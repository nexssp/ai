package nexssflow

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/nexssp/flow/compiler"
	"github.com/nexssp/flow/extensions/require"
	"github.com/nexssp/flow/native"
	"github.com/nexssp/flow/runner"

	"github.com/nexssp/ai/engine/llm"
	"github.com/nexssp/ai/retrieval"
)

type retrievalActionTestEmbedder struct{}

func (retrievalActionTestEmbedder) Embed(_ context.Context, texts ...string) ([][]float32, error) {
	vectors := make([][]float32, len(texts))
	for i := range texts {
		vectors[i] = []float32{1, 0}
	}
	return vectors, nil
}

func TestRetrievalAndRAGExampleRunsTypedActionsWithLatestFlow(t *testing.T) {
	service, err := retrieval.NewService(retrieval.Config{
		ChunkSize:       100,
		ChunkOverlap:    0,
		TopK:            3,
		MaxContextRunes: 500,
	}, retrievalActionTestEmbedder{}, retrieval.NewMemoryIndex())
	if err != nil {
		t.Fatal(err)
	}
	provider := &fakeProvider{}
	var providerDecl LLMDecl
	builder := func(_ GlobalOptions, decl LLMDecl) (llm.Provider, error) {
		providerDecl = decl
		return provider, nil
	}
	cfg, source, sourcePath := buildRetrievalExample(t, service, builder)

	indexRun := executeRetrievalExamplePipeline(t, cfg, source, sourcePath, "index_document", map[string]any{
		"namespace": "tenant-a",
		"id":        "handbook",
		"text":      "To reset the device, hold the blue button for five seconds.",
		"metadata":  map[string]any{"kind": "guide"},
	})
	indexed, ok := indexRun.Output.(retrieval.IndexResult)
	if !ok || indexed.Namespace != "tenant-a" || indexed.DocumentID != "handbook" || indexed.ChunkCount != 1 {
		t.Fatalf("index pipeline output = %#v (%T), want typed IndexResult", indexRun.Output, indexRun.Output)
	}

	searchRun := executeRetrievalExamplePipeline(t, cfg, source, sourcePath, "search_sources", map[string]any{
		"namespace": "tenant-a",
		"query":     "How do I reset the device?",
	})
	search, ok := searchRun.Output.(retrieval.SearchResponse)
	if !ok || len(search.Matches) != 1 || search.Matches[0].DocumentID != "handbook" {
		t.Fatalf("search pipeline output = %#v (%T), want typed SearchResponse with handbook", searchRun.Output, searchRun.Output)
	}

	answerRun := executeRetrievalExamplePipeline(t, cfg, source, sourcePath, "answer", map[string]any{
		"namespace":  "tenant-a",
		"query":      "How do I reset the device?",
		"model":      "test-chat-model",
		"max_tokens": 64,
	})
	answer, ok := answerRun.Output.(retrieval.AnswerResponse)
	if !ok || answer.Answer != "fake answer" || answer.Model != "test-chat-model" || len(answer.Sources) != 1 {
		t.Fatalf("answer pipeline output = %#v (%T), want typed AnswerResponse with one source", answerRun.Output, answerRun.Output)
	}
	if providerDecl.Model != "test-chat-model" || len(provider.request.Messages) != 2 || !strings.Contains(provider.request.Messages[1].Content, "untrusted reference data") {
		t.Fatalf("RAG provider request lost model or untrusted-source treatment: decl=%+v request=%+v", providerDecl, provider.request)
	}

	if _, err := runner.Execute(t.Context(), cfg, sourceForPipeline(source, "answer"), sourcePath, map[string]any{
		"namespace":  "tenant-a",
		"query":      "question",
		"unexpected": true,
	}); err == nil {
		t.Fatal("strict RAG request schema accepted an undeclared field")
	}

	deleteRun := executeRetrievalExamplePipeline(t, cfg, source, sourcePath, "delete_document", map[string]any{
		"namespace":   "tenant-a",
		"document_id": "handbook",
	})
	deleted, ok := deleteRun.Output.(RetrievalDeleteResponse)
	if !ok || !deleted.Deleted || deleted.Namespace != "tenant-a" || deleted.DocumentID != "handbook" {
		t.Fatalf("delete pipeline output = %#v (%T), want typed RetrievalDeleteResponse", deleteRun.Output, deleteRun.Output)
	}
}

func buildRetrievalExample(t *testing.T, service *retrieval.Service, builder ProviderBuilder) (cfg runner.Config, source, sourcePath string) {
	t.Helper()
	sourcePath = filepath.Join("..", "nflows", "retrieval_rag.nflow")
	sourceBytes, err := os.ReadFile(sourcePath)
	if err != nil {
		t.Fatalf("read NFlow example: %v", err)
	}
	source = string(sourceBytes)
	_, meta, err := compiler.Preprocess(t.Context(), compiler.NewDirectiveTable(require.Directive), source, sourcePath)
	if err != nil {
		t.Fatalf("preprocess NFlow example: %v", err)
	}
	requirements, err := require.SourceRequirements(meta)
	if err != nil || len(requirements) != 1 || requirements[0].Import != "github.com/nexssp/ai" {
		t.Fatalf("NFlow example requirements = %+v, err = %v", requirements, err)
	}
	global, err := options(requirements[0].Options)
	if err != nil {
		t.Fatalf("parse NFlow example options: %v", err)
	}
	bundle, err := NewBundleWithServices(global, builder, service)
	if err != nil {
		t.Fatalf("create AI bundle: %v", err)
	}
	bundle.Alias = requirements[0].Alias
	cfg, err = runner.BuildConfig(append(native.Bundles(), bundle))
	if err != nil {
		t.Fatalf("build Flow runner: %v", err)
	}
	return cfg, source, sourcePath
}

func executeRetrievalExamplePipeline(t *testing.T, cfg runner.Config, source, sourcePath, pipeline string, input map[string]any) runner.Execution {
	t.Helper()
	execution, err := runner.Execute(t.Context(), cfg, sourceForPipeline(source, pipeline), sourcePath, input)
	if err != nil {
		t.Fatalf("execute %s pipeline: %v", pipeline, err)
	}
	return execution
}

func sourceForPipeline(source, pipeline string) string {
	return strings.TrimSuffix(source, "pipeline.answer\n") + "pipeline." + pipeline + "\n"
}
