package nexssflow

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/nexssp/flow/compiler"
	"github.com/nexssp/flow/extensions/require"
	"github.com/nexssp/flow/extensions/symbols"
	"github.com/nexssp/flow/native"
	"github.com/nexssp/flow/runner"
	"github.com/nexssp/kernel/xerr"

	"github.com/nexssp/ai/engine/llm"
	"github.com/nexssp/ai/memory"
	"github.com/nexssp/ai/skills"
)

type routedMemoryWorkingStore struct {
	scopes    []memory.Scope
	turns     []memory.Turn
	recentErr error
}

func (s *routedMemoryWorkingStore) Recent(_ context.Context, scope memory.Scope, _ int) ([]memory.Turn, error) {
	s.scopes = append(s.scopes, scope)
	if s.recentErr != nil {
		return nil, s.recentErr
	}
	return nil, nil
}

func (s *routedMemoryWorkingStore) Append(_ context.Context, scope memory.Scope, turn memory.Turn) error {
	s.scopes = append(s.scopes, scope)
	s.turns = append(s.turns, turn)
	return nil
}

func (*routedMemoryWorkingStore) Older(context.Context, memory.Scope, time.Time, int) ([]memory.Turn, error) {
	return nil, nil
}

func (*routedMemoryWorkingStore) Delete(context.Context, memory.Scope, []string) error { return nil }

type routedMemoryLongTermStore struct {
	searchScopes []memory.Scope
	records      []memory.Record
}

func (s *routedMemoryLongTermStore) Search(_ context.Context, scope memory.Scope, _ string, _ int) ([]memory.SearchHit, error) {
	s.searchScopes = append(s.searchScopes, scope)
	return nil, nil
}

func (s *routedMemoryLongTermStore) Upsert(_ context.Context, record memory.Record) error {
	s.records = append(s.records, record)
	return nil
}

type memorySkillsRouteExample struct {
	config runner.Config
	source string
	path   string
}

func TestMemorySkillsRouteExampleUsesScopedServicesAndInertDrafts(t *testing.T) {
	t.Parallel()
	activeDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(activeDir, "summarize.md"), []byte("Summarize with accurate, short bullet points."), 0o600); err != nil {
		t.Fatal(err)
	}
	draftDir := filepath.Join(t.TempDir(), "drafts")
	catalog, err := skills.New(skills.Config{Directory: activeDir, DraftDirectory: draftDir})
	if err != nil {
		t.Fatal(err)
	}

	working := &routedMemoryWorkingStore{}
	semantic := &routedMemoryLongTermStore{}
	memoryService, err := memory.NewService(memory.Config{}, working, semantic, memory.CompleteFunc(func(context.Context, []memory.Message) (string, error) {
		return "memory answer", nil
	}))
	if err != nil {
		t.Fatal(err)
	}
	provider := &fakeProvider{}
	example := buildMemorySkillsRouteExample(t, memoryService, catalog, provider)

	memoryRun, err := runner.Execute(t.Context(), example.config, example.source, example.path, map[string]any{
		"route": "memory", "user_id": "user-1", "conversation_id": "chat-1", "memory_scope": "user",
		"user_message": "What is my preferred format?", "remember": true,
	})
	if err != nil {
		t.Fatalf("execute memory route: %v", err)
	}
	memoryResult, ok := memoryRun.Output.(memory.Response)
	if !ok || memoryResult.Assistant != "memory answer" {
		t.Fatalf("memory route output = %#v (%T), want memory.Response", memoryRun.Output, memoryRun.Output)
	}
	wantUserScope := memory.Scope{UserID: "user-1", MemoryScope: memory.MemoryScopeUser}
	if len(semantic.searchScopes) != 1 || semantic.searchScopes[0] != wantUserScope || len(semantic.records) != 1 || semantic.records[0].Scope != wantUserScope {
		t.Fatalf("memory route scopes: search=%+v records=%+v, want user-level scope", semantic.searchScopes, semantic.records)
	}
	wantWorkingScope := memory.Scope{UserID: "user-1", ConversationID: "chat-1", MemoryScope: memory.MemoryScopeConversation}
	if len(working.scopes) != 2 || working.scopes[0] != wantWorkingScope || working.scopes[1] != wantWorkingScope {
		t.Fatalf("working-history scopes = %+v, want conversation-only scope", working.scopes)
	}
	_, err = runner.Execute(t.Context(), example.config, example.source, example.path, map[string]any{
		"route": "memory", "user_id": "user-1", "conversation_id": "chat-default",
		"user_message": "Use the compatibility default.", "remember": true,
	})
	if err != nil {
		t.Fatalf("execute default conversation-scope route: %v", err)
	}
	wantConversationScope := memory.Scope{UserID: "user-1", ConversationID: "chat-default", MemoryScope: memory.MemoryScopeConversation}
	if len(semantic.searchScopes) != 2 || semantic.searchScopes[1] != wantConversationScope || len(semantic.records) != 2 || semantic.records[1].Scope != wantConversationScope {
		t.Fatalf("omitted memory_scope did not default to conversation scope: searches=%+v records=%+v", semantic.searchScopes, semantic.records)
	}

	skillRun, err := runner.Execute(t.Context(), example.config, example.source, example.path, map[string]any{
		"route": "skill", "memory_scope": "conversation", "skill_name": "summarize", "user_message": "Summarize this update.",
	})
	if err != nil {
		t.Fatalf("execute approved-skill route: %v", err)
	}
	skillResult, ok := skillRun.Output.(PromptResult)
	if !ok || skillResult.Content != "fake answer" || len(provider.request.Messages) != 2 || !strings.Contains(provider.request.Messages[0].Content, "accurate, short bullet points") {
		t.Fatalf("approved-skill route output=%#v request=%+v, want skill instructions in system prompt", skillRun.Output, provider.request)
	}

	draftRun, err := runner.Execute(t.Context(), example.config, example.source, example.path, map[string]any{
		"route": "draft", "draft_name": "new-helper", "draft_instructions": "Explain the steps and ask before taking action.",
	})
	if err != nil {
		t.Fatalf("execute draft route: %v", err)
	}
	draftResult, ok := draftRun.Output.(skills.DraftReceipt)
	if !ok || draftResult.Name != "new-helper" {
		t.Fatalf("draft route output = %#v (%T), want inert DraftReceipt", draftRun.Output, draftRun.Output)
	}
	if _, err := catalog.Get("new-helper"); err == nil {
		t.Fatal("NFlow draft action activated the skill without host review")
	}
	if _, err := os.Stat(filepath.Join(draftDir, draftResult.File)); err != nil {
		t.Fatalf("draft file was not written outside active catalog: %v", err)
	}

	working.recentErr = xerr.Unavailable("test outage")
	fallbackRun, err := runner.Execute(t.Context(), example.config, example.source, example.path, map[string]any{
		"route": "memory", "user_id": "user-2", "conversation_id": "chat-2", "memory_scope": "conversation",
		"user_message": "Use the fallback.",
	})
	if err != nil {
		t.Fatalf("memory failure should use the `.input` fallback: %v", err)
	}
	fallback, ok := fallbackRun.Output.(map[string]any)
	if !ok || fallback["status"] != "memory_unavailable" || fallback["user_id"] != "user-2" {
		t.Fatalf("memory fallback output = %#v, want memory_unavailable with original user_id", fallbackRun.Output)
	}
}

func buildMemorySkillsRouteExample(t *testing.T, memoryService *memory.Service, catalog *skills.Catalog, provider llm.Provider) memorySkillsRouteExample {
	t.Helper()
	path := filepath.Join("..", "nflows", "memory_skills_route.nflow")
	contents, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read routed memory/skills example: %v", err)
	}
	source := string(contents)
	_, meta, err := compiler.Preprocess(t.Context(), compiler.NewDirectiveTable(require.Directive, symbols.Directive), source, path)
	if err != nil {
		t.Fatalf("preprocess routed memory/skills example: %v", err)
	}
	requirements, err := require.SourceRequirements(meta)
	if err != nil || len(requirements) != 1 || requirements[0].Import != "github.com/nexssp/ai" {
		t.Fatalf("example requirements = %+v, err = %v", requirements, err)
	}
	global, err := options(requirements[0].Options)
	if err != nil {
		t.Fatalf("parse example options: %v", err)
	}
	bundle, err := NewBundleWithHostServices(global, func(GlobalOptions, LLMDecl) (llm.Provider, error) {
		return provider, nil
	}, HostServices{Memory: memoryService, Skills: catalog})
	if err != nil {
		t.Fatalf("create host-services bundle: %v", err)
	}
	bundle.Alias = requirements[0].Alias
	cfg, err := runner.BuildConfig(append(native.Bundles(), bundle))
	if err != nil {
		t.Fatalf("build Flow runner: %v", err)
	}
	return memorySkillsRouteExample{config: cfg, source: source, path: path}
}

var _ llm.Provider = (*fakeProvider)(nil)
