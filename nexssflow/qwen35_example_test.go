package nexssflow

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/nexssp/flow/compiler"
	"github.com/nexssp/flow/extensions/require"
	"github.com/nexssp/flow/native"
	"github.com/nexssp/flow/runner"

	"github.com/nexssp/ai/engine/llm"
)

const qwen35Revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

func TestQwen35ExampleUsesPinnedModelAndConfigurableLocalProvider(t *testing.T) {
	t.Parallel()
	path := filepath.Join("..", "nflows", "examples", "qwen35_openai_compatible.nflow")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	source := string(data)
	if !strings.Contains(source, qwen35Revision) || !strings.Contains(source, "Qwen/Qwen3.5-4B") {
		t.Fatal("Qwen3.5 example must keep the official model ID and immutable revision")
	}
	_, meta, err := compiler.Preprocess(t.Context(), compiler.NewDirectiveTable(require.Directive, namedLLMDirective), source, path)
	if err != nil {
		t.Fatalf("preprocess Qwen3.5 example: %v", err)
	}
	requirements, err := require.SourceRequirements(meta)
	if err != nil || len(requirements) != 1 || requirements[0].Import != "github.com/nexssp/ai" {
		t.Fatalf("example requirements = %+v, err = %v", requirements, err)
	}
	global, err := options(requirements[0].Options)
	if err != nil {
		t.Fatalf("parse @require options: %v", err)
	}
	provider := &fakeProvider{providerName: "openai-compatible"}
	var resolved GlobalOptions
	bundle, err := NewBundleWithProviderBuilder(global, func(options GlobalOptions, _ LLMDecl) (llm.Provider, error) {
		resolved = options
		return provider, nil
	})
	if err != nil {
		t.Fatalf("create AI bundle: %v", err)
	}
	bundle.Alias = requirements[0].Alias
	cfg, err := runner.BuildConfig(append(native.Bundles(), bundle))
	if err != nil {
		t.Fatalf("build Flow runner: %v", err)
	}
	run, err := runner.Execute(t.Context(), cfg, source, path, map[string]any{"prompt": "What should I know about this model?"})
	if err != nil {
		t.Fatalf("execute Qwen3.5 example: %v", err)
	}
	result, ok := run.Output.(PromptResult)
	if !ok || result.Content != "fake answer" || result.Provider != "openai-compatible" {
		t.Fatalf("Qwen3.5 result = %#v (%T), want configured OpenAI-compatible prompt result", run.Output, run.Output)
	}
	if resolved.Provider != "openai-compatible" || resolved.DefaultModel != "qwen35-4b" || resolved.BaseURL != "http://127.0.0.1:8000/v1" {
		t.Fatalf("resolved named-provider config = %+v, want local OpenAI-compatible endpoint and served model alias", resolved)
	}
	if provider.request.MaxTokens != 256 || len(provider.request.Messages) != 2 {
		t.Fatalf("provider request = %+v, want bounded prompt settings", provider.request)
	}
}
