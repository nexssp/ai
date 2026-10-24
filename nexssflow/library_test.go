package nexssflow

import (
	"context"
	"iter"
	"slices"
	"strings"
	"testing"

	"github.com/nexssp/flow/compiler"
	"github.com/nexssp/flow/extensions/require"
	"github.com/nexssp/flow/native"
	"github.com/nexssp/flow/runner"

	"github.com/nexssp/ai/engine/llm"
)

type fakeProvider struct {
	request llm.Request
	calls   int
}

func (p *fakeProvider) Name() string { return "fake" }

func (p *fakeProvider) Complete(_ context.Context, req llm.Request) (llm.Response, error) {
	p.request = req
	p.calls++
	return llm.Response{Content: "fake answer", Model: "fake-model", PromptTokens: 3, CompTokens: 2}, nil
}

func (p *fakeProvider) Stream(context.Context, llm.Request) (iter.Seq2[string, error], error) {
	return func(yield func(string, error) bool) {}, nil
}

func TestBundleRegistersAndRunsPromptActionThroughFlow(t *testing.T) {
	factory, ok := compiler.Lookup(ID)
	if !ok {
		t.Fatal("ai bundle factory was not registered")
	}

	provider := &fakeProvider{}
	var resolvedGlobal GlobalOptions
	bundle := bundleWithBuilder(GlobalOptions{DefaultModel: "configured-model"}, func(global GlobalOptions, _ LLMDecl) (llm.Provider, error) {
		resolvedGlobal = global
		return provider, nil
	})
	if bundle.ID != ID || len(bundle.Libraries) != 1 || bundle.Libraries[0].Name != "ai" {
		t.Fatalf("unexpected Flow bundle: %+v", bundle)
	}
	if len(bundle.Libraries[0].Aliases) != 0 {
		t.Fatal("Flow v0.42 bundles must expose canonical actions without Kernel aliases")
	}
	if factory == nil {
		t.Fatal("registered bundle factory is nil")
	}

	bundles := append(native.Bundles(), bundle)
	cfg, err := runner.BuildConfig(bundles)
	if err != nil {
		t.Fatalf("BuildConfig: %v", err)
	}
	if _, ok := cfg.Resolver.Action("ai.prompt"); !ok {
		t.Fatal("ai.prompt was not mounted in the Flow resolver")
	}
	if _, ok := cfg.Resolver.Action("ai.team"); !ok {
		t.Fatal("ai.team was not mounted in the Flow resolver")
	}

	execution, err := runner.Execute(context.Background(), cfg, "ai.prompt", "prompt-test.nflow", map[string]any{
		"job_id": "keep-me",
		"system": "Be concise",
		"user":   "What is 2+2?",
	})
	if err != nil {
		t.Fatalf("runner.Execute: %v", err)
	}
	out, ok := execution.Output.(map[string]any)
	if !ok {
		t.Fatalf("output type = %T, want map[string]any", execution.Output)
	}
	if out["content"] != "fake answer" || out["job_id"] != "keep-me" || out["prompt_tokens"] != 3 || out["total_tokens"] != 5 {
		t.Fatalf("unexpected merged output: %#v", out)
	}
	if provider.calls != 1 {
		t.Fatalf("provider calls = %d, want 1", provider.calls)
	}
	if resolvedGlobal.DefaultModel != "configured-model" || provider.request.Model != "" || len(provider.request.Messages) != 2 || provider.request.Messages[0].Content != "Be concise" || provider.request.Messages[1].Content != "What is 2+2?" {
		t.Fatalf("unexpected AI request: %+v", provider.request)
	}
}

func TestPromptActionRequiresUserWithoutCompletion(t *testing.T) {
	provider := &fakeProvider{}
	act := newPromptActionWith(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) {
		return provider, nil
	})
	_, err := act.DoAny(context.Background(), map[string]any{"system": "system only"})
	if err == nil {
		t.Fatal("expected a missing-user error")
	}
	if provider.calls != 0 {
		t.Fatal("provider completion ran despite the missing user field")
	}
}

func TestBundleDeclaresAndValidatesRequireOptions(t *testing.T) {
	bundle := Bundle(nil)
	want := []string{"provider", "model", "api_key_env", "timeout", "base_url"}
	if !slices.Equal(bundle.AcceptedOptions, want) {
		t.Fatalf("accepted @require options = %v, want %v", bundle.AcceptedOptions, want)
	}

	_, err := require.ResolveBundles([]require.Requirement{{
		Import:  ID,
		Options: map[string]string{"defualt_model": "typo"},
	}})
	if err == nil || !strings.Contains(err.Error(), "unknown option") {
		t.Fatalf("unknown @require option error = %v, want explicit option validation", err)
	}
}
