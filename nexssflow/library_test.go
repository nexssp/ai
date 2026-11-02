package nexssflow

import (
	"context"
	"encoding/json"
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
	model := req.Model
	if model == "" {
		model = "fake-model"
	}
	return llm.Response{Content: "fake answer", Model: model, PromptTokens: 3, CompTokens: 2}, nil
}

func (p *fakeProvider) Stream(context.Context, llm.Request) (iter.Seq2[string, error], error) {
	return func(_ func(string, error) bool) {}, nil
}

func TestBundleRegistersAndRunsTypedPromptActionThroughFlow(t *testing.T) {
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
		t.Fatal("Flow bundle must expose canonical actions without aliases")
	}
	if factory == nil {
		t.Fatal("registered bundle factory is nil")
	}

	bundles := append(native.Bundles(), bundle)
	cfg, err := runner.BuildConfig(bundles)
	if err != nil {
		t.Fatalf("BuildConfig: %v", err)
	}
	for _, name := range []string{"ai.prompt", "ai.team", "ai.decide", "ai.cost.estimate"} {
		if _, found := cfg.Resolver.Action(name); !found {
			t.Fatalf("%s was not mounted in the Flow resolver", name)
		}
	}

	contextJSON := json.RawMessage(`{"job":"keep-explicitly"}`)
	execution, err := runner.Execute(context.Background(), cfg, "ai.prompt", "prompt-test.nflow", map[string]any{
		"candidate_id": "candidate-a",
		"attempt_id":   "attempt-1",
		"context":      contextJSON,
		"system":       "Be concise",
		"user":         "What is 2+2?",
	})
	if err != nil {
		t.Fatalf("runner.Execute: %v", err)
	}
	out, ok := execution.Output.(PromptResult)
	if !ok {
		t.Fatalf("output type = %T, want PromptResult", execution.Output)
	}
	if out.Content != "fake answer" || out.CandidateID != "candidate-a" || out.AttemptID != "attempt-1" || out.PromptTokens != 3 || out.TotalTokens != 5 {
		t.Fatalf("unexpected typed output: %+v", out)
	}
	if string(out.Context) != string(contextJSON) {
		t.Fatalf("explicit context = %s, want %s", out.Context, contextJSON)
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
	act := newPromptActionWith(GlobalOptions{}, func(_ GlobalOptions, _ LLMDecl) (llm.Provider, error) {
		return provider, nil
	})
	_, err := act.DoAny(context.Background(), PromptRequest{System: "system only"})
	if err == nil {
		t.Fatal("expected a missing-user error")
	}
	if provider.calls != 0 {
		t.Fatal("provider completion ran despite the missing user field")
	}
}

func TestBundleDeclaresAndValidatesRequireOptions(t *testing.T) {
	bundle := Bundle(nil)
	want := []string{
		"provider", "model", "api_key_env", "timeout", "base_url",
		"decision_provider", "decision_model", "decision_api_key_env", "decision_base_url",
	}
	if !slices.Equal(bundle.AcceptedOptions, want) {
		t.Fatalf("accepted @require options = %v, want %v", bundle.AcceptedOptions, want)
	}

	_, err := require.ResolveBundles([]require.Requirement{{
		Import:  ID,
		Options: map[string]string{"default_model": "typo"},
	}})
	if err == nil || !strings.Contains(err.Error(), "unknown option") {
		t.Fatalf("unknown @require option error = %v, want explicit option validation", err)
	}
}
