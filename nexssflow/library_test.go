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
	aigateway "github.com/nexssp/ai/gateway"
)

type fakeProvider struct {
	request      llm.Request
	response     llm.Response
	calls        int
	providerName string
}

func (p *fakeProvider) Name() string {
	if p.providerName != "" {
		return p.providerName
	}
	return "fake"
}

func (p *fakeProvider) Complete(_ context.Context, req llm.Request) (llm.Response, error) {
	p.request = req
	p.calls++
	model := req.Model
	if model == "" {
		model = "fake-model"
	}
	if len(p.response.ToolCalls) > 0 {
		response := p.response
		if response.Model == "" {
			response.Model = model
		}
		return response, nil
	}
	return llm.Response{Content: "fake answer", Model: model, PromptTokens: 3, CompTokens: 2}, nil
}

func (p *fakeProvider) Stream(context.Context, llm.Request) (iter.Seq2[string, error], error) {
	return func(_ func(string, error) bool) {}, nil
}

func (p *fakeProvider) SupportsToolCalls() bool { return true }

func TestBundleRegistersAndRunsTypedPromptActionThroughFlow(t *testing.T) {
	factory, ok := compiler.Lookup(ID)
	if !ok {
		t.Fatal("ai bundle factory was not registered")
	}

	provider := &fakeProvider{response: llm.Response{
		Content: "fake answer", Model: "fake-model", PromptTokens: 3, CompTokens: 2,
		ToolCalls: []llm.ToolCall{{ID: "call-flow", Type: "function", Function: llm.FunctionCall{Name: "count_words", Arguments: "{}"}}},
	}}
	var resolvedGlobal GlobalOptions
	bundle, err := NewBundleWithProviderBuilder(GlobalOptions{DefaultModel: "configured-model"}, func(global GlobalOptions, _ LLMDecl) (llm.Provider, error) {
		resolvedGlobal = global
		return provider, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if bundle.ID != ID || len(bundle.Libraries) != 1 || bundle.Libraries[0].Name != "ai" {
		t.Fatalf("unexpected Flow bundle: %+v", bundle)
	}
	if len(bundle.Libraries[0].Aliases) != 0 {
		t.Fatal("Flow bundle must expose canonical actions without aliases")
	}
	if !slices.Contains(bundle.ExportedModules, llamaCppModule) {
		t.Fatalf("AI bundle does not export %q", llamaCppModule)
	}
	if factory == nil {
		t.Fatal("registered bundle factory is nil")
	}

	bundles := append(native.Bundles(), bundle)
	cfg, err := runner.BuildConfig(bundles)
	if err != nil {
		t.Fatalf("BuildConfig: %v", err)
	}
	for _, name := range []string{
		"ai.prompt", "ai.team", "ai.decide", "ai.cost.estimate", "ai.nflow.view",
		"ai.weights.adapters", "ai.weights.inspect", "ai.weights.patch",
		"ai.retrieval.index", "ai.retrieval.delete", "ai.retrieval.search", "ai.rag.answer",
		"ai.memory.reply", "ai.skills.list", "ai.skills.get", "ai.skills.draft",
		"external.exec",
	} {
		if _, found := cfg.Resolver.Action(name); !found {
			t.Fatalf("%s was not mounted in the Flow resolver", name)
		}
	}

	contextJSON := json.RawMessage(`{"job":"keep-explicitly"}`)
	execution, err := runner.Execute(context.Background(), cfg, "ai.prompt", "prompt-test.nflow", map[string]any{
		"candidate_id": "candidate-a",
		"attempt_id":   "attempt-1",
		"context":      contextJSON,
		"tools": []any{
			map[string]any{
				"type": "function",
				"function": map[string]any{
					"name": "count_words", "description": "Count words.",
					"parameters": map[string]any{"type": "object", "properties": map[string]any{}, "required": []any{}, "additionalProperties": false},
				},
			},
		},
		"tool_choice": "required",
		"system":      "Be concise",
		"user":        "What is 2+2?",
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
	if len(out.ToolCalls) != 1 || out.ToolCalls[0].ID != "call-flow" || out.ToolCalls[0].Function.Name != "count_words" || out.ToolCalls[0].Function.Arguments != "{}" {
		t.Fatalf("Flow output lost tool calls: %+v", out.ToolCalls)
	}
	if string(out.Context) != string(contextJSON) {
		t.Fatalf("explicit context = %s, want %s", out.Context, contextJSON)
	}
	if provider.calls != 1 {
		t.Fatalf("provider calls = %d, want 1", provider.calls)
	}
	if resolvedGlobal.DefaultModel != "configured-model" || provider.request.Model != "" || len(provider.request.Messages) != 2 || provider.request.Messages[0].Content != "Be concise" || provider.request.Messages[1].Content != "What is 2+2?" || len(provider.request.Tools) != 1 || provider.request.Tools[0].Function.Name != "count_words" || provider.request.ToolChoice != llm.ToolChoiceRequired {
		t.Fatalf("unexpected AI request: %+v", provider.request)
	}
	roundtrip, err := runner.Execute(context.Background(), cfg, "ai.prompt", "tool-roundtrip.nflow", map[string]any{
		"messages": []any{
			map[string]any{"role": "system", "content": "Use the tool result."},
			map[string]any{"role": "user", "content": "Count these words."},
			map[string]any{"role": "assistant", "tool_calls": []any{
				map[string]any{"id": "call-flow-roundtrip", "type": "function", "function": map[string]any{"name": "count_words", "arguments": "{}"}},
			}},
			map[string]any{"role": "tool", "tool_call_id": "call-flow-roundtrip", "name": "count_words", "content": "3\n"},
		},
	})
	if err != nil {
		t.Fatalf("runner.Execute message history: %v", err)
	}
	if _, ok := roundtrip.Output.(PromptResult); !ok {
		t.Fatalf("message history output type = %T, want PromptResult", roundtrip.Output)
	}
	if provider.calls != 2 || len(provider.request.Messages) != 4 || provider.request.Messages[2].Role != llm.RoleAssistant || len(provider.request.Messages[2].ToolCalls) != 1 || provider.request.Messages[2].ToolCalls[0].ID != "call-flow-roundtrip" || provider.request.Messages[3].Role != llm.RoleTool || provider.request.Messages[3].ToolCallID != "call-flow-roundtrip" || provider.request.Messages[3].Content != "3\n" {
		t.Fatalf("Flow message history did not round-trip: calls=%d messages=%+v", provider.calls, provider.request.Messages)
	}
}

func TestPipelineToolSpecsDeriveFromNFlowMetadata(t *testing.T) {
	provider := &fakeProvider{}
	bundle, err := NewBundleWithProviderBuilder(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) {
		return provider, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := runner.BuildConfig(append(native.Bundles(), bundle))
	if err != nil {
		t.Fatalf("BuildConfig: %v", err)
	}
	source := `@schema NoArguments {}
@schema QueryArguments {
	  query string ` + "`json:\"query\" validate:\"required\"`" + `
}
@pipeline count_words
  :tool-schema=!NoArguments
  :tool-description="Count words in the supplied query."
  noop
@end
@pipeline search_text
  :tool-schema=QueryArguments
  :tool-description="Search for a query in text."
  noop
@end
@pipeline ask_with_tools
  noop as request
  -> ai.tool_specs as prepared
  -> {
      user: "Count these words.",
      tools: prepared.ai_tool_specs,
      tool_choice: "required"
    } -> ai.prompt
@end
pipeline.ask_with_tools`
	if _, err := runner.Execute(context.Background(), cfg, source, "tool-specs.nflow", map[string]any{}); err != nil {
		t.Fatalf("execute NFlow-derived tool specs: %v", err)
	}
	if len(provider.request.Tools) != 2 {
		t.Fatalf("provider tools = %+v, want both annotated pipelines", provider.request.Tools)
	}
	tool := provider.request.Tools[0]
	if tool.Type != "function" || tool.Function.Name != "count_words" || tool.Function.Description != "Count words in the supplied query." {
		t.Fatalf("derived tool = %+v, want pipeline name and description", tool)
	}
	parameters, ok := tool.Function.Parameters.(map[string]any)
	if !ok {
		t.Fatalf("tool parameters have type %T, want JSON Schema object", tool.Function.Parameters)
	}
	if parameters["type"] != "object" || parameters["additionalProperties"] != false {
		t.Fatalf("tool parameter schema = %#v, want strict object schema", parameters)
	}
	properties, ok := parameters["properties"].(map[string]any)
	if !ok {
		t.Fatalf("tool properties have type %T, want object", parameters["properties"])
	}
	if len(properties) != 0 {
		t.Fatalf("empty tool properties = %#v, want no model arguments", properties)
	}
	tool = provider.request.Tools[1]
	if tool.Type != "function" || tool.Function.Name != "search_text" || tool.Function.Description != "Search for a query in text." {
		t.Fatalf("second derived tool = %+v, want second pipeline name and description", tool)
	}
	parameters, ok = tool.Function.Parameters.(map[string]any)
	if !ok {
		t.Fatalf("second tool parameters have type %T, want JSON Schema object", tool.Function.Parameters)
	}
	properties, ok = parameters["properties"].(map[string]any)
	if !ok {
		t.Fatalf("second tool properties have type %T, want object", parameters["properties"])
	}
	query, ok := properties["query"].(map[string]any)
	if !ok || query["type"] != "string" {
		t.Fatalf("query property schema = %#v, want string", properties["query"])
	}
	required, ok := parameters["required"].([]string)
	if !ok || !slices.Equal(required, []string{"query"}) {
		t.Fatalf("required fields = %#v, want [query]", parameters["required"])
	}
}

func TestNewBundleWithProviderBuilderRejectsNilBuilder(t *testing.T) {
	if _, err := NewBundleWithProviderBuilder(GlobalOptions{}, nil); err == nil {
		t.Fatal("NewBundleWithProviderBuilder accepted a nil provider builder")
	}
}

func TestFlowBundleCanInjectGateway(t *testing.T) {
	provider := &fakeProvider{}
	selectorCalled := false
	gateCalled := false
	gw, err := aigateway.New(aigateway.Config{
		Routes: []aigateway.Route{{Name: "candidate-route", Model: "routed-model", Provider: provider}},
		Select: func(_ context.Context, req llm.Request) (string, error) {
			selectorCalled = req.CandidateID == "candidate-fast"
			return "candidate-route", nil
		},
		Gates: []aigateway.Gate{func(_ context.Context, req llm.Request, route aigateway.Route) error {
			gateCalled = req.AttemptID == "attempt-9" && route.Name == "candidate-route"
			return nil
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	bundle, err := NewBundleWithProviderBuilder(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) {
		return gw, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := runner.BuildConfig(append(native.Bundles(), bundle))
	if err != nil {
		t.Fatalf("BuildConfig: %v", err)
	}
	execution, err := runner.Execute(context.Background(), cfg, "ai.prompt", "gateway-prompt-test.nflow", map[string]any{
		"candidate_id": "candidate-fast",
		"attempt_id":   "attempt-9",
		"model":        "caller-model",
		"user":         "route this prompt",
	})
	if err != nil {
		t.Fatalf("runner.Execute: %v", err)
	}
	out, ok := execution.Output.(PromptResult)
	if !ok {
		t.Fatalf("output type = %T, want PromptResult", execution.Output)
	}
	if !selectorCalled || !gateCalled || provider.calls != 1 || provider.request.Model != "routed-model" {
		t.Fatalf("gateway injection failed: selector=%v gate=%v calls=%d request=%#v", selectorCalled, gateCalled, provider.calls, provider.request)
	}
	if out.Model != "routed-model" || out.CandidateID != "candidate-fast" || out.AttemptID != "attempt-9" {
		t.Fatalf("unexpected gateway-routed Flow result: %+v", out)
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
