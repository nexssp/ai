package nexssflow

import (
	"errors"
	"testing"
	"time"

	"github.com/nexssp/cost"
	"github.com/nexssp/kernel/xerr"

	"github.com/nexssp/ai/engine/llm"
)

func TestPromptJSONModeIsTypedAndOptIn(t *testing.T) {
	for _, tc := range []struct {
		name string
		want bool
	}{
		{"default false", false},
		{"enabled", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			provider := &fakeProvider{}
			act := newPromptActionWith(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return provider, nil })
			_, err := act.DoAny(t.Context(), PromptRequest{User: "review", JSONMode: tc.want})
			if err != nil {
				t.Fatal(err)
			}
			if provider.request.JSONMode != tc.want {
				t.Fatalf("JSONMode = %v, want %v", provider.request.JSONMode, tc.want)
			}
		})
	}
}

func TestPromptPassesToolsAndReturnsToolCalls(t *testing.T) {
	tool := llm.ToolSpec{Type: "function", Function: llm.FunctionSpec{
		Name: "count_words", Description: "Count words in the caller's task.",
		Parameters: map[string]any{"type": "object", "properties": map[string]any{}, "required": []string{}, "additionalProperties": false},
	}}
	provider := &fakeProvider{response: llm.Response{
		Model:     "mercury-3",
		ToolCalls: []llm.ToolCall{{ID: "call-1", Type: "function", Function: llm.FunctionCall{Name: "count_words", Arguments: "{}"}}},
	}}
	act := newPromptActionWith(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return provider, nil })
	value, err := act.DoAny(t.Context(), PromptRequest{
		User: "Count the words in this text.", Tools: []llm.ToolSpec{tool}, ToolChoice: llm.ToolChoiceRequired,
	})
	if err != nil {
		t.Fatalf("ai.prompt: %v", err)
	}
	if len(provider.request.Tools) != 1 || provider.request.Tools[0].Function.Name != "count_words" || provider.request.ToolChoice != llm.ToolChoiceRequired {
		t.Fatalf("provider request lost tool configuration: %+v", provider.request)
	}
	result, ok := value.(PromptResult)
	if !ok {
		t.Fatalf("result type = %T, want PromptResult", value)
	}
	if len(result.ToolCalls) != 1 || result.ToolCalls[0].Function.Name != "count_words" || result.ToolCalls[0].Function.Arguments != "{}" {
		t.Fatalf("tool calls = %+v, want the provider call preserved", result.ToolCalls)
	}
}

func TestPromptCarriesExactPricingAndCorrelationMetadata(t *testing.T) {
	previous := llm.GlobalCatalog
	llm.GlobalCatalog = llm.NewPriceCatalog()
	defer func() { llm.GlobalCatalog = previous }()
	if err := llm.GlobalCatalog.Register("priced-model", llm.PriceCard{
		RateCardID: "client-eur-card",
		Currency:   cost.EUR,
		Peak:       llm.Price{PromptPer1k: 1, CompPer1k: 2},
	}); err != nil {
		t.Fatal(err)
	}

	expected := cost.EUR
	provider := &fakeProvider{}
	act := newPromptActionWith(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return provider, nil })
	value, err := act.DoAny(t.Context(), PromptRequest{
		CandidateID: "candidate-a", AttemptID: "attempt-4", Model: "priced-model",
		RateCardID: "client-eur-card", ExpectedCurrency: &expected,
		PromptTokens: 10, CachedPromptTokens: 3, CompletionTokens: 5,
		User: "route this request",
	})
	if err != nil {
		t.Fatalf("ai.prompt: %v", err)
	}
	result, ok := value.(PromptResult)
	if !ok {
		t.Fatalf("result type = %T, want PromptResult", value)
	}
	if result.CandidateID != "candidate-a" || result.AttemptID != "attempt-4" || result.RateCardID != "client-eur-card" || result.ExpectedCurrency == nil || *result.ExpectedCurrency != cost.EUR {
		t.Fatalf("prompt correlation/pricing echo = %+v", result)
	}
	if provider.request.CandidateID != "candidate-a" || provider.request.AttemptID != "attempt-4" || provider.request.RateCardID != "client-eur-card" || provider.request.ExpectedCurrency == nil || *provider.request.ExpectedCurrency != cost.EUR || provider.request.EstimatedPromptTokens != 10 || provider.request.EstimatedCachedPromptTokens != 3 || provider.request.EstimatedCompletionTokens != 5 {
		t.Fatalf("provider request lost accounting metadata: %+v", provider.request)
	}
	if result.Usage.RateCardID != "client-eur-card" || result.Usage.ExpectedCurrency == nil || *result.Usage.ExpectedCurrency != cost.EUR || !result.Usage.CostKnown || result.Usage.CostMicro != cost.Micro(7000) {
		t.Fatalf("usage did not use exact card: %+v", result.Usage)
	}
}

func TestPromptPricingMismatchStopsBeforeProviderDispatch(t *testing.T) {
	previous := llm.GlobalCatalog
	llm.GlobalCatalog = llm.NewPriceCatalog()
	defer func() { llm.GlobalCatalog = previous }()
	if err := llm.GlobalCatalog.Register("priced-model", llm.PriceCard{RateCardID: "configured-card", Currency: cost.EUR, Peak: llm.Price{PromptPer1k: 1}}); err != nil {
		t.Fatal(err)
	}

	expected := cost.USD
	provider := &fakeProvider{}
	act := newPromptActionWith(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return provider, nil })
	_, err := act.DoAny(t.Context(), PromptRequest{
		Model: "priced-model", RateCardID: "configured-card", ExpectedCurrency: &expected,
		PromptTokens: 10, User: "route this request",
	})
	var appErr *xerr.AppError
	if !errors.As(err, &appErr) || appErr.Kind != xerr.KindValidation {
		t.Fatalf("pricing mismatch error = %v, want terminal Validation", err)
	}
	if provider.calls != 0 {
		t.Fatalf("provider called %d times after pricing mismatch", provider.calls)
	}
}

type unsupportedToolProvider struct {
	*fakeProvider
}

func (*unsupportedToolProvider) SupportsToolCalls() bool { return false }

func TestPromptRejectsToolChoiceWithoutTools(t *testing.T) {
	provider := &fakeProvider{}
	act := newPromptActionWith(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return provider, nil })
	_, err := act.DoAny(t.Context(), PromptRequest{User: "Use a tool.", ToolChoice: llm.ToolChoiceRequired})
	var appErr *xerr.AppError
	if !errors.As(err, &appErr) || appErr.Kind != xerr.KindValidation {
		t.Fatalf("tool_choice without tools error = %v, want terminal Validation", err)
	}
	if provider.calls != 0 {
		t.Fatalf("provider calls = %d, want zero", provider.calls)
	}
}

func TestPromptRejectsUnsupportedToolChoiceBeforeDispatch(t *testing.T) {
	provider := &fakeProvider{}
	act := newPromptActionWith(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return provider, nil })
	tool := llm.ToolSpec{Type: "function", Function: llm.FunctionSpec{
		Name: "count_words", Parameters: map[string]any{"type": "object"},
	}}
	_, err := act.DoAny(t.Context(), PromptRequest{
		User: "Use a tool.", Tools: []llm.ToolSpec{tool}, ToolChoice: llm.ToolChoice("future-mode"),
	})
	var appErr *xerr.AppError
	if !errors.As(err, &appErr) || appErr.Kind != xerr.KindValidation {
		t.Fatalf("unknown tool_choice error = %v, want terminal Validation", err)
	}
	if provider.calls != 0 {
		t.Fatalf("provider calls = %d, want zero", provider.calls)
	}
}

func TestPromptRejectsProviderWithoutToolCallCapability(t *testing.T) {
	provider := &unsupportedToolProvider{fakeProvider: &fakeProvider{}}
	act := newPromptActionWith(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return provider, nil })
	tool := llm.ToolSpec{Type: "function", Function: llm.FunctionSpec{
		Name: "count_words", Parameters: map[string]any{"type": "object"},
	}}
	_, err := act.DoAny(t.Context(), PromptRequest{
		User: "Use a tool.", Tools: []llm.ToolSpec{tool}, ToolChoice: llm.ToolChoiceRequired,
	})
	var appErr *xerr.AppError
	if !errors.As(err, &appErr) || appErr.Kind != xerr.KindValidation {
		t.Fatalf("unsupported provider error = %v, want terminal Validation", err)
	}
	if provider.calls != 0 {
		t.Fatalf("provider calls = %d, want zero", provider.calls)
	}
}

func TestPromptPassesExplicitToolRoundTripMessages(t *testing.T) {
	messages := []llm.Message{
		{Role: llm.RoleSystem, Content: "Use the tool result."},
		{Role: llm.RoleUser, Content: "Count these words."},
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{
			ID: "call-roundtrip", Type: "function",
			Function: llm.FunctionCall{Name: "count_words", Arguments: "{}"},
		}}},
		{Role: llm.RoleTool, ToolCallID: "call-roundtrip", Name: "count_words", Content: "3\n"},
	}
	provider := &fakeProvider{}
	act := newPromptActionWith(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return provider, nil })
	_, err := act.DoAny(t.Context(), PromptRequest{Messages: messages})
	if err != nil {
		t.Fatalf("ai.prompt explicit message history: %v", err)
	}
	got := provider.request.Messages
	if len(got) != 4 || got[0].Role != llm.RoleSystem || got[1].Role != llm.RoleUser || got[2].Role != llm.RoleAssistant || got[3].Role != llm.RoleTool {
		t.Fatalf("provider messages = %+v, want system/user/assistant/tool conversation", got)
	}
	if len(got[2].ToolCalls) != 1 || got[2].ToolCalls[0].ID != "call-roundtrip" || got[2].ToolCalls[0].Function.Name != "count_words" || got[2].ToolCalls[0].Function.Arguments != "{}" {
		t.Fatalf("assistant tool-call message was not preserved: %+v", got[2])
	}
	if got[3].ToolCallID != "call-roundtrip" || got[3].Name != "count_words" || got[3].Content != "3\n" {
		t.Fatalf("tool-result message was not preserved: %+v", got[3])
	}
}

func TestPromptRejectsInvalidExplicitMessageHistoryBeforeDispatch(t *testing.T) {
	for _, tc := range []struct {
		name    string
		request PromptRequest
	}{
		{name: "no user message", request: PromptRequest{Messages: []llm.Message{{Role: llm.RoleSystem, Content: "system"}}}},
		{name: "mixed shorthand", request: PromptRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "history"}}, User: "also supplied"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			provider := &fakeProvider{}
			act := newPromptActionWith(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return provider, nil })
			_, err := act.DoAny(t.Context(), tc.request)
			var appErr *xerr.AppError
			if !errors.As(err, &appErr) || appErr.Kind != xerr.KindValidation {
				t.Fatalf("message history error = %v, want terminal Validation", err)
			}
			if provider.calls != 0 {
				t.Fatalf("provider calls = %d, want zero", provider.calls)
			}
		})
	}
}

func TestPromptForwardsAndValidatesOutputSchema(t *testing.T) {
	closed := false
	schema := llm.JSONSchema{
		Type: "object",
		Properties: map[string]llm.JSONSchema{
			"name":     {Type: "string"},
			"nickname": {Type: "string"},
		},
		Required:             []string{"name"},
		AdditionalProperties: &closed,
	}
	provider := &schemaCapableFakeProvider{fakeProvider: &fakeProvider{response: llm.Response{
		Content: `{"name":"Ada","nickname":null}`, Model: "schema-model",
	}}}
	act := newPromptActionWith(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return provider, nil })
	value, err := act.DoAny(t.Context(), PromptRequest{User: "Return a name", OutputSchema: &schema})
	if err != nil {
		t.Fatalf("strict schema request: %v", err)
	}
	result, ok := value.(PromptResult)
	if !ok {
		t.Fatalf("prompt result has type %T", value)
	}
	if provider.request.OutputSchema == nil || result.Content != `{"name":"Ada"}` || result.SchemaEnforcement != llm.SchemaEnforcementProvider {
		t.Fatalf("schema was not forwarded, normalized, and reported: request=%+v result=%+v", provider.request, result)
	}
}

func TestPromptSchemaAndToolChoiceSurviveObservableProvider(t *testing.T) {
	closed := false
	schema := llm.JSONSchema{Type: "object", Properties: map[string]llm.JSONSchema{"ok": {Type: "boolean"}}, Required: []string{"ok"}, AdditionalProperties: &closed}
	inner := &schemaCapableFakeProvider{fakeProvider: &fakeProvider{response: llm.Response{Content: `{"ok":true}`}}}
	provider := llm.NewObservableProvider(inner, func(string, string, string, int, int, int64, time.Duration, error) {})
	act := newPromptActionWith(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return provider, nil })
	tool := llm.ToolSpec{Type: "function", Function: llm.FunctionSpec{Name: "noop", Parameters: llm.JSONSchema{Type: "object"}}}
	if _, err := act.DoAny(t.Context(), PromptRequest{
		User: "Return JSON", OutputSchema: &schema, Tools: []llm.ToolSpec{tool}, ToolChoice: llm.ToolChoiceRequired,
	}); err != nil {
		t.Fatalf("schema and tool-choice request through observable provider: %v", err)
	}
	if inner.calls != 1 || inner.request.OutputSchema == nil || inner.request.ToolChoice != llm.ToolChoiceRequired {
		t.Fatalf("wrapped request was not preserved: calls=%d request=%+v", inner.calls, inner.request)
	}
}

func TestPromptStrictAndBestEffortSchemaCapabilityPolicy(t *testing.T) {
	closed := false
	schema := llm.JSONSchema{Type: "object", Properties: map[string]llm.JSONSchema{"ok": {Type: "boolean"}}, Required: []string{"ok"}, AdditionalProperties: &closed}
	strictProvider := &fakeProvider{response: llm.Response{Content: `{"ok":true}`}}
	act := newPromptActionWith(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return strictProvider, nil })
	if _, err := act.DoAny(t.Context(), PromptRequest{User: "Return JSON", OutputSchema: &schema}); err == nil {
		t.Fatal("strict mode accepted a provider with unknown schema capability")
	}
	if strictProvider.calls != 0 {
		t.Fatalf("unsupported strict schema dispatched %d provider calls", strictProvider.calls)
	}

	bestEffortProvider := &fakeProvider{response: llm.Response{Content: `{"ok":true}`}}
	act = newPromptActionWith(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return bestEffortProvider, nil })
	value, err := act.DoAny(t.Context(), PromptRequest{
		User: "Return JSON", OutputSchema: &schema, OutputSchemaMode: llm.SchemaModeBestEffort,
	})
	if err != nil {
		t.Fatalf("best-effort schema request: %v", err)
	}
	result, ok := value.(PromptResult)
	if !ok {
		t.Fatalf("prompt result has type %T", value)
	}
	if bestEffortProvider.request.OutputSchema != nil || result.SchemaEnforcement != llm.SchemaEnforcementLocal {
		t.Fatalf("best-effort was not explicitly local: request=%+v result=%+v", bestEffortProvider.request, result)
	}

	invalidProvider := &fakeProvider{response: llm.Response{Content: `{"ok":"yes"}`}}
	act = newPromptActionWith(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return invalidProvider, nil })
	if _, err := act.DoAny(t.Context(), PromptRequest{
		User: "Return JSON", OutputSchema: &schema, OutputSchemaMode: llm.SchemaModeBestEffort,
	}); err == nil {
		t.Fatal("best-effort local validation accepted invalid output")
	}
}

func TestPromptValidatesToolCallsAndHostResultsAtCompletionBoundary(t *testing.T) {
	closed := false
	argumentSchema := llm.JSONSchema{Type: "object", Properties: map[string]llm.JSONSchema{"count": {Type: "integer"}}, Required: []string{"count"}, AdditionalProperties: &closed}
	tool := llm.ToolSpec{Type: "function", Function: llm.FunctionSpec{
		Name: "count_words", Parameters: argumentSchema, ResultSchema: map[string]any{"type": "string"},
	}}
	badResponseProvider := &fakeProvider{response: llm.Response{ToolCalls: []llm.ToolCall{{
		ID: "call-bad", Type: "function", Function: llm.FunctionCall{Name: "count_words", Arguments: `{}`},
	}}}}
	act := newPromptActionWith(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return badResponseProvider, nil })
	if _, err := act.DoAny(t.Context(), PromptRequest{User: "Count", Tools: []llm.ToolSpec{tool}}); err == nil {
		t.Fatal("invalid provider tool arguments reached the caller")
	}
	if badResponseProvider.calls != 1 {
		t.Fatalf("response validation should follow the provider call, calls=%d", badResponseProvider.calls)
	}

	invalidHistoryProvider := &fakeProvider{}
	act = newPromptActionWith(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return invalidHistoryProvider, nil })
	_, err := act.DoAny(t.Context(), PromptRequest{
		Tools: []llm.ToolSpec{tool}, Messages: []llm.Message{
			{Role: llm.RoleUser, Content: "Count"},
			{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "history", Type: "function", Function: llm.FunctionCall{Name: "count_words", Arguments: `{}`}}}},
		},
	})
	if err == nil || invalidHistoryProvider.calls != 0 {
		t.Fatalf("invalid historical arguments were not rejected before dispatch: err=%v calls=%d", err, invalidHistoryProvider.calls)
	}

	invalidResultProvider := &fakeProvider{}
	act = newPromptActionWith(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return invalidResultProvider, nil })
	_, err = act.DoAny(t.Context(), PromptRequest{
		Tools: []llm.ToolSpec{tool}, Messages: []llm.Message{
			{Role: llm.RoleUser, Content: "Count"},
			{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "history", Type: "function", Function: llm.FunctionCall{Name: "count_words", Arguments: `{"count":2}`}}}},
			{Role: llm.RoleTool, ToolCallID: "history", Name: "count_words", Content: `2`},
		},
	})
	if err == nil || invalidResultProvider.calls != 0 {
		t.Fatalf("invalid host result was not rejected before dispatch: err=%v calls=%d", err, invalidResultProvider.calls)
	}
}

func TestPromptBestEffortDowngradesStrictToolArgumentsAndReportsLocalValidation(t *testing.T) {
	closed := false
	tool := llm.ToolSpec{Type: "function", Function: llm.FunctionSpec{
		Name: "noop", Parameters: llm.JSONSchema{Type: "object", Properties: map[string]llm.JSONSchema{}, AdditionalProperties: &closed}, Strict: true,
	}}
	provider := &fakeProvider{response: llm.Response{ToolCalls: []llm.ToolCall{{
		ID: "call-local", Type: "function", Function: llm.FunctionCall{Name: "noop", Arguments: `{}`},
	}}}}
	act := newPromptActionWith(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return provider, nil })
	value, err := act.DoAny(t.Context(), PromptRequest{
		User: "Use noop", Tools: []llm.ToolSpec{tool}, OutputSchemaMode: llm.SchemaModeBestEffort,
	})
	if err != nil {
		t.Fatalf("best-effort tool validation: %v", err)
	}
	result, ok := value.(PromptResult)
	if !ok {
		t.Fatalf("prompt result has type %T", value)
	}
	if provider.request.Tools[0].Function.Strict || result.SchemaEnforcement != llm.SchemaEnforcementLocal {
		t.Fatalf("strict tool was not downgraded and locally validated: request=%+v result=%+v", provider.request, result)
	}
}

type schemaCapableFakeProvider struct {
	*fakeProvider
}

func (*schemaCapableFakeProvider) Capabilities(string) llm.ProviderCapabilities {
	return llm.ProviderCapabilities{OutputSchema: llm.CapabilityConditional, OutputSchemaWithTools: llm.CapabilityConditional, StrictToolArguments: llm.CapabilityConditional}
}

func (*schemaCapableFakeProvider) SupportsOutputSchema(_ string, schema llm.JSONSchema, _ bool) (supported bool, reason string) {
	return schema.ValidateDefinition() == nil, "test schema rejected"
}
