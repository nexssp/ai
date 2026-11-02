package nexssflow

import (
	"errors"
	"testing"

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
