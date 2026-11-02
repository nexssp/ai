package llm

import (
	"strings"
	"testing"
	"time"

	"github.com/nexssp/cost"
)

func TestPriceCatalogStartsEmptyAndRequiresExternalCurrency(t *testing.T) {
	catalog := NewPriceCatalog()
	if got := catalog.Models(); len(got) != 0 {
		t.Fatalf("default catalog models = %v, want empty", got)
	}
	if strings.Contains(DefaultPricingTemplate, "USD") || strings.Contains(DefaultPricingTemplate, "gpt-") {
		t.Fatalf("default pricing template contains bundled model/currency claims: %s", DefaultPricingTemplate)
	}

	err := catalog.LoadFromYAML([]byte(`models:
  model-from-config:
    peak:
      prompt_per_1k: 1
      comp_per_1k: 2
`))
	if err == nil {
		t.Fatal("expected rate cards without currency to be rejected")
	}
	if got := catalog.Models(); len(got) != 0 {
		t.Fatalf("catalog mutated after invalid load: %v", got)
	}
}

func TestPriceCatalogLoadsCurrencyAndCalculatesFromExternalRateCard(t *testing.T) {
	catalog := NewPriceCatalog()
	const model = "external-rate-card-test-model"
	if err := catalog.LoadFromYAML([]byte(`currency: EUR
models:
  external-rate-card-test-model:
    rate_card_id: external-test-eur-2026
    peak:
      prompt_per_1k: 1
      prompt_cache_hit_per_1k: 0.5
      comp_per_1k: 2
`)); err != nil {
		t.Fatalf("LoadFromYAML: %v", err)
	}

	previous := GlobalCatalog
	GlobalCatalog = catalog
	defer func() { GlobalCatalog = previous }()

	amount, currency, ok := CalculateCost(model, 10, 5, 3, time.Now())
	if !ok || amount != cost.Micro(13500) || currency != cost.EUR {
		t.Fatalf("CalculateCost = (%v, %v, %v), want (13500 micro, EUR, true)", amount, currency, ok)
	}
}

func TestPriceCatalogRequiresCardIDAndCurrency(t *testing.T) {
	previous := GlobalCatalog
	GlobalCatalog = NewPriceCatalog()
	defer func() { GlobalCatalog = previous }()

	if err := GlobalCatalog.Register("missing-id", PriceCard{Currency: cost.EUR, Peak: Price{PromptPer1k: 1}}); err == nil {
		t.Fatal("expected rate card without ID to be rejected")
	}
	if err := GlobalCatalog.Register("missing-currency", PriceCard{RateCardID: "card-a", Peak: Price{PromptPer1k: 1}}); err == nil {
		t.Fatal("expected rate card without currency to be rejected")
	}
	if err := GlobalCatalog.LoadFromYAML([]byte(`currency: EUR
models:
  missing-rate-card-id:
    peak:
      prompt_per_1k: 1
`)); err == nil {
		t.Fatal("expected YAML card without ID to be rejected")
	}
}

func TestCostEstimateSelectsExactRateCardAndCurrency(t *testing.T) {
	previous := GlobalCatalog
	GlobalCatalog = NewPriceCatalog()
	defer func() { GlobalCatalog = previous }()

	const model = "two-rate-cards-model"
	if err := GlobalCatalog.Register(model, PriceCard{
		RateCardID: "customer-eur-2026",
		Currency:   cost.EUR,
		Peak:       Price{PromptPer1k: 1, CompPer1k: 2},
	}); err != nil {
		t.Fatal(err)
	}
	if err := GlobalCatalog.Register(model, PriceCard{
		RateCardID: "customer-usd-2026",
		Currency:   cost.USD,
		Peak:       Price{PromptPer1k: 10, CompPer1k: 20},
	}); err != nil {
		t.Fatal(err)
	}
	if _, _, ok := CalculateCost(model, 10, 0, 3, time.Now()); ok {
		t.Fatal("model-only lookup should fail closed when multiple cards exist")
	}
	amount, currency, ok := CalculateCostForRateCard(model, "customer-eur-2026", cost.EUR, 10, 0, 3, time.Now())
	if !ok || amount != cost.Micro(16000) || currency != cost.EUR {
		t.Fatalf("exact card estimate = (%v, %v, %v), want (16000 micro, EUR, true)", amount, currency, ok)
	}
	if _, _, ok := CalculateCostForRateCard(model, "customer-eur-2026", cost.USD, 10, 0, 3, time.Now()); ok {
		t.Fatal("currency mismatch should fail closed")
	}
	if _, _, ok := CalculateCostForRateCard(model, "missing-card", cost.EUR, 10, 0, 3, time.Now()); ok {
		t.Fatal("unknown card should fail closed")
	}
}

func TestUsageKeepsExplicitRateCardAndNeverFallsBackToModelOnly(t *testing.T) {
	previous := GlobalCatalog
	GlobalCatalog = NewPriceCatalog()
	defer func() { GlobalCatalog = previous }()

	const model = "usage-exact-card-model"
	for _, card := range []PriceCard{
		{RateCardID: "customer-eur", Currency: cost.EUR, Peak: Price{PromptPer1k: 1, CompPer1k: 2}},
		{RateCardID: "customer-usd", Currency: cost.USD, Peak: Price{PromptPer1k: 10, CompPer1k: 20}},
	} {
		if err := GlobalCatalog.Register(model, card); err != nil {
			t.Fatal(err)
		}
	}

	response := Response{Model: model, PromptTokens: 10, CompTokens: 3}
	usage := FromResponseForRateCard(response, "customer-eur", cost.EUR)
	if !usage.CostKnown || usage.CostMicro != cost.Micro(16000) || usage.Currency == nil || *usage.Currency != cost.EUR || usage.RateCardID != "customer-eur" || usage.ExpectedCurrency == nil || *usage.ExpectedCurrency != cost.EUR {
		t.Fatalf("explicit card usage = %+v", usage)
	}
	unknown := FromResponseForRateCard(response, "missing-card", cost.EUR)
	if unknown.CostKnown || unknown.CostMicros() != 0 || unknown.RateCardID != "missing-card" {
		t.Fatalf("unknown explicit card fell back to a model-only estimate: %+v", unknown)
	}
	if inferred := FromResponse(response); inferred.CostKnown {
		t.Fatalf("ambiguous model-only usage should remain unknown: %+v", inferred)
	}
}

func TestExactRateCardEstimateRoundsUpToMicro(t *testing.T) {
	previous := GlobalCatalog
	GlobalCatalog = NewPriceCatalog()
	defer func() { GlobalCatalog = previous }()
	if err := GlobalCatalog.Register("submicro-model", PriceCard{
		RateCardID: "submicro-eur",
		Currency:   cost.EUR,
		Peak:       Price{PromptPer1k: 0.0000009},
	}); err != nil {
		t.Fatal(err)
	}
	amount, currency, known := CalculateCostForRateCard("submicro-model", "submicro-eur", cost.EUR, 1, 0, 0, time.Now())
	if !known || amount != cost.Micro(1) || currency != cost.EUR {
		t.Fatalf("sub-micro exact estimate = (%v, %v, %v), want (1 micro, EUR, true)", amount, currency, known)
	}
}
