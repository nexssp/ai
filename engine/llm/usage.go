package llm

import (
	"time"

	"github.com/nexssp/cost"
)

// Usage is the LLM accounting that travels with every agent result.
//
// Any result that embeds Usage automatically satisfies the runner
// observer's token-cost interface, so the metrics table picks it up
// without per-result Tokens()/CostUSD() methods.
//
// Usage is a copyable value object. Currency is a pointer only because
// unknown-rate-card results have no currency value.
type Usage struct {
	Model              string         `json:"model,omitempty"`
	RateCardID         string         `json:"rate_card_id,omitempty"`
	ExpectedCurrency   *cost.Currency `json:"expected_currency,omitempty"`
	PromptTokens       int            `json:"prompt_tokens,omitempty"`
	CachedPromptTokens int            `json:"cached_prompt_tokens,omitempty"`
	CompTokens         int            `json:"comp_tokens,omitempty"`
	FinishReason       string         `json:"finish_reason,omitempty"`
	Duration           time.Duration  `json:"duration,omitempty"`
	CostMicro          cost.Micro     `json:"cost_micros,omitempty"`
	Currency           *cost.Currency `json:"currency,omitempty"`
	CostKnown          bool           `json:"cost_known"`
}

// FromResponse extracts Usage from a completed Response.
func FromResponse(r Response) Usage {
	return usageFromTokens(r.Model, r.PromptTokens, r.CachedPromptTokens, r.CompTokens, r.FinishReason, r.Duration, "", nil)
}

// FromResponseForRateCard extracts usage using the exact caller-selected card.
// A missing card or currency mismatch remains unknown; it never falls back to
// a model-only card lookup.
func FromResponseForRateCard(r Response, rateCardID string, expectedCurrency cost.Currency) Usage {
	expected := expectedCurrency
	return usageFromTokens(r.Model, r.PromptTokens, r.CachedPromptTokens, r.CompTokens, r.FinishReason, r.Duration, rateCardID, &expected)
}

func usageFromTokens(model string, promptTokens, cachedTokens, completionTokens int, finishReason string, duration time.Duration, rateCardID string, expectedCurrency *cost.Currency) Usage {
	u := Usage{
		Model:              model,
		RateCardID:         rateCardID,
		ExpectedCurrency:   expectedCurrency,
		PromptTokens:       promptTokens,
		CachedPromptTokens: cachedTokens,
		CompTokens:         completionTokens,
		FinishReason:       finishReason,
		Duration:           duration,
	}
	var amount cost.Micro
	var currency cost.Currency
	var known bool
	if rateCardID != "" {
		if expectedCurrency == nil {
			return u
		}
		amount, currency, known = CalculateCostForRateCard(model, rateCardID, *expectedCurrency, promptTokens, cachedTokens, completionTokens, time.Now())
	} else {
		amount, currency, known = CalculateCost(model, promptTokens, cachedTokens, completionTokens, time.Now())
	}
	u.CostMicro = amount
	u.CostKnown = known
	if known {
		u.Currency = &currency
	}
	return u
}

// Tokens returns (input, output) token counts.
func (u Usage) Tokens() (prompt, completion int) { return u.PromptTokens, u.CompTokens }

// CostMicros returns the cost in micro-units of the rate-card currency.
// Zero when the model has no rate card — that is intentional; the
// observer prints "-" rather than "0.000000" for unknown models.
func (u Usage) CostMicros() cost.Micro {
	if u.CostKnown {
		return u.CostMicro
	}
	var m cost.Micro
	var ok bool
	if u.RateCardID != "" {
		if u.ExpectedCurrency == nil {
			return 0
		}
		m, _, ok = CalculateCostForRateCard(u.Model, u.RateCardID, *u.ExpectedCurrency, u.PromptTokens, u.CachedPromptTokens, u.CompTokens, time.Now())
	} else {
		m, _, ok = CalculateCost(u.Model, u.PromptTokens, u.CachedPromptTokens, u.CompTokens, time.Now())
	}
	if !ok {
		return 0
	}

	return m
}

// CostUSD is the compat method expected by the runner observer's
// tokenCostReporter interface. The value is micro-units of whatever
// currency the rate card declares; the name is historical. Callers
// that need the currency should call CalculateCost directly.
func (u Usage) CostUSD() int64 { return int64(u.CostMicros()) }

// Truncated reports whether the provider stopped because it hit the
// max-tokens ceiling.
func (u Usage) Truncated() bool { return u.FinishReason == "length" }
