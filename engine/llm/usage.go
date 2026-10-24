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
// Usage is intentionally a value type: it is small (40 bytes on
// 64-bit), copyable, and safe to embed.
type Usage struct {
	Model              string        `json:"model,omitempty"`
	PromptTokens       int           `json:"prompt_tokens,omitempty"`
	CachedPromptTokens int           `json:"cached_prompt_tokens,omitempty"`
	CompTokens         int           `json:"comp_tokens,omitempty"`
	FinishReason       string        `json:"finish_reason,omitempty"`
	Duration           time.Duration `json:"duration,omitempty"`
}

// FromResponse extracts Usage from a completed Response.
func FromResponse(r Response) Usage {
	return Usage{
		Model:              r.Model,
		PromptTokens:       r.PromptTokens,
		CachedPromptTokens: r.CachedPromptTokens,
		CompTokens:         r.CompTokens,
		FinishReason:       r.FinishReason,
		Duration:           r.Duration,
	}
}

// Tokens returns (input, output) token counts.
func (u Usage) Tokens() (int, int) { return u.PromptTokens, u.CompTokens }

// CostMicros returns the cost in micro-units of the rate-card currency.
// Zero when the model has no rate card — that is intentional; the
// observer prints "-" rather than "0.000000" for unknown models.
func (u Usage) CostMicros() cost.Micro {
	m, _, ok := CalculateCost(
		u.Model,
		u.PromptTokens,
		u.CachedPromptTokens,
		u.CompTokens,
		time.Now(),
	)
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
