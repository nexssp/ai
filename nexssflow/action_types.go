package nexssflow

import (
	"encoding/json"

	"github.com/nexssp/cost"

	"github.com/nexssp/ai/engine/llm"
)

// PromptRequest is the strongly typed input to ai.prompt. CandidateID and
// AttemptID identify one provider attempt. Fallback is inert metadata for the
// enclosing NFlow error handler; ai.prompt never retries it itself.
type PromptRequest struct {
	CandidateID      string         `json:"candidate_id,omitempty"`
	AttemptID        string         `json:"attempt_id,omitempty"`
	RateCardID       string         `json:"rate_card_id,omitempty"`
	ExpectedCurrency *cost.Currency `json:"expected_currency,omitempty"`
	// PromptTokens, CachedPromptTokens, and CompletionTokens are the caller's
	// preflight basis, not provider-reported usage. The result Usage carries actuals.
	PromptTokens       int                   `json:"prompt_tokens,omitempty"`
	CachedPromptTokens int                   `json:"cached_prompt_tokens,omitempty"`
	CompletionTokens   int                   `json:"completion_tokens,omitempty"`
	Context            json.RawMessage       `json:"context,omitempty"`
	User               string                `json:"user"`
	System             string                `json:"system,omitempty"`
	Provider           string                `json:"provider,omitempty"`
	Model              string                `json:"model,omitempty"`
	BaseURL            string                `json:"base_url,omitempty"`
	MaxTokens          int                   `json:"max_tokens,omitempty"`
	Temperature        float64               `json:"temperature,omitempty"`
	JSONMode           bool                  `json:"json_mode,omitempty"`
	ReasoningEffort    llm.ReasoningEffort   `json:"reasoning_effort,omitempty"`
	Fallback           *CandidateAttemptPlan `json:"fallback,omitempty"`
}

// PromptResult is one provider attempt. It intentionally does not merge the
// entire incoming Flow state into the output; callers retain only declared,
// inspectable fields and can use IDs to account attempts independently.
type PromptResult struct {
	CandidateID        string          `json:"candidate_id,omitempty"`
	AttemptID          string          `json:"attempt_id,omitempty"`
	RateCardID         string          `json:"rate_card_id,omitempty"`
	ExpectedCurrency   *cost.Currency  `json:"expected_currency,omitempty"`
	Context            json.RawMessage `json:"context,omitempty"`
	Provider           string          `json:"provider"`
	Content            string          `json:"content"`
	Model              string          `json:"model"`
	PromptTokens       int             `json:"prompt_tokens"`
	CachedPromptTokens int             `json:"cached_prompt_tokens,omitempty"`
	CompTokens         int             `json:"comp_tokens"`
	TotalTokens        int             `json:"total_tokens"`
	Usage              llm.Usage       `json:"usage"`
}

// CandidateAttemptPlan is operator-supplied configuration for one candidate
// provider attempt. No provider, model, price, card, currency, or ID is inferred.
type CandidateAttemptPlan struct {
	CandidateID        string              `json:"candidate_id"`
	AttemptID          string              `json:"attempt_id"`
	Provider           string              `json:"provider"`
	Model              string              `json:"model"`
	BaseURL            string              `json:"base_url,omitempty"`
	RateCardID         string              `json:"rate_card_id"`
	ExpectedCurrency   cost.Currency       `json:"expected_currency"`
	PromptTokens       int                 `json:"prompt_tokens"`
	CachedPromptTokens int                 `json:"cached_prompt_tokens,omitempty"`
	CompletionTokens   int                 `json:"completion_tokens"`
	User               string              `json:"user"`
	System             string              `json:"system,omitempty"`
	MaxTokens          int                 `json:"max_tokens,omitempty"`
	Temperature        float64             `json:"temperature,omitempty"`
	JSONMode           bool                `json:"json_mode,omitempty"`
	ReasoningEffort    llm.ReasoningEffort `json:"reasoning_effort,omitempty"`
}

// DecisionCandidate is one allowlisted choice and its corresponding provider
// attempt plan. ID is the choice ID; CandidateID is reserved for the actual
// prompt/cost-estimate attempt held in Attempt.
type DecisionCandidate struct {
	ID          string               `json:"id"`
	Description string               `json:"description"`
	Attempt     CandidateAttemptPlan `json:"attempt"`
}

// DecisionActionRequest is the strongly typed input to ai.decide. The decision
// call has its own AttemptID and requires a positive MaxOutputTokens cap;
// candidate attempts remain separately identified.
type DecisionActionRequest struct {
	AttemptID          string                 `json:"attempt_id"`
	Provider           string                 `json:"provider,omitempty"`
	Model              string                 `json:"model,omitempty"`
	BaseURL            string                 `json:"base_url,omitempty"`
	RateCardID         string                 `json:"rate_card_id,omitempty"`
	ExpectedCurrency   *cost.Currency         `json:"expected_currency,omitempty"`
	PromptTokens       int                    `json:"prompt_tokens,omitempty"`
	CachedPromptTokens int                    `json:"cached_prompt_tokens,omitempty"`
	CompletionTokens   int                    `json:"completion_tokens,omitempty"`
	MaxOutputTokens    int                    `json:"max_output_tokens"`
	State              json.RawMessage        `json:"state"`
	Questions          []llm.DecisionQuestion `json:"questions"`
	AllowedIDs         []string               `json:"allowed_ids,omitempty"`
	Candidates         []DecisionCandidate    `json:"candidates,omitempty"`
}

// DecisionActionResult keeps provider answers typed and echoes the decision
// AttemptID and operator-supplied candidate attempt plans for NFlow routing.
type DecisionActionResult struct {
	AttemptID          string               `json:"attempt_id"`
	Model              string               `json:"model"`
	RateCardID         string               `json:"rate_card_id,omitempty"`
	ExpectedCurrency   *cost.Currency       `json:"expected_currency,omitempty"`
	PromptTokens       int                  `json:"prompt_tokens,omitempty"`
	CachedPromptTokens int                  `json:"cached_prompt_tokens,omitempty"`
	CompletionTokens   int                  `json:"completion_tokens,omitempty"`
	MaxOutputTokens    int                  `json:"max_output_tokens"`
	Answers            []llm.DecisionAnswer `json:"answers"`
	Usage              llm.Usage            `json:"usage"`
	Candidates         []DecisionCandidate  `json:"candidates,omitempty"`
}

// CostEstimateRequest estimates one attempt against an explicit external rate
// card. For a nested Decision, CompletionTokens must equal its positive
// MaxOutputTokens so the estimate uses the enforced worst-case output cap.
// The action computes no budget decision or reservation and never triggers fallback.
type CostEstimateRequest struct {
	CandidateID        string                 `json:"candidate_id,omitempty"`
	AttemptID          string                 `json:"attempt_id"`
	Model              string                 `json:"model"`
	RateCardID         string                 `json:"rate_card_id"`
	ExpectedCurrency   cost.Currency          `json:"expected_currency"`
	PromptTokens       int                    `json:"prompt_tokens"`
	CachedPromptTokens int                    `json:"cached_prompt_tokens,omitempty"`
	CompletionTokens   int                    `json:"completion_tokens"`
	Decision           *DecisionActionRequest `json:"decision,omitempty"`
	Prompt             *PromptRequest         `json:"prompt,omitempty"`
}

// CostEstimateResult reports the exact card calculation, or an explicit
// unknown estimate when the selected card is absent or currency mismatches.
type CostEstimateResult struct {
	CandidateID        string                 `json:"candidate_id,omitempty"`
	AttemptID          string                 `json:"attempt_id"`
	Model              string                 `json:"model"`
	RateCardID         string                 `json:"rate_card_id"`
	ExpectedCurrency   cost.Currency          `json:"expected_currency"`
	PromptTokens       int                    `json:"prompt_tokens"`
	CachedPromptTokens int                    `json:"cached_prompt_tokens"`
	CompletionTokens   int                    `json:"completion_tokens"`
	Known              bool                   `json:"known"`
	Issue              string                 `json:"issue,omitempty"`
	CostMicros         cost.Micro             `json:"cost_micros"`
	Currency           *cost.Currency         `json:"currency,omitempty"`
	Decision           *DecisionActionRequest `json:"decision,omitempty"`
	Prompt             *PromptRequest         `json:"prompt,omitempty"`
}
