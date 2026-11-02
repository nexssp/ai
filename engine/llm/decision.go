package llm

import (
	"context"
	"encoding/json"

	"github.com/nexssp/cost"
)

// DecisionQuestionType identifies a structured classification task supported
// by a decision-capable provider.
type DecisionQuestionType string

const (
	DecisionNoul    DecisionQuestionType = "noul"
	DecisionChoice  DecisionQuestionType = "choice"
	DecisionScore   DecisionQuestionType = "score"
	DecisionExtract DecisionQuestionType = "extract"
)

// DecisionCriterion names one candidate or one ordered score level. The
// provider adapter translates the slice into the wire format required by its
// API; callers never need to construct an untyped JSON object.
type DecisionCriterion struct {
	ID          string `json:"id"`
	Description string `json:"description"`
}

// DecisionQuestion describes one named question evaluated against shared
// state. Criteria are ordered for score questions and keyed by ID for choice
// and extract questions.
type DecisionQuestion struct {
	ID               string               `json:"id"`
	Type             DecisionQuestionType `json:"type"`
	Instructions     string               `json:"instructions,omitempty"`
	Criteria         []DecisionCriterion  `json:"criteria,omitempty"`
	TrueDescription  string               `json:"true,omitempty"`
	FalseDescription string               `json:"false,omitempty"`
	Source           string               `json:"source,omitempty"`
	Threshold        *float64             `json:"threshold,omitempty"`
	SnapToWords      *bool                `json:"snap_to_words,omitempty"`
}

// DecisionRequest is the typed provider request. MaxOutputTokens is a required
// positive hard cap; estimated token fields remain host-side accounting metadata.
// State is JSON so applications can pass context without a domain schema.
type DecisionRequest struct {
	Model                       string             `json:"model"`
	AttemptID                   string             `json:"attempt_id,omitempty"`
	RateCardID                  string             `json:"rate_card_id,omitempty"`
	ExpectedCurrency            *cost.Currency     `json:"expected_currency,omitempty"`
	EstimatedPromptTokens       int                `json:"estimated_prompt_tokens,omitempty"`
	EstimatedCachedPromptTokens int                `json:"estimated_cached_prompt_tokens,omitempty"`
	EstimatedCompletionTokens   int                `json:"estimated_completion_tokens,omitempty"`
	MaxOutputTokens             int                `json:"max_output_tokens"`
	State                       json.RawMessage    `json:"state"`
	Questions                   []DecisionQuestion `json:"questions"`
	AllowedIDs                  []string           `json:"allowed_ids,omitempty"`
}

// DecisionProbability is one named outcome probability returned by a provider.
type DecisionProbability struct {
	ID          string  `json:"id"`
	Probability float64 `json:"probability"`
}

// DecisionSpan is an exact source substring found by an extraction question.
type DecisionSpan struct {
	Text       string  `json:"text"`
	Start      int     `json:"start"`
	End        int     `json:"end"`
	Confidence float64 `json:"confidence"`
}

// DecisionExtraction contains exact spans and the probability that a field was
// not present in the source text.
type DecisionExtraction struct {
	Field                string         `json:"field"`
	Spans                []DecisionSpan `json:"spans"`
	NotStatedProbability float64        `json:"not_stated_probability"`
}

// DecisionAnswer is one typed answer, returned in the same order as the
// questions in DecisionRequest.
type DecisionAnswer struct {
	ID            string                `json:"id"`
	Type          DecisionQuestionType  `json:"type"`
	Noul          *float64              `json:"noul,omitempty"`
	Choice        string                `json:"choice,omitempty"`
	Score         *float64              `json:"score,omitempty"`
	Probabilities []DecisionProbability `json:"probabilities,omitempty"`
	Confidence    *float64              `json:"confidence,omitempty"`
	Legend        []DecisionCriterion   `json:"legend,omitempty"`
	Extractions   []DecisionExtraction  `json:"extractions,omitempty"`
}

// DecisionResponse is a structured result from a decision-capable model.
type DecisionResponse struct {
	Model   string           `json:"model"`
	Answers []DecisionAnswer `json:"answers"`
	Usage   Usage            `json:"usage"`
}

// DecisionProvider exposes structured decisions independently from chat
// completions. This matters for providers whose decision endpoint is not a
// chat-completions endpoint.
type DecisionProvider interface {
	Name() string
	Decide(ctx context.Context, req DecisionRequest) (DecisionResponse, error)
}
