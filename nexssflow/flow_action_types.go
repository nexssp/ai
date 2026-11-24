package nexssflow

import (
	"encoding/json"
	"fmt"

	"github.com/nexssp/cost"

	"github.com/nexssp/ai/engine/llm"
)

// flowJSONObject keeps opaque JSON objects compatible with Flow's typed schemas.
type flowJSONObject map[string]any

// Flow-facing requests use JSON strings for currency. cost.Currency is a packed
// [3]byte domain type with a custom JSON codec, so its wire shape cannot be
// represented as a primitive @schema field without an explicit adapter.
type flowCandidateAttemptPlan struct {
	CandidateID        string              `json:"candidate_id"`
	AttemptID          string              `json:"attempt_id"`
	Provider           string              `json:"provider"`
	Model              string              `json:"model"`
	BaseURL            string              `json:"base_url,omitempty"`
	RateCardID         string              `json:"rate_card_id"`
	ExpectedCurrency   string              `json:"expected_currency"`
	PromptTokens       int                 `json:"prompt_tokens"`
	CachedPromptTokens int                 `json:"cached_prompt_tokens,omitempty"`
	CompletionTokens   int                 `json:"completion_tokens"`
	User               string              `json:"user"`
	System             string              `json:"system,omitempty"`
	MaxTokens          int                 `json:"max_tokens,omitempty"`
	Temperature        *float64            `json:"temperature,omitempty"`
	JSONMode           bool                `json:"json_mode,omitempty"`
	ReasoningEffort    llm.ReasoningEffort `json:"reasoning_effort,omitempty"`
}

func (request flowCandidateAttemptPlan) toDomain() (CandidateAttemptPlan, error) {
	currency, err := parseFlowCurrency(request.ExpectedCurrency)
	if err != nil {
		return CandidateAttemptPlan{}, fmt.Errorf("candidate attempt expected_currency: %w", err)
	}
	return CandidateAttemptPlan{
		CandidateID:        request.CandidateID,
		AttemptID:          request.AttemptID,
		Provider:           request.Provider,
		Model:              request.Model,
		BaseURL:            request.BaseURL,
		RateCardID:         request.RateCardID,
		ExpectedCurrency:   currency,
		PromptTokens:       request.PromptTokens,
		CachedPromptTokens: request.CachedPromptTokens,
		CompletionTokens:   request.CompletionTokens,
		User:               request.User,
		System:             request.System,
		MaxTokens:          request.MaxTokens,
		Temperature:        optionalFloat64(request.Temperature),
		JSONMode:           request.JSONMode,
		ReasoningEffort:    request.ReasoningEffort,
	}, nil
}

type flowDecisionCandidate struct {
	ID          string                   `json:"id"`
	Description string                   `json:"description"`
	Attempt     flowCandidateAttemptPlan `json:"attempt"`
}

type flowDecisionActionRequest struct {
	AttemptID          string                  `json:"attempt_id"`
	Provider           string                  `json:"provider,omitempty"`
	Model              string                  `json:"model,omitempty"`
	BaseURL            string                  `json:"base_url,omitempty"`
	RateCardID         string                  `json:"rate_card_id,omitempty"`
	ExpectedCurrency   *string                 `json:"expected_currency,omitempty"`
	PromptTokens       int                     `json:"prompt_tokens,omitempty"`
	CachedPromptTokens int                     `json:"cached_prompt_tokens,omitempty"`
	CompletionTokens   int                     `json:"completion_tokens,omitempty"`
	MaxOutputTokens    int                     `json:"max_output_tokens"`
	State              flowJSONObject          `json:"state"`
	Questions          []llm.DecisionQuestion  `json:"questions"`
	AllowedIDs         []string                `json:"allowed_ids,omitempty"`
	Candidates         []flowDecisionCandidate `json:"candidates,omitempty"`
}

func (request flowDecisionActionRequest) toDomain() (DecisionActionRequest, error) {
	expectedCurrencyValue, hasExpectedCurrency, err := parseOptionalFlowCurrency(request.ExpectedCurrency)
	if err != nil {
		return DecisionActionRequest{}, err
	}
	state, err := flowJSONRawMessage(request.State, "state")
	if err != nil {
		return DecisionActionRequest{}, err
	}
	var candidates []DecisionCandidate
	if request.Candidates != nil {
		candidates = make([]DecisionCandidate, len(request.Candidates))
		for index := range request.Candidates {
			candidate := &request.Candidates[index]
			attempt, err := candidate.Attempt.toDomain()
			if err != nil {
				return DecisionActionRequest{}, fmt.Errorf("candidates[%d]: %w", index, err)
			}
			candidates[index] = DecisionCandidate{
				ID:          candidate.ID,
				Description: candidate.Description,
				Attempt:     attempt,
			}
		}
	}
	var expectedCurrency *cost.Currency
	if hasExpectedCurrency {
		expectedCurrency = &expectedCurrencyValue
	}
	return DecisionActionRequest{
		AttemptID:          request.AttemptID,
		Provider:           request.Provider,
		Model:              request.Model,
		BaseURL:            request.BaseURL,
		RateCardID:         request.RateCardID,
		ExpectedCurrency:   expectedCurrency,
		PromptTokens:       request.PromptTokens,
		CachedPromptTokens: request.CachedPromptTokens,
		CompletionTokens:   request.CompletionTokens,
		MaxOutputTokens:    request.MaxOutputTokens,
		State:              state,
		Questions:          request.Questions,
		AllowedIDs:         request.AllowedIDs,
		Candidates:         candidates,
	}, nil
}

type flowPromptRequest struct {
	CandidateID        string                    `json:"candidate_id,omitempty"`
	AttemptID          string                    `json:"attempt_id,omitempty"`
	RateCardID         string                    `json:"rate_card_id,omitempty"`
	ExpectedCurrency   *string                   `json:"expected_currency,omitempty"`
	PromptTokens       int                       `json:"prompt_tokens,omitempty"`
	CachedPromptTokens int                       `json:"cached_prompt_tokens,omitempty"`
	CompletionTokens   int                       `json:"completion_tokens,omitempty"`
	Context            flowJSONObject            `json:"context,omitempty"`
	Messages           []llm.Message             `json:"messages,omitempty"`
	User               string                    `json:"user,omitempty"`
	System             string                    `json:"system,omitempty"`
	Tools              []llm.ToolSpec            `json:"tools,omitempty"`
	ToolChoice         llm.ToolChoice            `json:"tool_choice,omitempty"`
	Provider           string                    `json:"provider,omitempty"`
	Model              string                    `json:"model,omitempty"`
	BaseURL            string                    `json:"base_url,omitempty"`
	MaxTokens          int                       `json:"max_tokens,omitempty"`
	Temperature        *float64                  `json:"temperature,omitempty"`
	JSONMode           bool                      `json:"json_mode,omitempty"`
	OutputSchema       flowJSONObject            `json:"output_schema,omitempty"`
	OutputSchemaMode   llm.SchemaMode            `json:"output_schema_mode,omitempty"`
	ReasoningEffort    llm.ReasoningEffort       `json:"reasoning_effort,omitempty"`
	Fallback           *flowCandidateAttemptPlan `json:"fallback,omitempty"`
}

func (request flowPromptRequest) toDomain() (PromptRequest, error) {
	expectedCurrencyValue, hasExpectedCurrency, err := parseOptionalFlowCurrency(request.ExpectedCurrency)
	if err != nil {
		return PromptRequest{}, err
	}
	contextJSON, err := flowJSONRawMessage(request.Context, "context")
	if err != nil {
		return PromptRequest{}, err
	}
	var outputSchema *llm.JSONSchema
	if request.OutputSchema != nil {
		outputSchemaJSON, err := flowJSONRawMessage(request.OutputSchema, "output_schema")
		if err != nil {
			return PromptRequest{}, err
		}
		var schema llm.JSONSchema
		if err := json.Unmarshal(outputSchemaJSON, &schema); err != nil {
			return PromptRequest{}, fmt.Errorf("output_schema must be a supported JSON Schema: %w", err)
		}
		if err := schema.ValidateDefinition(); err != nil {
			return PromptRequest{}, fmt.Errorf("output_schema: %w", err)
		}
		outputSchema = &schema
	}
	var fallback *CandidateAttemptPlan
	if request.Fallback != nil {
		converted, err := request.Fallback.toDomain()
		if err != nil {
			return PromptRequest{}, fmt.Errorf("fallback: %w", err)
		}
		fallback = &converted
	}
	var expectedCurrency *cost.Currency
	if hasExpectedCurrency {
		expectedCurrency = &expectedCurrencyValue
	}
	return PromptRequest{
		CandidateID:        request.CandidateID,
		AttemptID:          request.AttemptID,
		RateCardID:         request.RateCardID,
		ExpectedCurrency:   expectedCurrency,
		PromptTokens:       request.PromptTokens,
		CachedPromptTokens: request.CachedPromptTokens,
		CompletionTokens:   request.CompletionTokens,
		Context:            contextJSON,
		Messages:           request.Messages,
		User:               request.User,
		System:             request.System,
		Tools:              request.Tools,
		ToolChoice:         request.ToolChoice,
		Provider:           request.Provider,
		Model:              request.Model,
		BaseURL:            request.BaseURL,
		MaxTokens:          request.MaxTokens,
		Temperature:        optionalFloat64(request.Temperature),
		JSONMode:           request.JSONMode,
		OutputSchema:       outputSchema,
		OutputSchemaMode:   request.OutputSchemaMode,
		ReasoningEffort:    request.ReasoningEffort,
		Fallback:           fallback,
	}, nil
}

type flowCostEstimateRequest struct {
	CandidateID        string                     `json:"candidate_id,omitempty"`
	AttemptID          string                     `json:"attempt_id"`
	Model              string                     `json:"model"`
	RateCardID         string                     `json:"rate_card_id"`
	ExpectedCurrency   string                     `json:"expected_currency"`
	PromptTokens       int                        `json:"prompt_tokens"`
	CachedPromptTokens int                        `json:"cached_prompt_tokens,omitempty"`
	CompletionTokens   int                        `json:"completion_tokens"`
	Decision           *flowDecisionActionRequest `json:"decision,omitempty"`
	Prompt             *flowPromptRequest         `json:"prompt,omitempty"`
}

func (request flowCostEstimateRequest) toDomain() (CostEstimateRequest, error) {
	currency, err := parseFlowCurrency(request.ExpectedCurrency)
	if err != nil {
		return CostEstimateRequest{}, fmt.Errorf("expected_currency: %w", err)
	}
	var decision *DecisionActionRequest
	if request.Decision != nil {
		converted, err := request.Decision.toDomain()
		if err != nil {
			return CostEstimateRequest{}, fmt.Errorf("decision: %w", err)
		}
		decision = &converted
	}
	var prompt *PromptRequest
	if request.Prompt != nil {
		converted, err := request.Prompt.toDomain()
		if err != nil {
			return CostEstimateRequest{}, fmt.Errorf("prompt: %w", err)
		}
		prompt = &converted
	}
	return CostEstimateRequest{
		CandidateID:        request.CandidateID,
		AttemptID:          request.AttemptID,
		Model:              request.Model,
		RateCardID:         request.RateCardID,
		ExpectedCurrency:   currency,
		PromptTokens:       request.PromptTokens,
		CachedPromptTokens: request.CachedPromptTokens,
		CompletionTokens:   request.CompletionTokens,
		Decision:           decision,
		Prompt:             prompt,
	}, nil
}

func parseOptionalFlowCurrency(value *string) (cost.Currency, bool, error) {
	if value == nil {
		return cost.Currency{}, false, nil
	}
	currency, err := parseFlowCurrency(*value)
	if err != nil {
		return cost.Currency{}, false, err
	}
	return currency, true, nil
}

func flowJSONRawMessage(value flowJSONObject, field string) (json.RawMessage, error) {
	if value == nil {
		return json.RawMessage{}, nil
	}
	raw, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("%s must be a valid JSON value: %w", field, err)
	}
	return raw, nil
}

func optionalFloat64(value *float64) float64 {
	if value == nil {
		return 0
	}
	return *value
}

func parseFlowCurrency(value string) (cost.Currency, error) {
	var currency cost.Currency
	if err := currency.UnmarshalText([]byte(value)); err != nil {
		return cost.Currency{}, err
	}
	return currency, nil
}
