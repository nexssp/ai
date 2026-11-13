package llm

import (
	"context"
	"iter"
	"time"

	"github.com/nexssp/cost"
	"github.com/nexssp/kernel/action"
)

type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type Message struct {
	Role             Role       `json:"role"`
	Content          string     `json:"content"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	Name             string     `json:"name,omitempty"`
	ToolCallID       string     `json:"tool_call_id,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
}

type ToolSpec struct {
	Type     string       `json:"type"`
	Function FunctionSpec `json:"function"`
}

type FunctionSpec struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters"`
	Example     any    `json:"example,omitempty"`
}

// ToolChoice selects how a provider handles the supplied function tools.
// These common modes are supported by OpenAI-compatible Complete adapters.
type ToolChoice string

const (
	ToolChoiceAuto     ToolChoice = "auto"
	ToolChoiceNone     ToolChoice = "none"
	ToolChoiceRequired ToolChoice = "required"
)

// ToolCallingProvider advertises structured function-tool support for Complete.
// Stream remains text-only unless a separate structured streaming API is added.
type ToolCallingProvider interface {
	SupportsToolCalls() bool
}

// ReasoningEffort is a provider/model-specific reasoning budget. Not all
// providers accept every value; unsupported values are passed to the API and
// reported using its normal typed provider error.
type ReasoningEffort string

const (
	ReasoningEffortNone    ReasoningEffort = "none"
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	ReasoningEffortLow     ReasoningEffort = "low"
	ReasoningEffortMedium  ReasoningEffort = "medium"
	ReasoningEffortHigh    ReasoningEffort = "high"
	ReasoningEffortXHigh   ReasoningEffort = "xhigh"
	ReasoningEffortMax     ReasoningEffort = "max"
	ReasoningEffortInstant ReasoningEffort = "instant"
)

// JSONSchema is the typed subset of JSON Schema used for provider-constrained
// structured outputs. Provider adapters translate it to their native API shape.
type JSONSchema struct {
	Type                 string                `json:"type"`
	Properties           map[string]JSONSchema `json:"properties,omitempty"`
	Required             []string              `json:"required,omitempty"`
	Items                *JSONSchema           `json:"items,omitempty"`
	Enum                 []string              `json:"enum,omitempty"`
	AdditionalProperties *bool                 `json:"additionalProperties,omitempty"`
}

type Request struct {
	Model string `json:"model,omitempty"`
	// The following fields are host accounting metadata. Provider adapters build
	// their own wire DTOs and must not send these fields to upstream APIs.
	CandidateID                 string          `json:"candidate_id,omitempty"`
	AttemptID                   string          `json:"attempt_id,omitempty"`
	RateCardID                  string          `json:"rate_card_id,omitempty"`
	ExpectedCurrency            *cost.Currency  `json:"expected_currency,omitempty"`
	EstimatedPromptTokens       int             `json:"estimated_prompt_tokens,omitempty"`
	EstimatedCachedPromptTokens int             `json:"estimated_cached_prompt_tokens,omitempty"`
	EstimatedCompletionTokens   int             `json:"estimated_completion_tokens,omitempty"`
	Messages                    []Message       `json:"messages"`
	Temperature                 float64         `json:"temperature,omitempty"`
	MaxTokens                   int             `json:"max_tokens,omitempty"`
	JSONMode                    bool            `json:"json_mode,omitempty"`
	OutputSchema                *JSONSchema     `json:"output_schema,omitempty"` // Optional schema constraint; currently applied by Anthropic Messages.
	ReasoningEffort             ReasoningEffort `json:"reasoning_effort,omitempty"`
	Tools                       []ToolSpec      `json:"tools,omitempty"`
	ToolChoice                  ToolChoice      `json:"tool_choice,omitempty"`
	IncludeThinking             bool            `json:"include_thinking,omitempty"`
}

type Response struct {
	Content            string        `json:"content"`
	ReasoningContent   string        `json:"reasoning_content,omitempty"`
	Model              string        `json:"model"`
	PromptTokens       int           `json:"prompt_tokens"`
	CachedPromptTokens int           `json:"cached_prompt_tokens,omitempty"`
	CompTokens         int           `json:"completion_tokens"`
	FinishReason       string        `json:"finish_reason,omitempty"`
	ToolCalls          []ToolCall    `json:"tool_calls,omitempty"`
	Duration           time.Duration `json:"duration,omitempty"`
}

// Truncated reports whether the provider stopped because it hit the
// max_tokens ceiling. Callers should treat empty Content with
// Truncated()==true as "raise the limit", not "model failed".
func (r Response) Truncated() bool {
	return r.FinishReason == "length"
}

// CostMicros returns the cost of this turn in micro-units of the currency
// reported by the rate card. The currency itself is available via Cost().
func (r Response) CostMicros() cost.Micro {
	m, _, ok := CalculateCost(
		r.Model,
		r.PromptTokens,
		r.CachedPromptTokens,
		r.CompTokens,
		time.Now(),
	)
	if !ok {
		return 0
	}

	return m
}

// Cost returns both the amount and its currency.
func (r Response) Cost() (cost.Micro, cost.Currency, bool) { // bool: When ok is false the model has no rate card
	return CalculateCost(
		r.Model,
		r.PromptTokens,
		r.CachedPromptTokens,
		r.CompTokens,
		time.Now(),
	)
}

type Provider interface {
	Name() string
	Complete(ctx context.Context, req Request) (Response, error)
	Stream(ctx context.Context, req Request) (iter.Seq2[string, error], error)
}

func ProviderAction(name string, p Provider) *action.Builder[Request, Response] {
	if name == "" {
		name = "llm." + p.Name() + ".complete"
	}

	return action.New(name, func(ctx context.Context, req Request) (Response, error) {
		return p.Complete(ctx, req)
	}).Tag("ai", "llm", p.Name())
}

type ModelManager interface {
	Preload(ctx context.Context, model string) error
	Unload(ctx context.Context, model string) error
	IsLoaded(ctx context.Context, model string) bool
}

type Embedder interface {
	Embed(ctx context.Context, texts ...string) ([][]float32, error)
}

type TokenEstimator interface {
	EstimateTokens(messages []Message) int
}

type Config struct {
	Name         string        `json:"name,omitempty" yaml:"name,omitempty"`
	BaseURL      string        `json:"base_url,omitempty" yaml:"base_url,omitempty"`
	APIKey       string        `json:"api_key,omitempty" yaml:"api_key,omitempty"`
	DefaultModel string        `json:"default_model,omitempty" yaml:"default_model,omitempty"`
	Timeout      time.Duration `json:"timeout,omitempty" yaml:"timeout,omitempty"`
	MaxRetries   int           `json:"max_retries,omitempty" yaml:"max_retries,omitempty"`
}
