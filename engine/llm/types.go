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

type Request struct {
	Model           string     `json:"model,omitempty"`
	Messages        []Message  `json:"messages"`
	Temperature     float64    `json:"temperature,omitempty"`
	MaxTokens       int        `json:"max_tokens,omitempty"`
	JSONMode        bool       `json:"json_mode,omitempty"`
	Tools           []ToolSpec `json:"tools,omitempty"`
	IncludeThinking bool       `json:"include_thinking,omitempty"`
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

// CostMicros returns the cost of this turn in microdollars-equivalent
// units of the currency reported by the rate card. The currency itself
// is available via Cost() when needed.
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
