package llm

import (
	"bytes"
	"context"
	"io"
	"iter"
	"net/http"
	"strings"
	"time"

	"github.com/nexssp/kernel/action"
	"github.com/nexssp/kernel/xerr"
	"github.com/nexssp/transport/codec"
)

type OpenAIProvider struct {
	cfg        Config
	client     *http.Client
	codec      codec.Codec
	execAction *action.BuiltAction[Request, Response]
}

type openAIRequestPayload struct {
	Model           string                `json:"model"`
	Messages        []Message             `json:"messages"`
	Temperature     float64               `json:"temperature"`
	MaxTokens       int                   `json:"max_tokens,omitempty"`
	ReasoningEffort ReasoningEffort       `json:"reasoning_effort,omitempty"`
	ResponseFormat  *openAIResponseFormat `json:"response_format,omitempty"`
	Tools           []ToolSpec            `json:"tools,omitempty"`
	ToolChoice      ToolChoice            `json:"tool_choice,omitempty"`
	Stream          bool                  `json:"stream,omitempty"`
}

type openAIResponseFormat struct {
	Type string `json:"type"`
}

// openAIWireResponse captures reasoning_content (DeepSeek, vLLM, LMStudio,
// and any other OpenAI-compatible reasoning provider) plus DeepSeek's
// prompt cache hit/miss counters.
type openAIWireResponse struct {
	Model   string             `json:"model"`
	Choices []openAIWireChoice `json:"choices"`
	Usage   openAIWireUsage    `json:"usage"`
}

type openAIWireChoice struct {
	Message      openAIWireMessage `json:"message"`
	FinishReason string            `json:"finish_reason"`
}

type openAIWireMessage struct {
	Role             Role       `json:"role"`
	Content          string     `json:"content"`
	ReasoningContent string     `json:"reasoning_content"`
	ToolCalls        []ToolCall `json:"tool_calls"`
}

type openAIWireUsage struct {
	PromptTokens    int `json:"prompt_tokens"`
	CompTokens      int `json:"completion_tokens"`
	PromptCacheHit  int `json:"prompt_cache_hit_tokens"`
	PromptCacheMiss int `json:"prompt_cache_miss_tokens"`
}

func NewOpenAI(cfg Config) *OpenAIProvider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1"
	}

	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = 60 * time.Second
	}

	maxRetries := cfg.MaxRetries
	if maxRetries <= 0 {
		maxRetries = 2
	}

	p := &OpenAIProvider{
		cfg:    cfg,
		client: &http.Client{Timeout: timeout},
		codec:  codec.Default,
	}

	p.execAction = action.New("llm.openai.complete", func(ctx context.Context, req Request) (Response, error) {
		start := time.Now()

		if req.Model == "" {
			req.Model = p.cfg.DefaultModel
		}
		if req.Model == "" {
			return Response{}, xerr.Validation("llm: model is required in the request or provider configuration")
		}

		payload := openAIRequestPayload{
			Model: req.Model, Messages: req.Messages, Temperature: req.Temperature,
			MaxTokens: req.MaxTokens, ReasoningEffort: req.ReasoningEffort,
			ToolChoice: req.ToolChoice,
		}
		if req.JSONMode {
			payload.ResponseFormat = &openAIResponseFormat{Type: "json_object"}
		}
		if len(req.Tools) > 0 {
			payload.Tools = req.Tools
		}

		headers := map[string]string{}
		if p.cfg.APIKey != "" {
			headers["Authorization"] = "Bearer " + p.cfg.APIKey
		}

		raw, err := DoJSON[openAIWireResponse](ctx, HTTPRequest{
			URL:          strings.TrimRight(p.cfg.BaseURL, "/") + "/chat/completions",
			Headers:      headers,
			Body:         payload,
			Client:       p.client,
			Codec:        p.codec,
			ProviderName: p.Name(),
		})
		if err != nil {
			return Response{}, err
		}

		if len(raw.Choices) == 0 {
			return Response{}, xerr.Internal("llm: empty choices returned", nil)
		}

		choice := raw.Choices[0]
		model := raw.Model
		if model == "" {
			model = req.Model
		}

		return Response{
			Content:            choice.Message.Content,
			ReasoningContent:   choice.Message.ReasoningContent,
			Model:              model,
			PromptTokens:       raw.Usage.PromptTokens,
			CachedPromptTokens: raw.Usage.PromptCacheHit,
			CompTokens:         raw.Usage.CompTokens,
			FinishReason:       choice.FinishReason,
			ToolCalls:          choice.Message.ToolCalls,
			Duration:           time.Since(start),
		}, nil
	}).
		Internal().
		Retry(maxRetries, action.ExponentialBackoff(200*time.Millisecond, 2*time.Second)).
		Timeout(timeout).
		Build()

	return p
}

func (p *OpenAIProvider) Name() string {
	if p.cfg.Name != "" {
		return p.cfg.Name
	}

	return providerOpenAI
}

func (p *OpenAIProvider) SupportsToolCalls() bool { return true }

func (p *OpenAIProvider) Complete(ctx context.Context, req Request) (Response, error) {
	if req.Model == "" && p.cfg.DefaultModel == "" {
		return Response{}, xerr.Validation("llm: model is required in the request or provider configuration")
	}
	return p.execAction.Do(ctx, req)
}

func (p *OpenAIProvider) Stream(ctx context.Context, req Request) (iter.Seq2[string, error], error) {
	if req.Model == "" {
		req.Model = p.cfg.DefaultModel
	}
	if req.Model == "" {
		return nil, xerr.Validation("llm: model is required in the request or provider configuration")
	}

	payload := openAIRequestPayload{
		Model: req.Model, Messages: req.Messages, Temperature: req.Temperature,
		ReasoningEffort: req.ReasoningEffort, Stream: true,
	}

	bodyBytes, err := p.codec.Marshal(payload)
	if err != nil {
		return nil, xerr.Internal("llm: marshal stream request failed", err)
	}

	url := strings.TrimRight(p.cfg.BaseURL, "/") + "/chat/completions"

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, xerr.Internal("llm: build stream request failed", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	if p.cfg.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.cfg.APIKey)
	}

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, mapProviderTransportError(err)
	}

	if resp.StatusCode != http.StatusOK {
		respBytes, readErr := io.ReadAll(resp.Body)
		_ = resp.Body.Close()

		if readErr != nil {
			return nil, mapProviderTransportError(readErr)
		}
		return nil, FromStreamHTTPStatus(resp.StatusCode, respBytes, p.Name())
	}

	return ParseOpenAISSEStream(ctx, resp.Body), nil
}
