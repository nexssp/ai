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

// openAIWireResponse captures reasoning_content (DeepSeek, vLLM, LMStudio,
// and any other OpenAI-compatible reasoning provider) plus DeepSeek's
// prompt cache hit/miss counters.
type openAIWireResponse struct {
	Model   string `json:"model"`
	Choices []struct {
		Message struct {
			Role             Role       `json:"role"`
			Content          string     `json:"content"`
			ReasoningContent string     `json:"reasoning_content"`
			ToolCalls        []ToolCall `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens    int `json:"prompt_tokens"`
		CompTokens      int `json:"completion_tokens"`
		PromptCacheHit  int `json:"prompt_cache_hit_tokens"`
		PromptCacheMiss int `json:"prompt_cache_miss_tokens"`
	} `json:"usage"`
}

func NewOpenAI(cfg Config) *OpenAIProvider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1"
	}

	if cfg.DefaultModel == "" {
		cfg.DefaultModel = "gpt-4o-mini"
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

		payload := map[string]any{
			jsonKeyModel:    req.Model,
			jsonKeyMessages: req.Messages,
			"temperature":   req.Temperature,
		}
		if req.MaxTokens > 0 {
			payload["max_tokens"] = req.MaxTokens
		}

		if req.JSONMode {
			payload["response_format"] = map[string]string{"type": "json_object"}
		}

		if len(req.Tools) > 0 {
			payload["tools"] = req.Tools
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

		return Response{
			Content:            choice.Message.Content,
			ReasoningContent:   choice.Message.ReasoningContent,
			Model:              raw.Model,
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

func (p *OpenAIProvider) Complete(ctx context.Context, req Request) (Response, error) {
	return p.execAction.Do(ctx, req)
}

func (p *OpenAIProvider) Stream(ctx context.Context, req Request) (iter.Seq2[string, error], error) {
	if req.Model == "" {
		req.Model = p.cfg.DefaultModel
	}

	payload := map[string]any{
		jsonKeyModel:    req.Model,
		jsonKeyMessages: req.Messages,
		"temperature":   req.Temperature,
		"stream":        true,
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
		return nil, xerr.Unavailable("llm: stream request failed", err)
	}

	if resp.StatusCode != http.StatusOK {
		respBytes, readErr := io.ReadAll(resp.Body)
		_ = resp.Body.Close()

		if readErr != nil {
			return nil, xerr.Unavailable("llm: read stream error response", readErr)
		}
		return nil, FromHTTPStatus(resp.StatusCode, respBytes, p.Name())
	}

	return ParseOpenAISSEStream(ctx, resp.Body), nil
}
