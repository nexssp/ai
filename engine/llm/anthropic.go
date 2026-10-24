package llm

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"
	"time"

	"github.com/nexssp/kernel/action"
	"github.com/nexssp/kernel/xerr"
	"github.com/nexssp/transport/codec"
)

type AnthropicProvider struct {
	cfg        Config
	client     *http.Client
	codec      codec.Codec
	execAction *action.BuiltAction[Request, Response]
}

// anthropicWireResponse is the JSON shape of the /messages response.
type anthropicWireResponse struct {
	Model   string `json:"model"`
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	Usage struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

func NewAnthropic(cfg Config) *AnthropicProvider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.anthropic.com/v1"
	}

	if cfg.DefaultModel == "" {
		cfg.DefaultModel = "claude-3-5-sonnet-20241022"
	}

	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = 60 * time.Second
	}

	maxRetries := cfg.MaxRetries
	if maxRetries <= 0 {
		maxRetries = 2
	}

	p := &AnthropicProvider{
		cfg:    cfg,
		client: &http.Client{Timeout: timeout},
		codec:  codec.Default,
	}

	p.execAction = action.New("anthropic.complete", func(ctx context.Context, req Request) (Response, error) {
		start := time.Now()

		if req.Model == "" {
			req.Model = p.cfg.DefaultModel
		}

		payload := p.buildPayload(req, false)

		headers := map[string]string{
			"x-api-key":         p.cfg.APIKey,
			"anthropic-version": "2023-06-01",
		}

		raw, err := DoJSON[anthropicWireResponse](ctx, HTTPRequest{
			URL:          strings.TrimRight(p.cfg.BaseURL, "/") + "/messages",
			Headers:      headers,
			Body:         payload,
			Client:       p.client,
			Codec:        p.codec,
			ProviderName: p.Name(),
		})
		if err != nil {
			return Response{}, err
		}

		var fullText strings.Builder

		for _, block := range raw.Content {
			if block.Type == "text" {
				fullText.WriteString(block.Text)
			}
		}

		return Response{
			Content:      fullText.String(),
			Model:        raw.Model,
			PromptTokens: raw.Usage.InputTokens,
			CompTokens:   raw.Usage.OutputTokens,
			Duration:     time.Since(start),
		}, nil
	}).
		Internal().
		Retry(maxRetries, action.ExponentialBackoff(200*time.Millisecond, 2*time.Second)).
		Timeout(timeout).
		Build()

	return p
}

func (p *AnthropicProvider) Name() string {
	if p.cfg.Name != "" {
		return p.cfg.Name
	}

	return "anthropic"
}

func (p *AnthropicProvider) Complete(ctx context.Context, req Request) (Response, error) {
	return p.execAction.Do(ctx, req)
}

func (p *AnthropicProvider) Stream(ctx context.Context, req Request) (iter.Seq2[string, error], error) {
	if req.Model == "" {
		req.Model = p.cfg.DefaultModel
	}

	payload := p.buildPayload(req, true)

	bodyBytes, err := p.codec.Marshal(payload)
	if err != nil {
		return nil, xerr.Internal("anthropic: marshal stream request failed", err)
	}

	url := strings.TrimRight(p.cfg.BaseURL, "/") + "/messages"

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, xerr.Internal("anthropic: build stream request failed", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", p.cfg.APIKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, xerr.Unavailable("anthropic: stream request failed", err)
	}

	if resp.StatusCode != http.StatusOK {
		respBytes, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()

		return nil, xerr.Unavailable(fmt.Sprintf("anthropic: stream returned status %d: %s", resp.StatusCode, string(respBytes)))
	}

	return StreamSSE(ctx, resp.Body, DecodeAnthropicChunk), nil
}

// buildPayload converts an llm.Request into the Anthropic-native payload
// shape. Deterministic; no I/O; safe to call from tests.
func (p *AnthropicProvider) buildPayload(req Request, stream bool) map[string]any {
	var (
		systemPrompts []string
		messages      []map[string]string
	)

	for _, msg := range req.Messages {
		if msg.Role == RoleSystem {
			if msg.Content != "" {
				systemPrompts = append(systemPrompts, msg.Content)
			}
		} else {
			messages = append(messages, map[string]string{
				"role":    string(msg.Role),
				"content": msg.Content,
			})
		}
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 2048
	}

	payload := map[string]any{
		"model":      req.Model,
		"max_tokens": maxTokens,
		"messages":   messages,
	}
	if len(systemPrompts) > 0 {
		payload["system"] = strings.Join(systemPrompts, "\n\n")
	}

	if stream {
		payload["stream"] = true
	}

	return payload
}
