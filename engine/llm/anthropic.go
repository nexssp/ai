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

type AnthropicProvider struct {
	cfg        Config
	client     *http.Client
	codec      codec.Codec
	execAction *action.BuiltAction[Request, Response]
}

type anthropicRequestMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type anthropicRequestPayload struct {
	Model        string                    `json:"model"`
	MaxTokens    int                       `json:"max_tokens"`
	Messages     []anthropicRequestMessage `json:"messages"`
	System       string                    `json:"system,omitempty"`
	OutputConfig *anthropicOutputConfig    `json:"output_config,omitempty"`
	Stream       bool                      `json:"stream,omitempty"`
}

type anthropicOutputConfig struct {
	Format anthropicOutputFormat `json:"format"`
}

type anthropicOutputFormat struct {
	Type   string     `json:"type"`
	Schema JSONSchema `json:"schema"`
}

// anthropicWireResponse is the JSON shape of the /messages response.
type anthropicWireResponse struct {
	Model   string                  `json:"model"`
	Content []anthropicContentBlock `json:"content"`
	Usage   anthropicWireUsage      `json:"usage"`
}

type anthropicContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type anthropicWireUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

func NewAnthropic(cfg Config) *AnthropicProvider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.anthropic.com/v1"
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
		if req.Model == "" {
			return Response{}, xerr.Validation("anthropic: model is required in the request or provider configuration")
		}

		payload := p.buildPayload(req, false)

		headers := map[string]string{
			"X-Api-Key":         p.cfg.APIKey,
			"Anthropic-Version": "2023-06-01",
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

		model := raw.Model
		if model == "" {
			model = req.Model
		}
		return Response{
			Content:      fullText.String(),
			Model:        model,
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

	return providerAnthropic
}

func (p *AnthropicProvider) Complete(ctx context.Context, req Request) (Response, error) {
	if req.Model == "" && p.cfg.DefaultModel == "" {
		return Response{}, xerr.Validation("anthropic: model is required in the request or provider configuration")
	}
	return p.execAction.Do(ctx, req)
}

func (p *AnthropicProvider) Stream(ctx context.Context, req Request) (iter.Seq2[string, error], error) {
	if req.Model == "" {
		req.Model = p.cfg.DefaultModel
	}
	if req.Model == "" {
		return nil, xerr.Validation("anthropic: model is required in the request or provider configuration")
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
	httpReq.Header.Set("X-Api-Key", p.cfg.APIKey)
	httpReq.Header.Set("Anthropic-Version", "2023-06-01")

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

	return StreamSSE(ctx, resp.Body, DecodeAnthropicChunk), nil
}

// buildPayload converts an llm.Request into the Anthropic-native payload
// shape. Deterministic; no I/O; safe to call from tests.
func (p *AnthropicProvider) buildPayload(req Request, stream bool) anthropicRequestPayload {
	var (
		systemPrompts []string
		messages      []anthropicRequestMessage
	)

	for _, msg := range req.Messages {
		if msg.Role == RoleSystem {
			if msg.Content != "" {
				systemPrompts = append(systemPrompts, msg.Content)
			}
		} else {
			messages = append(messages, anthropicRequestMessage{Role: string(msg.Role), Content: msg.Content})
		}
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 2048
	}

	payload := anthropicRequestPayload{Model: req.Model, MaxTokens: maxTokens, Messages: messages}
	if len(systemPrompts) > 0 {
		payload.System = strings.Join(systemPrompts, "\n\n")
	}
	if req.OutputSchema != nil {
		payload.OutputConfig = &anthropicOutputConfig{Format: anthropicOutputFormat{Type: "json_schema", Schema: *req.OutputSchema}}
	}
	payload.Stream = stream
	return payload
}
