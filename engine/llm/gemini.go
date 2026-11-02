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

type GeminiProvider struct {
	cfg        Config
	client     *http.Client
	codec      codec.Codec
	execAction *action.BuiltAction[Request, Response]
}

type geminiPart struct {
	Text string `json:"text"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiGenerationConfig struct {
	ResponseMimeType string `json:"responseMimeType,omitempty"`
	MaxOutputTokens  int    `json:"maxOutputTokens,omitempty"`
}

type geminiRequestPayload struct {
	Contents          []geminiContent         `json:"contents"`
	SystemInstruction *geminiContent          `json:"systemInstruction,omitempty"`
	GenerationConfig  *geminiGenerationConfig `json:"generationConfig,omitempty"`
}

// geminiWireResponse is the JSON shape of the :generateContent response.
type geminiWireResponse struct {
	Candidates    []geminiWireCandidate `json:"candidates"`
	UsageMetadata geminiWireUsage       `json:"usageMetadata"`
}

type geminiWireCandidate struct {
	Content geminiWireContent `json:"content"`
}

type geminiWireContent struct {
	Parts []geminiWirePart `json:"parts"`
}

type geminiWirePart struct {
	Text string `json:"text"`
}

type geminiWireUsage struct {
	PromptTokenCount int `json:"promptTokenCount"`
	CandidatesTokens int `json:"candidatesTokenCount"`
}

func NewGemini(cfg Config) *GeminiProvider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://generativelanguage.googleapis.com/v1beta"
	}

	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = 60 * time.Second
	}

	maxRetries := cfg.MaxRetries
	if maxRetries <= 0 {
		maxRetries = 2
	}

	p := &GeminiProvider{
		cfg:    cfg,
		client: &http.Client{Timeout: timeout},
		codec:  codec.Default,
	}

	p.execAction = action.New("gemini.complete", func(ctx context.Context, req Request) (Response, error) {
		start := time.Now()

		if req.Model == "" {
			req.Model = p.cfg.DefaultModel
		}
		if req.Model == "" {
			return Response{}, xerr.Validation("gemini: model is required in the request or provider configuration")
		}

		// Gemini carries the API key in the URL, so no Authorization header.
		url := fmt.Sprintf("%s/models/%s:generateContent?key=%s",
			strings.TrimRight(p.cfg.BaseURL, "/"), req.Model, p.cfg.APIKey)

		raw, err := DoJSON[geminiWireResponse](ctx, HTTPRequest{
			URL:          url,
			Body:         p.buildPayload(req),
			Client:       p.client,
			Codec:        p.codec,
			ProviderName: p.Name(),
		})
		if err != nil {
			return Response{}, err
		}

		if len(raw.Candidates) == 0 || len(raw.Candidates[0].Content.Parts) == 0 {
			return Response{}, xerr.Internal("gemini: empty response candidates", nil)
		}

		return Response{
			Content:      raw.Candidates[0].Content.Parts[0].Text,
			Model:        req.Model,
			PromptTokens: raw.UsageMetadata.PromptTokenCount,
			CompTokens:   raw.UsageMetadata.CandidatesTokens,
			Duration:     time.Since(start),
		}, nil
	}).
		Internal().
		Retry(maxRetries, action.ExponentialBackoff(200*time.Millisecond, 2*time.Second)).
		Timeout(timeout).
		Build()

	return p
}

func (p *GeminiProvider) Name() string {
	if p.cfg.Name != "" {
		return p.cfg.Name
	}

	return providerGemini
}

func (p *GeminiProvider) Complete(ctx context.Context, req Request) (Response, error) {
	if req.Model == "" && p.cfg.DefaultModel == "" {
		return Response{}, xerr.Validation("gemini: model is required in the request or provider configuration")
	}
	return p.execAction.Do(ctx, req)
}

func (p *GeminiProvider) Stream(ctx context.Context, req Request) (iter.Seq2[string, error], error) {
	if req.Model == "" {
		req.Model = p.cfg.DefaultModel
	}
	if req.Model == "" {
		return nil, xerr.Validation("gemini: model is required in the request or provider configuration")
	}

	bodyBytes, err := p.codec.Marshal(p.buildPayload(req))
	if err != nil {
		return nil, xerr.Internal("gemini: marshal stream failed", err)
	}

	url := fmt.Sprintf("%s/models/%s:streamGenerateContent?alt=sse&key=%s",
		strings.TrimRight(p.cfg.BaseURL, "/"), req.Model, p.cfg.APIKey)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, xerr.Internal("gemini: build stream request failed", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

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

	return StreamSSE(ctx, resp.Body, DecodeGeminiChunk), nil
}

// buildPayload converts an llm.Request into the Gemini-native payload shape.
// Deterministic; no I/O; safe to call from tests.
func (p *GeminiProvider) buildPayload(req Request) geminiRequestPayload {
	var (
		contents       []geminiContent
		sysInstruction *geminiContent
	)

	for _, m := range req.Messages {
		if m.Role == RoleSystem {
			sysInstruction = &geminiContent{Parts: []geminiPart{{Text: m.Content}}}
		} else {
			role := roleUser
			if m.Role == RoleAssistant {
				role = roleModel
			}

			contents = append(contents, geminiContent{
				Role:  role,
				Parts: []geminiPart{{Text: m.Content}},
			})
		}
	}

	payload := geminiRequestPayload{Contents: contents, SystemInstruction: sysInstruction}
	if req.JSONMode || req.MaxTokens > 0 {
		generationConfig := &geminiGenerationConfig{MaxOutputTokens: req.MaxTokens}
		if req.JSONMode {
			generationConfig.ResponseMimeType = "application/json"
		}
		payload.GenerationConfig = generationConfig
	}
	return payload
}
