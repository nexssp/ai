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

// geminiWireResponse is the JSON shape of the :generateContent response.
type geminiWireResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount int `json:"promptTokenCount"`
		CandidatesTokens int `json:"candidatesTokenCount"`
	} `json:"usageMetadata"`
}

func NewGemini(cfg Config) *GeminiProvider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://generativelanguage.googleapis.com/v1beta"
	}

	if cfg.DefaultModel == "" {
		cfg.DefaultModel = "gemini-2.0-flash"
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
	return p.execAction.Do(ctx, req)
}

func (p *GeminiProvider) Stream(ctx context.Context, req Request) (iter.Seq2[string, error], error) {
	if req.Model == "" {
		req.Model = p.cfg.DefaultModel
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
		return nil, xerr.Unavailable("gemini: stream request failed", err)
	}

	if resp.StatusCode != http.StatusOK {
		respBytes, readErr := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if readErr != nil {
			return nil, xerr.Unavailable("gemini: read stream error response", readErr)
		}
		return nil, xerr.Unavailable(fmt.Sprintf("gemini: stream returned status %d: %s", resp.StatusCode, string(respBytes)))
	}

	return StreamSSE(ctx, resp.Body, DecodeGeminiChunk), nil
}

// buildPayload converts an llm.Request into the Gemini-native payload shape.
// Deterministic; no I/O; safe to call from tests.
func (p *GeminiProvider) buildPayload(req Request) map[string]any {
	type part struct {
		Text string `json:"text"`
	}

	type content struct {
		Role  string `json:"role"`
		Parts []part `json:"parts"`
	}

	var (
		contents       []content
		sysInstruction *content
	)

	for _, m := range req.Messages {
		if m.Role == RoleSystem {
			sysInstruction = &content{Parts: []part{{Text: m.Content}}}
		} else {
			role := roleUser
			if m.Role == RoleAssistant {
				role = roleModel
			}

			contents = append(contents, content{
				Role:  role,
				Parts: []part{{Text: m.Content}},
			})
		}
	}

	payload := map[string]any{"contents": contents}
	if sysInstruction != nil {
		payload["systemInstruction"] = sysInstruction
	}

	if req.JSONMode {
		payload["generationConfig"] = map[string]string{
			"responseMimeType": "application/json",
		}
	}

	return payload
}
