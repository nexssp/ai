// Ollama does not use SSE. Its /api/chat stream is newline-delimited JSON,
// terminated by a `done:true` field inside the last line. Its streaming
// reader therefore stays local to this file and is not routed through
// StreamSSE.
package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
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

type OllamaProvider struct {
	cfg        Config
	client     *http.Client
	codec      codec.Codec
	execAction *action.BuiltAction[Request, Response]
}

// ollamaWireResponse is the JSON shape of the /api/chat response.
type ollamaWireResponse struct {
	Model   string  `json:"model"`
	Message Message `json:"message"`
	Prompt  int     `json:"prompt_eval_count"`
	Eval    int     `json:"eval_count"`
}

func NewOllama(cfg Config) *OllamaProvider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "http://localhost:11434"
	}

	if cfg.DefaultModel == "" {
		cfg.DefaultModel = defaultOllamaModel
	}

	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = 120 * time.Second
	}

	maxRetries := cfg.MaxRetries
	if maxRetries <= 0 {
		maxRetries = 2
	}

	p := &OllamaProvider{
		cfg:    cfg,
		client: &http.Client{Timeout: timeout},
		codec:  codec.Default,
	}

	p.execAction = action.New("ollama.complete", func(ctx context.Context, req Request) (Response, error) {
		start := time.Now()

		if req.Model == "" {
			req.Model = p.cfg.DefaultModel
		}

		payload := map[string]any{
			jsonKeyModel:     req.Model,
			jsonKeyMessages:  req.Messages,
			"stream":         false,
			jsonKeyKeepAlive: keepAlive24h,
		}
		if req.JSONMode {
			payload["format"] = "json"
		}

		headers := map[string]string{}
		if p.cfg.APIKey != "" {
			headers["Authorization"] = "Bearer " + p.cfg.APIKey
		}

		raw, err := DoJSON[ollamaWireResponse](ctx, HTTPRequest{
			URL:          strings.TrimRight(p.cfg.BaseURL, "/") + "/api/chat",
			Headers:      headers,
			Body:         payload,
			Client:       p.client,
			Codec:        p.codec,
			ProviderName: p.Name(),
		})
		if err != nil {
			return Response{}, err
		}

		return Response{
			Content:      raw.Message.Content,
			Model:        raw.Model,
			PromptTokens: raw.Prompt,
			CompTokens:   raw.Eval,
			Duration:     time.Since(start),
		}, nil
	}).
		Internal().
		Retry(maxRetries, action.ExponentialBackoff(200*time.Millisecond, 2*time.Second)).
		Timeout(timeout).
		Build()

	return p
}

func (p *OllamaProvider) Name() string {
	if p.cfg.Name != "" {
		return p.cfg.Name
	}

	return providerOllama
}

func (p *OllamaProvider) Complete(ctx context.Context, req Request) (Response, error) {
	return p.execAction.Do(ctx, req)
}

// Stream uses newline-delimited JSON, not SSE. Ollama's /api/chat with
// stream:true sends one JSON object per line, terminated by a `done:true`
// field.
func (p *OllamaProvider) Stream(ctx context.Context, req Request) (iter.Seq2[string, error], error) {
	if req.Model == "" {
		req.Model = p.cfg.DefaultModel
	}

	payload := map[string]any{
		jsonKeyModel:     req.Model,
		jsonKeyMessages:  req.Messages,
		"stream":         true,
		jsonKeyKeepAlive: keepAlive24h,
	}

	bodyBytes, err := p.codec.Marshal(payload)
	if err != nil {
		return nil, xerr.Internal("ollama: marshal stream request failed", err)
	}

	url := strings.TrimRight(p.cfg.BaseURL, "/") + "/api/chat"

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, xerr.Internal("ollama: build stream request failed", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	if p.cfg.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.cfg.APIKey)
	}

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, xerr.Unavailable("ollama: stream request failed", err)
	}

	if resp.StatusCode != http.StatusOK {
		respBytes, readErr := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if readErr != nil {
			return nil, xerr.Unavailable("ollama: read stream error response", readErr)
		}
		return nil, xerr.Unavailable(fmt.Sprintf("ollama: stream returned status %d: %s", resp.StatusCode, string(respBytes)))
	}

	return streamOllamaJSONL(ctx, resp.Body), nil
}

// streamOllamaJSONL parses Ollama's newline-delimited JSON stream. Kept
// local to this file: Ollama is the only provider with this wire shape.
func streamOllamaJSONL(ctx context.Context, body io.ReadCloser) iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		defer body.Close()

		reader := bufio.NewReader(body)

		for {
			if ctx.Err() != nil {
				yield("", ctx.Err())

				return
			}

			line, err := reader.ReadBytes('\n')
			if err != nil {
				if err != io.EOF {
					yield("", err)
				}

				return
			}

			trimmed := bytes.TrimSpace(line)
			if len(trimmed) == 0 {
				continue
			}

			var chunk struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
				Done bool `json:"done"`
			}
			if err := json.Unmarshal(trimmed, &chunk); err != nil {
				continue
			}

			if chunk.Message.Content != "" {
				if !yield(chunk.Message.Content, nil) {
					return
				}
			}

			if chunk.Done {
				return
			}
		}
	}
}

func (p *OllamaProvider) Preload(ctx context.Context, model string) error {
	if model == "" {
		model = p.cfg.DefaultModel
	}

	payload := map[string]any{
		"model":      model,
		"keep_alive": -1,
	}

	bodyBytes, err := p.codec.Marshal(payload)
	if err != nil {
		return err
	}

	url := strings.TrimRight(p.cfg.BaseURL, "/") + "/api/generate"

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		body, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			return xerr.Unavailable("ollama preload response read failed", readErr)
		}

		return xerr.Unavailable(fmt.Sprintf("ollama preload failed (status %d: %s)", resp.StatusCode, string(body)))
	}

	return nil
}

func (p *OllamaProvider) Unload(ctx context.Context, model string) error {
	if model == "" {
		model = p.cfg.DefaultModel
	}

	payload := map[string]any{
		"model":      model,
		"keep_alive": 0,
	}

	bodyBytes, err := p.codec.Marshal(payload)
	if err != nil {
		return err
	}

	url := strings.TrimRight(p.cfg.BaseURL, "/") + "/api/generate"

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	return nil
}

func (p *OllamaProvider) IsLoaded(ctx context.Context, model string) bool {
	url := strings.TrimRight(p.cfg.BaseURL, "/") + "/api/ps"

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, http.NoBody)
	if err != nil {
		return false
	}

	resp, err := p.client.Do(req)
	if err != nil || resp.StatusCode != http.StatusOK {
		return false
	}
	defer func() { _ = resp.Body.Close() }()

	var psRes struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&psRes); err != nil {
		return false
	}

	for _, m := range psRes.Models {
		if m.Name == model || strings.HasPrefix(m.Name, model+":") {
			return true
		}
	}

	return false
}
