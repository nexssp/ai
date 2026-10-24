package llm

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/nexssp/kernel/xerr"
	"github.com/nexssp/transport/codec"
)

type openaiEmbedder struct {
	baseURL string
	apiKey  string
	model   string
	client  *http.Client
	codec   codec.Codec
}

// NewEmbedder creates an embedding generator compatible with OpenAI, OpenRouter, and Ollama.
func NewEmbedder(baseURL, apiKey, model string) Embedder {
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}

	if model == "" {
		model = "text-embedding-3-small"
	}

	return &openaiEmbedder{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		model:   model,
		client:  &http.Client{Timeout: 15 * time.Second},
		codec:   codec.Default,
	}
}

func (e *openaiEmbedder) Embed(ctx context.Context, texts ...string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	payload := map[string]any{
		"model": e.model,
		"input": texts,
	}

	bodyBytes, err := e.codec.Marshal(payload)
	if err != nil {
		return nil, xerr.Internal("embedder marshal error", err)
	}

	url := e.baseURL + "/embeddings"

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, xerr.Internal("embedder build request error", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	if e.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+e.apiKey)
	}

	resp, err := e.client.Do(httpReq)
	if err != nil {
		return nil, xerr.Unavailable("embedder api call failed", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, xerr.Internal("embedder read response error", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, xerr.Unavailable(fmt.Sprintf("embedder status %d: %s", resp.StatusCode, string(respBytes)))
	}

	var res struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}

	if err := e.codec.Unmarshal(respBytes, &res); err != nil {
		return nil, xerr.Internal("embedder unmarshal response error", err)
	}

	out := make([][]float32, len(res.Data))
	for i, d := range res.Data {
		out[i] = d.Embedding
	}

	return out, nil
}
