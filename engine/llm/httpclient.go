// Shared JSON POST transport for every non-streaming LLM provider call.
// Marshal, POST, status classification, body read, unmarshal — one place.
//
// Providers supply URL, headers, body, client, and codec. They do not
// repeat the boilerplate. Streaming stays in sse.go and the provider
// Stream methods; this helper is for the request/response cycle only.
package llm

import (
	"bytes"
	"context"
	"io"
	"net/http"

	"github.com/nexssp/kernel/xerr"
	"github.com/nexssp/transport/codec"
)

// HTTPRequest is the input to DoJSON. ProviderName is used only in error
// messages produced by FromHTTPStatus; leave empty to omit the prefix.
type HTTPRequest struct {
	URL          string
	Headers      map[string]string
	Body         any
	Client       *http.Client
	Codec        codec.Codec
	ProviderName string
}

// DoJSON performs a JSON POST, decodes the JSON response into Res, and maps
// non-2xx status codes through FromHTTPStatus. The function is the single
// HTTP entry point for every OpenAI-compatible, Anthropic, and Ollama
// non-streaming request.
func DoJSON[Res any](ctx context.Context, r HTTPRequest) (Res, error) {
	var zero Res

	if r.Client == nil {
		return zero, xerr.Internal("llm: DoJSON requires a non-nil HTTP client")
	}

	if r.Codec == nil {
		return zero, xerr.Internal("llm: DoJSON requires a non-nil codec")
	}

	bodyBytes, err := r.Codec.Marshal(r.Body)
	if err != nil {
		return zero, xerr.Internal("llm: marshal request failed", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, r.URL, bytes.NewReader(bodyBytes))
	if err != nil {
		return zero, xerr.Internal("llm: build request failed", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	for k, v := range r.Headers {
		httpReq.Header.Set(k, v)
	}

	resp, err := r.Client.Do(httpReq)
	if err != nil {
		return zero, xerr.MapTransportError(err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return zero, xerr.MapTransportError(err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return zero, FromHTTPStatus(resp.StatusCode, respBytes, r.ProviderName)
	}

	var out Res
	if err := r.Codec.Unmarshal(respBytes, &out); err != nil {
		return zero, xerr.Unavailable("llm: invalid provider response", err)
	}

	return out, nil
}
