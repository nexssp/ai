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
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"

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
		return zero, mapProviderTransportError(err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return zero, mapProviderTransportError(err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return zero, FromHTTPStatus(resp.StatusCode, respBytes, r.ProviderName)
	}

	var out Res
	if err := r.Codec.Unmarshal(respBytes, &out); err != nil {
		return zero, xerr.Internal("llm: invalid provider response", err)
	}

	return out, nil
}

func mapProviderTransportError(err error) error {
	if err == nil {
		return nil
	}
	cause := err
	for {
		urlErr, ok := errors.AsType[*url.Error](cause)
		if !ok {
			break
		}
		cause = urlErr.Err
	}
	if errors.Is(cause, context.DeadlineExceeded) {
		return xerr.Timeout("llm: provider transport timed out", err)
	}
	if errors.Is(cause, context.Canceled) {
		return xerr.Canceled("llm: provider transport was canceled", err)
	}
	if dnsErr, ok := errors.AsType[*net.DNSError](cause); ok {
		if dnsErr.IsTimeout {
			return xerr.Timeout("llm: provider DNS lookup timed out", err)
		}
		if dnsErr.IsTemporary {
			return xerr.Unavailable("llm: temporary provider DNS failure", err)
		}
		return xerr.Internal("llm: permanent provider DNS failure", err)
	}
	if netErr, ok := errors.AsType[net.Error](cause); ok {
		if netErr.Timeout() {
			return xerr.Timeout("llm: provider network timed out", err)
		}
		return xerr.Unavailable("llm: provider network failure", err)
	}
	if errors.Is(err, io.ErrUnexpectedEOF) {
		return xerr.Unavailable("llm: provider connection ended unexpectedly", err)
	}
	return xerr.Internal("llm: unclassified provider transport failure", err)
}
