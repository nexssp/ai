package llm

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/nexssp/kernel/xerr"
	"github.com/nexssp/transport/codec"
)

func requireHTTPErrorKind(t *testing.T, err error, want xerr.Kind) {
	t.Helper()
	var typed *xerr.AppError
	if !errors.As(err, &typed) || typed.Kind != want {
		t.Fatalf("error = %v (%T), want typed %s", err, err, want)
	}
}

func TestDoJSONHTTPErrorKinds(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
		body   string
		kind   xerr.Kind
	}{
		{"bad request", 400, `{}`, xerr.KindBadRequest},
		{"context length is terminal", 400, `{"error":{"code":"context_length_exceeded"}}`, xerr.KindBadRequest},
		{"overloaded code on bad request remains terminal", 400, `{"error":{"code":"overloaded"}}`, xerr.KindBadRequest},
		{"rate limit code on bad request remains terminal", 400, `{"error":{"code":"rate_limit_exceeded"}}`, xerr.KindBadRequest},
		{"unauthorized", 401, `{}`, xerr.KindUnauthorized},
		{"payment required quota", 402, `{"error":{"code":"insufficient_quota"}}`, xerr.KindForbidden},
		{"forbidden", 403, `{}`, xerr.KindForbidden},
		{"not found", 404, `{}`, xerr.KindNotFound},
		{"request timeout", 408, `{}`, xerr.KindTimeout},
		{"conflict", 409, `{}`, xerr.KindConflict},
		{"rate limit without provider code", 429, `{}`, xerr.KindTooManyRequests},
		{"structured rate limit", 429, `{"error":{"code":"rate_limit_exceeded"}}`, xerr.KindTooManyRequests},
		{"quota on rate limit status", 429, `{"error":{"code":"insufficient_quota"}}`, xerr.KindForbidden},
		{"unknown rate limit code is terminal", 429, `{"error":{"code":"provider_specific_unknown"}}`, xerr.KindInternal},
		{"server failure", 500, `{}`, xerr.KindUnavailable},
		{"not implemented", 501, `{}`, xerr.KindInternal},
		{"bad gateway", 502, `{}`, xerr.KindUnavailable},
		{"unavailable", 503, `{}`, xerr.KindUnavailable},
		{"gateway timeout", 504, `{}`, xerr.KindTimeout},
		{"invalid response", 200, `not JSON`, xerr.KindInternal},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tc.status)
				_, _ = io.WriteString(w, tc.body)
			}))
			defer server.Close()
			_, err := DoJSON[map[string]any](t.Context(), HTTPRequest{
				URL: server.URL, Body: map[string]any{"user": "test"},
				Client: server.Client(), Codec: codec.Default, ProviderName: "fake",
			})
			requireHTTPErrorKind(t, err, tc.kind)
		})
	}
}

func TestFromStreamHTTPStatusKinds(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
		kind   xerr.Kind
	}{
		{"authorization", http.StatusUnauthorized, xerr.KindUnauthorized},
		{"rate limit", http.StatusTooManyRequests, xerr.KindTooManyRequests},
		{"server unavailable", http.StatusServiceUnavailable, xerr.KindUnavailable},
		{"gateway timeout", http.StatusGatewayTimeout, xerr.KindTimeout},
		{"unexpected successful status", http.StatusCreated, xerr.KindInternal},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := FromStreamHTTPStatus(tc.status, []byte(`{}`), "fake")
			requireHTTPErrorKind(t, err, tc.kind)
		})
	}
}

type errorRoundTripper func(*http.Request) (*http.Response, error)

func (f errorRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

type failingResponseBody struct{ err error }

func (b failingResponseBody) Read([]byte) (int, error) { return 0, b.err }
func (b failingResponseBody) Close() error             { return nil }

func TestDoJSONTransportErrorKindsAndCauses(t *testing.T) {
	for _, tc := range []struct {
		name  string
		cause error
		kind  xerr.Kind
	}{
		{"deadline", context.DeadlineExceeded, xerr.KindTimeout},
		{"canceled", context.Canceled, xerr.KindCanceled},
		{"network timeout", &net.DNSError{Err: "timeout", IsTimeout: true}, xerr.KindTimeout},
		{"temporary DNS", &net.DNSError{Err: "temporary failure", IsTemporary: true}, xerr.KindUnavailable},
		{"permanent DNS", &net.DNSError{Err: "no such host", IsNotFound: true}, xerr.KindInternal},
		{"connection", &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("refused")}, xerr.KindUnavailable},
		{"unclassified transport failure", errors.New("transport failed"), xerr.KindInternal},
	} {
		for _, readBody := range []bool{false, true} {
			name := tc.name + "/request"
			if readBody {
				name = tc.name + "/body"
			}
			t.Run(name, func(t *testing.T) {
				client := &http.Client{Transport: errorRoundTripper(func(*http.Request) (*http.Response, error) {
					if !readBody {
						return nil, tc.cause
					}
					return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: failingResponseBody{tc.cause}}, nil
				})}
				_, err := DoJSON[map[string]any](t.Context(), HTTPRequest{URL: "http://fake.invalid", Client: client, Codec: codec.Default})
				requireHTTPErrorKind(t, err, tc.kind)
				if !errors.Is(err, tc.cause) {
					t.Fatalf("lost cause: %v", err)
				}
			})
		}
	}
}

func TestDoJSONInternalDefects(t *testing.T) {
	for _, tc := range []struct {
		name    string
		request HTTPRequest
	}{
		{"missing client", HTTPRequest{Codec: codec.Default}},
		{"missing codec", HTTPRequest{Client: http.DefaultClient}},
		{"marshal defect", HTTPRequest{Client: http.DefaultClient, Codec: codec.Default, Body: make(chan int)}},
		{"invalid constructed URL", HTTPRequest{Client: http.DefaultClient, Codec: codec.Default, URL: "://invalid"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := DoJSON[map[string]any](t.Context(), tc.request)
			requireHTTPErrorKind(t, err, xerr.KindInternal)
		})
	}
}

func TestDoJSONSuccessfulResponse(t *testing.T) {
	client := &http.Client{Transport: errorRoundTripper(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"answer":"ok"}`))}, nil
	})}
	got, err := DoJSON[map[string]string](t.Context(), HTTPRequest{URL: "http://fake.invalid", Client: client, Codec: codec.Default})
	if err != nil || got["answer"] != "ok" {
		t.Fatalf("response = %v, error = %v", got, err)
	}
}
