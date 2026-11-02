package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestInceptionNativeDecideFailsClosedWithoutDocumentedCap(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests++
		_, _ = w.Write([]byte(`{}`))
	}))
	defer server.Close()

	provider := NewInception(Config{BaseURL: server.URL + "/v1", DefaultModel: "mercury-decide"})
	_, err := provider.Decide(context.Background(), DecisionRequest{
		MaxOutputTokens: 16,
		State:           json.RawMessage(`{"source":"Acme is an example company."}`),
		Questions:       []DecisionQuestion{{ID: "q", Type: DecisionNoul, Instructions: "Is this supported?"}},
	})
	if err == nil || !strings.Contains(err.Error(), "does not document a provider-enforced max_output_tokens limit") {
		t.Fatalf("Decide error = %v, want explicit cap-support rejection", err)
	}
	if requests != 0 {
		t.Fatalf("HTTP requests = %d, want zero", requests)
	}
}

func TestInceptionDecideRejectsMissingCapBeforeDispatch(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests++
		_, _ = w.Write([]byte(`{}`))
	}))
	defer server.Close()

	provider := NewInception(Config{BaseURL: server.URL + "/v1", DefaultModel: "mercury-decide"})
	_, err := provider.Decide(context.Background(), DecisionRequest{
		State:     json.RawMessage(`{"x":1}`),
		Questions: []DecisionQuestion{{ID: "q", Type: DecisionNoul, Instructions: "Check this."}},
	})
	if err == nil || !strings.Contains(err.Error(), "max_output_tokens must be a positive integer") {
		t.Fatalf("Decide error = %v, want missing-cap validation", err)
	}
	if requests != 0 {
		t.Fatalf("HTTP requests = %d, want zero", requests)
	}
}
