package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestTypeSafeDecideFailsClosedWithoutDocumentedCap(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		_, _ = w.Write([]byte(`{}`))
	}))
	defer server.Close()

	provider := NewTypeSafe(Config{BaseURL: server.URL, DefaultModel: "jev-latest"})
	_, err := provider.Decide(context.Background(), DecisionRequest{
		MaxOutputTokens: 16,
		State:           json.RawMessage(`{"ctx":"context"}`),
		Questions:       []DecisionQuestion{{ID: "q", Type: DecisionNoul, Instructions: "Check this."}},
	})
	if err == nil || !strings.Contains(err.Error(), "does not document a provider-enforced max_output_tokens limit") {
		t.Fatalf("Decide error = %v, want explicit cap-support rejection", err)
	}
	if calls != 0 {
		t.Fatalf("HTTP calls = %d, want zero", calls)
	}
}

func TestTypeSafePreflightValidationRejectsMissingCapAndInvalidQuestions(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) { calls++ }))
	defer server.Close()
	provider := NewTypeSafe(Config{BaseURL: server.URL})
	base := DecisionRequest{
		Model: "caller-model", State: json.RawMessage(`{"x":1}`),
		Questions: []DecisionQuestion{{ID: "q", Type: DecisionNoul, Instructions: "check"}},
	}
	if _, err := provider.Decide(context.Background(), base); err == nil || !strings.Contains(err.Error(), "max_output_tokens must be a positive integer") {
		t.Fatalf("missing-cap error = %v", err)
	}
	base.MaxOutputTokens = 8
	base.Questions[0] = DecisionQuestion{ID: "route", Type: DecisionChoice, Criteria: []DecisionCriterion{{ID: "a", Description: "A"}}}
	base.AllowedIDs = []string{"b"}
	if _, err := provider.Decide(context.Background(), base); err == nil || !strings.Contains(err.Error(), "outside the allowlist") {
		t.Fatalf("allowlist error = %v", err)
	}
	base.AllowedIDs = []string{"a"}
	if _, err := provider.Decide(context.Background(), base); err == nil || !strings.Contains(err.Error(), "does not document a provider-enforced max_output_tokens limit") {
		t.Fatalf("cap-support error = %v", err)
	}
	if calls != 0 {
		t.Fatalf("HTTP calls = %d, want zero", calls)
	}
}

func TestFetchTypeSafeModelsUsesDocumentedCatalog(t *testing.T) {
	const key = "typesafe-models-key"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/models" {
			t.Errorf("request = %s %s", r.Method, r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer "+key {
			t.Errorf("Authorization = %q", got)
		}
		_, _ = w.Write([]byte(`{"models":[{"name":"jev-latest","description":"latest","release_date":"2026-10-14"},{"name":"jev-1.14.0","description":"pinned","release_date":"2026-10-14"}]}`))
	}))
	defer server.Close()

	models, err := FetchModels(context.Background(), "typesafe", key, server.URL)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 || models[0] != "jev-latest" || models[1] != "jev-1.14.0" {
		t.Fatalf("models = %v", models)
	}
}
