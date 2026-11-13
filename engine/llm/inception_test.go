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

func TestInceptionCompleteSupportsFunctionTools(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
			t.Errorf("request = %s %s, want POST /v1/chat/completions", r.Method, r.URL.Path)
		}
		var payload openAIRequestPayload
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode request: %v", err)
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		if payload.Model != "mercury-3" || payload.ToolChoice != ToolChoiceRequired || len(payload.Tools) != 1 || payload.Tools[0].Function.Name != "count_words" {
			t.Errorf("tool request = %+v, want required count_words tool", payload)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"mercury-3","choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call-inception","type":"function","function":{"name":"count_words","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`))
	}))
	defer server.Close()

	provider := NewInception(Config{BaseURL: server.URL + "/v1"})
	if !provider.SupportsToolCalls() {
		t.Fatal("Inception provider did not advertise its delegated tool-call support")
	}
	response, err := provider.Complete(context.Background(), Request{
		Model:    "mercury-3",
		Messages: []Message{{Role: RoleUser, Content: "Count the words."}},
		Tools: []ToolSpec{{Type: "function", Function: FunctionSpec{
			Name: "count_words", Description: "Count words.",
			Parameters: map[string]any{"type": "object", "properties": map[string]any{}, "required": []string{}, "additionalProperties": false},
		}}},
		ToolChoice: ToolChoiceRequired,
	})
	if err != nil {
		t.Fatalf("Inception Complete: %v", err)
	}
	if len(response.ToolCalls) != 1 || response.ToolCalls[0].ID != "call-inception" || response.ToolCalls[0].Function.Name != "count_words" {
		t.Fatalf("Inception response lost tool calls: %+v", response.ToolCalls)
	}
}
