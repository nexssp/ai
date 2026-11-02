package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

func TestAnthropicDecisionUsesNativeStructuredOutputSchema(t *testing.T) {
	requests := 0
	testAPIKey := t.Name()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.Method != http.MethodPost || r.URL.Path != "/v1/messages" {
			t.Errorf("request = %s %s, want POST /v1/messages", r.Method, r.URL.Path)
		}
		if r.Header.Get("X-Api-Key") != testAPIKey {
			t.Errorf("X-Api-Key = %q, want configured test key", r.Header.Get("X-Api-Key"))
		}

		var request anthropicRequestPayload
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode request: %v", err)
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		if request.OutputConfig == nil {
			t.Errorf("output_config is nil")
			http.Error(w, "missing output config", http.StatusBadRequest)
			return
		}
		if request.Model != "anthropic-schema-test-model" || request.OutputConfig.Format.Type != "json_schema" {
			t.Errorf("model/format = %q/%q", request.Model, request.OutputConfig.Format.Type)
		}
		if request.MaxTokens != 12 {
			t.Errorf("max_tokens = %d, want 12", request.MaxTokens)
		}
		answers := request.OutputConfig.Format.Schema.Properties["answers"]
		route := answers.Properties["route"]
		choice := route.Properties["choice"]
		if !reflect.DeepEqual(answers.Required, []string{"route"}) || answers.AdditionalProperties == nil || *answers.AdditionalProperties || !reflect.DeepEqual(choice.Enum, []string{"candidate-a", "candidate-b"}) {
			t.Errorf("decision schema lost required question or allowlist: answers=%+v route=%+v choice=%+v", answers, route, choice)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"anthropic-schema-test-model","content":[{"type":"text","text":"{\"answers\":{\"route\":{\"type\":\"choice\",\"choice\":\"candidate-b\"}}}"}],"usage":{"input_tokens":18,"output_tokens":4}}`))
	}))
	defer server.Close()

	provider := NewAnthropic(Config{BaseURL: server.URL + "/v1", APIKey: testAPIKey})
	decision := NewAnthropicDecisionProvider(provider)
	response, err := decision.Decide(context.Background(), DecisionRequest{
		Model:           "anthropic-schema-test-model",
		MaxOutputTokens: 12,
		State:           json.RawMessage(`{"task":"route"}`),
		AllowedIDs:      []string{"candidate-a", "candidate-b"},
		Questions: []DecisionQuestion{{
			ID: "route", Type: DecisionChoice,
			Criteria: []DecisionCriterion{
				{ID: "candidate-a", Description: "A"},
				{ID: "candidate-b", Description: "B"},
			},
		}},
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if requests != 1 || len(response.Answers) != 1 || response.Answers[0].Choice != "candidate-b" {
		t.Fatalf("unexpected structured decision: requests=%d response=%+v", requests, response)
	}
}
