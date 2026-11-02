package llm

import (
	"context"
	"encoding/json"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type jsonDecisionTestProvider struct {
	request  Request
	response Response
	calls    int
}

func (p *jsonDecisionTestProvider) Name() string { return "json-test" }

func (p *jsonDecisionTestProvider) Complete(_ context.Context, req Request) (Response, error) {
	p.request = req
	p.calls++
	return p.response, nil
}

func (p *jsonDecisionTestProvider) Stream(context.Context, Request) (iter.Seq2[string, error], error) {
	return func(func(string, error) bool) {}, nil
}

func TestJSONDecisionProviderReturnsTypedAnswersAndUsesJSONMode(t *testing.T) {
	provider := &jsonDecisionTestProvider{response: Response{
		Model: "served-model",
		Content: `{"answers":{
			"route":{"type":"choice","choice":"candidate-b","probabilities":{"candidate-a":0.1,"candidate-b":0.9},"confidence":0.9},
			"urgent":{"type":"noul","noul":0.8}
		}}`,
		PromptTokens: 20,
		CompTokens:   9,
	}}
	decision := NewJSONDecisionProvider(provider)
	result, err := decision.Decide(context.Background(), DecisionRequest{
		Model: "configured-model", MaxOutputTokens: 16,
		State:      json.RawMessage(`{"request":"route this"}`),
		AllowedIDs: []string{"candidate-a", "candidate-b"},
		Questions: []DecisionQuestion{
			{
				ID: "route", Type: DecisionChoice,
				Criteria: []DecisionCriterion{{ID: "candidate-a", Description: "A"}, {ID: "candidate-b", Description: "B"}},
			},
			{ID: "urgent", Type: DecisionNoul, Instructions: "Is this urgent?"},
		},
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if provider.calls != 1 || provider.request.Model != "configured-model" || provider.request.MaxTokens != 16 || !provider.request.JSONMode || provider.request.Temperature != 0 || len(provider.request.Messages) != 2 {
		t.Fatalf("chat request = %+v, calls=%d", provider.request, provider.calls)
	}
	if !strings.Contains(provider.request.Messages[0].Content, "JSON") || !strings.Contains(provider.request.Messages[1].Content, `"allowed_ids":["candidate-a","candidate-b"]`) {
		t.Fatalf("decision prompt lacks JSON contract or allowlist: %+v", provider.request.Messages)
	}
	if result.Model != "served-model" || len(result.Answers) != 2 || result.Answers[0].Choice != "candidate-b" || result.Answers[0].Probabilities[1].Probability != 0.9 || result.Answers[1].Noul == nil || *result.Answers[1].Noul != 0.8 {
		t.Fatalf("typed decision response = %+v", result)
	}
}

func TestJSONDecisionProviderValidatesChoiceAndExtractBeforeReturning(t *testing.T) {
	provider := &jsonDecisionTestProvider{response: Response{
		Content: `{"answers":{"route":{"type":"choice","choice":"not-declared"}}}`,
	}}
	decision := NewJSONDecisionProvider(provider)
	_, err := decision.Decide(context.Background(), DecisionRequest{
		Model: "configured-model", MaxOutputTokens: 16,
		State: json.RawMessage(`{"request":"route this"}`),
		Questions: []DecisionQuestion{{
			ID: "route", Type: DecisionChoice,
			Criteria: []DecisionCriterion{{ID: "candidate-a", Description: "A"}},
		}},
	})
	if err == nil {
		t.Fatal("expected undeclared choice to be rejected")
	}

	provider.response.Content = `{"answers":{"extract":{"type":"extract","spans":{"company":[{"text":"Acme","start":0,"end":3,"confidence":0.9}]},"not_stated":{"company":0.1}}}}`
	_, err = decision.Decide(context.Background(), DecisionRequest{
		Model: "configured-model", MaxOutputTokens: 16,
		State: json.RawMessage(`"Acme"`),
		Questions: []DecisionQuestion{{
			ID: "extract", Type: DecisionExtract,
			Instructions: "Extract the company.",
		}},
	})
	if err == nil {
		t.Fatal("expected source substring/offset mismatch to be rejected")
	}

	provider.response.Content = `{"answers":{"extract":{"type":"extract","spans":{"extract":[{"text":"café","start":0,"end":4,"confidence":0.9}],"person":[]},"not_stated":{"extract":0.1}}}}`
	_, err = decision.Decide(context.Background(), DecisionRequest{
		Model: "configured-model", MaxOutputTokens: 16,
		State: json.RawMessage(`"café"`),
		Questions: []DecisionQuestion{{
			ID: "extract", Type: DecisionExtract,
			Instructions: "Extract the exact text.",
		}},
	})
	if err == nil {
		t.Fatal("expected undeclared extract field to be rejected")
	}

	provider.response.Content = `{"answers":{"extract":{"type":"extract","spans":{"extract":[{"text":"café","start":0,"end":4,"confidence":0.9}]},"not_stated":{"extract":0.1}}}}`
	result, err := decision.Decide(context.Background(), DecisionRequest{
		Model: "configured-model", MaxOutputTokens: 16,
		State: json.RawMessage(`"café"`),
		Questions: []DecisionQuestion{{
			ID: "extract", Type: DecisionExtract,
			Instructions: "Extract the exact text.",
		}},
	})
	if err != nil {
		t.Fatalf("valid Unicode span: %v", err)
	}
	if len(result.Answers) != 1 || result.Answers[0].Extractions[0].Spans[0].End != 4 {
		t.Fatalf("Unicode span result = %+v", result)
	}
}

func TestJSONDecisionProviderRejectsExtraAnswerIDs(t *testing.T) {
	provider := &jsonDecisionTestProvider{response: Response{Content: `{"answers":{"route":{"type":"choice","choice":"candidate-a"},"rogue":{"type":"noul","noul":0.5}}}`}}
	decision := NewJSONDecisionProvider(provider)
	_, err := decision.Decide(context.Background(), DecisionRequest{
		Model: "configured-model", MaxOutputTokens: 16,
		State: json.RawMessage(`{"task":"route"}`),
		Questions: []DecisionQuestion{{
			ID: "route", Type: DecisionChoice,
			Criteria: []DecisionCriterion{{ID: "candidate-a", Description: "A"}},
		}},
	})
	if err == nil {
		t.Fatal("expected undeclared extra answer to be rejected")
	}
}

func TestJSONDecisionProviderRejectsMissingCapBeforeDispatch(t *testing.T) {
	provider := &jsonDecisionTestProvider{response: Response{Content: `{"answers":{"route":{"type":"noul","noul":0.5}}}`}}
	decision := NewJSONDecisionProvider(provider)
	_, err := decision.Decide(context.Background(), DecisionRequest{
		Model:     "configured-model",
		State:     json.RawMessage(`{"request":"route this"}`),
		Questions: []DecisionQuestion{{ID: "route", Type: DecisionNoul, Instructions: "Is this a route?"}},
	})
	if err == nil {
		t.Fatal("expected missing max_output_tokens to be rejected")
	}
	if provider.calls != 0 {
		t.Fatalf("provider calls = %d, want zero", provider.calls)
	}
}

func TestJSONDecisionProviderRejectsProviderUsageAboveCap(t *testing.T) {
	provider := &jsonDecisionTestProvider{response: Response{
		Content:    `{"answers":{"route":{"type":"noul","noul":0.5}}}`,
		CompTokens: 9,
	}}
	_, err := NewJSONDecisionProvider(provider).Decide(context.Background(), DecisionRequest{
		Model: "configured-model", MaxOutputTokens: 8,
		State:     json.RawMessage(`{"request":"route this"}`),
		Questions: []DecisionQuestion{{ID: "route", Type: DecisionNoul, Instructions: "Is this a route?"}},
	})
	if err == nil {
		t.Fatal("expected output usage above max_output_tokens to be rejected")
	}
	if provider.calls != 1 || provider.request.MaxTokens != 8 {
		t.Fatalf("provider call/cap = %d/%d, want 1/8", provider.calls, provider.request.MaxTokens)
	}
}

func TestJSONDecisionCapReachesActualOpenAIRequest(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.Method != http.MethodPost || r.URL.Path != "/chat/completions" {
			t.Errorf("request = %s %s, want POST /chat/completions", r.Method, r.URL.Path)
		}
		var payload openAIRequestPayload
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode OpenAI request: %v", err)
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		if payload.MaxTokens != 7 {
			t.Errorf("wire max_tokens = %d, want 7", payload.MaxTokens)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"wire-decision-model","choices":[{"message":{"role":"assistant","content":"{\"answers\":{\"route\":{\"type\":\"noul\",\"noul\":0.75}}}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3}}`))
	}))
	defer server.Close()

	decision := NewJSONDecisionProvider(NewOpenAI(Config{BaseURL: server.URL}))
	response, err := decision.Decide(context.Background(), DecisionRequest{
		Model: "wire-decision-model", MaxOutputTokens: 7,
		State:     json.RawMessage(`{"task":"route"}`),
		Questions: []DecisionQuestion{{ID: "route", Type: DecisionNoul, Instructions: "Is this a route?"}},
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if calls != 1 || len(response.Answers) != 1 || response.Answers[0].Noul == nil || *response.Answers[0].Noul != 0.75 {
		t.Fatalf("calls/response = %d/%+v", calls, response)
	}
}
