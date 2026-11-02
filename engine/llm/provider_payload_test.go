package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/nexssp/cost"
)

func TestOpenAICompleteUsesTypedWirePayload(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/chat/completions" {
			t.Errorf("request = %s %s, want POST /chat/completions", r.Method, r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("authorization = %q, want configured bearer token", r.Header.Get("Authorization"))
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		for _, forbidden := range [][]byte{[]byte(`"candidate_id"`), []byte(`"attempt_id"`), []byte(`"rate_card_id"`), []byte(`"expected_currency"`), []byte(`"estimated_prompt_tokens"`)} {
			if bytes.Contains(body, forbidden) {
				t.Errorf("host accounting metadata leaked in provider payload: %s", body)
			}
		}
		var payload openAIRequestPayload
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Errorf("decode typed OpenAI request: %v", err)
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		if payload.Model != "openai-wire-test" || payload.Temperature != 0.25 || payload.MaxTokens != 64 || payload.ReasoningEffort != ReasoningEffortHigh {
			t.Errorf("unexpected request fields: %+v", payload)
		}
		if len(payload.Messages) != 1 || payload.Messages[0].Role != RoleUser || payload.Messages[0].Content != "hello" {
			t.Errorf("messages = %+v", payload.Messages)
		}
		if payload.ResponseFormat == nil || payload.ResponseFormat.Type != "json_object" {
			t.Errorf("response format = %+v, want json_object", payload.ResponseFormat)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"openai-wire-test","choices":[{"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":2}}`))
	}))
	defer server.Close()

	provider := NewOpenAI(Config{BaseURL: server.URL, APIKey: "test-key"})
	expectedCurrency := cost.EUR
	response, err := provider.Complete(context.Background(), Request{
		Model:                 "openai-wire-test",
		CandidateID:           "candidate-x",
		AttemptID:             "attempt-x",
		RateCardID:            "card-eur-x",
		ExpectedCurrency:      &expectedCurrency,
		EstimatedPromptTokens: 20,
		Messages:              []Message{{Role: RoleUser, Content: "hello"}},
		Temperature:           0.25,
		MaxTokens:             64,
		JSONMode:              true,
		ReasoningEffort:       ReasoningEffortHigh,
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if response.Content != "answer" || response.PromptTokens != 7 || response.CompTokens != 2 {
		t.Fatalf("response = %+v", response)
	}
}

func TestOllamaCompleteUsesTypedWirePayload(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/api/chat" {
			t.Errorf("request = %s %s, want POST /api/chat", r.Method, r.URL.Path)
		}
		var payload ollamaChatPayload
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode typed Ollama request: %v", err)
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		if payload.Model != "ollama-wire-test" || payload.KeepAlive != keepAlive24h || payload.Format != "json" || payload.Stream {
			t.Errorf("unexpected Ollama request fields: %+v", payload)
		}
		if len(payload.Messages) != 1 || payload.Messages[0].Role != RoleUser || payload.Messages[0].Content != "hello" {
			t.Errorf("messages = %+v", payload.Messages)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"ollama-wire-test","message":{"role":"assistant","content":"answer"},"prompt_eval_count":4,"eval_count":3}`))
	}))
	defer server.Close()

	provider := NewOllama(Config{BaseURL: server.URL})
	response, err := provider.Complete(context.Background(), Request{
		Model:    "ollama-wire-test",
		Messages: []Message{{Role: RoleUser, Content: "hello"}},
		JSONMode: true,
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if response.Content != "answer" || response.PromptTokens != 4 || response.CompTokens != 3 {
		t.Fatalf("response = %+v", response)
	}
}

func TestGeminiCompleteUsesTypedWirePayload(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1beta/models/gemini-wire-test:generateContent" {
			t.Errorf("request = %s %s, want POST Gemini generateContent", r.Method, r.URL.Path)
		}
		if r.URL.Query().Get("key") != "test-key" {
			t.Errorf("query API key = %q, want configured key", r.URL.Query().Get("key"))
		}
		var payload geminiRequestPayload
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode typed Gemini request: %v", err)
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		if len(payload.Contents) != 1 || payload.Contents[0].Role != roleUser || len(payload.Contents[0].Parts) != 1 || payload.Contents[0].Parts[0].Text != "hello" {
			t.Errorf("contents = %+v", payload.Contents)
		}
		if payload.SystemInstruction == nil || payload.SystemInstruction.Parts[0].Text != "be concise" {
			t.Errorf("system instruction = %+v", payload.SystemInstruction)
		}
		if payload.GenerationConfig == nil || payload.GenerationConfig.ResponseMimeType != "application/json" || payload.GenerationConfig.MaxOutputTokens != 32 {
			t.Errorf("generation config = %+v", payload.GenerationConfig)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"answer"}]}}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2}}`))
	}))
	defer server.Close()

	provider := NewGemini(Config{BaseURL: server.URL + "/v1beta", APIKey: "test-key"})
	response, err := provider.Complete(context.Background(), Request{
		Model:     "gemini-wire-test",
		Messages:  []Message{{Role: RoleSystem, Content: "be concise"}, {Role: RoleUser, Content: "hello"}},
		MaxTokens: 32,
		JSONMode:  true,
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if response.Content != "answer" || response.PromptTokens != 5 || response.CompTokens != 2 {
		t.Fatalf("response = %+v", response)
	}
}

func TestEmbedderUsesTypedWirePayload(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/embeddings" {
			t.Errorf("request = %s %s, want POST /embeddings", r.Method, r.URL.Path)
		}
		var payload embeddingRequestPayload
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode typed embedding request: %v", err)
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		if payload.Model != "embedding-wire-test" || !reflect.DeepEqual(payload.Input, []string{"one", "two"}) {
			t.Errorf("embedding request = %+v", payload)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"embedding":[0.1,0.2]},{"embedding":[0.3,0.4]}]}`))
	}))
	defer server.Close()

	embedder := NewEmbedder(server.URL, "", "embedding-wire-test")
	vectors, err := embedder.Embed(context.Background(), "one", "two")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vectors) != 2 || len(vectors[0]) != 2 || vectors[0][0] != float32(0.1) || vectors[1][1] != float32(0.4) {
		t.Fatalf("vectors = %#v", vectors)
	}
}
