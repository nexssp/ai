package nexssflow

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestPromptActionUsesConfiguredProviderAndSendsExpectedRequest(t *testing.T) {
	const (
		keyEnvName = "LLM_API_KEY"
		testAPIKey = "local-test-token-not-a-secret"
	)
	t.Setenv(keyEnvName, testAPIKey)

	type observedRequest struct {
		method        string
		path          string
		contentType   string
		authorization string
		body          struct {
			Model    string `json:"model"`
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
			MaxTokens   int     `json:"max_tokens"`
			Temperature float64 `json:"temperature"`
		}
	}
	requests := make(chan observedRequest, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var observed observedRequest
		observed.method = r.Method
		observed.path = r.URL.Path
		observed.contentType = r.Header.Get("Content-Type")
		observed.authorization = r.Header.Get("Authorization")
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read request body", http.StatusBadRequest)
			return
		}
		if err := json.Unmarshal(body, &observed.body); err != nil {
			http.Error(w, "decode request body", http.StatusBadRequest)
			return
		}
		requests <- observed
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"model":"served-model","choices":[{"message":{"role":"assistant","content":"local fake answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2}}`)
	}))
	defer server.Close()
	t.Setenv("LLM_PROVIDER", "openai-compatible")
	t.Setenv("LLM_MODEL", "configured-local-model")
	t.Setenv("LLM_BASE_URL", server.URL+"/custom/v1/")

	act := newPromptAction(GlobalOptions{
		Timeout: 2 * time.Second,
	})
	got, err := act.DoAny(context.Background(), PromptRequest{
		CandidateID: "candidate-a",
		AttemptID:   "attempt-1",
		Context:     json.RawMessage(`{"job":"keep-explicit"}`),
		System:      "Be concise.",
		User:        "What is 2+2?",
		MaxTokens:   32,
		Temperature: 0.25,
	})
	if err != nil {
		t.Fatalf("prompt action: %v", err)
	}

	observed := <-requests
	if observed.method != http.MethodPost {
		t.Errorf("HTTP method = %q, want POST", observed.method)
	}
	if observed.path != "/custom/v1/chat/completions" {
		t.Errorf("request path = %q, want /custom/v1/chat/completions", observed.path)
	}
	if !strings.HasPrefix(observed.contentType, "application/json") {
		t.Errorf("Content-Type = %q, want application/json", observed.contentType)
	}
	if observed.authorization != "Bearer "+testAPIKey {
		t.Errorf("Authorization = %q, want configured test key", observed.authorization)
	}
	if observed.body.Model != "configured-local-model" {
		t.Errorf("request model = %q, want configured-local-model", observed.body.Model)
	}
	if len(observed.body.Messages) != 2 ||
		observed.body.Messages[0].Role != "system" || observed.body.Messages[0].Content != "Be concise." ||
		observed.body.Messages[1].Role != "user" || observed.body.Messages[1].Content != "What is 2+2?" {
		t.Errorf("request messages = %+v, want system and user prompts in order", observed.body.Messages)
	}
	if observed.body.MaxTokens != 32 || observed.body.Temperature != 0.25 {
		t.Errorf("request sampling settings = max_tokens:%d temperature:%v, want 32 and 0.25", observed.body.MaxTokens, observed.body.Temperature)
	}

	out, ok := got.(PromptResult)
	if !ok {
		t.Fatalf("action output type = %T, want PromptResult", got)
	}
	if out.Content != "local fake answer" || out.Model != "served-model" || out.CandidateID != "candidate-a" || out.AttemptID != "attempt-1" || out.PromptTokens != 4 || out.CompTokens != 2 || out.TotalTokens != 6 || string(out.Context) != `{"job":"keep-explicit"}` {
		t.Errorf("action output = %+v, want typed provider result and explicitly carried context", out)
	}
}

func TestBuildProviderRequiresExplicitProvider(t *testing.T) {
	t.Setenv("LLM_PROVIDER", "")
	if _, err := buildProvider(GlobalOptions{}, LLMDecl{}); err == nil || !strings.Contains(err.Error(), "LLM_PROVIDER") {
		t.Fatalf("buildProvider error = %v, want a clear provider configuration error", err)
	}
}

func TestBuildProviderAcceptsAvailableDrivers(t *testing.T) {
	t.Setenv("LLM_MODEL", "")
	t.Setenv("LLM_API_KEY", "")
	t.Setenv("LLM_BASE_URL", "https://custom.example/v1")
	for _, name := range []string{
		"openai-compatible", "openai", "groq", "nvidia", "nim", "deepseek", "openrouter",
		"lmstudio", "vllm", "xai", "inception", "anthropic", "claude", "gemini", "google", "ollama", "ollama_cloud",
	} {
		t.Run(name, func(t *testing.T) {
			t.Setenv("LLM_PROVIDER", name)
			provider, err := buildProvider(GlobalOptions{}, LLMDecl{})
			if err != nil {
				t.Fatalf("buildProvider(%q): %v", name, err)
			}
			if provider.Name() == "" {
				t.Fatalf("provider %q has an empty name", name)
			}
		})
	}
}
