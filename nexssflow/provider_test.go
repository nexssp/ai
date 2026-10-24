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
		keyEnvName = "SI_AI_TEST_PROVIDER_KEY"
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

	act := newPromptAction(GlobalOptions{
		APIKeyEnv:    keyEnvName,
		Timeout:      2 * time.Second,
		DefaultModel: "configured-local-model",
		Provider:     "openai",
		BaseURL:      server.URL + "/custom/v1/",
	})
	got, err := act.DoAny(context.Background(), map[string]any{
		"job_id":      "keep-me",
		"system":      "Be concise.",
		"user":        "What is 2+2?",
		"max_tokens":  32,
		"temperature": 0.25,
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

	out, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("action output type = %T, want map[string]any", got)
	}
	if out["content"] != "local fake answer" || out["model"] != "served-model" ||
		out["job_id"] != "keep-me" || out["prompt_tokens"] != 4 || out["comp_tokens"] != 2 || out["total_tokens"] != 6 {
		t.Errorf("action output = %#v, want provider response merged with input state", out)
	}
}
