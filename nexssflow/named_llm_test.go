package nexssflow

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/nexssp/flow/compiler"
	"github.com/nexssp/flow/extensions/require"
	"github.com/nexssp/flow/native"
	"github.com/nexssp/flow/runner"

	"github.com/nexssp/ai/engine/llm"
)

func TestNamedLLMHandlesUseIndependentKeysThroughNFlow(t *testing.T) {
	const (
		analystKey     = "fake-analyst-key"
		criticKey      = "fake-critic-key"
		synthesizerKey = "fake-synthesizer-key"
		sharedKey      = "must-not-be-used"
	)
	t.Setenv("GROQ_ANALYST_API_KEY", analystKey)
	t.Setenv("GROQ_CRITIC_API_KEY", criticKey)
	t.Setenv("GROQ_SYNTHESIZER_API_KEY", synthesizerKey)
	t.Setenv("GROQ_API_KEY", sharedKey)
	t.Setenv("LLM_API_KEY", sharedKey)

	type promptMessage struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	type promptRequest struct {
		Model    string          `json:"model"`
		Messages []promptMessage `json:"messages"`
	}
	type observedRequest struct {
		authorization string
		request       promptRequest
	}
	responses := []string{"analysis from analyst", "revision from critic", "final synthesis"}
	requests := make(chan observedRequest, len(responses))
	var mu sync.Mutex
	requestCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/chat/completions" {
			http.Error(w, "unexpected route", http.StatusNotFound)
			return
		}
		var request promptRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, "decode request", http.StatusBadRequest)
			return
		}
		mu.Lock()
		index := requestCount
		requestCount++
		mu.Unlock()
		if index >= len(responses) {
			http.Error(w, "unexpected extra request", http.StatusBadRequest)
			return
		}
		requests <- observedRequest{authorization: r.Header.Get("Authorization"), request: request}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": request.Model,
			"choices": []any{map[string]any{
				"message":       map[string]string{"role": "assistant", "content": responses[index]},
				"finish_reason": "stop",
			}},
			"usage": map[string]int{"prompt_tokens": 5, "completion_tokens": 3},
		})
	}))
	defer server.Close()
	t.Setenv("LLM_BASE_URL", server.URL)

	workflowPath := filepath.Join("..", "nflows", "examples", "three_named_groq.nflow")
	source, err := os.ReadFile(workflowPath)
	if err != nil {
		t.Fatalf("read NFlow example: %v", err)
	}
	_, discoveryMeta, err := compiler.Preprocess(
		context.Background(),
		compiler.NewDirectiveTable(require.Directive),
		string(source),
		workflowPath,
	)
	if err != nil {
		t.Fatalf("parse @require declaration: %v", err)
	}
	requirements, err := require.SourceRequirements(discoveryMeta)
	if err != nil {
		t.Fatalf("read @require declarations: %v", err)
	}
	if len(requirements) != 1 || requirements[0].Import != "github.com/nexssp/ai" || requirements[0].Alias != "ai" {
		t.Fatalf("requirements = %+v, want the ai bundle under alias ai", requirements)
	}
	bundles, err := require.ResolveBundles(requirements)
	if err != nil {
		t.Fatalf("resolve in-process AI bundle: %v", err)
	}
	cfg, err := runner.BuildConfig(append(native.Bundles(), bundles...))
	if err != nil {
		t.Fatalf("BuildConfig: %v", err)
	}

	execution, err := runner.Execute(context.Background(), cfg, string(source), workflowPath, map[string]any{
		"task": "Recommend a safe timeout policy for a public API.",
	})
	if err != nil {
		t.Fatalf("execute named LLM workflow: %v", err)
	}
	runNFlowAssertions(t, execution)
	output, ok := execution.Output.(PromptResult)
	if !ok || output.Content != responses[2] || output.Provider != "groq" {
		t.Fatalf("workflow result = %#v, want final Groq response", execution.Output)
	}
	for _, name := range []string{"llm.analyst", "llm.critic", "llm.synthesizer"} {
		if _, ok := execution.Resolver.Action(name); !ok {
			t.Fatalf("named action %q was not materialized", name)
		}
	}

	wantKeys := []string{analystKey, criticKey, synthesizerKey}
	wantPromptParts := [][]string{
		{"Recommend a safe timeout policy"},
		{responses[0]},
		{responses[0], responses[1]},
	}
	for i := range wantKeys {
		select {
		case observed := <-requests:
			if got, want := observed.authorization, "Bearer "+wantKeys[i]; got != want {
				t.Errorf("request %d Authorization = %q, want its named key", i+1, got)
			}
			if observed.request.Model != "llama-3.3-70b-versatile" {
				t.Errorf("request %d model = %q, want the declared model", i+1, observed.request.Model)
			}
			if len(observed.request.Messages) != 2 || observed.request.Messages[1].Role != "user" {
				t.Fatalf("request %d messages = %+v, want system and user prompts", i+1, observed.request.Messages)
			}
			for _, part := range wantPromptParts[i] {
				if !strings.Contains(observed.request.Messages[1].Content, part) {
					t.Errorf("request %d user prompt = %q, want it to include %q", i+1, observed.request.Messages[1].Content, part)
				}
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("timed out waiting for named LLM request %d", i+1)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if requestCount != len(responses) {
		t.Errorf("provider request count = %d, want %d", requestCount, len(responses))
	}
}

func TestNamedLLMWithMissingExplicitKeyDoesNotUseSharedFallback(t *testing.T) {
	t.Setenv("GROQ_PRIVATE_KEY", "")
	t.Setenv("GROQ_API_KEY", "shared-fallback-key")
	called := false
	bundle, err := NewBundleWithProviderBuilder(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) {
		called = true
		return nil, errors.New("provider builder must not run")
	})
	if err != nil {
		t.Fatalf("create AI bundle: %v", err)
	}
	bundle.Alias = "ai"
	cfg, err := runner.BuildConfig(append(native.Bundles(), bundle))
	if err != nil {
		t.Fatalf("BuildConfig: %v", err)
	}
	source := `@llm private { provider: "groq", model: "mock-model", api_key_env: "GROQ_PRIVATE_KEY" }
{ user: "This must fail before dispatch." } -> llm.private`
	_, err = runner.Execute(context.Background(), cfg, source, "missing-named-key.nflow", map[string]any{})
	if err == nil || !strings.Contains(err.Error(), "GROQ_PRIVATE_KEY") {
		t.Fatalf("execution error = %v, want missing named key validation", err)
	}
	if called {
		t.Fatal("provider builder ran despite the missing explicit named key")
	}
}
