package nexssflow

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/nexssp/flow/compiler"
	"github.com/nexssp/flow/extensions/require"
	"github.com/nexssp/flow/native"
	"github.com/nexssp/flow/runner"

	"github.com/nexssp/ai/engine/llm"
)

func TestRolesWorkflowSharesMessagesWithinOneFlowRun(t *testing.T) {
	const (
		keyEnvName = "SI_AI_TEST_PROVIDER_KEY"
		testAPIKey = "local-team-test-token"
		goal       = "Suggest a safe API timeout policy."
		firstReply = "The service must enforce a bounded request deadline."
		lastReply  = "Set a bounded deadline and return a controlled timeout response."
	)
	t.Setenv(keyEnvName, testAPIKey)

	type promptMessage struct {
		Role    string `json:"role"`
		Name    string `json:"name"`
		Content string `json:"content"`
	}
	type promptRequest struct {
		Model       string          `json:"model"`
		Messages    []promptMessage `json:"messages"`
		MaxTokens   int             `json:"max_tokens"`
		Temperature float64         `json:"temperature"`
	}
	requests := make(chan promptRequest, 2)
	var callCount atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/chat/completions" {
			http.Error(w, "unexpected endpoint", http.StatusNotFound)
			return
		}
		if got := r.Header.Get("Authorization"); got != "Bearer "+testAPIKey {
			http.Error(w, "unexpected authorization", http.StatusUnauthorized)
			return
		}
		var req promptRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "decode request", http.StatusBadRequest)
			return
		}
		requests <- req
		content := firstReply
		if callCount.Add(1) == 2 {
			content = lastReply
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"model":"local-fake-model","choices":[{"message":{"role":"assistant","content":`+mustJSON(t, content)+`},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3}}`)
	}))
	defer server.Close()

	workflowPath := filepath.Join("..", "nflows", "roles.nflow")
	source, err := os.ReadFile(workflowPath)
	if err != nil {
		t.Fatalf("read workflow: %v", err)
	}
	_, meta, err := compiler.Preprocess(context.Background(), native.Directives(), string(source), workflowPath)
	if err != nil {
		t.Fatalf("preprocess workflow: %v", err)
	}
	reqs, err := require.FromMeta(meta)
	if err != nil {
		t.Fatalf("preprocess from meta: %v", err)
	}
	if len(reqs) != 1 || reqs[0].Import != "github.com/nexssp/ai" {
		t.Fatalf("parsed @require declarations = %+v, want ai module", reqs)
	}
	requireOptions := reqs[0].Options
	requireOptions["base_url"] = server.URL
	global, err := options(requireOptions)
	if err != nil {
		t.Fatalf("read @require options: %v", err)
	}

	cfg, err := runner.BuildConfig(append(native.Bundles(), bundleWithBuilder(global, buildProvider)))
	if err != nil {
		t.Fatalf("BuildConfig: %v", err)
	}
	if _, ok := cfg.Resolver.Action("ai.team"); !ok {
		t.Fatal("ai.team was not mounted in the Flow resolver")
	}
	execution, err := runner.Execute(context.Background(), cfg, string(source), workflowPath, nil)
	if err != nil {
		t.Fatalf("execute roles workflow: %v", err)
	}
	out, ok := execution.Output.(map[string]any)
	if !ok {
		t.Fatalf("workflow output type = %T, want map[string]any", execution.Output)
	}
	if out["content"] != lastReply || out["goal"] != goal || out["prompt_tokens"] != 10 || out["comp_tokens"] != 6 || out["total_tokens"] != 16 {
		t.Fatalf("unexpected team output: %#v", out)
	}
	turns, ok := out["agents"].([]RoleTurn)
	if !ok || len(turns) != 2 {
		t.Fatalf("agent turns = %#v, want two role results", out["agents"])
	}
	if turns[0].Name != "analyst" || turns[0].Role != "Analyst" || turns[0].Content != firstReply ||
		turns[1].Name != "reviewer" || turns[1].Role != "Reviewer" || turns[1].Content != lastReply {
		t.Fatalf("unexpected role outputs: %+v", turns)
	}

	var observed [2]promptRequest
	for i := range observed {
		select {
		case observed[i] = <-requests:
		case <-time.After(2 * time.Second):
			t.Fatalf("role %d did not call the local fake provider", i+1)
		}
	}
	if len(observed[0].Messages) != 2 || observed[0].Messages[0].Role != "system" || !strings.Contains(observed[0].Messages[0].Content, "Analyst role agent \"analyst\"") ||
		observed[0].Messages[1].Role != "user" || observed[0].Messages[1].Content != goal {
		t.Fatalf("first role request does not contain its role prompt and goal: %+v", observed[0])
	}
	if observed[0].Model != "local-fake-model" || observed[0].MaxTokens != 120 || observed[0].Temperature != 0.1 {
		t.Errorf("first role request options = %+v", observed[0])
	}
	if len(observed[1].Messages) != 3 || observed[1].Messages[0].Role != "system" || !strings.Contains(observed[1].Messages[0].Content, "Reviewer role agent \"reviewer\"") ||
		observed[1].Messages[1].Role != "user" || observed[1].Messages[1].Content != goal ||
		observed[1].Messages[2].Role != "assistant" || observed[1].Messages[2].Name != "analyst" || observed[1].Messages[2].Content != firstReply {
		t.Fatalf("second role request did not receive the first role's named output as context: %+v", observed[1])
	}
}

func TestTeamRequiresAtLeastTwoRoleAgentsBeforeProviderAccess(t *testing.T) {
	providerBuilds := 0
	act := newTeamActionWith(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) {
		providerBuilds++
		return nil, nil
	})
	_, err := act.DoAny(context.Background(), map[string]any{
		"goal": "Only one role was declared.",
		"agents": []any{
			map[string]any{"name": "analyst", "role": "Analyst", "instructions": "Analyze."},
		},
	})
	if err == nil || !strings.Contains(err.Error(), "at least two role agents") {
		t.Fatalf("one-role error = %v, want minimum-team validation", err)
	}
	if providerBuilds != 0 {
		t.Fatalf("provider was built %d times for an invalid team, want zero", providerBuilds)
	}
}
