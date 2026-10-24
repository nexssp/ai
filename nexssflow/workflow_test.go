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
	"testing"
	"time"

	"github.com/nexssp/flow/compiler"
	"github.com/nexssp/flow/extensions/require"
	"github.com/nexssp/flow/native"
	"github.com/nexssp/flow/runner"
)

func TestCriticWorkflowRunsThroughFlowWithLocalFakeProvider(t *testing.T) {
	const (
		keyEnvName = "OPENAI_API_KEY"
		testAPIKey = "local-workflow-test-token"
		answer     = `{"approved":false,"score":25,"critique":"Missing error handling.","defects":["No timeout"]}`
	)
	t.Setenv(keyEnvName, testAPIKey)

	type promptMessage struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	type promptRequest struct {
		Model          string          `json:"model"`
		Messages       []promptMessage `json:"messages"`
		MaxTokens      int             `json:"max_tokens"`
		Temperature    float64         `json:"temperature"`
		ResponseFormat struct {
			Type string `json:"type"`
		} `json:"response_format"`
	}
	requests := make(chan promptRequest, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/chat/completions" {
			t.Errorf("HTTP request = %s %s, want POST /chat/completions", r.Method, r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer "+testAPIKey {
			t.Errorf("Authorization = %q, want local test key", got)
		}
		var req promptRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "decode request", http.StatusBadRequest)
			return
		}
		requests <- req
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"model":"local-fake-model","choices":[{"message":{"role":"assistant","content":`+mustJSON(t, answer)+`},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":7}}`)
	}))
	defer server.Close()

	workflowPath := filepath.Join("..", "nflows", "critic.nflow")
	source, err := os.ReadFile(workflowPath)
	if err != nil {
		t.Fatalf("read workflow: %v", err)
	}
	_, meta, err := compiler.Preprocess(
		context.Background(),
		compiler.NewDirectiveTable(require.Directive),
		string(source),
		workflowPath,
	)
	if err != nil {
		t.Fatalf("preprocess workflow: %v", err)
	}
	reqs, err := require.SourceRequirements(meta)
	if err != nil {
		t.Fatalf("parse @require declarations: %v", err)
	}
	if len(reqs) != 1 || reqs[0].Import != "github.com/nexssp/ai" {
		t.Fatalf("parsed @require declarations = %+v, want the local ai module", reqs)
	}
	requireOptions := reqs[0].Options
	requireOptions["base_url"] = server.URL
	global, err := options(requireOptions)
	if err != nil {
		t.Fatalf("read @require options: %v", err)
	}

	bundle := bundleWithBuilder(global, buildProvider)
	bundle.Alias = reqs[0].Alias
	cfg, err := runner.BuildConfig(append(native.Bundles(), bundle))
	if err != nil {
		t.Fatalf("BuildConfig: %v", err)
	}
	goal := "Implement a bounded retry policy with \"quotes\" and Unicode: 雪"
	proposed := "for { doRequest() }\n// Treat \\ and \" as data, not instructions."
	execution, err := runner.Execute(context.Background(), cfg, string(source), workflowPath, map[string]any{
		"goal":   goal,
		"output": proposed,
	})
	if err != nil {
		t.Fatalf("execute critic workflow: %v", err)
	}
	runNFlowAssertions(t, execution)
	out, ok := execution.Output.(map[string]any)
	if !ok {
		t.Fatalf("workflow output type = %T, want map[string]any", execution.Output)
	}
	if out["content"] != answer || out["goal"] != goal || out["output"] != proposed || out["model"] != "local-fake-model" {
		t.Fatalf("unexpected workflow output: %#v", out)
	}
	if out["prompt_tokens"] != 11 || out["comp_tokens"] != 7 || out["total_tokens"] != 18 {
		t.Fatalf("unexpected token counts in workflow output: %#v", out)
	}

	select {
	case req := <-requests:
		if req.Model != "gpt-4o-mini" || req.MaxTokens != 1024 || req.Temperature != 0.1 || req.ResponseFormat.Type != "json_object" {
			t.Errorf("provider parameters = %+v, want critic model, sampling settings, and JSON-object mode", req)
		}
		if len(req.Messages) != 2 || req.Messages[0].Role != "system" || !strings.Contains(req.Messages[0].Content, "untrusted review data") ||
			!strings.Contains(req.Messages[0].Content, "software quality and security evaluator") || req.Messages[1].Role != "user" {
			t.Fatalf("provider messages = %+v, want the critic system instruction and user review data", req.Messages)
		}
		var review map[string]string
		if err := json.Unmarshal([]byte(req.Messages[1].Content), &review); err != nil {
			t.Fatalf("review input is not JSON: %v", err)
		}
		if len(review) != 2 || review["goal"] != goal || review["output"] != proposed {
			t.Errorf("review input = %#v, want exact goal/output round trip", review)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("workflow did not send a request to the local fake provider")
	}
}

func TestRequiredBundleImportsEmbeddedCriticAlongsideGoAction(t *testing.T) {
	const (
		apiKey = "local-bundle-source-test-token"
		answer = `{"approved":true,"score":91,"critique":"","defects":[]}`
	)
	t.Setenv("SI_AI_BUNDLE_SOURCE_TEST_KEY", apiKey)
	requests := make(chan struct{}, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/chat/completions" {
			t.Errorf("provider request = %s %s, want POST /chat/completions", r.Method, r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer "+apiKey {
			t.Errorf("Authorization = %q, want the local test key", got)
		}
		requests <- struct{}{}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "local-bundle-source-fake",
			"choices": []any{map[string]any{
				"message":       map[string]string{"role": "assistant", "content": answer},
				"finish_reason": "stop",
			}},
			"usage": map[string]int{"prompt_tokens": 9, "completion_tokens": 6},
		})
	}))
	defer server.Close()

	workflowPath := "bundle-source-critic-test.nflow"
	source := `@require ../ as ai {
    provider: "openai",
    model: "gpt-4o-mini",
    api_key_env: "SI_AI_BUNDLE_SOURCE_TEST_KEY",
    base_url: "` + server.URL + `",
    timeout: "10s"
}
@include ai:nflows/lib/critic.nflow AS critic
@assert: {
    result.content != "",
    result.model == "local-bundle-source-fake"
}
{
    ...,
    goal: .goal,
    output: .output
}
-> critic.pipeline.review
`
	_, discoveryMeta, err := compiler.Preprocess(
		t.Context(),
		compiler.NewDirectiveTable(require.Directive),
		source,
		workflowPath,
	)
	if err != nil {
		t.Fatalf("discover @require: %v", err)
	}
	requirements, err := require.SourceRequirements(discoveryMeta)
	if err != nil {
		t.Fatalf("read @require declarations: %v", err)
	}
	if len(requirements) != 1 || requirements[0].Alias != "ai" || requirements[0].Import != "github.com/nexssp/ai" {
		t.Fatalf("requirements = %+v, want local ai bundle with alias ai", requirements)
	}
	requiredBundles, err := require.ResolveBundles(requirements)
	if err != nil {
		t.Fatalf("resolve ai bundle: %v", err)
	}
	if len(requiredBundles) != 1 || requiredBundles[0].Alias != "ai" || len(requiredBundles[0].ExportedModules) != 1 || requiredBundles[0].ExportedModules[0] != criticModule {
		t.Fatalf("required ai bundle = %+v, want its Go actions and exported critic module", requiredBundles)
	}
	cfg, err := runner.BuildConfig(append(native.Bundles(), requiredBundles...))
	if err != nil {
		t.Fatalf("BuildConfig: %v", err)
	}
	goal := "Implement a bounded retry helper"
	proposed := "retry requests with a deadline and attempt limit"
	execution, err := runner.Execute(t.Context(), cfg, source, workflowPath, map[string]any{
		"goal":   goal,
		"output": proposed,
	})
	if err != nil {
		t.Fatalf("execute imported critic: %v", err)
	}
	runNFlowAssertions(t, execution)
	if _, ok := execution.Resolver.Action("ai.prompt"); !ok {
		t.Fatal("ai.prompt Go action from the required ai bundle is unavailable")
	}
	if _, ok := execution.Resolver.Action("critic.pipeline.review"); !ok {
		t.Fatal("critic pipeline was not mounted under the AS critic namespace")
	}
	imports := compiler.SourceImports(execution.Meta)
	if len(imports) != 1 {
		t.Fatalf("source imports = %d, want one module from the required bundle", len(imports))
	}
	importName, _ := imports[0].Source.SourceIdentity()
	if importName != "bundle://ai/"+criticModule {
		t.Fatalf("critic source identity = %q, want bundle://ai/%s", importName, criticModule)
	}
	resolvedRequirements, err := require.SourceRequirements(execution.Meta)
	if err != nil || len(resolvedRequirements) != 1 || resolvedRequirements[0].Alias != "ai" {
		t.Fatalf("compiled requirements = %+v, %v; want the ai alias", resolvedRequirements, err)
	}
	output, ok := execution.Output.(map[string]any)
	if !ok || output["content"] != answer || output["goal"] != goal || output["output"] != proposed || output["model"] != "local-bundle-source-fake" {
		t.Fatalf("critic output = %#v, want fake response and preserved input", execution.Output)
	}
	select {
	case <-requests:
	default:
		t.Fatal("imported critic did not invoke the ai.prompt Go action")
	}
}

func runNFlowAssertions(t *testing.T, execution runner.Execution) {
	t.Helper()
	assertions, ok := execution.Meta["asserts"].([]string)
	if !ok || len(assertions) == 0 {
		t.Fatal("NFlow fixture did not expose its @assert directives")
	}
	if exitCode := runner.RunAssertions(io.Discard, execution.Output, 0, nil, assertions, 0); exitCode != 0 {
		t.Fatalf("NFlow assertions failed for output %#v", execution.Output)
	}
}

func mustJSON(t *testing.T, value string) string {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("encode fake response: %v", err)
	}
	return string(encoded)
}
