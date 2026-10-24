package nexssflow

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

const parallelRoleTestAPIKey = "local-parallel-role-test-token"

func TestParallelRoleWorkflowViaFlowCLIWithLocalFakeProvider(t *testing.T) {
	flowCLI := os.Getenv("NFLOW_BIN")
	flowCWD := os.Getenv("NFLOW_CWD")
	if flowCLI == "" || flowCWD == "" {
		t.Skip("set NFLOW_BIN and NFLOW_CWD to run this Flow CLI integration test")
	}
	if _, err := os.Stat(flowCLI); err != nil {
		t.Fatalf("Flow CLI %q: %v", flowCLI, err)
	}
	if info, err := os.Stat(flowCWD); err != nil || !info.IsDir() {
		t.Fatalf("NFLOW_CWD %q must be an existing directory (stat error: %v)", flowCWD, err)
	}

	const (
		analystFinding        = "analyst: identify timeout and retry trade-offs"
		securityFinding       = "security: bound retries and avoid request amplification"
		implementationFinding = "implementation: set a deadline and bounded backoff"
		synthesisContent      = "summary: " + analystFinding + " | " + securityFinding + " | " + implementationFinding
	)
	roleFindings := map[string]string{
		"ROLE: ANALYST":     analystFinding,
		"ROLE: SECURITY":    securityFinding,
		"ROLE: IMPLEMENTER": implementationFinding,
	}
	roleUsers := map[string]string{
		"ROLE: ANALYST":     "Assess reliability risks and expected failure modes for a 30-second API timeout.",
		"ROLE: SECURITY":    "Assess security and abuse risks of a 30-second API timeout.",
		"ROLE: IMPLEMENTER": "Recommend an implementation strategy for a 30-second API timeout.",
	}
	expectedSynthesisUser := "Combine independent findings in source order:\nAnalyst: " + analystFinding + "\nSecurity: " + securityFinding + "\nImplementation: " + implementationFinding

	t.Setenv("SI_AI_TEST_PROVIDER_KEY", parallelRoleTestAPIKey)
	var mu sync.Mutex
	var roleCalls, activeRoles, maxActiveRoles, synthesisCalls int
	roleCounts := make(map[string]int)
	var synthesisUsers []string
	rolesReleased := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
			t.Errorf("fake provider request = %s %s, want POST /v1/chat/completions", r.Method, r.URL.Path)
			http.Error(w, "unexpected path", http.StatusNotFound)
			return
		}
		if got := r.Header.Get("Authorization"); got != "Bearer "+parallelRoleTestAPIKey {
			t.Errorf("Authorization = %q, want the local test token", got)
		}

		var request struct {
			Model    string `json:"model"`
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode fake provider request: %v", err)
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
		if request.Model != "local-fake-model" {
			t.Errorf("model = %q, want local-fake-model", request.Model)
		}
		if len(request.Messages) != 2 || request.Messages[0].Role != "system" || request.Messages[1].Role != "user" {
			t.Errorf("messages = %+v, want one system and one user message", request.Messages)
			http.Error(w, "invalid messages", http.StatusBadRequest)
			return
		}
		system, user := request.Messages[0].Content, request.Messages[1].Content

		role := ""
		for marker := range roleFindings {
			if strings.Contains(system, marker) {
				role = marker
				break
			}
		}
		if role == "" && strings.Contains(system, "ROLE: SYNTHESIS") {
			mu.Lock()
			synthesisCalls++
			synthesisUsers = append(synthesisUsers, user)
			mu.Unlock()
			if user != expectedSynthesisUser {
				t.Errorf("synthesis user message = %q, want all three role results in source order: %q", user, expectedSynthesisUser)
			}
			writeParallelRoleCompletion(w, synthesisContent)
			return
		}
		if role == "" {
			t.Errorf("unrecognized system prompt: %q", system)
			http.Error(w, "unrecognized role", http.StatusBadRequest)
			return
		}
		if user != roleUsers[role] {
			t.Errorf("%s user message = %q, want %q", role, user, roleUsers[role])
		}

		mu.Lock()
		roleCalls++
		roleCounts[role]++
		activeRoles++
		if activeRoles > maxActiveRoles {
			maxActiveRoles = activeRoles
		}
		if roleCalls == len(roleFindings) {
			close(rolesReleased)
		}
		mu.Unlock()

		select {
		case <-rolesReleased:
		case <-time.After(4 * time.Second):
			t.Errorf("role request %q waited for the other independent role calls to overlap", role)
			http.Error(w, "role overlap barrier timed out", http.StatusGatewayTimeout)
			mu.Lock()
			activeRoles--
			mu.Unlock()
			return
		}
		writeParallelRoleCompletion(w, roleFindings[role])
		mu.Lock()
		activeRoles--
		mu.Unlock()
	}))
	defer server.Close()

	workflowPath := filepath.Join("..", "nflows", "parallel_roles.nflow")
	source, err := os.ReadFile(workflowPath)
	if err != nil {
		t.Fatalf("read workflow: %v", err)
	}
	text := string(source)
	oldModel := `model: "gpt-4o-mini",`
	newModel := `model: "local-fake-model",` + "\n" + `    base_url: "` + server.URL + `/v1",`
	if strings.Count(text, oldModel) != 1 {
		t.Fatalf("workflow must contain exactly one %q setting", oldModel)
	}
	text = strings.Replace(text, oldModel, newModel, 1)
	oldKeyEnv := `api_key_env: "OPENAI_API_KEY"`
	if strings.Count(text, oldKeyEnv) != 1 {
		t.Fatalf("workflow must contain exactly one %q setting", oldKeyEnv)
	}
	text = strings.Replace(text, oldKeyEnv, `api_key_env: "SI_AI_TEST_PROVIDER_KEY"`, 1)

	flowFile, err := os.CreateTemp(filepath.Join("..", "nflows"), ".parallel-roles-cli-*.nflow")
	if err != nil {
		t.Fatalf("create temporary workflow beside fixture: %v", err)
	}
	flowPath, err := filepath.Abs(flowFile.Name())
	if err != nil {
		_ = flowFile.Close()
		t.Fatalf("resolve temporary workflow path: %v", err)
	}
	defer func() { _ = os.Remove(flowPath) }()
	if _, err := flowFile.WriteString(text); err != nil {
		_ = flowFile.Close()
		t.Fatalf("write temporary workflow: %v", err)
	}
	if err := flowFile.Close(); err != nil {
		t.Fatalf("close temporary workflow: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, flowCLI, "run", flowPath, "--json")
	cmd.Dir = flowCWD
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("Flow CLI failed: %v\nstdout:\n%s\nstderr:\n%s", err, stdout.String(), stderr.String())
	}

	var output map[string]any
	if err := json.Unmarshal(bytes.TrimSpace(stdout.Bytes()), &output); err != nil {
		t.Fatalf("decode Flow CLI JSON output %q: %v\nstderr:\n%s", stdout.String(), err, stderr.String())
	}
	if output["content"] != synthesisContent {
		t.Fatalf("final content = %#v, want %q", output["content"], synthesisContent)
	}

	mu.Lock()
	defer mu.Unlock()
	if roleCalls != 3 || maxActiveRoles < 3 {
		t.Errorf("role calls=%d, maximum overlapping handlers=%d; want 3 calls and overlap of all 3", roleCalls, maxActiveRoles)
	}
	if len(roleCounts) != len(roleFindings) || roleCounts["ROLE: ANALYST"] != 1 || roleCounts["ROLE: SECURITY"] != 1 || roleCounts["ROLE: IMPLEMENTER"] != 1 {
		t.Errorf("role call counts = %#v, want one call per independent role", roleCounts)
	}
	if synthesisCalls != 1 || len(synthesisUsers) != 1 || synthesisUsers[0] != expectedSynthesisUser {
		t.Errorf("synthesis calls/users = %d/%q, want one request containing all three role results", synthesisCalls, synthesisUsers)
	}
}

func writeParallelRoleCompletion(w http.ResponseWriter, content string) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"model": "local-fake-model",
		"choices": []any{map[string]any{
			"message":       map[string]string{"role": "assistant", "content": content},
			"finish_reason": "stop",
		}},
		"usage": map[string]int{"prompt_tokens": 1, "completion_tokens": 1},
	})
}
