package nexssflow

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"

	"github.com/nexssp/flow/native"
	"github.com/nexssp/flow/runner"
)

func TestLlamaCppCompareExampleRunsBaselineAndCandidateViaArgv(t *testing.T) {
	packageDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	repoRoot := filepath.Dir(packageDir)
	sourcePath := filepath.Join(repoRoot, "nflows", "examples", "weights", "llama_cpp_compare.nflow")
	source, err := os.ReadFile(sourcePath)
	if err != nil {
		t.Fatalf("read llama.cpp comparison example: %v", err)
	}
	helperPath := buildLlamaCLIArgvRecorder(t)

	cfg, err := runner.BuildConfig(append(native.Bundles(), Bundle(nil)))
	if err != nil {
		t.Fatalf("BuildConfig: %v", err)
	}
	execution, err := runner.Execute(context.Background(), cfg, string(source), sourcePath, map[string]any{
		"llama_cli_path":       helperPath,
		"baseline_model_path":  "models/base model.gguf",
		"candidate_model_path": "models/candidate model.gguf",
		"prompt":               "Question: What is the capital of France? Answer:",
		"n_predict":            15,
		"temperature":          0.0,
	})
	if err != nil {
		t.Fatalf("execute llama.cpp comparison example: %v", err)
	}
	result, ok := execution.Output.(map[string]any)
	if !ok {
		t.Fatalf("comparison output type = %T, want map[string]any", execution.Output)
	}
	for label, modelPath := range map[string]string{
		"baseline":  "models/base model.gguf",
		"candidate": "models/candidate model.gguf",
	} {
		group, ok := result[label].(map[string]any)
		if !ok {
			t.Fatalf("%s output type = %T, want map[string]any", label, result[label])
		}
		stdout, ok := group["stdout"].(string)
		if !ok {
			t.Fatalf("%s stdout type = %T, want string", label, group["stdout"])
		}
		var args []string
		if err := json.Unmarshal([]byte(stdout), &args); err != nil {
			t.Fatalf("decode %s helper argv %q: %v", label, stdout, err)
		}
		want := []string{
			"-m", modelPath,
			"-p", "Question: What is the capital of France? Answer:",
			"-n", "15",
			"--temp", "0",
			"--log-disable",
		}
		if !reflect.DeepEqual(args, want) {
			t.Fatalf("%s argv = %#v, want %#v", label, args, want)
		}
		if group["exit_code"] != 0 {
			t.Fatalf("%s exit_code = %#v, want 0", label, group["exit_code"])
		}
	}
	if _, ok := result["interpretation"].(string); !ok {
		t.Fatalf("comparison interpretation = %#v, want explanatory string", result["interpretation"])
	}
}

func buildLlamaCLIArgvRecorder(t *testing.T) string {
	t.Helper()
	directory := t.TempDir()
	sourcePath := filepath.Join(directory, "argv-recorder.go")
	executablePath := filepath.Join(directory, "argv-recorder")
	if runtime.GOOS == "windows" {
		executablePath += ".exe"
	}
	const source = `package main
import (
  "encoding/json"
  "os"
)
func main() {
  _ = json.NewEncoder(os.Stdout).Encode(os.Args[1:])
}
`
	if err := os.WriteFile(sourcePath, []byte(source), 0o600); err != nil {
		t.Fatalf("write fake llama-cli: %v", err)
	}
	command := exec.CommandContext(t.Context(), "go", "build", "-o", executablePath, sourcePath)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("build fake llama-cli: %v\n%s", err, output)
	}
	return executablePath
}
