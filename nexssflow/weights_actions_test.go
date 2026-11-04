package nexssflow

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	flowtest "github.com/nexssp/flow/extensions/test"
	"github.com/nexssp/flow/native"
	"github.com/nexssp/flow/runner"

	"github.com/nexssp/ai/weights"
	"github.com/nexssp/ai/weights/safetensors"
)

func TestWeightsModuleExecutesThroughRequireAndImport(t *testing.T) {
	t.Chdir(t.TempDir())
	checkpointPath := "synthetic.safetensors"
	weight := make([]byte, 4)
	binary.LittleEndian.PutUint32(weight, math.Float32bits(1.25))
	if err := safetensors.Write(checkpointPath, map[string]safetensors.TensorData{
		"layer.weight": {DType: "F32", Shape: []uint64{1}, Data: weight},
	}, map[string]string{"fixture": "not-a-real-model"}, false); err != nil {
		t.Fatal(err)
	}

	cfg, err := runner.BuildConfig(append(native.Bundles(), Bundle(nil)))
	if err != nil {
		t.Fatalf("BuildConfig: %v", err)
	}
	source := `@require github.com/nexssp/ai as ai
@include ai:nflows/lib/weights.nflow AS weights

{
  model: {
    model_id: "synthetic-test",
    format: "safetensors",
    adapter: "safetensors.scalar-delta.v1",
    path: "synthetic.safetensors"
  },
  include_sha256: true,
  sample_elements: 1
} -> weights.pipeline.inspect
`
	execution, err := runner.Execute(context.Background(), cfg, source, "weights-module-test.nflow", nil)
	if err != nil {
		t.Fatalf("execute imported weights workflow: %v", err)
	}
	inspection, ok := execution.Output.(weights.Inspection)
	if !ok {
		t.Fatalf("workflow output type = %T, want weights.Inspection", execution.Output)
	}
	if inspection.Model.ID != "synthetic-test" || inspection.SHA256 == "" || len(inspection.Tensors) != 1 || len(inspection.Tensors[0].Sample) != 1 || inspection.Tensors[0].Sample[0] != 1.25 {
		t.Fatalf("unexpected NFlow inspection result: %+v", inspection)
	}
	patchSource := `@require github.com/nexssp/ai as ai
@include ai:nflows/lib/weights.nflow AS weights

{
  model: {
    model_id: "synthetic-test",
    format: "safetensors",
    adapter: "safetensors.scalar-delta.v1",
    path: "synthetic.safetensors"
  },
  output_path: "patched.safetensors",
  source_sha256: "` + inspection.SHA256 + `",
  edits: [{tensor: "layer.weight", index: 0, delta: 0.25, expected_value: 1.25}]
} -> weights.pipeline.patch
`
	patchExecution, err := runner.Execute(context.Background(), cfg, patchSource, "weights-patch-module-test.nflow", nil)
	if err != nil {
		t.Fatalf("execute imported weight patch: %v", err)
	}
	patch, ok := patchExecution.Output.(weights.PatchResult)
	if !ok || patch.InputSHA256 != inspection.SHA256 || patch.OutputSHA256 == "" || len(patch.Edits) != 1 || patch.Edits[0].After != 1.5 {
		t.Fatalf("unexpected NFlow patch result: %#v", patchExecution.Output)
	}
	patchedCheckpoint, err := safetensors.Open("patched.safetensors")
	if err != nil {
		t.Fatalf("open NFlow patch output: %v", err)
	}
	defer patchedCheckpoint.Close()
	patchedValue, err := patchedCheckpoint.ReadFloat("layer.weight", 0)
	if err != nil || patchedValue != 1.5 {
		t.Fatalf("NFlow patch output value=%v err=%v, want 1.5", patchedValue, err)
	}
}

func TestStandaloneWeightsInspectExampleRuns(t *testing.T) {
	packageDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	repoRoot := filepath.Dir(packageDir)
	sourcePath := filepath.Join(repoRoot, "nflows", "weights_inspect.nflow")
	source, err := os.ReadFile(sourcePath)
	if err != nil {
		t.Fatalf("read standalone example: %v", err)
	}
	t.Chdir(t.TempDir())
	checkpointPath := "synthetic.safetensors"
	weight := make([]byte, 4)
	binary.LittleEndian.PutUint32(weight, math.Float32bits(0.75))
	if err := safetensors.Write(checkpointPath, map[string]safetensors.TensorData{
		"layer.weight": {DType: "F32", Shape: []uint64{1}, Data: weight},
	}, nil, false); err != nil {
		t.Fatal(err)
	}
	cfg, err := runner.BuildConfig(append(native.Bundles(), Bundle(nil)))
	if err != nil {
		t.Fatalf("BuildConfig: %v", err)
	}
	execution, err := runner.Execute(context.Background(), cfg, string(source), sourcePath, map[string]any{
		"model": map[string]any{
			"model_id": "standalone-test",
			"format":   "safetensors",
			"adapter":  "safetensors.scalar-delta.v1",
			"path":     checkpointPath,
		},
	})
	if err != nil {
		t.Fatalf("execute standalone weights example: %v", err)
	}
	inspection, ok := execution.Output.(weights.Inspection)
	if !ok || inspection.Model.ID != "standalone-test" || inspection.SHA256 == "" {
		t.Fatalf("standalone example output = %#v, want hashed synthetic inspection", execution.Output)
	}
}

func TestWeightsNFlowAdapterContractWithTestDirective(t *testing.T) {
	packageDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	repoRoot := filepath.Dir(packageDir)
	sourcePath := filepath.Join(repoRoot, "nflows", "spec", "weights", "main.nflow")
	source, err := os.ReadFile(sourcePath)
	if err != nil {
		t.Fatalf("read NFlow @test contract: %v", err)
	}
	cfg, err := runner.BuildConfig(append(append(native.Bundles(), Bundle(nil)), flowtest.Bundle(nil)))
	if err != nil {
		t.Fatalf("BuildConfig with test extension: %v", err)
	}
	execution, err := runner.Execute(context.Background(), cfg, string(source), sourcePath, nil)
	if err != nil {
		t.Fatalf("execute NFlow @test adapter contract: %v", err)
	}
	if execution.Output == nil {
		t.Fatal("NFlow adapter contract returned no action output")
	}
}

func TestWeightsNFlowRejectsPyTorchWithoutAdapter(t *testing.T) {
	packageDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	repoRoot := filepath.Dir(packageDir)
	sourcePath := filepath.Join(repoRoot, "nflows", "spec", "weights", "main.nflow")
	source := `@require ../../../ as ai
@include ai:nflows/lib/weights.nflow AS weights

{
  model: {
    model_id: "pytorch-reject-test",
    format: "pytorch",
    path: "model.pt"
  }
} -> weights.pipeline.inspect
`
	cfg, err := runner.BuildConfig(append(native.Bundles(), Bundle(nil)))
	if err != nil {
		t.Fatalf("BuildConfig: %v", err)
	}
	_, err = runner.Execute(context.Background(), cfg, source, sourcePath, nil)
	if err == nil || !strings.Contains(err.Error(), "PyTorch") {
		t.Fatalf("NFlow inspect(PyTorch without adapter) error = %v, want safe rejection", err)
	}
}

func TestWeightsNFlowRejectsOutOfRangeSamplesAndMislabeledOutputs(t *testing.T) {
	packageDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	repoRoot := filepath.Dir(packageDir)
	sourcePath := filepath.Join(repoRoot, "nflows", "spec", "weights", "main.nflow")
	t.Chdir(t.TempDir())
	weight := make([]byte, 4)
	binary.LittleEndian.PutUint32(weight, math.Float32bits(1.25))
	if err := safetensors.Write("synthetic.safetensors", map[string]safetensors.TensorData{
		"layer.weight": {DType: "F32", Shape: []uint64{1}, Data: weight},
	}, nil, false); err != nil {
		t.Fatal(err)
	}
	cfg, err := runner.BuildConfig(append(native.Bundles(), Bundle(nil)))
	if err != nil {
		t.Fatalf("BuildConfig: %v", err)
	}
	base := `@require ../../../ as ai
@include ai:nflows/lib/weights.nflow AS weights
`
	cases := []struct {
		name   string
		source string
		want   string
	}{
		{
			name: "sample limit",
			source: base + `{
  model: { model_id: "boundary-test", format: "safetensors", adapter: "safetensors.scalar-delta.v1", path: "synthetic.safetensors" },
  sample_elements: 33
} -> weights.pipeline.inspect
`,
			want: "sample_elements",
		},
		{
			name: "output extension",
			source: base + `{
  model: { model_id: "boundary-test", format: "safetensors", adapter: "safetensors.scalar-delta.v1", path: "synthetic.safetensors" },
  output_path: "patched.bin",
  source_sha256: "` + strings.Repeat("0", 64) + `",
  edits: [{ tensor: "layer.weight", index: 0, delta: 0.25 }]
} -> weights.pipeline.patch
`,
			want: ".safetensors",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := runner.Execute(context.Background(), cfg, tc.source, sourcePath, nil)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("NFlow request error = %v, want substring %q", err, tc.want)
			}
		})
	}
}
