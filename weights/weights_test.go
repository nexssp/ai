package weights

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/nexssp/ai/weights/gguf"
	"github.com/nexssp/ai/weights/safetensors"
)

func TestAdaptersReturnsDefensiveSupportedAdapters(t *testing.T) {
	adapters := Adapters()
	if len(adapters) != 2 {
		t.Fatalf("Adapters() returned %d entries, want 2", len(adapters))
	}
	if adapters[0].ID != AdapterSafeTensorsScalarDelta || adapters[0].Format != FormatSafeTensors || !adapters[0].SupportsShardedFiles || adapters[0].SupportsInference {
		t.Fatalf("SafeTensors adapter advertises unexpected capability: %+v", adapters[0])
	}
	if adapters[1].ID != AdapterGGUFScalarDelta || adapters[1].Format != FormatGGUF || adapters[1].SupportsShardedFiles || adapters[1].SupportsInference {
		t.Fatalf("GGUF adapter advertises unexpected capability: %+v", adapters[1])
	}
	adapters[0].EditableDTypes[0] = "CORRUPTED"
	adapters[1].PreservedDTypes[0] = "CORRUPTED"
	if got := Adapters()[0].EditableDTypes[0]; got != "F16" {
		t.Fatalf("Adapters() leaked mutable SafeTensors state: first dtype = %q", got)
	}
	if got := Adapters()[1].PreservedDTypes[0]; got != "I8" {
		t.Fatalf("Adapters() leaked mutable GGUF state: first preserved dtype = %q", got)
	}
}

func TestInspectAndPatchSyntheticSafeTensors(t *testing.T) {
	t.Chdir(t.TempDir())
	source := "base.safetensors"
	output := "edited.safetensors"
	writeModelFixture(t, source)
	model := testModel(source)

	inspection, err := Inspect(context.Background(), InspectRequest{
		Model: model, IncludeSHA256: true, SampleElements: 2,
	})
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if inspection.Model.ID != model.ID || inspection.SizeBytes <= 0 || len(inspection.Tensors) != 2 {
		t.Fatalf("unexpected inspection: %+v", inspection)
	}
	if inspection.Tensors[0].Name != "layer.bias" || inspection.Tensors[1].Name != "layer.weight" {
		t.Fatalf("tensor names are not deterministic: %+v", inspection.Tensors)
	}
	if len(inspection.Tensors[1].Sample) != 2 || inspection.Tensors[1].Sample[0] != 1.5 || inspection.Tensors[1].Sample[1] != -2 {
		t.Fatalf("unexpected bounded samples: %+v", inspection.Tensors[1].Sample)
	}
	if inspection.SHA256 != testFileDigest(t, source) || len(inspection.Files) != 1 {
		t.Fatalf("inspection provenance is inconsistent: %+v", inspection)
	}

	expected := -2.0
	patched, err := Patch(context.Background(), PatchRequest{
		Model: model, OutputPath: output, SourceSHA256: inspection.SHA256,
		Edits: []Edit{{Tensor: "layer.weight", Index: 1, Delta: 0.25, ExpectedValue: &expected}},
	})
	if err != nil {
		t.Fatalf("Patch() error = %v", err)
	}
	if patched.InputSHA256 != inspection.SHA256 || patched.OutputSHA256 != testFileDigest(t, output) || patched.OutputSHA256 == patched.InputSHA256 {
		t.Fatalf("patch digests are inconsistent: %+v", patched)
	}
	if len(patched.Edits) != 1 || patched.Edits[0].Before != -2 || patched.Edits[0].After != -1.75 || len(patched.InputFiles) != 1 || len(patched.OutputFiles) != 1 {
		t.Fatalf("unexpected patch report: %+v", patched)
	}

	original, err := safetensors.Open(source)
	if err != nil {
		t.Fatal(err)
	}
	defer original.Close()
	originalValue, err := original.ReadFloat("layer.weight", 1)
	if err != nil || originalValue != -2 {
		t.Fatalf("source checkpoint was changed: value=%v err=%v", originalValue, err)
	}
	edited, err := safetensors.Open(output)
	if err != nil {
		t.Fatal(err)
	}
	defer edited.Close()
	editedValue, err := edited.ReadFloat("layer.weight", 1)
	if err != nil || editedValue != -1.75 {
		t.Fatalf("patched checkpoint has value=%v err=%v, want -1.75", editedValue, err)
	}
}

func TestInspectRejectsUnsupportedAdaptersAndFormats(t *testing.T) {
	t.Chdir(t.TempDir())
	path := "model.gguf"
	if err := os.WriteFile(path, []byte("GGUF\x00\x00\x00\x00"), 0o600); err != nil {
		t.Fatal(err)
	}
	request := InspectRequest{Model: Model{
		ID: "gguf-model", Format: FormatSafeTensors, Adapter: AdapterSafeTensorsScalarDelta, Path: path,
	}}
	if _, err := Inspect(context.Background(), request); !errors.Is(err, ErrUnsupportedFormat) {
		t.Fatalf("Inspect(GGUF under SafeTensors selection) error = %v, want ErrUnsupportedFormat", err)
	}
	request.Model.Format = "unknown"
	if _, err := Inspect(context.Background(), request); !errors.Is(err, ErrUnsupportedFormat) {
		t.Fatalf("Inspect(unknown format) error = %v, want ErrUnsupportedFormat", err)
	}
	request.Model.Format = FormatGGUF
	request.Model.Adapter = AdapterSafeTensorsScalarDelta
	if _, err := Inspect(context.Background(), request); !errors.Is(err, ErrUnsupportedAdapter) {
		t.Fatalf("Inspect(GGUF with wrong adapter) error = %v, want ErrUnsupportedAdapter", err)
	}
	request.Model.Format = FormatSafeTensors
	request.Model.Adapter = "unsupported.format.adapter"
	if _, err := Inspect(context.Background(), request); !errors.Is(err, ErrUnsupportedAdapter) {
		t.Fatalf("Inspect(unknown adapter) error = %v, want ErrUnsupportedAdapter", err)
	}
}

func TestInspectAndPatchSyntheticShardedSafeTensors(t *testing.T) {
	t.Chdir(t.TempDir())
	indexPath := writeShardedFixture(t, "shards")
	model := testModel(indexPath)
	inspection, err := Inspect(context.Background(), InspectRequest{Model: model, IncludeSHA256: true, SampleElements: 2})
	if err != nil {
		t.Fatalf("Inspect(sharded SafeTensors) error = %v", err)
	}
	if len(inspection.Tensors) != 2 || len(inspection.Files) != 3 || inspection.SHA256 == "" || inspection.SizeBytes <= 0 {
		t.Fatalf("incomplete sharded provenance: %+v", inspection)
	}
	if inspection.Tensors[1].Name != "layer.weight" || len(inspection.Tensors[1].Sample) != 2 || inspection.Tensors[1].Sample[1] != -2 {
		t.Fatalf("unexpected sharded tensor metadata: %+v", inspection.Tensors)
	}
	originalSourceHash := inspection.SHA256
	expected := -2.0
	patched, err := Patch(context.Background(), PatchRequest{
		Model: model, OutputPath: "output/model.safetensors.index.json", SourceSHA256: originalSourceHash,
		Edits: []Edit{{Tensor: "layer.weight", Index: 1, Delta: 0.5, ExpectedValue: &expected}},
	})
	if err != nil {
		t.Fatalf("Patch(sharded SafeTensors) error = %v", err)
	}
	if patched.InputSHA256 != inspection.SHA256 || patched.OutputSHA256 == "" || len(patched.InputFiles) != 3 || len(patched.OutputFiles) != 3 {
		t.Fatalf("sharded patch provenance is incomplete: %+v", patched)
	}
	if patched.OutputPath != "output/model.safetensors.index.json" || len(patched.Edits) != 1 || patched.Edits[0].After != -1.5 {
		t.Fatalf("unexpected sharded patch result: %+v", patched)
	}
	var outputIndex struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	indexBytes, err := os.ReadFile(patched.OutputPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(indexBytes, &outputIndex); err != nil || len(outputIndex.WeightMap) != 2 {
		t.Fatalf("output index is invalid: %v, index=%s", err, indexBytes)
	}
	outputModel := testModel(patched.OutputPath)
	outputInspection, err := Inspect(context.Background(), InspectRequest{Model: outputModel, IncludeSHA256: true, SampleElements: 2})
	if err != nil {
		t.Fatalf("reopen sharded output: %v", err)
	}
	if outputInspection.SHA256 != patched.OutputSHA256 || outputInspection.Tensors[1].Sample[1] != -1.5 {
		t.Fatalf("output index does not resolve patched tensors: %+v", outputInspection)
	}
	if _, err := os.Stat("shards/model-00001-of-00002.safetensors"); err != nil {
		t.Fatalf("source shard was removed: %v", err)
	}
	if originalSourceHash == patched.OutputSHA256 {
		t.Fatal("sharded output digest unexpectedly equals input digest")
	}
}

func TestShardedPatchRejectsStaleBundleDigestWithoutPublishing(t *testing.T) {
	workspace := t.TempDir()
	t.Chdir(workspace)
	indexPath := writeShardedFixture(t, "shards")
	model := testModel(indexPath)
	inspection, err := Inspect(context.Background(), InspectRequest{Model: model, IncludeSHA256: true})
	if err != nil {
		t.Fatal(err)
	}
	shardPath := confinedTestPath(t, "shards/model-00002-of-00002.safetensors")
	contents, err := os.ReadFile(shardPath)
	if err != nil {
		t.Fatal(err)
	}
	contents[len(contents)-1] ^= 0x01
	writeConfinedTestFile(t, workspace, "shards/model-00002-of-00002.safetensors", contents)
	outputIndex := "out/model.safetensors.index.json"
	_, err = Patch(context.Background(), PatchRequest{
		Model: model, OutputPath: outputIndex, SourceSHA256: inspection.SHA256,
		Edits: []Edit{{Tensor: "layer.weight", Index: 0, Delta: 0.25}},
	})
	if err == nil || !strings.Contains(err.Error(), "source checkpoint SHA-256 changed") {
		t.Fatalf("Patch(stale sharded digest) error = %v, want checksum mismatch", err)
	}
	if _, err := os.Stat(outputIndex); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("stale patch published an output index: %v", err)
	}
	entries, err := os.ReadDir("out")
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("stale patch left temporary or final shards: %v", entries)
	}
}

func TestShardedSafeTensorsRejectsMissingAndEscapingReferences(t *testing.T) {
	t.Chdir(t.TempDir())
	if err := os.Mkdir("bundle", 0o755); err != nil {
		t.Fatal(err)
	}
	for name, shard := range map[string]string{
		"missing.safetensors.index.json": "absent.safetensors",
		"escape.safetensors.index.json":  "../outside.safetensors",
	} {
		index := `{"weight_map":{"layer.weight":"` + shard + `"}}`
		if err := os.WriteFile(filepath.Join("bundle", name), []byte(index), 0o600); err != nil {
			t.Fatal(err)
		}
		_, err := Inspect(context.Background(), InspectRequest{Model: testModel(filepath.Join("bundle", name))})
		if err == nil {
			t.Errorf("Inspect(%q) unexpectedly accepted invalid shard reference %q", name, shard)
		}
	}
}

func TestInspectAndPatchSyntheticGGUFPreservesQuantizedBlocks(t *testing.T) {
	t.Chdir(t.TempDir())
	source := "base.gguf"
	quantized := writeGGUFFixture(t, source, 0)
	model := Model{ID: "synthetic-gguf", Format: FormatGGUF, Adapter: AdapterGGUFScalarDelta, Path: source}
	inspection, err := Inspect(context.Background(), InspectRequest{Model: model, IncludeSHA256: true, SampleElements: 2})
	if err != nil {
		t.Fatalf("Inspect(GGUF) error = %v", err)
	}
	if len(inspection.Tensors) != 2 || inspection.Tensors[0].Name != "layer.weight" || len(inspection.Tensors[0].Sample) != 2 || inspection.Tensors[0].Sample[0] != 1.25 {
		t.Fatalf("unexpected GGUF inspection: %+v", inspection)
	}
	if inspection.Tensors[1].DType != "Q4_0" || inspection.Tensors[1].Bytes != 18 || inspection.Tensors[1].Sample != nil {
		t.Fatalf("quantized GGUF tensor was not kept opaque: %+v", inspection.Tensors[1])
	}
	patched, err := Patch(context.Background(), PatchRequest{
		Model: model, OutputPath: "edited.gguf", SourceSHA256: inspection.SHA256,
		Edits: []Edit{{Tensor: "layer.weight", Index: 1, Delta: 0.5}},
	})
	if err != nil {
		t.Fatalf("Patch(GGUF) error = %v", err)
	}
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), ".weights-stage-") {
			t.Fatalf("successful GGUF patch left a staging file %q", entry.Name())
		}
	}
	outputInspection, err := Inspect(context.Background(), InspectRequest{Model: Model{ID: model.ID, Format: FormatGGUF, Adapter: AdapterGGUFScalarDelta, Path: patched.OutputPath}, IncludeSHA256: true, SampleElements: 2})
	if err != nil {
		t.Fatalf("Inspect(GGUF output) error = %v", err)
	}
	if outputInspection.SHA256 != patched.OutputSHA256 || outputInspection.Tensors[0].Sample[1] != -1.5 {
		t.Fatalf("GGUF output value/digest mismatch: %+v", outputInspection)
	}
	if !bytes.Equal(quantized, ggufTensorBytes(t, patched.OutputPath, "quant.weight")) {
		t.Fatal("quantized GGUF tensor payload changed during a float edit")
	}
	if testFileDigest(t, source) != inspection.SHA256 {
		t.Fatal("GGUF source file changed during patch")
	}
}

func TestGGUFRejectsQuantizedEditsMalformedLayoutsAndSplitFiles(t *testing.T) {
	t.Chdir(t.TempDir())
	writeGGUFFixture(t, "quant.gguf", 0)
	model := Model{ID: "synthetic-gguf", Format: FormatGGUF, Adapter: AdapterGGUFScalarDelta, Path: "quant.gguf"}
	inspection, err := Inspect(context.Background(), InspectRequest{Model: model, IncludeSHA256: true})
	if err != nil {
		t.Fatal(err)
	}
	_, err = Patch(context.Background(), PatchRequest{
		Model: model, OutputPath: "quant-edited.gguf", SourceSHA256: inspection.SHA256,
		Edits: []Edit{{Tensor: "quant.weight", Index: 0, Delta: 1}},
	})
	if err == nil || !strings.Contains(err.Error(), "preserved but not editable") {
		t.Fatalf("Patch(quantized tensor) error = %v, want explicit non-editable error", err)
	}
	if _, err := os.Stat("quant-edited.gguf"); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("failed quantized edit left an output: %v", err)
	}

	writeGGUFFixture(t, "split.gguf", 2)
	splitModel := model
	splitModel.Path = "split.gguf"
	if _, err := Inspect(context.Background(), InspectRequest{Model: splitModel}); !errors.Is(err, ErrUnsupportedLayout) {
		t.Fatalf("Inspect(split GGUF) error = %v, want ErrUnsupportedLayout", err)
	}
	if err := os.WriteFile("truncated.gguf", []byte("GGUF"), 0o600); err != nil {
		t.Fatal(err)
	}
	malformed := model
	malformed.Path = "truncated.gguf"
	if _, err := Inspect(context.Background(), InspectRequest{Model: malformed}); err == nil {
		t.Fatal("Inspect(truncated GGUF) succeeded")
	}
}

func TestGGUFPatchRejectsStaleDigestWithoutOutput(t *testing.T) {
	workspace := t.TempDir()
	t.Chdir(workspace)
	writeGGUFFixture(t, "stale.gguf", 0)
	model := Model{ID: "synthetic-gguf", Format: FormatGGUF, Adapter: AdapterGGUFScalarDelta, Path: "stale.gguf"}
	inspection, err := Inspect(context.Background(), InspectRequest{Model: model, IncludeSHA256: true})
	if err != nil {
		t.Fatal(err)
	}
	modelPath := confinedTestPath(t, "stale.gguf")
	contents, err := os.ReadFile(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	contents[len(contents)-1] ^= 0x01
	writeConfinedTestFile(t, workspace, "stale.gguf", contents)
	output := "stale-edited.gguf"
	_, err = Patch(context.Background(), PatchRequest{
		Model: model, OutputPath: output, SourceSHA256: inspection.SHA256,
		Edits: []Edit{{Tensor: "layer.weight", Index: 0, Delta: 0.25}},
	})
	if err == nil || !strings.Contains(err.Error(), "source GGUF SHA-256 changed") {
		t.Fatalf("Patch(stale GGUF digest) error = %v, want checksum mismatch", err)
	}
	if _, err := os.Stat(output); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("stale GGUF patch left an output: %v", err)
	}
}

func TestPyTorchCheckpointFailsClosedWithoutAdvertisingSupport(t *testing.T) {
	model := Model{ID: "synthetic-pytorch", Format: FormatPyTorch, Path: "model.pt"}
	if _, err := Inspect(context.Background(), InspectRequest{Model: model}); !errors.Is(err, ErrUnsafeSerialization) {
		t.Fatalf("Inspect(PyTorch) error = %v, want ErrUnsafeSerialization", err)
	}
	for _, adapter := range Adapters() {
		if adapter.Format == FormatPyTorch {
			t.Fatalf("PyTorch reject-only path must not be advertised as supported: %+v", adapter)
		}
	}
}

func TestInspectRejectsOversizedSamples(t *testing.T) {
	t.Chdir(t.TempDir())
	path := "valid.safetensors"
	writeModelFixture(t, path)
	if _, err := Inspect(context.Background(), InspectRequest{Model: testModel(path), SampleElements: MaxInspectSamples + 1}); err == nil {
		t.Fatal("Inspect() accepted a sample limit above MaxInspectSamples")
	}
}

func TestInspectRejectsAbsoluteAndSymlinkEscapePaths(t *testing.T) {
	root := t.TempDir()
	outside := filepath.Join(t.TempDir(), "outside.safetensors")
	writeModelFixture(t, outside)
	t.Chdir(root)
	if err := os.Symlink(filepath.Dir(outside), "escape"); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	for _, path := range []string{outside, filepath.Join("escape", filepath.Base(outside))} {
		_, err := Inspect(context.Background(), InspectRequest{Model: testModel(path)})
		if err == nil {
			t.Errorf("Inspect(%q) succeeded outside the workspace root", path)
		}
	}
}

func TestPatchRequiresCurrentSourceHashAndLeavesNoOutput(t *testing.T) {
	t.Chdir(t.TempDir())
	source := "base.safetensors"
	output := "edited.safetensors"
	writeModelFixture(t, source)
	request := PatchRequest{
		Model: testModel(source), OutputPath: output,
		SourceSHA256: strings.Repeat("0", sha256.Size*2),
		Edits:        []Edit{{Tensor: "layer.bias", Index: 0, Delta: 0.5}},
	}
	_, err := Patch(context.Background(), request)
	if err == nil || !strings.Contains(err.Error(), "SHA-256 changed") {
		t.Fatalf("Patch() error = %v, want checksum mismatch", err)
	}
	if _, statErr := os.Stat(output); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("failed patch left an output: stat error %v", statErr)
	}
}

func TestPatchRejectsOutputOutsideWorkspace(t *testing.T) {
	workspace := t.TempDir()
	outsideDir := t.TempDir()
	t.Chdir(workspace)
	source := "base.safetensors"
	writeModelFixture(t, source)
	outside := filepath.Join(outsideDir, "edited.safetensors")
	for _, output := range []string{outside, filepath.Join("..", filepath.Base(outside))} {
		_, err := Patch(context.Background(), PatchRequest{
			Model: testModel(source), OutputPath: output, SourceSHA256: testFileDigest(t, source),
			Edits: []Edit{{Tensor: "layer.bias", Index: 0, Delta: 0.5}},
		})
		if err == nil {
			t.Errorf("Patch(output_path=%q) succeeded outside the workspace", output)
		}
	}
	if _, err := os.Stat(outside); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("outside output was created: stat error %v", err)
	}
	if err := os.Symlink(outside, "output-link.safetensors"); err == nil {
		_, err = Patch(context.Background(), PatchRequest{
			Model: testModel(source), OutputPath: "output-link.safetensors", SourceSHA256: testFileDigest(t, source),
			Edits: []Edit{{Tensor: "layer.bias", Index: 0, Delta: 0.5}},
		})
		if err == nil {
			t.Fatal("Patch accepted a symlink output path")
		}
	}
}

func TestPatchRejectsUnsupportedFloatOverflow(t *testing.T) {
	t.Chdir(t.TempDir())
	source := "base.safetensors"
	writeModelFixture(t, source)
	_, err := Patch(context.Background(), PatchRequest{
		Model: testModel(source), OutputPath: "overflow.safetensors",
		SourceSHA256: testFileDigest(t, source),
		Edits:        []Edit{{Tensor: "layer.weight", Index: 0, Delta: math.MaxFloat64}},
	})
	if err == nil || !strings.Contains(err.Error(), "overflows F32") {
		t.Fatalf("Patch() error = %v, want non-finite/overflow rejection", err)
	}
}

func TestPatchRejectsMislabeledSafeTensorsOutput(t *testing.T) {
	t.Chdir(t.TempDir())
	source := "base.safetensors"
	writeModelFixture(t, source)
	_, err := Patch(context.Background(), PatchRequest{
		Model: testModel(source), OutputPath: "edited.bin",
		SourceSHA256: testFileDigest(t, source),
		Edits:        []Edit{{Tensor: "layer.weight", Index: 0, Delta: 0.5}},
	})
	if err == nil || !strings.Contains(err.Error(), "must end in .safetensors") {
		t.Fatalf("Patch() error = %v, want a SafeTensors extension error", err)
	}
	if _, statErr := os.Stat("edited.bin"); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("mislabeled output was created: %v", statErr)
	}
}

func TestInspectAndPatchRejectCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Inspect(ctx, InspectRequest{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("Inspect(canceled context) error = %v, want context.Canceled", err)
	}
	if _, err := Patch(ctx, PatchRequest{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("Patch(canceled context) error = %v, want context.Canceled", err)
	}
}

func confinedTestPath(t *testing.T, relative string) string {
	t.Helper()
	if filepath.IsAbs(relative) {
		t.Fatalf("test path must be relative: %q", relative)
	}
	root, err := filepath.Abs(".")
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, filepath.Clean(relative))
	rel, err := filepath.Rel(root, path)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		t.Fatalf("test path escapes temporary working directory: %q", relative)
	}
	return path
}

func testModel(path string) Model {
	return Model{ID: "synthetic-test-model", Format: FormatSafeTensors, Adapter: AdapterSafeTensorsScalarDelta, Path: path}
}

func writeModelFixture(t *testing.T, path string) {
	t.Helper()
	weight := make([]byte, 8)
	binary.LittleEndian.PutUint32(weight[0:4], math.Float32bits(1.5))
	binary.LittleEndian.PutUint32(weight[4:8], math.Float32bits(-2))
	bias := make([]byte, 4)
	binary.LittleEndian.PutUint32(bias, math.Float32bits(0.25))
	if err := safetensors.Write(path, map[string]safetensors.TensorData{
		"layer.weight": {DType: "F32", Shape: []uint64{2}, Data: weight},
		"layer.bias":   {DType: "F32", Shape: []uint64{1}, Data: bias},
	}, map[string]string{"architecture": "synthetic-only"}, false); err != nil {
		t.Fatal(err)
	}
}

func writeShardedFixture(t *testing.T, dir string) string {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	weight := make([]byte, 8)
	binary.LittleEndian.PutUint32(weight[0:4], math.Float32bits(1.5))
	binary.LittleEndian.PutUint32(weight[4:8], math.Float32bits(-2))
	bias := make([]byte, 4)
	binary.LittleEndian.PutUint32(bias, math.Float32bits(0.25))
	shard1 := filepath.Join(dir, "model-00001-of-00002.safetensors")
	shard2 := filepath.Join(dir, "model-00002-of-00002.safetensors")
	if err := safetensors.Write(shard1, map[string]safetensors.TensorData{
		"layer.weight": {DType: "F32", Shape: []uint64{2}, Data: weight},
	}, nil, false); err != nil {
		t.Fatal(err)
	}
	if err := safetensors.Write(shard2, map[string]safetensors.TensorData{
		"layer.bias": {DType: "F32", Shape: []uint64{1}, Data: bias},
	}, nil, false); err != nil {
		t.Fatal(err)
	}
	index := map[string]any{
		"metadata": map[string]any{"total_size": 12},
		"weight_map": map[string]string{
			"layer.weight": filepath.Base(shard1),
			"layer.bias":   filepath.Base(shard2),
		},
	}
	indexBytes, err := json.Marshal(index)
	if err != nil {
		t.Fatal(err)
	}
	indexPath := filepath.Join(dir, "model.safetensors.index.json")
	if err := os.WriteFile(indexPath, indexBytes, 0o600); err != nil {
		t.Fatal(err)
	}
	return indexPath
}

func writeGGUFFixture(t *testing.T, path string, splitCount uint32) []byte {
	t.Helper()
	var header bytes.Buffer
	header.WriteString("GGUF")
	_ = binary.Write(&header, binary.LittleEndian, uint32(3))
	_ = binary.Write(&header, binary.LittleEndian, uint64(2))
	metadataCount := uint64(0)
	if splitCount > 0 {
		metadataCount = 2
	}
	_ = binary.Write(&header, binary.LittleEndian, metadataCount)
	if splitCount > 0 {
		writeGGUFString(&header, "split.no")
		_ = binary.Write(&header, binary.LittleEndian, uint32(4))
		_ = binary.Write(&header, binary.LittleEndian, uint32(0))
		writeGGUFString(&header, "split.count")
		_ = binary.Write(&header, binary.LittleEndian, uint32(4))
		_ = binary.Write(&header, binary.LittleEndian, splitCount)
	}
	writeGGUFTensorInfo(&header, "layer.weight", 1, 2, 0, 0)
	writeGGUFTensorInfo(&header, "quant.weight", 1, 32, 2, 32)
	dataStart := (header.Len() + 31) &^ 31
	whole := make([]byte, dataStart+32+18)
	copy(whole, header.Bytes())
	binary.LittleEndian.PutUint32(whole[dataStart:dataStart+4], math.Float32bits(1.25))
	binary.LittleEndian.PutUint32(whole[dataStart+4:dataStart+8], math.Float32bits(-2))
	for i := range 18 {
		whole[dataStart+32+i] = byte(0x40 + i)
	}
	if err := os.WriteFile(path, whole, 0o600); err != nil {
		t.Fatal(err)
	}
	return append([]byte(nil), whole[dataStart+32:dataStart+32+18]...)
}

func writeGGUFString(buffer *bytes.Buffer, value string) {
	_ = binary.Write(buffer, binary.LittleEndian, uint64(len(value)))
	buffer.WriteString(value)
}

func writeGGUFTensorInfo(buffer *bytes.Buffer, name string, dimensions uint32, firstDimension uint64, typeID uint32, offset uint64) {
	writeGGUFString(buffer, name)
	_ = binary.Write(buffer, binary.LittleEndian, dimensions)
	_ = binary.Write(buffer, binary.LittleEndian, firstDimension)
	_ = binary.Write(buffer, binary.LittleEndian, typeID)
	_ = binary.Write(buffer, binary.LittleEndian, offset)
}

func ggufTensorBytes(t *testing.T, path, tensor string) []byte {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := gguf.OpenFile(context.Background(), file, path)
	if err != nil {
		t.Fatal(err)
	}
	defer parsed.Close()
	info, exists := parsed.Tensors[tensor]
	if !exists {
		t.Fatalf("GGUF tensor %q not found", tensor)
	}
	all, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.DataPos < 0 {
		t.Fatal("GGUF tensor data position is negative")
	}
	byteLength, err := checkedTestInt64(info.Bytes)
	if err != nil {
		t.Fatal(err)
	}
	if info.DataPos > math.MaxInt64-byteLength {
		t.Fatal("GGUF tensor byte range overflows int64")
	}
	end := info.DataPos + byteLength
	if end > int64(len(all)) {
		t.Fatalf("GGUF tensor byte range ends at %d beyond file length %d", end, len(all))
	}
	return append([]byte(nil), all[info.DataPos:end]...)
}

func checkedTestInt64(value uint64) (int64, error) {
	if value > math.MaxInt64 {
		return 0, errors.New("value exceeds int64")
	}
	return int64(value), nil
}

func writeConfinedTestFile(t *testing.T, rootPath, relativePath string, contents []byte) {
	t.Helper()
	root, err := os.OpenRoot(rootPath)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if closeErr := root.Close(); closeErr != nil {
			t.Errorf("close test root: %v", closeErr)
		}
	}()
	file, err := root.OpenFile(relativePath, os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if closeErr := file.Close(); closeErr != nil {
			t.Errorf("close fixture file: %v", closeErr)
		}
	}()
	if written, err := file.Write(contents); err != nil {
		t.Fatalf("write fixture file: %v", err)
	} else if written != len(contents) {
		t.Fatalf("write fixture file: wrote %d of %d bytes", written, len(contents))
	}
}

func testFileDigest(t *testing.T, path string) string {
	t.Helper()
	contents, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(contents)
	return hex.EncodeToString(digest[:])
}
