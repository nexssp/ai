package safetensors

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPatchSupportsCommonFloatTypesAndPreservesOtherBytes(t *testing.T) {
	dir := t.TempDir()
	source := filepath.Join(dir, "base.safetensors")
	output := filepath.Join(dir, "patched.safetensors")
	writeFixture(t, source)

	want := []struct {
		name                 string
		before, delta, after float64
	}{
		{"f32", 1.5, 0.25, 1.75},
		{"bf16", 2, 0.5, 2.5},
		{"f16", -2, 0.5, -1.5},
		{"f64", 4.25, 0.5, 4.75},
	}
	edits := make([]Edit, 0, len(want))
	for _, item := range want {
		before := item.before
		edits = append(edits, Edit{Tensor: item.name, Index: 0, Delta: item.delta, ExpectedValue: &before})
	}
	result, err := Patch(context.Background(), source, output, edits, PatchOptions{ExpectedSHA256: fileSHA256(t, source)})
	if err != nil {
		t.Fatalf("Patch() error = %v", err)
	}
	if len(result.Edits) != len(want) {
		t.Fatalf("Patch() returned %d edits, want %d", len(result.Edits), len(want))
	}
	if result.InputSHA256 != fileSHA256(t, source) || result.OutputSHA256 != fileSHA256(t, output) {
		t.Fatalf("patch digests do not match files: input=%s output=%s", result.InputSHA256, result.OutputSHA256)
	}
	for i, item := range want {
		if math.Abs(result.Edits[i].Before-item.before) > 1e-9 || math.Abs(result.Edits[i].After-item.after) > 1e-9 {
			t.Errorf("edit %s = before %.9g after %.9g, want %.9g / %.9g", item.name, result.Edits[i].Before, result.Edits[i].After, item.before, item.after)
		}
	}

	original, err := Open(source)
	if err != nil {
		t.Fatal(err)
	}
	defer original.Close()
	stillOriginal, err := original.ReadFloat("f32", 0)
	if err != nil || stillOriginal != 1.5 {
		t.Fatalf("source changed: value=%v err=%v", stillOriginal, err)
	}
	patched, err := Open(output)
	if err != nil {
		t.Fatal(err)
	}
	defer patched.Close()
	for _, item := range want {
		got, err := patched.ReadFloat(item.name, 0)
		if err != nil || math.Abs(got-item.after) > 1e-9 {
			t.Errorf("patched %s = %v, err=%v; want %v", item.name, got, err, item.after)
		}
	}
	if patched.Metadata["note"] != "keep me" {
		t.Errorf("metadata not preserved: %#v", patched.Metadata)
	}
	origBytes, err := os.ReadFile(source)
	if err != nil {
		t.Fatal(err)
	}
	patchedBytes, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	origOpaque := tensorBytes(t, source, "opaque")
	patchedOpaque := tensorBytes(t, output, "opaque")
	if !bytes.Equal(origOpaque, patchedOpaque) {
		t.Errorf("non-float tensor bytes changed: %x != %x", origOpaque, patchedOpaque)
	}
	if len(origBytes) != len(patchedBytes) {
		t.Errorf("file size changed from %d to %d", len(origBytes), len(patchedBytes))
	}
}

func TestPatchRejectsStaleValueWithoutCreatingOutput(t *testing.T) {
	dir := t.TempDir()
	source := filepath.Join(dir, "base.safetensors")
	output := filepath.Join(dir, "never-created.safetensors")
	writeFixture(t, source)
	wrong := 99.0
	_, err := Patch(context.Background(), source, output, []Edit{{Tensor: "f32", Index: 0, Delta: 1, ExpectedValue: &wrong}}, PatchOptions{ExpectedSHA256: fileSHA256(t, source)})
	if err == nil || !strings.Contains(err.Error(), "stale weight") {
		t.Fatalf("Patch() error = %v, want stale weight error", err)
	}
	if _, statErr := os.Stat(output); !os.IsNotExist(statErr) {
		t.Fatalf("failed patch left output behind: stat error %v", statErr)
	}
}

func TestPatchRejectsWrongSourceDigestWithoutCreatingOutput(t *testing.T) {
	dir := t.TempDir()
	source := filepath.Join(dir, "base.safetensors")
	output := filepath.Join(dir, "wrong-digest.safetensors")
	writeFixture(t, source)
	_, err := Patch(context.Background(), source, output, []Edit{{Tensor: "f32", Index: 0, Delta: 1}}, PatchOptions{
		ExpectedSHA256: strings.Repeat("0", sha256.Size*2),
	})
	if err == nil || !strings.Contains(err.Error(), "SHA-256 changed") {
		t.Fatalf("Patch() error = %v, want source digest mismatch", err)
	}
	if _, statErr := os.Stat(output); !os.IsNotExist(statErr) {
		t.Fatalf("digest mismatch left output behind: stat error %v", statErr)
	}
}

func TestPatchNeverOverwritesInputAndProtectsExistingOutput(t *testing.T) {
	dir := t.TempDir()
	source := filepath.Join(dir, "base.safetensors")
	writeFixture(t, source)
	options := PatchOptions{ExpectedSHA256: fileSHA256(t, source)}
	_, err := Patch(context.Background(), source, source, []Edit{{Tensor: "f32", Index: 0, Delta: 1}}, options)
	if err == nil || !strings.Contains(err.Error(), "must differ") {
		t.Fatalf("same-path patch error = %v", err)
	}
	output := filepath.Join(dir, "existing.safetensors")
	if err := os.WriteFile(output, []byte("keep"), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err = Patch(context.Background(), source, output, []Edit{{Tensor: "f32", Index: 0, Delta: 1}}, options)
	if err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("existing output patch error = %v", err)
	}
	contents, err := os.ReadFile(output)
	if err != nil || string(contents) != "keep" {
		t.Fatalf("existing output changed: %q, err=%v", contents, err)
	}
}

func TestOpenRejectsDuplicateKeysAndInvalidCoverage(t *testing.T) {
	dir := t.TempDir()
	cases := []struct {
		name, header string
		body         []byte
		want         string
	}{
		{"duplicate top-level", `{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, make([]byte, 4), "duplicate"},
		{"hole", `{"x":{"dtype":"F32","shape":[1],"data_offsets":[1,5]}}`, make([]byte, 5), "hole"},
		{"size mismatch", `{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}`, make([]byte, 4), "require 8"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(dir, tc.name+".safetensors")
			writeRawFile(t, path, tc.header, tc.body)
			_, err := Open(path)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Open() error = %v, want substring %q", err, tc.want)
			}
		})
	}
}

func TestInspectReturnsMetadataAndLimitedPrefix(t *testing.T) {
	path := filepath.Join(t.TempDir(), "base.safetensors")
	writeFixture(t, path)
	items, err := Inspect(path, "f32", 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 1 || items[0].Name != "f32" || len(items[0].Sample) != 1 || items[0].Sample[0] != 1.5 {
		t.Fatalf("Inspect() = %#v", items)
	}
}

func TestFloat16RoundTrips(t *testing.T) {
	for _, value := range []float32{0, 1, -2, 0.5, 65504, 0.00006103515625} {
		half := float32ToHalf(value)
		got := halfToFloat32(half)
		if math.Abs(float64(got-value)) > math.Max(1e-6, math.Abs(float64(value))*1e-3) {
			t.Errorf("half conversion %g -> %g", value, got)
		}
	}
}

func writeFixture(t *testing.T, path string) {
	t.Helper()
	f32 := make([]byte, 8)
	binary.LittleEndian.PutUint32(f32[0:4], math.Float32bits(1.5))
	binary.LittleEndian.PutUint32(f32[4:8], math.Float32bits(-1))
	f64 := make([]byte, 8)
	binary.LittleEndian.PutUint64(f64, math.Float64bits(4.25))
	if err := Write(path, map[string]TensorData{
		"f32":    {DType: "F32", Shape: []uint64{2}, Data: f32},
		"bf16":   {DType: "BF16", Shape: []uint64{1}, Data: []byte{0x00, 0x40}},
		"f16":    {DType: "F16", Shape: []uint64{1}, Data: []byte{0x00, 0xc0}},
		"f64":    {DType: "F64", Shape: []uint64{1}, Data: f64},
		"opaque": {DType: "I32", Shape: []uint64{1}, Data: []byte{1, 2, 3, 4}},
	}, map[string]string{"note": "keep me"}, false); err != nil {
		t.Fatal(err)
	}
}

func tensorBytes(t *testing.T, path, name string) []byte {
	t.Helper()
	file, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	info := file.Tensors[name]
	data := make([]byte, info.DataOffsets[1]-info.DataOffsets[0])
	offset, err := checkedFileOffset(file.DataStart, info.DataOffsets[0])
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.file.ReadAt(data, offset); err != nil {
		t.Fatal(err)
	}
	return data
}

func writeRawFile(t *testing.T, path, header string, body []byte) {
	t.Helper()
	padding := header
	for len(padding)%8 != 0 {
		padding += " "
	}
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	var prefix [8]byte
	binary.LittleEndian.PutUint64(prefix[:], uint64(len(padding)))
	if _, err := file.Write(prefix[:]); err != nil {
		t.Fatal(err)
	}
	if _, err := file.WriteString(padding); err != nil {
		t.Fatal(err)
	}
	if _, err := file.Write(body); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
}

func fileSHA256(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:])
}
