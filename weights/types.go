// Package weights provides bounded, format-aware checkpoint inspection and editing.
//
// The package never loads a model, infers its architecture, runs inference, or
// evaluates behavior. Callers select an explicit format and adapter. Edits are
// low-level tensor-file changes and require independent compatibility checks and
// held-out evaluation before being treated as useful.
package weights

import "github.com/nexssp/ai/weights/safetensors"

const (
	// FormatSafeTensors identifies single-file and indexed sharded SafeTensors checkpoints.
	FormatSafeTensors = "safetensors"
	// FormatGGUF identifies supported little-endian GGUF v2/v3 checkpoints.
	FormatGGUF = "gguf"
	// FormatPyTorch identifies PyTorch .pt/.pth/.bin checkpoint requests. This
	// format is currently rejected because no safe tensor-only reader is included.
	FormatPyTorch = "pytorch"

	// AdapterSafeTensorsScalarDelta edits scalar float values in single or indexed
	// sharded SafeTensors files.
	AdapterSafeTensorsScalarDelta = "safetensors.scalar-delta.v1"
	// AdapterGGUFScalarDelta edits scalar values only in unquantized GGUF float
	// tensors. Quantized tensor payloads are validated and preserved byte-for-byte.
	AdapterGGUFScalarDelta = "gguf.float-scalar-delta.v1"

	// MaxInspectSamples bounds optional tensor values returned by Inspect.
	MaxInspectSamples = 32
	// MaxPatchEdits bounds one atomic patch operation.
	MaxPatchEdits = 16_384
)

// AdapterInfo describes a concrete supported file-editing adapter. Scope is
// intentionally explicit so callers do not mistake a byte-format adapter for a
// model-family loader or inference runtime.
type AdapterInfo struct {
	ID                   string   `json:"id"`
	Format               string   `json:"format"`
	Scope                string   `json:"scope"`
	EditableDTypes       []string `json:"editable_dtypes"`
	PreservedDTypes      []string `json:"preserved_dtypes"`
	SupportsShardedFiles bool     `json:"supports_sharded_files"`
	SupportsInference    bool     `json:"supports_inference"`
}

// Model selects a local checkpoint and the exact adapter used to interpret it.
// ID is a caller-provided label; this package does not authenticate model IDs or
// claim architecture-level compatibility. Path is relative to the process
// working directory and is opened through a Nexss xfs confined root. For
// SafeTensors it may name a .safetensors file or the standard .safetensors.index.json.
// GGUF currently accepts one .gguf file. PyTorch pickle-based checkpoints fail closed.
type Model struct {
	ID      string `json:"model_id"`
	Format  string `json:"format"`
	Adapter string `json:"adapter"`
	Path    string `json:"path"`
}

// InspectRequest asks for tensor metadata and, optionally, a small sample and a
// source digest. A nonzero SampleElements value is capped at MaxInspectSamples.
type InspectRequest struct {
	Model          Model  `json:"model"`
	IncludeSHA256  bool   `json:"include_sha256,omitempty"`
	SampleElements uint64 `json:"sample_elements,omitempty"`
}

// TensorInfo describes one tensor. Sample contains values only when explicitly
// requested and only for supported floating-point dtypes.
type TensorInfo struct {
	Name     string    `json:"name"`
	DType    string    `json:"dtype"`
	Shape    []uint64  `json:"shape"`
	Elements uint64    `json:"elements"`
	Bytes    uint64    `json:"bytes"`
	Sample   []float64 `json:"sample,omitempty"`
}

// FileDigest reports the exact path, size, and SHA-256 for one checkpoint member.
type FileDigest struct {
	Path      string `json:"path"`
	SizeBytes int64  `json:"size_bytes"`
	SHA256    string `json:"sha256"`
}

// Inspection reports the selected checkpoint's format-level structure. It is
// not a claim that the checkpoint can be loaded by any particular model runtime.
// For indexed sharded SafeTensors, SHA256 is a deterministic digest over the
// sorted member paths and their raw file digests; Files lists each member.
type Inspection struct {
	Model     Model        `json:"model"`
	SizeBytes int64        `json:"size_bytes"`
	SHA256    string       `json:"sha256,omitempty"`
	Files     []FileDigest `json:"files,omitempty"`
	Tensors   []TensorInfo `json:"tensors"`
}

// Edit adds Delta to one flat row-major tensor element. ExpectedValue is an
// optional additional stale-value guard; SourceSHA256 on PatchRequest binds the
// entire edit to the exact inspected checkpoint file or bundle.
type Edit = safetensors.Edit

// AppliedEdit records the rounded value stored in the output checkpoint.
type AppliedEdit = safetensors.AppliedEdit

// PatchRequest applies a bounded patch to a copy of the selected checkpoint.
// OutputPath is workspace-relative; paths outside the working directory and
// symlink escapes are rejected. Source files are never overwritten.
type PatchRequest struct {
	Model        Model  `json:"model"`
	OutputPath   string `json:"output_path"`
	SourceSHA256 string `json:"source_sha256"`
	Edits        []Edit `json:"edits"`
	Overwrite    bool   `json:"overwrite,omitempty"`
}

// PatchResult is an integrity and provenance report for the committed output.
// Bundle hashes are derived deterministically from the sorted member paths and
// per-file SHA-256 digests.
type PatchResult struct {
	Model        Model         `json:"model"`
	InputPath    string        `json:"input_path"`
	OutputPath   string        `json:"output_path"`
	InputSHA256  string        `json:"input_sha256"`
	OutputSHA256 string        `json:"output_sha256"`
	InputFiles   []FileDigest  `json:"input_files,omitempty"`
	OutputFiles  []FileDigest  `json:"output_files,omitempty"`
	Edits        []AppliedEdit `json:"edits"`
}
