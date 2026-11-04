# Checkpoint weight editing

`github.com/nexssp/ai/weights` provides a bounded local file-editing primitive. It inspects tensor metadata and can add caller-supplied scalar deltas to a checksum-verified copy. It does **not** load a model, infer architecture, run inference, generate edits, or evaluate behavior. A tensor-level edit is not evidence of compatibility or usefulness.

Callers choose the exact `model_id` label, `format`, `adapter`, and workspace-relative `path`. Model IDs are opaque labels, not registry lookups or proof of compatibility. Paths are opened through Nexss `kernel/xfs`, confined to the process working directory, and cannot escape through absolute paths or symlinks. Run from a workspace where the intended files are available.

## What works

| Format | Selection | Current support |
| --- | --- | --- |
| Single SafeTensors | `format: "safetensors"`, `adapter: "safetensors.scalar-delta.v1"`, `path: "model.safetensors"` | Bounded metadata inspection and optional samples. Scalar edits for F16, BF16, F32, and F64; supported integer, boolean, and 8-bit float tensors are preserved. |
| Indexed sharded SafeTensors | Same format and adapter; `path` is the standard `.safetensors.index.json` | Validates the index `weight_map`, every referenced sibling shard, and exact tensor-to-shard membership. Hashes the index and all shard files. Patching writes new shards and atomically publishes a new index last. |
| GGUF | `format: "gguf"`, `adapter: "gguf.float-scalar-delta.v1"`, `path: "model.gguf"` | Little-endian GGUF v2/v3, bounded header/layout validation, and scalar edits for F16, BF16, F32, and F64. Recognized integer and quantized block encodings are size-checked and preserved byte-for-byte, never dequantized or edited. GGUF split bundles are rejected. |
| PyTorch `.pt` / `.pth` / `.bin` | `format: "pytorch"` | **Not supported.** Requests fail with `ErrUnsafeSerialization`; this format is not advertised as an implemented adapter. |

SafeTensors is the preferred interchange format for tensor-only state dictionaries. A sharded index uses `weight_map` entries that map tensor names to shard filenames; the index and shards must be siblings within the workspace. For format details, see the [SafeTensors specification](https://github.com/safetensors/safetensors) and Hugging Face's [SafeTensors metadata-parsing guide](https://huggingface.co/docs/safetensors/en/metadata_parsing).

The GGUF adapter validates the declared type, dimensions, storage size, alignment, and non-overlapping byte ranges before exposing tensor metadata. Quantized data is block-encoded; it is not a flat float array. See the [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md). Unknown future GGML type IDs and unsupported storage layouts fail closed. The adapter is an editor/inspector, not a GGUF model loader.

## Why PyTorch checkpoints are not parsed

Typical `torch.save` checkpoints contain a pickle object graph (`data.pkl`) that describes tensor names, shapes, strides, storage offsets, and storage references, alongside raw storage files. Reading the ZIP storage entries alone cannot recover correct tensor metadata. PyTorch's `weights_only=True` still uses a restricted unpickler; it is not a non-pickle format, and PyTorch documents remaining resource-exhaustion and other risks. See the official [PyTorch serialization notes](https://docs.pytorch.org/docs/2.15/notes/serialization.html).

This package does not call Python, `torch.load`, or a general pickle VM. A future safe PyTorch subset would need a bounded, non-executing pickle interpreter with a strict opcode/global allowlist for plain tensor state dictionaries, validated ZIP/storage mapping, limits on memory and nesting, and adversarial synthetic fixtures. It must reject unknown object graphs and constructors rather than expanding into arbitrary pickle support. Until then, convert a tensor-only state dictionary to SafeTensors using a trusted producer.

## Inspection and provenance

`Inspect` returns deterministic tensor names, dtype, shape, element count, and stored byte size. Samples are optional and capped at 32 leading elements per editable tensor; quantized GGUF tensors remain opaque and have no float sample. `include_sha256` (Go `IncludeSHA256`) performs a complete read of each checkpoint member. For a single file, `SHA256` is the raw file digest. For a sharded bundle, it is a deterministic aggregate over sorted relative paths, file sizes, and each raw file digest; `Files` reports the per-member values. This binds a later patch to the inspected index and all shard bytes.

`Patch` requires the exact `source_sha256` reported by inspection and writes a distinct output. Set `include_sha256: true` during inspection to obtain that digest. It supports at most 16,384 edits per call, checks optional `expected_value` guards, rejects duplicate/out-of-range edits and non-finite/overflowing values, and reports both input and output digests. For sharded SafeTensors, set `output_path` to a new `.safetensors.index.json`; new shard files are written in that directory and the index is published last. If an operation is interrupted after shard creation but before index publication, unreferenced output shard files may remain; the source bundle remains untouched. Overwriting an existing sharded index intentionally leaves its prior nonce-named shard files in place so readers that already loaded the old index are not broken; prefer versioned output indexes and remove obsolete shards only after confirming they are no longer referenced. Existing output is protected unless `overwrite: true` is explicitly set. Non-overwrite publication uses an atomic same-directory hard link and fails if another process creates the output first, so the workspace filesystem must support hard links. Use one writer per output path; concurrent overwrite writers are not coordinated.

## Go API

Single-file SafeTensors:

```go
model := weights.Model{
    ID:      "local-experiment-a",
    Format:  weights.FormatSafeTensors,
    Adapter: weights.AdapterSafeTensorsScalarDelta,
    Path:    "models/base.safetensors",
}

inspection, err := weights.Inspect(ctx, weights.InspectRequest{
    Model: model, IncludeSHA256: true, SampleElements: 4,
})
if err != nil {
    return err
}

result, err := weights.Patch(ctx, weights.PatchRequest{
    Model:        model,
    OutputPath:   "models/experiment-a.safetensors",
    SourceSHA256: inspection.SHA256,
    Edits: []weights.Edit{{
        Tensor: "layer.weight", Index: 12, Delta: 0.0005,
    }},
})
if err != nil {
    return err
}
_ = result.OutputSHA256 // record it with independent compatibility checks/evaluation
```

For GGUF, select `weights.FormatGGUF` and `weights.AdapterGGUFScalarDelta`, then use `.gguf` input and output paths. For indexed SafeTensors, use the same SafeTensors format and adapter but set the input path to `models/base.safetensors.index.json`; patch to a different index path such as `models/experiment-a.safetensors.index.json`. Inspect output again before handing it to a model runtime.

`weights.Adapters()` returns defensive copies of the implemented adapter capabilities. Neither adapter establishes that a specific architecture/runtime will accept the result.

## NFlow

The embedded `nflows/lib/weights.nflow` module exports `weights.pipeline.list_adapters`, `weights.pipeline.inspect`, and `weights.pipeline.patch`:

```nflow
@include ai:nflows/lib/weights.nflow AS weights

{
  model: {
    model_id: "local-experiment-a",
    format: "safetensors",
    adapter: "safetensors.scalar-delta.v1",
    path: "models/base.safetensors.index.json"
  },
  include_sha256: true,
  sample_elements: 4
} -> weights.pipeline.inspect
```

A patch request uses the same model selector, a distinct `output_path`, the inspection's `sha256` as `source_sha256`, and an `edits` array of `{tensor, index, delta}` entries. `expected_value` is an optional extra stale-value guard. The separate inspection and patch calls let the caller review the exact input digest before writing.

## Optional local llama.cpp evaluation

The weights package itself never starts an inference process. For a local smoke comparison, the AI NFlow bundle exports `nflows/lib/llama_cpp.nflow`, which wraps Flow's standard `external.exec` action and invokes an already-installed `llama-cli` with an argv array. The paired example [`nflows/examples/weights/llama_cpp_compare.nflow`](../nflows/examples/weights/llama_cpp_compare.nflow) runs one prompt against baseline and candidate files and returns raw output for review. No model or executable is downloaded by this project; provide paths to files you manage locally.

`external.exec` is arbitrary process execution, not a sandbox. Run only trusted flows in a trusted local runtime. The wrapper avoids shell parsing and validates basic generation bounds, but it does not constrain which executable the caller selects, cap output, measure model quality, or manage llama.cpp resources. Review outputs over a predeclared validation set, then use held-out checks before claiming improvement. Quantized GGUF checkpoints can be used for inference if llama.cpp supports them, but the current weight editor cannot apply scalar deltas to quantized tensors.

## Scope and testing

The weights-package tests use only small synthetic SafeTensors shards and GGUF bytes; they do not download model weights or run model inference. The NFlow integration tests use a local fake executable to verify argv handling without llama.cpp or model files. Checks cover format parsing, sharded index integrity, source/output digests, path confinement, quantized-byte preservation, failure cleanup, and the NFlow adapter contract. These are serialization and invocation checks, not model-quality or held-out behavior evaluation. The package is a constrained format primitive, not a general Weight-Space Programming system.
