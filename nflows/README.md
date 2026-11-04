# ai as a reusable NFlow package

`nexssflow` keeps provider protocols, HTTP requests, credentials, timeouts, and normalized errors in Go, while NFlow defines orchestration, routing graphs, and reusable workflow logic. This checkout uses Flow `v0.67.1`; the `nflow` runtime/CLI is installed separately from the Go module.

## Load the package

From a workflow in this checkout:

```nflow
@require ../ as ai {
    timeout: "30s"
}
```

Set `LLM_PROVIDER` to any provider supported by `engine/llm`, and provide `LLM_MODEL`; no workflow picks a provider or model silently. `LLM_BASE_URL` and `LLM_API_KEY` are optional overrides. `openai-compatible` is available for custom Chat Completions endpoints. `ai.decide` additionally requires a driver with the structured/native or supported JSON-mode capability it needs. `api_key_env` and `decision_api_key_env` are environment-variable names, never secret values. Accepted options are `provider`, `model`, `api_key_env`, `timeout`, `base_url`, `decision_provider`, `decision_model`, `decision_api_key_env`, and `decision_base_url`.

## Typed actions

The library exports `ai.prompt`, `ai.team`, `ai.decide`, `ai.cost.estimate`, and the bounded `ai.weights.adapters`, `ai.weights.inspect`, and `ai.weights.patch` actions. Each action is backed by a Go request struct and a typed result. `ai.prompt` returns content, model/provider, token counts, usage, and explicitly carried IDs/context; it does not merge arbitrary Flow state fields. `ai.team` executes two or more declared role turns sequentially and returns typed role outputs and per-turn usage.

`ai.decide` requires a positive `max_output_tokens` cap. Anthropic's official Structured Outputs API (`decision_provider: "anthropic"`) receives the cap in `max_tokens` and a typed JSON Schema in `output_config.format`. OpenAI, Gemini, xAI, DeepSeek, and Inception chat (`decision_provider: "inception-chat"`) use locally validated JSON-mode adapters; the cap is sent using each provider's documented field, and reported output usage above the cap is rejected. The documented Inception native `/v1/decisions` and TypeSafe `/v1/systemone` request schemas expose no provider-enforced output cap, so those native adapters fail closed before HTTP dispatch. JSON mode is not claimed to be equivalent to a native decision endpoint or schema-constrained output. Unverified provider names are rejected rather than silently adapted.

The decision action validates the candidate allowlist and returns a typed `DecisionResponse` with ordered `[]DecisionAnswer`; NFlow owns the routing graph and branch behavior. See [`decision_route.nflow`](decision_route.nflow) for a working `match` example. NFlow `@schema` field names are declared in snake_case and used directly without `json` tags. The workflow uses `ai.nflow.view` only after typed action outputs, then branches on `decision_result.answers[0].choice`. `ai.cost.estimate` uses `max_output_tokens` as the nested decision's completion-cost basis, but it does not bill, reserve, or authorize selector spend. The application host guard must approve and reserve cost before provider dispatch.

## Vector retrieval and grounded answers

The bundle also exposes `ai.retrieval.index`, `ai.retrieval.search`, `ai.retrieval.delete`, and `ai.rag.answer`. Each has a concrete Go request/response type; `ai.rag.answer` returns the exact source chunks supplied to the model. Retrieved excerpts are marked as untrusted reference data, and the answer action does not claim to verify model-generated citations. All index/search/delete operations require a namespace, which callers should treat as an authorization and data-isolation boundary.

Hosts inject a configured `retrieval.Service` when building the bundle. The service owns AI policy (chunking, embeddings, search, and prompt assembly); its `VectorIndex` is the adapter boundary, while generic persistence belongs in `nexssp/storage`.

```go
service, err := retrieval.NewService(retrieval.Config{}, embedder, vectorIndex)
if err != nil {
    return err
}
bundle, err := nexssflow.NewBundleWithServices(globalOptions, providerBuilder, service)
if err != nil {
    return err
}
```

[`retrieval_rag.nflow`](retrieval_rag.nflow) contains strict, snake_case `@schema` contracts and reusable pipelines for indexing, deletion, search, and answering. Its root pipeline runs `answer`; the integration test executes each pipeline through Flow `v0.67.1` using an in-memory index and fake embedder/provider. A host must inject the retrieval service before running the example. `MemoryIndex` is only for tests and local demonstrations; production persistence should use a host-selected adapter.

## Checkpoint weights (bounded adapter)

The bundle embeds [`lib/weights.nflow`](lib/weights.nflow), which provides `weights.pipeline.list_adapters`, `weights.pipeline.inspect`, and `weights.pipeline.patch` after the module is imported. Implemented adapters are `safetensors.scalar-delta.v1` for single or standard indexed-sharded SafeTensors and `gguf.float-scalar-delta.v1` for a single little-endian GGUF v2/v3 file. Paths are workspace-relative to the process working directory and confined through Nexss `kernel/xfs`; outputs are distinct from all source files.

```nflow
@require github.com/nexssp/ai main as ai
@include ai:nflows/lib/weights.nflow AS weights

{
  model: {
    model_id: "local-experiment-a",
    format: "safetensors",
    adapter: "safetensors.scalar-delta.v1",
    path: "models/base.safetensors"
  },
  include_sha256: true,
  sample_elements: 4
} -> weights.pipeline.inspect
```

Set `include_sha256: true` on inspection, then pass that exact `sha256` in a separate patch call with a caller-supplied `edits` list. For a sharded SafeTensors input, pass the `.safetensors.index.json` path; its digest covers the index and every referenced shard. Use one writer per output path; non-overwrite commits are atomic and reject a competing creator, but concurrent writers that explicitly set `overwrite: true` are not coordinated. Prefer versioned output indexes for sharded checkpoints: replacing an index leaves its previous shard files intact so existing readers are not broken, and cleanup is an operator responsibility. GGUF edits are limited to F16, BF16, F32, and F64 tensors; recognized quantized blocks are preserved but not edited or dequantized, and GGUF split bundles are rejected. PyTorch `.pt`/`.pth`/`.bin` checkpoints are rejected because no safe non-executing tensor-only reader is included; the code does not invoke pickle or Python. Model IDs are opaque labels; the core does not branch by model name. The weights actions do not load or architecture-validate models, generate WSP proposals, score outputs, or evaluate held-out behavior; the optional llama.cpp wrapper below only runs local generations for manual review. See the [standalone inspection workflow](weights_inspect.nflow) and [weights guide](../weights/README.md) for the detailed support matrix and security boundary.

## Local llama.cpp evaluation

The AI bundle also exports [`lib/llama_cpp.nflow`](lib/llama_cpp.nflow), a typed wrapper around Flow's existing `external.exec` action. Flow's standard native bundle already registers `external.exec`, so this project does not add a second subprocess implementation or download a platform-specific executable. Install/build `llama-cli` and obtain the GGUF files through your own trusted process; the workflow accepts the local executable and model paths and does not fetch either one.

```nflow
@require github.com/nexssp/ai as ai
@include ai:nflows/lib/llama_cpp.nflow AS llama_cpp

{
  llama_cli_path: "llama-cli", # or a path to an already-installed local executable
  model_path: "models/base.gguf",
  prompt: "Question: What is the capital of France? Answer:",
  n_predict: 15,
  temperature: 0.0
} -> llama_cpp.pipeline.evaluate
```

The wrapper passes separate argv strings (`-m`, model, `-p`, prompt, and the bounded generation settings); it does not interpolate a shell command. It returns the CLI's raw `stdout`, `stderr`, and exit metadata instead of stripping the prompt or guessing which output is the answer. [`examples/weights/llama_cpp_compare.nflow`](examples/weights/llama_cpp_compare.nflow) runs the same prompt against a baseline and candidate and returns both raw outputs for manual review. Repeat that comparison over a predeclared validation set; this example does not score quality, establish a causal improvement, or release/deploy a model.

Important: `external.exec` runs arbitrary host processes. Treat it as a privileged capability and use only trusted workflows; do not expose it to untrusted tenants or allow an LLM to choose the executable. The local llama.cpp installation and model remain operator-managed. The GGUF editing adapter still edits only supported unquantized float tensors; quantized tensors are preserved, not edited. Inference compatibility and held-out behavior must be checked independently.

## Reusable critic workflow

The bundle embeds and exports `nflows/lib/critic.nflow`; import it without copying the module:

```nflow
@include ai:nflows/lib/critic.nflow AS critic
{
    goal: "Check whether this output answers the request safely.",
    output: "The proposed output to review."
} -> critic.pipeline.review
```

The critic sends the goal and proposed output as explicit prompt context and asks for JSON text with `approved`, `score`, `critique`, and `defects` fields. Importing the module does not run the review. Other fixtures include [`critic.nflow`](critic.nflow), [`roles.nflow`](roles.nflow), and [`parallel_roles.nflow`](parallel_roles.nflow).

## Checks

The tests use fake providers and need no live API credentials:

```sh
go test ./...
task check
```
