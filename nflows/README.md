# ai as a reusable NFlow package

`nexssflow` keeps provider protocols, HTTP requests, credentials, timeouts, and normalized errors in Go, while NFlow defines orchestration, routing graphs, and reusable workflow logic. This checkout uses Flow `v0.67.1`; the `nflow` runtime/CLI is installed separately from the Go module.

## Load the package

From a workflow in this checkout:

```nflow
@require ../ as ai {
    provider: "openai",
    model: "your-model-id",
    api_key_env: "OPENAI_API_KEY",
    decision_provider: "inception",
    decision_model: "mercury-decide",
    decision_api_key_env: "INCEPTION_API_KEY",
    timeout: "30s"
}
```

The `model` and `decision_model` values must be selected for the configured provider account; no default model or rate card is bundled. Inception's official Mercury Decide model ID is `mercury-decide`; see the [Decisions API docs](https://docs.inceptionlabs.ai/capabilities/decisions) and [model list](https://docs.inceptionlabs.ai/get-started/models). `api_key_env` and `decision_api_key_env` are environment-variable names, never secret values. Accepted options are `provider`, `model`, `api_key_env`, `timeout`, `base_url`, `decision_provider`, `decision_model`, `decision_api_key_env`, and `decision_base_url`.

## Typed actions

The library exports `ai.prompt`, `ai.team`, `ai.decide`, and `ai.cost.estimate`. Each action is backed by a Go request struct and a typed result. `ai.prompt` returns content, model/provider, token counts, usage, and explicitly carried IDs/context; it does not merge arbitrary Flow state fields. `ai.team` executes two or more declared role turns sequentially and returns typed role outputs and per-turn usage.

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
