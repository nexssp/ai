# Nexss AI

Reusable Go packages for LLM providers, Nexss Flow workflows, and host-integrated conversation memory.

The repository is the Go module [`github.com/nexssp/ai`](https://github.com/nexssp/ai). The current source is on `main`; no release tag is assumed here. Until a release is published, Go consumers can select the branch with `@main` (or pin a commit for reproducible builds).

## Install

This module declares Go `1.27.3` in [`go.mod`](go.mod). Add the current branch to a Go application with:

```sh
go get github.com/nexssp/ai@main
```

## Quick start: call a model from Go

Set the provider API key and an actual model ID supported by that account. Presets provide endpoints and credential environment names; they do not select a model.

```sh
export OPENAI_API_KEY="your-api-key"
export LLM_MODEL="your-model-id"
```

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"

    "github.com/nexssp/ai/engine/llm"
)

func main() {
    provider, err := llm.NewProvider("openai", llm.Config{
        DefaultModel: os.Getenv("LLM_MODEL"),
    })
    if err != nil {
        log.Fatal(err)
    }

    response, err := provider.Complete(context.Background(), llm.Request{
        Messages: []llm.Message{{
            Role:    llm.RoleUser,
            Content: "Explain what a Go interface is in one sentence.",
        }},
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(response.Content)
}
```

A request can override its model and include messages, temperature, maximum tokens, JSON mode, tool definitions, or a typed reasoning-effort value where the selected provider/model supports it. Unsupported options are left to the provider API to reject; the shared interface does not claim identical capabilities across vendors.

## Packages

| Package | What it provides |
| --- | --- |
| [`engine/llm`](engine/llm) | Provider interfaces and implementations, chat and streaming requests, model discovery, provider chains and routing helpers, structured decision DTOs, and externally configured cost estimates. |
| [`nexssflow`](nexssflow) | A Nexss Flow v0.65.0+ bundle named `ai` with typed `ai.prompt`, `ai.team`, `ai.decide`, and `ai.cost.estimate` actions, plus embedded workflows. |
| [`memory`](memory) | A host-side hybrid conversation-memory service with bounded Redis working history and application-supplied long-term memory and LLM adapters. See the [memory guide](memory/README.md). |
| [`extensions/ai_assist`](extensions/ai_assist) | An experimental placeholder bundle. Its asynchronous error-analysis hook is currently a TODO; it does not provide working model analysis yet. |

### Providers and models

`llm.NewProvider` supports these provider names and aliases:

- **OpenAI-compatible chat APIs:** `openai`, `groq`, `nvidia`, `nim`, `deepseek`, `openrouter`, `lmstudio`, `vllm`, `xai`, and `inception`.
- **Provider-specific APIs:** `anthropic` / `claude`, `gemini` / `google`, and `ollama` / `ollama_cloud`.

Presets contain endpoints and credential environment names only. Model IDs are supplied by the caller or provider configuration, or fetched from a live provider catalog. If discovery is unavailable, the library returns an error rather than inventing a fallback model. Credentials are read from the corresponding environment variables when available: `OPENAI_API_KEY`, `GROQ_API_KEY`, `NVIDIA_API_KEY` (also `NVIDIA_NIM_API_KEY` or `NVCF_API_KEY`), `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY` (also `OPENROUTER_KEY`), `LMSTUDIO_API_KEY`, `VLLM_API_KEY`, `XAI_API_KEY`, `INCEPTION_API_KEY`, `ANTHROPIC_API_KEY`, and `GEMINI_API_KEY`. Ollama can be used locally without an API key. Override preset endpoints with `llm.Config.BaseURL` and consult the provider for model names and API compatibility.

### Decision-capable models and providers

`ai.decide` requires a positive `max_output_tokens` cap. Cap-aware chat adapters pass that cap to the provider API (`max_tokens` for OpenAI-compatible and Anthropic APIs; `maxOutputTokens` for Gemini) and reject responses whose reported output usage exceeds it. The documented native Inception `/v1/decisions` and TypeSafe `/v1/systemone` APIs have no provider-enforced output-token cap, so those adapters fail closed before dispatch; use a cap-aware chat adapter such as `inception-chat` instead. Inception's [Decisions API](https://docs.inceptionlabs.ai/capabilities/decisions) and TypeSafe's [System One request schema](https://docs.typesafe.ai/api/system-one) document their current request fields.

Anthropic is supported through its official [Structured Outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) API: `decision_provider: "anthropic"` sends the typed per-question schema in `output_config.format`. Choose a model from Anthropic's currently supported structured-output model list; no model ID is selected automatically.

The same typed `DecisionProvider` contract can use locally validated JSON-mode chat adapters with `openai`, `gemini`, `xai`, and `deepseek`. Their official APIs document structured or JSON output capabilities: [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs), [Gemini structured outputs](https://ai.google.dev/gemini-api/docs/structured-output), [xAI structured outputs](https://docs.x.ai/developers/model-capabilities/text/structured-outputs), and [DeepSeek JSON Output](https://api-docs.deepseek.com/guides/json_mode/). The current adapter asks for JSON, parses the answer into Go structs, and checks declared question/choice IDs and extraction spans locally; it does not send each vendor's stricter schema format. Malformed, blank, undeclared, or out-of-allowlist answers return an error rather than triggering a fallback.

### Rate cards and cost estimates

The default rate-card catalog is empty. The library does not bundle model IDs, token rates, currencies, or billing windows. Load a rate card from application-managed YAML, a file, or a URL before relying on estimates. Each configured card must declare a currency, globally or on the card; unknown or currency-less rates remain unknown rather than appearing as zero-cost.

```yaml
currency: EUR
models:
  exact-model-id-from-provider:
    peak:
      prompt_per_1k: 0.0
      prompt_cache_hit_per_1k: 0.0
    comp_per_1k: 0.0
```

Replace the zeroes and model ID with values verified for the account and provider; they are placeholders, not published prices. If pricing differs by time, configure the `window` and `off_peak` fields as appropriate. `ai.cost.estimate` requires the exact `rate_card_id` and `expected_currency`; for a nested decision, its completion-token basis must equal `max_output_tokens`, so the estimate uses the same cap sent to the provider. Explicit-card estimates round up to the next micro-unit. The estimate action only calculates and reports a number: it does not bill, reserve, authorize a selector, or enforce a budget. The application host guard must approve and reserve selector cost before dispatching a decision. Neither it nor `llm.CalculateCostForRateCard` selects a candidate or retries a failed call.

## Use from Nexss Flow

The Flow bundle is provided by `github.com/nexssp/ai/nexssflow`. In a workflow, select the current branch with the separate `main` version selector and replace the chat model placeholder with an ID supported by the configured account:

```nflow
@require github.com/nexssp/ai main as ai {
    provider: "openai",
    model: "your-model-id",
    api_key_env: "OPENAI_API_KEY",
    decision_provider: "inception-chat",
    decision_model: "your-decision-chat-model-id",
    decision_api_key_env: "INCEPTION_API_KEY",
    timeout: "30s"
}

{
    user: "Explain what a Go interface is in one sentence.",
    max_tokens: 160,
    temperature: 0.2
} -> ai.prompt
```

`api_key_env` and `decision_api_key_env` are **environment-variable names**, not secret values. The Flow bundle accepts `provider`, `model`, `api_key_env`, `timeout`, `base_url`, `decision_provider`, `decision_model`, `decision_api_key_env`, and `decision_base_url` as `@require` options. The normal chat model and decision model are selected independently; there is no implicit candidate-model fallback. The default provider is OpenAI and the default timeout is 60 seconds, but no default model is selected. `base_url` must be an absolute HTTP(S) URL. You can also use a local requirement while developing in this checkout; included workflows use `@require ../` from the `nflows/` directory. The Flow runtime or CLI is separate from this Go module.

### Typed Flow actions

Each action has a Go request struct and a typed result. `ai.prompt` accepts a non-empty `user`, optional `system`, provider/model/base URL, token and sampling settings, JSON mode, reasoning effort, candidate and attempt IDs, an explicit nested `context`, and optional exact `rate_card_id`/`expected_currency` attribution with caller token estimates. When pricing metadata is supplied, it must match the configured card for the explicit model before provider dispatch. Its `PromptResult` echoes candidate/attempt and pricing identity and reports actual provider usage. Accounting fields are host metadata and are not sent in provider wire payloads. It does **not** merge arbitrary input keys into the result; carry information needed later in the named `context` field.

`ai.team` runs two or more declared role turns **in order** during one Flow invocation. Each role needs a unique `name`, a `role`, and `instructions`; it may override provider, model, base URL, and reasoning effort. Every role sees the original goal and earlier role answers. `TeamResult` includes the goal, typed role turns, transcript, final content/model, aggregate token counts, and per-turn usage. Cost is left per turn because roles may use different currencies or rate cards. This is temporary in-run orchestration, not a persistent agent, session, or memory store.

`ai.decide` requires a separately configured provider and model plus a positive `max_output_tokens` value. Its optional pricing metadata stays separate from model identity and is echoed with the attempt ID. Its `allowed_ids` constrain choice criteria and answers to the supplied candidate IDs; the action validates the choice, but it does not own the application's routing graph, budget policy, selector authorization/reservation, or fallback. Keep those controls in the application host guard and caller's NFlow graph. The runnable [`decision_route.nflow`](nflows/decision_route.nflow) shows `ai.decide` followed by a Flow `match` over the typed answer, with a final manual-review branch. Flow expressions access Go structs by exported field names, so the example reads `result.Answers[0].Choice`.

A decision request uses typed question definitions. For example, the workflow fixture supplies `state`, `allowed_ids`, and a `choice` question with an explicit candidate description for each ID. `ai.decide` returns the model, typed answers, probabilities/spans where supplied, and per-call usage. For a nested decision, `ai.cost.estimate` requires the completion-token basis to equal `max_output_tokens`; missing, invalid, or mismatched caps are rejected. It includes the configured currency only when the exact rate card matches and remains a calculation, not a billing or reservation gate.

The exported embedded module [`nflows/lib/critic.nflow`](nflows/lib/critic.nflow) defines `critic.pipeline.review`; import it and call the pipeline without copying the module into an application:

```nflow
@include ai:nflows/lib/critic.nflow AS critic

{
    goal: "Check whether this output answers the request safely.",
    output: "The proposed output to review."
} -> critic.pipeline.review
```

The critic sends the goal and proposed output as explicit prompt context and asks for JSON text with `approved`, `score`, `critique`, and `defects` fields. Importing the module does not run the review. Other examples are [`nflows/critic.nflow`](nflows/critic.nflow) for a standalone wrapper, [`nflows/roles.nflow`](nflows/roles.nflow) for ordered roles, and [`nflows/parallel_roles.nflow`](nflows/parallel_roles.nflow) for independent prompts composed in parallel by Flow before synthesis.

## Conversation memory

The `memory` package is a host-side service, not a `.nflow` bundle. It combines recent scoped turns from a `WorkingStore` with optional relevant results from a `LongTermStore`, constructs the prompt, and calls an injected `Completer`. Redis is the included working-history backend; applications provide their own semantic-memory adapter and LLM completer. Importance-based promotion uses a simple keyword scorer by default, which can be replaced with an application-specific scorer. There is no built-in LEANN adapter, scheduler, or active-scope discovery. See [`memory/README.md`](memory/README.md) for setup, request flow, promotion, and failure handling.

## Development

The repository's contribution guide routes development checks through [Task](https://taskfile.dev); see [`CONTRIBUTING.md`](CONTRIBUTING.md) for tool setup and contribution rules. Useful targets include:

```sh
task setup
task test
task ci
```

`task test` runs the short Go test suite; `task ci` runs lint and uncached race-enabled tests. Tests use fake providers and do not require live LLM credentials. The project is licensed under [Apache-2.0](LICENSE).
