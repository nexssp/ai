# Nexss AI

Reusable Go packages for LLM providers, Nexss Flow workflows, and host-integrated conversation memory.

The repository is the Go module [`github.com/nexssp/ai`](https://github.com/nexssp/ai). The current source is on `main`; no release tag is assumed here. Until a release is published, Go consumers can select the branch with `@main` (or pin a commit for reproducible builds).

## Install

This module declares Go `1.27.3` in [`go.mod`](go.mod). Add the current branch to a Go application with:

```sh
go get github.com/nexssp/ai@main
```

## Quick start: call a model from Go

Set the provider's API key in the environment. The preset below reads `OPENAI_API_KEY` automatically when no key is supplied in `llm.Config`.

```sh
export OPENAI_API_KEY="your-api-key"
```

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/nexssp/ai/engine/llm"
)

func main() {
	provider, err := llm.NewProvider("openai", llm.Config{})
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

Pass `llm.Config` to set a different `DefaultModel`, `BaseURL`, `APIKey`, `Timeout`, or `MaxRetries`. A request can override its model and include messages, temperature, maximum tokens, JSON mode, tool definitions, or a request to include reasoning where the provider supports it. The shared interface is intentionally provider-neutral; individual APIs do not support every option in the same way.

## Packages

| Package | What it provides |
| --- | --- |
| [`engine/llm`](engine/llm) | Provider interfaces and implementations, chat and streaming requests, embedding generation, model discovery, model routing/fallback helpers, Kernel action adapters, and configurable cost estimates. |
| [`nexssflow`](nexssflow) | A Nexss Flow bundle named `ai` with `ai.prompt` and `ai.team`, plus an embedded critic workflow. |
| [`memory`](memory) | A host-side hybrid conversation-memory service with bounded Redis working history and application-supplied long-term memory and LLM adapters. See the [memory guide](memory/README.md). |
| [`extensions/ai_assist`](extensions/ai_assist) | An experimental placeholder bundle. Its asynchronous error-analysis hook is currently a TODO; it does not provide working model analysis yet. |

### Providers

`llm.NewProvider` supports these provider names and aliases:

- **OpenAI-compatible chat APIs:** `openai`, `groq`, `nvidia`, `nim`, `deepseek`, `openrouter`, `lmstudio`, and `vllm`.
- **Provider-specific APIs:** `anthropic` / `claude`, `gemini` / `google`, and `ollama` / `ollama_cloud`.

The presets supply default endpoints and models. Credentials are read from the corresponding environment variables when available: `OPENAI_API_KEY`, `GROQ_API_KEY`, `NVIDIA_API_KEY` (also `NVIDIA_NIM_API_KEY` or `NVCF_API_KEY` for NVIDIA), `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY` (also `OPENROUTER_KEY`), `LMSTUDIO_API_KEY`, `VLLM_API_KEY`, `ANTHROPIC_API_KEY`, and `GEMINI_API_KEY`. Ollama can be used locally without an API key. You can override preset endpoints with `llm.Config.BaseURL`; use a provider's own documentation for model names and API compatibility.

The common `llm.Provider` interface exposes `Complete` and `Stream`. The package also includes an OpenAI-compatible `NewEmbedder`, model discovery through `FetchModels`, and helpers for provider chains, transient-error fallback, tiered routing, and expert routing. Model discovery may return a built-in fallback list when a live catalog cannot be reached. Cost estimates depend on the configured rate card and exact model name; review pricing data before using estimates for billing decisions.

## Use from Nexss Flow

The Flow bundle is provided by `github.com/nexssp/ai/nexssflow`. In a workflow, select the current branch with the separate `main` version selector:

```nflow
@require github.com/nexssp/ai main as ai {
    provider: "openai",
    model: "gpt-4o-mini",
    api_key_env: "OPENAI_API_KEY",
    timeout: "30s"
}

{
    user: "Explain what a Go interface is in one sentence.",
    max_tokens: 160,
    temperature: 0.2
} -> ai.prompt
```

`api_key_env` is the **name** of an environment variable, not the secret value. The Flow bundle accepts `provider`, `model`, `api_key_env`, `timeout`, and `base_url` as `@require` options. Its default provider is OpenAI and its default timeout is 60 seconds. `base_url` must be an absolute HTTP(S) URL. You can also use a local requirement while developing in this checkout; the included workflows use `@require ../` from the `nflows/` directory. The Flow runtime or CLI is separate from this Go module.

### Flow actions

`ai.prompt` requires a non-empty `user` string. It accepts optional `system`, `provider`, `model`, `base_url`, `max_tokens`, `temperature`, and `json_mode` values. The action preserves the incoming top-level state and adds `content`, `model`, `prompt_tokens`, `comp_tokens`, and `total_tokens` to the result.

`ai.team` runs two or more declared role turns **in order** during one Flow invocation. Each role needs a unique `name`, a `role`, and `instructions`; it may override the model. Every role sees the original goal and the earlier roles' answers. The result includes the role turns, conversation, final content/model, and aggregate token counts. This is temporary in-run orchestration, not a persistent agent, session, or memory store.

```nflow
{
    goal: "Suggest a safe API timeout policy.",
    agents: [
        {name: "analyst", role: "Analyst", instructions: "Identify one key reliability constraint."},
        {name: "reviewer", role: "Reviewer", instructions: "Use the analyst's finding and propose a concise policy."}
    ],
    max_tokens: 160,
    temperature: 0.2
} -> ai.team
```

The exported embedded module [`nflows/lib/critic.nflow`](nflows/lib/critic.nflow) defines `critic.pipeline.review`; import it and call the pipeline without copying the module into an application:

```nflow
@include ai:nflows/lib/critic.nflow AS critic

{
    goal: "Check whether this output answers the request safely.",
    output: "The proposed output to review."
} -> critic.pipeline.review
```

The critic asks `ai.prompt` for JSON text with `approved`, `score`, `critique`, and `defects` fields; its response is available as `result.content`. Importing the module does not run the review. Other examples are [`nflows/critic.nflow`](nflows/critic.nflow) for a standalone wrapper, [`nflows/roles.nflow`](nflows/roles.nflow) for ordered roles, and [`nflows/parallel_roles.nflow`](nflows/parallel_roles.nflow) for independent prompts composed in parallel by Flow before synthesis.

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
