# Structured outputs and tool schemas

The AI bundle uses one recursive JSON Schema contract for assistant output and function arguments. Assistant output is declared with `ai.prompt.output_schema`; tool arguments come from `:tool-schema` metadata or a `ToolSpec`; `:tool-result-schema` is a host-only contract for validating tool results before they are sent back to a model. Result schemas are not serialized into vendor tool definitions.

## Capability matrix

These values describe this repository's active adapter routes, not every feature a vendor offers. `Conditional` means the chosen model, server, engine, or endpoint must also support the feature; an API error is returned rather than retrying with another model or route. `Unknown` means support is not established for that adapter/model combination. The runtime exposes the same distinctions through `llm.CapabilityProvider`.

| Adapter route | JSON mode | Function tools | Assistant output schema | Output schema with tools | Strict tool arguments | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| OpenAI | Supported | Conditional | Conditional | Unknown | Conditional | Strict output uses the Chat Completions JSON Schema format; strict schemas require an object root and closed objects. |
| Groq | Conditional | Conditional | Conditional | Unsupported | Unknown | Groq documents structured outputs and tools as incompatible in one call. |
| NVIDIA / NIM | Conditional | Conditional | Conditional | Unknown | Conditional | Served model, engine, and tool parser determine support. |
| DeepSeek | Supported | Conditional | Unsupported | Unsupported | Unsupported | The chat adapter documents JSON mode, not constrained assistant JSON Schema; strict tool mode is not enabled by this adapter. |
| OpenRouter | Conditional | Conditional | Conditional | Unknown | Conditional | Backend/model-dependent; a successful route is not a guarantee for every backend. |
| LM Studio | Conditional | Conditional | Conditional | Unknown | Unknown | Depends on model and server configuration. |
| vLLM | Conditional | Conditional | Conditional | Unknown | Conditional | Depends on version, model backend, parser, template, and server configuration. |
| xAI | Conditional | Conditional | Conditional for Grok 4 family; otherwise Unknown | Conditional for Grok 4 family; otherwise Unknown | Conditional | The adapter uses Chat Completions; structured output with tools is model-family dependent. |
| Inception | Conditional | Conditional | Conditional | Unknown | Conditional | The OpenAI-compatible chat route exposes schema and strict-tool fields; the combination is not established here. |
| Anthropic Messages | Unsupported | Conditional | Conditional | Conditional | Conditional | Uses native Messages output format and `tool_use` / `tool_result` blocks. |
| Gemini | Conditional | Conditional | Conditional | Conditional for `gemini-3*` / `gemini-4*`; otherwise Unknown | Unknown | Uses native `responseSchema`, function declarations, and function-response history. `responseSchema` omits unsupported `additionalProperties`; such contracts require best-effort local validation. |
| Ollama local `/api/chat` | Conditional | Conditional | Conditional | Unknown | Unknown | Uses the local server configured by the caller. |
| Ollama Cloud endpoint | Unknown | Conditional | Unsupported | Unsupported | Unsupported | The native Ollama route does not claim structured-output support for Cloud. The Cloud OpenAI-compatible API is a different route. |
| Generic `openai-compatible` or custom provider | Unknown unless explicitly declared | Unknown unless explicitly declared | Unknown unless explicitly declared | Unknown unless explicitly declared | Unknown unless explicitly declared | The destination vendor/model is caller-selected, so vendor features are not inferred from the adapter label. |

Tool-result history is conditional on the concrete route for all mapped providers except Anthropic Messages, where native `tool_use` / `tool_result` history is supported by this adapter. Streaming schemas and tool calls are unsupported in every concrete adapter above and are rejected by its text-only streaming method; generic custom routes remain unknown until declared.

The route-level values are backed by the provider API documentation: [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs), [Groq Structured Outputs](https://console.groq.com/docs/structured-outputs) and [Tool Use](https://console.groq.com/docs/tool-use/overview), [NVIDIA NIM tool calling](https://docs.nvidia.com/nim/large-language-models/latest/advanced-use-cases/tool-calling-and-mcp.html), [DeepSeek JSON Output](https://api-docs.deepseek.com/guides/json_mode/) and [Tool Calls](https://api-docs.deepseek.com/guides/tool_calls/), [OpenRouter Structured Outputs](https://openrouter.ai/docs/guides/features/structured-outputs), [LM Studio Structured Output](https://lmstudio.ai/docs/developer/openai-compat/structured-output), [vLLM Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/) and [Tool Calling](https://docs.vllm.ai/en/latest/features/tool_calling/), [xAI Structured Outputs](https://docs.x.ai/developers/model-capabilities/text/structured-outputs), [Inception Structured Outputs](https://docs.inceptionlabs.ai/capabilities/structured-outputs) and [Tool Use](https://docs.inceptionlabs.ai/capabilities/tool-use), [Anthropic Messages](https://docs.anthropic.com/en/api/messages) and [Strict Tool Use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/strict-tool-use), [Gemini Schema](https://ai.google.dev/api/rest/v1beta/Schema), [Gemini Structured Output](https://ai.google.dev/gemini-api/docs/generate-content/structured-output) and [Function Calling](https://ai.google.dev/gemini-api/docs/generate-content/function-calling), and [Ollama Structured Outputs](https://docs.ollama.com/capabilities/structured-outputs), [Tool Calling](https://docs.ollama.com/capabilities/tool-calling), and [Cloud API](https://docs.ollama.com/cloud/api). Provider APIs and model capabilities change; keep the matrix conservative when the repository's route or combination is not verified.

## Enforcement and fallback

`output_schema_mode` defaults to `strict`. Strict mode validates the contract before dispatch and stops before the provider call if the active adapter cannot represent the schema or its capability is unsupported/unknown. A conditional capability is sent to the selected route; if that model or server rejects it, the error is returned without switching providers.

Set `output_schema_mode: "best_effort"` to opt into local-only fallback. When the provider cannot enforce the output contract, the adapter omits `output_schema` from its vendor payload and the bundle validates the generated JSON locally. For a strict tool schema (`:tool-schema=!Name`), best-effort may remove only the vendor strict flag; returned arguments are still parsed and checked against the declared schema before being returned to Flow. For assistant output, the result reports `schema_enforcement: "provider"` or `"local_validation"`; tool arguments are locally checked in either mode. This mode does not add datatype instructions to prompts or retry with another route.

The supported contract covers recursive objects and arrays, required and optional properties, string enums, nullable fields, and `additionalProperties`. An untyped `{}` schema accepts any JSON value in local validation, but is not representable in OpenAI strict mode. Unsupported JSON Schema keywords (for example `pattern`, `minimum`, and `$ref`) are rejected rather than silently discarded. `format` is preserved for provider APIs but is not interpreted by the local validator. OpenAI strict output requires a closed object root; optional properties are represented using nullable values and normalized back to omitted fields after validation. Gemini's documented `responseSchema` subset does not include `additionalProperties`, so a closed-object contract containing that keyword is not advertised as provider-enforced there.

All returned tool calls are checked for valid JSON object arguments, matching declared function names, unique IDs, and schema compliance before they reach Flow. Historical tool arguments are parsed and checked when their tool contract is available. Tool-result contracts are validated on the host and are never sent in the function declaration.

## NFlow contracts

Visible `@schema` declarations are materialized as reusable JSON-shaped contracts through `ai.schemas.get` and `ai.schemas.list`, so they can be projected and passed between Flow actions. `ai.tool_specs.list` continues to return typed tool specs; its optional host-only `result_schema` is also a JSON-shaped object and is parsed into the same recursive contract for validation. A pipeline can declare its model arguments separately from its result contract:

```nflow
@schema LookupArguments {
  query string `validate:"required"`
}

@schema LookupResult {
  count int `validate:"required"`
}

@pipeline lookup
  :tool-schema=!LookupArguments
  :tool-description="Look up the supplied query and return its count."
  :tool-result-schema=LookupResult
  # pipeline implementation
@end
```

A caller can fetch a visible schema and pass it directly to `ai.prompt`:

```nflow
{ name: "LookupResult" } -> ai.schemas.get as contract
-> { user: "Return the answer as JSON.", output_schema: contract } -> ai.prompt
```

`ollama_cloud` retains the provider's existing local default endpoint (`http://localhost:11434`). The alias does not redirect a caller to hosted Ollama. Cloud behavior requires an explicitly configured endpoint and is distinct from the native local `/api/chat` route.
