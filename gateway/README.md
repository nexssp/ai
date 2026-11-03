# AI gateway

`gateway` is a configurable wrapper around `engine/llm.Provider`. The host supplies a closed route table, a selector that chooses a route name, optional pre-dispatch gates, and post-call observers. The package has no built-in provider preference, model default, retry rule, or hidden fallback.

## Example

```go
routes := []gateway.Route{
    {Name: "fast", Model: "account-model-fast", Provider: fastProvider},
    {Name: "quality", Model: "account-model-quality", Provider: qualityProvider},
}

gw, err := gateway.New(gateway.Config{
    Routes: routes,
    Select: func(_ context.Context, req llm.Request) (string, error) {
        // Use validated host metadata or an application-owned policy.
        switch req.CandidateID {
        case "fast-candidate":
            return "fast", nil
        case "quality-candidate":
            return "quality", nil
        default:
            return "", fmt.Errorf("candidate has no configured route")
        }
    },
    Gates: []gateway.Gate{
        func(ctx context.Context, req llm.Request, route gateway.Route) error {
            // Check an allowlist or perform an application-owned preflight here.
            // A gate error guarantees the provider is not invoked.
            return nil
        },
    },
    Observers: []gateway.Observer{
        func(_ context.Context, event gateway.Event) {
            // Record route, outcome, duration, and usage without prompt/answer text.
        },
    },
})
if err != nil {
    return err
}

answer, err := gw.Complete(ctx, llm.Request{
    CandidateID: "fast-candidate",
    Messages:    []llm.Message{{Role: llm.RoleUser, Content: "..."}},
})
```

A route may override an unpriced request's model; if `Route.Model` is empty, the caller's model is preserved. To protect exact-card estimates, a request carrying a rate-card ID, expected currency, or token estimate is rejected if the route would change its model. The selector can only choose a route present in the configured route table. A gate runs after selection but before provider dispatch. Gates and selectors are host callbacks, must treat request data as read-only, and must be concurrency-safe when a gateway is shared.

`Gateway` implements `llm.Provider`, so a host can inject it into an AI Flow bundle with `nexssflow.NewBundleWithProviderBuilder`. This gives an application such as Jumalu a reusable route/guard extension point while keeping the workflow's candidate ordering and exact-card preflight visible. `ai.team` forwards candidate/attempt IDs but does not carry exact-card estimate metadata; use the explicit `ai.prompt` attempt path for exact-card budget gates. The host still owns durable reservations and settlement, authorization, retry policy, and audit retention; gateway observers are telemetry callbacks, not an accounting boundary. Existing `engine/llm` provider-chain and fallback helpers remain available and are not invoked implicitly by this gateway.

For non-stream calls, the observer receives route, model, candidate/attempt/rate-card identifiers, usage, duration, outcome, and the original error; it does not receive prompt or response content. Provider errors can still contain upstream details, so do not log them blindly. Observers are synchronous and should be quick, concurrency-safe, and non-panicking; a panic propagates to the caller. The current stream interface provides chunks and errors only, so stream events report zero token usage. A stream sequence is single-use; no event is produced if it is never consumed, and an early consumer stop is reported as `abandoned`.

## Ownership boundaries

- Keep model routing policy, request gates, and AI-specific cache/RAG/memory semantics in the application or `nexssp/ai`. The [`retrieval`](../retrieval) package owns chunking, embeddings, search semantics, prompt assembly, and the `VectorIndex` adapter boundary; [`memory`](../memory) remains the separate conversation-memory service.
- Keep generic persistence capabilities and backend adapters in [`nexssp/storage`](https://github.com/nexssp/storage/tree/v0.5.0). Do not add model selection, embeddings, RAG, prompt policy, or semantic similarity behavior to that general storage module. A vector backend should implement the AI-owned `retrieval.VectorIndex` contract in an adapter.
- Keep MCP protocol servers and transport concerns in [`nexssp/transportai/tmcp`](https://github.com/nexssp/transportai/tree/v0.11.0/tmcp). A later AI client adapter can translate model tool calls to MCP calls without making the gateway own the protocol server.
- Keep the application host as the authority for durable budget reservations/settlement, authorization, audit retention, and retry policy.

The retrieval package supplies a basic vector-search/RAG core and adapter extension point, but no persistent vector database, semantic cache, MCP client, tool-execution loop, persistent agent harness, or evaluation framework is bundled here.
