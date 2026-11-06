# Hybrid conversation memory

The `memory` package is a host-side Go service for applications built with Flow. It combines bounded Redis working history with an injected long-term semantic store and a configured model completer. The host owns all storage and provider connections; this package is not itself a `.nflow` bundle.

## Choose `memory_scope` explicitly

Every request identifies both the authenticated `UserID` and active `ConversationID`. Recent working history always uses both IDs and therefore never crosses conversations. `Scope.MemoryScope` selects the destination for long-term recall and promotion:

- `conversation` stores and recalls durable memories for this user in this conversation only.
- `user` stores and recalls durable memories for this user across their conversations. The `LongTermStore` receives that user ID with an empty conversation ID, so the adapter must enforce user isolation.

An omitted `memory_scope` remains the backward-compatible `conversation` behavior. Set user and conversation IDs from trusted host/session identity—not from untrusted model output. User-scoped memory is still private to one user; it is not global across users.

## Request flow

For each incoming message, `Service.Reply` validates the full conversation identity and memory scope, reads recent turns using the conversation-only scope, searches long-term memory using the requested `memory_scope`, assembles the prompt, and calls the supplied completer. After successful completion it appends the exchange to conversation-local working history. Explicit `Remember` or an importance score at/above the configured threshold upserts the exchange into the requested long-term scope.

The default importance scorer is a small configurable keyword matcher; it is a transparent baseline, not a claim of semantic importance detection. Hosts can inject a classifier or model-backed `ImportanceScorer`. A custom `ShouldRecall` policy can skip long-term search. Retrieved memories are added in a separate system message and labeled as untrusted reference data.

## Example

```go
working, err := memory.NewRedisWorkingStore(redisClient, memory.RedisConfig{
    Prefix:   "myapp:conversation-memory",
    MaxTurns: 32,
    TTL:      14 * 24 * time.Hour,
})
if err != nil {
    return err
}

service, err := memory.NewService(memory.Config{
    SystemPrompt:        "Answer clearly and use relevant user context.",
    RecentTurns:         8,
    RecallTopK:          4,
    ImportanceThreshold: 0.8,
    Keywords:            []string{"remember", "my preference is", "my timezone is"},
}, working, semanticStore, chatCompleter)
if err != nil {
    return err
}

reply, err := service.Reply(ctx, memory.Request{
    Scope: memory.Scope{
        UserID:         authenticatedUserID,
        ConversationID: activeConversationID,
        MemoryScope:    memory.MemoryScopeUser, // or MemoryScopeConversation
    },
    UserMessage: text,
})
```

## Backends and LEANN

`WorkingStore` is implemented by `RedisWorkingStore`. Keys are derived from both user and conversation IDs, Redis list updates and retention trimming run in a transaction, and history is bounded (256 turns by default). The caller owns the Redis client lifecycle.

`LongTermStore` defines `Search(scope, query, limit)` and idempotent `Upsert(record)` operations. It receives an explicit `MemoryScope`; for user scope, `ConversationID` is empty, while conversation scope includes both IDs. LEANN can be integrated behind this interface, but there is no LEANN adapter in this package: implement the adapter against the LEANN runtime/version actually deployed rather than relying on a guessed wire format.

## Background promotion

A host scheduler may call `PromoteOlder` for each active conversation with a cutoff and batch limit. The service reads and deletes working turns using both user and conversation IDs, and uses each turn's recorded `memory_scope` when promoting it. Upserts use stable turn IDs, so retrying after an upsert succeeds but deletion fails is safe provided the long-term adapter honors idempotent upsert-by-ID. This package does not own a scheduler or discover active scopes; the application does.

## Failure behavior

Retrieval or completion failures stop the request before it is persisted. A Redis or long-term write failure is returned after the model answer has been generated; in that case the `Response` may contain `Assistant`, and callers should report/retry persistence separately rather than blindly repeating a potentially costly model call. The package does not silently swallow memory failures.
