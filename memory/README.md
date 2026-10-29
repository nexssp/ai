# Hybrid conversation memory

`extensions/memory` is a host-side Go service for applications built with Flow. It combines a bounded Redis working history with an injected long-term semantic store and a model completer. It is deliberately not a `.nflow` DSL bundle: the host owns the LLM and storage connections and supplies them when constructing the service.

## Request flow

For each incoming message, `Service.Reply` validates the user/conversation scope, reads the latest configured number of turns, searches long-term memory with the current message, assembles the system prompt + retrieved memories + recent turns + current user message, and calls the supplied completer. After a successful completion it appends the user/assistant exchange to working memory. An explicit `Remember` flag or an importance score at/above the configured threshold immediately upserts the exchange into long-term memory.

The default importance scorer is a small configurable keyword matcher; it is a transparent baseline, not a claim of semantic importance detection. Hosts can inject a classifier or model-backed `ImportanceScorer`. A custom `ShouldRecall` policy can skip long-term search for messages that do not need it. Retrieved memories are added in a separate system message and explicitly labeled as untrusted reference data.

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
    Scope:       memory.Scope{UserID: userID, ConversationID: conversationID},
    UserMessage: text,
})
```

## Backends and LEANN

`WorkingStore` is implemented by `RedisWorkingStore`. Keys are derived from both user and conversation IDs, Redis list updates and retention trimming run in a transaction, and history is bounded (256 turns by default). The caller owns the Redis client lifecycle.

`LongTermStore` defines `Search(scope, query, limit)` and idempotent `Upsert(record)` operations. The full scope is passed on every call so adapters can keep users and conversations isolated. LEANN can be integrated behind this interface, but there is no LEANN adapter in this package: the upstream project currently documents Python APIs and a CLI, not a stable Go or HTTP service contract. Implement the adapter against the LEANN runtime/version actually deployed rather than relying on a guessed wire format.

## Background promotion

A host scheduler may call `PromoteOlder` for each active scope with a cutoff and batch limit. The service upserts each aged turn using its stable turn ID, then removes only those IDs from Redis. If a write succeeds but deletion fails, a retry is safe provided the long-term adapter honors idempotent upsert-by-ID. This package does not own a scheduler or discover active scopes; the application does.

## Failure behavior

Retrieval or completion failures stop the request before it is persisted. A Redis or long-term write failure is returned after the model answer has been generated; in that case the `Response` may contain `Assistant`, and callers should report/retry persistence separately rather than blindly repeating a potentially costly model call. The package does not silently swallow memory failures.
