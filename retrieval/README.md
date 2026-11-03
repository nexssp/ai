# Vector retrieval and RAG

The `retrieval` package adds AI-owned vector retrieval and a small retrieval-augmented generation (RAG) service. It uses an injected embedder and a `VectorIndex` interface; it does not choose a vector database, model, hosted provider, or persistence format. `llm.Embedder` already satisfies the package's embedder interface, and any `llm.Provider`—including the AI gateway—can generate the final answer.

## Index and query

```go
embedder := llm.NewEmbedder(baseURL, apiKey, "your-embedding-model")
index := retrieval.NewMemoryIndex() // Local/demo only; not persistent.
service, err := retrieval.NewService(retrieval.Config{
    ChunkSize:       800,
    ChunkOverlap:    120,
    TopK:            5,
    MaxContextRunes: 6000,
}, embedder, index)
if err != nil {
    return err
}

_, err = service.Index(ctx, retrieval.Document{
    Namespace: "tenant-123",
    ID:        "handbook-v1",
    Text:      handbookText,
    Metadata:  map[string]string{"kind": "handbook"},
})
if err != nil {
    return err
}

results, err := service.Search(ctx, retrieval.SearchRequest{
    Namespace: "tenant-123",
    Query:     "How do I request time off?",
})
```

`Index` splits text on Unicode code-point boundaries, embeds the chunks in one batch, and atomically replaces the previous chunks for that document. The overlap and chunk size are character counts, not tokenizer-aware token limits. A re-indexed shorter document will not retain stale trailing chunks. The namespace is mandatory on indexing, searching, and deletion; use it as an authorization/data-isolation boundary and ensure every persistent adapter honors it.

## Grounded answer

```go
answer, err := service.Answer(ctx, chatProvider, retrieval.AnswerRequest{
    Namespace: "tenant-123",
    Query:     "How do I request time off?",
    Model:     "your-chat-model",
    MaxTokens: 300,
})
if err != nil {
    return err
}
fmt.Println(answer.Answer)
for _, source := range answer.Sources {
    fmt.Printf("%s (%s): %.3f\n", source.DocumentID, source.ChunkID, source.Score)
}
```

The answer prompt tells the model to treat retrieved passages as untrusted reference data, cite them by source number, and admit when they do not support an answer. The response returns the exact source chunks used, but citations remain model-generated and must not be treated as proof of correctness. Prompt instructions are not a security boundary; applications must still enforce access control before indexing and retrieval, and should apply their own output validation where needed.

## NFlow actions

The AI Flow bundle exposes `ai.retrieval.index`, `ai.retrieval.search`, `ai.retrieval.delete`, and `ai.rag.answer` with concrete typed request and response structs. Configure the embedder and `VectorIndex` in a `retrieval.Service`, then inject it into the AI bundle with `nexssflow.NewBundleWithServices`; the regular provider builder supplies the chat provider for `ai.rag.answer`. The action descriptions and [`../nflows/retrieval_rag.nflow`](../nflows/retrieval_rag.nflow) document the snake_case request/response schemas and runnable index/search/delete/answer pipelines. The example's answer pipeline requires an already configured service and an indexed namespace.

## Persistence and ownership

`VectorIndex` is the AI-facing adapter contract: it replaces a document's chunks atomically, deletes that document within a namespace, and returns namespace-scoped, descending similarity matches. A persistent vector database adapter can live alongside the selected storage backend. Keep generic persistence, connection management, and database operations in `nexssp/storage`; keep chunking, embeddings, similarity/RAG semantics, prompt assembly, and model policy in `nexssp/ai`.

`MemoryIndex` provides normalized cosine similarity in process memory for tests and local demos only. Its contents disappear on restart; it is not a substitute for a persistent vector database. The current increment supplies the retrieval/RAG core and extension point, not a production vector DB adapter, reranker, hybrid keyword search, or token-aware chunker.
