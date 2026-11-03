// Package retrieval provides provider-neutral vector retrieval and a small RAG service.
// Persistent vector databases are supplied by adapters that implement VectorIndex.
package retrieval

import "context"

// Embedder converts text into vectors. llm.Embedder satisfies this interface.
type Embedder interface {
	Embed(ctx context.Context, texts ...string) ([][]float32, error)
}

// Document is a source to index. Namespace is a required isolation boundary;
// callers should use one namespace per security/data-access domain.
type Document struct {
	Namespace string            `json:"namespace"`
	ID        string            `json:"id"`
	Text      string            `json:"text"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

// VectorRecord is one embedded text chunk stored by a VectorIndex adapter.
type VectorRecord struct {
	Namespace  string            `json:"namespace"`
	DocumentID string            `json:"document_id"`
	ChunkID    string            `json:"chunk_id"`
	Text       string            `json:"text"`
	Metadata   map[string]string `json:"metadata,omitempty"`
	Vector     []float32         `json:"vector"`
}

// VectorQuery asks an index for its best matches within one namespace. Scores
// use the normalized [0,1] similarity convention where larger is better.
type VectorQuery struct {
	Namespace string
	Vector    []float32
	Limit     int
	MinScore  float32
}

// Match is a retrieved chunk, including source identifiers for citations.
type Match struct {
	Namespace  string            `json:"namespace"`
	DocumentID string            `json:"document_id"`
	ChunkID    string            `json:"chunk_id"`
	Text       string            `json:"text"`
	Metadata   map[string]string `json:"metadata,omitempty"`
	Score      float32           `json:"score"`
}

// VectorIndex is the AI-owned boundary implemented by a persistent vector
// adapter. ReplaceDocument must atomically replace all chunks for that document
// in the namespace; Search must return only that namespace and rank by descending
// normalized similarity. Generic persistence and database policy belong in the
// adapter, not this package.
type VectorIndex interface {
	ReplaceDocument(ctx context.Context, namespace, documentID string, records []VectorRecord) error
	DeleteDocument(ctx context.Context, namespace, documentID string) error
	Search(ctx context.Context, query VectorQuery) ([]Match, error)
}

// Config controls chunking, retrieval, and the amount of source text passed to
// the generator. Character limits count Unicode code points, not model tokens.
type Config struct {
	ChunkSize       int
	ChunkOverlap    int
	TopK            int
	MinScore        float32
	MaxContextRunes int
}

// IndexResult reports a completed document replacement.
type IndexResult struct {
	Namespace  string `json:"namespace"`
	DocumentID string `json:"document_id"`
	ChunkCount int    `json:"chunk_count"`
}

// SearchRequest retrieves source chunks for a natural-language query.
type SearchRequest struct {
	Namespace string   `json:"namespace"`
	Query     string   `json:"query"`
	TopK      int      `json:"top_k,omitempty"`
	MinScore  *float32 `json:"min_score,omitempty"`
}

// SearchResponse is the typed result of a retrieval operation.
type SearchResponse struct {
	Matches []Match `json:"matches"`
}

// AnswerRequest asks a configured LLM provider to answer with retrieved sources.
type AnswerRequest struct {
	Namespace    string   `json:"namespace"`
	Query        string   `json:"query"`
	TopK         int      `json:"top_k,omitempty"`
	MinScore     *float32 `json:"min_score,omitempty"`
	Model        string   `json:"model,omitempty"`
	SystemPrompt string   `json:"system_prompt,omitempty"`
	MaxTokens    int      `json:"max_tokens,omitempty"`
}

// AnswerResponse contains the model output and the exact retrieved passages
// supplied as context so callers can inspect or display citations.
type AnswerResponse struct {
	Answer  string  `json:"answer"`
	Sources []Match `json:"sources"`
	Model   string  `json:"model,omitempty"`
}
