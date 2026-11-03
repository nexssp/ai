package nexssflow

// RetrievalDeleteRequest identifies one document to remove from one namespace.
type RetrievalDeleteRequest struct {
	Namespace  string `json:"namespace"`
	DocumentID string `json:"document_id"`
}

// RetrievalDeleteResponse confirms the namespace-scoped document deletion.
type RetrievalDeleteResponse struct {
	Namespace  string `json:"namespace"`
	DocumentID string `json:"document_id"`
	Deleted    bool   `json:"deleted"`
}

// RAGAnswerActionRequest selects a retrieval namespace, question, and optional
// provider settings for one answer grounded in the retrieved source excerpts.
type RAGAnswerActionRequest struct {
	Namespace    string   `json:"namespace"`
	Query        string   `json:"query"`
	TopK         int      `json:"top_k,omitempty"`
	MinScore     *float32 `json:"min_score,omitempty"`
	Provider     string   `json:"provider,omitempty"`
	Model        string   `json:"model,omitempty"`
	BaseURL      string   `json:"base_url,omitempty"`
	SystemPrompt string   `json:"system_prompt,omitempty"`
	MaxTokens    int      `json:"max_tokens,omitempty"`
}
