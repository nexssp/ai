// Package memory provides a host-side hybrid conversation-memory service for Flow applications.
// It keeps recent turns in a bounded working store, retrieves scoped long-term memories,
// builds an LLM prompt, and promotes important or aged turns through backend interfaces.
//
// Typical use:
//
//	working, _ := memory.NewRedisWorkingStore(redisClient, memory.RedisConfig{})
//	service, _ := memory.NewService(memory.Config{}, working, leannAdapter, chatAdapter)
//	reply, err := service.Reply(ctx, memory.Request{Scope: scope, UserMessage: "..."})
package memory

import (
	"context"
	"time"
)

// Scope is the isolation boundary for conversation and durable memory.
// Backends must use both fields when reading, searching, writing, and deleting.
type Scope struct {
	UserID         string `json:"user_id"`
	ConversationID string `json:"conversation_id"`
}

// Turn is one completed user/assistant exchange stored in working memory.
type Turn struct {
	ID               string    `json:"id"`
	User             string    `json:"user"`
	Assistant        string    `json:"assistant"`
	CreatedAt        time.Time `json:"created_at"`
	ImportanceScore  float64   `json:"importance_score,omitempty"`
	ExplicitRemember bool      `json:"explicit_remember,omitempty"`
}

// SearchHit is a relevant long-term memory returned for the current query.
type SearchHit struct {
	ID    string  `json:"id"`
	Text  string  `json:"text"`
	Score float64 `json:"score,omitempty"`
}

// Record is an idempotently upserted long-term memory. ID is derived from
// the source turn, so a retried promotion replaces rather than duplicates it.
type Record struct {
	ID         string    `json:"id"`
	Scope      Scope     `json:"scope"`
	Text       string    `json:"text"`
	Score      float64   `json:"score"`
	SourceTurn string    `json:"source_turn"`
	CreatedAt  time.Time `json:"created_at"`
	Reason     string    `json:"reason"`
}

// Message is a role/content pair in the prompt sent to a chat model.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// WorkingStore holds bounded, ordered short-term conversation turns.
// Recent must return turns oldest-to-newest. Delete removes only the supplied IDs.
type WorkingStore interface {
	Recent(ctx context.Context, scope Scope, limit int) ([]Turn, error)
	Append(ctx context.Context, scope Scope, turn Turn) error
	Older(ctx context.Context, scope Scope, before time.Time, limit int) ([]Turn, error)
	Delete(ctx context.Context, scope Scope, turnIDs []string) error
}

// LongTermStore is the adapter boundary for LEANN or another semantic-memory backend.
// Search and Upsert must honor the complete Scope; Upsert must be idempotent by record ID.
type LongTermStore interface {
	Search(ctx context.Context, scope Scope, query string, limit int) ([]SearchHit, error)
	Upsert(ctx context.Context, record Record) error
}

// Completer calls the configured LLM with the assembled prompt.
type Completer interface {
	Complete(ctx context.Context, messages []Message) (string, error)
}

// CompleteFunc adapts a function to Completer.
type CompleteFunc func(context.Context, []Message) (string, error)

// Complete implements Completer.
func (f CompleteFunc) Complete(ctx context.Context, messages []Message) (string, error) {
	return f(ctx, messages)
}

// ImportanceScorer estimates whether a user message merits immediate promotion.
// The returned score must be in [0, 1].
type ImportanceScorer interface {
	Score(ctx context.Context, userMessage string) (float64, error)
}

// ImportanceScoreFunc adapts a function to ImportanceScorer.
type ImportanceScoreFunc func(context.Context, string) (float64, error)

// Score implements ImportanceScorer.
func (f ImportanceScoreFunc) Score(ctx context.Context, message string) (float64, error) {
	return f(ctx, message)
}

// RecallPolicy can skip semantic search for messages that do not need older context.
type RecallPolicy func(ctx context.Context, scope Scope, query string) bool

// Config controls prompt assembly, retrieval, and automatic memory promotion.
type Config struct {
	SystemPrompt        string
	RecentTurns         int
	RecallTopK          int
	RecallMinScore      float64
	MaxRecallCharacters int
	ImportanceThreshold float64
	Keywords            []string
	Scorer              ImportanceScorer
	ShouldRecall        RecallPolicy
}

// Request is one incoming conversation message.
type Request struct {
	Scope        Scope
	UserMessage  string
	SystemPrompt string
	Remember     bool // explicit user/application request to retain this turn
}

// Response is the generated answer and memory actions completed for this turn.
type Response struct {
	Assistant  string
	TurnID     string
	Recalled   []SearchHit
	Remembered bool
	Importance float64
}
