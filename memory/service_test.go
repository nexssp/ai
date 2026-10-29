package memory

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/nexssp/kernel/xtest/ktest"
)

type testWorkingStore struct {
	turns     map[string][]Turn
	appendErr error
	deleteErr error
}

func newTestWorkingStore() *testWorkingStore {
	return &testWorkingStore{turns: make(map[string][]Turn)}
}

func (s *testWorkingStore) Recent(_ context.Context, scope Scope, limit int) ([]Turn, error) {
	all := s.turns[scopeKey(scope)]
	if limit < len(all) {
		all = all[len(all)-limit:]
	}
	return append([]Turn(nil), all...), nil
}

func (s *testWorkingStore) Append(_ context.Context, scope Scope, turn Turn) error {
	if s.appendErr != nil {
		return s.appendErr
	}
	key := scopeKey(scope)
	s.turns[key] = append(s.turns[key], turn)
	return nil
}

func (s *testWorkingStore) Older(_ context.Context, scope Scope, before time.Time, limit int) ([]Turn, error) {
	all := s.turns[scopeKey(scope)]
	out := make([]Turn, 0, limit)
	for _, turn := range all {
		if turn.CreatedAt.Before(before) {
			out = append(out, turn)
			if len(out) == limit {
				break
			}
		}
	}
	return out, nil
}

func (s *testWorkingStore) Delete(_ context.Context, scope Scope, turnIDs []string) error {
	if s.deleteErr != nil {
		return s.deleteErr
	}
	wanted := make(map[string]bool, len(turnIDs))
	for _, id := range turnIDs {
		wanted[id] = true
	}
	key := scopeKey(scope)
	kept := make([]Turn, 0, len(s.turns[key]))
	for _, turn := range s.turns[key] {
		if !wanted[turn.ID] {
			kept = append(kept, turn)
		}
	}
	s.turns[key] = kept
	return nil
}

type testSemanticStore struct {
	hits     []SearchHit
	records  []Record
	searched Scope
}

func (s *testSemanticStore) Search(_ context.Context, scope Scope, _ string, _ int) ([]SearchHit, error) {
	s.searched = scope
	return append([]SearchHit(nil), s.hits...), nil
}

func (s *testSemanticStore) Upsert(_ context.Context, record Record) error {
	s.records = append(s.records, record)
	return nil
}

func scopeKey(scope Scope) string { return scope.UserID + "/" + scope.ConversationID }

func TestReplyAssemblesHybridContextAndPromotesImportantTurn(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	scope := Scope{UserID: "user-1", ConversationID: "chat-1"}
	working := newTestWorkingStore()
	working.turns[scopeKey(scope)] = []Turn{{
		ID: "previous", User: "I work in design.", Assistant: "Got it.", CreatedAt: time.Now().Add(-time.Minute),
	}}
	semantic := &testSemanticStore{hits: []SearchHit{{ID: "fact-1", Text: "The user's preferred format is concise.", Score: 0.95}}}
	var gotPrompt []Message
	model := CompleteFunc(func(_ context.Context, messages []Message) (string, error) {
		gotPrompt = append([]Message(nil), messages...)
		return "Understood.", nil
	})
	service, err := NewService(Config{
		SystemPrompt:        "Be helpful.",
		RecentTurns:         4,
		RecallTopK:          3,
		ImportanceThreshold: 0.75,
		Scorer: ImportanceScoreFunc(func(context.Context, string) (float64, error) {
			return 0.8, nil
		}),
	}, working, semantic, model)
	ktest.RequireNoError(t, err)

	response, err := service.Reply(ctx, Request{Scope: scope, UserMessage: "What format should you use?"})
	ktest.RequireNoError(t, err)
	ktest.RequireEqual(t, response.Assistant, "Understood.")
	ktest.RequireCondition(t, response.Remembered, "important turn was not promoted")
	ktest.RequireEqual(t, len(response.Recalled), 1)
	ktest.RequireEqual(t, semantic.searched, scope)
	ktest.RequireEqual(t, len(gotPrompt), 5)
	ktest.RequireEqual(t, gotPrompt[0], Message{Role: "system", Content: "Be helpful."})
	ktest.RequireEqual(t, gotPrompt[1].Role, "system")
	ktest.RequireStringContains(t, gotPrompt[1].Content, "untrusted reference data")
	ktest.RequireEqual(t, gotPrompt[2], Message{Role: "user", Content: "I work in design."})
	ktest.RequireEqual(t, gotPrompt[3], Message{Role: "assistant", Content: "Got it."})
	ktest.RequireEqual(t, gotPrompt[4], Message{Role: "user", Content: "What format should you use?"})
	ktest.RequireEqual(t, len(working.turns[scopeKey(scope)]), 2)
	ktest.RequireEqual(t, working.turns[scopeKey(scope)][1].Assistant, "Understood.")
	ktest.RequireEqual(t, len(semantic.records), 1)
	ktest.RequireEqual(t, semantic.records[0].Reason, "importance-threshold")
	ktest.RequireEqual(t, semantic.records[0].ID, response.TurnID)
}

func TestReplyExplicitRememberBypassesScorer(t *testing.T) {
	t.Parallel()
	working := newTestWorkingStore()
	semantic := &testSemanticStore{}
	service, err := NewService(Config{Scorer: ImportanceScoreFunc(func(context.Context, string) (float64, error) {
		return 0, errors.New("must not run")
	})}, working, semantic, CompleteFunc(func(context.Context, []Message) (string, error) { return "saved", nil }))
	ktest.RequireNoError(t, err)

	response, err := service.Reply(context.Background(), Request{
		Scope: Scope{UserID: "u", ConversationID: "c"}, UserMessage: "Keep this in mind.", Remember: true,
	})
	ktest.RequireNoError(t, err)
	ktest.RequireCondition(t, response.Remembered, "explicitly retained turn was not promoted")
	ktest.RequireEqual(t, response.Importance, 1.0)
	ktest.RequireEqual(t, semantic.records[0].Reason, "explicit")
}

func TestReplyDoesNotPersistWhenCompletionFails(t *testing.T) {
	t.Parallel()
	working := newTestWorkingStore()
	service, err := NewService(Config{}, working, &testSemanticStore{}, CompleteFunc(func(context.Context, []Message) (string, error) {
		return "", errors.New("model unavailable")
	}))
	ktest.RequireNoError(t, err)

	_, err = service.Reply(context.Background(), Request{Scope: Scope{UserID: "u", ConversationID: "c"}, UserMessage: "hello"})
	ktest.RequireCondition(t, err != nil, "expected completion error")
	ktest.RequireEqual(t, len(working.turns), 0)
}

func TestPromoteOlderDeletesOnlySuccessfullyPromotedTurns(t *testing.T) {
	t.Parallel()
	now := time.Now().UTC()
	scope := Scope{UserID: "u", ConversationID: "c"}
	working := newTestWorkingStore()
	working.turns[scopeKey(scope)] = []Turn{
		{ID: "old-1", User: "first", Assistant: "answer", CreatedAt: now.Add(-48 * time.Hour)},
		{ID: "old-2", User: "second", Assistant: "answer", CreatedAt: now.Add(-24 * time.Hour)},
		{ID: "new", User: "recent", Assistant: "answer", CreatedAt: now},
	}
	semantic := &testSemanticStore{}
	service, err := NewService(Config{}, working, semantic, CompleteFunc(func(context.Context, []Message) (string, error) { return "", nil }))
	ktest.RequireNoError(t, err)

	count, err := service.PromoteOlder(context.Background(), scope, now.Add(-time.Hour), 1)
	ktest.RequireNoError(t, err)
	ktest.RequireEqual(t, count, 1)
	ktest.RequireEqual(t, len(semantic.records), 1)
	ktest.RequireEqual(t, semantic.records[0].ID, "old-1")
	remaining := working.turns[scopeKey(scope)]
	ktest.RequireEqual(t, len(remaining), 2)
	ktest.RequireEqual(t, remaining[0].ID, "old-2")
	ktest.RequireEqual(t, remaining[1].ID, "new")
}

func TestReplyRequiresCompleteScope(t *testing.T) {
	t.Parallel()
	service, err := NewService(Config{}, newTestWorkingStore(), &testSemanticStore{}, CompleteFunc(func(context.Context, []Message) (string, error) { return "", nil }))
	ktest.RequireNoError(t, err)
	_, err = service.Reply(context.Background(), Request{Scope: Scope{UserID: "u"}, UserMessage: "hello"})
	ktest.RequireCondition(t, err != nil, "incomplete scope should fail")
}
