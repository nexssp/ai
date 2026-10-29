package memory

import (
	"context"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/nexssp/kernel/xtest/ktest"
	"github.com/redis/go-redis/v9"
)

func TestRedisWorkingStoreBoundsHistoryAndIsolatesScopes(t *testing.T) {
	t.Parallel()
	server := miniredis.RunT(t)
	client := redis.NewClient(&redis.Options{Addr: server.Addr()})
	t.Cleanup(func() { _ = client.Close() })
	store, err := NewRedisWorkingStore(client, RedisConfig{Prefix: "test:memory", MaxTurns: 2})
	ktest.RequireNoError(t, err)
	ctx := context.Background()
	first := Scope{UserID: "user-a", ConversationID: "chat"}
	second := Scope{UserID: "user-b", ConversationID: "chat"}
	base := time.Now().UTC()

	for i, id := range []string{"turn-1", "turn-2", "turn-3"} {
		err := store.Append(ctx, first, Turn{ID: id, User: id, Assistant: "ok", CreatedAt: base.Add(time.Duration(i) * time.Minute)})
		ktest.RequireNoError(t, err)
	}
	ktest.RequireNoError(t, store.Append(ctx, second, Turn{ID: "other", User: "private", Assistant: "ok", CreatedAt: base}))

	got, err := store.Recent(ctx, first, 10)
	ktest.RequireNoError(t, err)
	ktest.RequireEqual(t, len(got), 2)
	ktest.RequireEqual(t, got[0].ID, "turn-2")
	ktest.RequireEqual(t, got[1].ID, "turn-3")
	other, err := store.Recent(ctx, second, 10)
	ktest.RequireNoError(t, err)
	ktest.RequireEqual(t, len(other), 1)
	ktest.RequireEqual(t, other[0].ID, "other")
}

func TestRedisWorkingStoreOlderAndDeleteSpecificTurns(t *testing.T) {
	t.Parallel()
	server := miniredis.RunT(t)
	client := redis.NewClient(&redis.Options{Addr: server.Addr()})
	t.Cleanup(func() { _ = client.Close() })
	store, err := NewRedisWorkingStore(client, RedisConfig{MaxTurns: 10})
	ktest.RequireNoError(t, err)
	ctx := context.Background()
	scope := Scope{UserID: "u", ConversationID: "c"}
	now := time.Now().UTC()
	for _, turn := range []Turn{
		{ID: "old", User: "old", CreatedAt: now.Add(-time.Hour)},
		{ID: "new", User: "new", CreatedAt: now},
	} {
		ktest.RequireNoError(t, store.Append(ctx, scope, turn))
	}

	old, err := store.Older(ctx, scope, now.Add(-time.Minute), 10)
	ktest.RequireNoError(t, err)
	ktest.RequireEqual(t, len(old), 1)
	ktest.RequireEqual(t, old[0].ID, "old")
	ktest.RequireNoError(t, store.Delete(ctx, scope, []string{"old"}))
	remaining, err := store.Recent(ctx, scope, 10)
	ktest.RequireNoError(t, err)
	ktest.RequireEqual(t, len(remaining), 1)
	ktest.RequireEqual(t, remaining[0].ID, "new")
}
