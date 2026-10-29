package memory

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"time"

	"github.com/nexssp/kernel/xerr"
	"github.com/redis/go-redis/v9"
)

const (
	defaultRedisPrefix   = "nflow:memory:working"
	defaultRedisMaxTurns = 256
)

// RedisConfig controls Redis key names, the maximum turns retained per scope,
// and optional expiration. A positive MaxTurns is always enforced.
type RedisConfig struct {
	Prefix   string
	MaxTurns int
	TTL      time.Duration
}

// RedisWorkingStore stores completed turns in one bounded Redis list per scope.
type RedisWorkingStore struct {
	client   redis.UniversalClient
	prefix   string
	maxTurns int
	ttl      time.Duration
}

// NewRedisWorkingStore creates a Redis-backed working store. It does not take
// ownership of the client; the caller remains responsible for closing it.
func NewRedisWorkingStore(client redis.UniversalClient, config RedisConfig) (*RedisWorkingStore, error) {
	if client == nil {
		return nil, xerr.Validation("memory/redis: Redis client is required")
	}
	if config.Prefix == "" {
		config.Prefix = defaultRedisPrefix
	}
	if strings.TrimSpace(config.Prefix) != config.Prefix || strings.ContainsAny(config.Prefix, " \t\r\n") {
		return nil, xerr.Validation("memory/redis: prefix must not contain whitespace")
	}
	if config.MaxTurns < 0 {
		return nil, xerr.Validation("memory/redis: maximum turns cannot be negative")
	}
	if config.MaxTurns == 0 {
		config.MaxTurns = defaultRedisMaxTurns
	}
	if config.TTL < 0 {
		return nil, xerr.Validation("memory/redis: TTL cannot be negative")
	}
	return &RedisWorkingStore{
		client:   client,
		prefix:   config.Prefix,
		maxTurns: config.MaxTurns,
		ttl:      config.TTL,
	}, nil
}

// Recent returns the newest limit turns in chronological order.
func (s *RedisWorkingStore) Recent(ctx context.Context, scope Scope, limit int) ([]Turn, error) {
	if err := validateScope(scope); err != nil {
		return nil, err
	}
	if limit <= 0 {
		return []Turn{}, nil
	}
	values, err := s.client.LRange(ctx, s.key(scope), -int64(limit), -1).Result()
	if err != nil {
		return nil, xerr.Unavailable("memory/redis: read recent turns", err)
	}
	turns := make([]Turn, 0, len(values))
	for _, value := range values {
		turn, err := decodeTurn(value)
		if err != nil {
			return nil, err
		}
		turns = append(turns, turn)
	}
	return turns, nil
}

// Append adds a turn atomically and trims the oldest turns over the configured cap.
func (s *RedisWorkingStore) Append(ctx context.Context, scope Scope, turn Turn) error {
	if err := validateScope(scope); err != nil {
		return err
	}
	if turn.ID == "" {
		return xerr.Validation("memory/redis: turn ID is required")
	}
	if turn.CreatedAt.IsZero() {
		turn.CreatedAt = time.Now().UTC()
	}
	encoded, err := json.Marshal(turn)
	if err != nil {
		return xerr.Internal("memory/redis: encode turn", err)
	}
	key := s.key(scope)
	_, err = s.client.TxPipelined(ctx, func(pipe redis.Pipeliner) error {
		pipe.RPush(ctx, key, string(encoded))
		pipe.LTrim(ctx, key, -int64(s.maxTurns), -1)
		if s.ttl > 0 {
			pipe.Expire(ctx, key, s.ttl)
		}
		return nil
	})
	if err != nil {
		return xerr.Unavailable("memory/redis: append and trim turns", err)
	}
	return nil
}

// Older returns at most limit turns with timestamps strictly before before.
func (s *RedisWorkingStore) Older(ctx context.Context, scope Scope, before time.Time, limit int) ([]Turn, error) {
	if err := validateScope(scope); err != nil {
		return nil, err
	}
	if before.IsZero() || limit <= 0 {
		return []Turn{}, nil
	}
	values, err := s.client.LRange(ctx, s.key(scope), 0, -1).Result()
	if err != nil {
		return nil, xerr.Unavailable("memory/redis: scan old turns", err)
	}
	turns := make([]Turn, 0, min(limit, len(values)))
	for _, value := range values {
		turn, err := decodeTurn(value)
		if err != nil {
			return nil, err
		}
		if turn.CreatedAt.Before(before) {
			turns = append(turns, turn)
			if len(turns) == limit {
				break
			}
		}
	}
	return turns, nil
}

// Delete removes only turns whose IDs are supplied. It preserves concurrent
// appends and turns that were not successfully promoted.
func (s *RedisWorkingStore) Delete(ctx context.Context, scope Scope, turnIDs []string) error {
	if err := validateScope(scope); err != nil {
		return err
	}
	if len(turnIDs) == 0 {
		return nil
	}
	wanted := make(map[string]struct{}, len(turnIDs))
	for _, id := range turnIDs {
		if id != "" {
			wanted[id] = struct{}{}
		}
	}
	if len(wanted) == 0 {
		return nil
	}
	key := s.key(scope)
	values, err := s.client.LRange(ctx, key, 0, -1).Result()
	if err != nil {
		return xerr.Unavailable("memory/redis: inspect turns to delete", err)
	}
	encoded := make([]string, 0, len(wanted))
	found := make(map[string]struct{}, len(wanted))
	for _, value := range values {
		turn, decodeErr := decodeTurn(value)
		if decodeErr != nil {
			return decodeErr
		}
		if _, ok := wanted[turn.ID]; ok {
			if _, duplicate := found[turn.ID]; !duplicate {
				encoded = append(encoded, value)
				found[turn.ID] = struct{}{}
			}
		}
	}
	if len(encoded) == 0 {
		return nil
	}
	_, err = s.client.TxPipelined(ctx, func(pipe redis.Pipeliner) error {
		for _, value := range encoded {
			pipe.LRem(ctx, key, 1, value)
		}
		return nil
	})
	if err != nil {
		return xerr.Unavailable("memory/redis: delete promoted turns", err)
	}
	return nil
}

func (s *RedisWorkingStore) key(scope Scope) string {
	identity := scope.UserID + "\x00" + scope.ConversationID
	digest := sha256.Sum256([]byte(identity))
	return s.prefix + ":" + hex.EncodeToString(digest[:])
}

func decodeTurn(value string) (Turn, error) {
	var turn Turn
	if err := json.Unmarshal([]byte(value), &turn); err != nil {
		return Turn{}, xerr.Internal("memory/redis: decode stored turn", err)
	}
	if turn.ID == "" {
		return Turn{}, xerr.Validation("memory/redis: stored turn has no ID")
	}
	return turn, nil
}
