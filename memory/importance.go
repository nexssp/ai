package memory

import (
	"context"
	"strings"
)

// KeywordScorer gives a configurable score when a message contains any configured
// retention phrase. It is a transparent baseline; production callers can inject
// a classifier or model-backed ImportanceScorer instead.
type KeywordScorer struct {
	Keywords   []string
	MatchScore float64
}

// Score returns MatchScore for a case-insensitive substring match, or zero.
func (s KeywordScorer) Score(_ context.Context, message string) (float64, error) {
	score := s.MatchScore
	if score == 0 {
		score = 0.9
	}
	message = strings.ToLower(message)
	for _, keyword := range s.Keywords {
		keyword = strings.ToLower(strings.TrimSpace(keyword))
		if keyword != "" && strings.Contains(message, keyword) {
			return score, nil
		}
	}
	return 0, nil
}

var defaultKeywords = []string{
	"remember",
	"my name is",
	"i prefer",
	"i like",
	"i dislike",
	"my timezone is",
	"i work at",
	"my role is",
}
