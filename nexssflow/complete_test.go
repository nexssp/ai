package nexssflow

import (
	"testing"

	"github.com/nexssp/ai/engine/llm"
)

func TestPromptJSONModeIsOptIn(t *testing.T) {
	for _, tc := range []struct {
		name  string
		value any
		want  bool
	}{
		{"absent", nil, false},
		{"false", false, false},
		{"true", true, true},
		{"string is not a boolean", "true", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			provider := &fakeProvider{}
			act := newPromptActionWith(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return provider, nil })
			if _, err := act.DoAny(t.Context(), map[string]any{"user": "review", "json_mode": tc.value}); err != nil {
				t.Fatal(err)
			}
			if provider.request.JSONMode != tc.want {
				t.Fatalf("JSONMode = %v, want %v", provider.request.JSONMode, tc.want)
			}
		})
	}
}
