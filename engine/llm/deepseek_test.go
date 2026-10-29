package llm

import (
	"strings"
	"testing"
)

func TestDeepSeekStreamFilterChunkBoundaries(t *testing.T) {
	tests := []struct {
		name   string
		chunks []string
		want   string
	}{
		{
			name:   "split thought tags",
			chunks: []string{"before<th", "ink>hidden </th", "ink>after"},
			want:   "beforeafter",
		},
		{
			name:   "unfinished opening tag remains visible",
			chunks: []string{"visible<th"},
			want:   "visible<th",
		},
		{
			name:   "unfinished thought remains hidden",
			chunks: []string{"<think>hidden</th"},
			want:   "",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var filter deepSeekStreamFilter
			var output strings.Builder
			yield := func(chunk string, err error) bool {
				if err != nil {
					t.Fatalf("unexpected stream error: %v", err)
				}
				output.WriteString(chunk)
				return true
			}

			for _, chunk := range test.chunks {
				if !filter.consume(chunk, yield) {
					t.Fatal("stream filter stopped unexpectedly")
				}
			}
			filter.flush(yield)

			if got := output.String(); got != test.want {
				t.Fatalf("filtered stream = %q, want %q", got, test.want)
			}
		})
	}
}
