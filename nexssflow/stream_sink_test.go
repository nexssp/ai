package nexssflow

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/nexssp/ai/engine/llm"
)

func TestLineStreamSinkWritesTokenDeltaAsNDJSON(t *testing.T) {
	var output bytes.Buffer
	sink := lineStreamSink{writer: &output}
	if err := sink.Emit(context.Background(), llm.StreamEvent{Kind: llm.EventTokenDelta, Delta: "first\nsecond"}); err != nil {
		t.Fatal(err)
	}

	line := strings.TrimSpace(output.String())
	if !strings.HasPrefix(line, streamEventPrefix) {
		t.Fatalf("stream line = %q, want prefix %q", line, streamEventPrefix)
	}
	var event struct {
		Type  string `json:"type"`
		Delta string `json:"delta"`
	}
	if err := json.Unmarshal([]byte(strings.TrimPrefix(line, streamEventPrefix)), &event); err != nil {
		t.Fatalf("decode stream event: %v", err)
	}
	if event.Type != "token.delta" || event.Delta != "first\nsecond" {
		t.Fatalf("event = %+v, want token.delta with exact delta", event)
	}
}

func TestLineStreamSinkIgnoresNonTokenEvents(t *testing.T) {
	var output bytes.Buffer
	sink := lineStreamSink{writer: &output}
	if err := sink.Emit(context.Background(), llm.StreamEvent{Kind: llm.EventCompletionFinished}); err != nil {
		t.Fatal(err)
	}
	if output.Len() != 0 {
		t.Fatalf("non-token event wrote %q", output.String())
	}
}
