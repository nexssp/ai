package nexssflow

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/nexssp/transport/thttp"

	"github.com/nexssp/ai/engine/llm"
)

const (
	streamEventsEnv    = "NEXSS_AI_STREAM_EVENTS"
	streamEventPrefix  = "NEXSS_AI_STREAM_EVENT:"
	streamEventsStdout = "stdout"
)

var streamEventMu sync.Mutex

type lineStreamSink struct {
	writer io.Writer
}

func (sink lineStreamSink) Emit(ctx context.Context, event llm.StreamEvent) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if event.Kind != llm.EventTokenDelta {
		return nil
	}
	payload, err := json.Marshal(struct {
		Type  string `json:"type"`
		Delta string `json:"delta"`
	}{Type: event.Kind.String(), Delta: event.Delta})
	if err != nil {
		return fmt.Errorf("encode AI stream event: %w", err)
	}
	streamEventMu.Lock()
	defer streamEventMu.Unlock()
	if _, err := fmt.Fprintf(sink.writer, "%s%s\n", streamEventPrefix, payload); err != nil {
		return fmt.Errorf("write AI stream event: %w", err)
	}
	return nil
}

type transportStreamSink struct {
	writer thttp.StreamEventWriter
}

func (sink transportStreamSink) Emit(ctx context.Context, event llm.StreamEvent) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if event.Kind != llm.EventTokenDelta {
		return nil
	}
	return sink.writer.WriteEvent(ctx, thttp.StreamEvent{Type: event.Kind.String(), Delta: event.Delta})
}

func streamSinkFor(ctx context.Context) llm.StreamSink {
	if sink, ok := llm.StreamSinkFromContext(ctx); ok {
		return sink
	}
	if writer, ok := thttp.StreamEventWriterFromContext(ctx); ok {
		return transportStreamSink{writer: writer}
	}
	if os.Getenv(streamEventsEnv) == streamEventsStdout {
		return lineStreamSink{writer: os.Stdout}
	}
	return nil
}
