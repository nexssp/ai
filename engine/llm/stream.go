package llm

import (
	"context"
	"time"
)

type streamSinkContextKey struct{}

// WithStreamSink attaches a run-scoped sink for token events.
func WithStreamSink(ctx context.Context, sink StreamSink) context.Context {
	return context.WithValue(ctx, streamSinkContextKey{}, sink)
}

// StreamSinkFromContext returns the run-scoped sink, if one was attached.
func StreamSinkFromContext(ctx context.Context) (StreamSink, bool) {
	sink, ok := ctx.Value(streamSinkContextKey{}).(StreamSink)
	return sink, ok && sink != nil
}

type StreamEventKind uint8

const (
	EventRunStarted StreamEventKind = iota
	EventTurnStarted
	EventCompletionStarted
	EventTokenDelta
	EventCompletionFinished
	EventToolCallStarted
	EventToolCallFinished
	EventTurnFinished
	EventRunFinished
	EventError
)

func (k StreamEventKind) String() string {
	switch k {
	case EventRunStarted:
		return "run.started"
	case EventTurnStarted:
		return "turn.started"
	case EventCompletionStarted:
		return "completion.started"
	case EventTokenDelta:
		return "token.delta"
	case EventCompletionFinished:
		return "completion.finished"
	case EventToolCallStarted:
		return "tool_call.started"
	case EventToolCallFinished:
		return "tool_call.finished"
	case EventTurnFinished:
		return "turn.finished"
	case EventRunFinished:
		return "run.finished"
	case EventError:
		return "error"
	default:
		return "unknown"
	}
}

// StreamEvent represents an ephemeral lifecycle or streaming event.
type StreamEvent struct {
	Sequence   uint64          `json:"sequence"`
	Time       time.Time       `json:"time"`
	Kind       StreamEventKind `json:"kind"`
	RunID      string          `json:"run_id,omitempty"`
	SessionID  string          `json:"session_id,omitempty"`
	Turn       int             `json:"turn,omitempty"`
	Action     string          `json:"action,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	ToolName   string          `json:"tool_name,omitempty"`
	Delta      string          `json:"delta,omitempty"`
	ToolResult string          `json:"tool_result,omitempty"`
	Response   *Response       `json:"response,omitempty"`
	Error      error           `json:"error,omitempty"`
}

// StreamSink receives zero-allocation streaming events across run boundaries.
type StreamSink interface {
	Emit(context.Context, StreamEvent) error
}
