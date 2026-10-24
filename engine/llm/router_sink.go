package llm

import (
	"context"
	"sync/atomic"

	"github.com/nexssp/kernel/observe"
)

// RouterSink counts MoE router events. It is a plain observe.Sink so it
// can be attached with .AnyHook(observe.Hook(sink)) to any action set.
type RouterSink struct {
	executed atomic.Uint64
	retries  atomic.Uint64
	errors   atomic.Uint64
}

func NewRouterSink() *RouterSink { return &RouterSink{} }

func (s *RouterSink) Emit(_ context.Context, ev observe.Event) {
	switch ev.Kind {
	case observe.KindSuccess:
		s.executed.Add(1)
	case observe.KindRetry:
		s.retries.Add(1)
	case observe.KindError:
		s.errors.Add(1)
	}
}

type SinkStats struct {
	Executed uint64 `json:"executed"`
	Retries  uint64 `json:"retries"`
	Errors   uint64 `json:"errors"`
}

func (s *RouterSink) Stats() SinkStats {
	return SinkStats{
		Executed: s.executed.Load(),
		Retries:  s.retries.Load(),
		Errors:   s.errors.Load(),
	}
}
