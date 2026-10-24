// ai/llm/observe.go
package llm

import (
	"context"
	"fmt"
	"iter"
	"time"
)

// ProviderTrace is the callback invoked by ObservableProvider around
// every completion attempt.
//
//	kind       one of "start", "finish", "warn", "error"
//	provider   the provider's Name() (e.g. "deepseek")
//	model      the resolved model name; "auto" until known, then the
//	           server-reported name from the response
//	in, out    token counts (0 for "start" and "error")
//	costMicro  cost in the currency's micro units (0 when unknown)
//	dur        wall-clock time of the call (0 for "start")
//	err        non-nil for "error" and "warn"
type ProviderTrace func(
	kind, provider, model string,
	in, out int,
	costMicro int64,
	dur time.Duration,
	err error,
)

// ObservableProvider wraps any Provider and fires a ProviderTrace
// around every Complete call. Stream is not traced; use the runner's
// stream events for streaming observability.
type ObservableProvider struct {
	inner Provider
	emit  ProviderTrace
}

// NewObservableProvider returns inner wrapped with emit. If either
// inner or emit is nil the function returns inner unchanged, so
// callers can chain it unconditionally.
func NewObservableProvider(inner Provider, emit ProviderTrace) Provider {
	if inner == nil || emit == nil {
		return inner
	}

	return &ObservableProvider{inner: inner, emit: emit}
}

func (p *ObservableProvider) Name() string { return p.inner.Name() }

func (p *ObservableProvider) Complete(ctx context.Context, req Request) (Response, error) {
	model := req.Model
	if model == "" {
		model = "auto"
	}

	p.emit("start", p.inner.Name(), model, 0, 0, 0, 0, nil)

	start := time.Now()
	res, err := p.inner.Complete(ctx, req)
	dur := time.Since(start)

	if err != nil {
		p.emit("error", p.inner.Name(), model, 0, 0, 0, dur, err)

		return res, err
	}

	// res.CostMicros() returns cost.Micro; the callback takes int64.
	// Cast at the boundary so this file does not need nexssp/cost.
	p.emit("finish", p.inner.Name(), res.Model,
		res.PromptTokens, res.CompTokens,
		int64(res.CostMicros()), dur, nil)

	// Truncation is silent at the HTTP level but fatal downstream: the
	// JSON decoder sees "unexpected end of JSON input" and the operator
	// has no idea the model was cut off. Surface it immediately.
	if res.Truncated() {
		p.emit("warn", p.inner.Name(), res.Model,
			res.PromptTokens, res.CompTokens,
			int64(res.CostMicros()), dur,
			fmt.Errorf("response truncated (finish_reason=length); raise @config:max_tokens"))
	}

	return res, nil
}

func (p *ObservableProvider) Stream(ctx context.Context, req Request) (iter.Seq2[string, error], error) {
	return p.inner.Stream(ctx, req)
}
