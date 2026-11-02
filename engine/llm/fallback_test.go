package llm

import (
	"context"
	"errors"
	"iter"
	"testing"

	"github.com/nexssp/kernel/xerr"
)

type fallbackTestProvider struct {
	name        string
	completeErr error
	streamErr   error
	calls       int
}

func (p *fallbackTestProvider) Name() string { return p.name }

func (p *fallbackTestProvider) Complete(context.Context, Request) (Response, error) {
	p.calls++
	return Response{Content: p.name}, p.completeErr
}

func (p *fallbackTestProvider) Stream(context.Context, Request) (iter.Seq2[string, error], error) {
	p.calls++
	return func(func(string, error) bool) {}, p.streamErr
}

func TestFallbackDefaultRetriesOnlyTransientProviderKinds(t *testing.T) {
	cases := []struct {
		kind         xerr.Kind
		wantFallback bool
	}{
		{xerr.KindUnavailable, true},
		{xerr.KindTooManyRequests, true},
		{xerr.KindRateLimit, true},
		{xerr.KindTimeout, true},
		{xerr.KindCircuitBreaker, true},
		{xerr.KindBadRequest, false},
		{xerr.KindUnauthorized, false},
		{xerr.KindForbidden, false},
		{xerr.KindConflict, false},
		{xerr.KindValidation, false},
		{xerr.KindInternal, false},
		{xerr.KindCanceled, false},
	}
	for _, tc := range cases {
		t.Run(string(tc.kind), func(t *testing.T) {
			primary := &fallbackTestProvider{name: "primary", completeErr: &xerr.AppError{Kind: tc.kind, Message: "provider failure"}}
			secondary := &fallbackTestProvider{name: "secondary"}
			provider := NewFallbackProvider(primary, secondary, FallbackConfig{})
			response, err := provider.Complete(context.Background(), Request{})
			if tc.wantFallback {
				if err != nil || response.Content != "secondary" || secondary.calls != 1 {
					t.Fatalf("response=%+v err=%v secondary calls=%d, want successful fallback", response, err, secondary.calls)
				}
			} else {
				if secondary.calls != 0 {
					t.Fatalf("secondary calls=%d for non-transient %s", secondary.calls, tc.kind)
				}
				if got := xerr.KindFrom(err); got != tc.kind {
					t.Fatalf("error kind=%s, want original %s: %v", got, tc.kind, err)
				}
			}
		})
	}
}

func TestFallbackCustomPredicateCannotRetryBudgetRejections(t *testing.T) {
	primary := &fallbackTestProvider{name: "primary", completeErr: xerr.Validation("budget preflight rejected")}
	secondary := &fallbackTestProvider{name: "secondary"}
	provider := NewFallbackProvider(primary, secondary, FallbackConfig{
		ShouldFallback: func(error) bool { return true },
	})
	_, err := provider.Complete(context.Background(), Request{})
	if got := xerr.KindFrom(err); got != xerr.KindValidation {
		t.Fatalf("budget rejection kind=%s, want Validation: %v", got, err)
	}
	if secondary.calls != 0 {
		t.Fatalf("secondary called %d times after budget preflight rejection", secondary.calls)
	}
}

func TestFallbackGateFailureIsTerminalAndPreservesCause(t *testing.T) {
	primary := &fallbackTestProvider{name: "primary", completeErr: xerr.Unavailable("primary temporarily unavailable")}
	secondary := &fallbackTestProvider{name: "secondary"}
	gateErr := xerr.Timeout("authorization service timed out")
	provider := NewFallbackProvider(primary, secondary, FallbackConfig{
		OnFallback: func(context.Context, FallbackInfo) error { return gateErr },
	})
	_, err := provider.Complete(context.Background(), Request{})
	if got := xerr.KindFrom(err); got != xerr.KindValidation {
		t.Fatalf("fallback-gate kind=%s, want terminal Validation: %v", got, err)
	}
	if !errors.Is(err, gateErr) || secondary.calls != 0 {
		t.Fatalf("gate cause not preserved or secondary called: err=%v calls=%d", err, secondary.calls)
	}
}

func TestFallbackFailurePreservesSecondaryClassification(t *testing.T) {
	primaryErr := xerr.Unavailable("primary outage")
	secondaryErr := xerr.Unauthorized("secondary key invalid")
	primary := &fallbackTestProvider{name: "primary", completeErr: primaryErr}
	secondary := &fallbackTestProvider{name: "secondary", completeErr: secondaryErr}
	provider := NewFallbackProvider(primary, secondary, FallbackConfig{})
	_, err := provider.Complete(context.Background(), Request{})
	if got := xerr.KindFrom(err); got != xerr.KindUnauthorized {
		t.Fatalf("final error kind=%s, want secondary Unauthorized: %v", got, err)
	}
	var fallbackErr *FallbackError
	if !errors.As(err, &fallbackErr) || !errors.Is(err, primaryErr) || !errors.Is(err, secondaryErr) {
		t.Fatalf("fallback causes were not preserved: %v", err)
	}
}
