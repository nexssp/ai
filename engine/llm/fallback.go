package llm

import (
	"context"
	"errors"
	"fmt"
	"iter"

	"github.com/nexssp/kernel/xerr"
)

// FallbackInfo captures primary-to-secondary provider transition context.
type FallbackInfo struct {
	PrimaryRequest  Request
	FallbackRequest Request
	PrimaryError    error
}

// FallbackConfig controls provider failover conditions and request transforms.
type FallbackConfig struct {
	ShouldFallback func(error) bool
	Transform      func(Request) Request
	OnFallback     func(context.Context, FallbackInfo) error
}

func (c FallbackConfig) normalize() FallbackConfig {
	if c.ShouldFallback == nil {
		c.ShouldFallback = func(err error) bool {
			kind := xerr.KindFrom(err)

			return kind == xerr.KindUnavailable || kind == xerr.KindTooManyRequests || kind == xerr.KindRateLimit
		}
	}

	if c.Transform == nil {
		c.Transform = func(req Request) Request { return req }
	}

	return c
}

// FallbackError wraps primary and secondary errors for errors.Is / errors.As matching.
type FallbackError struct {
	Primary   error
	Secondary error
}

func (e *FallbackError) Error() string {
	return fmt.Sprintf("fallback failed: primary: %v; secondary: %v", e.Primary, e.Secondary)
}

func (e *FallbackError) Unwrap() []error {
	return []error{e.Primary, e.Secondary}
}

// FallbackProvider executes secondary provider on transient primary failure.
type FallbackProvider struct {
	primary   Provider
	secondary Provider
	cfg       FallbackConfig
}

func NewFallbackProvider(primary, secondary Provider, cfg FallbackConfig) *FallbackProvider {
	return &FallbackProvider{
		primary:   primary,
		secondary: secondary,
		cfg:       cfg.normalize(),
	}
}

func (p *FallbackProvider) Name() string {
	return p.primary.Name() + "+fallback"
}

func (p *FallbackProvider) Complete(ctx context.Context, req Request) (Response, error) {
	if p == nil || p.primary == nil || p.secondary == nil {
		return Response{}, errors.New("llm: fallback requires primary and secondary providers")
	}

	res, primaryErr := p.primary.Complete(ctx, req)
	if primaryErr == nil || !p.cfg.ShouldFallback(primaryErr) || ctx.Err() != nil {
		return res, primaryErr
	}

	fallbackReq := p.cfg.Transform(req)
	if p.cfg.OnFallback != nil {
		if err := p.cfg.OnFallback(ctx, FallbackInfo{
			PrimaryRequest:  req,
			FallbackRequest: fallbackReq,
			PrimaryError:    primaryErr,
		}); err != nil {
			return Response{}, err
		}
	}

	fallbackRes, fallbackErr := p.secondary.Complete(ctx, fallbackReq)
	if fallbackErr != nil {
		return fallbackRes, &FallbackError{Primary: primaryErr, Secondary: fallbackErr}
	}

	return fallbackRes, nil
}

func (p *FallbackProvider) Stream(ctx context.Context, req Request) (iter.Seq2[string, error], error) {
	if p == nil || p.primary == nil || p.secondary == nil {
		return nil, errors.New("llm: fallback requires primary and secondary providers")
	}

	stream, primaryErr := p.primary.Stream(ctx, req)
	if primaryErr == nil {
		return stream, nil
	}

	if !p.cfg.ShouldFallback(primaryErr) || ctx.Err() != nil {
		return nil, primaryErr
	}

	fallbackReq := p.cfg.Transform(req)
	if p.cfg.OnFallback != nil {
		if err := p.cfg.OnFallback(ctx, FallbackInfo{
			PrimaryRequest:  req,
			FallbackRequest: fallbackReq,
			PrimaryError:    primaryErr,
		}); err != nil {
			return nil, err
		}
	}

	return p.secondary.Stream(ctx, fallbackReq)
}
