// Package gateway provides a configurable provider router for application-owned
// model routes. It deliberately leaves provider selection policy, budgets,
// persistence, retries, and protocol adapters to the host application.
package gateway

import (
	"context"
	"fmt"
	"iter"
	"reflect"
	"sync/atomic"
	"time"

	"github.com/nexssp/kernel/xerr"

	"github.com/nexssp/ai/engine/llm"
)

// Route is one explicitly configured destination. Name is the stable route key
// returned by Selector. Model, when non-empty, overrides Request.Model for this
// destination; otherwise the caller's requested model is preserved.
type Route struct {
	Name     string
	Model    string
	Provider llm.Provider
}

// Selector chooses one configured route by name for each request. It cannot add
// providers at runtime: Gateway rejects names that are absent from Config.Routes.
// Treat req as read-only; nested slices and pointers may alias caller data. A
// Selector must be safe for concurrent calls when its Gateway is shared.
type Selector func(context.Context, llm.Request) (string, error)

// Gate runs after route selection and before any provider call. Returning an
// error rejects the request and guarantees that the selected provider is not
// invoked. Treat req and route as read-only; nested values may alias gateway
// state. Gates must be safe for concurrent calls when the Gateway is shared.
type Gate func(context.Context, llm.Request, Route) error

// Outcome describes the terminal state reported to an Observer.
type Outcome string

const (
	OutcomeSucceeded Outcome = "succeeded"
	OutcomeFailed    Outcome = "failed"
	OutcomeRejected  Outcome = "rejected"
	OutcomeAbandoned Outcome = "abandoned"
)

// Event contains call metadata and usage, but never prompt or response content.
// For streams, usage fields are zero because llm.Provider.Stream does not expose
// terminal usage. Err is the selection, gate, provider, or gateway error;
// selector and gate errors may be wrapped with gateway context. Applications
// should avoid logging it blindly because provider errors may include upstream
// details.
type Event struct {
	Outcome            Outcome
	Route              string
	Provider           string
	Model              string
	CandidateID        string
	AttemptID          string
	RateCardID         string
	Stream             bool
	PromptTokens       int
	CachedPromptTokens int
	CompletionTokens   int
	Duration           time.Duration
	Err                error
}

// Observer receives one terminal event for a completed, rejected, or failed
// non-stream call. For streams it receives an event when the returned sequence
// is consumed to completion, fails, or is stopped early by its consumer. An
// unconsumed stream produces no terminal event. Observers are synchronous; a
// panic propagates to the caller, so observers must not panic. They must be safe
// for concurrent calls when the Gateway is shared.
type Observer func(context.Context, Event)

// Config defines every route plus the host's route selector, gates, and hooks.
// No route, model, fallback, or gate is selected implicitly.
type Config struct {
	// Name is the name returned by Gateway.Name. Empty selects "gateway".
	Name string
	// Routes is the closed set of destinations this gateway may invoke.
	Routes []Route
	// Select is required and returns the name of one route in Routes.
	Select Selector
	// Gates run in order after route selection and before provider dispatch.
	Gates []Gate
	// Observers run after a request reaches a terminal outcome. They return no
	// errors, but panics propagate. They are not an accounting boundary.
	Observers []Observer
}

// Gateway selects one configured provider route, applies gates, then delegates
// the request. Its configuration is immutable after construction and may be
// used concurrently provided callbacks and providers are concurrency-safe.
type Gateway struct {
	name      string
	routes    map[string]Route
	selectFn  Selector
	gates     []Gate
	observers []Observer
}

// New validates and copies the configuration. Route names must be unique and
// every configured route must have a provider.
func New(cfg Config) (*Gateway, error) {
	if cfg.Select == nil {
		return nil, xerr.Validation("gateway: selector is required")
	}
	if len(cfg.Routes) == 0 {
		return nil, xerr.Validation("gateway: at least one route is required")
	}

	routes := make(map[string]Route, len(cfg.Routes))
	for i, route := range cfg.Routes {
		if route.Name == "" {
			return nil, xerr.Validation(fmt.Sprintf("gateway: route %d has an empty name", i))
		}
		if isNilProvider(route.Provider) {
			return nil, xerr.Validation(fmt.Sprintf("gateway: route %q has no provider", route.Name))
		}
		if _, exists := routes[route.Name]; exists {
			return nil, xerr.Conflict(fmt.Sprintf("gateway: duplicate route name %q", route.Name))
		}
		routes[route.Name] = route
	}
	for i, gate := range cfg.Gates {
		if gate == nil {
			return nil, xerr.Validation(fmt.Sprintf("gateway: gate %d is nil", i))
		}
	}
	for i, observer := range cfg.Observers {
		if observer == nil {
			return nil, xerr.Validation(fmt.Sprintf("gateway: observer %d is nil", i))
		}
	}

	name := cfg.Name
	if name == "" {
		name = "gateway"
	}

	return &Gateway{
		name:      name,
		routes:    routes,
		selectFn:  cfg.Select,
		gates:     append([]Gate(nil), cfg.Gates...),
		observers: append([]Observer(nil), cfg.Observers...),
	}, nil
}

// Name implements llm.Provider.
func (g *Gateway) Name() string {
	if g == nil || g.name == "" {
		return "gateway"
	}
	return g.name
}

// Complete selects a configured route, runs every gate, then makes one provider call.
func (g *Gateway) Complete(ctx context.Context, req llm.Request) (llm.Response, error) {
	started := time.Now()
	route, routedReq, rejected, err := g.resolve(ctx, req)
	if err != nil {
		event := newEvent(routedReq, route, false)
		event.Err = err
		event.Outcome = OutcomeFailed
		if rejected {
			event.Outcome = OutcomeRejected
		}
		event.Duration = time.Since(started)
		g.observe(ctx, event)
		return llm.Response{}, err
	}

	response, callErr := route.Provider.Complete(ctx, routedReq)
	event := newEvent(routedReq, route, false)
	event.Outcome = OutcomeSucceeded
	if callErr != nil {
		event.Outcome = OutcomeFailed
	}
	event.Err = callErr
	if response.Model != "" {
		event.Model = response.Model
	}
	event.PromptTokens = response.PromptTokens
	event.CachedPromptTokens = response.CachedPromptTokens
	event.CompletionTokens = response.CompTokens
	event.Duration = time.Since(started)
	g.observe(ctx, event)

	return response, callErr
}

// Stream selects a configured route, runs every gate, then opens one provider
// stream. The sequence is single-use; an unconsumed sequence produces no event.
// The observer runs when it reaches a terminal state; if the consumer stops
// early, the outcome is OutcomeAbandoned. Stream events do not include usage.
func (g *Gateway) Stream(ctx context.Context, req llm.Request) (iter.Seq2[string, error], error) {
	started := time.Now()
	route, routedReq, rejected, err := g.resolve(ctx, req)
	if err != nil {
		event := newEvent(routedReq, route, true)
		event.Err = err
		event.Outcome = OutcomeFailed
		if rejected {
			event.Outcome = OutcomeRejected
		}
		event.Duration = time.Since(started)
		g.observe(ctx, event)
		return nil, err
	}

	sequence, err := route.Provider.Stream(ctx, routedReq)
	if err != nil {
		event := newEvent(routedReq, route, true)
		event.Outcome = OutcomeFailed
		event.Err = err
		event.Duration = time.Since(started)
		g.observe(ctx, event)
		return nil, err
	}
	if sequence == nil {
		err = xerr.Internal("gateway: provider returned a nil stream")
		event := newEvent(routedReq, route, true)
		event.Outcome = OutcomeFailed
		event.Err = err
		event.Duration = time.Since(started)
		g.observe(ctx, event)
		return nil, err
	}

	var consumed atomic.Bool
	return func(yield func(string, error) bool) {
		if !consumed.CompareAndSwap(false, true) {
			yield("", xerr.Conflict("gateway: stream sequence can only be consumed once"))
			return
		}
		event := newEvent(routedReq, route, true)
		event.Outcome = OutcomeAbandoned
		defer func() {
			event.Duration = time.Since(started)
			g.observe(ctx, event)
		}()

		for chunk, streamErr := range sequence {
			if streamErr != nil {
				event.Outcome = OutcomeFailed
				event.Err = streamErr
				yield(chunk, streamErr)
				return
			}
			if !yield(chunk, nil) {
				return
			}
		}
		event.Outcome = OutcomeSucceeded
	}, nil
}

func (g *Gateway) resolve(ctx context.Context, req llm.Request) (Route, llm.Request, bool, error) {
	if g == nil || g.selectFn == nil {
		return Route{}, req, false, xerr.Internal("gateway: gateway is not initialized")
	}

	name, err := g.selectFn(ctx, req)
	if err != nil {
		return Route{}, req, false, fmt.Errorf("gateway: select route: %w", err)
	}
	route, ok := g.routes[name]
	if !ok {
		return Route{}, req, false, xerr.NotFound(fmt.Sprintf("gateway: selector returned unknown route %q", name))
	}

	routedReq := req
	if route.Model != "" {
		if route.Model != req.Model && hasPricingMetadata(req) {
			return route, req, true, xerr.Conflict(fmt.Sprintf("gateway: refusing to change model on a priced request from %q to %q", req.Model, route.Model))
		}
		routedReq.Model = route.Model
	}
	for _, gate := range g.gates {
		if err := gate(ctx, routedReq, route); err != nil {
			return route, routedReq, true, fmt.Errorf("gateway: request rejected by gate: %w", err)
		}
	}

	return route, routedReq, false, nil
}

func hasPricingMetadata(req llm.Request) bool {
	return req.RateCardID != "" || req.ExpectedCurrency != nil ||
		req.EstimatedPromptTokens > 0 || req.EstimatedCachedPromptTokens > 0 || req.EstimatedCompletionTokens > 0
}

func isNilProvider(provider llm.Provider) bool {
	if provider == nil {
		return true
	}
	value := reflect.ValueOf(provider)
	kind := value.Kind()
	if kind == reflect.Chan || kind == reflect.Func || kind == reflect.Interface ||
		kind == reflect.Map || kind == reflect.Pointer || kind == reflect.Slice {
		return value.IsNil()
	}
	return false
}

func newEvent(req llm.Request, route Route, stream bool) Event {
	event := Event{
		Route:       route.Name,
		Model:       req.Model,
		CandidateID: req.CandidateID,
		AttemptID:   req.AttemptID,
		RateCardID:  req.RateCardID,
		Stream:      stream,
	}
	if route.Provider != nil {
		event.Provider = route.Provider.Name()
	}
	return event
}

func (g *Gateway) observe(ctx context.Context, event Event) {
	if g == nil {
		return
	}
	for _, observer := range g.observers {
		observer(ctx, event)
	}
}
