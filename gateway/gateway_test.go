package gateway

import (
	"context"
	"errors"
	"iter"
	"strings"
	"testing"

	"github.com/nexssp/kernel/xerr"

	"github.com/nexssp/ai/engine/llm"
)

type fakeProvider struct {
	name          string
	response      llm.Response
	completeErr   error
	completeCalls int
	request       llm.Request
	stream        iter.Seq2[string, error]
	streamErr     error
	streamCalls   int
}

func (p *fakeProvider) Name() string { return p.name }

func (p *fakeProvider) Complete(_ context.Context, req llm.Request) (llm.Response, error) {
	p.completeCalls++
	p.request = req
	return p.response, p.completeErr
}

func (p *fakeProvider) Stream(_ context.Context, req llm.Request) (iter.Seq2[string, error], error) {
	p.streamCalls++
	p.request = req
	return p.stream, p.streamErr
}

func TestCompleteSelectsConfiguredRouteGatesAndObserves(t *testing.T) {
	provider := &fakeProvider{
		name: "provider-fast",
		response: llm.Response{
			Content:            "private answer",
			Model:              "fast-model-v2",
			PromptTokens:       11,
			CachedPromptTokens: 3,
			CompTokens:         7,
		},
	}
	var gotEvent Event
	gateCalled := false

	gateway, err := New(Config{
		Routes: []Route{{Name: "fast-route", Model: "fast-model-v2", Provider: provider}},
		Select: func(_ context.Context, req llm.Request) (string, error) {
			if req.Model != "fast-alias" {
				t.Fatalf("selector saw model %q", req.Model)
			}
			return "fast-route", nil
		},
		Gates: []Gate{func(_ context.Context, req llm.Request, route Route) error {
			gateCalled = true
			if req.AttemptID != "attempt-7" || req.Model != "fast-model-v2" || route.Name != "fast-route" {
				t.Fatalf("gate received unexpected request or route: %#v %#v", req, route)
			}
			return nil
		}},
		Observers: []Observer{func(_ context.Context, event Event) { gotEvent = event }},
	})
	if err != nil {
		t.Fatal(err)
	}

	response, err := gateway.Complete(context.Background(), llm.Request{
		Model:       "fast-alias",
		CandidateID: "candidate-2",
		AttemptID:   "attempt-7",
		Messages:    []llm.Message{{Role: llm.RoleUser, Content: "private prompt"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if response.Content != "private answer" || provider.completeCalls != 1 || !gateCalled {
		t.Fatalf("unexpected dispatch: response=%#v calls=%d gateCalled=%v", response, provider.completeCalls, gateCalled)
	}
	if provider.request.Model != "fast-model-v2" {
		t.Fatalf("provider model = %q, want configured model override", provider.request.Model)
	}
	if gotEvent.Outcome != OutcomeSucceeded || gotEvent.Route != "fast-route" || gotEvent.Provider != "provider-fast" {
		t.Fatalf("unexpected event identity/outcome: %#v", gotEvent)
	}
	if gotEvent.Model != "fast-model-v2" || gotEvent.PromptTokens != 11 || gotEvent.CachedPromptTokens != 3 || gotEvent.CompletionTokens != 7 {
		t.Fatalf("unexpected event usage: %#v", gotEvent)
	}
	if gotEvent.CandidateID != "candidate-2" || gotEvent.AttemptID != "attempt-7" {
		t.Fatalf("request metadata missing from event: %#v", gotEvent)
	}
	if gotEvent.Stream || gotEvent.Err != nil || gotEvent.Duration < 0 {
		t.Fatalf("unexpected event status: %#v", gotEvent)
	}
}

func TestGateRejectsBeforeProviderCall(t *testing.T) {
	wantErr := xerr.Forbidden("budget denied")
	provider := &fakeProvider{name: "restricted"}
	var gotEvent Event
	gateway, err := New(Config{
		Routes:    []Route{{Name: "restricted-route", Provider: provider}},
		Select:    func(context.Context, llm.Request) (string, error) { return "restricted-route", nil },
		Gates:     []Gate{func(context.Context, llm.Request, Route) error { return wantErr }},
		Observers: []Observer{func(_ context.Context, event Event) { gotEvent = event }},
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = gateway.Complete(context.Background(), llm.Request{})
	if !errors.Is(err, wantErr) {
		t.Fatalf("Complete() error = %v, want wrapped %v", err, wantErr)
	}
	if got := xerr.KindFrom(err); got != xerr.KindForbidden {
		t.Fatalf("Complete() error kind = %q, want %q", got, xerr.KindForbidden)
	}
	if provider.completeCalls != 0 {
		t.Fatalf("provider called %d times after gate rejection", provider.completeCalls)
	}
	if gotEvent.Outcome != OutcomeRejected || gotEvent.Route != "restricted-route" || !errors.Is(gotEvent.Err, wantErr) {
		t.Fatalf("unexpected rejection event: %#v", gotEvent)
	}
}

func TestSelectorErrorIsObserved(t *testing.T) {
	wantErr := xerr.Unavailable("no eligible model")
	var gotEvent Event
	gateway, err := New(Config{
		Routes:    []Route{{Name: "route", Provider: &fakeProvider{name: "provider"}}},
		Select:    func(context.Context, llm.Request) (string, error) { return "", wantErr },
		Observers: []Observer{func(_ context.Context, event Event) { gotEvent = event }},
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = gateway.Complete(context.Background(), llm.Request{Model: "unknown"})
	if !errors.Is(err, wantErr) {
		t.Fatalf("Complete() error = %v, want wrapped %v", err, wantErr)
	}
	if got := xerr.KindFrom(err); got != xerr.KindUnavailable {
		t.Fatalf("selector error kind = %q, want %q", got, xerr.KindUnavailable)
	}
	if gotEvent.Outcome != OutcomeFailed || gotEvent.Route != "" || !errors.Is(gotEvent.Err, wantErr) {
		t.Fatalf("unexpected selection failure event: %#v", gotEvent)
	}
}

func TestUnknownRouteIsRejectedWithoutDispatch(t *testing.T) {
	provider := &fakeProvider{name: "provider"}
	var gotEvent Event
	gateway, err := New(Config{
		Routes:    []Route{{Name: "configured", Provider: provider}},
		Select:    func(context.Context, llm.Request) (string, error) { return "not-configured", nil },
		Observers: []Observer{func(_ context.Context, event Event) { gotEvent = event }},
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = gateway.Complete(context.Background(), llm.Request{})
	if err == nil {
		t.Fatal("Complete() accepted an unconfigured route")
	}
	if got := xerr.KindFrom(err); got != xerr.KindNotFound {
		t.Fatalf("unknown-route error kind = %q, want %q", got, xerr.KindNotFound)
	}
	if provider.completeCalls != 0 || gotEvent.Outcome != OutcomeFailed {
		t.Fatalf("unknown route dispatched or was not observed: calls=%d event=%#v", provider.completeCalls, gotEvent)
	}
}

func TestStreamReportsCompletionAndEarlyStop(t *testing.T) {
	makeGateway := func(t *testing.T) (*Gateway, *fakeProvider, *[]Event) {
		t.Helper()
		provider := &fakeProvider{
			name: "streamer",
			stream: func(yield func(string, error) bool) {
				if !yield("one", nil) {
					return
				}
				yield("two", nil)
			},
		}
		events := make([]Event, 0, 1)
		gateway, err := New(Config{
			Routes:    []Route{{Name: "stream-route", Provider: provider}},
			Select:    func(context.Context, llm.Request) (string, error) { return "stream-route", nil },
			Observers: []Observer{func(_ context.Context, event Event) { events = append(events, event) }},
		})
		if err != nil {
			t.Fatal(err)
		}
		return gateway, provider, &events
	}

	t.Run("consumed", func(t *testing.T) {
		gateway, provider, events := makeGateway(t)
		sequence, err := gateway.Stream(context.Background(), llm.Request{Model: "streamer"})
		if err != nil {
			t.Fatal(err)
		}
		var chunks []string
		for chunk, streamErr := range sequence {
			if streamErr != nil {
				t.Fatal(streamErr)
			}
			chunks = append(chunks, chunk)
		}
		if provider.streamCalls != 1 || len(chunks) != 2 || chunks[0] != "one" || chunks[1] != "two" {
			t.Fatalf("unexpected stream: calls=%d chunks=%v", provider.streamCalls, chunks)
		}
		if len(*events) != 1 || (*events)[0].Outcome != OutcomeSucceeded || !(*events)[0].Stream {
			t.Fatalf("unexpected completion event: %#v", *events)
		}
		if (*events)[0].PromptTokens != 0 || (*events)[0].CachedPromptTokens != 0 || (*events)[0].CompletionTokens != 0 {
			t.Fatalf("stream event unexpectedly reports unavailable token usage: %#v", (*events)[0])
		}
	})

	t.Run("consumer stops early", func(t *testing.T) {
		gateway, _, events := makeGateway(t)
		sequence, err := gateway.Stream(context.Background(), llm.Request{})
		if err != nil {
			t.Fatal(err)
		}
		sequence(func(string, error) bool { return false })
		if len(*events) != 1 || (*events)[0].Outcome != OutcomeAbandoned || !(*events)[0].Stream {
			t.Fatalf("unexpected abandoned-stream event: %#v", *events)
		}
	})

	t.Run("single use", func(t *testing.T) {
		gateway, _, events := makeGateway(t)
		sequence, err := gateway.Stream(context.Background(), llm.Request{})
		if err != nil {
			t.Fatal(err)
		}
		sequence(func(string, error) bool { return true })
		var secondErr error
		sequence(func(_ string, err error) bool {
			secondErr = err
			return false
		})
		if secondErr == nil || xerr.KindFrom(secondErr) != xerr.KindConflict || !strings.Contains(secondErr.Error(), "gateway: stream sequence can only be consumed once") {
			t.Fatalf("second stream iteration error = %v", secondErr)
		}
		if len(*events) != 1 {
			t.Fatalf("observer events after repeated iteration = %d, want 1", len(*events))
		}
	})
}

func TestNewValidatesConfiguration(t *testing.T) {
	provider := &fakeProvider{name: "provider"}
	selector := func(context.Context, llm.Request) (string, error) { return "route", nil }
	var typedNilProvider *fakeProvider
	cases := []struct {
		name string
		cfg  Config
		want xerr.Kind
	}{
		{name: "missing selector", cfg: Config{}, want: xerr.KindValidation},
		{name: "empty route table", cfg: Config{Select: selector}, want: xerr.KindValidation},
		{name: "unnamed route", cfg: Config{Select: selector, Routes: []Route{{Provider: provider}}}, want: xerr.KindValidation},
		{name: "missing provider", cfg: Config{Select: selector, Routes: []Route{{Name: "route"}}}, want: xerr.KindValidation},
		{name: "typed nil provider", cfg: Config{Select: selector, Routes: []Route{{Name: "typed-nil", Provider: typedNilProvider}}}, want: xerr.KindValidation},
		{name: "duplicate route", cfg: Config{Select: selector, Routes: []Route{{Name: "route", Provider: provider}, {Name: "route", Provider: provider}}}, want: xerr.KindConflict},
		{name: "nil gate", cfg: Config{Select: selector, Routes: []Route{{Name: "route", Provider: provider}}, Gates: []Gate{nil}}, want: xerr.KindValidation},
		{name: "nil observer", cfg: Config{Select: selector, Routes: []Route{{Name: "route", Provider: provider}}, Observers: []Observer{nil}}, want: xerr.KindValidation},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			if _, err := New(test.cfg); err == nil {
				t.Fatal("New accepted invalid configuration")
			} else if got := xerr.KindFrom(err); got != test.want {
				t.Fatalf("New() error kind = %q, want %q (error: %v)", got, test.want, err)
			}
		})
	}
}

func TestProviderFailureIsObserved(t *testing.T) {
	wantErr := errors.New("upstream failed")
	provider := &fakeProvider{name: "provider", completeErr: wantErr}
	var gotEvent Event
	gateway, err := New(Config{
		Routes:    []Route{{Name: "route", Provider: provider}},
		Select:    func(context.Context, llm.Request) (string, error) { return "route", nil },
		Observers: []Observer{func(_ context.Context, event Event) { gotEvent = event }},
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = gateway.Complete(context.Background(), llm.Request{})
	if !errors.Is(err, wantErr) {
		t.Fatalf("Complete() error = %v, want %v", err, wantErr)
	}
	if gotEvent.Outcome != OutcomeFailed || !errors.Is(gotEvent.Err, wantErr) {
		t.Fatalf("unexpected provider failure event: %#v", gotEvent)
	}
	if gotEvent.Duration < 0 {
		t.Fatalf("negative duration: %v", gotEvent.Duration)
	}
}

func TestPricedRequestRejectsRouteModelOverride(t *testing.T) {
	provider := &fakeProvider{name: "provider"}
	var gotEvent Event
	gateway, err := New(Config{
		Routes:    []Route{{Name: "route", Model: "model-b", Provider: provider}},
		Select:    func(context.Context, llm.Request) (string, error) { return "route", nil },
		Observers: []Observer{func(_ context.Context, event Event) { gotEvent = event }},
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = gateway.Complete(context.Background(), llm.Request{Model: "model-a", RateCardID: "card-a"})
	if err == nil {
		t.Fatal("Complete() accepted a priced request routed to a different model")
	}
	if got := xerr.KindFrom(err); got != xerr.KindConflict {
		t.Fatalf("priced model override error kind = %q, want %q", got, xerr.KindConflict)
	}
	if provider.completeCalls != 0 {
		t.Fatalf("provider called %d times after priced model mismatch", provider.completeCalls)
	}
	if gotEvent.Outcome != OutcomeRejected || gotEvent.Model != "model-a" {
		t.Fatalf("unexpected model-mismatch event: %#v", gotEvent)
	}
}

func TestPricedRequestAllowsMatchingConfiguredModel(t *testing.T) {
	provider := &fakeProvider{name: "provider", response: llm.Response{Model: "model-a"}}
	gateway, err := New(Config{
		Routes: []Route{{Name: "route", Model: "model-a", Provider: provider}},
		Select: func(context.Context, llm.Request) (string, error) { return "route", nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = gateway.Complete(context.Background(), llm.Request{Model: "model-a", RateCardID: "card-a"})
	if err != nil {
		t.Fatalf("matching priced model was rejected: %v", err)
	}
	if provider.completeCalls != 1 {
		t.Fatalf("provider calls = %d, want 1", provider.completeCalls)
	}
}

func TestStreamReportsOpenAndIterationErrors(t *testing.T) {
	streamErr := errors.New("stream iteration failed")
	provider := &fakeProvider{
		name: "provider",
		stream: func(yield func(string, error) bool) {
			yield("partial", streamErr)
		},
	}
	var events []Event
	gateway, err := New(Config{
		Routes:    []Route{{Name: "route", Provider: provider}},
		Select:    func(context.Context, llm.Request) (string, error) { return "route", nil },
		Observers: []Observer{func(_ context.Context, event Event) { events = append(events, event) }},
	})
	if err != nil {
		t.Fatal(err)
	}
	sequence, err := gateway.Stream(context.Background(), llm.Request{})
	if err != nil {
		t.Fatal(err)
	}
	for _, nextErr := range sequence {
		if !errors.Is(nextErr, streamErr) {
			t.Fatalf("stream iteration error = %v, want %v", nextErr, streamErr)
		}
	}
	if len(events) != 1 || events[0].Outcome != OutcomeFailed || !errors.Is(events[0].Err, streamErr) {
		t.Fatalf("unexpected stream failure event: %#v", events)
	}

	provider.stream = nil
	sequence, err = gateway.Stream(context.Background(), llm.Request{})
	if err == nil || sequence != nil || xerr.KindFrom(err) != xerr.KindInternal {
		t.Fatalf("nil provider sequence result = %v, %v; want error and nil sequence", sequence, err)
	}
	if len(events) != 2 || events[1].Outcome != OutcomeFailed || events[1].Err == nil {
		t.Fatalf("nil provider sequence was not observed as failure: %#v", events)
	}
}
