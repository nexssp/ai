package nexssflow

import (
	"context"
	"errors"
	"fmt"
	"net"
	"strings"
	"testing"
	"time"

	"github.com/nexssp/flow/native"
	"github.com/nexssp/flow/runner"
	"github.com/nexssp/kernel/action"
	"github.com/nexssp/kernel/xerr"

	"github.com/nexssp/ai/engine/llm"
)

func requireKind(t *testing.T, err error, want xerr.Kind) *xerr.AppError {
	t.Helper()
	if err == nil {
		t.Fatalf("expected %s error", want)
	}
	var app *xerr.AppError
	if !errors.As(err, &app) {
		t.Fatalf("error type %T (%v), want *xerr.AppError", err, err)
	}
	if app.Kind != want {
		t.Fatalf("error kind = %s, want %s: %v", app.Kind, want, err)
	}
	return app
}

func testAgents() []any {
	return []any{
		map[string]any{"name": "planner", "role": "planner", "instructions": "plan"},
		map[string]any{"name": "reviewer", "role": "reviewer", "instructions": "review"},
	}
}

func TestOptionsTimeoutValidation(t *testing.T) {
	for _, value := range []string{"nonsense", "0s", "-1s"} {
		t.Run(value, func(t *testing.T) {
			_, err := options(map[string]string{"timeout": value})
			ae := requireKind(t, err, xerr.KindValidation)
			if value == "nonsense" && !strings.Contains(ae.Message, "invalid timeout") {
				t.Fatalf("message lacks invalid timeout detail: %q", ae.Message)
			}
		})
	}
	got, err := options(map[string]string{"timeout": "150ms"})
	if err != nil || got.Timeout != 150*time.Millisecond {
		t.Fatalf("valid timeout: got %v, err %v", got.Timeout, err)
	}
	defaults, err := options(nil)
	if err != nil || defaults.Timeout != 60*time.Second {
		t.Fatalf("default timeout: got %v, err %v", defaults.Timeout, err)
	}
}

func TestOptionsRejectUnknownRequireOption(t *testing.T) {
	_, err := options(map[string]string{"defualt_model": "model-name"})
	ae := requireKind(t, err, xerr.KindValidation)
	if !strings.Contains(ae.Message, "unknown @require option \"defualt_model\"") {
		t.Fatalf("unexpected unknown-option message: %q", ae.Message)
	}
}

func TestBundleInvalidTimeoutReachesBothActions(t *testing.T) {
	bundle := Bundle(map[string]string{"timeout": "bad"})
	for _, act := range bundle.Libraries[0].Actions {
		req := map[string]any{"user": "hello", "goal": "hello", "agents": testAgents()}
		_, err := act.DoAny(context.Background(), req)
		if ae := requireKind(t, err, xerr.KindValidation); !strings.Contains(ae.Message, "invalid timeout") {
			t.Errorf("validation detail lost: %q", ae.Message)
		}
	}
}

func TestPromptValidatesUserBeforeBuildingProvider(t *testing.T) {
	builds := 0
	act := newPromptActionWith(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) {
		builds++
		return &fakeProvider{}, nil
	})
	for _, req := range []map[string]any{{}, {"user": "  "}} {
		_, err := act.DoAny(context.Background(), req)
		requireKind(t, err, xerr.KindValidation)
	}
	if builds != 0 {
		t.Fatalf("provider builder called %d times for invalid user", builds)
	}
}

func TestParseRoleAgentsValidationPaths(t *testing.T) {
	cases := []struct {
		name string
		raw  any
	}{
		{"wrong container", "agents"},
		{"too few", []any{map[string]any{"name": "a", "role": "r", "instructions": "i"}}},
		{"invalid element", []any{42, map[string]any{"name": "b", "role": "r", "instructions": "i"}}},
		{"missing fields", []any{map[string]any{}, map[string]any{"name": "b", "role": "r", "instructions": "i"}}},
		{"duplicate name", []any{map[string]any{"name": "same", "role": "r", "instructions": "i"}, map[string]any{"name": "same", "role": "s", "instructions": "j"}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parseRoleAgents(tc.raw)
			requireKind(t, err, xerr.KindValidation)
		})
	}
}

func TestTeamRequiresGoalBeforeProvider(t *testing.T) {
	builds := 0
	act := newTeamActionWith(GlobalOptions{}, func(GlobalOptions, LLMDecl) (llm.Provider, error) {
		builds++
		return &fakeProvider{}, nil
	})
	_, err := act.DoAny(context.Background(), map[string]any{"agents": testAgents()})
	requireKind(t, err, xerr.KindValidation)
	if builds != 0 {
		t.Fatalf("provider builder called %d times without goal", builds)
	}
}

type nilFakeProvider struct{ *fakeProvider }

type errorFakeProvider struct {
	*fakeProvider
	err error
}

func (p *errorFakeProvider) Complete(context.Context, llm.Request) (llm.Response, error) {
	return llm.Response{}, p.err
}

func actionFor(t *testing.T, name string, builder providerBuilder) action.AnyAction {
	t.Helper()
	if name == "ai.prompt" {
		return newPromptActionWith(GlobalOptions{}, builder)
	}
	return newTeamActionWith(GlobalOptions{}, builder)
}

func requestFor(name string) map[string]any {
	if name == "ai.prompt" {
		return map[string]any{"user": "hello"}
	}
	return map[string]any{"goal": "hello", "agents": testAgents()}
}

func TestActionsRejectNilBuildersAndProviders(t *testing.T) {
	for _, name := range []string{"ai.prompt", "ai.team"} {
		t.Run(name+" nil builder", func(t *testing.T) {
			_, err := actionFor(t, name, nil).DoAny(context.Background(), requestFor(name))
			requireKind(t, err, xerr.KindInternal)
		})
		t.Run(name+" nil provider", func(t *testing.T) {
			_, err := actionFor(t, name, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return nil, nil }).DoAny(context.Background(), requestFor(name))
			requireKind(t, err, xerr.KindInternal)
		})
		t.Run(name+" typed nil provider", func(t *testing.T) {
			var provider *nilFakeProvider
			_, err := actionFor(t, name, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return provider, nil }).DoAny(context.Background(), requestFor(name))
			requireKind(t, err, xerr.KindInternal)
		})
	}
}

func TestBuilderErrorsAreClassifiedAndPreserved(t *testing.T) {
	plain := errors.New("builder bug")
	_, err := actionFor(t, "ai.prompt", func(GlobalOptions, LLMDecl) (llm.Provider, error) { return nil, plain }).DoAny(context.Background(), requestFor("ai.prompt"))
	requireKind(t, err, xerr.KindInternal)

	typed := xerr.Validation("bad provider config")
	typed.ValidationDetails = xerr.ValidationDetails{{Field: "base_url", Validation: "must be URL"}}
	for _, name := range []string{"ai.prompt", "ai.team"} {
		_, err := actionFor(t, name, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return nil, typed }).DoAny(context.Background(), requestFor(name))
		ae := requireKind(t, err, xerr.KindValidation)
		if !errors.Is(err, typed) || !strings.Contains(ae.Message, typed.Message) || len(ae.ValidationDetails) != 1 || ae.ValidationDetails[0].Field != "base_url" {
			t.Fatalf("%s did not preserve typed builder error: %#v", name, ae)
		}
		if name == "ai.team" && (!strings.Contains(ae.Message, `agent "planner"`) || typed.Message != "bad provider config") {
			t.Fatalf("team context absent or source error mutated: wrapped=%q original=%q", ae.Message, typed.Message)
		}
	}
}

func TestCompletionErrorsClassified(t *testing.T) {
	netErr := &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("offline")}
	cases := []struct {
		name string
		err  error
		kind xerr.Kind
	}{
		{"canceled", context.Canceled, xerr.KindCanceled},
		{"deadline", context.DeadlineExceeded, xerr.KindTimeout},
		{"network", netErr, xerr.KindUnavailable},
		{"unknown", errors.New("unexpected"), xerr.KindInternal},
	}
	for _, tc := range cases {
		for _, name := range []string{"ai.prompt", "ai.team"} {
			t.Run(tc.name+"/"+name, func(t *testing.T) {
				provider := &errorFakeProvider{fakeProvider: &fakeProvider{}, err: tc.err}
				_, err := actionFor(t, name, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return provider, nil }).DoAny(context.Background(), requestFor(name))
				ae := requireKind(t, err, tc.kind)
				if name == "ai.team" && !strings.Contains(ae.Message, `agent "planner"`) {
					t.Fatalf("team error lacks agent context: %q", ae.Message)
				}
			})
		}
	}
}

func TestTypedCompletionErrorAndUnknownProviderValidation(t *testing.T) {
	typed := xerr.Validation("provider rejected request")
	typed.ValidationDetails = xerr.ValidationDetails{{Field: "prompt", Validation: "required"}}
	for _, name := range []string{"ai.prompt", "ai.team"} {
		provider := &errorFakeProvider{fakeProvider: &fakeProvider{}, err: fmt.Errorf("wrapped typed: %w", typed)}
		_, err := actionFor(t, name, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return provider, nil }).DoAny(context.Background(), requestFor(name))
		ae := requireKind(t, err, xerr.KindValidation)
		if !errors.Is(err, typed) || !errors.As(err, new(*xerr.AppError)) || len(ae.ValidationDetails) != 1 || ae.ValidationDetails[0].Field != "prompt" {
			t.Fatalf("typed completion error lost in %s: %#v", name, ae)
		}
	}
	_, err := buildProvider(GlobalOptions{}, LLMDecl{Provider: "not-a-provider"})
	requireKind(t, err, xerr.KindBadRequest)
	_, err = buildProvider(GlobalOptions{}, LLMDecl{BaseURL: "not-a-url"})
	requireKind(t, err, xerr.KindValidation)
}

func TestCompletionErrorWrapKeepsCause(t *testing.T) {
	cause := fmt.Errorf("wrapped: %w", context.Canceled)
	err := completionError(cause)
	requireKind(t, err, xerr.KindCanceled)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("completion classification lost cause: %v", err)
	}
}

func TestAllTypedKindsSurviveBuilderAndCompletion(t *testing.T) {
	for _, kind := range xerr.AllKinds() {
		t.Run(string(kind), func(t *testing.T) {
			sentinel := errors.New("sentinel cause")
			original := &xerr.AppError{Kind: kind, Message: "typed failure", Cause: sentinel, ValidationDetails: xerr.ValidationDetails{{Field: "x", Validation: "invalid"}}}
			wrapped := fmt.Errorf("outer wrapper: %w", original)
			for _, source := range []string{"builder", "completion"} {
				for _, name := range []string{"ai.prompt", "ai.team"} {
					t.Run(source+"/"+name, func(t *testing.T) {
						var err error
						if source == "builder" {
							_, err = actionFor(t, name, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return nil, wrapped }).DoAny(context.Background(), requestFor(name))
						} else {
							provider := &errorFakeProvider{fakeProvider: &fakeProvider{}, err: wrapped}
							_, err = actionFor(t, name, func(GlobalOptions, LLMDecl) (llm.Provider, error) { return provider, nil }).DoAny(context.Background(), requestFor(name))
						}
						ae := requireKind(t, err, kind)
						var found *xerr.AppError
						if !errors.Is(err, sentinel) || !errors.Is(err, original) || !errors.Is(err, wrapped) || !errors.As(err, &found) || found.Kind != kind {
							t.Fatalf("typed chain not preserved: %v", err)
						}
						if len(ae.ValidationDetails) != 1 || ae.ValidationDetails[0].Field != "x" {
							t.Fatalf("validation details lost: %#v", ae.ValidationDetails)
						}
					})
				}
			}
		})
	}
}

func TestRunnerOnErrorSeesBundleTimeoutValidationForBothActions(t *testing.T) {
	cases := []struct {
		name   string
		source string
	}{
		{"ai.prompt", `ai.prompt @{ user: "hello" } ? on_error { xerr.KindValidation -> { handled: true, kind: "Validation" }, else -> error.reraise }`},
		{"ai.team", `ai.team @{ goal: "hello", agents: [{name: "planner", role: "planner", instructions: "plan"}, {name: "reviewer", role: "reviewer", instructions: "review"}] } ? on_error { xerr.KindValidation -> { handled: true, kind: "Validation" }, else -> error.reraise }`},
	}
	cfg, err := runner.BuildConfig(append(native.Bundles(), Bundle(map[string]string{"timeout": "bad"})))
	if err != nil {
		t.Fatalf("BuildConfig: %v", err)
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			execution, err := runner.Execute(context.Background(), cfg, tc.source, tc.name+".nflow", nil)
			if err != nil {
				t.Fatalf("runner.Execute: %v", err)
			}
			out, ok := execution.Output.(map[string]any)
			if !ok || out["handled"] != true || out["kind"] != "Validation" {
				t.Fatalf("on_error output = %#v", execution.Output)
			}
		})
	}
}
