package nexssflow

import (
	"context"
	"iter"
	"os"
	"path/filepath"
	"testing"

	"github.com/nexssp/cost"
	"github.com/nexssp/flow/compiler"
	"github.com/nexssp/flow/extensions/require"
	"github.com/nexssp/flow/native"
	"github.com/nexssp/flow/runner"
	"github.com/nexssp/kernel/xerr"

	"github.com/nexssp/ai/engine/llm"
)

type routerPromptProvider struct {
	name  string
	model string
	err   error
	calls int
}

func (p *routerPromptProvider) Name() string { return p.name }

func (p *routerPromptProvider) Complete(_ context.Context, req llm.Request) (llm.Response, error) {
	p.calls++
	if p.err != nil {
		return llm.Response{}, p.err
	}
	return llm.Response{
		Model:        req.Model,
		Content:      "answer from " + req.Model,
		PromptTokens: 10,
		CompTokens:   3,
	}, nil
}

func (p *routerPromptProvider) Stream(context.Context, llm.Request) (iter.Seq2[string, error], error) {
	return nil, nil
}

type routerRun struct {
	execution runner.Execution
	err       error
	a         *routerPromptProvider
	b         *routerPromptProvider
	decision  *staticDecisionProvider
}

func runDecisionRouter(t *testing.T, choice string, providerAErr error, decisionCardID, candidateACardID string) routerRun {
	t.Helper()
	previousCatalog := llm.GlobalCatalog
	llm.GlobalCatalog = llm.NewPriceCatalog()
	t.Cleanup(func() { llm.GlobalCatalog = previousCatalog })
	if decisionCardID != "missing-decision-card" {
		registerRouterRateCard(t, "decision-model", decisionCardID)
	}
	if candidateACardID != "missing-candidate-card" {
		registerRouterRateCard(t, "candidate-model-a", candidateACardID)
	}
	registerRouterRateCard(t, "candidate-model-b", "candidate-card-b")

	path := filepath.Join("..", "nflows", "decision_route.nflow")
	source, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read decision router: %v", err)
	}
	_, meta, err := compiler.Preprocess(t.Context(), compiler.NewDirectiveTable(require.Directive), string(source), path)
	if err != nil {
		t.Fatalf("preprocess decision router: %v", err)
	}
	requirements, err := require.SourceRequirements(meta)
	if err != nil || len(requirements) != 1 || requirements[0].Import != "github.com/nexssp/ai" {
		t.Fatalf("router requirements = %+v, err = %v", requirements, err)
	}
	global, err := options(requirements[0].Options)
	if err != nil {
		t.Fatalf("parse router @require options: %v", err)
	}

	a := &routerPromptProvider{name: "fake-a", model: "candidate-model-a", err: providerAErr}
	b := &routerPromptProvider{name: "fake-b", model: "candidate-model-b"}
	decision := &staticDecisionProvider{response: llm.DecisionResponse{
		Model:   "decision-model",
		Answers: []llm.DecisionAnswer{{ID: "route", Type: llm.DecisionChoice, Choice: choice}},
	}}
	builder := func(_ GlobalOptions, decl LLMDecl) (llm.Provider, error) {
		switch decl.Model {
		case "candidate-model-a":
			return a, nil
		case "candidate-model-b":
			return b, nil
		default:
			return nil, xerr.Validation("unexpected model in router test: " + decl.Model)
		}
	}
	bundle := bundleWithBuilders(global, builder, func(_ GlobalOptions, _ DecisionDecl) (llm.DecisionProvider, error) {
		return decision, nil
	})
	bundle.Alias = requirements[0].Alias
	cfg, err := runner.BuildConfig(append(native.Bundles(), bundle))
	if err != nil {
		t.Fatalf("build router runner: %v", err)
	}
	input := map[string]any{
		"task":                       "Route the request",
		"request":                    "Resolve a test request",
		"decision_attempt_id":        "decision-attempt-1",
		"decision_provider":          "fake-decision",
		"decision_model":             "decision-model",
		"decision_rate_card_id":      decisionCardID,
		"decision_expected_currency": "USD",
		"decision_prompt_tokens":     10,
		"decision_max_output_tokens": 5,
		"candidate_a":                routerCandidate("candidate-a", "Candidate A", "candidate-attempt-a", "candidate-model-a", candidateACardID),
		"candidate_b":                routerCandidate("candidate-b", "Candidate B", "candidate-attempt-b", "candidate-model-b", "candidate-card-b"),
	}
	execution, runErr := runner.Execute(t.Context(), cfg, string(source), path, input)
	return routerRun{execution: execution, err: runErr, a: a, b: b, decision: decision}
}

func routerCandidate(id, description, attemptID, model, rateCardID string) map[string]any {
	return map[string]any{
		"id":          id,
		"description": description,
		"attempt": map[string]any{
			"candidate_id":      id,
			"attempt_id":        attemptID,
			"provider":          "fake",
			"model":             model,
			"rate_card_id":      rateCardID,
			"expected_currency": "USD",
			"prompt_tokens":     10,
			"completion_tokens": 5,
			"user":              "route this request",
		},
	}
}

func registerRouterRateCard(t *testing.T, model, rateCardID string) {
	t.Helper()
	if err := llm.GlobalCatalog.Register(model, llm.PriceCard{
		RateCardID: rateCardID,
		Currency:   cost.USD,
		Peak:       llm.Price{PromptPer1k: 1, CompPer1k: 2},
	}); err != nil {
		t.Fatalf("register test rate card %s/%s: %v", model, rateCardID, err)
	}
}

func TestDecisionRouteRunsSelectedCandidate(t *testing.T) {
	run := runDecisionRouter(t, "candidate-b", nil, "decision-card", "candidate-card-a")
	if run.err != nil {
		t.Fatalf("execute router: %v", run.err)
	}
	result, ok := run.execution.Output.(PromptResult)
	if !ok || result.CandidateID != "candidate-b" || result.AttemptID != "candidate-attempt-b" || result.Model != "candidate-model-b" {
		t.Fatalf("router output = %#v, want selected candidate B PromptResult", run.execution.Output)
	}
	if run.a.calls != 0 || run.b.calls != 1 || run.decision.calls != 1 {
		t.Fatalf("provider calls A/B/decision = %d/%d/%d, want 0/1/1", run.a.calls, run.b.calls, run.decision.calls)
	}
}

func TestDecisionRouteUsesFallbackOnlyForTransientProviderFailure(t *testing.T) {
	run := runDecisionRouter(t, "candidate-a", xerr.Unavailable("temporary outage"), "decision-card", "candidate-card-a")
	if run.err != nil {
		t.Fatalf("execute router: %v", run.err)
	}
	result, ok := run.execution.Output.(PromptResult)
	if !ok || result.CandidateID != "candidate-b" || result.AttemptID != "candidate-attempt-b" || result.Model != "candidate-model-b" {
		t.Fatalf("router output = %#v, want fallback candidate B PromptResult", run.execution.Output)
	}
	if run.a.calls != 1 || run.b.calls != 1 || run.decision.calls != 1 {
		t.Fatalf("provider calls A/B/decision = %d/%d/%d, want 1/1/1", run.a.calls, run.b.calls, run.decision.calls)
	}
}

func TestDecisionRouteDoesNotFallbackOnPermanentProviderFailure(t *testing.T) {
	run := runDecisionRouter(t, "candidate-a", xerr.Unauthorized("bad credential"), "decision-card", "candidate-card-a")
	if run.err == nil {
		t.Fatal("permanent provider failure should propagate")
	}
	if run.a.calls != 1 || run.b.calls != 0 || run.decision.calls != 1 {
		t.Fatalf("provider calls A/B/decision = %d/%d/%d, want 1/0/1", run.a.calls, run.b.calls, run.decision.calls)
	}
}

func TestDecisionRoutePreflightRejectionsDoNotCallOrFallback(t *testing.T) {
	t.Run("decision", func(t *testing.T) {
		run := runDecisionRouter(t, "candidate-a", nil, "missing-decision-card", "candidate-card-a")
		if run.err != nil {
			t.Fatalf("execute router: %v", run.err)
		}
		result, ok := run.execution.Output.(map[string]any)
		if !ok || result["status"] != "decision_preflight_rejected" {
			t.Fatalf("router output = %#v, want decision_preflight_rejected", run.execution.Output)
		}
		if run.decision.calls != 0 || run.a.calls != 0 || run.b.calls != 0 {
			t.Fatalf("provider calls A/B/decision = %d/%d/%d, want 0/0/0", run.a.calls, run.b.calls, run.decision.calls)
		}
	})
	t.Run("selected candidate", func(t *testing.T) {
		run := runDecisionRouter(t, "candidate-a", nil, "decision-card", "missing-candidate-card")
		if run.err != nil {
			t.Fatalf("execute router: %v", run.err)
		}
		result, ok := run.execution.Output.(map[string]any)
		if !ok || result["status"] != "primary_preflight_rejected" || result["fallback_invoked"] != false {
			t.Fatalf("router output = %#v, want rejected primary and no fallback", run.execution.Output)
		}
		if run.a.calls != 0 || run.b.calls != 0 || run.decision.calls != 1 {
			t.Fatalf("provider calls A/B/decision = %d/%d/%d, want 0/0/1", run.a.calls, run.b.calls, run.decision.calls)
		}
	})
}
