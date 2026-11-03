package nexssflow

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/nexssp/cost"
	"github.com/nexssp/flow/native"
	"github.com/nexssp/flow/runner"

	"github.com/nexssp/ai/engine/llm"
)

type fakeDecisionProvider struct {
	request llm.DecisionRequest
	calls   int
}

func (p *fakeDecisionProvider) Name() string { return "fake-decision" }

func (p *fakeDecisionProvider) Decide(_ context.Context, req llm.DecisionRequest) (llm.DecisionResponse, error) {
	p.request = req
	p.calls++
	return llm.DecisionResponse{
		Model: "served-decision-model",
		Answers: []llm.DecisionAnswer{{
			ID: "route", Type: llm.DecisionChoice, Choice: "candidate-b",
		}},
	}, nil
}

func TestDecisionActionUsesTypedRequestAndDedicatedProvider(t *testing.T) {
	provider := &fakeDecisionProvider{}
	act := newDecisionActionWith(GlobalOptions{DecisionModel: "configured-decision-model"}, func(_ GlobalOptions, decl DecisionDecl) (llm.DecisionProvider, error) {
		if decl.Model != "" {
			t.Errorf("per-action model = %q, want empty so global decision model is used", decl.Model)
		}
		return provider, nil
	})
	request := DecisionActionRequest{
		AttemptID:       "decision-attempt-7",
		MaxOutputTokens: 32,
		State:           json.RawMessage(`{"task":"route"}`),
		AllowedIDs:      []string{"candidate-a", "candidate-b"},
		Candidates: []DecisionCandidate{
			{ID: "candidate-a", Description: "Candidate A", Attempt: CandidateAttemptPlan{CandidateID: "candidate-a", AttemptID: "candidate-attempt-a", Provider: "fake", Model: "model-a", RateCardID: "card-a", ExpectedCurrency: cost.EUR, PromptTokens: 10, CompletionTokens: 5, User: "route this"}},
			{ID: "candidate-b", Description: "Candidate B", Attempt: CandidateAttemptPlan{CandidateID: "candidate-b", AttemptID: "candidate-attempt-b", Provider: "fake", Model: "model-b", RateCardID: "card-b", ExpectedCurrency: cost.EUR, PromptTokens: 10, CompletionTokens: 5, User: "route this"}},
		},
		Questions: []llm.DecisionQuestion{{
			ID: "route", Type: llm.DecisionChoice,
			Criteria: []llm.DecisionCriterion{
				{ID: "candidate-a", Description: "Candidate A"},
				{ID: "candidate-b", Description: "Candidate B"},
			},
		}},
	}
	result, err := act.DoAny(context.Background(), request)
	if err != nil {
		t.Fatalf("ai.decide: %v", err)
	}
	response, ok := result.(DecisionActionResult)
	if !ok {
		t.Fatalf("result type = %T, want DecisionActionResult", result)
	}
	if provider.calls != 1 || provider.request.Model != "configured-decision-model" || provider.request.MaxOutputTokens != 32 || string(provider.request.State) != string(request.State) || len(provider.request.AllowedIDs) != 2 || response.Answers[0].Choice != "candidate-b" {
		t.Fatalf("unexpected request/result: request=%+v response=%+v calls=%d", provider.request, response, provider.calls)
	}
	if response.AttemptID != request.AttemptID || len(response.Candidates) != 2 || response.Candidates[1].Attempt.Model != "model-b" {
		t.Fatalf("decision correlation/candidates were not echoed: %+v", response)
	}
	if response.MaxOutputTokens != request.MaxOutputTokens {
		t.Fatalf("output cap was not echoed: %+v", response)
	}
}

func TestDecisionActionRejectsMissingOrInvalidCapBeforeProviderBuild(t *testing.T) {
	buildCalls := 0
	act := newDecisionActionWith(GlobalOptions{}, func(_ GlobalOptions, _ DecisionDecl) (llm.DecisionProvider, error) {
		buildCalls++
		return &fakeDecisionProvider{}, nil
	})
	for _, maxOutputTokens := range []int{0, -1} {
		_, err := act.DoAny(context.Background(), DecisionActionRequest{
			AttemptID:       "attempt-without-valid-cap",
			Model:           "decision-model",
			MaxOutputTokens: maxOutputTokens,
			State:           json.RawMessage(`{"task":"route"}`),
			Questions:       []llm.DecisionQuestion{{ID: "q", Type: llm.DecisionNoul, Instructions: "Check this."}},
		})
		if err == nil {
			t.Fatalf("cap %d should be rejected", maxOutputTokens)
		}
	}
	if buildCalls != 0 {
		t.Fatalf("provider build calls = %d, want zero", buildCalls)
	}
}

func TestDecisionActionCarriesExactPricingMetadata(t *testing.T) {
	previous := llm.GlobalCatalog
	llm.GlobalCatalog = llm.NewPriceCatalog()
	defer func() { llm.GlobalCatalog = previous }()
	if err := llm.GlobalCatalog.Register("decision-model", llm.PriceCard{RateCardID: "decision-eur-card", Currency: cost.EUR, Peak: llm.Price{PromptPer1k: 1}}); err != nil {
		t.Fatal(err)
	}

	provider := &fakeDecisionProvider{}
	expected := cost.EUR
	act := newDecisionActionWith(GlobalOptions{}, func(_ GlobalOptions, _ DecisionDecl) (llm.DecisionProvider, error) { return provider, nil })
	request := DecisionActionRequest{
		AttemptID: "decision-attempt-9", Model: "decision-model",
		RateCardID: "decision-eur-card", ExpectedCurrency: &expected,
		PromptTokens: 20, CachedPromptTokens: 4, CompletionTokens: 8,
		MaxOutputTokens: 8,
		State:           json.RawMessage(`{"task":"route"}`),
		Questions:       []llm.DecisionQuestion{{ID: "route", Type: llm.DecisionNoul, Instructions: "score the route"}},
	}
	value, err := act.DoAny(context.Background(), request)
	if err != nil {
		t.Fatalf("ai.decide: %v", err)
	}
	result, ok := value.(DecisionActionResult)
	if !ok {
		t.Fatalf("ai.decide result has unexpected type %T", value)
	}
	if provider.request.AttemptID != request.AttemptID || provider.request.RateCardID != request.RateCardID || provider.request.ExpectedCurrency == nil || *provider.request.ExpectedCurrency != cost.EUR || provider.request.EstimatedPromptTokens != 20 || provider.request.EstimatedCachedPromptTokens != 4 || provider.request.EstimatedCompletionTokens != 8 || provider.request.MaxOutputTokens != 8 {
		t.Fatalf("decision provider request lost pricing metadata: %+v", provider.request)
	}
	if result.AttemptID != request.AttemptID || result.RateCardID != request.RateCardID || result.ExpectedCurrency == nil || *result.ExpectedCurrency != cost.EUR || result.PromptTokens != 20 || result.CachedPromptTokens != 4 || result.CompletionTokens != 8 || result.MaxOutputTokens != 8 {
		t.Fatalf("decision result lost pricing/correlation echo: %+v", result)
	}
}

func TestDecisionProviderMustBeConfiguredSeparatelyFromCandidateModel(t *testing.T) {
	if _, err := buildDecisionProvider(GlobalOptions{
		DecisionProvider: "inception",
		DefaultModel:     "candidate-model",
	}, DecisionDecl{}); err == nil {
		t.Fatal("candidate model must not be reused as the decision model")
	}

	provider, err := buildDecisionProvider(GlobalOptions{
		DecisionProvider: "inception",
		DecisionModel:    "configured-decision-model",
	}, DecisionDecl{})
	if err != nil {
		t.Fatalf("build configured decision provider: %v", err)
	}
	if provider.Name() != "inception" {
		t.Fatalf("decision provider name = %q, want inception", provider.Name())
	}
	typesafeProvider, err := buildDecisionProvider(GlobalOptions{DecisionProvider: "typesafe", DecisionModel: "jev-1.14.0"}, DecisionDecl{})
	if err != nil || typesafeProvider.Name() != "typesafe" {
		t.Fatalf("build TypeSafe decision provider: provider=%v err=%v", typesafeProvider, err)
	}

	jsonProvider, err := buildDecisionProvider(GlobalOptions{
		DecisionProvider: "openai",
		DecisionModel:    "some-model",
	}, DecisionDecl{})
	if err != nil {
		t.Fatalf("build verified JSON-mode decision provider: %v", err)
	}
	if jsonProvider.Name() != "openai" {
		t.Fatalf("JSON-mode decision provider name = %q, want openai", jsonProvider.Name())
	}
	anthropicProvider, err := buildDecisionProvider(GlobalOptions{
		DecisionProvider: "anthropic",
		DecisionModel:    "some-model",
	}, DecisionDecl{})
	if err != nil || anthropicProvider.Name() != "anthropic" {
		t.Fatalf("build Anthropic schema-output decision provider: provider=%v err=%v", anthropicProvider, err)
	}

	mercuryChat, err := buildDecisionProvider(GlobalOptions{
		DecisionProvider: "inception-chat",
		DecisionModel:    "mercury-3",
	}, DecisionDecl{})
	if err != nil || mercuryChat.Name() != "inception" {
		t.Fatalf("build Mercury chat structured-output adapter: provider=%v err=%v", mercuryChat, err)
	}
}

func TestNFlowOwnsDecisionRoutingGraph(t *testing.T) {
	providerAnswer := llm.DecisionResponse{
		Model:   "configured-decision-model",
		Answers: []llm.DecisionAnswer{{ID: "route", Type: llm.DecisionChoice, Choice: "candidate-b"}},
	}
	providerResult := &staticDecisionProvider{response: providerAnswer}
	bundle := bundleWithBuilders(GlobalOptions{}, nil, func(GlobalOptions, DecisionDecl) (llm.DecisionProvider, error) {
		return providerResult, nil
	})
	cfg, err := runner.BuildConfig(append(native.Bundles(), bundle))
	if err != nil {
		t.Fatalf("BuildConfig: %v", err)
	}
	source := `{
		attempt_id: "decision-attempt-1",
		max_output_tokens: 16,
		state: {task: "route"},
	    allowed_ids: ["candidate-a", "candidate-b"],
	    questions: [{id: "route", type: "choice", criteria: [
	        {id: "candidate-a", description: "A"},
	        {id: "candidate-b", description: "B"}
		    ]}],
		candidates: [
			{id: "candidate-a", description: "A", attempt: {candidate_id: "candidate-a", attempt_id: "candidate-attempt-a", provider: "fake", model: "model-a", rate_card_id: "rate-a", expected_currency: "EUR", prompt_tokens: 10, completion_tokens: 5, user: "request"}},
			{id: "candidate-b", description: "B", attempt: {candidate_id: "candidate-b", attempt_id: "candidate-attempt-b", provider: "fake", model: "model-b", rate_card_id: "rate-b", expected_currency: "EUR", prompt_tokens: 10, completion_tokens: 5, user: "request"}}
		]
		} -> ai.decide -> ai.nflow.view as decision_result
		-> match {
			decision_result.answers[0].choice == "candidate-a" -> const @{route: "candidate-a"},
			decision_result.answers[0].choice == "candidate-b" && decision_result.attempt_id == "decision-attempt-1" && decision_result.candidates[1].attempt.model == "model-b" -> const @{route: "candidate-b"},
		    _ -> const @{route: "manual-review"},
		}`
	execution, err := runner.Execute(context.Background(), cfg, source, "decision-route-test.nflow", nil)
	if err != nil {
		t.Fatalf("runner.Execute: %v", err)
	}
	output, ok := execution.Output.(map[string]any)
	if !ok || output["route"] != "candidate-b" {
		t.Fatalf("NFlow route result = %#v, want candidate-b", execution.Output)
	}
}

type staticDecisionProvider struct {
	response llm.DecisionResponse
	calls    int
}

func (p *staticDecisionProvider) Name() string { return "static-decision" }

func (p *staticDecisionProvider) Decide(context.Context, llm.DecisionRequest) (llm.DecisionResponse, error) {
	p.calls++
	return p.response, nil
}

func TestCostEstimateActionUsesExternalCurrencyAndAttemptIDs(t *testing.T) {
	previous := llm.GlobalCatalog
	llm.GlobalCatalog = llm.NewPriceCatalog()
	defer func() { llm.GlobalCatalog = previous }()
	if err := llm.GlobalCatalog.Register("cost-action-test-model", llm.PriceCard{
		RateCardID: "test-pln-card-2026",
		Currency:   cost.PLN,
		Peak: llm.Price{
			PromptPer1k:      1,
			PromptCacheHit1k: 0.5,
			CompPer1k:        2,
		},
	}); err != nil {
		t.Fatal(err)
	}

	act := newCostEstimateAction()
	expectedCurrency := cost.PLN
	result, err := act.DoAny(context.Background(), CostEstimateRequest{
		CandidateID:        "candidate-a",
		AttemptID:          "attempt-3",
		Model:              "cost-action-test-model",
		RateCardID:         "test-pln-card-2026",
		ExpectedCurrency:   cost.PLN,
		PromptTokens:       10,
		CachedPromptTokens: 5,
		CompletionTokens:   3,
		Prompt:             &PromptRequest{CandidateID: "candidate-a", AttemptID: "attempt-3", Model: "cost-action-test-model", RateCardID: "test-pln-card-2026", ExpectedCurrency: &expectedCurrency, PromptTokens: 10, CachedPromptTokens: 5, CompletionTokens: 3, Provider: "fake", User: "route this request"},
	})
	if err != nil {
		t.Fatalf("ai.cost.estimate: %v", err)
	}
	estimate, ok := result.(CostEstimateResult)
	if !ok {
		t.Fatalf("result type = %T, want CostEstimateResult", result)
	}
	if !estimate.Known || estimate.CostMicros != cost.Micro(13500) || estimate.Currency == nil || *estimate.Currency != cost.PLN || estimate.ExpectedCurrency != cost.PLN || estimate.CandidateID != "candidate-a" || estimate.AttemptID != "attempt-3" {
		t.Fatalf("unexpected typed cost estimate: %+v", estimate)
	}
	if estimate.RateCardID != "test-pln-card-2026" || estimate.PromptTokens != 10 || estimate.CachedPromptTokens != 5 || estimate.CompletionTokens != 3 || estimate.Prompt == nil || estimate.Prompt.User != "route this request" {
		t.Fatalf("rate card/prompt = %q/%+v", estimate.RateCardID, estimate.Prompt)
	}
}

func TestCostEstimateUsesOuterCandidateIDWhenNestedPromptOmitsIt(t *testing.T) {
	previous := llm.GlobalCatalog
	llm.GlobalCatalog = llm.NewPriceCatalog()
	defer func() { llm.GlobalCatalog = previous }()
	if err := llm.GlobalCatalog.Register("route-model", llm.PriceCard{RateCardID: "route-eur", Currency: cost.EUR, Peak: llm.Price{PromptPer1k: 1}}); err != nil {
		t.Fatal(err)
	}

	expected := cost.EUR
	value, err := newCostEstimateAction().DoAny(context.Background(), CostEstimateRequest{
		CandidateID: "candidate-from-route", AttemptID: "attempt-from-route", Model: "route-model",
		RateCardID: "route-eur", ExpectedCurrency: cost.EUR, PromptTokens: 10, CompletionTokens: 5,
		Prompt: &PromptRequest{
			AttemptID: "attempt-from-route", Model: "route-model", RateCardID: "route-eur",
			ExpectedCurrency: &expected, PromptTokens: 10, CompletionTokens: 5, User: "route this request",
		},
	})
	if err != nil {
		t.Fatalf("ai.cost.estimate: %v", err)
	}
	result, ok := value.(CostEstimateResult)
	if !ok {
		t.Fatalf("ai.cost.estimate result has unexpected type %T", value)
	}
	if !result.Known || result.CandidateID != "candidate-from-route" || result.AttemptID != "attempt-from-route" {
		t.Fatalf("outer candidate/attempt were not echoed: %+v", result)
	}
}

func TestCostEstimateActionFailsClosedOnRateCardOrCurrencyMismatch(t *testing.T) {
	previous := llm.GlobalCatalog
	llm.GlobalCatalog = llm.NewPriceCatalog()
	defer func() { llm.GlobalCatalog = previous }()
	if err := llm.GlobalCatalog.Register("model-a", llm.PriceCard{RateCardID: "eur-card", Currency: cost.EUR, Peak: llm.Price{PromptPer1k: 1}}); err != nil {
		t.Fatal(err)
	}
	act := newCostEstimateAction()
	expectedCurrency := cost.USD
	base := CostEstimateRequest{CandidateID: "candidate-a", AttemptID: "attempt-a", Model: "model-a", RateCardID: "eur-card", ExpectedCurrency: cost.USD, PromptTokens: 10, CompletionTokens: 5, Prompt: &PromptRequest{CandidateID: "candidate-a", AttemptID: "attempt-a", Model: "model-a", RateCardID: "eur-card", ExpectedCurrency: &expectedCurrency, PromptTokens: 10, CompletionTokens: 5, Provider: "fake", User: "hello"}}
	result, err := act.DoAny(context.Background(), base)
	if err != nil {
		t.Fatalf("known request should return an explicit unknown estimate: %v", err)
	}
	estimate, ok := result.(CostEstimateResult)
	if !ok {
		t.Fatalf("ai.cost.estimate result has unexpected type %T", result)
	}
	if estimate.Known || estimate.Issue != "currency_mismatch" || estimate.CostMicros != 0 {
		t.Fatalf("currency mismatch did not fail closed: %+v", estimate)
	}
	base.ExpectedCurrency = cost.EUR
	base.RateCardID = "missing-card"
	expectedCurrency = cost.EUR
	base.Prompt.RateCardID = "missing-card"
	result, err = act.DoAny(context.Background(), base)
	if err != nil {
		t.Fatal(err)
	}
	estimate, ok = result.(CostEstimateResult)
	if !ok {
		t.Fatalf("ai.cost.estimate result has unexpected type %T", result)
	}
	if estimate.Known || estimate.Issue != "rate_card_not_found" || estimate.CostMicros != 0 {
		t.Fatalf("unknown card did not fail closed: %+v", estimate)
	}
	base.ExpectedCurrency = cost.Currency{}
	if _, err := act.DoAny(context.Background(), base); err == nil {
		t.Fatal("missing expected currency should fail validation")
	}
}

func TestCostEstimateActionDecodesNestedDecisionTarget(t *testing.T) {
	act := newCostEstimateAction()
	result, err := act.DoAny(context.Background(), map[string]any{
		"attempt_id":        "decision-attempt-1",
		"model":             "decision-model",
		"rate_card_id":      "unconfigured-card",
		"expected_currency": "USD",
		"prompt_tokens":     10,
		"completion_tokens": 5,
		"decision": map[string]any{
			"attempt_id":        "decision-attempt-1",
			"provider":          "fake-decision",
			"model":             "decision-model",
			"completion_tokens": 5,
			"max_output_tokens": 5,
			"state":             map[string]any{"task": "route"},
			"questions":         []any{},
		},
	})
	if err != nil {
		t.Fatalf("ai.cost.estimate with nested decision: %v", err)
	}
	estimate, ok := result.(CostEstimateResult)
	if !ok || estimate.Decision == nil || estimate.Decision.AttemptID != "decision-attempt-1" {
		t.Fatalf("decoded decision target = %#v, want echoed typed decision attempt", result)
	}
}

func TestDecisionCostEstimateUsesExactOutputCapAndFailsClosed(t *testing.T) {
	previous := llm.GlobalCatalog
	llm.GlobalCatalog = llm.NewPriceCatalog()
	defer func() { llm.GlobalCatalog = previous }()
	if err := llm.GlobalCatalog.Register("decision-estimate-model", llm.PriceCard{
		RateCardID: "decision-cap-card",
		Currency:   cost.EUR,
		Peak:       llm.Price{PromptPer1k: 1, CompPer1k: 2},
	}); err != nil {
		t.Fatal(err)
	}

	expected := cost.EUR
	base := CostEstimateRequest{
		AttemptID: "decision-estimate-attempt", Model: "decision-estimate-model",
		RateCardID: "decision-cap-card", ExpectedCurrency: cost.EUR,
		PromptTokens: 10, CompletionTokens: 8,
		Decision: &DecisionActionRequest{
			AttemptID: "decision-estimate-attempt", Model: "decision-estimate-model",
			RateCardID: "decision-cap-card", ExpectedCurrency: &expected,
			PromptTokens: 10, CompletionTokens: 8, MaxOutputTokens: 8,
		},
	}
	value, err := newCostEstimateAction().DoAny(context.Background(), base)
	if err != nil {
		t.Fatalf("cost estimate: %v", err)
	}
	result, ok := value.(CostEstimateResult)
	if !ok {
		t.Fatalf("ai.cost.estimate result has unexpected type %T", value)
	}
	want, currency, known := llm.CalculateCostForRateCard("decision-estimate-model", "decision-cap-card", cost.EUR, 10, 0, 8, currentTime())
	if !known || result.CompletionTokens != 8 || result.CostMicros != want || result.Currency == nil || *result.Currency != currency {
		t.Fatalf("estimate did not use exact output cap: got=%+v want_cost=%d", result, want)
	}

	for _, maxOutputTokens := range []int{0, -1} {
		invalid := base
		decision := *base.Decision
		decision.MaxOutputTokens = maxOutputTokens
		invalid.Decision = &decision
		if _, err := newCostEstimateAction().DoAny(context.Background(), invalid); err == nil {
			t.Fatalf("cap %d should fail closed", maxOutputTokens)
		}
	}
	mismatched := base
	mismatched.CompletionTokens = 7
	if _, err := newCostEstimateAction().DoAny(context.Background(), mismatched); err == nil {
		t.Fatal("estimate completion basis differing from max_output_tokens should fail closed")
	}
}
