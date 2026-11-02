package nexssflow

import (
	"context"
	"strings"
	"time"

	"github.com/nexssp/cost"
	"github.com/nexssp/kernel/action"
	"github.com/nexssp/kernel/xerr"

	"github.com/nexssp/ai/engine/llm"
)

var currentTime = time.Now

// newDecisionActionWith creates a typed action that delegates through the
// configured DecisionProvider. Candidate prompt attempts are metadata only;
// they are echoed for NFlow routing and never passed to the decision provider.
func newDecisionActionWith(global GlobalOptions, build decisionProviderBuilder) action.AnyAction {
	return action.New[DecisionActionRequest, DecisionActionResult]("ai.decide", func(ctx context.Context, req DecisionActionRequest) (DecisionActionResult, error) {
		if build == nil {
			return DecisionActionResult{}, xerr.Internal("ai.decide: decision provider builder is unavailable")
		}
		if strings.TrimSpace(req.AttemptID) == "" {
			return DecisionActionResult{}, xerr.Validation("ai.decide: attempt_id is required")
		}
		if req.MaxOutputTokens <= 0 {
			return DecisionActionResult{}, xerr.Validation("ai.decide: max_output_tokens must be a positive integer")
		}
		decisionModel := firstNonEmpty(req.Model, global.DecisionModel)
		priced := req.RateCardID != "" || req.ExpectedCurrency != nil
		if err := validatePricingFields("ai.decide", decisionModel, req.RateCardID, req.ExpectedCurrency, false); err != nil {
			return DecisionActionResult{}, err
		}
		if priced {
			if err := validateConfiguredPricing("ai.decide", decisionModel, req.RateCardID, req.ExpectedCurrency); err != nil {
				return DecisionActionResult{}, err
			}
		}
		if err := validateTokenBasis("ai.decide", req.PromptTokens, req.CachedPromptTokens, req.CompletionTokens, priced); err != nil {
			return DecisionActionResult{}, err
		}
		if err := validateDecisionCandidates(req); err != nil {
			return DecisionActionResult{}, err
		}
		provider, err := build(global, DecisionDecl{
			Provider: req.Provider,
			Model:    req.Model,
			BaseURL:  req.BaseURL,
		})
		if err != nil {
			return DecisionActionResult{}, completionError(err)
		}
		if provider == nil {
			return DecisionActionResult{}, xerr.Internal("ai.decide: provider builder returned nil")
		}
		response, err := provider.Decide(ctx, llm.DecisionRequest{
			Model:                       decisionModel,
			AttemptID:                   req.AttemptID,
			RateCardID:                  req.RateCardID,
			ExpectedCurrency:            cloneCurrency(req.ExpectedCurrency),
			EstimatedPromptTokens:       req.PromptTokens,
			EstimatedCachedPromptTokens: req.CachedPromptTokens,
			EstimatedCompletionTokens:   req.CompletionTokens,
			MaxOutputTokens:             req.MaxOutputTokens,
			State:                       req.State,
			Questions:                   req.Questions,
			AllowedIDs:                  req.AllowedIDs,
		})
		if err != nil {
			return DecisionActionResult{}, completionError(err)
		}
		if response.Usage.CompTokens > req.MaxOutputTokens {
			return DecisionActionResult{}, xerr.Internal("ai.decide: provider reported output usage above max_output_tokens")
		}
		return DecisionActionResult{
			AttemptID:          req.AttemptID,
			Model:              response.Model,
			RateCardID:         req.RateCardID,
			ExpectedCurrency:   cloneCurrency(req.ExpectedCurrency),
			PromptTokens:       req.PromptTokens,
			CachedPromptTokens: req.CachedPromptTokens,
			CompletionTokens:   req.CompletionTokens,
			MaxOutputTokens:    req.MaxOutputTokens,
			Answers:            response.Answers,
			Usage:              response.Usage,
			Candidates:         append([]DecisionCandidate(nil), req.Candidates...),
		}, nil
	}).Description("Run a structured decision with a separately configured decision model").Tag("ai", "decision").Build()
}

func validateDecisionCandidates(req DecisionActionRequest) error {
	if len(req.Candidates) == 0 {
		return nil
	}
	allowed := make(map[string]struct{}, len(req.AllowedIDs))
	for _, id := range req.AllowedIDs {
		allowed[id] = struct{}{}
	}
	seenIDs := make(map[string]struct{}, len(req.Candidates))
	seenAttempts := make(map[string]struct{}, len(req.Candidates)+1)
	seenAttempts[req.AttemptID] = struct{}{}
	for i := range req.Candidates {
		candidate := &req.Candidates[i]
		id := strings.TrimSpace(candidate.ID)
		if id == "" || strings.TrimSpace(candidate.Description) == "" {
			return xerr.Validation("ai.decide: candidates require an ID and description")
		}
		if _, exists := seenIDs[id]; exists {
			return xerr.Validation("ai.decide: candidate IDs must be unique")
		}
		seenIDs[id] = struct{}{}
		if _, ok := allowed[id]; !ok {
			return xerr.Validation("ai.decide: candidate ID must be included in allowed_ids: " + id)
		}
		attempt := candidate.Attempt
		if attempt.CandidateID != id || strings.TrimSpace(attempt.CandidateID) != attempt.CandidateID || strings.TrimSpace(attempt.AttemptID) == "" || strings.TrimSpace(attempt.AttemptID) != attempt.AttemptID || strings.TrimSpace(attempt.Provider) == "" || strings.TrimSpace(attempt.Model) == "" {
			return xerr.Validation("ai.decide: candidate prompt requires matching candidate_id, attempt_id, provider, and model")
		}
		if _, exists := seenAttempts[attempt.AttemptID]; exists {
			return xerr.Validation("ai.decide: candidate attempt IDs must be unique")
		}
		seenAttempts[attempt.AttemptID] = struct{}{}
		if strings.TrimSpace(attempt.RateCardID) == "" || strings.TrimSpace(attempt.RateCardID) != attempt.RateCardID || !validCurrency(attempt.ExpectedCurrency) {
			return xerr.Validation("ai.decide: candidate requires rate_card_id and expected_currency")
		}
		if err := validateTokenBasis("ai.decide candidate", attempt.PromptTokens, attempt.CachedPromptTokens, attempt.CompletionTokens, true); err != nil {
			return err
		}
	}
	return nil
}

// newCostEstimateAction is a pure calculation boundary. Selector approval and
// budget reservation belong to the host guard, not this estimate action.
func newCostEstimateAction() action.AnyAction {
	return action.New[CostEstimateRequest, CostEstimateResult]("ai.cost.estimate", func(_ context.Context, req CostEstimateRequest) (CostEstimateResult, error) {
		if err := validateCostEstimateRequest(req); err != nil {
			return CostEstimateResult{}, err
		}
		return calculateCostEstimate(req), nil
	}).Description("Estimate one attempt from an exact configured rate card; this action does not bill, reserve selector budget, enforce limits, or trigger fallback").Tag("ai", "cost", "estimate").Build()
}

func validateCostEstimateRequest(req CostEstimateRequest) error {
	if strings.TrimSpace(req.Model) == "" || strings.TrimSpace(req.Model) != req.Model {
		return xerr.Validation("ai.cost.estimate: model is required")
	}
	if strings.TrimSpace(req.AttemptID) == "" {
		return xerr.Validation("ai.cost.estimate: attempt_id is required")
	}
	if strings.TrimSpace(req.RateCardID) == "" || strings.TrimSpace(req.RateCardID) != req.RateCardID {
		return xerr.Validation("ai.cost.estimate: rate_card_id is required")
	}
	if !validCurrency(req.ExpectedCurrency) {
		return xerr.Validation("ai.cost.estimate: expected_currency is required")
	}
	if err := validateTokenBasis("ai.cost.estimate", req.PromptTokens, req.CachedPromptTokens, req.CompletionTokens, true); err != nil {
		return err
	}
	if req.Decision != nil && req.Prompt != nil {
		return xerr.Validation("ai.cost.estimate: set either decision or prompt, not both")
	}
	if req.Decision != nil {
		if err := validateDecisionCostPreflight(req); err != nil {
			return err
		}
	}
	if req.Prompt != nil {
		if err := validatePromptCostPreflight(req); err != nil {
			return err
		}
	}
	return nil
}

func validateDecisionCostPreflight(req CostEstimateRequest) error {
	decision := req.Decision
	if strings.TrimSpace(decision.AttemptID) == "" || req.CandidateID != "" || req.AttemptID != decision.AttemptID || req.Model != decision.Model {
		return xerr.Validation("ai.cost.estimate: decision preflight IDs and model must match the estimate")
	}
	if decision.MaxOutputTokens <= 0 {
		return xerr.Validation("ai.cost.estimate: decision max_output_tokens must be a positive integer")
	}
	if req.CompletionTokens != decision.MaxOutputTokens || decision.CompletionTokens != decision.MaxOutputTokens {
		return xerr.Validation("ai.cost.estimate: decision completion token basis must equal max_output_tokens")
	}
	if decision.RateCardID == "" && decision.ExpectedCurrency == nil {
		return nil
	}
	if err := validatePricingFields("ai.cost.estimate decision", decision.Model, decision.RateCardID, decision.ExpectedCurrency, true); err != nil {
		return err
	}
	if decision.RateCardID != req.RateCardID || *decision.ExpectedCurrency != req.ExpectedCurrency {
		return xerr.Validation("ai.cost.estimate: decision pricing identity must match the estimate")
	}
	if err := validateTokenBasis("ai.cost.estimate decision", decision.PromptTokens, decision.CachedPromptTokens, decision.CompletionTokens, true); err != nil {
		return err
	}
	if decision.PromptTokens != req.PromptTokens || decision.CachedPromptTokens != req.CachedPromptTokens || decision.CompletionTokens != req.CompletionTokens {
		return xerr.Validation("ai.cost.estimate: decision token basis must match the estimate")
	}
	return nil
}

func validatePromptCostPreflight(req CostEstimateRequest) error {
	prompt := req.Prompt
	if req.CandidateID == "" || (prompt.CandidateID != "" && req.CandidateID != prompt.CandidateID) || req.AttemptID != prompt.AttemptID || req.Model != prompt.Model {
		return xerr.Validation("ai.cost.estimate: prompt preflight IDs and model must match the estimate")
	}
	if err := validatePricingFields("ai.cost.estimate prompt", prompt.Model, prompt.RateCardID, prompt.ExpectedCurrency, true); err != nil {
		return err
	}
	if prompt.RateCardID != req.RateCardID || *prompt.ExpectedCurrency != req.ExpectedCurrency {
		return xerr.Validation("ai.cost.estimate: prompt pricing identity must match the estimate")
	}
	if err := validateTokenBasis("ai.cost.estimate prompt", prompt.PromptTokens, prompt.CachedPromptTokens, prompt.CompletionTokens, true); err != nil {
		return err
	}
	if prompt.PromptTokens != req.PromptTokens || prompt.CachedPromptTokens != req.CachedPromptTokens || prompt.CompletionTokens != req.CompletionTokens {
		return xerr.Validation("ai.cost.estimate: prompt token basis must match the estimate")
	}
	return nil
}

func calculateCostEstimate(req CostEstimateRequest) CostEstimateResult {
	amount, currency, known := llm.CalculateCostForRateCard(
		req.Model,
		req.RateCardID,
		req.ExpectedCurrency,
		req.PromptTokens,
		req.CachedPromptTokens,
		req.CompletionTokens,
		currentTime(),
	)
	result := CostEstimateResult{
		CandidateID:        req.CandidateID,
		AttemptID:          req.AttemptID,
		Model:              req.Model,
		RateCardID:         req.RateCardID,
		ExpectedCurrency:   req.ExpectedCurrency,
		PromptTokens:       req.PromptTokens,
		CachedPromptTokens: req.CachedPromptTokens,
		CompletionTokens:   req.CompletionTokens,
		Known:              known,
		CostMicros:         amount,
		Decision:           req.Decision,
		Prompt:             req.Prompt,
	}
	if known {
		result.Currency = &currency
		return result
	}
	card, ok := llm.GlobalCatalog.GetRateCard(req.Model, req.RateCardID)
	switch {
	case !ok:
		result.Issue = "rate_card_not_found"
	case card.Currency == (cost.Currency{}):
		result.Issue = "rate_card_currency_missing"
	case card.Currency != req.ExpectedCurrency:
		result.Issue = "currency_mismatch"
	default:
		result.Issue = "invalid_rate_card"
	}
	return result
}
