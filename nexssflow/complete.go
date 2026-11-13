package nexssflow

import (
	"context"
	"strings"

	"github.com/nexssp/kernel/xerr"

	"github.com/nexssp/ai/engine/llm"
)

func validateUser(user string) error {
	if strings.TrimSpace(user) == "" {
		return xerr.Validation("ai.prompt: user field is required")
	}
	return nil
}

func validatePromptMessages(req PromptRequest) error {
	if len(req.Messages) == 0 {
		return validateUser(req.User)
	}
	if req.User != "" || req.System != "" {
		return xerr.Validation("ai.prompt: messages cannot be combined with user or system")
	}
	for _, message := range req.Messages {
		if message.Role == llm.RoleUser && strings.TrimSpace(message.Content) != "" {
			return nil
		}
	}
	return xerr.Validation("ai.prompt: messages must include a non-empty user message")
}

func validateToolOptions(req PromptRequest) error {
	if len(req.Tools) == 0 {
		if req.ToolChoice != "" {
			return xerr.Validation("ai.prompt: tool_choice requires at least one tool")
		}
		return nil
	}
	switch req.ToolChoice {
	case "", llm.ToolChoiceAuto, llm.ToolChoiceNone, llm.ToolChoiceRequired:
		return nil
	default:
		return xerr.Validation("ai.prompt: tool_choice must be auto, none, or required")
	}
}

// complete runs exactly one provider attempt and returns an explicit typed
// result. Candidate selection, budget checks, and fallback policy belong to
// the calling workflow, not this action.
func complete(ctx context.Context, provider llm.Provider, req PromptRequest) (PromptResult, error) {
	if err := validateToolOptions(req); err != nil {
		return PromptResult{}, err
	}
	priced := req.RateCardID != "" || req.ExpectedCurrency != nil
	if err := validatePricingFields("ai.prompt", req.Model, req.RateCardID, req.ExpectedCurrency, false); err != nil {
		return PromptResult{}, err
	}
	if priced {
		if err := validateConfiguredPricing("ai.prompt", req.Model, req.RateCardID, req.ExpectedCurrency); err != nil {
			return PromptResult{}, err
		}
	}
	if err := validateTokenBasis("ai.prompt", req.PromptTokens, req.CachedPromptTokens, req.CompletionTokens, priced); err != nil {
		return PromptResult{}, err
	}
	if req.MaxTokens < 0 {
		return PromptResult{}, xerr.Validation("ai.prompt: max_tokens cannot be negative")
	}
	if err := requireProvider(provider); err != nil {
		return PromptResult{}, err
	}
	if len(req.Tools) > 0 {
		toolProvider, ok := provider.(llm.ToolCallingProvider)
		if !ok || !toolProvider.SupportsToolCalls() {
			return PromptResult{}, xerr.Validation("ai.prompt: provider " + provider.Name() + " does not support non-streaming function tool calls")
		}
	}

	messages := req.Messages
	if len(messages) == 0 {
		messages = make([]llm.Message, 0, 2)
		if req.System != "" {
			messages = append(messages, llm.Message{Role: llm.RoleSystem, Content: req.System})
		}
		messages = append(messages, llm.Message{Role: llm.RoleUser, Content: req.User})
	}

	response, err := provider.Complete(ctx, llm.Request{
		Model:                       req.Model,
		CandidateID:                 req.CandidateID,
		AttemptID:                   req.AttemptID,
		RateCardID:                  req.RateCardID,
		ExpectedCurrency:            cloneCurrency(req.ExpectedCurrency),
		EstimatedPromptTokens:       req.PromptTokens,
		EstimatedCachedPromptTokens: req.CachedPromptTokens,
		EstimatedCompletionTokens:   req.CompletionTokens,
		Messages:                    messages,
		Tools:                       req.Tools,
		ToolChoice:                  req.ToolChoice,
		MaxTokens:                   req.MaxTokens,
		Temperature:                 req.Temperature,
		JSONMode:                    req.JSONMode,
		ReasoningEffort:             req.ReasoningEffort,
	})
	if err != nil {
		return PromptResult{}, completionError(err)
	}

	model := response.Model
	if model == "" {
		model = req.Model
	}
	response.Model = model
	usage := llm.FromResponse(response)
	if priced {
		usage = llm.FromResponseForRateCard(response, req.RateCardID, *req.ExpectedCurrency)
	}
	return PromptResult{
		CandidateID:        req.CandidateID,
		AttemptID:          req.AttemptID,
		RateCardID:         req.RateCardID,
		ExpectedCurrency:   cloneCurrency(req.ExpectedCurrency),
		Context:            req.Context,
		Provider:           provider.Name(),
		Content:            response.Content,
		ToolCalls:          response.ToolCalls,
		Model:              model,
		PromptTokens:       response.PromptTokens,
		CachedPromptTokens: response.CachedPromptTokens,
		CompTokens:         response.CompTokens,
		TotalTokens:        response.PromptTokens + response.CompTokens,
		Usage:              usage,
	}, nil
}
