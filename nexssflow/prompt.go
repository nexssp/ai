package nexssflow

import (
	"context"

	"github.com/nexssp/kernel/action"
	"github.com/nexssp/kernel/xerr"

	"github.com/nexssp/ai/engine/llm"
)

// ProviderBuilder lets a host supply a Provider implementation, such as a
// gateway, to the ai.prompt, ai.team, and ai.rag.answer Flow actions.
type ProviderBuilder func(GlobalOptions, LLMDecl) (llm.Provider, error)

type providerBuilder = ProviderBuilder

// newPromptAction builds the reusable ai.prompt action.
func newPromptAction(global GlobalOptions) action.AnyAction {
	return newPromptActionWith(global, buildProvider)
}

// newPromptActionWith keeps provider construction injectable for deterministic tests.
func newPromptActionWith(global GlobalOptions, build providerBuilder) *action.BuiltAction[flowPromptRequest, PromptResult] {
	return action.New[flowPromptRequest, PromptResult]("ai.prompt", func(ctx context.Context, flowRequest flowPromptRequest) (PromptResult, error) {
		req, conversionErr := flowRequest.toDomain()
		if conversionErr != nil {
			return PromptResult{}, xerr.Validation("ai.prompt: invalid currency metadata: " + conversionErr.Error())
		}
		if err := validatePromptMessages(req); err != nil {
			return PromptResult{}, err
		}
		if build == nil {
			return PromptResult{}, xerr.Internal("ai.prompt: provider builder is unavailable")
		}
		provider, err := build(global, LLMDecl{
			Provider: req.Provider,
			Model:    req.Model,
			BaseURL:  req.BaseURL,
		})
		if err != nil {
			return PromptResult{}, completionError(err)
		}
		return complete(ctx, provider, req)
	}).Description("Run one LLM attempt and return typed content and usage").Tag("ai", "prompt").Build()
}
