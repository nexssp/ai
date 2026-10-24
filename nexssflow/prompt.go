package nexssflow

import (
	"context"

	"github.com/nexssp/kernel/action"
	"github.com/nexssp/kernel/xerr"

	"github.com/nexssp/ai/engine/llm"
)

type providerBuilder func(GlobalOptions, LLMDecl) (llm.Provider, error)

// newPromptAction builds the archive's generic ai.prompt action. The provider
// is resolved per call, preserving the legacy action's mixed-provider behavior.
func newPromptAction(global GlobalOptions) action.AnyAction {
	return newPromptActionWith(global, buildProvider)
}

// newPromptActionWith keeps the provider seam explicit so tests can exercise
// the real action without making a provider or network call.
func newPromptActionWith(global GlobalOptions, build providerBuilder) action.AnyAction {
	return action.New("ai.prompt", func(ctx context.Context, req map[string]any) (map[string]any, error) {
		if build == nil {
			return nil, xerr.Internal("ai: provider builder is unavailable")
		}
		if err := validateUser(req); err != nil {
			return nil, err
		}
		provider, err := build(global, LLMDecl{
			Provider: stringOf(req["provider"]),
			Model:    stringOf(req["model"]),
			BaseURL:  stringOf(req["base_url"]),
		})
		if err != nil {
			return nil, completionError(err)
		}
		system := stringOf(req["system"])
		model := stringOf(req["model"])
		return completeAndMerge(ctx, provider, req, system, model)
	}).Description("Prompt an LLM and merge its response into the current Flow state").Tag("ai", "prompt").Build()
}
