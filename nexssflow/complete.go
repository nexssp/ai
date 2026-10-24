package nexssflow

import (
	"context"
	"strings"

	"github.com/nexssp/kernel/xerr"

	"github.com/nexssp/ai/engine/llm"
)

func validateUser(req map[string]any) error {
	if strings.TrimSpace(stringOf(req["user"])) == "" {
		return xerr.Validation("ai: user field is required")
	}
	return nil
}

// completeAndMerge runs one untyped completion and folds the response
// into the incoming map. Pipeline state survives; content, model, and
// token counts become additional keys.
//
// The merge is the whole point of CQRS at the action boundary: a
// pipeline stage must not lose `goal`, `attempt`, or any other field
// the surrounding flow depends on just because this call completed.
func completeAndMerge(
	ctx context.Context,
	provider llm.Provider,
	req map[string]any,
	system, model string,
) (map[string]any, error) {
	if err := validateUser(req); err != nil {
		return nil, err
	}
	if err := requireProvider(provider); err != nil {
		return nil, err
	}
	user := stringOf(req["user"])

	messages := make([]llm.Message, 0, 2)
	if system != "" {
		messages = append(messages, llm.Message{Role: llm.RoleSystem, Content: system})
	}
	messages = append(messages, llm.Message{Role: llm.RoleUser, Content: user})

	jsonMode, _ := req["json_mode"].(bool)
	resp, err := provider.Complete(ctx, llm.Request{
		Model:       model,
		Messages:    messages,
		MaxTokens:   intOf(req["max_tokens"]),
		Temperature: floatOf(req["temperature"]),
		JSONMode:    jsonMode,
	})
	if err != nil {
		return nil, completionError(err)
	}

	out := shallowCopy(req)
	out["content"] = resp.Content
	out["model"] = resp.Model
	out["prompt_tokens"] = resp.PromptTokens
	out["comp_tokens"] = resp.CompTokens
	out["total_tokens"] = resp.PromptTokens + resp.CompTokens

	return out, nil
}
