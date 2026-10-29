package llm

import (
	"bytes"
	"context"
	"strings"
	"text/template"

	"github.com/nexssp/kernel/action"
	"github.com/nexssp/kernel/ai/dag"
	"github.com/nexssp/kernel/xerr"
	"github.com/nexssp/transport/codec"
)

func cleanJSONPayload(raw string) []byte {
	trimmed := strings.TrimSpace(raw)
	if strings.HasPrefix(trimmed, "```") {
		// Strip the opening ```json or ``` code fence.
		if idx := strings.IndexByte(trimmed, '\n'); idx != -1 {
			trimmed = trimmed[idx+1:]
		}
		// Strip the closing ``` code fence.
		if idx := strings.LastIndex(trimmed, "```"); idx != -1 {
			trimmed = trimmed[:idx]
		}

		trimmed = strings.TrimSpace(trimmed)
	}

	return []byte(trimmed)
}

func NewNode(id, _ string, p Provider, systemPrompt, userPromptTmpl string) action.AnyAction {
	parsedTmpl := template.Must(template.New(id).Parse(userPromptTmpl))

	return action.New(id, func(ctx context.Context, nCtx *dag.NodeContext) (string, error) {
		renderedUserPrompt, err := renderPrompt(parsedTmpl, nCtx.Input)
		if err != nil {
			return "", err
		}

		req := Request{
			Messages: []Message{
				{Role: RoleSystem, Content: systemPrompt},
				{Role: RoleUser, Content: renderedUserPrompt},
			},
		}

		res, err := p.Complete(ctx, req)
		if err != nil {
			return "", err
		}

		return res.Content, nil
	}).Build()
}

func NewStructuredNode[T any](id, _ string, p Provider, systemPrompt, userPromptTmpl string) action.AnyAction {
	parsedTmpl := template.Must(template.New(id).Parse(userPromptTmpl))

	return action.New(id, func(ctx context.Context, nCtx *dag.NodeContext) (T, error) {
		var zero T

		renderedUserPrompt, err := renderPrompt(parsedTmpl, nCtx.Input)
		if err != nil {
			return zero, err
		}

		req := Request{
			JSONMode: true,
			Messages: []Message{
				{Role: RoleSystem, Content: systemPrompt + "\nRespond with valid JSON."},
				{Role: RoleUser, Content: renderedUserPrompt},
			},
		}

		res, err := p.Complete(ctx, req)
		if err != nil {
			return zero, err
		}

		payload := cleanJSONPayload(res.Content)

		var target T
		if err := codec.Default.Unmarshal(payload, &target); err != nil {
			return zero, xerr.Internal("llm: failed to decode structured output", err)
		}

		return target, nil
	}).Build()
}

// renderPrompt executes tmpl against the node's read-only input view.
//
// ReadState is the correct parameter type here: it is what
// dag.NodeContext.Input actually holds. ReadState.Data() returns a
// shallow copy of the underlying map, which is all we need — the
// template only reads values, never mutates them.
func renderPrompt(tmpl *template.Template, state dag.ReadState) (string, error) {
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, state.Data()); err != nil {
		return "", xerr.Internal("llm: prompt template rendering failed", err)
	}
	return buf.String(), nil
}
