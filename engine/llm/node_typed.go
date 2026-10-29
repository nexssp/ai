package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/nexssp/kernel/action"
	"github.com/nexssp/kernel/xerr"
)

// TypedReq is the input contract for every NewTypedNode-produced
// action. A caller builds one of these, the action forwards it to the
// provider in JSON mode, and the model's reply is unmarshalled into
// Res.
//
// The two fields are free-form strings so the node can be composed
// without knowing anything about the caller's domain. System carries
// the reviewer/architect/planner persona; User carries the payload
// rendered as text by an upstream projection.
type TypedReq struct {
	System string `json:"system,omitempty"`
	User   string `json:"user" validate:"required"`
}

// NewTypedNode returns an action that:
//
//  1. Sends {system, user} to the provider with JSONMode enabled.
//  2. Decodes the model's reply into Res.
//  3. Returns Res as the action output.
//
// This is the single primitive that lets every ai/* subsystem drop
// its own hand-rolled "prompt → call → unmarshal" boilerplate and
// keep only the DTO. The heavy lifting (structured output enforcement,
// JSON decoding, error normalisation) lives in exactly one place.
//
// The name is required and becomes the action's registry identity
// (e.g. "ai.structured.review_report"). Callers typically pass a
// fully qualified name so the .nflow file can reference it directly.
func NewTypedNode[Res any](
	name string,
	provider Provider,
	defaultSystem string,
) action.AnyAction {
	if name == "" {
		panic("flow: NewTypedNode requires a non-empty name")
	}
	if provider == nil {
		panic("flow: NewTypedNode requires a non-nil provider for " + name)
	}

	return action.New(name, func(ctx context.Context, req TypedReq) (Res, error) {
		var zero Res

		if strings.TrimSpace(req.User) == "" {
			return zero, xerr.BadRequest("typed node: user prompt is required")
		}

		system := req.System
		if system == "" {
			system = defaultSystem
		}

		messages := make([]Message, 0, 2)
		if system != "" {
			messages = append(messages, Message{
				Role:    RoleSystem,
				Content: system,
			})
		}
		messages = append(messages, Message{
			Role:    RoleUser,
			Content: req.User,
		})

		resp, err := provider.Complete(ctx, Request{
			Messages: messages,
			JSONMode: true,
		})
		if err != nil {
			return zero, xerr.Unavailable(
				fmt.Sprintf("typed node %q: provider failed", name), err,
			)
		}

		// Some reasoning providers put the JSON in ReasoningContent
		// when the visible Content was truncated. Prefer Content, fall
		// back to ReasoningContent only when Content is empty — never
		// the other way around, because ReasoningContent can contain
		// prose mixed with JSON.
		raw := strings.TrimSpace(resp.Content)
		if raw == "" {
			raw = strings.TrimSpace(resp.ReasoningContent)
		}
		if raw == "" {
			return zero, xerr.Validation(
				fmt.Sprintf("typed node %q: provider returned empty content", name),
			)
		}

		raw = stripCodeFence(raw)

		var out Res
		dec := json.NewDecoder(strings.NewReader(raw))
		dec.DisallowUnknownFields()

		if err := dec.Decode(&out); err != nil {
			preview := raw
			if len(preview) > 200 {
				preview = preview[:200] + "…"
			}

			return zero, xerr.Validation(fmt.Sprintf(
				"typed node %q: invalid JSON: %v\n              ├── preview   : %q",
				name, err, preview,
			))
		}

		// Reject trailing content after the object. This catches
		// "here is the JSON: {...} hope this helps" — a common LLM
		// failure mode that silently corrupts downstream parsing.
		var trailing any
		if dec.More() {
			return zero, xerr.Validation(fmt.Sprintf(
				"typed node %q: trailing data after JSON object", name,
			))
		}
		_ = trailing

		return out, nil
	}).
		Description("Structured LLM completion; returns "+fmt.Sprintf("%T", *new(Res))).
		Tag("ai", "structured").
		Build()
}

// stripCodeFence removes a markdown ```json ... ``` wrapper that a
// model occasionally adds despite JSONMode. Bounded, no regex.
func stripCodeFence(s string) string {
	if !strings.HasPrefix(s, "```") {
		return s
	}

	// Skip the opening fence and its optional language tag up to the
	// first newline. If there is no newline, the input is not a fence.
	before, _, ok := strings.Cut(s, "\n")
	if !ok {
		return s
	}
	nl := len(before)

	body := s[nl+1:]

	before, _, ok = strings.CutLast(body, "```")
	end := len(before)
	if !ok {
		return body
	}

	return strings.TrimSpace(body[:end])
}
