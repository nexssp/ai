package nexssflow

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/nexssp/kernel/action"
	"github.com/nexssp/kernel/xerr"
)

// NFlowViewRequest carries one typed action result through the view bridge.
// Its JSON payload is private because the shape is the result type's shape, not
// a new public request DTO.
type NFlowViewRequest struct {
	payload json.RawMessage
}

// UnmarshalJSON preserves the arbitrary typed-result shape without exposing a
// map as the action request type.
func (r *NFlowViewRequest) UnmarshalJSON(data []byte) error {
	if r == nil {
		return errors.New("NFlow view request cannot be nil")
	}
	if !json.Valid(data) {
		return errors.New("NFlow view request must be valid JSON")
	}
	r.payload = append(r.payload[:0], data...)
	return nil
}

// MarshalJSON lets Kernel's struct-to-struct coercion preserve the original
// typed result when the request has already been decoded.
func (r NFlowViewRequest) MarshalJSON() ([]byte, error) {
	if !json.Valid(r.payload) {
		return nil, errors.New("NFlow view request has no valid JSON payload")
	}
	return r.payload, nil
}

// NFlowViewResponse is the typed action response. The response's JSON object is
// projected to a map only by the type-erased Flow bridge below, where dynamic
// NFlow expression selectors require it.
type NFlowViewResponse struct {
	payload json.RawMessage
}

func (r NFlowViewResponse) MarshalJSON() ([]byte, error) {
	if !json.Valid(r.payload) {
		return nil, errors.New("NFlow view response has no valid JSON payload")
	}
	return r.payload, nil
}

// nflowViewAction adapts the typed action's JSON-shaped response to the runtime
// object needed by Flow's existing snake_case selectors. It deliberately does
// not expose TypedPayload: Flow cannot statically prove arbitrary JSON shapes.
type nflowViewAction struct {
	typed   action.AnyAction
	payload action.TypedPayload
}

func newNFlowViewAction() action.AnyAction {
	typed := action.New[NFlowViewRequest, NFlowViewResponse]("ai.nflow.view", func(_ context.Context, request NFlowViewRequest) (NFlowViewResponse, error) {
		if len(request.payload) == 0 {
			return NFlowViewResponse{}, xerr.Validation("ai.nflow.view: action result must be an object")
		}
		return NFlowViewResponse{payload: append(json.RawMessage(nil), request.payload...)}, nil
	}).Description("Expose a typed AI action result as its JSON-visible NFlow object").Tag("ai", "nflow").Build()
	return &nflowViewAction{typed: typed, payload: typed}
}

func (a *nflowViewAction) Describe() *action.Meta { return a.typed.Describe() }

func (a *nflowViewAction) GetBindings() []action.Binding { return a.typed.GetBindings() }

func (a *nflowViewAction) GetAnyHooks() []action.AnyHook { return a.typed.GetAnyHooks() }

func (a *nflowViewAction) AddAnyHook(hooks ...action.AnyHook) { a.typed.AddAnyHook(hooks...) }

func (a *nflowViewAction) CloneWithHooks(hooks ...action.AnyHook) action.AnyAction {
	return &nflowViewAction{typed: a.typed.CloneWithHooks(hooks...), payload: a.payload}
}

func (a *nflowViewAction) ExecuteDecoded(ctx context.Context, decode action.DecodeFunc) (any, error) {
	value, err := a.typed.ExecuteDecoded(ctx, decode)
	if err != nil {
		return nil, err
	}
	return nflowViewExpressionValue(value)
}

func (a *nflowViewAction) DoAny(ctx context.Context, request any) (any, error) {
	value, err := a.typed.DoAny(ctx, request)
	if err != nil {
		return nil, err
	}
	return nflowViewExpressionValue(value)
}

func nflowViewExpressionValue(value any) (map[string]any, error) {
	response, ok := value.(NFlowViewResponse)
	if !ok {
		return nil, xerr.Internal("ai.nflow.view: typed action returned an unexpected response")
	}
	var view map[string]any
	if err := json.Unmarshal(response.payload, &view); err != nil {
		return nil, xerr.Validation("ai.nflow.view: decode action result", err)
	}
	if view == nil {
		return nil, xerr.Validation("ai.nflow.view: action result must be an object")
	}
	return view, nil
}
