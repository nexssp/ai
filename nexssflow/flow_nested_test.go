package nexssflow

import (
	"context"
	"testing"

	"github.com/nexssp/flow/native"
	"github.com/nexssp/flow/runner"
	"github.com/nexssp/kernel/action"
)

type nestedDecisionEnvelope struct {
	AttemptID string                 `json:"attempt_id"`
	Decision  *DecisionActionRequest `json:"decision"`
}

func TestFlowNestedTypedActionConversion(t *testing.T) {
	input := map[string]any{
		"decision": map[string]any{
			"attempt_id":        "decision-attempt-1",
			"provider":          "fake-decision",
			"model":             "decision-model",
			"completion_tokens": 16,
			"max_output_tokens": 16,
			"state":             map[string]any{"task": "route"},
			"questions":         []any{},
		},
	}
	converted, err := action.Coerce[nestedDecisionEnvelope](input)
	if err != nil || converted.Decision == nil || converted.Decision.AttemptID != "decision-attempt-1" {
		t.Fatalf("Kernel action.Coerce nested pointer = %+v, err=%v", converted, err)
	}

	bundle := bundleWithBuilders(GlobalOptions{}, nil, nil)
	bundle.Libraries[0].Actions = append(bundle.Libraries[0].Actions, action.New[nestedDecisionEnvelope, string]("test.read_nested", func(_ context.Context, req nestedDecisionEnvelope) (string, error) {
		if req.Decision == nil {
			return "nil", nil
		}
		return req.Decision.AttemptID, nil
	}).Build())
	cfg, err := runner.BuildConfig(append(native.Bundles(), bundle))
	if err != nil {
		t.Fatalf("BuildConfig: %v", err)
	}
	source := `{
		decision: {
				attempt_id: "decision-attempt-1",
				provider: "fake-decision",
				model: "decision-model",
				completion_tokens: 16,
				max_output_tokens: 16,
				state: {task: "route"},
			questions: []
		}
	} -> test.read_nested`
	execution, err := runner.Execute(context.Background(), cfg, source, "nested-decision-test.nflow", nil)
	if err != nil {
		t.Fatalf("runner.Execute: %v", err)
	}
	if execution.Output != "decision-attempt-1" {
		t.Fatalf("typed action received nested attempt_id = %#v, want decision-attempt-1", execution.Output)
	}
}

func TestFlowChainsNestedTypedOutputIntoTypedAction(t *testing.T) {
	bundle := bundleWithBuilders(GlobalOptions{}, nil, nil)
	bundle.Libraries[0].Actions = append(bundle.Libraries[0].Actions,
		action.New[struct{}, nestedDecisionEnvelope]("test.output_nested", func(context.Context, struct{}) (nestedDecisionEnvelope, error) {
			return nestedDecisionEnvelope{AttemptID: "decision-attempt-1", Decision: &DecisionActionRequest{AttemptID: "decision-attempt-1", Model: "decision-model", CompletionTokens: 16, MaxOutputTokens: 16}}, nil
		}).Build(),
		action.New[nestedDecisionEnvelope, string]("test.read_output", func(_ context.Context, req nestedDecisionEnvelope) (string, error) {
			if req.Decision == nil {
				return "nil", nil
			}
			return req.AttemptID + ":" + req.Decision.AttemptID, nil
		}).Build(),
	)
	cfg, err := runner.BuildConfig(append(native.Bundles(), bundle))
	if err != nil {
		t.Fatalf("BuildConfig: %v", err)
	}
	execution, err := runner.Execute(context.Background(), cfg, "test.output_nested -> test.read_output", "nested-output-test.nflow", nil)
	if err != nil {
		t.Fatalf("runner.Execute: %v", err)
	}
	if execution.Output != "decision-attempt-1:decision-attempt-1" {
		t.Fatalf("typed output chaining = %#v, want both nested typed fields", execution.Output)
	}
}

func TestFlowProjectionCanAccessTypedResultThroughExplicitRoot(t *testing.T) {
	bundle := bundleWithBuilders(GlobalOptions{}, nil, nil)
	bundle.Libraries[0].Actions = append(bundle.Libraries[0].Actions, action.New[struct{}, nestedDecisionEnvelope]("test.output_nested", func(context.Context, struct{}) (nestedDecisionEnvelope, error) {
		return nestedDecisionEnvelope{AttemptID: "decision-attempt-1", Decision: &DecisionActionRequest{AttemptID: "decision-attempt-1", Model: "decision-model", CompletionTokens: 16, MaxOutputTokens: 16}}, nil
	}).Build())
	cfg, err := runner.BuildConfig(append(native.Bundles(), bundle))
	if err != nil {
		t.Fatalf("BuildConfig: %v", err)
	}
	source := `test.output_nested -> {attempt_id: __root__.AttemptID, model: __root__.Decision.Model}`
	execution, err := runner.Execute(context.Background(), cfg, source, "nested-output-root-test.nflow", nil)
	if err != nil {
		t.Fatalf("runner.Execute: %v", err)
	}
	output, ok := execution.Output.(map[string]any)
	if !ok || output["attempt_id"] != "decision-attempt-1" || output["model"] != "decision-model" {
		t.Fatalf("explicit-root typed output projection = %#v, want decision attempt fields", execution.Output)
	}
}
