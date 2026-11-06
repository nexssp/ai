package nexssflow

import (
	"context"
	"testing"

	"github.com/nexssp/kernel/action"
)

type nflowViewFixture struct {
	CandidateID string `json:"candidate_id"`
	Attempt     struct {
		AttemptID string `json:"attempt_id"`
		Provider  string `json:"provider"`
	} `json:"attempt"`
}

func TestNFlowViewUsesJSONFieldNames(t *testing.T) {
	fixture := nflowViewFixture{CandidateID: "candidate-a"}
	fixture.Attempt.AttemptID = "attempt-a"
	fixture.Attempt.Provider = "openai"

	viewAction := newNFlowViewAction()
	if _, ok := viewAction.(action.TypedPayload); ok {
		t.Fatalf("view action type %T must not expose a static payload contract", viewAction)
	}
	value, err := viewAction.DoAny(context.Background(), fixture)
	if err != nil {
		t.Fatalf("view action: %v", err)
	}
	view, ok := value.(map[string]any)
	if !ok {
		t.Fatalf("view type = %T, want map[string]any", value)
	}
	if view["candidate_id"] != "candidate-a" {
		t.Fatalf("candidate_id = %#v, want candidate-a", view["candidate_id"])
	}
	attempt, ok := view["attempt"].(map[string]any)
	if !ok {
		t.Fatalf("attempt type = %T, want map[string]any", view["attempt"])
	}
	if attempt["attempt_id"] != "attempt-a" || attempt["provider"] != "openai" {
		t.Fatalf("attempt view = %#v", attempt)
	}
}
