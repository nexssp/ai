package nexssflow

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestFlowJSONObjectAdaptersCopyIntoDomainRawMessages(t *testing.T) {
	state := flowJSONObject{"task": "route", "request": "hello"}
	wantState, err := json.Marshal(state)
	if err != nil {
		t.Fatalf("marshal expected state: %v", err)
	}
	decision, err := (flowDecisionActionRequest{State: state}).toDomain()
	if err != nil {
		t.Fatalf("convert decision state: %v", err)
	}
	if !bytes.Equal(decision.State, wantState) {
		t.Fatalf("decision state = %s, want %s", decision.State, wantState)
	}

	contextValue := flowJSONObject{"job": "keep-explicitly"}
	wantContext, err := json.Marshal(contextValue)
	if err != nil {
		t.Fatalf("marshal expected context: %v", err)
	}
	prompt, err := (flowPromptRequest{Context: contextValue}).toDomain()
	if err != nil {
		t.Fatalf("convert prompt context: %v", err)
	}
	if !bytes.Equal(prompt.Context, wantContext) {
		t.Fatalf("prompt context = %s, want %s", prompt.Context, wantContext)
	}

	state["task"] = "changed"
	contextValue["job"] = "changed"
	if !bytes.Equal(decision.State, wantState) {
		t.Fatalf("decision state aliases Flow JSON input: got %s, want %s", decision.State, wantState)
	}
	if !bytes.Equal(prompt.Context, wantContext) {
		t.Fatalf("prompt context aliases Flow JSON input: got %s, want %s", prompt.Context, wantContext)
	}
}

func TestFlowJSONObjectAdaptersRejectInvalidJSONValues(t *testing.T) {
	invalidJSON := flowJSONObject{"unsupported": make(chan int)}
	if _, err := (flowDecisionActionRequest{State: invalidJSON}).toDomain(); err == nil || !strings.Contains(err.Error(), "state must be a valid JSON value") {
		t.Fatalf("decision state error = %v, want invalid JSON value error", err)
	}
	if _, err := (flowPromptRequest{Context: invalidJSON}).toDomain(); err == nil || !strings.Contains(err.Error(), "context must be a valid JSON value") {
		t.Fatalf("prompt context error = %v, want invalid JSON value error", err)
	}
}
