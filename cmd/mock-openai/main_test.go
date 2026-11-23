package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHandlerReturnsPredefinedResponsesInOrder(t *testing.T) {
	handler := newHandler(parseTestResponses(t, `["analyst draft","reviewer final"]`))
	for i, want := range []string{"analyst draft", "reviewer final"} {
		recorder := postCompletion(handler, `{"model":"mercury-3"}`)
		if recorder.Code != http.StatusOK {
			t.Fatalf("response %d status = %d, want %d: %s", i+1, recorder.Code, http.StatusOK, recorder.Body.String())
		}
		var got chatCompletion
		if err := json.Unmarshal(recorder.Body.Bytes(), &got); err != nil {
			t.Fatalf("decode response %d: %v", i+1, err)
		}
		if got.Object != "chat.completion" || got.Model != "mercury-3" || len(got.Choices) != 1 {
			t.Fatalf("response %d metadata = %#v", i+1, got)
		}
		if got.Choices[0].Message.Role != "assistant" || got.Choices[0].Message.Content == nil || *got.Choices[0].Message.Content != want {
			t.Fatalf("response %d message = %#v, want assistant %q", i+1, got.Choices[0].Message, want)
		}
		if got.ID != fmt.Sprintf("chatcmpl-mock-%d", i+1) {
			t.Fatalf("response %d id = %q", i+1, got.ID)
		}
	}
}

func TestHandlerReturnsFunctionToolCalls(t *testing.T) {
	handler := newHandler(parseTestResponses(t, `[{"tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Warsaw\"}"}}]}]`))
	recorder := postCompletion(handler, `{"model":"mock-model"}`)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d: %s", recorder.Code, http.StatusOK, recorder.Body.String())
	}
	var got chatCompletion
	if err := json.Unmarshal(recorder.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode tool-call response: %v", err)
	}
	if len(got.Choices) != 1 || got.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("choice = %#v, want finish_reason tool_calls", got.Choices)
	}
	message := got.Choices[0].Message
	if message.Content != nil {
		t.Fatalf("content = %v, want null for a tool-call-only response", message.Content)
	}
	if len(message.ToolCalls) != 1 {
		t.Fatalf("tool_calls = %#v, want one call", message.ToolCalls)
	}
	call := message.ToolCalls[0]
	if call.ID != "call-1" || call.Type != "function" || call.Function.Name != "lookup" || call.Function.Arguments != `{"city":"Warsaw"}` {
		t.Fatalf("tool call = %#v", call)
	}
}

func TestHandlerReturnsScriptedOpenAIErrorAndContinues(t *testing.T) {
	handler := newHandler(parseTestResponses(t, `[{"status":429,"error":{"message":"scripted failure","type":"invalid_request_error","code":"mock_error"}},"after error"]`))
	errorRecorder := postCompletion(handler, `{"model":"mock-model"}`)
	if errorRecorder.Code != http.StatusTooManyRequests {
		t.Fatalf("error status = %d, want %d", errorRecorder.Code, http.StatusTooManyRequests)
	}
	var gotError apiError
	if err := json.Unmarshal(errorRecorder.Body.Bytes(), &gotError); err != nil {
		t.Fatalf("decode error response: %v", err)
	}
	if gotError.Error.Message != "scripted failure" || gotError.Error.Type != "invalid_request_error" || gotError.Error.Code != "mock_error" {
		t.Fatalf("error = %#v", gotError.Error)
	}

	nextRecorder := postCompletion(handler, `{"model":"mock-model"}`)
	var got chatCompletion
	if err := json.Unmarshal(nextRecorder.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response after error: %v", err)
	}
	if nextRecorder.Code != http.StatusOK || got.Choices[0].Message.Content == nil || *got.Choices[0].Message.Content != "after error" {
		t.Fatalf("response after error = status %d, body %s", nextRecorder.Code, nextRecorder.Body.String())
	}

	defaultStatusHandler := newHandler(parseTestResponses(t, `[{"error":{"message":"default status","type":"invalid_request_error"}}]`))
	defaultStatusRecorder := postCompletion(defaultStatusHandler, `{"model":"mock-model"}`)
	if defaultStatusRecorder.Code != http.StatusBadRequest {
		t.Fatalf("default error status = %d, want %d", defaultStatusRecorder.Code, http.StatusBadRequest)
	}
}

func TestHandlerRejectsExhaustedResponses(t *testing.T) {
	handler := newHandler(parseTestResponses(t, `["only response"]`))
	first := postCompletion(handler, `{"model":"mock-model"}`)
	if first.Code != http.StatusOK {
		t.Fatalf("first response status = %d, want %d", first.Code, http.StatusOK)
	}
	second := postCompletion(handler, `{"model":"mock-model"}`)
	if second.Code != http.StatusBadRequest {
		t.Fatalf("exhausted response status = %d, want %d", second.Code, http.StatusBadRequest)
	}
	var got apiError
	if err := json.Unmarshal(second.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode exhaustion error: %v", err)
	}
	if got.Error.Code != "mock_responses_exhausted" {
		t.Fatalf("error code = %q, want mock_responses_exhausted", got.Error.Code)
	}
}

func TestParseResponsesRejectsInvalidScripts(t *testing.T) {
	for _, raw := range []string{
		`[]`,
		`[{}]`,
		`[{"status":200,"error":{"message":"bad status","type":"invalid_request_error"}}]`,
		`[{"tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"not-json"}}]}]`,
	} {
		if _, err := parseResponses(raw); err == nil {
			t.Errorf("parseResponses(%s) succeeded, want error", raw)
		}
	}
}

func TestListenLoopbackRejectsNonLoopbackAddresses(t *testing.T) {
	for _, address := range []string{"0.0.0.0:8765", ":8765", "192.0.2.1:8765", "localhost:8765"} {
		if listener, err := listenLoopback(address); err == nil {
			_ = listener.Close()
			t.Errorf("listenLoopback(%q) succeeded, want rejection", address)
		}
	}
	listener, err := listenLoopback("127.0.0.1:0")
	if err != nil {
		t.Fatalf("loopback listener rejected: %v", err)
	}
	_ = listener.Close()
}

func TestHandlerRejectsMalformedJSON(t *testing.T) {
	recorder := postCompletion(newHandler(parseTestResponses(t, `["unused"]`)), "{")
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusBadRequest)
	}
}

func parseTestResponses(t *testing.T, raw string) []scriptedReply {
	t.Helper()
	responses, err := parseResponses(raw)
	if err != nil {
		t.Fatalf("parse responses: %v", err)
	}
	return responses
}

func postCompletion(handler http.Handler, body string) *httptest.ResponseRecorder {
	recorder := httptest.NewRecorder()
	request := httptest.NewRequestWithContext(context.Background(), http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	handler.ServeHTTP(recorder, request)
	return recorder
}
