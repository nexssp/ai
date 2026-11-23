package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

const (
	defaultResponses        = `["Mock response 1.","Mock response 2."]`
	invalidRequestErrorType = "invalid_request_error"
)

// responseList hands out one configured response per valid chat-completion request.
type responseList struct {
	mu        sync.Mutex
	responses []scriptedReply
	next      int
}

type scriptedReply struct {
	Content   *string            `json:"content,omitempty"`
	ToolCalls []functionToolCall `json:"tool_calls,omitempty"`
	Error     *apiErrorDetail    `json:"error,omitempty"`
	Status    int                `json:"status,omitempty"`
}

type functionToolCall struct {
	ID       string               `json:"id"`
	Type     string               `json:"type"`
	Function scriptedFunctionCall `json:"function"`
}

type scriptedFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type chatRequest struct {
	Model string `json:"model"`
}

type chatCompletion struct {
	ID      string           `json:"id"`
	Object  string           `json:"object"`
	Created int64            `json:"created"`
	Model   string           `json:"model"`
	Choices []completionItem `json:"choices"`
	Usage   completionUsage  `json:"usage"`
}

type completionItem struct {
	Index        int               `json:"index"`
	Message      completionMessage `json:"message"`
	FinishReason string            `json:"finish_reason"`
}

type completionMessage struct {
	Role      string             `json:"role"`
	Content   *string            `json:"content"`
	ToolCalls []functionToolCall `json:"tool_calls,omitempty"`
}

type completionUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type apiError struct {
	Error apiErrorDetail `json:"error"`
}

type apiErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}

func main() {
	listenAddress := flag.String("listen", "127.0.0.1:8765", "loopback TCP address to listen on")
	responseJSON := flag.String("responses", defaultResponses, "JSON array of content strings or scripted response objects, returned in order")
	flag.Parse()

	responses, err := parseResponses(*responseJSON)
	if err != nil {
		slog.Error("parse mock responses", "error", err)
		os.Exit(1)
	}
	listener, err := listenLoopback(*listenAddress)
	if err != nil {
		slog.Error("listen for mock", "error", err)
		os.Exit(1)
	}
	server := &http.Server{
		Handler:           newHandler(responses),
		ReadHeaderTimeout: 5 * time.Second,
	}
	slog.Info("OpenAI-compatible mock listening", "address", listener.Addr(), "responses", len(responses))
	if err := server.Serve(listener); err != nil {
		slog.Error("serve mock", "error", err)
		os.Exit(1)
	}
}

func parseResponses(raw string) ([]scriptedReply, error) {
	var entries []json.RawMessage
	if err := json.Unmarshal([]byte(raw), &entries); err != nil {
		return nil, fmt.Errorf("invalid -responses JSON: %w", err)
	}
	if len(entries) == 0 {
		return nil, errors.New("-responses must contain at least one response")
	}

	responses := make([]scriptedReply, len(entries))
	for i, entry := range entries {
		var content string
		if err := json.Unmarshal(entry, &content); err == nil {
			responses[i].Content = &content
		} else if err := json.Unmarshal(entry, &responses[i]); err != nil {
			return nil, fmt.Errorf("invalid -responses[%d]: expected a string or response object: %w", i, err)
		}
		if err := validateReply(responses[i]); err != nil {
			return nil, fmt.Errorf("invalid -responses[%d]: %w", i, err)
		}
	}
	return responses, nil
}

func validateReply(reply scriptedReply) error {
	if reply.Error != nil {
		if reply.Content != nil || len(reply.ToolCalls) != 0 {
			return errors.New("error replies cannot also contain content or tool_calls")
		}
		if strings.TrimSpace(reply.Error.Message) == "" || strings.TrimSpace(reply.Error.Type) == "" {
			return errors.New("error replies require non-empty error.message and error.type")
		}
		if reply.Status != 0 && (reply.Status < 400 || reply.Status > 599) {
			return errors.New("error status must be between 400 and 599")
		}
		return nil
	}
	if reply.Status != 0 {
		return errors.New("status is only valid for an error reply")
	}
	if reply.Content == nil && len(reply.ToolCalls) == 0 {
		return errors.New("response requires content or at least one function tool_call")
	}
	if reply.Content != nil && strings.TrimSpace(*reply.Content) == "" && len(reply.ToolCalls) == 0 {
		return errors.New("content-only replies must not be blank")
	}
	for i, call := range reply.ToolCalls {
		if strings.TrimSpace(call.ID) == "" || call.Type != "function" || strings.TrimSpace(call.Function.Name) == "" {
			return fmt.Errorf("tool_calls[%d] requires id, type \"function\", and function.name", i)
		}
		if !json.Valid([]byte(call.Function.Arguments)) {
			return fmt.Errorf("tool_calls[%d].function.arguments must be a JSON string", i)
		}
	}
	return nil
}

func listenLoopback(address string) (net.Listener, error) {
	host, _, err := net.SplitHostPort(address)
	if err != nil {
		return nil, fmt.Errorf("invalid listen address %q: %w", address, err)
	}
	ip := net.ParseIP(host)
	if ip == nil || !ip.IsLoopback() {
		return nil, fmt.Errorf("listen address %q must use a loopback IP literal", address)
	}
	var listenConfig net.ListenConfig
	return listenConfig.Listen(context.Background(), "tcp", address)
}

func newHandler(responses []scriptedReply) http.Handler {
	queue := &responseList{responses: responses}
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeAPIError(w, http.StatusMethodNotAllowed, apiErrorDetail{
				Message: "method not allowed",
				Type:    invalidRequestErrorType,
				Code:    "method_not_allowed",
			})
			return
		}
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		if _, err := w.Write([]byte("ok\n")); err != nil {
			slog.Error("write health response", "error", err)
		}
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", http.MethodPost)
			writeAPIError(w, http.StatusMethodNotAllowed, apiErrorDetail{
				Message: "method not allowed",
				Type:    invalidRequestErrorType,
				Code:    "method_not_allowed",
			})
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
		defer r.Body.Close()
		var request chatRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			writeAPIError(w, http.StatusBadRequest, apiErrorDetail{
				Message: "invalid JSON request",
				Type:    invalidRequestErrorType,
				Code:    "invalid_json",
			})
			return
		}

		reply, index, ok := queue.take()
		if !ok {
			writeAPIError(w, http.StatusBadRequest, apiErrorDetail{
				Message: "predefined mock responses exhausted",
				Type:    invalidRequestErrorType,
				Code:    "mock_responses_exhausted",
			})
			return
		}
		if reply.Error != nil {
			status := reply.Status
			if status == 0 {
				status = http.StatusBadRequest
			}
			writeAPIError(w, status, *reply.Error)
			return
		}
		model := request.Model
		if model == "" {
			model = "mock-model"
		}
		finishReason := "stop"
		if len(reply.ToolCalls) > 0 {
			finishReason = "tool_calls"
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(chatCompletion{
			ID:      fmt.Sprintf("chatcmpl-mock-%d", index),
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   model,
			Choices: []completionItem{{
				Index: 0,
				Message: completionMessage{
					Role:      "assistant",
					Content:   reply.Content,
					ToolCalls: reply.ToolCalls,
				},
				FinishReason: finishReason,
			}},
			Usage: completionUsage{},
		}); err != nil {
			slog.Error("write chat completion response", "error", err)
		}
	})
	return mux
}

func (q *responseList) take() (scriptedReply, int, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.next >= len(q.responses) {
		return scriptedReply{}, 0, false
	}
	q.next++
	return q.responses[q.next-1], q.next, true
}

func writeAPIError(w http.ResponseWriter, status int, detail apiErrorDetail) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(apiError{Error: detail}); err != nil {
		slog.Error("write API error response", "error", err)
	}
}
