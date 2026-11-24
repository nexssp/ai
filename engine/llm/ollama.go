// Ollama does not use SSE. Its /api/chat stream is newline-delimited JSON,
// terminated by a `done:true` field inside the last line. Its streaming
// reader therefore stays local to this file and is not routed through
// StreamSSE.
package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"
	"time"

	"github.com/nexssp/kernel/action"
	"github.com/nexssp/kernel/xerr"
	"github.com/nexssp/transport/codec"
)

type OllamaProvider struct {
	cfg        Config
	client     *http.Client
	codec      codec.Codec
	execAction *action.BuiltAction[Request, Response]
}

type ollamaChatPayload struct {
	Model     string          `json:"model"`
	Messages  []ollamaMessage `json:"messages"`
	Stream    bool            `json:"stream"`
	KeepAlive string          `json:"keep_alive"`
	Format    any             `json:"format,omitempty"`
	Tools     []ollamaTool    `json:"tools,omitempty"`
}

type ollamaMessage struct {
	Role      Role             `json:"role"`
	Content   string           `json:"content,omitempty"`
	Name      string           `json:"name,omitempty"`
	ToolCalls []ollamaToolCall `json:"tool_calls,omitempty"`
}

type ollamaTool struct {
	Type     string             `json:"type"`
	Function ollamaFunctionSpec `json:"function"`
}

type ollamaFunctionSpec struct {
	Name        string     `json:"name"`
	Description string     `json:"description,omitempty"`
	Parameters  JSONSchema `json:"parameters"`
}

type ollamaToolCall struct {
	Function ollamaFunctionCall `json:"function"`
}

type ollamaFunctionCall struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

type ollamaLifecyclePayload struct {
	Model     string `json:"model"`
	KeepAlive int    `json:"keep_alive"`
}

// ollamaWireResponse is the JSON shape of the /api/chat response.
type ollamaWireResponse struct {
	Model      string        `json:"model"`
	Message    ollamaMessage `json:"message"`
	Prompt     int           `json:"prompt_eval_count"`
	Eval       int           `json:"eval_count"`
	DoneReason string        `json:"done_reason"`
}

type ollamaStreamChunk struct {
	Message ollamaStreamMessage `json:"message"`
	Done    bool                `json:"done"`
}

type ollamaStreamMessage struct {
	Content string `json:"content"`
}

type ollamaRunningModelsResponse struct {
	Models []ollamaRunningModel `json:"models"`
}

type ollamaRunningModel struct {
	Name string `json:"name"`
}

func NewOllama(cfg Config) *OllamaProvider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "http://localhost:11434"
	}

	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = 120 * time.Second
	}

	maxRetries := cfg.MaxRetries
	if maxRetries <= 0 {
		maxRetries = 2
	}

	p := &OllamaProvider{
		cfg:    cfg,
		client: &http.Client{Timeout: timeout},
		codec:  codec.Default,
	}

	p.execAction = action.New("ollama.complete", func(ctx context.Context, req Request) (Response, error) {
		start := time.Now()

		if req.Model == "" {
			req.Model = p.cfg.DefaultModel
		}
		if req.Model == "" {
			return Response{}, xerr.Validation("ollama: model is required in the request or provider configuration")
		}

		payload, err := p.buildPayload(req, false)
		if err != nil {
			return Response{}, err
		}

		headers := map[string]string{}
		if p.cfg.APIKey != "" {
			headers["Authorization"] = "Bearer " + p.cfg.APIKey
		}

		raw, err := DoJSON[ollamaWireResponse](ctx, HTTPRequest{
			URL:          strings.TrimRight(p.cfg.BaseURL, "/") + "/api/chat",
			Headers:      headers,
			Body:         payload,
			Client:       p.client,
			Codec:        p.codec,
			ProviderName: p.Name(),
		})
		if err != nil {
			return Response{}, err
		}

		toolCalls := make([]ToolCall, 0, len(raw.Message.ToolCalls))
		for index, call := range raw.Message.ToolCalls {
			arguments, err := json.Marshal(call.Function.Arguments)
			if err != nil {
				return Response{}, xerr.Internal("ollama: encode tool call arguments failed", err)
			}
			toolCalls = append(toolCalls, ToolCall{ID: fmt.Sprintf("ollama-call-%d-%d", start.UnixNano(), index), Type: toolTypeFunction, Function: FunctionCall{Name: call.Function.Name, Arguments: string(arguments)}})
		}
		return Response{
			Content:      raw.Message.Content,
			Model:        raw.Model,
			PromptTokens: raw.Prompt,
			CompTokens:   raw.Eval,
			ToolCalls:    toolCalls,
			FinishReason: raw.DoneReason,
			Duration:     time.Since(start),
		}, nil
	}).
		Internal().
		Retry(maxRetries, action.ExponentialBackoff(200*time.Millisecond, 2*time.Second)).
		Timeout(timeout).
		Build()

	return p
}

func (p *OllamaProvider) Name() string {
	if p.cfg.Name != "" {
		return p.cfg.Name
	}

	return providerOllama
}

func (p *OllamaProvider) SupportsToolCalls() bool { return true }

func (p *OllamaProvider) SupportsToolChoice(choice ToolChoice) bool {
	return choice == ToolChoiceAuto
}

func (p *OllamaProvider) Capabilities(model string) ProviderCapabilities {
	return CapabilitiesForRoute(p.Name(), configuredModel(model, p.cfg.DefaultModel), p.cfg.BaseURL)
}

func (p *OllamaProvider) SupportsOutputSchema(model string, schema JSONSchema, withTools bool) (supported bool, reason string) {
	capabilities := p.Capabilities(model)
	level := capabilities.OutputSchema
	if withTools {
		level = capabilities.OutputSchemaWithTools
	}
	if level != CapabilitySupported && level != CapabilityConditional {
		return false, "route capability is " + string(level)
	}
	if err := schema.ValidateDefinition(); err != nil {
		return false, err.Error()
	}
	return true, ""
}

func (p *OllamaProvider) buildPayload(req Request, stream bool) (ollamaChatPayload, error) {
	payload := ollamaChatPayload{Model: req.Model, Stream: stream, KeepAlive: keepAlive24h}
	if req.OutputSchema != nil {
		payload.Format = *req.OutputSchema
	} else if req.JSONMode {
		payload.Format = "json"
	}
	for _, message := range req.Messages {
		converted := ollamaMessage{Role: message.Role, Content: message.Content, Name: message.Name}
		for _, call := range message.ToolCalls {
			arguments, err := DecodeJSONValue(call.Function.Arguments)
			if err != nil {
				return ollamaChatPayload{}, fmt.Errorf("ollama: tool call %q arguments: %w", call.Function.Name, err)
			}
			object, ok := arguments.(map[string]any)
			if !ok {
				return ollamaChatPayload{}, fmt.Errorf("ollama: tool call %q arguments must be a JSON object", call.Function.Name)
			}
			converted.ToolCalls = append(converted.ToolCalls, ollamaToolCall{Function: ollamaFunctionCall{Name: call.Function.Name, Arguments: object}})
		}
		payload.Messages = append(payload.Messages, converted)
	}
	if len(req.Tools) > 0 {
		for _, spec := range req.Tools {
			schema, err := ParseJSONSchema(spec.Function.Parameters)
			if err != nil {
				return ollamaChatPayload{}, fmt.Errorf("ollama: tool %q parameters: %w", spec.Function.Name, err)
			}
			payload.Tools = append(payload.Tools, ollamaTool{Type: spec.Type, Function: ollamaFunctionSpec{Name: spec.Function.Name, Description: spec.Function.Description, Parameters: schema}})
		}
	}
	if req.ToolChoice != "" && req.ToolChoice != ToolChoiceAuto {
		return ollamaChatPayload{}, fmt.Errorf("ollama: tool choice %q is not supported by the native /api/chat route", req.ToolChoice)
	}
	return payload, nil
}

func (p *OllamaProvider) Complete(ctx context.Context, req Request) (Response, error) {
	if req.Model == "" && p.cfg.DefaultModel == "" {
		return Response{}, xerr.Validation("ollama: model is required in the request or provider configuration")
	}
	return p.execAction.Do(ctx, req)
}

// Stream uses newline-delimited JSON, not SSE. Ollama's /api/chat with
// stream:true sends one JSON object per line, terminated by a `done:true`
// field.
func (p *OllamaProvider) Stream(ctx context.Context, req Request) (iter.Seq2[string, error], error) {
	if req.OutputSchema != nil || len(req.Tools) > 0 {
		return nil, xerr.Validation("ollama: structured outputs and tool calls are not supported by the text-only streaming adapter")
	}
	if req.Model == "" {
		req.Model = p.cfg.DefaultModel
	}
	if req.Model == "" {
		return nil, xerr.Validation("ollama: model is required in the request or provider configuration")
	}

	payload, err := p.buildPayload(req, true)
	if err != nil {
		return nil, err
	}

	bodyBytes, err := p.codec.Marshal(payload)
	if err != nil {
		return nil, xerr.Internal("ollama: marshal stream request failed", err)
	}

	url := strings.TrimRight(p.cfg.BaseURL, "/") + "/api/chat"

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, xerr.Internal("ollama: build stream request failed", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	if p.cfg.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.cfg.APIKey)
	}

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, mapProviderTransportError(err)
	}

	if resp.StatusCode != http.StatusOK {
		respBytes, readErr := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if readErr != nil {
			return nil, mapProviderTransportError(readErr)
		}
		return nil, FromStreamHTTPStatus(resp.StatusCode, respBytes, p.Name())
	}

	return streamOllamaJSONL(ctx, resp.Body), nil
}

// streamOllamaJSONL parses Ollama's newline-delimited JSON stream. Kept
// local to this file: Ollama is the only provider with this wire shape.
func streamOllamaJSONL(ctx context.Context, body io.ReadCloser) iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		defer body.Close()

		reader := bufio.NewReader(body)

		for {
			if ctx.Err() != nil {
				yield("", ctx.Err())

				return
			}

			line, err := reader.ReadBytes('\n')
			if err != nil {
				if err != io.EOF {
					yield("", err)
				}

				return
			}

			trimmed := bytes.TrimSpace(line)
			if len(trimmed) == 0 {
				continue
			}

			var chunk ollamaStreamChunk
			if err := json.Unmarshal(trimmed, &chunk); err != nil {
				continue
			}

			if chunk.Message.Content != "" {
				if !yield(chunk.Message.Content, nil) {
					return
				}
			}

			if chunk.Done {
				return
			}
		}
	}
}

func (p *OllamaProvider) Preload(ctx context.Context, model string) error {
	if model == "" {
		model = p.cfg.DefaultModel
	}
	if model == "" {
		return xerr.Validation("ollama: model is required for preload")
	}

	payload := ollamaLifecyclePayload{Model: model, KeepAlive: -1}

	bodyBytes, err := p.codec.Marshal(payload)
	if err != nil {
		return err
	}

	url := strings.TrimRight(p.cfg.BaseURL, "/") + "/api/generate"

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		body, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			return xerr.Unavailable("ollama preload response read failed", readErr)
		}

		return xerr.Unavailable(fmt.Sprintf("ollama preload failed (status %d: %s)", resp.StatusCode, string(body)))
	}

	return nil
}

func (p *OllamaProvider) Unload(ctx context.Context, model string) error {
	if model == "" {
		model = p.cfg.DefaultModel
	}
	if model == "" {
		return xerr.Validation("ollama: model is required for unload")
	}

	payload := ollamaLifecyclePayload{Model: model, KeepAlive: 0}

	bodyBytes, err := p.codec.Marshal(payload)
	if err != nil {
		return err
	}

	url := strings.TrimRight(p.cfg.BaseURL, "/") + "/api/generate"

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	return nil
}

func (p *OllamaProvider) IsLoaded(ctx context.Context, model string) bool {
	url := strings.TrimRight(p.cfg.BaseURL, "/") + "/api/ps"

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, http.NoBody)
	if err != nil {
		return false
	}

	resp, err := p.client.Do(req)
	if err != nil || resp.StatusCode != http.StatusOK {
		return false
	}
	defer func() { _ = resp.Body.Close() }()

	var psRes ollamaRunningModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&psRes); err != nil {
		return false
	}

	for _, m := range psRes.Models {
		if m.Name == model || strings.HasPrefix(m.Name, model+":") {
			return true
		}
	}

	return false
}
