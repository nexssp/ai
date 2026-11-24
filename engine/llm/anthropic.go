package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
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

type AnthropicProvider struct {
	cfg        Config
	client     *http.Client
	codec      codec.Codec
	execAction *action.BuiltAction[Request, Response]
}

type anthropicRequestMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type anthropicTool struct {
	Name        string     `json:"name"`
	Description string     `json:"description,omitempty"`
	InputSchema JSONSchema `json:"input_schema"`
	Strict      bool       `json:"strict,omitempty"`
}

type anthropicToolChoice struct {
	Type string `json:"type"`
}

type anthropicRequestPayload struct {
	Model        string                    `json:"model"`
	MaxTokens    int                       `json:"max_tokens"`
	Messages     []anthropicRequestMessage `json:"messages"`
	System       string                    `json:"system,omitempty"`
	OutputConfig *anthropicOutputConfig    `json:"output_config,omitempty"`
	Tools        []anthropicTool           `json:"tools,omitempty"`
	ToolChoice   *anthropicToolChoice      `json:"tool_choice,omitempty"`
	Stream       bool                      `json:"stream,omitempty"`
}

type anthropicOutputConfig struct {
	Format anthropicOutputFormat `json:"format"`
}

type anthropicOutputFormat struct {
	Type   string     `json:"type"`
	Schema JSONSchema `json:"schema"`
}

// anthropicWireResponse is the JSON shape of the /messages response.
type anthropicWireResponse struct {
	Model      string                  `json:"model"`
	Content    []anthropicContentBlock `json:"content"`
	Usage      anthropicWireUsage      `json:"usage"`
	StopReason string                  `json:"stop_reason"`
}

type anthropicContentBlock struct {
	Type  string          `json:"type"`
	Text  string          `json:"text,omitempty"`
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
}

type anthropicWireUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

func NewAnthropic(cfg Config) *AnthropicProvider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.anthropic.com/v1"
	}

	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = 60 * time.Second
	}

	maxRetries := cfg.MaxRetries
	if maxRetries <= 0 {
		maxRetries = 2
	}

	p := &AnthropicProvider{
		cfg:    cfg,
		client: &http.Client{Timeout: timeout},
		codec:  codec.Default,
	}

	p.execAction = action.New("anthropic.complete", func(ctx context.Context, req Request) (Response, error) {
		start := time.Now()

		if req.Model == "" {
			req.Model = p.cfg.DefaultModel
		}
		if req.Model == "" {
			return Response{}, xerr.Validation("anthropic: model is required in the request or provider configuration")
		}

		payload, err := p.buildPayload(req, false)
		if err != nil {
			return Response{}, err
		}

		headers := map[string]string{
			"X-Api-Key":         p.cfg.APIKey,
			"Anthropic-Version": "2023-06-01",
		}

		raw, err := DoJSON[anthropicWireResponse](ctx, HTTPRequest{
			URL:          strings.TrimRight(p.cfg.BaseURL, "/") + "/messages",
			Headers:      headers,
			Body:         payload,
			Client:       p.client,
			Codec:        p.codec,
			ProviderName: p.Name(),
		})
		if err != nil {
			return Response{}, err
		}

		var fullText strings.Builder
		var toolCalls []ToolCall

		for _, block := range raw.Content {
			switch block.Type {
			case "text":
				fullText.WriteString(block.Text)
			case "tool_use":
				arguments := string(block.Input)
				if len(block.Input) == 0 {
					arguments = "{}"
				}
				toolCalls = append(toolCalls, ToolCall{ID: block.ID, Type: toolTypeFunction, Function: FunctionCall{Name: block.Name, Arguments: arguments}})
			}
		}

		model := raw.Model
		if model == "" {
			model = req.Model
		}
		return Response{
			Content:      fullText.String(),
			Model:        model,
			PromptTokens: raw.Usage.InputTokens,
			CompTokens:   raw.Usage.OutputTokens,
			FinishReason: raw.StopReason,
			ToolCalls:    toolCalls,
			Duration:     time.Since(start),
		}, nil
	}).
		Internal().
		Retry(maxRetries, action.ExponentialBackoff(200*time.Millisecond, 2*time.Second)).
		Timeout(timeout).
		Build()

	return p
}

func (p *AnthropicProvider) Name() string {
	if p.cfg.Name != "" {
		return p.cfg.Name
	}

	return providerAnthropic
}

func (p *AnthropicProvider) SupportsToolCalls() bool { return true }

func (p *AnthropicProvider) SupportsToolChoice(choice ToolChoice) bool {
	return choice == ToolChoiceAuto || choice == ToolChoiceRequired
}

func (p *AnthropicProvider) Capabilities(model string) ProviderCapabilities {
	return CapabilitiesForRoute(p.Name(), configuredModel(model, p.cfg.DefaultModel), p.cfg.BaseURL)
}

func (p *AnthropicProvider) SupportsOutputSchema(model string, schema JSONSchema, withTools bool) (supported bool, reason string) {
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

func (p *AnthropicProvider) Complete(ctx context.Context, req Request) (Response, error) {
	if req.Model == "" && p.cfg.DefaultModel == "" {
		return Response{}, xerr.Validation("anthropic: model is required in the request or provider configuration")
	}
	return p.execAction.Do(ctx, req)
}

func (p *AnthropicProvider) Stream(ctx context.Context, req Request) (iter.Seq2[string, error], error) {
	if req.OutputSchema != nil || len(req.Tools) > 0 {
		return nil, xerr.Validation("anthropic: structured outputs and tool calls are not supported by the text-only streaming adapter")
	}
	if req.Model == "" {
		req.Model = p.cfg.DefaultModel
	}
	if req.Model == "" {
		return nil, xerr.Validation("anthropic: model is required in the request or provider configuration")
	}

	payload, err := p.buildPayload(req, true)
	if err != nil {
		return nil, err
	}

	bodyBytes, err := p.codec.Marshal(payload)
	if err != nil {
		return nil, xerr.Internal("anthropic: marshal stream request failed", err)
	}

	url := strings.TrimRight(p.cfg.BaseURL, "/") + "/messages"

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, xerr.Internal("anthropic: build stream request failed", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("X-Api-Key", p.cfg.APIKey)
	httpReq.Header.Set("Anthropic-Version", "2023-06-01")

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

	return StreamSSE(ctx, resp.Body, DecodeAnthropicChunk), nil
}

// buildPayload converts an llm.Request into the Anthropic-native payload
// shape. Deterministic; no I/O; safe to call from tests.
type anthropicRequestBlock struct {
	Type      string `json:"type"`
	Text      string `json:"text,omitempty"`
	ID        string `json:"id,omitempty"`
	Name      string `json:"name,omitempty"`
	Input     any    `json:"input,omitempty"`
	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   string `json:"content,omitempty"`
}

func (p *AnthropicProvider) buildPayload(req Request, stream bool) (anthropicRequestPayload, error) {
	systemPrompts, messages, err := anthropicMessages(req.Messages)
	if err != nil {
		return anthropicRequestPayload{}, err
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 2048
	}

	payload := anthropicRequestPayload{Model: req.Model, MaxTokens: maxTokens, Messages: messages, Stream: stream}
	if len(systemPrompts) > 0 {
		payload.System = strings.Join(systemPrompts, "\n\n")
	}
	if req.OutputSchema != nil {
		payload.OutputConfig = &anthropicOutputConfig{Format: anthropicOutputFormat{Type: "json_schema", Schema: *req.OutputSchema}}
	}
	for _, spec := range req.Tools {
		schema, err := ParseJSONSchema(spec.Function.Parameters)
		if err != nil {
			return anthropicRequestPayload{}, fmt.Errorf("anthropic: tool %q parameters: %w", spec.Function.Name, err)
		}
		payload.Tools = append(payload.Tools, anthropicTool{Name: spec.Function.Name, Description: spec.Function.Description, InputSchema: schema, Strict: spec.Function.Strict})
	}
	if req.ToolChoice != "" {
		choiceType := map[ToolChoice]string{ToolChoiceAuto: "auto", ToolChoiceRequired: "any"}[req.ToolChoice]
		if choiceType == "" {
			return anthropicRequestPayload{}, fmt.Errorf("anthropic: unsupported tool choice %q", req.ToolChoice)
		}
		payload.ToolChoice = &anthropicToolChoice{Type: choiceType}
	}
	return payload, nil
}

func anthropicMessages(source []Message) ([]string, []anthropicRequestMessage, error) {
	var systemPrompts []string
	var messages []anthropicRequestMessage
	for index := 0; index < len(source); index++ {
		message := source[index]
		switch message.Role {
		case RoleSystem:
			if message.Content != "" {
				systemPrompts = append(systemPrompts, message.Content)
			}
		case RoleUser:
			messages = append(messages, anthropicRequestMessage{Role: "user", Content: message.Content})
		case RoleAssistant:
			converted, err := anthropicAssistantMessage(message)
			if err != nil {
				return nil, nil, err
			}
			messages = append(messages, converted)
		case RoleTool:
			converted, nextIndex, err := anthropicToolResultMessage(source, index)
			if err != nil {
				return nil, nil, err
			}
			messages = append(messages, converted)
			index = nextIndex - 1
		default:
			return nil, nil, fmt.Errorf("anthropic: unsupported message role %q", message.Role)
		}
	}
	return systemPrompts, messages, nil
}

func anthropicAssistantMessage(message Message) (anthropicRequestMessage, error) {
	if len(message.ToolCalls) == 0 {
		return anthropicRequestMessage{Role: "assistant", Content: message.Content}, nil
	}
	blocks := make([]anthropicRequestBlock, 0, len(message.ToolCalls)+1)
	if message.Content != "" {
		blocks = append(blocks, anthropicRequestBlock{Type: "text", Text: message.Content})
	}
	for _, call := range message.ToolCalls {
		input, err := DecodeJSONValue(call.Function.Arguments)
		if err != nil {
			return anthropicRequestMessage{}, fmt.Errorf("anthropic: tool call %q arguments: %w", call.Function.Name, err)
		}
		if _, ok := input.(map[string]any); !ok {
			return anthropicRequestMessage{}, fmt.Errorf("anthropic: tool call %q arguments must be a JSON object", call.Function.Name)
		}
		blocks = append(blocks, anthropicRequestBlock{Type: "tool_use", ID: call.ID, Name: call.Function.Name, Input: input})
	}
	return anthropicRequestMessage{Role: "assistant", Content: blocks}, nil
}

func anthropicToolResultMessage(messages []Message, start int) (anthropicRequestMessage, int, error) {
	blocks := make([]anthropicRequestBlock, 0, 2)
	index := start
	for index < len(messages) && messages[index].Role == RoleTool {
		result := messages[index]
		if result.ToolCallID == "" {
			return anthropicRequestMessage{}, 0, errors.New("anthropic: tool result is missing tool_call_id")
		}
		blocks = append(blocks, anthropicRequestBlock{Type: "tool_result", ToolUseID: result.ToolCallID, Content: result.Content})
		index++
	}
	return anthropicRequestMessage{Role: "user", Content: blocks}, index, nil
}
