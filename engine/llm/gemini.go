package llm

import (
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

type GeminiProvider struct {
	cfg        Config
	client     *http.Client
	codec      codec.Codec
	execAction *action.BuiltAction[Request, Response]
}

type geminiPart struct {
	Text             string                  `json:"text,omitempty"`
	FunctionCall     *geminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResponse `json:"functionResponse,omitempty"`
}

type geminiFunctionCall struct {
	ID   string         `json:"id,omitempty"`
	Name string         `json:"name"`
	Args map[string]any `json:"args,omitempty"`
}

type geminiFunctionResponse struct {
	ID       string         `json:"id,omitempty"`
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiGenerationConfig struct {
	ResponseMimeType string        `json:"responseMimeType,omitempty"`
	MaxOutputTokens  int           `json:"maxOutputTokens,omitempty"`
	ResponseSchema   *geminiSchema `json:"responseSchema,omitempty"`
}

type geminiSchema struct {
	Type        string                  `json:"type,omitempty"`
	Description string                  `json:"description,omitempty"`
	Format      string                  `json:"format,omitempty"`
	Nullable    bool                    `json:"nullable,omitempty"`
	Properties  map[string]geminiSchema `json:"properties,omitempty"`
	Required    []string                `json:"required,omitempty"`
	Items       *geminiSchema           `json:"items,omitempty"`
	Enum        []string                `json:"enum,omitempty"`
}

type geminiFunctionDeclaration struct {
	Name        string       `json:"name"`
	Description string       `json:"description,omitempty"`
	Parameters  geminiSchema `json:"parameters"`
}

type geminiTool struct {
	FunctionDeclarations []geminiFunctionDeclaration `json:"functionDeclarations"`
}

type geminiFunctionCallingConfig struct {
	Mode                 string   `json:"mode,omitempty"`
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

type geminiToolConfig struct {
	FunctionCallingConfig geminiFunctionCallingConfig `json:"functionCallingConfig"`
}

type geminiRequestPayload struct {
	Contents          []geminiContent         `json:"contents"`
	SystemInstruction *geminiContent          `json:"systemInstruction,omitempty"`
	GenerationConfig  *geminiGenerationConfig `json:"generationConfig,omitempty"`
	Tools             []geminiTool            `json:"tools,omitempty"`
	ToolConfig        *geminiToolConfig       `json:"toolConfig,omitempty"`
}

// geminiWireResponse is the JSON shape of the :generateContent response.
type geminiWireResponse struct {
	Candidates    []geminiWireCandidate `json:"candidates"`
	UsageMetadata geminiWireUsage       `json:"usageMetadata"`
}

type geminiWireCandidate struct {
	Content      geminiWireContent `json:"content"`
	FinishReason string            `json:"finishReason"`
}

type geminiWireContent struct {
	Parts []geminiWirePart `json:"parts"`
}

type geminiWirePart struct {
	Text         string              `json:"text"`
	FunctionCall *geminiFunctionCall `json:"functionCall,omitempty"`
}

type geminiWireUsage struct {
	PromptTokenCount int `json:"promptTokenCount"`
	CandidatesTokens int `json:"candidatesTokenCount"`
}

func NewGemini(cfg Config) *GeminiProvider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://generativelanguage.googleapis.com/v1beta"
	}

	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = 60 * time.Second
	}

	maxRetries := cfg.MaxRetries
	if maxRetries <= 0 {
		maxRetries = 2
	}

	p := &GeminiProvider{
		cfg:    cfg,
		client: &http.Client{Timeout: timeout},
		codec:  codec.Default,
	}

	p.execAction = action.New("gemini.complete", func(ctx context.Context, req Request) (Response, error) {
		start := time.Now()

		if req.Model == "" {
			req.Model = p.cfg.DefaultModel
		}
		if req.Model == "" {
			return Response{}, xerr.Validation("gemini: model is required in the request or provider configuration")
		}

		payload, err := p.buildPayload(req)
		if err != nil {
			return Response{}, err
		}

		// Gemini carries the API key in the URL, so no Authorization header.
		url := fmt.Sprintf("%s/models/%s:generateContent?key=%s",
			strings.TrimRight(p.cfg.BaseURL, "/"), req.Model, p.cfg.APIKey)

		raw, err := DoJSON[geminiWireResponse](ctx, HTTPRequest{
			URL:          url,
			Body:         payload,
			Client:       p.client,
			Codec:        p.codec,
			ProviderName: p.Name(),
		})
		if err != nil {
			return Response{}, err
		}

		if len(raw.Candidates) == 0 {
			return Response{}, xerr.Internal("gemini: empty response candidates", nil)
		}
		var content strings.Builder
		var toolCalls []ToolCall
		for index, part := range raw.Candidates[0].Content.Parts {
			content.WriteString(part.Text)
			if part.FunctionCall == nil {
				continue
			}
			arguments, err := json.Marshal(part.FunctionCall.Args)
			if err != nil {
				return Response{}, xerr.Internal("gemini: encode function call arguments failed", err)
			}
			id := part.FunctionCall.ID
			if id == "" {
				id = fmt.Sprintf("gemini-call-%d-%d", start.UnixNano(), index)
			}
			toolCalls = append(toolCalls, ToolCall{ID: id, Type: toolTypeFunction, Function: FunctionCall{Name: part.FunctionCall.Name, Arguments: string(arguments)}})
		}

		return Response{
			Content:      content.String(),
			Model:        req.Model,
			PromptTokens: raw.UsageMetadata.PromptTokenCount,
			CompTokens:   raw.UsageMetadata.CandidatesTokens,
			FinishReason: raw.Candidates[0].FinishReason,
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

func (p *GeminiProvider) Name() string {
	if p.cfg.Name != "" {
		return p.cfg.Name
	}

	return providerGemini
}

func (p *GeminiProvider) SupportsToolCalls() bool { return true }

func (p *GeminiProvider) SupportsToolChoice(choice ToolChoice) bool {
	return choice == ToolChoiceAuto || choice == ToolChoiceNone || choice == ToolChoiceRequired
}

func (p *GeminiProvider) Capabilities(model string) ProviderCapabilities {
	return CapabilitiesForRoute(p.Name(), configuredModel(model, p.cfg.DefaultModel), p.cfg.BaseURL)
}

func (p *GeminiProvider) SupportsOutputSchema(model string, schema JSONSchema, withTools bool) (supported bool, reason string) {
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
	if hasJSONSchemaAdditionalProperties(schema) {
		return false, "Gemini responseSchema does not expose additionalProperties; use best_effort for local validation"
	}
	return true, ""
}

func (p *GeminiProvider) Complete(ctx context.Context, req Request) (Response, error) {
	if req.Model == "" && p.cfg.DefaultModel == "" {
		return Response{}, xerr.Validation("gemini: model is required in the request or provider configuration")
	}
	return p.execAction.Do(ctx, req)
}

func (p *GeminiProvider) Stream(ctx context.Context, req Request) (iter.Seq2[string, error], error) {
	if req.OutputSchema != nil || len(req.Tools) > 0 {
		return nil, xerr.Validation("gemini: structured outputs and function calls are not supported by the text-only streaming adapter")
	}
	if req.Model == "" {
		req.Model = p.cfg.DefaultModel
	}
	if req.Model == "" {
		return nil, xerr.Validation("gemini: model is required in the request or provider configuration")
	}

	payload, err := p.buildPayload(req)
	if err != nil {
		return nil, err
	}
	bodyBytes, err := p.codec.Marshal(payload)
	if err != nil {
		return nil, xerr.Internal("gemini: marshal stream failed", err)
	}

	url := fmt.Sprintf("%s/models/%s:streamGenerateContent?alt=sse&key=%s",
		strings.TrimRight(p.cfg.BaseURL, "/"), req.Model, p.cfg.APIKey)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, xerr.Internal("gemini: build stream request failed", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

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

	return StreamSSE(ctx, resp.Body, DecodeGeminiChunk), nil
}

// buildPayload converts an llm.Request into the Gemini-native payload shape.
// Deterministic; no I/O; safe to call from tests.
func (p *GeminiProvider) buildPayload(req Request) (geminiRequestPayload, error) {
	contents, systemPrompts, err := geminiMessages(req.Messages)
	if err != nil {
		return geminiRequestPayload{}, err
	}

	payload := geminiRequestPayload{Contents: contents}
	if len(systemPrompts) > 0 {
		payload.SystemInstruction = &geminiContent{Parts: []geminiPart{{Text: strings.Join(systemPrompts, "\n\n")}}}
	}
	if req.JSONMode || req.MaxTokens > 0 || req.OutputSchema != nil {
		generationConfig := &geminiGenerationConfig{MaxOutputTokens: req.MaxTokens}
		if req.JSONMode || req.OutputSchema != nil {
			generationConfig.ResponseMimeType = "application/json"
		}
		if req.OutputSchema != nil {
			converted := geminiSchemaFromJSON(*req.OutputSchema)
			generationConfig.ResponseSchema = &converted
		}
		payload.GenerationConfig = generationConfig
	}
	for _, spec := range req.Tools {
		schema, err := ParseJSONSchema(spec.Function.Parameters)
		if err != nil {
			return geminiRequestPayload{}, fmt.Errorf("gemini: tool %q parameters: %w", spec.Function.Name, err)
		}
		payload.Tools = append(payload.Tools, geminiTool{FunctionDeclarations: []geminiFunctionDeclaration{{Name: spec.Function.Name, Description: spec.Function.Description, Parameters: geminiSchemaFromJSON(schema)}}})
	}
	if req.ToolChoice != "" {
		mode := map[ToolChoice]string{ToolChoiceAuto: "AUTO", ToolChoiceNone: "NONE", ToolChoiceRequired: "ANY"}[req.ToolChoice]
		if mode == "" {
			return geminiRequestPayload{}, fmt.Errorf("gemini: unsupported tool choice %q", req.ToolChoice)
		}
		payload.ToolConfig = &geminiToolConfig{FunctionCallingConfig: geminiFunctionCallingConfig{Mode: mode}}
	}
	return payload, nil
}

func geminiMessages(source []Message) ([]geminiContent, []string, error) {
	callNames := make(map[string]string)
	for _, message := range source {
		for _, call := range message.ToolCalls {
			callNames[call.ID] = call.Function.Name
		}
	}
	var contents []geminiContent
	var systemPrompts []string
	for index := 0; index < len(source); index++ {
		message := source[index]
		switch message.Role {
		case RoleSystem:
			if message.Content != "" {
				systemPrompts = append(systemPrompts, message.Content)
			}
		case RoleUser:
			contents = append(contents, geminiContent{Role: roleUser, Parts: []geminiPart{{Text: message.Content}}})
		case RoleAssistant:
			converted, err := geminiAssistantMessage(message, index)
			if err != nil {
				return nil, nil, err
			}
			contents = append(contents, converted)
		case RoleTool:
			converted, nextIndex, err := geminiToolResultMessage(source, index, callNames)
			if err != nil {
				return nil, nil, err
			}
			contents = append(contents, converted)
			index = nextIndex - 1
		default:
			return nil, nil, fmt.Errorf("gemini: unsupported message role %q", message.Role)
		}
	}
	return contents, systemPrompts, nil
}

func geminiAssistantMessage(message Message, messageIndex int) (geminiContent, error) {
	parts := make([]geminiPart, 0, len(message.ToolCalls)+1)
	if message.Content != "" {
		parts = append(parts, geminiPart{Text: message.Content})
	}
	for _, call := range message.ToolCalls {
		args, err := DecodeJSONValue(call.Function.Arguments)
		if err != nil {
			return geminiContent{}, fmt.Errorf("gemini: tool call %q arguments: %w", call.Function.Name, err)
		}
		object, ok := args.(map[string]any)
		if !ok {
			return geminiContent{}, fmt.Errorf("gemini: tool call %q arguments must be a JSON object", call.Function.Name)
		}
		parts = append(parts, geminiPart{FunctionCall: &geminiFunctionCall{ID: call.ID, Name: call.Function.Name, Args: object}})
	}
	if len(parts) == 0 {
		return geminiContent{}, fmt.Errorf("gemini: assistant message %d has no content or function calls", messageIndex)
	}
	return geminiContent{Role: roleModel, Parts: parts}, nil
}

func geminiToolResultMessage(messages []Message, start int, callNames map[string]string) (geminiContent, int, error) {
	parts := make([]geminiPart, 0, 2)
	index := start
	for index < len(messages) && messages[index].Role == RoleTool {
		result := messages[index]
		name := result.Name
		if name == "" {
			name = callNames[result.ToolCallID]
		}
		if name == "" {
			return geminiContent{}, 0, fmt.Errorf("gemini: tool result %q is missing a function name", result.ToolCallID)
		}
		response, err := DecodeJSONValue(result.Content)
		if err != nil {
			response = map[string]any{"result": result.Content}
		}
		object, ok := response.(map[string]any)
		if !ok {
			object = map[string]any{"result": response}
		}
		parts = append(parts, geminiPart{FunctionResponse: &geminiFunctionResponse{ID: result.ToolCallID, Name: name, Response: object}})
		index++
	}
	return geminiContent{Role: roleUser, Parts: parts}, index, nil
}

func geminiSchemaFromJSON(schema JSONSchema) geminiSchema {
	typeNames := map[string]string{schemaTypeObject: "OBJECT", schemaTypeArray: "ARRAY", schemaTypeString: "STRING", schemaTypeNumber: "NUMBER", schemaTypeInteger: "INTEGER", schemaTypeBoolean: "BOOLEAN", "null": "NULL"}
	converted := geminiSchema{Type: typeNames[schema.Type], Description: schema.Description, Format: schema.Format, Nullable: schema.Nullable, Required: append([]string(nil), schema.Required...), Enum: append([]string(nil), schema.Enum...)}
	if len(schema.Properties) > 0 {
		converted.Properties = make(map[string]geminiSchema, len(schema.Properties))
		for name := range schema.Properties {
			property := schema.Properties[name]
			converted.Properties[name] = geminiSchemaFromJSON(property)
		}
	}
	if schema.Items != nil {
		item := geminiSchemaFromJSON(*schema.Items)
		converted.Items = &item
	}
	return converted
}
