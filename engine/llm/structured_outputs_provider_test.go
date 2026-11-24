package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func schemaTestObject() JSONSchema {
	closed := false
	return JSONSchema{
		Type:                 "object",
		Properties:           map[string]JSONSchema{"answer": {Type: "string"}},
		Required:             []string{"answer"},
		AdditionalProperties: &closed,
	}
}

func toolTestSchema() JSONSchema {
	closed := false
	return JSONSchema{
		Type:                 "object",
		Properties:           map[string]JSONSchema{"query": {Type: "string"}},
		Required:             []string{"query"},
		AdditionalProperties: &closed,
	}
}

func toolTestHistory() []Message {
	return []Message{
		{Role: RoleUser, Content: "Search for a result."},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{
			ID: "call-1", Type: toolTypeFunction,
			Function: FunctionCall{Name: "search", Arguments: `{"query":"NFlow"}`},
		}}},
		{Role: RoleTool, ToolCallID: "call-1", Name: "search", Content: `{"count":1}`},
	}
}

func TestOpenAICompleteEncodesStrictOutputSchema(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload openAIRequestPayload
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode OpenAI request: %v", err)
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		if payload.ResponseFormat == nil || payload.ResponseFormat.Type != "json_schema" || payload.ResponseFormat.JSONSchema == nil || !payload.ResponseFormat.JSONSchema.Strict {
			t.Errorf("response format = %+v, want strict json_schema", payload.ResponseFormat)
		} else {
			schema := payload.ResponseFormat.JSONSchema.Schema
			if schema.Type != "object" || !strings.Contains(strings.Join(schema.Required, ","), "answer") || schema.AdditionalProperties == nil || *schema.AdditionalProperties {
				t.Errorf("encoded output schema = %+v", schema)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"schema-test","choices":[{"message":{"role":"assistant","content":"{\"answer\":\"ready\"}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}`))
	}))
	defer server.Close()

	provider := NewOpenAI(Config{BaseURL: server.URL})
	response, err := provider.Complete(context.Background(), Request{
		Model: "schema-test", Messages: []Message{{Role: RoleUser, Content: "Answer."}}, OutputSchema: new(schemaTestObject()),
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if response.Content != `{"answer":"ready"}` {
		t.Fatalf("content = %q", response.Content)
	}
}

func TestOpenAIToolsEncodeStrictArgumentsWithoutHostResultSchema(t *testing.T) {
	tools, err := openAITools([]ToolSpec{{Type: toolTypeFunction, Function: FunctionSpec{
		Name: "search", Parameters: toolTestSchema(), Strict: true, ResultSchema: map[string]any{"type": "object"},
	}}})
	if err != nil {
		t.Fatalf("openAITools: %v", err)
	}
	if len(tools) != 1 || !tools[0].Function.Strict {
		t.Fatalf("encoded tools = %+v", tools)
	}
	parameters, ok := tools[0].Function.Parameters.(JSONSchema)
	if !ok || parameters.AdditionalProperties == nil || *parameters.AdditionalProperties || !strings.Contains(strings.Join(parameters.Required, ","), "query") {
		t.Fatalf("encoded function parameters = %#v", tools[0].Function.Parameters)
	}
	data, err := json.Marshal(tools)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "result_schema") {
		t.Fatalf("host-only result schema leaked to provider: %s", data)
	}
}

func TestAnthropicPayloadMapsOutputToolsAndHistory(t *testing.T) {
	provider := NewAnthropic(Config{})
	payload, err := provider.buildPayload(Request{
		Model: "claude-test", Messages: toolTestHistory(), OutputSchema: new(schemaTestObject()),
		Tools: []ToolSpec{{Type: toolTypeFunction, Function: FunctionSpec{
			Name: "search", Description: "Search.", Parameters: toolTestSchema(), Strict: true, ResultSchema: map[string]any{"type": "object"},
		}}}, ToolChoice: ToolChoiceRequired,
	}, false)
	if err != nil {
		t.Fatalf("buildPayload: %v", err)
	}
	if payload.OutputConfig == nil || payload.OutputConfig.Format.Type != "json_schema" || payload.OutputConfig.Format.Schema.Properties["answer"].Type != "string" {
		t.Fatalf("output config = %+v", payload.OutputConfig)
	}
	if len(payload.Tools) != 1 || payload.Tools[0].Name != "search" || !payload.Tools[0].Strict || payload.ToolChoice == nil || payload.ToolChoice.Type != "any" {
		t.Fatalf("tools/choice = %+v/%+v", payload.Tools, payload.ToolChoice)
	}
	assistantBlocks, ok := payload.Messages[1].Content.([]anthropicRequestBlock)
	if !ok || len(assistantBlocks) != 1 || assistantBlocks[0].Type != "tool_use" || assistantBlocks[0].ID != "call-1" || assistantBlocks[0].Name != "search" {
		t.Fatalf("assistant history = %#v", payload.Messages[1].Content)
	}
	resultBlocks, ok := payload.Messages[2].Content.([]anthropicRequestBlock)
	if !ok || len(resultBlocks) != 1 || resultBlocks[0].Type != "tool_result" || resultBlocks[0].ToolUseID != "call-1" || resultBlocks[0].Content != `{"count":1}` {
		t.Fatalf("tool-result history = %#v", payload.Messages[2].Content)
	}
	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "result_schema") {
		t.Fatalf("host-only result schema leaked to provider: %s", data)
	}
}

func TestGeminiPayloadMapsOutputToolsAndHistory(t *testing.T) {
	provider := NewGemini(Config{})
	payload, err := provider.buildPayload(Request{
		Model: "gemini-3-test", Messages: toolTestHistory(), OutputSchema: new(schemaTestObject()),
		Tools:      []ToolSpec{{Type: toolTypeFunction, Function: FunctionSpec{Name: "search", Description: "Search.", Parameters: toolTestSchema()}}},
		ToolChoice: ToolChoiceRequired,
	})
	if err != nil {
		t.Fatalf("buildPayload: %v", err)
	}
	if payload.GenerationConfig == nil || payload.GenerationConfig.ResponseMimeType != "application/json" || payload.GenerationConfig.ResponseSchema == nil || payload.GenerationConfig.ResponseSchema.Type != "OBJECT" {
		t.Fatalf("generation config = %+v", payload.GenerationConfig)
	}
	if len(payload.Tools) != 1 || len(payload.Tools[0].FunctionDeclarations) != 1 || payload.Tools[0].FunctionDeclarations[0].Name != "search" || payload.ToolConfig == nil || payload.ToolConfig.FunctionCallingConfig.Mode != "ANY" {
		t.Fatalf("tools/config = %+v/%+v", payload.Tools, payload.ToolConfig)
	}
	call := payload.Contents[1].Parts[0].FunctionCall
	if call == nil || call.ID != "call-1" || call.Name != "search" || call.Args["query"] != "NFlow" {
		t.Fatalf("function-call history = %+v", call)
	}
	result := payload.Contents[2].Parts[0].FunctionResponse
	if result == nil || result.ID != "call-1" || result.Name != "search" || result.Response["count"] != json.Number("1") {
		t.Fatalf("function-response history = %+v", result)
	}
}

func TestOllamaPayloadMapsOutputToolsAndHistory(t *testing.T) {
	provider := NewOllama(Config{})
	payload, err := provider.buildPayload(Request{
		Model: "ollama-test", Messages: toolTestHistory(), OutputSchema: new(schemaTestObject()),
		Tools:      []ToolSpec{{Type: toolTypeFunction, Function: FunctionSpec{Name: "search", Description: "Search.", Parameters: toolTestSchema()}}},
		ToolChoice: ToolChoiceAuto,
	}, false)
	if err != nil {
		t.Fatalf("buildPayload: %v", err)
	}
	format, ok := payload.Format.(JSONSchema)
	if !ok || format.Type != "object" || format.Properties["answer"].Type != "string" {
		t.Fatalf("format = %#v", payload.Format)
	}
	if len(payload.Tools) != 1 || payload.Tools[0].Function.Name != "search" || payload.Tools[0].Function.Parameters.Properties["query"].Type != "string" {
		t.Fatalf("tools = %+v", payload.Tools)
	}
	if len(payload.Messages) != 3 || len(payload.Messages[1].ToolCalls) != 1 || payload.Messages[1].ToolCalls[0].Function.Name != "search" || payload.Messages[2].Role != RoleTool || payload.Messages[2].Name != "search" || payload.Messages[2].Content != `{"count":1}` {
		t.Fatalf("tool history = %+v", payload.Messages)
	}
}
