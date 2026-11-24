package llm

import (
	"context"
	"encoding/json"
	"iter"
	"strings"
	"testing"
)

func TestJSONSchemaValidatesRecursiveObjectsArraysAndEnums(t *testing.T) {
	closed := false
	item := JSONSchema{
		Type: "object",
		Properties: map[string]JSONSchema{
			"kind":  {Type: "string", Enum: []string{"red", "blue"}},
			"score": {Type: "integer"},
		},
		Required:             []string{"kind"},
		AdditionalProperties: &closed,
	}
	schema := JSONSchema{
		Type: "object",
		Properties: map[string]JSONSchema{
			"name":  {Type: "string"},
			"items": {Type: "array", Items: &item},
		},
		Required:             []string{"name", "items"},
		AdditionalProperties: &closed,
	}

	if err := ValidateJSONContent(`{"name":"test","items":[{"kind":"red","score":2}]}`, schema); err != nil {
		t.Fatalf("valid nested content rejected: %v", err)
	}
	for _, content := range []string{
		`{"name":"test","items":[{"kind":"green"}]}`,
		`{"name":"test","items":[{"kind":"red","extra":true}]}`,
		`{"name":"test","items":[] ,"extra":true}`,
		`{"items":[]}`,
	} {
		if err := ValidateJSONContent(content, schema); err == nil {
			t.Errorf("invalid content accepted: %s", content)
		}
	}
}

func TestDecodeJSONValueRejectsMalformedAndTrailingJSON(t *testing.T) {
	for _, content := range []string{`{"x":`, `{} {}`, `{} garbage`} {
		if _, err := DecodeJSONValue(content); err == nil {
			t.Errorf("invalid JSON accepted: %q", content)
		}
	}
}

func TestParseJSONSchemaRejectsUnsupportedKeywordsRecursively(t *testing.T) {
	for _, raw := range []string{
		`{"type":"string","pattern":"^safe$"}`,
		`{"type":"object","properties":{"count":{"type":"integer","minimum":1}}}`,
		`{"enum":["small","large"]}`,
		`null`,
	} {
		if _, err := ParseJSONSchema(json.RawMessage(raw)); err == nil {
			t.Errorf("schema with unsupported content accepted: %s", raw)
		}
	}
}

func TestJSONSchemaMarshalKeepsEmptyObjectProperties(t *testing.T) {
	data, err := json.Marshal(JSONSchema{Type: "object", Properties: map[string]JSONSchema{}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"properties":{}`) {
		t.Fatalf("empty object schema lost properties map: %s", data)
	}
	if strings.Contains(string(data), `"properties":null`) {
		t.Fatalf("empty object schema emitted null properties: %s", data)
	}
}

func TestJSONSchemaMarshalPreservesUnconstrainedAndNullTypes(t *testing.T) {
	for _, tc := range []struct {
		name   string
		schema JSONSchema
		want   string
	}{
		{name: "unconstrained", schema: JSONSchema{}, want: `{}`},
		{name: "unconstrained property", schema: JSONSchema{Type: "object", Properties: map[string]JSONSchema{"value": {}}}, want: `{"type":"object","properties":{"value":{}}}`},
		{name: "redundant nullable null", schema: JSONSchema{Type: "null", Nullable: true}, want: `{"type":"null"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			data, err := json.Marshal(tc.schema)
			if err != nil {
				t.Fatal(err)
			}
			if string(data) != tc.want {
				t.Fatalf("JSON Schema = %s, want %s", data, tc.want)
			}
			if _, err := ParseJSONSchema(data); err != nil {
				t.Fatalf("serialized schema cannot be parsed: %v", err)
			}
		})
	}
}

func TestOpenAIStrictSchemaClosesAndNormalizesOptionalFields(t *testing.T) {
	closed := false
	schema := JSONSchema{
		Type: "object",
		Properties: map[string]JSONSchema{
			"name":     {Type: "string"},
			"nickname": {Type: "string"},
		},
		Required:             []string{"name"},
		AdditionalProperties: &closed,
	}
	if err := ValidateOpenAIStrictSchema(schema); err != nil {
		t.Fatalf("closed schema should be strict-compatible: %v", err)
	}
	strict := OpenAIStrictSchema(schema)
	if len(strict.Required) != 2 || !strict.Properties["nickname"].Nullable {
		t.Fatalf("optional field was not represented for strict mode: %+v", strict)
	}
	content, err := NormalizeJSONContent(`{"name":"Ada","nickname":null}`, schema)
	if err != nil {
		t.Fatalf("normalize optional null: %v", err)
	}
	if content != `{"name":"Ada"}` {
		t.Fatalf("normalized content = %s, want optional null omitted", content)
	}

	open := JSONSchema{Type: "object", Properties: map[string]JSONSchema{}}
	if err := ValidateOpenAIStrictSchema(open); err == nil {
		t.Fatal("open object schema unexpectedly passed strict preflight")
	}
	unconstrained := JSONSchema{
		Type: "object", Properties: map[string]JSONSchema{"value": {}},
		AdditionalProperties: &closed,
	}
	if err := ValidateOpenAIStrictSchema(unconstrained); err == nil {
		t.Fatal("unconstrained property unexpectedly passed strict preflight")
	}
}

func TestValidateToolSpecsCallsAndResults(t *testing.T) {
	closed := false
	arguments := JSONSchema{
		Type: "object", Properties: map[string]JSONSchema{"count": {Type: "integer"}},
		Required: []string{"count"}, AdditionalProperties: &closed,
	}
	tool := ToolSpec{Type: "function", Function: FunctionSpec{
		Name: "count_words", Parameters: arguments, ResultSchema: map[string]any{"type": "integer"},
	}}
	if err := ValidateToolSpecs([]ToolSpec{tool}); err != nil {
		t.Fatalf("valid tool spec rejected: %v", err)
	}
	validCall := []ToolCall{{ID: "call-1", Type: "function", Function: FunctionCall{Name: "count_words", Arguments: `{"count":2}`}}}
	if err := ValidateAndNormalizeToolCalls(validCall, []ToolSpec{tool}); err != nil {
		t.Fatalf("valid tool call rejected: %v", err)
	}
	for _, call := range [][]ToolCall{
		{{ID: "call-1", Type: "function", Function: FunctionCall{Name: "count_words", Arguments: `{}`}}},
		{{ID: "call-1", Type: "function", Function: FunctionCall{Name: "other", Arguments: `{"count":2}`}}},
		{{ID: "call-1", Type: "function", Function: FunctionCall{Name: "count_words", Arguments: `{"count":2} {}`}}},
		{{ID: "call-1", Type: "function", Function: FunctionCall{Name: "count_words", Arguments: `{"count":2}`}}, {ID: "call-1", Type: "function", Function: FunctionCall{Name: "count_words", Arguments: `{"count":2}`}}},
	} {
		if err := ValidateAndNormalizeToolCalls(call, []ToolSpec{tool}); err == nil {
			t.Errorf("invalid tool call accepted: %+v", call)
		}
	}
	if err := ValidateToolResults([]Message{{Role: RoleTool, Name: "count_words", Content: `"two"`}}, []ToolSpec{tool}); err == nil {
		t.Fatal("invalid host tool result passed its declared result schema")
	}
}

func TestValidateMessageToolCallsChecksHistoricalArguments(t *testing.T) {
	closed := false
	tool := ToolSpec{Type: "function", Function: FunctionSpec{
		Name: "count_words",
		Parameters: JSONSchema{
			Type: "object", Properties: map[string]JSONSchema{"count": {Type: "integer"}},
			Required: []string{"count"}, AdditionalProperties: &closed,
		},
	}}
	messages := []Message{{Role: RoleAssistant, ToolCalls: []ToolCall{{
		ID: "history-call", Type: "function", Function: FunctionCall{Name: "count_words", Arguments: `{}`},
	}}}}
	if err := ValidateMessageToolCalls(messages, []ToolSpec{tool}); err == nil {
		t.Fatal("historical tool call bypassed its argument schema")
	}
}

func TestValidateToolResultsRequireUniquePrecedingCalls(t *testing.T) {
	tool := ToolSpec{Type: "function", Function: FunctionSpec{Name: "count_words", Parameters: JSONSchema{Type: "object"}, ResultSchema: map[string]any{"type": "string"}}}
	call := Message{Role: RoleAssistant, ToolCalls: []ToolCall{{
		ID: "call-1", Type: toolTypeFunction, Function: FunctionCall{Name: "count_words", Arguments: `{}`},
	}}}
	validResult := Message{Role: RoleTool, ToolCallID: "call-1", Name: "count_words", Content: `"three"`}

	for _, tc := range []struct {
		name     string
		messages []Message
	}{
		{name: "missing id", messages: []Message{call, {Role: RoleTool, Name: "count_words", Content: `"three"`}}},
		{name: "orphan id", messages: []Message{{Role: RoleTool, ToolCallID: "orphan", Name: "count_words", Content: `"three"`}}},
		{name: "duplicate result", messages: []Message{call, validResult, validResult}},
		{name: "mismatched name", messages: []Message{call, {Role: RoleTool, ToolCallID: "call-1", Name: "other", Content: `"three"`}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := ValidateToolResults(tc.messages, []ToolSpec{tool}); err == nil {
				t.Fatal("invalid tool-result history accepted")
			}
		})
	}
	if err := ValidateToolResults([]Message{call, validResult}, []ToolSpec{tool}); err != nil {
		t.Fatalf("valid linked tool result rejected: %v", err)
	}
}

func TestCapabilitiesRemainUnknownForCustomProvidersAndOllamaCloudAliasStaysLocal(t *testing.T) {
	capabilities := CapabilitiesFor(namedProvider("openai"), "model")
	if capabilities.OutputSchema != CapabilityUnknown || capabilities.ToolCalls != CapabilityUnknown {
		t.Fatalf("undeclared custom provider capabilities were guessed: %+v", capabilities)
	}

	local := NewOllama(Config{Name: "ollama_cloud"})
	if local.cfg.BaseURL != "http://localhost:11434" {
		t.Fatalf("ollama_cloud default endpoint = %q, want existing local default", local.cfg.BaseURL)
	}
	if got := local.Capabilities("model").OutputSchema; got != CapabilityConditional {
		t.Fatalf("local ollama_cloud output schema capability = %q, want conditional", got)
	}
	cloud := NewOllama(Config{Name: "ollama_cloud", BaseURL: "https://ollama.com"})
	if got := cloud.Capabilities("model").OutputSchema; got != CapabilityUnsupported {
		t.Fatalf("explicit Ollama Cloud output schema capability = %q, want unsupported", got)
	}
}

func TestProviderFinishReasonsRecognizeTruncationAliases(t *testing.T) {
	for _, reason := range []string{"length", "max_tokens", "MAX_TOKENS"} {
		if !(Response{FinishReason: reason}).Truncated() {
			t.Errorf("finish reason %q not recognized as truncation", reason)
		}
	}
	if (Response{FinishReason: "stop"}).Truncated() {
		t.Fatal("ordinary stop reason recognized as truncation")
	}
}

func TestGeminiSchemaMappingOmitsUnsupportedAdditionalProperties(t *testing.T) {
	closed := false
	schema := JSONSchema{Type: "object", Properties: map[string]JSONSchema{}, AdditionalProperties: &closed}
	data, err := json.Marshal(geminiSchemaFromJSON(schema))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "additionalProperties") {
		t.Fatalf("unsupported Gemini field was serialized: %s", data)
	}
	if supported, _ := NewGemini(Config{DefaultModel: "gemini-3-test"}).SupportsOutputSchema("", schema, false); supported {
		t.Fatal("Gemini advertised a schema using unsupported additionalProperties")
	}
}

type namedProvider string

func (p namedProvider) Name() string { return string(p) }
func (namedProvider) Complete(context.Context, Request) (Response, error) {
	return Response{}, nil
}

func (namedProvider) Stream(context.Context, Request) (iter.Seq2[string, error], error) {
	return func(func(string, error) bool) {}, nil
}
