package llm

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"sort"
	"strings"
)

// MarshalJSON preserves the established JSONSchema API while supporting the
// nullable type unions emitted by common constrained-output APIs.
func (s JSONSchema) MarshalJSON() ([]byte, error) {
	type schemaJSON struct {
		Type                 any                    `json:"type,omitempty"`
		Description          string                 `json:"description,omitempty"`
		Format               string                 `json:"format,omitempty"`
		Properties           *map[string]JSONSchema `json:"properties,omitempty"`
		Required             []string               `json:"required,omitempty"`
		Items                *JSONSchema            `json:"items,omitempty"`
		Enum                 []string               `json:"enum,omitempty"`
		AdditionalProperties *bool                  `json:"additionalProperties,omitempty"`
	}

	var typeValue any
	if s.Type != "" {
		typeValue = s.Type
		if s.Nullable && s.Type != "null" {
			typeValue = []string{s.Type, "null"}
		}
	}
	var properties *map[string]JSONSchema
	if s.Type == schemaTypeObject || len(s.Properties) > 0 {
		objectProperties := s.Properties
		if objectProperties == nil {
			objectProperties = map[string]JSONSchema{}
		}
		properties = &objectProperties
	}
	return json.Marshal(schemaJSON{
		Type: typeValue, Description: s.Description, Format: s.Format,
		Properties: properties, Required: s.Required, Items: s.Items,
		Enum: s.Enum, AdditionalProperties: s.AdditionalProperties,
	})
}

type jsonSchemaWire struct {
	Type                 json.RawMessage       `json:"type"`
	Description          string                `json:"description"`
	Format               string                `json:"format"`
	Properties           map[string]JSONSchema `json:"properties"`
	Required             []string              `json:"required"`
	Items                *JSONSchema           `json:"items"`
	Enum                 []string              `json:"enum"`
	AdditionalProperties *bool                 `json:"additionalProperties"`
	Nullable             bool                  `json:"nullable"`
}

// UnmarshalJSON accepts one type or the common nullable pair ["type", "null"].
func (s *JSONSchema) UnmarshalJSON(data []byte) error {
	if s == nil {
		return errors.New("llm: cannot unmarshal JSON Schema into nil receiver")
	}
	var keys map[string]json.RawMessage
	if err := json.Unmarshal(data, &keys); err != nil {
		return err
	}
	if keys == nil {
		return errors.New("llm: JSON Schema must be an object")
	}
	if err := validateJSONSchemaKeys(keys); err != nil {
		return err
	}
	var wire jsonSchemaWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	if err := normalizeAdditionalPropertiesAlias(keys, &wire); err != nil {
		return err
	}
	schemaType, nullable, err := parseJSONSchemaType(wire.Type, wire.Nullable)
	if err != nil {
		return err
	}
	*s = JSONSchema{
		Type: schemaType, Nullable: nullable, Description: wire.Description, Format: wire.Format,
		Properties: wire.Properties, Required: wire.Required, Items: wire.Items,
		Enum: wire.Enum, AdditionalProperties: wire.AdditionalProperties,
	}
	return nil
}

func validateJSONSchemaKeys(keys map[string]json.RawMessage) error {
	allowed := map[string]struct{}{
		"type": {}, "description": {}, "format": {}, "properties": {}, "required": {},
		"items": {}, "enum": {}, "additionalProperties": {}, "additional_properties": {}, "nullable": {},
	}
	for key := range keys {
		if _, ok := allowed[key]; !ok {
			return fmt.Errorf("llm: unsupported JSON Schema keyword %q", key)
		}
	}
	return nil
}

func normalizeAdditionalPropertiesAlias(keys map[string]json.RawMessage, wire *jsonSchemaWire) error {
	alias, exists := keys["additional_properties"]
	if !exists {
		return nil
	}
	var aliasValue *bool
	if err := json.Unmarshal(alias, &aliasValue); err != nil {
		return fmt.Errorf("llm: decode additional_properties: %w", err)
	}
	if _, duplicate := keys["additionalProperties"]; duplicate {
		canonicalValue := wire.AdditionalProperties
		if (canonicalValue == nil) != (aliasValue == nil) || (canonicalValue != nil && *canonicalValue != *aliasValue) {
			return errors.New("llm: additionalProperties and additional_properties disagree")
		}
		return nil
	}
	wire.AdditionalProperties = aliasValue
	return nil
}

func parseJSONSchemaType(raw json.RawMessage, nullable bool) (schemaType string, resultNullable bool, parseErr error) {
	if len(raw) == 0 {
		return "", nullable, nil
	}
	if err := json.Unmarshal(raw, &schemaType); err == nil {
		return schemaType, nullable, nil
	}
	var union []string
	if err := json.Unmarshal(raw, &union); err != nil || len(union) != 2 {
		return "", false, errors.New("llm: JSON Schema type must be a string or nullable type pair")
	}
	sawNull := false
	for _, item := range union {
		if item == "null" {
			if sawNull {
				return "", false, errors.New("llm: duplicate null type in JSON Schema nullable pair")
			}
			sawNull = true
			continue
		}
		if schemaType != "" {
			return "", false, errors.New("llm: JSON Schema type unions other than nullable pairs are unsupported")
		}
		schemaType = item
	}
	if schemaType == "" || !sawNull {
		return "", false, errors.New("llm: JSON Schema type must be a string or nullable type pair")
	}
	return schemaType, nullable || sawNull, nil
}

// ParseJSONSchema converts a JSON-shaped value, commonly supplied by a Flow
// request or a compatibility caller, into the shared validated contract.
func ParseJSONSchema(value any) (JSONSchema, error) {
	switch schema := value.(type) {
	case JSONSchema:
		if err := schema.ValidateDefinition(); err != nil {
			return JSONSchema{}, err
		}
		return schema, nil
	case *JSONSchema:
		if schema == nil {
			return JSONSchema{}, errors.New("llm: JSON Schema is nil")
		}
		return ParseJSONSchema(*schema)
	case json.RawMessage:
		return parseJSONSchemaBytes(schema)
	case []byte:
		return parseJSONSchemaBytes(schema)
	case nil:
		return JSONSchema{}, errors.New("llm: JSON Schema is required")
	default:
		data, err := json.Marshal(value)
		if err != nil {
			return JSONSchema{}, fmt.Errorf("llm: encode JSON Schema: %w", err)
		}
		return parseJSONSchemaBytes(data)
	}
}

func parseJSONSchemaBytes(data []byte) (JSONSchema, error) {
	var schema JSONSchema
	if err := json.Unmarshal(data, &schema); err != nil {
		return JSONSchema{}, fmt.Errorf("llm: decode JSON Schema: %w", err)
	}
	if err := schema.ValidateDefinition(); err != nil {
		return JSONSchema{}, err
	}
	return schema, nil
}

// ValidateDefinition verifies the recursively supported JSON Schema subset.
func (s JSONSchema) ValidateDefinition() error {
	return s.validateDefinition("$")
}

func hasJSONSchemaAdditionalProperties(schema JSONSchema) bool {
	if schema.AdditionalProperties != nil {
		return true
	}
	for name := range schema.Properties {
		property := schema.Properties[name]
		if hasJSONSchemaAdditionalProperties(property) {
			return true
		}
	}
	return schema.Items != nil && hasJSONSchemaAdditionalProperties(*schema.Items)
}

func (s JSONSchema) validateDefinition(path string) error {
	switch s.Type {
	case "", schemaTypeObject, schemaTypeArray, schemaTypeString, schemaTypeNumber, schemaTypeInteger, "boolean", "null":
	default:
		return fmt.Errorf("llm: unsupported JSON Schema type %q at %s", s.Type, path)
	}
	if len(s.Enum) > 0 && s.Type != schemaTypeString {
		return fmt.Errorf("llm: string enum requires type string at %s", path)
	}
	if s.Type == schemaTypeArray && s.Items == nil {
		return fmt.Errorf("llm: array schema requires items at %s", path)
	}
	if s.Type != schemaTypeObject && (len(s.Properties) > 0 || len(s.Required) > 0 || s.AdditionalProperties != nil) {
		return fmt.Errorf("llm: object keywords require type object at %s", path)
	}
	for _, name := range s.Required {
		if _, exists := s.Properties[name]; !exists {
			return fmt.Errorf("llm: required field %q has no property schema at %s", name, path)
		}
	}
	for name := range s.Properties {
		child := s.Properties[name]
		if err := child.validateDefinition(path + "." + name); err != nil {
			return err
		}
	}
	if s.Items != nil {
		if err := s.Items.validateDefinition(path + "[]"); err != nil {
			return err
		}
	}
	return nil
}

// DecodeJSONValue accepts exactly one JSON value and rejects trailing values.
func DecodeJSONValue(content string) (any, error) {
	decoder := json.NewDecoder(strings.NewReader(content))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, fmt.Errorf("llm: invalid JSON: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return nil, errors.New("llm: JSON contains trailing data")
		}
		return nil, fmt.Errorf("llm: invalid trailing JSON data: %w", err)
	}
	return value, nil
}

// ValidateJSONContent validates one complete JSON document against a schema.
func ValidateJSONContent(content string, schema JSONSchema) error {
	if err := schema.ValidateDefinition(); err != nil {
		return err
	}
	value, err := DecodeJSONValue(content)
	if err != nil {
		return err
	}
	return schema.ValidateValue(value)
}

// ValidateValue validates a decoded JSON value recursively against the schema.
func (s JSONSchema) ValidateValue(value any) error {
	if err := s.ValidateDefinition(); err != nil {
		return err
	}
	return s.validateValue(value, "$")
}

func (s JSONSchema) validateValue(value any, path string) error {
	if value == nil && (s.Nullable || s.Type == "null" || s.Type == "") {
		return nil
	}
	if value == nil {
		return fmt.Errorf("llm: expected %s at %s, got null", s.Type, path)
	}
	switch s.Type {
	case schemaTypeObject:
		return s.validateObject(value, path)
	case schemaTypeArray:
		return s.validateArray(value, path)
	case schemaTypeString:
		return s.validateString(value, path)
	case schemaTypeInteger:
		if !isJSONInteger(value) {
			return typeError(schemaTypeInteger, value, path)
		}
	case schemaTypeNumber:
		if !isJSONNumber(value) {
			return typeError("number", value, path)
		}
	case "boolean":
		if _, ok := value.(bool); !ok {
			return typeError("boolean", value, path)
		}
	case "null":
		return fmt.Errorf("llm: expected null at %s, got %s", path, jsonKind(value))
	}
	return nil
}

func typeError(want string, value any, path string) error {
	return fmt.Errorf("llm: expected %s at %s, got %s", want, path, jsonKind(value))
}

func jsonKind(value any) string {
	switch value.(type) {
	case nil:
		return "null"
	case string:
		return "string"
	case bool:
		return "boolean"
	case json.Number, float32, float64, int, int32, int64, uint, uint32, uint64:
		return "number"
	case []any:
		return schemaTypeArray
	case map[string]any:
		return schemaTypeObject
	default:
		return fmt.Sprintf("%T", value)
	}
}

func isJSONNumber(value any) bool {
	switch number := value.(type) {
	case json.Number:
		_, err := number.Float64()
		return err == nil
	case float32:
		return !math.IsNaN(float64(number)) && !math.IsInf(float64(number), 0)
	case float64:
		return !math.IsNaN(number) && !math.IsInf(number, 0)
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		return true
	default:
		return false
	}
}

func isJSONInteger(value any) bool {
	switch number := value.(type) {
	case json.Number:
		if _, err := number.Int64(); err == nil {
			return true
		}
		floating, err := number.Float64()
		return err == nil && !math.IsNaN(floating) && !math.IsInf(floating, 0) && math.Trunc(floating) == floating
	case float32:
		return !math.IsNaN(float64(number)) && !math.IsInf(float64(number), 0) && math.Trunc(float64(number)) == float64(number)
	case float64:
		return !math.IsNaN(number) && !math.IsInf(number, 0) && math.Trunc(number) == number
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		return true
	default:
		return false
	}
}

// CanonicalJSON round-trips a JSON-shaped value to detach mutable caller data.
func CanonicalJSON(value any) (any, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("llm: encode JSON value: %w", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var out any
	if err := decoder.Decode(&out); err != nil {
		return nil, err
	}
	return out, nil
}

// OpenAIStrictSchema converts the shared optional-field contract to the strict
// subset required by OpenAI-compatible structured outputs. Optional properties
// become required nullable properties; callers normalize null back to omission.
func OpenAIStrictSchema(schema JSONSchema) JSONSchema {
	out := schema
	if schema.Type == schemaTypeObject {
		out.Properties = make(map[string]JSONSchema, len(schema.Properties))
		out.Required = make([]string, 0, len(schema.Properties))
		required := make(map[string]bool, len(schema.Required))
		for _, name := range schema.Required {
			required[name] = true
		}
		for name := range schema.Properties {
			property := schema.Properties[name]
			converted := OpenAIStrictSchema(property)
			if !required[name] && converted.Type != "" {
				converted.Nullable = true
			}
			out.Properties[name] = converted
			out.Required = append(out.Required, name)
		}
		sort.Strings(out.Required)
		closed := false
		out.AdditionalProperties = &closed
	}
	if schema.Items != nil {
		item := OpenAIStrictSchema(*schema.Items)
		out.Items = &item
	}
	return out
}

// ValidateOpenAIStrictSchema rejects contracts that OpenAI strict mode cannot
// preserve. Objects must be explicitly closed; open maps cannot be silently
// converted into closed objects.
func ValidateOpenAIStrictSchema(schema JSONSchema) error {
	if err := schema.ValidateDefinition(); err != nil {
		return err
	}
	if schema.Type != schemaTypeObject {
		return errors.New("llm: OpenAI strict output schema root must be an object")
	}
	return validateOpenAIStrictSchema(schema, "$")
}

func validateOpenAIStrictSchema(schema JSONSchema, path string) error {
	if schema.Type == "" {
		return fmt.Errorf("llm: OpenAI strict schema requires a concrete type at %s", path)
	}
	if schema.Type == schemaTypeObject {
		if schema.AdditionalProperties == nil || *schema.AdditionalProperties {
			return fmt.Errorf("llm: OpenAI strict schema requires additionalProperties:false at %s", path)
		}
		for name := range schema.Properties {
			property := schema.Properties[name]
			if err := validateOpenAIStrictSchema(property, path+"."+name); err != nil {
				return err
			}
		}
	}
	if schema.Items != nil {
		return validateOpenAIStrictSchema(*schema.Items, path+"[]")
	}
	return nil
}

// NormalizeOptionalNulls removes provider-required null placeholders only when
// the shared contract declares the property optional and non-nullable.
func NormalizeOptionalNulls(value any, schema JSONSchema) any {
	switch schema.Type {
	case schemaTypeObject:
		object, ok := value.(map[string]any)
		if !ok {
			return value
		}
		required := make(map[string]bool, len(schema.Required))
		for _, name := range schema.Required {
			required[name] = true
		}
		out := make(map[string]any, len(object))
		for name, childValue := range object {
			child, hasSchema := schema.Properties[name]
			if hasSchema && childValue == nil && !required[name] && !child.Nullable && child.Type != "" {
				continue
			}
			if hasSchema {
				childValue = NormalizeOptionalNulls(childValue, child)
			}
			out[name] = childValue
		}
		return out
	case schemaTypeArray:
		items, ok := value.([]any)
		if !ok || schema.Items == nil {
			return value
		}
		out := make([]any, len(items))
		for index, item := range items {
			out[index] = NormalizeOptionalNulls(item, *schema.Items)
		}
		return out
	default:
		return value
	}
}

// NormalizeJSONContent validates and canonicalizes output while translating
// strict-provider null placeholders for optional fields back to omission.
func NormalizeJSONContent(content string, schema JSONSchema) (string, error) {
	value, err := DecodeJSONValue(content)
	if err != nil {
		return "", err
	}
	value = NormalizeOptionalNulls(value, schema)
	if validationErr := schema.ValidateValue(value); validationErr != nil {
		return "", validationErr
	}
	data, err := json.Marshal(value)
	if err != nil {
		return "", fmt.Errorf("llm: encode normalized JSON output: %w", err)
	}
	return string(data), nil
}

// ValidateToolSpecs checks that tool names are unique and each argument/result
// contract is a valid object schema before a provider request is sent.
func ValidateToolSpecs(tools []ToolSpec) error {
	seen := make(map[string]struct{}, len(tools))
	for index, tool := range tools {
		name := tool.Function.Name
		if tool.Type != toolTypeFunction {
			return fmt.Errorf("llm: tool %d has unsupported type %q", index, tool.Type)
		}
		if strings.TrimSpace(name) == "" {
			return fmt.Errorf("llm: tool %d is missing a function name", index)
		}
		if _, exists := seen[name]; exists {
			return fmt.Errorf("llm: duplicate function tool %q", name)
		}
		seen[name] = struct{}{}
		schema, err := ParseJSONSchema(tool.Function.Parameters)
		if err != nil {
			return fmt.Errorf("llm: tool %q parameters: %w", name, err)
		}
		if schema.Type != schemaTypeObject {
			return fmt.Errorf("llm: tool %q parameters must use an object schema", name)
		}
		if tool.Function.ResultSchema != nil {
			if _, err := ParseJSONSchema(tool.Function.ResultSchema); err != nil {
				return fmt.Errorf("llm: tool %q result schema: %w", name, err)
			}
		}
	}
	return nil
}

// ValidateAndNormalizeToolCalls validates provider-produced calls against the
// request's argument contracts, rejecting malformed, trailing, or unknown data.
func ValidateAndNormalizeToolCalls(calls []ToolCall, tools []ToolSpec) error {
	if len(calls) == 0 {
		return nil
	}
	if len(tools) == 0 {
		return errors.New("llm: provider returned tool calls when no tools were supplied")
	}
	byName := make(map[string]JSONSchema, len(tools))
	for _, tool := range tools {
		schema, err := ParseJSONSchema(tool.Function.Parameters)
		if err != nil {
			return fmt.Errorf("llm: tool %q parameters: %w", tool.Function.Name, err)
		}
		byName[tool.Function.Name] = schema
	}
	seenIDs := make(map[string]struct{}, len(calls))
	for index := range calls {
		call := &calls[index]
		if call.ID == "" {
			return fmt.Errorf("llm: tool call %d is missing an id", index)
		}
		if _, exists := seenIDs[call.ID]; exists {
			return fmt.Errorf("llm: duplicate tool call id %q", call.ID)
		}
		seenIDs[call.ID] = struct{}{}
		if call.Type != "" && call.Type != toolTypeFunction {
			return fmt.Errorf("llm: tool call %q has unsupported type %q", call.ID, call.Type)
		}
		schema, exists := byName[call.Function.Name]
		if !exists {
			return fmt.Errorf("llm: provider returned unknown tool %q", call.Function.Name)
		}
		value, err := DecodeJSONValue(call.Function.Arguments)
		if err != nil {
			return fmt.Errorf("llm: tool call %q arguments: %w", call.ID, err)
		}
		if _, ok := value.(map[string]any); !ok {
			return fmt.Errorf("llm: tool call %q arguments must be a JSON object", call.ID)
		}
		value = NormalizeOptionalNulls(value, schema)
		if validationErr := schema.ValidateValue(value); validationErr != nil {
			return fmt.Errorf("llm: tool call %q arguments: %w", call.ID, validationErr)
		}
		encoded, err := json.Marshal(value)
		if err != nil {
			return fmt.Errorf("llm: encode normalized tool call %q arguments: %w", call.ID, err)
		}
		call.Function.Arguments = string(encoded)
	}
	return nil
}

// ValidateMessageToolCalls parses historical assistant calls and applies any
// matching argument contract before a conversation reaches a provider.
func ValidateMessageToolCalls(messages []Message, tools []ToolSpec) error {
	byName := make(map[string]JSONSchema, len(tools))
	for _, tool := range tools {
		schema, err := ParseJSONSchema(tool.Function.Parameters)
		if err != nil {
			return fmt.Errorf("llm: tool %q parameters: %w", tool.Function.Name, err)
		}
		byName[tool.Function.Name] = schema
	}
	seenIDs := make(map[string]struct{})
	for messageIndex, message := range messages {
		if message.Role != RoleAssistant {
			if len(message.ToolCalls) > 0 {
				return fmt.Errorf("llm: non-assistant message %d contains tool calls", messageIndex)
			}
			continue
		}
		for callIndex, call := range message.ToolCalls {
			if call.ID == "" {
				return fmt.Errorf("llm: historical tool call %d in message %d is missing an id", callIndex, messageIndex)
			}
			if _, exists := seenIDs[call.ID]; exists {
				return fmt.Errorf("llm: duplicate historical tool call id %q", call.ID)
			}
			seenIDs[call.ID] = struct{}{}
			if call.Type != "" && call.Type != toolTypeFunction {
				return fmt.Errorf("llm: historical tool call %q has unsupported type %q", call.ID, call.Type)
			}
			value, err := DecodeJSONValue(call.Function.Arguments)
			if err != nil {
				return fmt.Errorf("llm: historical tool call %q arguments: %w", call.ID, err)
			}
			if _, ok := value.(map[string]any); !ok {
				return fmt.Errorf("llm: historical tool call %q arguments must be a JSON object", call.ID)
			}
			if schema, exists := byName[call.Function.Name]; exists {
				value = NormalizeOptionalNulls(value, schema)
				if err := schema.ValidateValue(value); err != nil {
					return fmt.Errorf("llm: historical tool call %q arguments: %w", call.ID, err)
				}
			}
		}
	}
	return nil
}

// ValidateToolResults checks tool-result messages for every explicitly declared
// host result contract. Result contracts are never sent to provider APIs.
func ValidateToolResults(messages []Message, tools []ToolSpec) error {
	if len(messages) == 0 {
		return nil
	}
	byName := make(map[string]map[string]any, len(tools))
	for index := range tools {
		byName[tools[index].Function.Name] = tools[index].Function.ResultSchema
	}
	pendingCalls := make(map[string]string)
	seenCallIDs := make(map[string]struct{})
	completedCalls := make(map[string]struct{})
	for index, message := range messages {
		if message.Role == RoleAssistant {
			for _, call := range message.ToolCalls {
				if call.ID == "" {
					return fmt.Errorf("llm: historical tool call in message %d is missing an id", index)
				}
				if _, exists := seenCallIDs[call.ID]; exists {
					return fmt.Errorf("llm: duplicate historical tool call id %q", call.ID)
				}
				seenCallIDs[call.ID] = struct{}{}
				pendingCalls[call.ID] = call.Function.Name
			}
			continue
		}
		if message.Role != RoleTool {
			continue
		}
		if message.ToolCallID == "" {
			return fmt.Errorf("llm: tool result message %d is missing tool_call_id", index)
		}
		callName, exists := pendingCalls[message.ToolCallID]
		if !exists {
			if _, completed := completedCalls[message.ToolCallID]; completed {
				return fmt.Errorf("llm: duplicate tool result for call %q", message.ToolCallID)
			}
			return fmt.Errorf("llm: tool result %q has no preceding assistant tool call", message.ToolCallID)
		}
		if message.Name != "" && callName != message.Name {
			return fmt.Errorf("llm: tool result %q name does not match its tool call", message.ToolCallID)
		}
		delete(pendingCalls, message.ToolCallID)
		completedCalls[message.ToolCallID] = struct{}{}
		schemaValue, known := byName[callName]
		if !known {
			for _, candidate := range byName {
				if candidate != nil {
					return fmt.Errorf("llm: tool result message %d cannot be matched to a declared result schema", index)
				}
			}
			continue
		}
		if schemaValue == nil {
			continue
		}
		schema, err := ParseJSONSchema(schemaValue)
		if err != nil {
			return fmt.Errorf("llm: tool %q result schema: %w", callName, err)
		}
		if err := ValidateJSONContent(message.Content, schema); err != nil {
			return fmt.Errorf("llm: tool result %q: %w", callName, err)
		}
	}
	return nil
}

func (s JSONSchema) validateObject(value any, path string) error {
	object, ok := value.(map[string]any)
	if !ok {
		return typeError(schemaTypeObject, value, path)
	}
	for _, name := range s.Required {
		if _, exists := object[name]; !exists {
			return fmt.Errorf("llm: missing required property %q at %s", name, path)
		}
	}
	for name, childValue := range object {
		child, exists := s.Properties[name]
		if !exists {
			if s.AdditionalProperties != nil && !*s.AdditionalProperties {
				return fmt.Errorf("llm: unexpected property %q at %s", name, path)
			}
			continue
		}
		if err := child.validateValue(childValue, path+"."+name); err != nil {
			return err
		}
	}
	return nil
}

func (s JSONSchema) validateArray(value any, path string) error {
	items, ok := value.([]any)
	if !ok {
		return typeError(schemaTypeArray, value, path)
	}
	for index, item := range items {
		if err := s.Items.validateValue(item, fmt.Sprintf("%s[%d]", path, index)); err != nil {
			return err
		}
	}
	return nil
}

func (s JSONSchema) validateString(value any, path string) error {
	text, ok := value.(string)
	if !ok {
		return typeError(schemaTypeString, value, path)
	}
	if len(s.Enum) == 0 || slices.Contains(s.Enum, text) {
		return nil
	}
	return fmt.Errorf("llm: value at %s is not in the allowed enum", path)
}
