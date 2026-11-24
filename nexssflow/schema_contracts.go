package nexssflow

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/nexssp/flow/compiler"
	schemaext "github.com/nexssp/flow/extensions/schema"
	"github.com/nexssp/kernel/action"
	"github.com/nexssp/kernel/xerr"

	"github.com/nexssp/ai/engine/llm"
)

const (
	schemaGetActionName  = "ai.schemas.get"
	schemaListActionName = "ai.schemas.list"
)

type schemaGetRequest struct {
	Name string `json:"name"`
}

// schemaContractsMaterializer exposes visible @schema declarations as reusable
// provider-neutral JSON Schema contracts for ai.prompt and tool declarations.
func schemaContractsMaterializer() compiler.Materializer {
	return func(req compiler.MaterializeReq) error {
		contracts := schemaContractsFromMeta(req.Meta)
		get := action.New[schemaGetRequest, map[string]any](schemaGetActionName, func(_ context.Context, input schemaGetRequest) (map[string]any, error) {
			schema, ok := contracts[input.Name]
			if !ok {
				return nil, xerr.Validation(fmt.Sprintf("ai.schemas.get: unknown schema %q", input.Name))
			}
			value, err := flowJSONSchemaValue(schema)
			if err != nil {
				return nil, xerr.Internal("ai.schemas.get: encode schema", err)
			}
			return value, nil
		}).Description("Return a provider-neutral JSON Schema contract derived from @schema").Tag("ai", "schema").Build()
		list := action.New(schemaListActionName, func(_ context.Context, _ any) (map[string]map[string]any, error) {
			values := make(map[string]map[string]any, len(contracts))
			for name := range contracts {
				schema := contracts[name]
				value, err := flowJSONSchemaValue(schema)
				if err != nil {
					return nil, xerr.Internal("ai.schemas.list: encode schema", err)
				}
				values[name] = value
			}
			return values, nil
		}).Description("List provider-neutral JSON Schema contracts derived from @schema").Tag("ai", "schema").Build()
		owner := compiler.DeclarationOwner(req.Meta, ID, "schemas", schemaListActionName)
		if err := req.Resolver.Mount(action.Library{Name: owner, Actions: []action.AnyAction{get, list}}); err != nil {
			return xerr.Internal("ai: mount schema contract actions", err)
		}
		return nil
	}
}

func schemaContractsFromMeta(meta map[string]any) map[string]llm.JSONSchema {
	schemas := schemaext.VisibleSchemasFromMeta(meta)
	contracts := make(map[string]llm.JSONSchema, len(schemas))
	for _, schema := range schemas {
		contracts[schema.Name] = flowSchemaToJSONSchema(schema, true)
	}
	return contracts
}

// flowSchemaToJSONSchema recursively uses Flow's linked NestedSchema metadata.
func flowSchemaToJSONSchema(source schemaext.Schema, closed bool) llm.JSONSchema {
	properties := make(map[string]llm.JSONSchema, len(source.Fields))
	var required []string
	for index := range source.Fields {
		field := &source.Fields[index]
		properties[field.JSONName] = flowFieldToJSONSchema(field, closed)
		if flowFieldRequired(field) {
			required = append(required, field.JSONName)
		}
	}
	result := llm.JSONSchema{Type: "object", Properties: properties, Required: required}
	if closed {
		value := false
		result.AdditionalProperties = &value
	}
	return result
}

func flowFieldToJSONSchema(field *schemaext.Field, closed bool) llm.JSONSchema {
	value := flowFieldBaseSchema(field, closed)
	if field.Slice {
		item := value
		value = llm.JSONSchema{Type: "array", Items: &item}
	}
	if field.Pointer {
		value.Nullable = true
	}
	if description := firstNonEmpty(field.Tags["usage"], field.Tags["description"]); description != "" {
		value.Description = description
	}
	applyFlowFieldEnum(&value, field)
	return value
}

func flowFieldBaseSchema(field *schemaext.Field, closed bool) llm.JSONSchema {
	var value llm.JSONSchema
	switch {
	case field.Map:
		allow := true
		value = llm.JSONSchema{Type: "object", AdditionalProperties: &allow}
	case field.NestedSchema != nil:
		value = flowSchemaToJSONSchema(*field.NestedSchema, closed)
	default:
		switch field.Kind {
		case schemaext.KindString, schemaext.KindSecret:
			value.Type = "string"
			if field.Kind == schemaext.KindSecret {
				value.Format = "password"
			}
		case schemaext.KindInt, schemaext.KindDuration:
			value.Type = "integer"
		case schemaext.KindFloat:
			value.Type = "number"
		case schemaext.KindBool:
			value.Type = "boolean"
		case schemaext.KindStruct:
			value.Type = "object"
		case schemaext.KindAny:
			// An untyped field intentionally accepts any JSON value.
		}
	}
	return value
}

func applyFlowFieldEnum(value *llm.JSONSchema, field *schemaext.Field) {
	enumSchema := value
	if value.Type == "array" && value.Items != nil {
		enumSchema = value.Items
	}
	if len(field.EnumValues) > 0 {
		enumSchema.Type = "string"
		enumSchema.Enum = append([]string(nil), field.EnumValues...)
	} else if enum := field.Tags["enum"]; enum != "" {
		for item := range strings.SplitSeq(enum, ",") {
			if item = strings.TrimSpace(item); item != "" {
				enumSchema.Enum = append(enumSchema.Enum, item)
			}
		}
		if len(enumSchema.Enum) > 0 && enumSchema.Type == "" {
			enumSchema.Type = "string"
		}
	}
}

func flowJSONSchemaValue(schema llm.JSONSchema) (map[string]any, error) {
	data, err := json.Marshal(schema)
	if err != nil {
		return nil, err
	}
	var value map[string]any
	if err := json.Unmarshal(data, &value); err != nil {
		return nil, err
	}
	return value, nil
}

func flowFieldRequired(field *schemaext.Field) bool {
	if field.Required {
		return true
	}
	for part := range strings.SplitSeq(field.Tags["validate"], ",") {
		if strings.TrimSpace(part) == "required" {
			return true
		}
	}
	return false
}
