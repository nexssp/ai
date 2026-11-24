package nexssflow

import (
	"context"
	"fmt"
	"maps"
	"sort"
	"strings"

	"github.com/nexssp/flow/compiler"
	schemaext "github.com/nexssp/flow/extensions/schema"
	"github.com/nexssp/kernel/action"
	"github.com/nexssp/kernel/xerr"

	"github.com/nexssp/ai/engine/llm"
)

const (
	toolSchemaModifierName       = "tool-schema"
	toolDescriptionModifierName  = "tool-description"
	toolResultSchemaModifierName = "tool-result-schema"
	toolSpecsActionName          = "ai.tool_specs"
	toolSpecsListActionName      = "ai.tool_specs.list"
	toolSpecsFieldName           = "ai_tool_specs"
)

// pipelineToolModifiers mark the subset of @pipeline declarations that may be
// exposed to an LLM. The schema is separate from the pipeline's runtime input.
func pipelineToolModifiers() []compiler.Modifier {
	return []compiler.Modifier{
		compiler.WithUnique(compiler.String(toolSchemaModifierName, func(b *action.Builder[any, any], value string) *action.Builder[any, any] {
			return b.Tag("ai.tool_schema:" + value)
		})),
		compiler.WithUnique(compiler.String(toolDescriptionModifierName, func(b *action.Builder[any, any], value string) *action.Builder[any, any] {
			return b.Tag("ai.tool_description:" + value)
		})),
		compiler.WithUnique(compiler.String(toolResultSchemaModifierName, func(b *action.Builder[any, any], value string) *action.Builder[any, any] {
			return b.Tag("ai.tool_result_schema:" + value)
		})),
	}
}

func aiMaterializer(global GlobalOptions, builder providerBuilder) compiler.Materializer {
	namedLLM := namedLLMMaterializer(global, builder)
	toolSpecs := pipelineToolSpecsMaterializer()
	schemaContracts := schemaContractsMaterializer()
	return func(req compiler.MaterializeReq) error {
		if err := namedLLM(req); err != nil {
			return err
		}
		if err := toolSpecs(req); err != nil {
			return err
		}
		return schemaContracts(req)
	}
}

func pipelineToolSpecsMaterializer() compiler.Materializer {
	return func(req compiler.MaterializeReq) error {
		specs, err := toolSpecsFromMeta(req.Meta)
		if err != nil {
			return err
		}
		act := action.New(toolSpecsActionName, func(_ context.Context, input map[string]any) (map[string]any, error) {
			if _, exists := input[toolSpecsFieldName]; exists {
				return nil, xerr.Validation("ai.tool_specs: input field ai_tool_specs is reserved")
			}
			output := make(map[string]any, len(input)+1)
			maps.Copy(output, input)
			output[toolSpecsFieldName] = append([]llm.ToolSpec(nil), specs...)
			return output, nil
		}).Description("Attach function tool specs derived from @pipeline and @schema declarations").Tag("ai", "tools").Build()
		listAct := action.New(toolSpecsListActionName, func(_ context.Context, _ any) ([]llm.ToolSpec, error) {
			return append([]llm.ToolSpec(nil), specs...), nil
		}).Description("Return function tool specs derived from @pipeline and @schema declarations").Tag("ai", "tools").Build()
		owner := compiler.DeclarationOwner(req.Meta, ID, "tool_specs", toolSpecsActionName)
		if err := req.Resolver.Mount(action.Library{
			Name:    owner,
			Actions: []action.AnyAction{act, listAct},
		}); err != nil {
			return xerr.Internal("ai: mount pipeline tool spec action", err)
		}
		return nil
	}
}

type pipelineToolConfig struct {
	schemaName       string
	description      string
	resultSchemaName string
	hasSchema        bool
	hasDescription   bool
	hasResultSchema  bool
}

func (config *pipelineToolConfig) add(pipelineName, raw string) error {
	name := compiler.ModifierName(raw)
	_, value, hasValue := strings.Cut(raw, "=")
	switch name {
	case toolSchemaModifierName:
		if err := requireToolModifierValue(pipelineName, toolSchemaModifierName, value, hasValue); err != nil {
			return err
		}
		if config.hasSchema {
			return fmt.Errorf("@pipeline %q: :%s may be specified only once", pipelineName, toolSchemaModifierName)
		}
		config.schemaName, config.hasSchema = strings.TrimSpace(value), true
	case toolDescriptionModifierName:
		if err := requireToolModifierValue(pipelineName, toolDescriptionModifierName, value, hasValue); err != nil {
			return err
		}
		if config.hasDescription {
			return fmt.Errorf("@pipeline %q: :%s may be specified only once", pipelineName, toolDescriptionModifierName)
		}
		config.description, config.hasDescription = strings.TrimSpace(value), true
	case toolResultSchemaModifierName:
		if err := requireToolModifierValue(pipelineName, toolResultSchemaModifierName, value, hasValue); err != nil {
			return err
		}
		if config.hasResultSchema {
			return fmt.Errorf("@pipeline %q: :%s may be specified only once", pipelineName, toolResultSchemaModifierName)
		}
		config.resultSchemaName, config.hasResultSchema = strings.TrimSpace(value), true
	}
	return nil
}

func requireToolModifierValue(pipelineName, modifier, value string, hasValue bool) error {
	if hasValue && strings.TrimSpace(value) != "" {
		return nil
	}
	if modifier == toolDescriptionModifierName {
		return fmt.Errorf("@pipeline %q: :%s requires a description", pipelineName, modifier)
	}
	return fmt.Errorf("@pipeline %q: :%s requires a schema name", pipelineName, modifier)
}

func toolSpecsFromMeta(meta map[string]any) ([]llm.ToolSpec, error) {
	if meta == nil {
		return nil, nil
	}
	pipelines, _ := meta["pipelines"].(map[string]string)
	modifiers, _ := meta["pipeline_modifiers"].(map[string][]string)
	names := make([]string, 0, len(pipelines))
	for name := range pipelines {
		names = append(names, name)
	}
	sort.Strings(names)

	specs := make([]llm.ToolSpec, 0, len(names))
	for _, pipelineName := range names {
		config, err := pipelineToolConfigFromMeta(pipelineName, modifiers[pipelineName])
		if err != nil {
			return nil, err
		}
		spec, exposed, err := pipelineToolSpecFromMeta(meta, pipelineName, config)
		if err != nil {
			return nil, err
		}
		if exposed {
			specs = append(specs, spec)
		}
	}
	return specs, nil
}

func pipelineToolConfigFromMeta(pipelineName string, modifiers []string) (pipelineToolConfig, error) {
	var config pipelineToolConfig
	for _, raw := range modifiers {
		if err := config.add(pipelineName, raw); err != nil {
			return pipelineToolConfig{}, err
		}
	}
	return config, nil
}

func pipelineToolSpecFromMeta(meta map[string]any, pipelineName string, config pipelineToolConfig) (llm.ToolSpec, bool, error) {
	if !config.hasSchema && !config.hasDescription && !config.hasResultSchema {
		return llm.ToolSpec{}, false, nil
	}
	if !config.hasSchema || !config.hasDescription {
		return llm.ToolSpec{}, false, fmt.Errorf("@pipeline %q: :%s and :%s must be declared together", pipelineName, toolSchemaModifierName, toolDescriptionModifierName)
	}
	strict := strings.HasPrefix(config.schemaName, "!")
	schemaName := strings.TrimPrefix(config.schemaName, "!")
	if schemaName == "" {
		return llm.ToolSpec{}, false, fmt.Errorf("@pipeline %q: :%s requires a schema name after the strictness marker", pipelineName, toolSchemaModifierName)
	}
	targetSchema, found := schemaext.ByName(meta, schemaName)
	if !found {
		return llm.ToolSpec{}, false, fmt.Errorf("@pipeline %q: :%s=%s: unknown schema", pipelineName, toolSchemaModifierName, schemaName)
	}
	parameters, err := flowJSONSchemaValue(flowSchemaToJSONSchema(targetSchema, strict))
	if err != nil {
		return llm.ToolSpec{}, false, fmt.Errorf("@pipeline %q: encode tool schema: %w", pipelineName, err)
	}
	var resultSchema map[string]any
	if config.hasResultSchema {
		resultSchema, err = toolResultSchemaFromMeta(meta, pipelineName, config)
		if err != nil {
			return llm.ToolSpec{}, false, err
		}
	}
	return llm.ToolSpec{
		Type: "function",
		Function: llm.FunctionSpec{
			Name: pipelineName, Description: config.description,
			Parameters: parameters, Strict: strict, ResultSchema: resultSchema,
		},
	}, true, nil
}

func toolResultSchemaFromMeta(meta map[string]any, pipelineName string, config pipelineToolConfig) (map[string]any, error) {
	target, found := schemaext.ByName(meta, config.resultSchemaName)
	if !found {
		return nil, fmt.Errorf("@pipeline %q: :%s=%s: unknown schema", pipelineName, toolResultSchemaModifierName, config.resultSchemaName)
	}
	converted := flowSchemaToJSONSchema(target, true)
	value, err := flowJSONSchemaValue(converted)
	if err != nil {
		return nil, fmt.Errorf("@pipeline %q: encode tool result schema: %w", pipelineName, err)
	}
	return value, nil
}
