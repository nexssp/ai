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
	toolSchemaModifierName      = "tool-schema"
	toolDescriptionModifierName = "tool-description"
	toolSpecsActionName         = "ai.tool_specs"
	toolSpecsFieldName          = "ai_tool_specs"
)

// pipelineToolModifiers mark the subset of @pipeline declarations that may be
// exposed to an LLM. The schema is separate from the pipeline's runtime input:
// callers can keep orchestration fields out of the model-visible argument list.
func pipelineToolModifiers() []compiler.Modifier {
	return []compiler.Modifier{
		compiler.WithUnique(compiler.String(toolSchemaModifierName, func(b *action.Builder[any, any], value string) *action.Builder[any, any] {
			return b.Tag("ai.tool_schema:" + value)
		})),
		compiler.WithUnique(compiler.String(toolDescriptionModifierName, func(b *action.Builder[any, any], value string) *action.Builder[any, any] {
			return b.Tag("ai.tool_description:" + value)
		})),
	}
}

func aiMaterializer(global GlobalOptions, builder providerBuilder) compiler.Materializer {
	namedLLM := namedLLMMaterializer(global, builder)
	toolSpecs := pipelineToolSpecsMaterializer()
	return func(req compiler.MaterializeReq) error {
		if err := namedLLM(req); err != nil {
			return err
		}
		return toolSpecs(req)
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
		owner := compiler.DeclarationOwner(req.Meta, ID, "tool_specs", toolSpecsActionName)
		if err := req.Resolver.Mount(action.Library{
			Name:    owner,
			Actions: []action.AnyAction{act},
		}); err != nil {
			return xerr.Internal("ai: mount pipeline tool spec action", err)
		}
		return nil
	}
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
		var schemaName, description string
		var hasSchema, hasDescription bool
		for _, raw := range modifiers[pipelineName] {
			name := compiler.ModifierName(raw)
			_, value, hasValue := strings.Cut(raw, "=")
			switch name {
			case toolSchemaModifierName:
				if !hasValue || strings.TrimSpace(value) == "" {
					return nil, fmt.Errorf("@pipeline %q: :%s requires a schema name", pipelineName, toolSchemaModifierName)
				}
				if hasSchema {
					return nil, fmt.Errorf("@pipeline %q: :%s may be specified only once", pipelineName, toolSchemaModifierName)
				}
				schemaName, hasSchema = strings.TrimSpace(value), true
			case toolDescriptionModifierName:
				if !hasValue || strings.TrimSpace(value) == "" {
					return nil, fmt.Errorf("@pipeline %q: :%s requires a description", pipelineName, toolDescriptionModifierName)
				}
				if hasDescription {
					return nil, fmt.Errorf("@pipeline %q: :%s may be specified only once", pipelineName, toolDescriptionModifierName)
				}
				description, hasDescription = strings.TrimSpace(value), true
			}
		}

		if !hasSchema && !hasDescription {
			continue
		}
		if !hasSchema || !hasDescription {
			return nil, fmt.Errorf("@pipeline %q: :%s and :%s must be declared together", pipelineName, toolSchemaModifierName, toolDescriptionModifierName)
		}
		strict := strings.HasPrefix(schemaName, "!")
		schemaName = strings.TrimPrefix(schemaName, "!")
		if schemaName == "" {
			return nil, fmt.Errorf("@pipeline %q: :%s requires a schema name after the strictness marker", pipelineName, toolSchemaModifierName)
		}
		targetSchema, found := schemaext.ByName(meta, schemaName)
		if !found {
			return nil, fmt.Errorf("@pipeline %q: :%s=%s: unknown schema", pipelineName, toolSchemaModifierName, schemaName)
		}
		parameters := schemaext.ToJSONSchema(targetSchema)
		if strict {
			parameters["additionalProperties"] = false
		}
		specs = append(specs, llm.ToolSpec{
			Type: "function",
			Function: llm.FunctionSpec{
				Name:        pipelineName,
				Description: description,
				Parameters:  parameters,
			},
		})
	}
	return specs, nil
}
