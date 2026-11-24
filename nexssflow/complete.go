package nexssflow

import (
	"context"
	"strings"
	"time"

	"github.com/nexssp/kernel/xerr"

	"github.com/nexssp/ai/engine/llm"
)

func validateUser(user string) error {
	if strings.TrimSpace(user) == "" {
		return xerr.Validation("ai.prompt: user field is required")
	}
	return nil
}

func validatePromptMessages(req PromptRequest) error {
	if len(req.Messages) == 0 {
		return validateUser(req.User)
	}
	if req.User != "" || req.System != "" {
		return xerr.Validation("ai.prompt: messages cannot be combined with user or system")
	}
	for _, message := range req.Messages {
		if message.Role == llm.RoleUser && strings.TrimSpace(message.Content) != "" {
			return nil
		}
	}
	return xerr.Validation("ai.prompt: messages must include a non-empty user message")
}

func validateToolOptions(req PromptRequest) error {
	if len(req.Tools) == 0 {
		if req.ToolChoice != "" {
			return xerr.Validation("ai.prompt: tool_choice requires at least one tool")
		}
		return nil
	}
	switch req.ToolChoice {
	case "", llm.ToolChoiceAuto, llm.ToolChoiceNone, llm.ToolChoiceRequired:
		return nil
	default:
		return xerr.Validation("ai.prompt: tool_choice must be auto, none, or required")
	}
}

func schemaModeFor(req PromptRequest) (llm.SchemaMode, error) {
	mode := req.OutputSchemaMode
	if mode == "" {
		mode = llm.SchemaModeStrict
	}
	switch mode {
	case llm.SchemaModeStrict, llm.SchemaModeBestEffort:
		return mode, nil
	default:
		return "", xerr.Validation("ai.prompt: output_schema_mode must be strict or best_effort")
	}
}

func schemaSupport(provider llm.Provider, model string, schema llm.JSONSchema, withTools bool) (supported bool, reason string) {
	if support, ok := provider.(llm.OutputSchemaSupportProvider); ok {
		return support.SupportsOutputSchema(model, schema, withTools)
	}
	capabilities := llm.CapabilitiesFor(provider, model)
	level := capabilities.OutputSchema
	if withTools {
		level = capabilities.OutputSchemaWithTools
	}
	if level == llm.CapabilitySupported || level == llm.CapabilityConditional {
		return true, ""
	}
	return false, "route capability is " + string(level)
}

func capabilityAllows(level llm.CapabilityLevel) bool {
	return level == llm.CapabilitySupported || level == llm.CapabilityConditional
}

// complete runs exactly one provider attempt; selection and fallback belong to the caller.
func complete(ctx context.Context, provider llm.Provider, req PromptRequest) (PromptResult, error) {
	if err := validateToolOptions(req); err != nil {
		return PromptResult{}, err
	}
	mode, err := schemaModeFor(req)
	if err != nil {
		return PromptResult{}, err
	}
	priced, err := validatePricingAndLimits(req)
	if err != nil {
		return PromptResult{}, err
	}
	if providerErr := requireProvider(provider); providerErr != nil {
		return PromptResult{}, providerErr
	}
	if req.Stream && (len(req.Tools) > 0 || req.OutputSchema != nil) {
		return PromptResult{}, xerr.Validation("ai.prompt: streaming supports text-only responses without tools or output_schema")
	}
	messages, err := prepareCompletionMessages(provider, req)
	if err != nil {
		return PromptResult{}, err
	}
	tools, localToolValidation, err := prepareToolContracts(provider, req, mode)
	if err != nil {
		return PromptResult{}, err
	}
	outputSchema, localOutputValidation, err := prepareOutputContract(provider, req, tools, mode)
	if err != nil {
		return PromptResult{}, err
	}
	if req.Stream {
		return streamCompletion(ctx, provider, req, mode, messages, tools, outputSchema)
	}

	response, err := provider.Complete(ctx, requestForCompletion(req, mode, messages, tools, outputSchema))
	if err != nil {
		return PromptResult{}, completionError(err)
	}
	return finishCompletion(provider, req, response, tools, outputSchema, priced, localToolValidation || localOutputValidation, true)
}

func streamCompletion(ctx context.Context, provider llm.Provider, req PromptRequest, mode llm.SchemaMode, messages []llm.Message, tools []llm.ToolSpec, outputSchema *llm.JSONSchema) (PromptResult, error) {
	sequence, err := provider.Stream(ctx, requestForCompletion(req, mode, messages, tools, outputSchema))
	if err != nil {
		return PromptResult{}, completionError(err)
	}
	if sequence == nil {
		return PromptResult{}, xerr.Internal("ai.prompt: provider returned a nil stream")
	}

	sink := streamSinkFor(ctx)
	var content strings.Builder
	var sequenceNumber uint64
	for delta, streamErr := range sequence {
		if streamErr != nil {
			return PromptResult{}, completionError(streamErr)
		}
		content.WriteString(delta)
		if sink != nil && delta != "" {
			if err := sink.Emit(ctx, llm.StreamEvent{
				Sequence: sequenceNumber,
				Time:     time.Now(),
				Kind:     llm.EventTokenDelta,
				Action:   "ai.prompt",
				Delta:    delta,
			}); err != nil {
				return PromptResult{}, completionError(err)
			}
			sequenceNumber++
		}
	}
	return finishCompletion(provider, req, llm.Response{Model: req.Model, Content: content.String()}, tools, outputSchema, false, false, false)
}

func validatePricingAndLimits(req PromptRequest) (bool, error) {
	priced := req.RateCardID != "" || req.ExpectedCurrency != nil
	if err := validatePricingFields("ai.prompt", req.Model, req.RateCardID, req.ExpectedCurrency, false); err != nil {
		return false, err
	}
	if priced {
		if err := validateConfiguredPricing("ai.prompt", req.Model, req.RateCardID, req.ExpectedCurrency); err != nil {
			return false, err
		}
	}
	if err := validateTokenBasis("ai.prompt", req.PromptTokens, req.CachedPromptTokens, req.CompletionTokens, priced); err != nil {
		return false, err
	}
	if req.MaxTokens < 0 {
		return false, xerr.Validation("ai.prompt: max_tokens cannot be negative")
	}
	return priced, nil
}

func prepareCompletionMessages(provider llm.Provider, req PromptRequest) ([]llm.Message, error) {
	if err := validatePromptMessages(req); err != nil {
		return nil, err
	}
	if len(req.Tools) > 0 {
		if err := llm.ValidateToolSpecs(req.Tools); err != nil {
			return nil, xerr.Validation("ai.prompt: " + err.Error())
		}
		toolProvider, ok := provider.(llm.ToolCallingProvider)
		if !ok || !toolProvider.SupportsToolCalls() {
			return nil, xerr.Validation("ai.prompt: provider " + provider.Name() + " does not support non-streaming function tool calls")
		}
	}
	if req.ToolChoice != "" {
		choiceProvider, ok := provider.(llm.ToolChoiceProvider)
		if !ok || !choiceProvider.SupportsToolChoice(req.ToolChoice) {
			return nil, xerr.Validation("ai.prompt: provider " + provider.Name() + " does not support tool_choice " + string(req.ToolChoice))
		}
	}
	messages := req.Messages
	if len(messages) == 0 {
		messages = make([]llm.Message, 0, 2)
		if req.System != "" {
			messages = append(messages, llm.Message{Role: llm.RoleSystem, Content: req.System})
		}
		messages = append(messages, llm.Message{Role: llm.RoleUser, Content: req.User})
	}
	if err := llm.ValidateMessageToolCalls(messages, req.Tools); err != nil {
		return nil, xerr.Validation("ai.prompt: " + err.Error())
	}
	if err := llm.ValidateToolResults(messages, req.Tools); err != nil {
		return nil, xerr.Validation("ai.prompt: " + err.Error())
	}
	return messages, nil
}

func prepareToolContracts(provider llm.Provider, req PromptRequest, mode llm.SchemaMode) ([]llm.ToolSpec, bool, error) {
	tools := append([]llm.ToolSpec(nil), req.Tools...)
	capabilities := llm.CapabilitiesFor(provider, req.Model)
	localValidation := false
	for index := range tools {
		if !tools[index].Function.Strict || capabilityAllows(capabilities.StrictToolArguments) {
			continue
		}
		if mode != llm.SchemaModeBestEffort {
			return nil, false, xerr.Validation("ai.prompt: provider " + provider.Name() + " cannot guarantee strict tool arguments (capability " + string(capabilities.StrictToolArguments) + "); select output_schema_mode=best_effort for local validation")
		}
		tools[index].Function.Strict = false
		localValidation = true
	}
	return tools, localValidation, nil
}

func prepareOutputContract(provider llm.Provider, req PromptRequest, tools []llm.ToolSpec, mode llm.SchemaMode) (*llm.JSONSchema, bool, error) {
	if req.OutputSchema == nil {
		return nil, false, nil
	}
	if err := req.OutputSchema.ValidateDefinition(); err != nil {
		return nil, false, xerr.Validation("ai.prompt: output_schema: " + err.Error())
	}
	supported, reason := schemaSupport(provider, req.Model, *req.OutputSchema, len(tools) > 0)
	if !supported && mode != llm.SchemaModeBestEffort {
		return nil, false, xerr.Validation("ai.prompt: provider " + provider.Name() + " cannot enforce the requested output schema (" + reason + "); select output_schema_mode=best_effort for local validation")
	}
	if !supported {
		return nil, true, nil
	}
	return req.OutputSchema, false, nil
}

func requestForCompletion(req PromptRequest, mode llm.SchemaMode, messages []llm.Message, tools []llm.ToolSpec, outputSchema *llm.JSONSchema) llm.Request {
	return llm.Request{
		Model: req.Model, CandidateID: req.CandidateID, AttemptID: req.AttemptID,
		RateCardID: req.RateCardID, ExpectedCurrency: cloneCurrency(req.ExpectedCurrency),
		EstimatedPromptTokens: req.PromptTokens, EstimatedCachedPromptTokens: req.CachedPromptTokens,
		EstimatedCompletionTokens: req.CompletionTokens, Messages: messages, Tools: tools,
		ToolChoice: req.ToolChoice, MaxTokens: req.MaxTokens, Temperature: req.Temperature,
		JSONMode: req.JSONMode, OutputSchema: outputSchema, OutputSchemaMode: mode,
		ReasoningEffort: req.ReasoningEffort,
	}
}

func finishCompletion(provider llm.Provider, req PromptRequest, response llm.Response, tools []llm.ToolSpec, outputSchema *llm.JSONSchema, priced, localValidation, usageAvailable bool) (PromptResult, error) {
	if err := llm.ValidateAndNormalizeToolCalls(response.ToolCalls, tools); err != nil {
		return PromptResult{}, xerr.Validation("ai.prompt: provider returned invalid tool calls: " + err.Error())
	}
	enforcement := llm.SchemaEnforcementNone
	if req.OutputSchema != nil {
		if outputSchema != nil {
			enforcement = llm.SchemaEnforcementProvider
		} else {
			enforcement = llm.SchemaEnforcementLocal
		}
		if len(response.ToolCalls) == 0 || strings.TrimSpace(response.Content) != "" {
			content, err := llm.NormalizeJSONContent(response.Content, *req.OutputSchema)
			if err != nil {
				return PromptResult{}, xerr.Validation("ai.prompt: assistant output does not satisfy output_schema: " + err.Error())
			}
			response.Content = content
		}
	}
	if localValidation {
		enforcement = llm.SchemaEnforcementLocal
	}
	response.SchemaEnforcement = enforcement

	model := response.Model
	if model == "" {
		model = req.Model
	}
	response.Model = model
	var usage llm.Usage
	if usageAvailable {
		usage = llm.FromResponse(response)
		if priced {
			usage = llm.FromResponseForRateCard(response, req.RateCardID, *req.ExpectedCurrency)
		}
	}
	return PromptResult{
		CandidateID: req.CandidateID, AttemptID: req.AttemptID, RateCardID: req.RateCardID,
		ExpectedCurrency: cloneCurrency(req.ExpectedCurrency), Context: req.Context,
		Provider: provider.Name(), Content: response.Content, ToolCalls: response.ToolCalls,
		SchemaEnforcement: enforcement, Model: model, PromptTokens: response.PromptTokens,
		CachedPromptTokens: response.CachedPromptTokens, CompTokens: response.CompTokens,
		TotalTokens: response.PromptTokens + response.CompTokens, UsageAvailable: usageAvailable, Usage: usage,
	}, nil
}
