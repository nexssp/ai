package llm

import (
	"net/url"
	"strings"
)

// CapabilityLevel is deliberately not a bool: an API can implement a feature
// only for particular models or server configurations, or may be unaudited.
type CapabilityLevel string

const (
	CapabilityUnknown     CapabilityLevel = "unknown"
	CapabilityUnsupported CapabilityLevel = "unsupported"
	CapabilityConditional CapabilityLevel = "conditional"
	CapabilitySupported   CapabilityLevel = "supported"
)

// ProviderCapabilities describes the active adapter route, not every API a
// vendor offers. Conditional means the caller-selected model/server must also
// support the feature; PAI never retries by changing models or routes.
type ProviderCapabilities struct {
	JSONMode              CapabilityLevel `json:"json_mode"`
	OutputSchema          CapabilityLevel `json:"output_schema"`
	ToolCalls             CapabilityLevel `json:"tool_calls"`
	OutputSchemaWithTools CapabilityLevel `json:"output_schema_with_tools"`
	StrictToolArguments   CapabilityLevel `json:"strict_tool_arguments"`
	ToolResultHistory     CapabilityLevel `json:"tool_result_history"`
	StreamingOutputSchema CapabilityLevel `json:"streaming_output_schema"`
	StreamingToolCalls    CapabilityLevel `json:"streaming_tool_calls"`
}

// CapabilityProvider exposes provider-specific route/model feature knowledge.
type CapabilityProvider interface {
	Capabilities(model string) ProviderCapabilities
}

// OutputSchemaSupportProvider refines route-level capability metadata for a
// particular schema and feature combination.
type OutputSchemaSupportProvider interface {
	SupportsOutputSchema(model string, schema JSONSchema, withTools bool) (supported bool, reason string)
}

// CapabilitiesFor obtains capability information without guessing about custom
// Provider implementations, for which every undeclared feature is unknown.
func CapabilitiesFor(provider Provider, model string) ProviderCapabilities {
	if provider == nil {
		return unknownCapabilities()
	}
	if source, ok := provider.(CapabilityProvider); ok {
		return source.Capabilities(model)
	}
	return unknownCapabilities()
}

// CapabilitiesForRoute is also used by adapters with thin provider wrappers.
// baseURL is inspected only to distinguish an explicitly configured Ollama
// Cloud endpoint; it never causes a route change.
func CapabilitiesForRoute(name, model, baseURL string) ProviderCapabilities {
	name = strings.ToLower(strings.TrimSpace(name))
	model = strings.ToLower(strings.TrimSpace(model))
	switch name {
	case "openai":
		return ProviderCapabilities{
			JSONMode: CapabilitySupported, OutputSchema: CapabilityConditional,
			ToolCalls: CapabilityConditional, OutputSchemaWithTools: CapabilityUnknown,
			StrictToolArguments: CapabilityConditional, ToolResultHistory: CapabilityConditional,
			StreamingOutputSchema: CapabilityUnsupported, StreamingToolCalls: CapabilityUnsupported,
		}
	case "groq":
		return ProviderCapabilities{
			JSONMode: CapabilityConditional, OutputSchema: CapabilityConditional,
			ToolCalls: CapabilityConditional, OutputSchemaWithTools: CapabilityUnsupported,
			StrictToolArguments: CapabilityUnknown, ToolResultHistory: CapabilityConditional,
			StreamingOutputSchema: CapabilityUnsupported, StreamingToolCalls: CapabilityUnsupported,
		}
	case "nvidia", "nim":
		return ProviderCapabilities{
			JSONMode: CapabilityConditional, OutputSchema: CapabilityConditional,
			ToolCalls: CapabilityConditional, OutputSchemaWithTools: CapabilityUnknown,
			StrictToolArguments: CapabilityConditional, ToolResultHistory: CapabilityConditional,
			StreamingOutputSchema: CapabilityUnsupported, StreamingToolCalls: CapabilityUnsupported,
		}
	case "deepseek":
		return ProviderCapabilities{
			JSONMode: CapabilitySupported, OutputSchema: CapabilityUnsupported,
			ToolCalls: CapabilityConditional, OutputSchemaWithTools: CapabilityUnsupported,
			StrictToolArguments: CapabilityUnsupported, ToolResultHistory: CapabilityConditional,
			StreamingOutputSchema: CapabilityUnsupported, StreamingToolCalls: CapabilityUnsupported,
		}
	case "openrouter":
		return ProviderCapabilities{
			JSONMode: CapabilityConditional, OutputSchema: CapabilityConditional,
			ToolCalls: CapabilityConditional, OutputSchemaWithTools: CapabilityUnknown,
			StrictToolArguments: CapabilityConditional, ToolResultHistory: CapabilityConditional,
			StreamingOutputSchema: CapabilityUnsupported, StreamingToolCalls: CapabilityUnsupported,
		}
	case "lmstudio":
		return ProviderCapabilities{
			JSONMode: CapabilityConditional, OutputSchema: CapabilityConditional,
			ToolCalls: CapabilityConditional, OutputSchemaWithTools: CapabilityUnknown,
			StrictToolArguments: CapabilityUnknown, ToolResultHistory: CapabilityConditional,
			StreamingOutputSchema: CapabilityUnsupported, StreamingToolCalls: CapabilityUnsupported,
		}
	case "vllm":
		return ProviderCapabilities{
			JSONMode: CapabilityConditional, OutputSchema: CapabilityConditional,
			ToolCalls: CapabilityConditional, OutputSchemaWithTools: CapabilityUnknown,
			StrictToolArguments: CapabilityConditional, ToolResultHistory: CapabilityConditional,
			StreamingOutputSchema: CapabilityUnsupported, StreamingToolCalls: CapabilityUnsupported,
		}
	case "xai":
		if strings.HasPrefix(model, "grok-4") {
			return ProviderCapabilities{
				JSONMode: CapabilityConditional, OutputSchema: CapabilityConditional,
				ToolCalls: CapabilityConditional, OutputSchemaWithTools: CapabilityConditional,
				StrictToolArguments: CapabilityConditional, ToolResultHistory: CapabilityConditional,
				StreamingOutputSchema: CapabilityUnsupported, StreamingToolCalls: CapabilityUnsupported,
			}
		}
		return ProviderCapabilities{
			JSONMode: CapabilityConditional, OutputSchema: CapabilityUnknown,
			ToolCalls: CapabilityConditional, OutputSchemaWithTools: CapabilityUnknown,
			StrictToolArguments: CapabilityConditional, ToolResultHistory: CapabilityConditional,
			StreamingOutputSchema: CapabilityUnsupported, StreamingToolCalls: CapabilityUnsupported,
		}
	case "inception":
		return ProviderCapabilities{
			JSONMode: CapabilityConditional, OutputSchema: CapabilityConditional,
			ToolCalls: CapabilityConditional, OutputSchemaWithTools: CapabilityUnknown,
			StrictToolArguments: CapabilityConditional, ToolResultHistory: CapabilityConditional,
			StreamingOutputSchema: CapabilityUnsupported, StreamingToolCalls: CapabilityUnsupported,
		}
	case providerAnthropic, providerClaude:
		return ProviderCapabilities{
			JSONMode: CapabilityUnsupported, OutputSchema: CapabilityConditional,
			ToolCalls: CapabilityConditional, OutputSchemaWithTools: CapabilityConditional,
			StrictToolArguments: CapabilityConditional, ToolResultHistory: CapabilitySupported,
			StreamingOutputSchema: CapabilityUnsupported, StreamingToolCalls: CapabilityUnsupported,
		}
	case providerGemini, providerGoogle:
		combined := CapabilityUnknown
		if strings.HasPrefix(model, "gemini-3") || strings.HasPrefix(model, "gemini-4") {
			combined = CapabilityConditional
		}
		return ProviderCapabilities{
			JSONMode: CapabilityConditional, OutputSchema: CapabilityConditional,
			ToolCalls: CapabilityConditional, OutputSchemaWithTools: combined,
			StrictToolArguments: CapabilityUnknown, ToolResultHistory: CapabilityConditional,
			StreamingOutputSchema: CapabilityUnsupported, StreamingToolCalls: CapabilityUnsupported,
		}
	case providerOllama, providerOllamaCloud:
		if isOllamaCloudURL(baseURL) {
			return ProviderCapabilities{
				JSONMode: CapabilityUnknown, OutputSchema: CapabilityUnsupported,
				ToolCalls: CapabilityConditional, OutputSchemaWithTools: CapabilityUnsupported,
				StrictToolArguments: CapabilityUnsupported, ToolResultHistory: CapabilityConditional,
				StreamingOutputSchema: CapabilityUnsupported, StreamingToolCalls: CapabilityUnsupported,
			}
		}
		return ProviderCapabilities{
			JSONMode: CapabilityConditional, OutputSchema: CapabilityConditional,
			ToolCalls: CapabilityConditional, OutputSchemaWithTools: CapabilityUnknown,
			StrictToolArguments: CapabilityUnknown, ToolResultHistory: CapabilityConditional,
			StreamingOutputSchema: CapabilityUnsupported, StreamingToolCalls: CapabilityUnsupported,
		}
	case "openai-compatible":
		return unknownCapabilities()
	case "typesafe", "inception-chat":
		return ProviderCapabilities{
			JSONMode: CapabilityUnsupported, OutputSchema: CapabilityUnsupported,
			ToolCalls: CapabilityUnsupported, OutputSchemaWithTools: CapabilityUnsupported,
			StrictToolArguments: CapabilityUnsupported, ToolResultHistory: CapabilityUnsupported,
			StreamingOutputSchema: CapabilityUnsupported, StreamingToolCalls: CapabilityUnsupported,
		}
	default:
		return unknownCapabilities()
	}
}

func configuredModel(model, defaultModel string) string {
	if strings.TrimSpace(model) == "" {
		return defaultModel
	}
	return model
}

func unknownCapabilities() ProviderCapabilities {
	return ProviderCapabilities{
		JSONMode: CapabilityUnknown, OutputSchema: CapabilityUnknown,
		ToolCalls: CapabilityUnknown, OutputSchemaWithTools: CapabilityUnknown,
		StrictToolArguments: CapabilityUnknown, ToolResultHistory: CapabilityUnknown,
		StreamingOutputSchema: CapabilityUnknown, StreamingToolCalls: CapabilityUnknown,
	}
}

func isOllamaCloudURL(rawURL string) bool {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return false
	}
	host := strings.ToLower(parsed.Hostname())
	return host == "ollama.com" || strings.HasSuffix(host, ".ollama.com")
}
