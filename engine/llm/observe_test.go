package llm

import (
	"testing"
	"time"
)

func TestObservableProviderForwardsRequestCapabilities(t *testing.T) {
	inner := NewOpenAI(Config{DefaultModel: "gpt-4o"})
	wrapped := NewObservableProvider(inner, func(string, string, string, int, int, int64, time.Duration, error) {})

	capabilities, ok := wrapped.(CapabilityProvider)
	if !ok {
		t.Fatal("observable provider does not expose capabilities")
	}
	if got := capabilities.Capabilities("gpt-4o").OutputSchema; got != CapabilityConditional {
		t.Fatalf("wrapped output-schema capability = %q, want conditional", got)
	}

	schemaSupport, ok := wrapped.(OutputSchemaSupportProvider)
	if !ok {
		t.Fatal("observable provider does not expose output-schema support")
	}
	closed := false
	schema := JSONSchema{Type: "object", AdditionalProperties: &closed}
	if supported, reason := schemaSupport.SupportsOutputSchema("gpt-4o", schema, false); !supported {
		t.Fatalf("wrapped output schema rejected: %s", reason)
	}

	toolChoice, ok := wrapped.(ToolChoiceProvider)
	if !ok || !toolChoice.SupportsToolChoice(ToolChoiceRequired) {
		t.Fatal("observable provider did not forward required tool-choice support")
	}
}
