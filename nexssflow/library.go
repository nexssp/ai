// Package nexssflow exposes reusable AI actions to Flow v0.67.1 and newer.
package nexssflow

import (
	"fmt"
	"slices"
	"time"

	"github.com/nexssp/flow/compiler"
	"github.com/nexssp/kernel/action"
	"github.com/nexssp/kernel/xerr"

	siaimodules "github.com/nexssp/ai"
	"github.com/nexssp/ai/engine/llm"
	"github.com/nexssp/ai/retrieval"
)

const ID = "ai"

const (
	criticModule   = "nflows/lib/critic.nflow"
	weightsModule  = "nflows/lib/weights.nflow"
	llamaCppModule = "nflows/lib/llama_cpp.nflow"
)

var acceptedOptions = []string{
	"provider", "model", "api_key_env", "timeout", "base_url",
	"decision_provider", "decision_model", "decision_api_key_env", "decision_base_url",
}

func init() {
	compiler.Register(ID, Bundle)
}

// GlobalOptions are supplied by Flow's @require options block.
type GlobalOptions struct {
	APIKeyEnv         string
	Timeout           time.Duration
	DefaultModel      string
	Provider          string
	BaseURL           string
	DecisionAPIKeyEnv string
	DecisionModel     string
	DecisionProvider  string
	DecisionBaseURL   string
}

// LLMDecl contains the per-call values accepted by ai.prompt.
type LLMDecl struct {
	Provider string
	Model    string
	BaseURL  string
}

// DecisionDecl selects the separately configured decision-capable provider.
type DecisionDecl struct {
	Provider string
	Model    string
	BaseURL  string
}

// Bundle exposes AI actions as a Flow v0.67.1 library using canonical names.
func Bundle(raw map[string]string) compiler.Bundle {
	global, err := options(raw)
	builder := providerBuilder(buildProvider)
	if err != nil {
		builder = func(GlobalOptions, LLMDecl) (llm.Provider, error) { return nil, err }
	}
	return bundleWithBuilder(global, builder)
}

// NewBundleWithProviderBuilder creates an AI Flow bundle whose prompt and team
// actions use a host-supplied provider builder. Hosts can use it to inject a
// gateway while retaining the standard typed actions and default decision
// provider builder.
func NewBundleWithProviderBuilder(global GlobalOptions, builder ProviderBuilder) (compiler.Bundle, error) {
	if builder == nil {
		return compiler.Bundle{}, xerr.Validation("ai: provider builder is required")
	}
	return bundleWithBuilder(global, builder), nil
}

// NewBundleWithServices creates a Flow bundle with an injected retrieval
// service and provider builder. The service owns the embedder and vector index;
// a host may use retrieval.NewMemoryIndex for tests/demos or provide an adapter
// backed by its selected persistence implementation.
func NewBundleWithServices(global GlobalOptions, builder ProviderBuilder, service *retrieval.Service) (compiler.Bundle, error) {
	if builder == nil {
		return compiler.Bundle{}, xerr.Validation("ai: provider builder is required")
	}
	if service == nil {
		return compiler.Bundle{}, xerr.Validation("ai: retrieval service is required")
	}
	return bundleWithServices(global, builder, service), nil
}

func options(raw map[string]string) (GlobalOptions, error) {
	for name := range raw {
		if !slices.Contains(acceptedOptions, name) {
			return GlobalOptions{}, xerr.Validation(fmt.Sprintf("ai: unknown @require option %q", name))
		}
	}
	global := GlobalOptions{
		APIKeyEnv:         raw["api_key_env"],
		Timeout:           60 * time.Second,
		DefaultModel:      raw["model"],
		Provider:          raw["provider"],
		BaseURL:           raw["base_url"],
		DecisionAPIKeyEnv: raw["decision_api_key_env"],
		DecisionModel:     raw["decision_model"],
		DecisionProvider:  raw["decision_provider"],
		DecisionBaseURL:   raw["decision_base_url"],
	}
	if raw["timeout"] != "" {
		d, err := time.ParseDuration(raw["timeout"])
		if err != nil {
			return global, xerr.Validation(fmt.Sprintf("ai: invalid timeout %q", raw["timeout"]), err)
		}
		if d <= 0 {
			return global, xerr.Validation("ai: timeout must be positive")
		}
		global.Timeout = d
	}
	return global, nil
}

func bundleWithBuilder(global GlobalOptions, builder providerBuilder) compiler.Bundle {
	return bundleWithServices(global, builder, nil)
}

func bundleWithBuilders(global GlobalOptions, builder providerBuilder, decisions decisionProviderBuilder) compiler.Bundle {
	return bundleWithBuilderAndServices(global, builder, decisions, nil)
}

func bundleWithServices(global GlobalOptions, builder providerBuilder, service *retrieval.Service) compiler.Bundle {
	return bundleWithBuilderAndServices(global, builder, buildDecisionProvider, service)
}

func bundleWithBuilderAndServices(global GlobalOptions, builder providerBuilder, decisions decisionProviderBuilder, service *retrieval.Service) compiler.Bundle {
	actions := []action.AnyAction{
		newPromptActionWith(global, builder),
		newTeamActionWith(global, builder),
		newDecisionActionWith(global, decisions),
		newCostEstimateAction(),
		newNFlowViewAction(),
	}
	actions = append(actions, newWeightActions()...)
	actions = append(actions, newRetrievalActions(global, builder, service)...)
	return compiler.Bundle{
		ID:              ID,
		SourceModules:   siaimodules.SourceModules,
		ExportedModules: []string{criticModule, weightsModule, llamaCppModule},
		AcceptedOptions: append([]string(nil), acceptedOptions...),
		Libraries: []action.Library{{
			Name:    "ai",
			Actions: actions,
		}},
	}
}
