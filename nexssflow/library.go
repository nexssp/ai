// Package nexssflow exposes reusable AI actions to Flow v0.42 and newer.
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
)

const ID = "ai"

const criticModule = "nflows/lib/critic.nflow"

var acceptedOptions = []string{"provider", "model", "api_key_env", "timeout", "base_url"}

func init() {
	compiler.Register(ID, Bundle)
}

// GlobalOptions are supplied by Flow's @require options block.
type GlobalOptions struct {
	APIKeyEnv    string
	Timeout      time.Duration
	DefaultModel string
	Provider     string
	BaseURL      string
}

// LLMDecl contains the per-call values accepted by ai.prompt.
type LLMDecl struct {
	Provider string
	Model    string
	BaseURL  string
}

// Bundle exposes AI actions as a Flow v0.42 library using canonical names.
func Bundle(raw map[string]string) compiler.Bundle {
	global, err := options(raw)
	builder := providerBuilder(buildProvider)
	if err != nil {
		builder = func(GlobalOptions, LLMDecl) (llm.Provider, error) { return nil, err }
	}
	return bundleWithBuilder(global, builder)
}

func options(raw map[string]string) (GlobalOptions, error) {
	for name := range raw {
		if !slices.Contains(acceptedOptions, name) {
			return GlobalOptions{}, xerr.Validation(fmt.Sprintf("ai: unknown @require option %q", name))
		}
	}
	global := GlobalOptions{
		APIKeyEnv:    raw["api_key_env"],
		Timeout:      60 * time.Second,
		DefaultModel: raw["model"],
		Provider:     raw["provider"],
		BaseURL:      raw["base_url"],
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
	return compiler.Bundle{
		ID:              ID,
		SourceModules:   siaimodules.SourceModules,
		ExportedModules: []string{criticModule},
		AcceptedOptions: append([]string(nil), acceptedOptions...),
		Libraries: []action.Library{{
			Name: "ai",
			Actions: []action.AnyAction{
				newPromptActionWith(global, builder),
				newTeamActionWith(global, builder),
			},
		}},
	}
}
