package nexssflow

import (
	"net/url"
	"os"

	"github.com/nexssp/kernel/xerr"

	"github.com/nexssp/ai/engine/llm"
)

// buildProvider resolves a provider from a declaration, falling back to
// the global options, then to preset defaults, then to environment
// variables owned by the preset. The optional @require api_key_env and
// base_url values override the preset defaults.
func buildProvider(global GlobalOptions, decl LLMDecl) (llm.Provider, error) {
	providerName := firstNonEmpty(decl.Provider, global.Provider, "openai")
	baseURL := firstNonEmpty(decl.BaseURL, global.BaseURL)
	if baseURL != "" {
		parsed, err := url.Parse(baseURL)
		if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Hostname() == "" {
			return nil, xerr.Validation("ai: base_url must be an absolute HTTP(S) URL")
		}
	}
	if global.Timeout < 0 {
		return nil, xerr.Validation("ai: timeout must be positive")
	}

	envName := global.APIKeyEnv
	apiKey := ""
	if envName != "" {
		apiKey = os.Getenv(envName)
	}

	cfg := llm.Config{
		Name:         providerName,
		BaseURL:      baseURL,
		APIKey:       apiKey,
		DefaultModel: firstNonEmpty(decl.Model, global.DefaultModel),
		Timeout:      global.Timeout,
	}
	return llm.NewProvider(providerName, cfg)
}
