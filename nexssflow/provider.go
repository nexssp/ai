package nexssflow

import (
	"net/url"
	"os"

	"github.com/nexssp/kernel/xerr"

	"github.com/nexssp/ai/engine/llm"
)

// buildProvider resolves a one-model call from per-call or global configuration.
func buildProvider(global GlobalOptions, decl LLMDecl) (llm.Provider, error) {
	providerName := firstNonEmpty(decl.Provider, global.Provider, "openai")
	baseURL := firstNonEmpty(decl.BaseURL, global.BaseURL)
	cfg, err := providerConfig(global, providerName, firstNonEmpty(decl.Model, global.DefaultModel), baseURL, global.APIKeyEnv)
	if err != nil {
		return nil, err
	}
	return llm.NewProvider(providerName, cfg)
}

type decisionProviderBuilder func(GlobalOptions, DecisionDecl) (llm.DecisionProvider, error)

// buildDecisionProvider does not inherit the candidate model. The caller must
// configure a dedicated decision model either per action or in @require.
func buildDecisionProvider(global GlobalOptions, decl DecisionDecl) (llm.DecisionProvider, error) {
	providerName := firstNonEmpty(decl.Provider, global.DecisionProvider, global.Provider)
	model := firstNonEmpty(decl.Model, global.DecisionModel)
	if providerName == "" {
		return nil, xerr.Validation("ai.decide: decision_provider is required")
	}
	if model == "" {
		return nil, xerr.Validation("ai.decide: decision_model is required")
	}
	baseURL := firstNonEmpty(decl.BaseURL, global.DecisionBaseURL)
	apiKeyEnv := firstNonEmpty(global.DecisionAPIKeyEnv, global.APIKeyEnv)
	cfg, err := providerConfig(global, providerName, model, baseURL, apiKeyEnv)
	if err != nil {
		return nil, err
	}
	if providerName == "typesafe" {
		return llm.NewTypeSafe(cfg), nil
	}
	constructionName := providerName
	if providerName == "inception-chat" {
		constructionName = "inception"
		cfg.Name = constructionName
	}
	provider, err := llm.NewProvider(constructionName, cfg)
	if err != nil {
		return nil, err
	}
	if providerName == "inception" {
		decisionProvider, ok := provider.(llm.DecisionProvider)
		if !ok {
			return nil, xerr.Internal("ai.decide: Inception provider lacks its native decision interface")
		}
		return decisionProvider, nil
	}
	switch providerName {
	case "openai", "xai", "deepseek", "gemini", "inception-chat":
		return llm.NewJSONDecisionProvider(provider), nil
	case "anthropic", "claude":
		anthropic, ok := provider.(*llm.AnthropicProvider)
		if !ok {
			return nil, xerr.Internal("ai.decide: Anthropic provider does not implement structured outputs")
		}
		return llm.NewAnthropicDecisionProvider(anthropic), nil
	default:
		return nil, xerr.Validation("ai.decide: provider must expose native decisions, JSON Schema output, or verified JSON-mode chat output")
	}
}

func providerConfig(global GlobalOptions, providerName, model, baseURL, apiKeyEnv string) (llm.Config, error) {
	if baseURL != "" {
		parsed, err := url.Parse(baseURL)
		if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Hostname() == "" {
			return llm.Config{}, xerr.Validation("ai: base_url must be an absolute HTTP(S) URL")
		}
	}
	if global.Timeout < 0 {
		return llm.Config{}, xerr.Validation("ai: timeout must be positive")
	}
	apiKey := ""
	if apiKeyEnv != "" {
		apiKey = os.Getenv(apiKeyEnv)
	}
	return llm.Config{
		Name:         providerName,
		BaseURL:      baseURL,
		APIKey:       apiKey,
		DefaultModel: model,
		Timeout:      global.Timeout,
	}, nil
}
