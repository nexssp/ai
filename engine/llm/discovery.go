package llm

import (
	"context"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/nexssp/kernel/xerr"
	"github.com/nexssp/transport/codec"
)

var discoveryClient = &http.Client{
	Timeout: 5 * time.Second,
	Transport: &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 20,
		IdleConnTimeout:     90 * time.Second,
	},
}

type namedModelCatalogWire struct {
	Models []modelNameWire `json:"models"`
}

type modelNameWire struct {
	Name string `json:"name"`
}

type openAIModelCatalogWire struct {
	Data []modelIDWire `json:"data"`
}

type modelIDWire struct {
	ID string `json:"id"`
}

// FetchModels queries an upstream provider for its active model catalog. It
// returns no fabricated model IDs when the provider catalog is unavailable.
func FetchModels(ctx context.Context, providerType, apiKey, baseURL string) ([]string, error) {
	switch strings.ToLower(providerType) {
	case providerOpenAICompatible:
		if strings.TrimSpace(baseURL) == "" {
			return nil, xerr.Validation("llm: openai-compatible model discovery requires a base URL")
		}
		return fetchOpenAIModels(ctx, providerType, apiKey, baseURL)
	case providerOllama, "ollama_cloud":
		return fetchOllamaModels(ctx, apiKey, baseURL)
	case "typesafe":
		return fetchTypeSafeModels(ctx, apiKey, baseURL)
	case providerOpenAI, "groq", providerNVIDIA, "nim", providerDeepSeek, providerOpenRouter, providerGemini, providerLMStudio, "local", "vllm", "vast", providerXAI, providerInception:
		return fetchOpenAIModels(ctx, providerType, apiKey, baseURL)
	default:
		return nil, xerr.BadRequest("llm: model discovery does not support provider " + providerType)
	}
}

func fetchTypeSafeModels(ctx context.Context, apiKey, baseURL string) ([]string, error) {
	url := typeSafeModelsURL(baseURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, http.NoBody)
	if err != nil {
		return nil, xerr.Internal("llm: build TypeSafe model discovery request", err)
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	resp, err := discoveryClient.Do(req)
	if err != nil {
		return nil, xerr.MapTransportError(err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return nil, FromHTTPStatus(resp.StatusCode, nil, "typesafe")
	}
	var catalog namedModelCatalogWire
	if err := codec.Default.NewDecoder(resp.Body).Decode(&catalog); err != nil {
		return nil, xerr.Unavailable("llm: invalid TypeSafe model catalog", err)
	}
	models := make([]string, 0, len(catalog.Models))
	for _, model := range catalog.Models {
		if model.Name != "" {
			models = append(models, model.Name)
		}
	}
	return models, nil
}

func fetchOllamaModels(ctx context.Context, apiKey, baseURL string) ([]string, error) {
	url := resolveOllamaURL(baseURL)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, http.NoBody)
	if err != nil {
		return nil, xerr.Internal("llm: build Ollama model discovery request", err)
	}

	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	resp, err := discoveryClient.Do(req)
	if err != nil {
		return nil, xerr.MapTransportError(err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return nil, FromHTTPStatus(resp.StatusCode, nil, providerOllama)
	}

	var res namedModelCatalogWire
	if err := codec.Default.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, xerr.Unavailable("llm: invalid Ollama model catalog", err)
	}

	list := make([]string, 0, len(res.Models))
	for _, model := range res.Models {
		if model.Name != "" {
			list = append(list, model.Name)
		}
	}
	return list, nil
}

func fetchOpenAIModels(ctx context.Context, pType, apiKey, baseURL string) ([]string, error) {
	url := resolveProviderURL(pType, baseURL)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, http.NoBody)
	if err != nil {
		return nil, xerr.Internal("llm: build model discovery request", err)
	}

	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	resp, err := discoveryClient.Do(req)
	if err != nil {
		return nil, xerr.MapTransportError(err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return nil, FromHTTPStatus(resp.StatusCode, nil, pType)
	}

	var res openAIModelCatalogWire
	if err := codec.Default.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, xerr.Unavailable("llm: invalid model catalog", err)
	}

	list := make([]string, 0, len(res.Data))
	for _, model := range res.Data {
		if model.ID != "" {
			list = append(list, model.ID)
		}
	}
	return list, nil
}

func resolveOllamaURL(baseURL string) string {
	if baseURL != "" {
		if strings.HasSuffix(baseURL, "/v1/chat/completions") {
			return strings.Replace(baseURL, "/v1/chat/completions", "/api/tags", 1)
		}

		return strings.TrimRight(baseURL, "/") + "/api/tags"
	}

	if host := os.Getenv("OLLAMA_BASE_URL"); host != "" {
		return strings.TrimRight(host, "/") + "/api/tags"
	}

	if host := os.Getenv("OLLAMA_HOST"); host != "" {
		return strings.TrimRight(host, "/") + "/api/tags"
	}

	return "http://localhost:11434/api/tags"
}

func resolveProviderURL(providerType, baseURL string) string {
	if baseURL != "" {
		if strings.HasSuffix(baseURL, "/chat/completions") {
			return strings.Replace(baseURL, "/chat/completions", "/models", 1)
		}

		return strings.TrimRight(baseURL, "/") + "/models"
	}

	switch strings.ToLower(providerType) {
	case "groq":
		return "https://api.groq.com/openai/v1/models"
	case providerNVIDIA, "nim":
		return "https://integrate.api.nvidia.com/v1/models"
	case providerDeepSeek:
		return "https://api.deepseek.com/models"
	case providerOpenRouter:
		return "https://openrouter.ai/api/v1/models"
	case providerLMStudio, "local":
		return "http://127.0.0.1:1234/v1/models"
	case "vllm", "vast":
		return "http://localhost:8000/v1/models"
	case providerXAI:
		return "https://api.x.ai/v1/models"
	case providerInception:
		return "https://api.inceptionlabs.ai/v1/models"
	default:
		return "https://api.openai.com/v1/models"
	}
}
