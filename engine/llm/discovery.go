package llm

import (
	"context"
	"net/http"
	"os"
	"strings"
	"time"

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

// FetchModels queries an upstream provider for its active model catalog.
func FetchModels(ctx context.Context, providerType, apiKey, baseURL string) ([]string, error) {
	switch strings.ToLower(providerType) {
	case "ollama", "ollama_cloud":
		return fetchOllamaModels(ctx, apiKey, baseURL)
	case "openai", "groq", "nvidia", "nim", "deepseek", "openrouter", "gemini", "lmstudio", "local", "vllm", "vast":
		return fetchOpenAIModels(ctx, providerType, apiKey, baseURL)
	default:
		return getFallbackModels(providerType), nil
	}
}

func fetchOllamaModels(ctx context.Context, apiKey, baseURL string) ([]string, error) {
	url := resolveOllamaURL(baseURL)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, http.NoBody)
	if err != nil {
		return getFallbackModels("ollama"), err
	}

	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	resp, err := discoveryClient.Do(req)
	if err != nil || resp.StatusCode != http.StatusOK {
		if resp != nil && resp.Body != nil {
			_ = resp.Body.Close()
		}

		return getFallbackModels("ollama"), nil
	}
	defer func() { _ = resp.Body.Close() }()

	var res struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}

	if err := codec.Default.NewDecoder(resp.Body).Decode(&res); err == nil && len(res.Models) > 0 {
		list := make([]string, len(res.Models))
		for i, m := range res.Models {
			list[i] = m.Name
		}

		return list, nil
	}

	return getFallbackModels("ollama"), nil
}

func fetchOpenAIModels(ctx context.Context, pType, apiKey, baseURL string) ([]string, error) {
	url := resolveProviderURL(pType, baseURL)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, http.NoBody)
	if err != nil {
		return getFallbackModels(pType), err
	}

	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	resp, err := discoveryClient.Do(req)
	if err != nil || resp.StatusCode != http.StatusOK {
		if resp != nil && resp.Body != nil {
			_ = resp.Body.Close()
		}

		return getFallbackModels(pType), nil
	}
	defer func() { _ = resp.Body.Close() }()

	var res struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}

	if err := codec.Default.NewDecoder(resp.Body).Decode(&res); err == nil && len(res.Data) > 0 {
		list := make([]string, len(res.Data))
		for i, m := range res.Data {
			list[i] = m.ID
		}

		return list, nil
	}

	return getFallbackModels(pType), nil
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
	case "nvidia", "nim":
		return "https://integrate.api.nvidia.com/v1/models"
	case "deepseek":
		return "https://api.deepseek.com/models"
	case "openrouter":
		return "https://openrouter.ai/api/v1/models"
	case "lmstudio", "local":
		return "http://127.0.0.1:1234/v1/models"
	case "vllm", "vast":
		return "http://localhost:8000/v1/models"
	default:
		return "https://api.openai.com/v1/models"
	}
}

func getFallbackModels(providerType string) []string {
	switch strings.ToLower(providerType) {
	case "groq":
		return []string{"llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"}
	case "nvidia", "nim":
		return []string{"meta/llama-3.3-70b-instruct", "nvidia/llama-3.1-nemotron-70b-instruct", "deepseek-ai/deepseek-r1"}
	case "ollama":
		return []string{"qwen2.5-coder:7b", "mistral:latest", "llama3.2:3b"}
	case "deepseek":
		return []string{"deepseek-chat", "deepseek-reasoner"}
	case "openrouter":
		return []string{"meta-llama/llama-3.3-70b-instruct:free", "qwen/qwen-2.5-coder-32b-instruct"}
	case "gemini":
		return []string{"gemini-2.0-flash", "gemini-1.5-pro"}
	case "lmstudio", "local", "vllm", "vast":
		return []string{"google/gemma-2-9b", "qwen2.5-coder:7b"}
	default:
		return []string{"gpt-4o-mini", "gpt-4o"}
	}
}
