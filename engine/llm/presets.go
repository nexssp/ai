package llm

import (
	"fmt"
	"os"
	"strings"

	"github.com/nexssp/kernel/xerr"
)

type ProviderPreset struct {
	Name         string
	DefaultURL   string
	DefaultModel string
	EnvKeyNames  []string
	IsDeepSeek   bool
}

var Presets = map[string]ProviderPreset{
	"openai": {
		Name:         "openai",
		DefaultURL:   "https://api.openai.com/v1",
		DefaultModel: "gpt-4o-mini",
		EnvKeyNames:  []string{"OPENAI_API_KEY"},
	},
	"groq": {
		Name:         "groq",
		DefaultURL:   "https://api.groq.com/openai/v1",
		DefaultModel: "llama-3.1-8b-instant",
		EnvKeyNames:  []string{"GROQ_API_KEY"},
	},
	"nvidia": {
		Name:         "nvidia",
		DefaultURL:   "https://integrate.api.nvidia.com/v1",
		DefaultModel: "meta/llama-3.3-70b-instruct",
		EnvKeyNames:  []string{"NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY", "NVCF_API_KEY"},
	},
	"nim": {
		Name:         "nim",
		DefaultURL:   "http://localhost:8000/v1",
		DefaultModel: "meta/llama-3.3-70b-instruct",
		EnvKeyNames:  []string{"NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY"},
	},
	"deepseek": {
		Name:         "deepseek",
		DefaultURL:   "https://api.deepseek.com/v1",
		DefaultModel: "deepseek-flash",
		EnvKeyNames:  []string{"DEEPSEEK_API_KEY"},
		IsDeepSeek:   true,
	},
	"openrouter": {
		Name:         "openrouter",
		DefaultURL:   "https://openrouter.ai/api/v1",
		DefaultModel: "meta-llama/llama-3.3-70b-instruct:free",
		EnvKeyNames:  []string{"OPENROUTER_API_KEY", "OPENROUTER_KEY"},
	},
	"lmstudio": {
		Name:         "lmstudio",
		DefaultURL:   "http://127.0.0.1:1234/v1",
		DefaultModel: "qwen2.5-coder:7b",
		EnvKeyNames:  []string{"LMSTUDIO_API_KEY"},
	},
	"vllm": {
		Name:         "vllm",
		DefaultURL:   "http://localhost:8000/v1",
		DefaultModel: "default",
		EnvKeyNames:  []string{"VLLM_API_KEY"},
	},
}

// NewProvider resolves and constructs a configured LLM Provider.
func NewProvider(providerType string, cfg Config) (Provider, error) {
	pType := strings.ToLower(strings.TrimSpace(providerType))

	if preset, ok := Presets[pType]; ok {
		if cfg.Name == "" {
			cfg.Name = preset.Name
		}

		if cfg.BaseURL == "" {
			cfg.BaseURL = preset.DefaultURL
		}

		if cfg.DefaultModel == "" {
			cfg.DefaultModel = preset.DefaultModel
		}

		if cfg.APIKey == "" {
			for _, key := range preset.EnvKeyNames {
				if val := os.Getenv(key); val != "" {
					cfg.APIKey = val

					break
				}
			}
		}

		if preset.IsDeepSeek {
			return NewDeepSeek(cfg), nil
		}

		return NewOpenAI(cfg), nil
	}

	switch pType {
	case "anthropic", "claude":
		if cfg.APIKey == "" {
			cfg.APIKey = os.Getenv("ANTHROPIC_API_KEY")
		}

		return NewAnthropic(cfg), nil
	case "gemini", "google":
		if cfg.APIKey == "" {
			cfg.APIKey = os.Getenv("GEMINI_API_KEY")
		}

		return NewGemini(cfg), nil
	case "ollama", "ollama_cloud":
		return NewOllama(cfg), nil
	default:
		return nil, xerr.BadRequest(fmt.Sprintf("unknown LLM provider: %q", providerType))
	}
}
