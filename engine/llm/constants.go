package llm

const (
	providerAnthropic  = "anthropic"
	providerGemini     = "gemini"
	providerOllama     = "ollama"
	providerOpenAI     = "openai"
	providerNVIDIA     = "nvidia"
	providerOpenRouter = "openrouter"
	providerLMStudio   = "lmstudio"
	providerDeepSeek   = "deepseek"
	roleUser           = "user"
	roleModel          = "model"
	defaultOllamaModel = "qwen2.5-coder:7b"
	jsonKeyModel       = "model"
	jsonKeyMessages    = "messages"
	jsonKeyKeepAlive   = "keep_alive"
	keepAlive24h       = "24h"
	thinkOpenTag       = "<think>"
	thinkCloseTag      = "</think>"
)
