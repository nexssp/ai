package nexssflow

import (
	"context"
	"errors"
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/nexssp/flow/compiler"
	"github.com/nexssp/kernel/action"
	"github.com/nexssp/kernel/xerr"

	"github.com/nexssp/ai/engine/llm"
)

const namedLLMMetadataKey = "ai.named_llms"

var namedLLMDirective = compiler.Directive{
	Name:        "llm",
	Example:     `@llm critic { provider: "groq", model: "your-model", api_key_env: "GROQ_CRITIC_API_KEY" }`,
	Handler:     handleNamedLLMDirective,
	OpaqueBlock: true,
}

var namedLLMOptions = []string{"provider", "model", "api_key_env", "base_url", "timeout"}

type namedLLMDeclaration struct {
	Name      string
	Provider  string
	Model     string
	APIKeyEnv string
	BaseURL   string
	Timeout   time.Duration
}

func handleNamedLLMDirective(_ context.Context, req compiler.DirectiveReq) (compiler.DirectiveRes, error) {
	line := strings.TrimSpace(req.Lines[req.I])
	rest := strings.TrimSpace(strings.TrimPrefix(line, "@llm"))
	name, _, hasOpen := strings.Cut(rest, "{")
	pos := compiler.Position{File: req.File, Line: req.I + 1}
	if !hasOpen {
		return compiler.DirectiveRes{}, compiler.SourceError(pos, "@llm: expected `@llm NAME { provider: ..., model: ... }`")
	}
	name = strings.TrimSpace(name)
	if !validNamedLLMName(name) {
		return compiler.DirectiveRes{}, compiler.SourceError(pos, "@llm: invalid name %q", name)
	}
	options, next, err := readNamedLLMOptions(req.Lines, req.I)
	if err != nil {
		return compiler.DirectiveRes{}, compiler.SourceError(pos, "@llm %s: %v", name, err)
	}
	declaration, err := parseNamedLLMDeclaration(name, options)
	if err != nil {
		return compiler.DirectiveRes{}, compiler.SourceError(pos, "@llm %s: %v", name, err)
	}
	if err := req.Declare("llm", name); err != nil {
		return compiler.DirectiveRes{}, err
	}
	var declarations []namedLLMDeclaration
	if raw, exists := req.Out[namedLLMMetadataKey]; exists {
		var ok bool
		declarations, ok = raw.([]namedLLMDeclaration)
		if !ok {
			return compiler.DirectiveRes{}, compiler.SourceError(pos, "@llm: invalid declaration metadata type %T", raw)
		}
	}
	req.Out[namedLLMMetadataKey] = append(declarations, declaration)
	return compiler.DirectiveRes{Next: next}, nil
}

func readNamedLLMOptions(lines []string, start int) (options map[string]string, next int, err error) {
	entries, next, err := compiler.ReadBlock(lines, start)
	if err != nil {
		return nil, start, err
	}
	options = make(map[string]string, len(entries))
	for _, entry := range entries {
		key, value, ok := strings.Cut(entry, ":")
		if !ok || strings.TrimSpace(key) == "" {
			return nil, start, fmt.Errorf("expected `key: value`, got %q", entry)
		}
		key = strings.TrimSpace(key)
		if _, exists := options[key]; exists {
			return nil, start, fmt.Errorf("duplicate option %q", key)
		}
		options[key] = strings.TrimSpace(value)
	}
	return options, next, nil
}

func parseNamedLLMDeclaration(name string, raw map[string]string) (namedLLMDeclaration, error) {
	keys := make([]string, 0, len(raw))
	for key := range raw {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	for _, key := range keys {
		if !slices.Contains(namedLLMOptions, key) {
			return namedLLMDeclaration{}, fmt.Errorf("unknown option %q", key)
		}
	}

	provider, err := namedLLMString(raw, "provider")
	if err != nil {
		return namedLLMDeclaration{}, err
	}
	provider = strings.ToLower(strings.TrimSpace(provider))
	if provider == "" {
		return namedLLMDeclaration{}, errors.New("provider is required")
	}
	if !llm.IsKnownProvider(provider) {
		return namedLLMDeclaration{}, fmt.Errorf("unknown provider %q", provider)
	}
	model, err := namedLLMString(raw, "model")
	if err != nil {
		return namedLLMDeclaration{}, err
	}
	if strings.TrimSpace(model) == "" {
		return namedLLMDeclaration{}, errors.New("model is required")
	}
	apiKeyEnv, err := namedLLMString(raw, "api_key_env")
	if err != nil {
		return namedLLMDeclaration{}, err
	}
	baseURL, err := namedLLMString(raw, "base_url")
	if err != nil {
		return namedLLMDeclaration{}, err
	}
	timeout := time.Duration(0)
	if rawTimeout, exists := raw["timeout"]; exists {
		timeoutText, parseErr := namedLLMString(raw, "timeout")
		if parseErr != nil {
			return namedLLMDeclaration{}, parseErr
		}
		timeout, err = time.ParseDuration(timeoutText)
		if err != nil {
			return namedLLMDeclaration{}, fmt.Errorf("invalid timeout %q: %w", timeoutText, err)
		}
		if timeout <= 0 {
			return namedLLMDeclaration{}, fmt.Errorf("timeout must be positive (got %q)", rawTimeout)
		}
	}
	return namedLLMDeclaration{
		Name:      name,
		Provider:  provider,
		Model:     model,
		APIKeyEnv: apiKeyEnv,
		BaseURL:   baseURL,
		Timeout:   timeout,
	}, nil
}

func namedLLMString(options map[string]string, key string) (string, error) {
	raw, exists := options[key]
	if !exists {
		return "", nil
	}
	value := strings.TrimSpace(raw)
	if value == "" {
		return "", fmt.Errorf("%s must not be empty", key)
	}
	if value[0] == '"' || value[0] == '`' {
		unquoted, err := strconv.Unquote(value)
		if err != nil {
			return "", fmt.Errorf("invalid %s string: %w", key, err)
		}
		return unquoted, nil
	}
	if value[0] == '\'' {
		if len(value) < 2 || value[len(value)-1] != '\'' {
			return "", fmt.Errorf("invalid %s string %q", key, value)
		}
		return value[1 : len(value)-1], nil
	}
	return value, nil
}

func validNamedLLMName(name string) bool {
	if name == "" {
		return false
	}
	for i, r := range name {
		valid := r == '_' || r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z'
		if i > 0 {
			valid = valid || r >= '0' && r <= '9' || r == '-'
		}
		if !valid {
			return false
		}
	}
	return true
}

func namedLLMMaterializer(global GlobalOptions, builder providerBuilder) compiler.Materializer {
	return func(req compiler.MaterializeReq) error {
		if req.Meta == nil {
			return nil
		}
		raw, exists := req.Meta[namedLLMMetadataKey]
		if !exists {
			return nil
		}
		declarations, ok := raw.([]namedLLMDeclaration)
		if !ok {
			return xerr.Internal(fmt.Sprintf("ai: invalid named LLM metadata type %T", raw))
		}
		for _, declaration := range declarations {
			namedGlobal := global
			namedGlobal.Provider = declaration.Provider
			namedGlobal.DefaultModel = declaration.Model
			if declaration.APIKeyEnv != "" {
				namedGlobal.APIKeyEnv = declaration.APIKeyEnv
			}
			if declaration.BaseURL != "" {
				namedGlobal.BaseURL = declaration.BaseURL
			}
			if declaration.Timeout > 0 {
				namedGlobal.Timeout = declaration.Timeout
			}
			namedBuilder := builder
			if declaration.APIKeyEnv != "" {
				namedBuilder = func(options GlobalOptions, prompt LLMDecl) (llm.Provider, error) {
					if value, exists := os.LookupEnv(declaration.APIKeyEnv); !exists || strings.TrimSpace(value) == "" {
						return nil, xerr.Validation(fmt.Sprintf("ai: named LLM %q requires a non-empty %s environment variable", declaration.Name, declaration.APIKeyEnv))
					}
					if builder == nil {
						return nil, xerr.Internal("ai: provider builder is unavailable")
					}
					return builder(options, prompt)
				}
			}
			act := newPromptActionWith(namedGlobal, namedBuilder).ToBuilder().
				Name("llm."+declaration.Name).
				Description("Run one prompt with the named LLM configuration "+declaration.Name).
				Tag("ai", "llm", declaration.Name).
				Build()
			owner := compiler.DeclarationOwner(req.Meta, "llm", declaration.Name, "llm."+declaration.Name)
			if err := req.Resolver.Mount(action.Library{
				Name:    owner,
				Actions: []action.AnyAction{act},
			}); err != nil {
				return xerr.Internal(fmt.Sprintf("ai: mount named LLM %q", declaration.Name), err)
			}
		}
		return nil
	}
}
