// Schema and loader for the _experts.yml manifest.
//
// The ExpertsManifest struct owns the `experts:` block. Consumers that need
// additional top-level keys (jumalu's `graphs:`) embed this struct and add
// their own fields; yaml.v3 inlines anonymous embedded structs, so a single
// YAML file is decoded into the merged schema without a custom unmarshaler.
package llm

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/nexssp/cost"
	"github.com/nexssp/kernel/xerr"
	"gopkg.in/yaml.v3"
)

// ExpertProviderConfig is the per-provider entry under an expert.
//
// TimeoutMS accepts both the current `timeout_ms:` YAML key and the
// historical `timeoutms:` key. The legacy form exists because an earlier
// loader carried the field with no yaml tag, making the literal lowercased
// Go field name the effective YAML key. Both forms are decoded; the
// canonical form wins if both are present.
type ExpertProviderConfig struct {
	Provider              string        `json:"provider" yaml:"provider"`
	Model                 string        `json:"model" yaml:"model"`
	RateCardID            string        `json:"rate_card_id,omitempty" yaml:"rate_card_id,omitempty"`
	Currency              cost.Currency `json:"currency,omitempty" yaml:"currency,omitempty"`
	Temperature           float64       `json:"temperature,omitempty" yaml:"temperature,omitempty"`
	MaxTokens             int           `json:"max_tokens,omitempty" yaml:"max_tokens,omitempty"`
	TimeoutMS             int64         `json:"timeout_ms,omitempty" yaml:"timeout_ms,omitempty"`
	SystemPrefix          string        `json:"system_prefix,omitempty" yaml:"system_prefix,omitempty"`
	PromptPrice1k         float64       `json:"prompt_price_1k,omitempty" yaml:"prompt_price_1k,omitempty"`
	PromptCacheHitPrice1k float64       `json:"prompt_cache_hit_price_1k,omitempty" yaml:"prompt_cache_hit_price_1k,omitempty"`
	CompletionPrice1k     float64       `json:"completion_price_1k,omitempty" yaml:"completion_price_1k,omitempty"`
}

// UnmarshalYAML decodes both `timeout_ms` and the legacy `timeoutms` key.
// Everything else uses the default yaml.v3 behavior via a shadow struct.
func (c *ExpertProviderConfig) UnmarshalYAML(value *yaml.Node) error {
	type rawConfig struct {
		Provider              string        `yaml:"provider"`
		Model                 string        `yaml:"model"`
		RateCardID            string        `yaml:"rate_card_id,omitempty"`
		Currency              cost.Currency `yaml:"currency,omitempty"`
		Temperature           float64       `yaml:"temperature,omitempty"`
		MaxTokens             int           `yaml:"max_tokens,omitempty"`
		TimeoutMS             int64         `yaml:"timeout_ms,omitempty"`
		TimeoutMSLegacy       int64         `yaml:"timeoutms,omitempty"`
		SystemPrefix          string        `yaml:"system_prefix,omitempty"`
		PromptPrice1k         float64       `yaml:"prompt_price_1k,omitempty"`
		PromptCacheHitPrice1k float64       `yaml:"prompt_cache_hit_price_1k,omitempty"`
		CompletionPrice1k     float64       `yaml:"completion_price_1k,omitempty"`
	}

	var raw rawConfig
	if err := value.Decode(&raw); err != nil {
		return err
	}

	*c = ExpertProviderConfig{
		Provider:              raw.Provider,
		Model:                 raw.Model,
		RateCardID:            raw.RateCardID,
		Currency:              raw.Currency,
		Temperature:           raw.Temperature,
		MaxTokens:             raw.MaxTokens,
		TimeoutMS:             raw.TimeoutMS,
		SystemPrefix:          raw.SystemPrefix,
		PromptPrice1k:         raw.PromptPrice1k,
		PromptCacheHitPrice1k: raw.PromptCacheHitPrice1k,
		CompletionPrice1k:     raw.CompletionPrice1k,
	}

	// Legacy fallback: only used when the canonical key is absent.
	if c.TimeoutMS == 0 && raw.TimeoutMSLegacy != 0 {
		c.TimeoutMS = raw.TimeoutMSLegacy
	}

	return nil
}

// ExpertSpec is a single named expert entry.
type ExpertSpec struct {
	Key         string                 `json:"key" yaml:"key"`
	Name        string                 `json:"name" yaml:"name"`
	Description string                 `json:"description" yaml:"description"`
	Keywords    []string               `json:"keywords" yaml:"keywords"`
	Providers   []ExpertProviderConfig `json:"providers" yaml:"providers"`
}

// ExpertsManifest is the shared top-level shape of an _experts.yml.
//
// Consumers that need additional top-level keys should embed this struct
// anonymously; yaml.v3 will inline the embedded fields.
type ExpertsManifest struct {
	Experts map[string]ExpertSpec `json:"experts" yaml:"experts"`
}

func LoadExpertsManifest(data []byte) (*ExpertsManifest, error) {
	var manifest ExpertsManifest

	dec := yaml.NewDecoder(bytes.NewReader(data))
	dec.KnownFields(false)

	if err := dec.Decode(&manifest); err != nil {
		return nil, xerr.BadRequest("llm: failed to parse experts manifest", err)
	}

	return &manifest, nil
}

func LoadExpertsFile(filePath string) (*ExpertsManifest, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		message := fmt.Sprintf("llm: could not read experts file %q", filePath)
		if errors.Is(err, os.ErrNotExist) {
			return nil, xerr.NotFound(message, err)
		}
		return nil, xerr.Unavailable(message, err)
	}

	return LoadExpertsManifest(data)
}

// BuildMoEExperts constructs the named Provider map from a manifest.
//
// Side effect: registers a rate card into GlobalCatalog only when the expert
// explicitly supplies its rate_card_id, currency, and rates. Provider/model
// identities and rates are external configuration; this function does not
// infer a card identity, currency, or price.
//
// When an expert declares more than one provider, the returned provider is
// a ProviderChain composed in declaration order. The chain inherits the
// default fallback policy (xerr.IsTransient) and no observer.
//
// The graph compilation layer (jumalu) is expected to walk its own
// `graphs:` block after this call returns. This function does not compile
// or execute anything.
func BuildMoEExperts(manifest *ExpertsManifest) (map[string]Provider, error) {
	if manifest == nil || len(manifest.Experts) == 0 {
		return nil, xerr.BadRequest("llm: experts manifest is empty")
	}
	out := make(map[string]Provider, len(manifest.Experts))
	keys := make([]string, 0, len(manifest.Experts))
	for k := range manifest.Experts {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	for _, key := range keys {
		provider, built, err := buildMoEExpert(key, manifest.Experts[key])
		if err != nil {
			return nil, err
		}
		if built {
			out[key] = provider
		}
	}
	if len(out) == 0 {
		return nil, xerr.BadRequest("llm: no usable experts in manifest")
	}
	return out, nil
}

func buildMoEExpert(key string, spec ExpertSpec) (Provider, bool, error) {
	if len(spec.Providers) == 0 {
		return nil, false, nil
	}
	chain := make([]Provider, 0, len(spec.Providers))
	for _, config := range spec.Providers {
		provider, err := buildMoEExpertProvider(key, config)
		if err != nil {
			return nil, false, err
		}
		if provider != nil {
			chain = append(chain, provider)
		}
	}
	if len(chain) == 0 {
		return nil, false, nil
	}
	if len(chain) == 1 {
		return chain[0], true, nil
	}
	return NewProviderChain(chain[0], chain[1:]...), true, nil
}

func buildMoEExpertProvider(key string, config ExpertProviderConfig) (Provider, error) {
	if strings.TrimSpace(config.Model) == "" {
		return nil, xerr.Validation(fmt.Sprintf("llm: expert %q requires an explicit model identifier", key))
	}
	hasRates := config.PromptPrice1k != 0 || config.PromptCacheHitPrice1k != 0 || config.CompletionPrice1k != 0
	if hasRates || config.Currency != (cost.Currency{}) || strings.TrimSpace(config.RateCardID) != "" {
		if strings.TrimSpace(config.RateCardID) == "" {
			return nil, xerr.Validation(fmt.Sprintf("llm: expert %q rate card requires an explicit rate_card_id", key))
		}
		if config.Currency == (cost.Currency{}) {
			return nil, xerr.Validation(fmt.Sprintf("llm: expert %q rate card requires an explicit currency", key))
		}
		if err := GlobalCatalog.Register(config.Model, PriceCard{
			RateCardID: config.RateCardID,
			Currency:   config.Currency,
			Peak: Price{
				PromptPer1k:      config.PromptPrice1k,
				PromptCacheHit1k: config.PromptCacheHitPrice1k,
				CompPer1k:        config.CompletionPrice1k,
			},
		}); err != nil {
			return nil, withErrorContext(fmt.Sprintf("llm: expert %q rate card", key), err)
		}
	}
	timeout := time.Duration(config.TimeoutMS) * time.Millisecond
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	provider, err := NewProvider(config.Provider, Config{
		Name:         key,
		DefaultModel: config.Model,
		Timeout:      timeout,
	})
	if err != nil {
		return nil, withErrorContext(fmt.Sprintf("llm: expert %q provider %q", key, config.Provider), err)
	}
	return provider, nil
}

// BuildMoEFromManifest is the legacy entry point. It wraps the expert map
// into a MoEProvider for callers that want a single routing provider.
//
// New callers should prefer BuildMoEExperts and construct their own router
// (TieredModelRouter, ProviderChain, or a custom dispatcher).
func BuildMoEFromManifest(manifest *ExpertsManifest, defaultProvider Provider) (*MoEProvider, error) {
	experts, err := BuildMoEExperts(manifest)
	if err != nil {
		return nil, err
	}

	moe := NewMoE("experts_manifest_router", defaultProvider)
	for name, prov := range experts {
		moe.RegisterExpert(name, prov)
	}

	return moe, nil
}

// MatchExpertByKeywords returns the first expert whose keyword list contains
// a case-insensitive substring of query. Returns "general" when nothing
// matches or the manifest is empty.
func (m *ExpertsManifest) MatchExpertByKeywords(query string) string {
	if m == nil || len(m.Experts) == 0 {
		return "general"
	}

	clean := strings.ToLower(query)

	keys := make([]string, 0, len(m.Experts))
	for k := range m.Experts {
		keys = append(keys, k)
	}

	slices.Sort(keys)

	for _, key := range keys {
		spec := m.Experts[key]
		for _, kw := range spec.Keywords {
			if strings.Contains(clean, strings.ToLower(kw)) {
				return key
			}
		}
	}

	return "general"
}

// ResolveConfigPath applies the standard fail-fast resolution chain: flag
// wins, then env, then the first existing default. An explicit override
// that points to a missing file is a hard error.
func ResolveConfigPath(flag, envKey string, defaults ...string) (string, error) {
	if flag != "" {
		if err := statConfigPath(flag); err != nil {
			return "", xerr.NotFound("config file from flag not found: "+flag, err)
		}

		return flag, nil
	}

	if envKey != "" {
		if v := os.Getenv(envKey); v != "" {
			if err := statConfigPath(v); err != nil {
				return "", xerr.NotFound(fmt.Sprintf("%s=%s not found", envKey, v), err)
			}

			return v, nil
		}
	}

	for _, def := range defaults {
		if err := statConfigPath(def); err == nil {
			return def, nil
		} else if !errors.Is(err, os.ErrNotExist) {
			return "", xerr.Unavailable("llm: check default config path", err)
		}
	}

	return "", xerr.NotFound(fmt.Sprintf("no config file found (checked: %v)", defaults))
}

func statConfigPath(filePath string) (err error) {
	absolutePath, err := filepath.Abs(filePath)
	if err != nil {
		return xerr.Validation("llm: config path is invalid", err)
	}
	rootPath := filepath.VolumeName(absolutePath) + string(filepath.Separator)
	root, err := os.OpenRoot(rootPath)
	if err != nil {
		return xerr.Unavailable("llm: open config filesystem root", err)
	}
	defer func() {
		err = errors.Join(err, root.Close())
	}()

	relativePath, err := filepath.Rel(rootPath, absolutePath)
	if err != nil {
		return xerr.Validation("llm: config path is invalid", err)
	}
	if !filepath.IsLocal(relativePath) {
		return xerr.Validation("llm: config path escapes its filesystem root")
	}
	_, err = root.Stat(relativePath)
	return err
}
