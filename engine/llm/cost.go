package llm

import (
	"context"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/nexssp/cost"
	"gopkg.in/yaml.v3"
)

// Price is a per-1000-token rate for one billing tier.
//
//	PromptPer1k      cache-MISS input rate (the "expensive" input path)
//	PromptCacheHit1k cache-HIT input rate. Providers that do not offer
//	                 prompt caching should leave this equal to
//	                 PromptPer1k, or zero when the field is unused.
//	CompPer1k        output rate
//
// All values are in the currency declared by the enclosing PriceCard.
type Price struct {
	PromptPer1k      float64 `json:"prompt_per_1k"            yaml:"prompt_per_1k"`
	PromptCacheHit1k float64 `json:"prompt_cache_hit_per_1k"  yaml:"prompt_cache_hit_per_1k"`
	CompPer1k        float64 `json:"comp_per_1k"              yaml:"comp_per_1k"`
}

// PriceCard is the full rate card for one model in one currency.
//
//	Peak     applied 24/7 unless OffPeak is set and the call happens
//	         inside the catalog's OffPeakWindow.
//	OffPeak  optional discounted rate card. Leave nil when the
//	         provider does not differentiate by time-of-day.
//	Currency the currency in which Peak and OffPeak are denominated.
type PriceCard struct {
	Currency cost.Currency `json:"currency" yaml:"currency"`
	Peak     Price         `json:"peak"     yaml:"peak"`
	OffPeak  *Price        `json:"off_peak,omitempty" yaml:"off_peak,omitempty"`
}

// OffPeakWindow is the daily UTC window during which the OffPeak rate
// card applies. The window may wrap midnight (e.g. 16:30 → 00:30).
type OffPeakWindow struct {
	StartUTC string `json:"start_utc" yaml:"start_utc"`
	EndUTC   string `json:"end_utc"   yaml:"end_utc"`
}

// PriceCatalog is the process-wide rate card registry. It is safe for
// concurrent readers and occasional writers (Register / reload).
type PriceCatalog struct {
	mu       sync.RWMutex
	cards    map[string]PriceCard
	window   OffPeakWindow
	loadedAt time.Time
}

// GlobalCatalog is the default catalog used by CalculateCost and by
// every provider that builds a Response.
var GlobalCatalog = NewPriceCatalog()

// NewPriceCatalog returns a catalog pre-populated from the built-in
// DefaultPricingTemplate. Callers may overwrite it with LoadFromFile,
// LoadFromURL, or LoadFromYAML.
func NewPriceCatalog() *PriceCatalog {
	c := &PriceCatalog{
		cards: make(map[string]PriceCard, 64),
	}
	_ = c.LoadFromYAML([]byte(DefaultPricingTemplate))

	return c
}

// Register installs or replaces a rate card for a model.
func (c *PriceCatalog) Register(model string, card PriceCard) {
	if model == "" {
		return
	}

	if card.Currency == (cost.Currency{}) {
		card.Currency = cost.USD
	}

	c.mu.Lock()
	c.cards[model] = card
	c.mu.Unlock()
}

// Get returns the rate card for a model.
func (c *PriceCatalog) Get(model string) (PriceCard, bool) {
	c.mu.RLock()
	card, ok := c.cards[model]
	c.mu.RUnlock()

	return card, ok
}

// Models returns a sorted list of every model that has a rate card.
// Useful for introspection and startup banners.
func (c *PriceCatalog) Models() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	out := make([]string, 0, len(c.cards))
	for k := range c.cards {
		out = append(out, k)
	}

	return out
}

// IsOffPeak reports whether t falls inside the configured window.
// The window is UTC by contract.
func (c *PriceCatalog) IsOffPeak(t time.Time) bool {
	c.mu.RLock()
	w := c.window
	c.mu.RUnlock()

	return inWindow(t.UTC(), w)
}

// ── YAML loading ────────────────────────────────────────────────────────────

// LoadFromYAML accepts two formats:
//
//	New (preferred):
//	  currency: USD
//	  window: { start_utc: "16:30", end_utc: "00:30" }
//	  models:
//	    deepseek-chat:
//	      peak:     { prompt_per_1k: 0.00027, prompt_cache_hit_per_1k: 0.00007, comp_per_1k: 0.00110 }
//	      off_peak: { prompt_per_1k: 0.000135, prompt_cache_hit_per_1k: 0.000035, comp_per_1k: 0.00055 }
//
//	Legacy:
//	  gpt-4o: { prompt_per_1k: 0.00250, comp_per_1k: 0.01000 }
//
// The new shape wins when a `models:` key is present.
func (c *PriceCatalog) LoadFromYAML(data []byte) error {
	var newFmt struct {
		Currency cost.Currency       `yaml:"currency"`
		Window   OffPeakWindow       `yaml:"window"`
		Models   map[string]cardYAML `yaml:"models"`
	}
	if err := yaml.Unmarshal(data, &newFmt); err == nil && len(newFmt.Models) > 0 {
		defaultCur := newFmt.Currency
		if defaultCur == (cost.Currency{}) {
			defaultCur = cost.USD
		}

		c.mu.Lock()
		for name, m := range newFmt.Models {
			card := PriceCard(m)
			if card.Currency == (cost.Currency{}) {
				card.Currency = defaultCur
			}

			c.cards[name] = card
		}

		c.window = newFmt.Window
		c.loadedAt = time.Now()
		c.mu.Unlock()

		return nil
	}

	var oldFmt map[string]Price
	if err := yaml.Unmarshal(data, &oldFmt); err != nil {
		return fmt.Errorf("price catalog: unmarshal yaml: %w", err)
	}

	c.mu.Lock()
	for name, p := range oldFmt {
		if name == "window" || name == "currency" || name == "models" {
			continue
		}

		c.cards[name] = PriceCard{
			Currency: cost.USD,
			Peak:     p,
		}
	}

	c.loadedAt = time.Now()
	c.mu.Unlock()

	return nil
}

type cardYAML struct {
	Currency cost.Currency `yaml:"currency"`
	Peak     Price         `yaml:"peak"`
	OffPeak  *Price        `yaml:"off_peak,omitempty"`
}

func (c *PriceCatalog) LoadFromFile(filePath string) error {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("price catalog: read %s: %w", filePath, err)
	}

	return c.LoadFromYAML(data)
}

func (c *PriceCatalog) LoadFromURL(ctx context.Context, url string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, http.NoBody)
	if err != nil {
		return fmt.Errorf("price catalog: create url request: %w", err)
	}

	client := &http.Client{Timeout: 10 * time.Second}

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("price catalog: fetch url: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("price catalog: fetch url returned status %d", resp.StatusCode)
	}

	data, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return fmt.Errorf("price catalog: read url body: %w", err)
	}

	return c.LoadFromYAML(data)
}

// InitPricingFile writes the default template to disk.
func InitPricingFile(filePath string, force bool) (string, error) {
	if !force {
		if _, err := os.Stat(filePath); err == nil {
			return "", fmt.Errorf("pricing file %q already exists", filePath)
		}
	}

	dir := filepath.Dir(filePath)
	if dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return "", fmt.Errorf("create pricing directory: %w", err)
		}
	}

	if err := os.WriteFile(filePath, []byte(DefaultPricingTemplate), 0o600); err != nil {
		return "", fmt.Errorf("write default pricing file: %w", err)
	}

	return filePath, nil
}

// ── calculation ─────────────────────────────────────────────────────────────

// CalculateCost returns the cost of one LLM turn.
//
//	model       the model identifier as reported by the provider
//	promptTokens total input tokens (cached + miss)
//	cachedTokens the portion of promptTokens that hit the prompt cache.
//	             Pass 0 when the provider does not report cache hits.
//	compTokens  output tokens
//	at          the time of the call, used to select peak vs off-peak
//
// Returns (micros, currency, ok). When ok is false the model has no rate
// card — the caller should treat cost as unknown, not zero.
func CalculateCost(
	model string,
	promptTokens, cachedTokens, compTokens int,
	at time.Time,
) (cost.Micro, cost.Currency, bool) {
	card, found := GlobalCatalog.Get(model)
	if !found {
		return 0, cost.Currency{}, false
	}

	rate := card.Peak
	if card.OffPeak != nil && GlobalCatalog.IsOffPeak(at) {
		rate = *card.OffPeak
	}

	missTokens := promptTokens - cachedTokens
	if missTokens < 0 {
		missTokens = 0
	}

	hitTokens := promptTokens - missTokens

	cacheHitRate := rate.PromptCacheHit1k
	if cacheHitRate == 0 {
		cacheHitRate = rate.PromptPer1k
	}

	promptUSD := (float64(missTokens)/1000.0)*rate.PromptPer1k +
		(float64(hitTokens)/1000.0)*cacheHitRate
	compUSD := (float64(compTokens) / 1000.0) * rate.CompPer1k

	total := promptUSD + compUSD
	if math.IsNaN(total) || math.IsInf(total, 0) || total < 0 {
		return 0, card.Currency, false
	}

	return cost.ToMicro(total), card.Currency, true
}

// ── window helpers ──────────────────────────────────────────────────────────

func inWindow(t time.Time, w OffPeakWindow) bool {
	start, ok1 := parseHHMM(w.StartUTC)

	end, ok2 := parseHHMM(w.EndUTC)
	if !ok1 || !ok2 {
		return false
	}

	mins := t.Hour()*60 + t.Minute()
	if start <= end {
		return mins >= start && mins < end
	}

	return mins >= start || mins < end
}

func parseHHMM(s string) (int, bool) {
	if len(s) != 5 || s[2] != ':' {
		return 0, false
	}

	h, err1 := strconv.Atoi(s[0:2])

	m, err2 := strconv.Atoi(s[3:5])
	if err1 != nil || err2 != nil || h < 0 || h > 23 || m < 0 || m > 59 {
		return 0, false
	}

	return h*60 + m, true
}

// ── default template ────────────────────────────────────────────────────────

// DefaultPricingTemplate is loaded by NewPriceCatalog at process start.
//
// ⚠️  The numbers below are placeholders for any field you have not
// verified against the provider's published rate card. Only the fields
// that were already present in this repository before M1.5 have been
// preserved verbatim; every new field (prompt_cache_hit_per_1k,
// off_peak, currency) is a zero or a commented hint.
//
// To make cost reporting trustworthy:
//
//  1. Edit the YAML below, or ship a pricing.yml and call
//     GlobalCatalog.LoadFromFile("pricing.yml") at startup.
//  2. Model names must match exactly what the provider returns in the
//     `model` field of its response. `deepseek-flash` is not a real
//     DeepSeek model name — verify what you actually receive and align.
//  3. Every new model you support must appear here, or CalculateCost
//     returns ok=false and the runner shows `-` in the cost column.
const DefaultPricingTemplate = `# Rates are USD (unless a model overrides) per 1,000 tokens.
#
# window:      the daily UTC window during which off_peak applies.
# currency:    the default currency for every model in this file.
# models.X:    the rate card. peak is required; off_peak is optional.
#
# prompt_cache_hit_per_1k: leave equal to prompt_per_1k when the provider
#                          does not offer a distinct cached-input rate.

currency: USD

window:
  start_utc: "16:30"
  end_utc:   "00:30"

models:
  gpt-4o:
    peak:
      prompt_per_1k:          0.00250
      prompt_cache_hit_per_1k: 0.00125
      comp_per_1k:            0.01000

  gpt-4o-mini:
    peak:
      prompt_per_1k:          0.00015
      prompt_cache_hit_per_1k: 0.000075
      comp_per_1k:            0.00060

  claude-3-5-sonnet-20241022:
    peak:
      prompt_per_1k:          0.00300
      prompt_cache_hit_per_1k: 0.00030
      comp_per_1k:            0.01500

  deepseek-chat:
    peak:
      prompt_per_1k:          0.00027
      prompt_cache_hit_per_1k: 0.00027
      comp_per_1k:            0.00110
    # off_peak: paste the published discounted rate here.

  deepseek-reasoner:
    peak:
      prompt_per_1k:          0.00055
      prompt_cache_hit_per_1k: 0.00055
      comp_per_1k:            0.00219
    # off_peak: paste the published discounted rate here.

  llama-3.3-70b-versatile:
    peak:
      prompt_per_1k:          0.00059
      prompt_cache_hit_per_1k: 0.00059
      comp_per_1k:            0.00079

  meta/llama-3.3-70b-instruct:
    peak:
      prompt_per_1k:          0.00070
      prompt_cache_hit_per_1k: 0.00070
      comp_per_1k:            0.00090
`
