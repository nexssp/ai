package llm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/nexssp/cost"
	"github.com/nexssp/kernel/xerr"
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
	RateCardID string        `json:"rate_card_id" yaml:"rate_card_id"`
	Currency   cost.Currency `json:"currency" yaml:"currency"`
	Peak       Price         `json:"peak"     yaml:"peak"`
	OffPeak    *Price        `json:"off_peak,omitempty" yaml:"off_peak,omitempty"`
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
	cards    map[string]map[string]PriceCard
	window   OffPeakWindow
	loadedAt time.Time
}

// GlobalCatalog is the default catalog used by CalculateCost and by
// every provider that builds a Response.
var GlobalCatalog = NewPriceCatalog()

// NewPriceCatalog returns an empty-by-default catalog. Applications load their
// provider/model rate cards, currencies, and billing windows from configuration.
func NewPriceCatalog() *PriceCatalog {
	c := &PriceCatalog{
		cards: make(map[string]map[string]PriceCard, 64),
	}
	if err := c.LoadFromYAML([]byte(DefaultPricingTemplate)); err != nil {
		panic(err)
	}

	return c
}

// Register installs or replaces a rate card for a model. Both the model and
// card identity are required; an incomplete card is never registered.
func (c *PriceCatalog) Register(model string, card PriceCard) error {
	if strings.TrimSpace(model) == "" {
		return xerr.Validation("price catalog: model identifier is required")
	}
	if strings.TrimSpace(card.RateCardID) == "" {
		return xerr.Validation("price catalog: rate_card_id is required for model " + model)
	}
	if card.Currency == (cost.Currency{}) {
		return xerr.Validation("price catalog: currency is required for rate card " + card.RateCardID)
	}

	c.mu.Lock()
	if c.cards[model] == nil {
		c.cards[model] = make(map[string]PriceCard)
	}
	c.cards[model][card.RateCardID] = card
	c.mu.Unlock()
	return nil
}

// Get returns a model's card only when exactly one card is configured for it.
// Ambiguous model-only estimates fail closed; use GetRateCard with an explicit ID.
func (c *PriceCatalog) Get(model string) (PriceCard, bool) {
	c.mu.RLock()
	byID := c.cards[model]
	c.mu.RUnlock()
	if len(byID) != 1 {
		return PriceCard{}, false
	}
	for _, card := range byID {
		return card, true
	}
	return PriceCard{}, false
}

// GetRateCard returns the exact card registered for the model and ID.
func (c *PriceCatalog) GetRateCard(model, rateCardID string) (PriceCard, bool) {
	c.mu.RLock()
	card, ok := c.cards[model][rateCardID]
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

// LoadFromYAML accepts an explicit models map with per-card or top-level
// currency. Every model entry must carry an explicit rate_card_id; cards
// without an explicit currency are rejected instead of assuming one.
func (c *PriceCatalog) LoadFromYAML(data []byte) error {
	var newFmt priceCatalogYAML
	if err := yaml.Unmarshal(data, &newFmt); err != nil {
		return xerr.BadRequest("price catalog: unmarshal yaml", err)
	}
	if newFmt.Models == nil {
		return xerr.Validation("price catalog: models map is required")
	}

	cards := make(map[string]map[string]PriceCard, len(newFmt.Models))
	for name, m := range newFmt.Models {
		if name == "" {
			return xerr.Validation("price catalog: model identifier is required")
		}
		card := PriceCard(m)
		if strings.TrimSpace(card.RateCardID) == "" {
			return xerr.Validation("price catalog: rate_card_id is required for model " + name)
		}
		if card.Currency == (cost.Currency{}) {
			card.Currency = newFmt.Currency
		}
		if card.Currency == (cost.Currency{}) {
			return xerr.Validation("price catalog: currency is required for model " + name)
		}
		cards[name] = map[string]PriceCard{card.RateCardID: card}
	}

	c.mu.Lock()
	for name, byID := range cards {
		if c.cards[name] == nil {
			c.cards[name] = make(map[string]PriceCard)
		}
		maps.Copy(c.cards[name], byID)
	}
	c.window = newFmt.Window
	c.loadedAt = time.Now()
	c.mu.Unlock()
	return nil
}

type cardYAML struct {
	RateCardID string        `yaml:"rate_card_id"`
	Currency   cost.Currency `yaml:"currency"`
	Peak       Price         `yaml:"peak"`
	OffPeak    *Price        `yaml:"off_peak,omitempty"`
}

type priceCatalogYAML struct {
	Currency cost.Currency       `yaml:"currency"`
	Window   OffPeakWindow       `yaml:"window"`
	Models   map[string]cardYAML `yaml:"models"`
}

func (c *PriceCatalog) LoadFromFile(filePath string) error {
	data, err := os.ReadFile(filePath)
	if err != nil {
		message := "price catalog: read " + filePath
		if errors.Is(err, os.ErrNotExist) {
			return xerr.NotFound(message, err)
		}
		return xerr.Unavailable(message, err)
	}

	return c.LoadFromYAML(data)
}

func (c *PriceCatalog) LoadFromURL(ctx context.Context, url string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, http.NoBody)
	if err != nil {
		return xerr.BadRequest("price catalog: create URL request", err)
	}

	client := &http.Client{Timeout: 10 * time.Second}

	resp, err := client.Do(req)
	if err != nil {
		return xerr.Unavailable("price catalog: fetch URL", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return xerr.Unavailable(fmt.Sprintf("price catalog: fetch URL returned status %d", resp.StatusCode))
	}

	data, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return xerr.Unavailable("price catalog: read URL body", err)
	}

	return c.LoadFromYAML(data)
}

// InitPricingFile writes the default template to disk.
func InitPricingFile(filePath string, force bool) (string, error) {
	if !force {
		if _, err := os.Stat(filePath); err == nil {
			return "", xerr.Conflict(fmt.Sprintf("pricing file %q already exists", filePath))
		} else if !errors.Is(err, os.ErrNotExist) {
			return "", xerr.Unavailable("price catalog: check existing pricing file", err)
		}
	}

	dir := filepath.Dir(filePath)
	if dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return "", xerr.Unavailable("price catalog: create pricing directory", err)
		}
	}

	if err := os.WriteFile(filePath, []byte(DefaultPricingTemplate), 0o600); err != nil {
		return "", xerr.Unavailable("price catalog: write default pricing file", err)
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
	if !found || card.Currency == (cost.Currency{}) {
		return 0, cost.Currency{}, false
	}
	return calculateCardCost(card, promptTokens, cachedTokens, compTokens, at)
}

// CalculateCostForRateCard estimates cost only when the exact configured
// rate-card identity and expected currency both match. A missing card,
// currency, or currency mismatch returns known=false and never a zero-cost
// success. Explicit-card estimates round up to the next micro-unit so a
// budget preflight cannot pass on a sub-micro rounding underestimate.
func CalculateCostForRateCard(
	model, rateCardID string,
	expectedCurrency cost.Currency,
	promptTokens, cachedTokens, compTokens int,
	at time.Time,
) (cost.Micro, cost.Currency, bool) {
	if strings.TrimSpace(model) == "" || strings.TrimSpace(rateCardID) == "" || expectedCurrency == (cost.Currency{}) {
		return 0, cost.Currency{}, false
	}
	card, found := GlobalCatalog.GetRateCard(model, rateCardID)
	if !found || card.Currency == (cost.Currency{}) || card.Currency != expectedCurrency {
		return 0, cost.Currency{}, false
	}
	return calculateCardCostConservative(card, promptTokens, cachedTokens, compTokens, at)
}

func calculateCardCost(card PriceCard, promptTokens, cachedTokens, compTokens int, at time.Time) (cost.Micro, cost.Currency, bool) {
	return calculateCardCostRounded(card, promptTokens, cachedTokens, compTokens, at, false)
}

func calculateCardCostConservative(card PriceCard, promptTokens, cachedTokens, compTokens int, at time.Time) (cost.Micro, cost.Currency, bool) {
	return calculateCardCostRounded(card, promptTokens, cachedTokens, compTokens, at, true)
}

func calculateCardCostRounded(card PriceCard, promptTokens, cachedTokens, compTokens int, at time.Time, roundUp bool) (cost.Micro, cost.Currency, bool) {
	if card.Currency == (cost.Currency{}) {
		return 0, cost.Currency{}, false
	}
	rate := card.Peak
	if card.OffPeak != nil && GlobalCatalog.IsOffPeak(at) {
		rate = *card.OffPeak
	}

	missTokens := max(0, promptTokens-cachedTokens)

	hitTokens := promptTokens - missTokens

	cacheHitRate := rate.PromptCacheHit1k
	if cacheHitRate == 0 {
		cacheHitRate = rate.PromptPer1k
	}

	promptAmount := (float64(missTokens)/1000.0)*rate.PromptPer1k +
		(float64(hitTokens)/1000.0)*cacheHitRate
	completionAmount := (float64(compTokens) / 1000.0) * rate.CompPer1k

	total := promptAmount + completionAmount
	if math.IsNaN(total) || math.IsInf(total, 0) || total < 0 {
		return 0, card.Currency, false
	}
	if roundUp {
		micros := total * 1_000_000
		if math.IsNaN(micros) || math.IsInf(micros, 0) || micros >= float64(math.MaxInt64) {
			return 0, card.Currency, false
		}
		return cost.Micro(math.Ceil(micros)), card.Currency, true
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

// DefaultPricingTemplate intentionally has no provider/model IDs, rates,
// currency, or billing window. Applications must load those from their
// operator-managed rate-card configuration.
const DefaultPricingTemplate = `# Add externally verified rate cards below.
# Set currency globally or on each model entry; no currency is assumed.
models: {}
`
