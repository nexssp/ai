package llm

import (
	"context"

	"github.com/nexssp/kernel/xerr"
)

// TieredModelRouter dynamically escalates queries to higher-tier models if smaller models fail.
type TieredModelRouter struct {
	FastSmall Provider // Tier 1: Local / Groq (0$ or sub-cent)
	Standard  Provider // Tier 2: DeepSeek-V3 / GPT-4o-mini
	HeavyDeep Provider // Tier 3: Claude 3.5 Sonnet / o1 (High reasoning)
}

func NewTieredModelRouter(fast, standard, heavy Provider) *TieredModelRouter {
	return &TieredModelRouter{
		FastSmall: fast,
		Standard:  standard,
		HeavyDeep: heavy,
	}
}

// CompleteAdaptive routes queries through increasing capability tiers.
func (r *TieredModelRouter) CompleteAdaptive(ctx context.Context, req Request, minTier int) (Response, int, error) {
	// Tier 1: Fast / Edge / Local
	if minTier <= 1 && r.FastSmall != nil {
		res, err := r.FastSmall.Complete(ctx, req)
		if err == nil && len(res.Content) > 0 {
			return res, 1, nil
		}
	}

	// Tier 2: Standard Workhorse
	if minTier <= 2 && r.Standard != nil {
		res, err := r.Standard.Complete(ctx, req)
		if err == nil && len(res.Content) > 0 {
			return res, 2, nil
		}
	}

	// Tier 3: Heavy Reasoning Model
	if r.HeavyDeep != nil {
		res, err := r.HeavyDeep.Complete(ctx, req)
		if err == nil {
			return res, 3, nil
		}

		return Response{}, 3, err
	}

	return Response{}, 0, xerr.Unavailable("all LLM tiers exhausted", nil)
}
