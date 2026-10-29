package llm

import (
	"context"
	"iter"
	"sync/atomic"
	"time"

	"github.com/nexssp/kernel/xerr"
)

type ProviderChain struct {
	providers   []Provider
	activeIdx   atomic.Int64
	cooldowns   []atomic.Int64
	failCounts  []atomic.Uint32
	switchCount atomic.Uint64
}

func NewProviderChain(primary Provider, fallbacks ...Provider) *ProviderChain {
	var all []Provider
	if primary != nil {
		all = append(all, primary)
	}

	for _, fb := range fallbacks {
		if fb != nil {
			all = append(all, fb)
		}
	}

	return &ProviderChain{
		providers:  all,
		cooldowns:  make([]atomic.Int64, len(all)),
		failCounts: make([]atomic.Uint32, len(all)),
	}
}

func (c *ProviderChain) Name() string {
	total := len(c.providers)
	if total == 0 {
		return "empty_chain"
	}

	idx := int(c.activeIdx.Load() % int64(total))

	return c.providers[idx].Name()
}

func (c *ProviderChain) isTransient(err error) bool {
	return xerr.IsTransient(err)
}

func (c *ProviderChain) Complete(ctx context.Context, req Request) (Response, error) {
	total := len(c.providers)
	if total == 0 {
		return Response{}, xerr.Unavailable("no providers configured in chain")
	}

	startIdx := int(c.activeIdx.Load()) % total
	now := time.Now().Unix()

	var lastErr error

	for i := range total {
		curr := (startIdx + i) % total
		if c.cooldowns[curr].Load() > now && i < total-1 {
			continue
		}

		resp, err := c.providers[curr].Complete(ctx, req)
		if err == nil {
			c.failCounts[curr].Store(0)

			if int64(curr) != c.activeIdx.Load() && i == 0 {
				c.activeIdx.Store(int64(curr))
			}

			return resp, nil
		}

		lastErr = err
		if !c.isTransient(err) {
			return resp, err
		}

		fails := c.failCounts[curr].Add(1)
		c.cooldowns[curr].Store(now + int64(1<<min(fails, 6)))
		c.activeIdx.Store(int64((curr + 1) % total))
		c.switchCount.Add(1)
	}

	return Response{}, xerr.Unavailable("all providers exhausted", lastErr)
}

func (c *ProviderChain) Stream(ctx context.Context, req Request) (iter.Seq2[string, error], error) {
	total := len(c.providers)
	if total == 0 {
		return nil, xerr.Unavailable("no providers configured in chain")
	}

	idx := int(c.activeIdx.Load() % int64(total))

	seq, err := c.providers[idx].Stream(ctx, req)
	if err != nil && c.isTransient(err) && total > 1 {
		next := (idx + 1) % total
		c.activeIdx.Store(int64(next))

		return c.providers[next].Stream(ctx, req)
	}

	return seq, err
}
