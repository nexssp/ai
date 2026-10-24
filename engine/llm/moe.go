package llm

import (
	"context"
	"fmt"
	"iter"

	"github.com/nexssp/kernel/xerr"
)

type MoEProvider struct {
	name        string
	experts     map[string]Provider
	defaultProv Provider
}

func NewMoE(name string, defaultProvider Provider) *MoEProvider {
	return &MoEProvider{
		name:        name,
		experts:     make(map[string]Provider),
		defaultProv: defaultProvider,
	}
}

func (m *MoEProvider) RegisterExpert(expertName string, p Provider) *MoEProvider {
	m.experts[expertName] = p

	return m
}

func (m *MoEProvider) Name() string { return m.name }

func (m *MoEProvider) Complete(ctx context.Context, req Request) (Response, error) {
	if prov, ok := m.experts[req.Model]; ok {
		return prov.Complete(ctx, req)
	}

	res, err := m.defaultProv.Complete(ctx, req)
	if err == nil {
		return res, nil
	}

	// Automatic fallback to available experts if default provider fails
	for name, expert := range m.experts {
		if res, expErr := expert.Complete(ctx, req); expErr == nil {
			res.Model = name + " (expert fallback)"

			return res, nil
		}
	}

	return Response{}, xerr.Unavailable("moe: all expert providers failed", err)
}

func (m *MoEProvider) Stream(ctx context.Context, req Request) (iter.Seq2[string, error], error) {
	if prov, ok := m.experts[req.Model]; ok {
		return prov.Stream(ctx, req)
	}

	return m.defaultProv.Stream(ctx, req)
}

// ── ModelManager Implementation ───────────────────────────────────────────────

func (m *MoEProvider) Preload(ctx context.Context, _ string) error {
	for name, expert := range m.experts {
		if err := Preload(ctx, expert, ""); err != nil {
			return fmt.Errorf("moe: expert %q preload failed: %w", name, err)
		}
	}

	return Preload(ctx, m.defaultProv, "")
}

func (m *MoEProvider) Unload(ctx context.Context, _ string) error {
	var firstErr error
	for name, expert := range m.experts {
		if err := Unload(ctx, expert, ""); err != nil && firstErr == nil {
			firstErr = fmt.Errorf("moe: expert %q unload failed: %w", name, err)
		}
	}

	if err := Unload(ctx, m.defaultProv, ""); err != nil && firstErr == nil {
		firstErr = err
	}

	return firstErr
}

func (m *MoEProvider) IsLoaded(ctx context.Context, _ string) bool {
	for _, expert := range m.experts {
		if !IsLoaded(ctx, expert, "") {
			return false
		}
	}

	return IsLoaded(ctx, m.defaultProv, "")
}
