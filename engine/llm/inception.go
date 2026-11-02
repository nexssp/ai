package llm

import (
	"context"
	"iter"

	"github.com/nexssp/kernel/xerr"
)

// InceptionProvider supports OpenAI-compatible chat completions and exposes
// the native decision interface. The documented /decisions API has no
// provider-enforced output-token cap, so Decide refuses dispatch rather than
// silently ignoring DecisionRequest.MaxOutputTokens. Use the cap-aware
// inception-chat adapter for structured JSON decisions instead.
type InceptionProvider struct {
	cfg  Config
	chat *OpenAIProvider
}

func NewInception(cfg Config) *InceptionProvider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.inceptionlabs.ai/v1"
	}
	if cfg.Name == "" {
		cfg.Name = providerInception
	}
	return &InceptionProvider{cfg: cfg, chat: NewOpenAI(cfg)}
}

func (p *InceptionProvider) Name() string { return p.cfg.Name }

func (p *InceptionProvider) Complete(ctx context.Context, req Request) (Response, error) {
	return p.chat.Complete(ctx, req)
}

func (p *InceptionProvider) Stream(ctx context.Context, req Request) (iter.Seq2[string, error], error) {
	return p.chat.Stream(ctx, req)
}

func (p *InceptionProvider) Decide(_ context.Context, req DecisionRequest) (DecisionResponse, error) {
	if _, _, err := validateDecisionRequest(req, p.cfg.DefaultModel, p.Name()); err != nil {
		return DecisionResponse{}, err
	}
	return DecisionResponse{}, xerr.Validation("inception: /v1/decisions does not document a provider-enforced max_output_tokens limit; refusing an uncapped decision request")
}

var (
	_ Provider         = (*InceptionProvider)(nil)
	_ DecisionProvider = (*InceptionProvider)(nil)
)
