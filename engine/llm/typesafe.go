package llm

import (
	"context"
	"strings"

	"github.com/nexssp/kernel/xerr"
)

// TypeSafeProvider represents TypeSafe's native SystemOne decision endpoint.
// The documented API has no provider-enforced output-token cap, so Decide
// refuses dispatch rather than silently ignoring DecisionRequest.MaxOutputTokens.
type TypeSafeProvider struct {
	cfg Config
}

func NewTypeSafe(cfg Config) *TypeSafeProvider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.typesafe.ai"
	}
	if cfg.Name == "" {
		cfg.Name = "typesafe"
	}
	return &TypeSafeProvider{cfg: cfg}
}

func (p *TypeSafeProvider) Name() string { return p.cfg.Name }

func (p *TypeSafeProvider) Decide(_ context.Context, req DecisionRequest) (DecisionResponse, error) {
	if _, _, err := validateDecisionRequest(req, p.cfg.DefaultModel, p.Name()); err != nil {
		return DecisionResponse{}, err
	}
	return DecisionResponse{}, xerr.Validation("typesafe: System One does not document a provider-enforced max_output_tokens limit; refusing an uncapped decision request")
}

var _ DecisionProvider = (*TypeSafeProvider)(nil)

func typeSafeModelsURL(baseURL string) string {
	baseURL = strings.TrimRight(baseURL, "/")
	if strings.HasSuffix(baseURL, "/v1/systemone") {
		return strings.TrimSuffix(baseURL, "/systemone") + "/models"
	}
	if strings.HasSuffix(baseURL, "/v1") {
		return baseURL + "/models"
	}
	return baseURL + "/v1/models"
}
