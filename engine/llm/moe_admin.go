package llm

import (
	"context"
	"slices"

	"github.com/nexssp/kernel/action"
	"github.com/nexssp/transport/thttp"
)

// MoEAdminConfig wires the MoE admin surface to a RouterSink and a
// manifest reload callback supplied by the caller.
type MoEAdminConfig struct {
	Sink           *RouterSink
	ReloadFunc     func(context.Context) (int, error)
	KnownProviders []string
}

type MoEStatsRes struct {
	Executed uint64 `json:"executed"`
	Retries  uint64 `json:"retries"`
	Errors   uint64 `json:"errors"`
}

type MoEReloadRes struct {
	Experts int `json:"experts_reloaded"`
}

type MoEModelsReq struct {
	Provider string `json:"provider" validate:"required"`
	APIKey   string `json:"api_key,omitempty"`
	BaseURL  string `json:"base_url,omitempty"`
}

type MoEModelsRes struct {
	Provider string   `json:"provider"`
	Models   []string `json:"models"`
}

func BuildMoEAdminActions(cfg MoEAdminConfig) []action.AnyAction {
	stats := action.New("moe.stats.get", func(_ context.Context, _ struct{}) (MoEStatsRes, error) {
		if cfg.Sink == nil {
			return MoEStatsRes{}, nil
		}

		s := cfg.Sink.Stats()

		return MoEStatsRes(s), nil
	}).
		Description("Snapshot of MoE routing counters").
		Tag("admin", "moe").
		System().
		Route(thttp.GET("/api/v1/moe/stats")).
		Build()

	reload := action.New("moe.experts.reload", func(ctx context.Context, _ struct{}) (MoEReloadRes, error) {
		if cfg.ReloadFunc == nil {
			return MoEReloadRes{}, nil
		}

		n, err := cfg.ReloadFunc(ctx)
		if err != nil {
			return MoEReloadRes{}, err
		}

		return MoEReloadRes{Experts: n}, nil
	}).
		Description("Re-reads the experts manifest without restart").
		Tag("admin", "moe").
		System().
		Route(thttp.POST("/api/v1/moe/experts/reload")).
		Build()

	models := action.New("moe.models.list", func(ctx context.Context, req MoEModelsReq) (MoEModelsRes, error) {
		if len(cfg.KnownProviders) > 0 && !slices.Contains(cfg.KnownProviders, req.Provider) {
			return MoEModelsRes{}, nil
		}
		list, err := FetchModels(ctx, req.Provider, req.APIKey, req.BaseURL)
		if err != nil {
			return MoEModelsRes{}, err
		}

		return MoEModelsRes{Provider: req.Provider, Models: list}, nil
	}).
		Description("Lists models available on the given provider").
		Tag("admin", "moe").
		System().
		Route(thttp.GET("/api/v1/moe/models")).
		Build()

	return []action.AnyAction{stats, reload, models}
}
