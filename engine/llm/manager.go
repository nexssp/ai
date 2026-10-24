package llm

import "context"

// Preload guarantees the model is loaded in memory/VRAM and ready for traffic.
func Preload(ctx context.Context, p Provider, model string) error {
	if mgr, ok := p.(ModelManager); ok {
		return mgr.Preload(ctx, model)
	}

	return nil
}

// Unload releases model resources from VRAM if supported.
func Unload(ctx context.Context, p Provider, model string) error {
	if mgr, ok := p.(ModelManager); ok {
		return mgr.Unload(ctx, model)
	}

	return nil
}

// IsLoaded checks whether the model is actively resident in VRAM.
func IsLoaded(ctx context.Context, p Provider, model string) bool {
	if mgr, ok := p.(ModelManager); ok {
		return mgr.IsLoaded(ctx, model)
	}

	return true
}
