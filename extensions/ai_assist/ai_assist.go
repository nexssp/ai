package ai_assist

import (
	"context"

	"github.com/nexssp/flow/compiler"
	"github.com/nexssp/kernel/action"
	"github.com/nexssp/kernel/xctx"
)

func init() {
	compiler.Register("ai_assist", Bundle)

	// Register a Kernel hook following the pattern used by Flow's hook extension.
	action.MustRegisterHook("ai.error_analyzer", func() action.AnyHook {
		return action.AnyHook{
			OnError: func(ctx context.Context, req any, err error, meta *action.Meta) {
				// Clone the context for asynchronous analysis so the request path does not wait for it.
				bgCtx := xctx.CloneForAsync(ctx)
				go func(_ context.Context) {
					// TODO: Send the error and request payload to OpenAI or Claude here.
					// Example: "Action <meta.Name> failed with <err>. Payload: <req>."
				}(bgCtx)
			},
		}
	})
}

func Bundle(_ map[string]string) compiler.Bundle {
	return compiler.Bundle{ID: "ai_assist"}
}
