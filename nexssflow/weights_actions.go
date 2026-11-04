package nexssflow

import (
	"context"

	"github.com/nexssp/kernel/action"
	"github.com/nexssp/kernel/xerr"

	"github.com/nexssp/ai/weights"
)

func newWeightActions() []action.AnyAction {
	return []action.AnyAction{
		action.New[struct{}, []weights.AdapterInfo]("ai.weights.adapters", func(context.Context, struct{}) ([]weights.AdapterInfo, error) {
			return weights.Adapters(), nil
		}).Description("List implemented SafeTensors and GGUF checkpoint-editing adapters; PyTorch pickle checkpoints are not supported; format adapters do not imply model-family compatibility or evaluation").Tag("ai", "weights", "adapters").Build(),
		action.New[weights.InspectRequest, weights.Inspection]("ai.weights.inspect", func(ctx context.Context, request weights.InspectRequest) (weights.Inspection, error) {
			result, err := weights.Inspect(ctx, request)
			if err != nil {
				return weights.Inspection{}, xerr.From(err)
			}
			return result, nil
		}).Description("Inspect one SafeTensors file, an indexed sharded SafeTensors bundle, or one supported GGUF file using the explicitly selected format and adapter; PyTorch pickle checkpoints are rejected; optional SHA-256 reads every bundle member").Tag("ai", "weights", "inspect").Build(),
		action.New[weights.PatchRequest, weights.PatchResult]("ai.weights.patch", func(ctx context.Context, request weights.PatchRequest) (weights.PatchResult, error) {
			result, err := weights.Patch(ctx, request)
			if err != nil {
				return weights.PatchResult{}, xerr.From(err)
			}
			return result, nil
		}).Description("Apply bounded scalar edits to a checksum-verified SafeTensors or GGUF copy; sharded SafeTensors outputs publish the index last; never overwrites a source checkpoint").Tag("ai", "weights", "patch").Build(),
	}
}
