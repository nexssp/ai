package weights

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/nexssp/ai/weights/safetensors"
)

// Inspect validates the caller-selected adapter, reads tensor metadata, and
// optionally returns a bounded sample and SHA-256 provenance. Bundle hashes cover
// every indexed shard as well as the index JSON. Paths are confined to the
// process working directory by Nexss xfs.
func Inspect(ctx context.Context, request InspectRequest) (Inspection, error) {
	if ctx == nil {
		return Inspection{}, errors.New("weights: context is required")
	}
	if contextErr := ctx.Err(); contextErr != nil {
		return Inspection{}, contextErr
	}
	if request.SampleElements > MaxInspectSamples {
		return Inspection{}, fmt.Errorf("weights: sample_elements must be between 0 and %d", MaxInspectSamples)
	}
	checkpoint, err := openCheckpoint(ctx, request.Model)
	if err != nil {
		return Inspection{}, err
	}
	defer checkpoint.Close()
	if contextErr := ctx.Err(); contextErr != nil {
		return Inspection{}, contextErr
	}
	result := Inspection{
		Model:   request.Model,
		Tensors: make([]TensorInfo, 0, len(checkpoint.tensors)),
	}
	for _, name := range sortedTensorNames(checkpoint.tensors) {
		if contextErr := ctx.Err(); contextErr != nil {
			return Inspection{}, contextErr
		}
		info := checkpoint.tensors[name]
		summary := TensorInfo{
			Name: name, DType: info.DType, Shape: slices.Clone(info.Shape),
			Elements: info.Elements, Bytes: info.Bytes,
		}
		if request.SampleElements > 0 && info.Elements > 0 && editableDType(checkpoint.format, info.DType) {
			summary.Sample, err = checkpoint.readFloatPrefix(name, request.SampleElements)
			if err != nil {
				return Inspection{}, fmt.Errorf("weights: sample tensor %q: %w", name, err)
			}
		}
		result.Tensors = append(result.Tensors, summary)
	}
	for _, size := range checkpoint.fileSizes {
		if size > math.MaxInt64-result.SizeBytes {
			return Inspection{}, errors.New("weights: total checkpoint size overflows int64")
		}
		result.SizeBytes += size
	}
	if request.IncludeSHA256 {
		result.Files, result.SHA256, result.SizeBytes, err = checkpoint.provenance(ctx)
		if err != nil {
			return Inspection{}, fmt.Errorf("weights: hash checkpoint provenance: %w", err)
		}
	}
	return result, nil
}

// Patch validates the selected format adapter and applies at most MaxPatchEdits
// scalar deltas to a checksum-verified copy. The source is never overwritten;
// replacing an existing distinct output is opt-in.
func Patch(ctx context.Context, request PatchRequest) (PatchResult, error) {
	if ctx == nil {
		return PatchResult{}, errors.New("weights: context is required")
	}
	if contextErr := ctx.Err(); contextErr != nil {
		return PatchResult{}, contextErr
	}
	if strings.TrimSpace(request.OutputPath) == "" {
		return PatchResult{}, errors.New("weights: output_path is required")
	}
	if len(request.Edits) == 0 {
		return PatchResult{}, errors.New("weights: at least one edit is required")
	}
	if len(request.Edits) > MaxPatchEdits {
		return PatchResult{}, fmt.Errorf("weights: edit count %d exceeds limit %d", len(request.Edits), MaxPatchEdits)
	}
	checkpoint, err := openCheckpoint(ctx, request.Model)
	if err != nil {
		return PatchResult{}, err
	}
	defer checkpoint.Close()
	if contextErr := ctx.Err(); contextErr != nil {
		return PatchResult{}, contextErr
	}
	if checkpoint.format == FormatSafeTensors && !checkpoint.sharded && !strings.HasSuffix(strings.ToLower(request.OutputPath), ".safetensors") {
		return PatchResult{}, errors.New("weights: SafeTensors output_path must end in .safetensors")
	}
	if checkpoint.sharded {
		return patchShardedSafeTensors(ctx, checkpoint, request)
	}
	switch checkpoint.format {
	case FormatSafeTensors:
		result, err := safetensors.PatchWithinRoot(ctx, checkpoint.root, checkpoint.commitRoot, request.Model.Path, request.OutputPath, request.Edits, safetensors.PatchOptions{
			ExpectedSHA256: request.SourceSHA256,
			Overwrite:      request.Overwrite,
		})
		if err != nil {
			return PatchResult{}, fmt.Errorf("weights: patch SafeTensors checkpoint: %w", err)
		}
		inputInfo, err := checkpoint.root.Stat(request.Model.Path)
		if err != nil {
			return PatchResult{}, fmt.Errorf("weights: stat input after patch: %w", err)
		}
		outputInfo, err := checkpoint.root.Stat(request.OutputPath)
		if err != nil {
			return PatchResult{}, fmt.Errorf("weights: stat output after patch: %w", err)
		}
		return PatchResult{
			Model: request.Model, InputPath: result.InputPath, OutputPath: result.OutputPath,
			InputSHA256: result.InputSHA256, OutputSHA256: result.OutputSHA256,
			InputFiles:  []FileDigest{{Path: request.Model.Path, SizeBytes: inputInfo.Size(), SHA256: result.InputSHA256}},
			OutputFiles: []FileDigest{{Path: request.OutputPath, SizeBytes: outputInfo.Size(), SHA256: result.OutputSHA256}},
			Edits:       result.Edits,
		}, nil
	case FormatGGUF:
		return patchGGUF(ctx, checkpoint, request)
	default:
		return PatchResult{}, fmt.Errorf("%w: patching is not implemented for format %q", ErrUnsupportedFormat, checkpoint.format)
	}
}

func sortedTensorNames(tensors map[string]TensorInfo) []string {
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	slices.Sort(names)
	return names
}
