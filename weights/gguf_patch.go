package weights

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/nexssp/kernel/xfs"

	"github.com/nexssp/ai/weights/gguf"
	"github.com/nexssp/ai/weights/internal/filecommit"
)

func patchGGUF(ctx context.Context, checkpoint *openedCheckpoint, request PatchRequest) (result PatchResult, returnErr error) {
	outputPath, validateErr := validateGGUFOutput(checkpoint, request)
	if validateErr != nil {
		return PatchResult{}, validateErr
	}
	outputDir := filepath.Dir(outputPath)
	if outputDir != "." {
		if mkdirErr := checkpoint.root.MkdirAll(outputDir, 0o755); mkdirErr != nil {
			return PatchResult{}, fmt.Errorf("weights: create GGUF output directory: %w", mkdirErr)
		}
	}
	stagePath, stage, stageErr := createStageFile(checkpoint.root, outputDir)
	if stageErr != nil {
		return PatchResult{}, stageErr
	}
	defer func() {
		var cleanupErr error
		if stage != nil {
			cleanupErr = errors.Join(cleanupErr, stage.Close())
		}
		if removeErr := checkpoint.root.Remove(stagePath); removeErr != nil && !errors.Is(removeErr, os.ErrNotExist) {
			cleanupErr = errors.Join(cleanupErr, fmt.Errorf("weights: remove GGUF staging file %q: %w", stagePath, removeErr))
		}
		returnErr = errors.Join(returnErr, cleanupErr)
	}()

	inputDigest, inputSize, sourcePerm, sourceErr := stageGGUFSource(ctx, checkpoint, request, stage)
	if sourceErr != nil {
		return PatchResult{}, sourceErr
	}
	if closeErr := stage.Close(); closeErr != nil {
		return PatchResult{}, fmt.Errorf("weights: close staged GGUF file: %w", closeErr)
	}
	stage = nil

	applied, applyErr := applyGGUFEdits(ctx, checkpoint.root, stagePath, request.Edits)
	if applyErr != nil {
		return PatchResult{}, applyErr
	}
	outputDigest, finalizeErr := finalizeGGUFStage(ctx, checkpoint.root, stagePath, sourcePerm, inputDigest, applied)
	if finalizeErr != nil {
		return PatchResult{}, finalizeErr
	}
	if ctxErr := ctx.Err(); ctxErr != nil {
		return PatchResult{}, ctxErr
	}
	if publishErr := publishGGUFStage(checkpoint, stagePath, outputPath, request.Overwrite); publishErr != nil {
		return PatchResult{}, publishErr
	}
	return makeGGUFResult(request, outputPath, inputDigest, inputSize, outputDigest, applied), nil
}

func validateGGUFOutput(checkpoint *openedCheckpoint, request PatchRequest) (string, error) {
	if err := validateDigest(request.SourceSHA256); err != nil {
		return "", err
	}
	outputPath, err := xfs.Rel(request.OutputPath)
	if err != nil {
		return "", fmt.Errorf("weights: validate GGUF output path: %w", err)
	}
	if !strings.HasSuffix(strings.ToLower(outputPath), ".gguf") {
		return "", errors.New("weights: GGUF output_path must end in .gguf")
	}
	if info, err := checkpoint.root.Lstat(outputPath); err == nil {
		if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
			return "", errors.New("weights: GGUF output_path must be a regular file, not a symlink or directory")
		}
		if !request.Overwrite {
			return "", fmt.Errorf("weights: output %q already exists; set overwrite=true only when replacing this output is intended", outputPath)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", fmt.Errorf("weights: inspect GGUF output path: %w", err)
	}
	if err := rejectOutputAliases(checkpoint, outputPath); err != nil {
		return "", err
	}
	return outputPath, nil
}

func stageGGUFSource(ctx context.Context, checkpoint *openedCheckpoint, request PatchRequest, stage *os.File) (digest string, size int64, sourcePerm os.FileMode, returnErr error) {
	digest, size, sourcePerm, err := copyAndHash(ctx, checkpoint.root, request.Model.Path, stage)
	if err != nil {
		return "", 0, 0, fmt.Errorf("weights: stage GGUF source: %w", err)
	}
	expected := strings.ToLower(strings.TrimSpace(request.SourceSHA256))
	if digest != expected {
		return "", 0, 0, fmt.Errorf("weights: source GGUF SHA-256 changed: expected %s, copied %s", expected, digest)
	}
	if err := stage.Sync(); err != nil {
		return "", 0, 0, fmt.Errorf("weights: sync staged GGUF file: %w", err)
	}
	return digest, size, sourcePerm, nil
}

func applyGGUFEdits(ctx context.Context, root *xfs.Root, stagePath string, edits []Edit) ([]gguf.AppliedEdit, error) {
	editableFile, err := root.OpenFile(stagePath, os.O_RDWR, 0)
	if err != nil {
		return nil, fmt.Errorf("weights: reopen staged GGUF file: %w", err)
	}
	parsed, err := gguf.OpenFile(ctx, editableFile, stagePath)
	if err != nil {
		return nil, fmt.Errorf("weights: parse staged GGUF file: %w", err)
	}
	if parsed.SplitCount > 1 {
		_ = parsed.Close()
		return nil, fmt.Errorf("%w: GGUF split files are not supported by this adapter", ErrUnsupportedLayout)
	}
	ggufEdits := make([]gguf.Edit, len(edits))
	for i, edit := range edits {
		ggufEdits[i] = gguf.Edit{Tensor: edit.Tensor, Index: edit.Index, Delta: edit.Delta, ExpectedValue: edit.ExpectedValue}
	}
	applied, applyErr := parsed.ApplyEdits(ctx, ggufEdits)
	closeErr := parsed.Close()
	if applyErr != nil {
		return nil, fmt.Errorf("weights: apply GGUF edits: %w", applyErr)
	}
	if closeErr != nil {
		return nil, fmt.Errorf("weights: close edited GGUF staging file: %w", closeErr)
	}
	return applied, nil
}

func finalizeGGUFStage(ctx context.Context, root *xfs.Root, stagePath string, sourcePerm os.FileMode, inputDigest string, applied []gguf.AppliedEdit) (result FileDigest, returnErr error) {
	syncFile, openErr := root.OpenFile(stagePath, os.O_RDWR, 0)
	if openErr != nil {
		return FileDigest{}, fmt.Errorf("weights: reopen edited GGUF staging file: %w", openErr)
	}
	if chmodErr := syncFile.Chmod(sourcePerm); chmodErr != nil {
		_ = syncFile.Close()
		return FileDigest{}, fmt.Errorf("weights: restore GGUF output permissions: %w", chmodErr)
	}
	if syncErr := syncFile.Sync(); syncErr != nil {
		_ = syncFile.Close()
		return FileDigest{}, fmt.Errorf("weights: sync edited GGUF staging file: %w", syncErr)
	}
	if closeErr := syncFile.Close(); closeErr != nil {
		return FileDigest{}, fmt.Errorf("weights: close synced GGUF staging file: %w", closeErr)
	}
	if verifyErr := verifyGGUFEdits(ctx, root, stagePath, applied); verifyErr != nil {
		return FileDigest{}, fmt.Errorf("weights: verify edited GGUF staging file: %w", verifyErr)
	}
	digest, digestErr := digestPath(ctx, root, stagePath)
	if digestErr != nil {
		return FileDigest{}, fmt.Errorf("weights: hash edited GGUF staging file: %w", digestErr)
	}
	if digest.SHA256 == inputDigest {
		return FileDigest{}, errors.New("weights: GGUF patch produced no byte-level change")
	}
	return digest, nil
}

func publishGGUFStage(checkpoint *openedCheckpoint, stagePath, outputPath string, overwrite bool) error {
	if overwrite {
		if err := checkpoint.root.Rename(stagePath, outputPath); err != nil {
			return fmt.Errorf("weights: atomically replace GGUF output: %w", err)
		}
	} else if err := filecommit.LinkNoReplace(checkpoint.root, checkpoint.commitRoot, stagePath, outputPath); err != nil {
		return fmt.Errorf("weights: atomically publish GGUF output without replacement: %w", err)
	}
	return nil
}

func makeGGUFResult(request PatchRequest, outputPath, inputDigest string, inputSize int64, outputDigest FileDigest, edits []gguf.AppliedEdit) PatchResult {
	outputDigest.Path = outputPath
	applied := make([]AppliedEdit, len(edits))
	for i, edit := range edits {
		applied[i] = AppliedEdit{Tensor: edit.Tensor, Index: edit.Index, Before: edit.Before, After: edit.After}
	}
	return PatchResult{
		Model: request.Model, InputPath: request.Model.Path, OutputPath: outputPath,
		InputSHA256: inputDigest, OutputSHA256: outputDigest.SHA256,
		InputFiles:  []FileDigest{{Path: request.Model.Path, SizeBytes: inputSize, SHA256: inputDigest}},
		OutputFiles: []FileDigest{outputDigest}, Edits: applied,
	}
}

func verifyGGUFEdits(ctx context.Context, root *xfs.Root, path string, edits []gguf.AppliedEdit) error {
	file, err := root.Open(path)
	if err != nil {
		return err
	}
	parsed, err := gguf.OpenFile(ctx, file, path)
	if err != nil {
		return err
	}
	defer parsed.Close()
	for _, edit := range edits {
		if err := ctx.Err(); err != nil {
			return err
		}
		value, err := parsed.ReadFloat(edit.Tensor, edit.Index)
		if err != nil {
			return err
		}
		if value != edit.After {
			return fmt.Errorf("tensor %s[%d] stores %.9g after write; expected %.9g", edit.Tensor, edit.Index, value, edit.After)
		}
	}
	return nil
}
