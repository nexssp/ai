package weights

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/nexssp/kernel/xfs"

	"github.com/nexssp/ai/weights/internal/filecommit"
	"github.com/nexssp/ai/weights/safetensors"
)

type stagedShard struct {
	sourcePath string
	stagePath  string
	finalPath  string
	digest     string
	size       int64
	edits      []Edit
	published  bool
}

//nolint:gocyclo // Keep stage, rollback, and publish state visible together to preserve transaction cleanup ordering.
func patchShardedSafeTensors(ctx context.Context, checkpoint *openedCheckpoint, request PatchRequest) (result PatchResult, returnErr error) {
	if err := validateDigest(request.SourceSHA256); err != nil {
		return PatchResult{}, err
	}
	outputIndex, pathErr := xfs.Rel(request.OutputPath)
	if pathErr != nil {
		return PatchResult{}, fmt.Errorf("weights: validate output index path: %w", pathErr)
	}
	if !strings.HasSuffix(strings.ToLower(outputIndex), ".safetensors.index.json") {
		return PatchResult{}, errors.New("weights: a sharded SafeTensors output_path must end in .safetensors.index.json")
	}
	for _, sourcePath := range checkpoint.sourcePaths {
		if filepath.Clean(sourcePath) == filepath.Clean(outputIndex) {
			return PatchResult{}, errors.New("weights: output_path must differ from every input file; source checkpoints are never overwritten")
		}
	}
	outputDir := filepath.Dir(outputIndex)
	if outputDir != "." {
		if mkdirErr := checkpoint.root.MkdirAll(outputDir, 0o755); mkdirErr != nil {
			return PatchResult{}, fmt.Errorf("weights: create sharded output directory: %w", mkdirErr)
		}
	}
	if info, statErr := checkpoint.root.Lstat(outputIndex); statErr == nil {
		if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
			return PatchResult{}, errors.New("weights: output index must be a regular file, not a symlink or directory")
		}
		if !request.Overwrite {
			return PatchResult{}, fmt.Errorf("weights: output %q already exists; set overwrite=true only when replacing this output is intended", request.OutputPath)
		}
	} else if !errors.Is(statErr, os.ErrNotExist) {
		return PatchResult{}, fmt.Errorf("weights: inspect output index: %w", statErr)
	}
	if aliasErr := rejectOutputAliases(checkpoint, outputIndex); aliasErr != nil {
		return PatchResult{}, aliasErr
	}

	groupedEdits := make(map[string][]Edit)
	for _, edit := range request.Edits {
		info, exists := checkpoint.tensors[edit.Tensor]
		if !exists {
			return PatchResult{}, fmt.Errorf("weights: tensor %q does not exist", edit.Tensor)
		}
		if !editableDType(FormatSafeTensors, info.DType) {
			return PatchResult{}, fmt.Errorf("weights: tensor %q uses %s; editing supports F16, BF16, F32, and F64", edit.Tensor, info.DType)
		}
		if edit.Index >= info.Elements {
			return PatchResult{}, fmt.Errorf("weights: index %d is outside tensor %q with %d elements", edit.Index, edit.Tensor, info.Elements)
		}
		groupedEdits[checkpoint.tensorPaths[edit.Tensor]] = append(groupedEdits[checkpoint.tensorPaths[edit.Tensor]], edit)
	}

	shardPaths := make([]string, 0, len(checkpoint.sourcePaths)-1)
	for _, path := range checkpoint.sourcePaths {
		if path != checkpoint.indexPath {
			shardPaths = append(shardPaths, path)
		}
	}
	if len(shardPaths) == 0 {
		return PatchResult{}, errors.New("weights: sharded index has no tensor shards")
	}
	nonce, nonceErr := randomSuffix()
	if nonceErr != nil {
		return PatchResult{}, fmt.Errorf("weights: create output shard identifiers: %w", nonceErr)
	}
	baseName := filepath.Base(outputIndex)
	base := baseName[:len(baseName)-len(".safetensors.index.json")]
	if base == "" {
		base = "weights"
	}
	for len(base) > 120 {
		_, size := utf8.DecodeLastRuneInString(base)
		base = base[:len(base)-size]
	}
	stages := make([]stagedShard, 0, len(shardPaths))
	indexStagePath := ""
	committed := false
	defer func() {
		var cleanupErr error
		remove := func(path string) {
			if removeErr := checkpoint.root.Remove(path); removeErr != nil && !errors.Is(removeErr, os.ErrNotExist) {
				cleanupErr = errors.Join(cleanupErr, fmt.Errorf("weights: remove temporary output %q: %w", path, removeErr))
			}
		}
		if indexStagePath != "" {
			remove(indexStagePath)
		}
		if !committed {
			for _, shard := range stages {
				if shard.stagePath != "" {
					remove(shard.stagePath)
				}
				if shard.published {
					remove(shard.finalPath)
				}
			}
		}
		returnErr = errors.Join(returnErr, cleanupErr)
	}()

	for i, sourcePath := range shardPaths {
		if contextErr := ctx.Err(); contextErr != nil {
			return PatchResult{}, contextErr
		}
		finalName := fmt.Sprintf("%s-%s-%05d-of-%05d.safetensors", base, nonce, i+1, len(shardPaths))
		finalPath, pathErr := xfs.Rel(filepath.Join(outputDir, finalName))
		if pathErr != nil {
			return PatchResult{}, fmt.Errorf("weights: validate output shard path: %w", pathErr)
		}
		if _, statErr := checkpoint.root.Lstat(finalPath); statErr == nil {
			return PatchResult{}, fmt.Errorf("weights: generated output shard %q already exists", finalPath)
		} else if !errors.Is(statErr, os.ErrNotExist) {
			return PatchResult{}, fmt.Errorf("weights: inspect output shard %q: %w", finalPath, statErr)
		}
		stagePath, stageFile, err := createStageFile(checkpoint.root, outputDir)
		if err != nil {
			return PatchResult{}, err
		}
		stage := stagedShard{sourcePath: sourcePath, stagePath: stagePath, finalPath: finalPath, edits: groupedEdits[sourcePath]}
		stages = append(stages, stage)
		digest, size, perm, err := copyAndHash(ctx, checkpoint.root, sourcePath, stageFile)
		if err != nil {
			_ = stageFile.Close()
			return PatchResult{}, fmt.Errorf("weights: stage SafeTensors shard %q: %w", sourcePath, err)
		}
		if err := stageFile.Chmod(perm); err != nil {
			_ = stageFile.Close()
			return PatchResult{}, fmt.Errorf("weights: set staged shard permissions: %w", err)
		}
		if err := stageFile.Sync(); err != nil {
			_ = stageFile.Close()
			return PatchResult{}, fmt.Errorf("weights: sync staged shard: %w", err)
		}
		if err := stageFile.Close(); err != nil {
			return PatchResult{}, fmt.Errorf("weights: close staged shard: %w", err)
		}
		stages[len(stages)-1].digest = digest
		stages[len(stages)-1].size = size
	}

	indexDigest, err := digestPath(ctx, checkpoint.root, checkpoint.indexPath)
	if err != nil {
		return PatchResult{}, fmt.Errorf("weights: hash SafeTensors index: %w", err)
	}
	indexHash := sha256.Sum256(checkpoint.index.RawBytes)
	if indexDigest.SHA256 != hex.EncodeToString(indexHash[:]) || indexDigest.SizeBytes != int64(len(checkpoint.index.RawBytes)) {
		return PatchResult{}, errors.New("weights: SafeTensors index changed while preparing the patch")
	}
	inputFiles := make([]FileDigest, 0, len(stages)+1)
	inputFiles = append(inputFiles, indexDigest)
	for _, shard := range stages {
		inputFiles = append(inputFiles, FileDigest{Path: shard.sourcePath, SizeBytes: shard.size, SHA256: shard.digest})
	}
	if actual := aggregateDigest(inputFiles); !strings.EqualFold(actual, request.SourceSHA256) {
		return PatchResult{}, fmt.Errorf("weights: source checkpoint SHA-256 changed: expected %s, staged %s", strings.ToLower(request.SourceSHA256), actual)
	}

	outputShardBySource := make(map[string]string, len(stages))
	applied := make([]AppliedEdit, 0, len(request.Edits))
	outputFiles := make([]FileDigest, 0, len(stages)+1)
	for i := range stages {
		if contextErr := ctx.Err(); contextErr != nil {
			return PatchResult{}, contextErr
		}
		stage := &stages[i]
		outputDigest := stage.digest
		if len(stage.edits) > 0 {
			patchedStage, stagePathErr := uniquePath(checkpoint.root, outputDir, ".weights-patched-")
			if stagePathErr != nil {
				return PatchResult{}, stagePathErr
			}
			patched, patchErr := safetensors.PatchWithinRoot(ctx, checkpoint.root, checkpoint.commitRoot, stage.stagePath, patchedStage, stage.edits, safetensors.PatchOptions{
				ExpectedSHA256: stage.digest,
			})
			if patchErr != nil {
				return PatchResult{}, fmt.Errorf("weights: patch staged shard %q: %w", stage.sourcePath, patchErr)
			}
			if removeErr := checkpoint.root.Remove(stage.stagePath); removeErr != nil {
				cleanupErr := checkpoint.root.Remove(patchedStage)
				if cleanupErr != nil && !errors.Is(cleanupErr, os.ErrNotExist) {
					removeErr = errors.Join(removeErr, fmt.Errorf("weights: remove patched shard stage %q: %w", patchedStage, cleanupErr))
				}
				return PatchResult{}, fmt.Errorf("weights: remove unchanged shard stage: %w", removeErr)
			}
			stage.stagePath = patchedStage
			outputDigest = patched.OutputSHA256
			applied = append(applied, patched.Edits...)
		}
		if publishErr := filecommit.LinkNoReplace(checkpoint.root, checkpoint.commitRoot, stage.stagePath, stage.finalPath); publishErr != nil {
			return PatchResult{}, fmt.Errorf("weights: publish output shard %q without replacement: %w", stage.finalPath, publishErr)
		}
		stage.published = true
		if removeErr := checkpoint.root.Remove(stage.stagePath); removeErr != nil {
			return PatchResult{}, fmt.Errorf("weights: remove staged output shard %q: %w", stage.stagePath, removeErr)
		}
		stage.stagePath = ""
		outputShardBySource[stage.sourcePath] = stage.finalPath
		stat, statErr := checkpoint.root.Stat(stage.finalPath)
		if statErr != nil {
			return PatchResult{}, fmt.Errorf("weights: stat committed output shard: %w", statErr)
		}
		outputFiles = append(outputFiles, FileDigest{Path: stage.finalPath, SizeBytes: stat.Size(), SHA256: outputDigest})
	}

	newWeightMap := make(map[string]string, len(checkpoint.index.WeightMap))
	for tensorName := range checkpoint.index.WeightMap {
		outputShard, exists := outputShardBySource[checkpoint.tensorPaths[tensorName]]
		if !exists {
			return PatchResult{}, fmt.Errorf("weights: no staged output shard for tensor %q", tensorName)
		}
		newWeightMap[tensorName] = filepath.Base(outputShard)
	}
	newDocument := make(map[string]json.RawMessage, len(checkpoint.index.Document))
	for key, value := range checkpoint.index.Document {
		newDocument[key] = slices.Clone(value)
	}
	weightJSON, err := json.Marshal(newWeightMap)
	if err != nil {
		return PatchResult{}, fmt.Errorf("weights: encode output weight_map: %w", err)
	}
	newDocument["weight_map"] = weightJSON
	indexOutput, err := json.Marshal(newDocument)
	if err != nil {
		return PatchResult{}, fmt.Errorf("weights: encode output SafeTensors index: %w", err)
	}
	indexOutput = append(indexOutput, '\n')
	indexOutputDigest := sha256.Sum256(indexOutput)
	outputFiles = append(outputFiles, FileDigest{
		Path: outputIndex, SizeBytes: int64(len(indexOutput)), SHA256: hex.EncodeToString(indexOutputDigest[:]),
	})
	if err := ctx.Err(); err != nil {
		return PatchResult{}, err
	}
	if request.Overwrite {
		if err := checkpoint.root.WriteFileAtomic(outputIndex, indexOutput, 0o600); err != nil {
			return PatchResult{}, fmt.Errorf("weights: atomically replace output SafeTensors index: %w", err)
		}
	} else {
		indexStagePath, stageFile, err := createStageFile(checkpoint.root, outputDir)
		if err != nil {
			return PatchResult{}, err
		}
		written, writeErr := stageFile.Write(indexOutput)
		if writeErr != nil || written != len(indexOutput) {
			_ = stageFile.Close()
			if writeErr != nil {
				return PatchResult{}, fmt.Errorf("weights: write staged SafeTensors index: %w", writeErr)
			}
			return PatchResult{}, fmt.Errorf("weights: write staged SafeTensors index: %w", io.ErrShortWrite)
		}
		if err := stageFile.Sync(); err != nil {
			_ = stageFile.Close()
			return PatchResult{}, fmt.Errorf("weights: sync staged SafeTensors index: %w", err)
		}
		if err := stageFile.Close(); err != nil {
			return PatchResult{}, fmt.Errorf("weights: close staged SafeTensors index: %w", err)
		}
		if err := filecommit.LinkNoReplace(checkpoint.root, checkpoint.commitRoot, indexStagePath, outputIndex); err != nil {
			return PatchResult{}, fmt.Errorf("weights: atomically publish output SafeTensors index without replacement: %w", err)
		}
	}
	committed = true
	slices.SortFunc(applied, func(a, b AppliedEdit) int {
		if a.Tensor < b.Tensor {
			return -1
		}
		if a.Tensor > b.Tensor {
			return 1
		}
		if a.Index < b.Index {
			return -1
		}
		if a.Index > b.Index {
			return 1
		}
		return 0
	})
	slices.SortFunc(inputFiles, func(a, b FileDigest) int { return strings.Compare(a.Path, b.Path) })
	slices.SortFunc(outputFiles, func(a, b FileDigest) int { return strings.Compare(a.Path, b.Path) })
	return PatchResult{
		Model: request.Model, InputPath: request.Model.Path, OutputPath: outputIndex,
		InputSHA256: aggregateDigest(inputFiles), OutputSHA256: aggregateDigest(outputFiles),
		InputFiles: inputFiles, OutputFiles: outputFiles, Edits: applied,
	}, nil
}

func copyAndHash(ctx context.Context, root *xfs.Root, sourcePath string, dst *os.File) (digest string, total int64, mode os.FileMode, err error) {
	source, err := root.Open(sourcePath)
	if err != nil {
		return "", 0, 0, err
	}
	defer source.Close()
	stat, err := source.Stat()
	if err != nil {
		return "", 0, 0, err
	}
	if !stat.Mode().IsRegular() {
		return "", 0, 0, fmt.Errorf("%q is not a regular file", sourcePath)
	}
	hasher := sha256.New()
	buffer := make([]byte, 4<<20)
	for {
		if err := ctx.Err(); err != nil {
			return "", 0, 0, err
		}
		n, readErr := source.Read(buffer)
		if n > 0 {
			written, writeErr := dst.Write(buffer[:n])
			if writeErr != nil {
				return "", 0, 0, writeErr
			}
			if written != n {
				return "", 0, 0, io.ErrShortWrite
			}
			_, _ = hasher.Write(buffer[:n])
			total += int64(n)
		}
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return "", 0, 0, readErr
		}
	}
	if total != stat.Size() {
		return "", 0, 0, fmt.Errorf("source size changed while copying: stat=%d copied=%d", stat.Size(), total)
	}
	return hex.EncodeToString(hasher.Sum(nil)), total, stat.Mode().Perm(), nil
}

func createStageFile(root *xfs.Root, dir string) (string, *os.File, error) {
	for range 10 {
		suffix, err := randomSuffix()
		if err != nil {
			return "", nil, fmt.Errorf("weights: generate temporary filename: %w", err)
		}
		path, err := xfs.Rel(filepath.Join(dir, ".weights-stage-"+suffix+".tmp"))
		if err != nil {
			return "", nil, fmt.Errorf("weights: validate temporary filename: %w", err)
		}
		file, err := root.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
		if errors.Is(err, os.ErrExist) {
			continue
		}
		if err != nil {
			return "", nil, fmt.Errorf("weights: create temporary shard: %w", err)
		}
		return path, file, nil
	}
	return "", nil, errors.New("weights: could not create a unique temporary shard after retries")
}

func uniquePath(root *xfs.Root, dir, prefix string) (string, error) {
	for range 10 {
		suffix, err := randomSuffix()
		if err != nil {
			return "", fmt.Errorf("weights: generate temporary filename: %w", err)
		}
		path, err := xfs.Rel(filepath.Join(dir, prefix+suffix+".tmp"))
		if err != nil {
			return "", fmt.Errorf("weights: validate temporary filename: %w", err)
		}
		if _, err := root.Lstat(path); errors.Is(err, os.ErrNotExist) {
			return path, nil
		} else if err != nil {
			return "", fmt.Errorf("weights: inspect temporary path: %w", err)
		}
	}
	return "", errors.New("weights: could not find an unused temporary path after retries")
}

func randomSuffix() (string, error) {
	var raw [8]byte
	if _, err := rand.Read(raw[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(raw[:]), nil
}

func validateDigest(value string) error {
	value = strings.TrimSpace(value)
	if len(value) != sha256.Size*2 {
		return errors.New("weights: source_sha256 must be a 64-character hexadecimal digest")
	}
	if _, err := hex.DecodeString(value); err != nil {
		return fmt.Errorf("weights: invalid source_sha256: %w", err)
	}
	return nil
}

func rejectOutputAliases(checkpoint *openedCheckpoint, outputPath string) error {
	outInfo, err := checkpoint.root.Lstat(outputPath)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("weights: inspect output path: %w", err)
	}
	if outInfo.Mode()&os.ModeSymlink != 0 {
		return errors.New("weights: output path cannot be a symbolic link")
	}
	for _, sourcePath := range checkpoint.sourcePaths {
		source, err := checkpoint.root.Open(sourcePath)
		if err != nil {
			return fmt.Errorf("weights: open input while checking output path: %w", err)
		}
		sourceInfo, statErr := source.Stat()
		_ = source.Close()
		if statErr != nil {
			return fmt.Errorf("weights: stat input while checking output path: %w", statErr)
		}
		if os.SameFile(sourceInfo, outInfo) {
			return errors.New("weights: output_path refers to the same file as an input checkpoint member")
		}
	}
	return nil
}
