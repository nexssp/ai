package safetensors

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/nexssp/kernel/xfs"

	"github.com/nexssp/ai/weights/internal/filecommit"
)

type TensorData struct {
	DType string
	Shape []uint64
	Data  []byte
}

type TensorSummary struct {
	Name     string    `json:"name"`
	DType    string    `json:"dtype"`
	Shape    []uint64  `json:"shape"`
	Elements uint64    `json:"elements"`
	Bytes    uint64    `json:"bytes"`
	Sample   []float64 `json:"sample,omitempty"`
}

type Edit struct {
	Tensor        string   `json:"tensor"`
	Index         uint64   `json:"index"`
	Delta         float64  `json:"delta"`
	ExpectedValue *float64 `json:"expected_value,omitempty"`
}

type AppliedEdit struct {
	Tensor string  `json:"tensor"`
	Index  uint64  `json:"index"`
	Before float64 `json:"before"`
	After  float64 `json:"after"`
}

type PatchResult struct {
	InputPath    string        `json:"input_path"`
	OutputPath   string        `json:"output_path"`
	InputSHA256  string        `json:"input_sha256"`
	OutputSHA256 string        `json:"output_sha256"`
	Edits        []AppliedEdit `json:"edits"`
}

// PatchOptions binds a patch to the exact source checkpoint inspected by the caller.
type PatchOptions struct {
	ExpectedSHA256 string
	Overwrite      bool
}

type preparedEdit struct {
	edit   AppliedEdit
	offset int64
	bytes  []byte
}

// Write creates a SafeTensors file. The destination is never replaced unless overwrite is true.
func Write(path string, tensors map[string]TensorData, metadata map[string]string, overwrite bool) error {
	if len(tensors) == 0 {
		return errors.New("at least one tensor is required")
	}
	names, header, _, err := prepareWriteHeader(tensors, metadata)
	if err != nil {
		return err
	}
	headerJSON, err := json.Marshal(header)
	if err != nil {
		return fmt.Errorf("encode SafeTensors header: %w", err)
	}
	// The spec permits trailing spaces; padding to 8 bytes keeps data aligned.
	for len(headerJSON)%8 != 0 {
		headerJSON = append(headerJSON, ' ')
	}
	if len(headerJSON) > MaxHeaderBytes {
		return fmt.Errorf("encoded header exceeds %d bytes", MaxHeaderBytes)
	}
	return writeAtomic(path, overwrite, func(dst *os.File) error {
		var prefix [8]byte
		binary.LittleEndian.PutUint64(prefix[:], uint64(len(headerJSON)))
		if _, err := dst.Write(prefix[:]); err != nil {
			return err
		}
		if _, err := dst.Write(headerJSON); err != nil {
			return err
		}
		for _, name := range names {
			if _, err := dst.Write(tensors[name].Data); err != nil {
				return err
			}
		}
		return nil
	})
}

func prepareWriteHeader(tensors map[string]TensorData, metadata map[string]string) (names []string, header map[string]any, totalBytes uint64, returnErr error) {
	names = make([]string, 0, len(tensors))
	for name := range tensors {
		if strings.TrimSpace(name) == "" || name == "__metadata__" {
			return nil, nil, 0, fmt.Errorf("invalid tensor name %q", name)
		}
		names = append(names, name)
	}
	sort.Strings(names)
	header = make(map[string]any, len(tensors)+1)
	bodyBytes := uint64(0)
	for _, name := range names {
		tensor := tensors[name]
		size, err := dtypeSize(tensor.DType)
		if err != nil {
			return nil, nil, 0, fmt.Errorf("tensor %q: %w", name, err)
		}
		elements, err := elementCount(tensor.Shape)
		if err != nil {
			return nil, nil, 0, fmt.Errorf("tensor %q: %w", name, err)
		}
		want, err := multiply(elements, size)
		if err != nil {
			return nil, nil, 0, fmt.Errorf("tensor %q: %w", name, err)
		}
		if want != uint64(len(tensor.Data)) {
			return nil, nil, 0, fmt.Errorf("tensor %q has %d data bytes; dtype and shape require %d", name, len(tensor.Data), want)
		}
		end, err := add(bodyBytes, want)
		if err != nil {
			return nil, nil, 0, err
		}
		header[name] = map[string]any{"dtype": tensor.DType, "shape": tensor.Shape, "data_offsets": []uint64{bodyBytes, end}}
		bodyBytes = end
	}
	if metadata != nil {
		copyMetadata := make(map[string]string, len(metadata))
		for key, value := range metadata {
			if key == "" {
				return nil, nil, 0, errors.New("metadata keys cannot be empty")
			}
			copyMetadata[key] = value
		}
		header["__metadata__"] = copyMetadata
	}
	return names, header, bodyBytes, nil
}

func add(a, b uint64) (uint64, error) {
	if b > math.MaxUint64-a {
		return 0, errors.New("SafeTensors file size overflow")
	}
	return a + b, nil
}

func Inspect(path, tensorName string, sampleLimit uint64) ([]TensorSummary, error) {
	file, err := Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	if sampleLimit == 0 {
		sampleLimit = 8
	}
	if sampleLimit > 128 {
		sampleLimit = 128
	}
	if tensorName != "" {
		if _, ok := file.Tensors[tensorName]; !ok {
			return nil, fmt.Errorf("tensor %q does not exist", tensorName)
		}
	}
	names := make([]string, 0, len(file.Tensors))
	for name := range file.Tensors {
		if tensorName == "" || name == tensorName {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	out := make([]TensorSummary, 0, len(names))
	for _, name := range names {
		info := file.Tensors[name]
		size, err := dtypeSize(info.DType)
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", name, err)
		}
		span := TensorSummary{
			Name: name, DType: info.DType, Shape: append([]uint64(nil), info.Shape...),
			Elements: info.Elements, Bytes: info.DataOffsets[1] - info.DataOffsets[0],
		}
		if isEditableFloat(info.DType) && info.Elements > 0 {
			span.Sample, err = file.ReadFloatPrefix(name, sampleLimit)
			if err != nil {
				return nil, err
			}
		} else if size == 0 {
			return nil, fmt.Errorf("unsupported dtype %q", info.DType)
		}
		out = append(out, span)
	}
	return out, nil
}

// Patch applies finite deltas to a verified copy of source. The original is never edited.
// Index addresses the flat row-major element order used by SafeTensors.
func Patch(ctx context.Context, sourcePath, outputPath string, edits []Edit, options PatchOptions) (PatchResult, error) {
	if ctx == nil {
		return PatchResult{}, errors.New("context is required")
	}
	if len(edits) == 0 {
		return PatchResult{}, errors.New("at least one edit is required")
	}
	expectedSHA256 := strings.ToLower(strings.TrimSpace(options.ExpectedSHA256))
	if len(expectedSHA256) != sha256.Size*2 {
		return PatchResult{}, errors.New("expected source SHA-256 must be a 64-character hexadecimal digest")
	}
	if _, err := hex.DecodeString(expectedSHA256); err != nil {
		return PatchResult{}, fmt.Errorf("invalid expected source SHA-256: %w", err)
	}
	in, err := Open(sourcePath)
	if err != nil {
		return PatchResult{}, err
	}
	defer in.Close()
	if pathErr := ensureDistinctPaths(sourcePath, outputPath); pathErr != nil {
		return PatchResult{}, pathErr
	}
	preparedEdits, err := prepareEdits(ctx, in, edits)
	if err != nil {
		return PatchResult{}, err
	}
	if ctxErr := ctx.Err(); ctxErr != nil {
		return PatchResult{}, ctxErr
	}
	inputDigest, outputDigest, err := copyAndPatch(ctx, in, outputPath, preparedEdits, options.Overwrite, expectedSHA256)
	if err != nil {
		return PatchResult{}, err
	}
	applied := make([]AppliedEdit, len(preparedEdits))
	for i := range preparedEdits {
		applied[i] = preparedEdits[i].edit
	}
	return PatchResult{InputPath: sourcePath, OutputPath: outputPath, InputSHA256: inputDigest, OutputSHA256: outputDigest, Edits: applied}, nil
}

// PatchWithinRoot applies a patch through matching, already-open workspace roots.
// Both paths must be relative to those roots. Writes are streamed to a
// same-directory temporary file and atomically published without replacement
// unless overwrite is explicitly enabled.
func PatchWithinRoot(ctx context.Context, root *xfs.Root, commitRoot *os.Root, sourcePath, outputPath string, edits []Edit, options PatchOptions) (PatchResult, error) {
	if ctx == nil || root == nil || commitRoot == nil {
		return PatchResult{}, errors.New("context and matching workspace roots are required")
	}
	if err := ctx.Err(); err != nil {
		return PatchResult{}, err
	}
	if err := filecommit.VerifyRoots(root, commitRoot); err != nil {
		return PatchResult{}, err
	}
	if len(edits) == 0 {
		return PatchResult{}, errors.New("at least one edit is required")
	}
	expectedSHA256 := strings.ToLower(strings.TrimSpace(options.ExpectedSHA256))
	if len(expectedSHA256) != sha256.Size*2 {
		return PatchResult{}, errors.New("expected source SHA-256 must be a 64-character hexadecimal digest")
	}
	if _, err := hex.DecodeString(expectedSHA256); err != nil {
		return PatchResult{}, fmt.Errorf("invalid expected source SHA-256: %w", err)
	}
	inFile, err := root.Open(sourcePath)
	if err != nil {
		return PatchResult{}, fmt.Errorf("open workspace-relative input: %w", err)
	}
	in, err := OpenFile(inFile, sourcePath)
	if err != nil {
		return PatchResult{}, err
	}
	defer in.Close()
	if pathErr := ensureDistinctPathsWithinRoot(root, in, sourcePath, outputPath); pathErr != nil {
		return PatchResult{}, pathErr
	}
	preparedEdits, err := prepareEdits(ctx, in, edits)
	if err != nil {
		return PatchResult{}, err
	}
	inputDigest, outputDigest, err := copyAndPatchWithinRoot(ctx, root, commitRoot, in, outputPath, preparedEdits, options.Overwrite, expectedSHA256)
	if err != nil {
		return PatchResult{}, err
	}
	applied := make([]AppliedEdit, len(preparedEdits))
	for i := range preparedEdits {
		applied[i] = preparedEdits[i].edit
	}
	return PatchResult{InputPath: sourcePath, OutputPath: outputPath, InputSHA256: inputDigest, OutputSHA256: outputDigest, Edits: applied}, nil
}

func prepareEdits(ctx context.Context, in *File, edits []Edit) ([]preparedEdit, error) {
	preparedEdits := make([]preparedEdit, 0, len(edits))
	seen := make(map[string]struct{}, len(edits))
	for _, edit := range edits {
		prepared, key, err := prepareOneEdit(ctx, in, edit)
		if err != nil {
			return nil, err
		}
		if _, exists := seen[key]; exists {
			return nil, fmt.Errorf("duplicate edit for tensor %q index %d", edit.Tensor, edit.Index)
		}
		seen[key] = struct{}{}
		preparedEdits = append(preparedEdits, prepared)
	}
	return preparedEdits, nil
}

func prepareOneEdit(ctx context.Context, in *File, edit Edit) (preparedEdit, string, error) {
	if err := ctx.Err(); err != nil {
		return preparedEdit{}, "", err
	}
	if strings.TrimSpace(edit.Tensor) == "" {
		return preparedEdit{}, "", errors.New("edit tensor name is required")
	}
	if math.IsNaN(edit.Delta) || math.IsInf(edit.Delta, 0) {
		return preparedEdit{}, "", fmt.Errorf("edit for tensor %q has a non-finite delta", edit.Tensor)
	}
	info, ok := in.Tensors[edit.Tensor]
	if !ok {
		return preparedEdit{}, "", fmt.Errorf("tensor %q does not exist", edit.Tensor)
	}
	if !isEditableFloat(info.DType) {
		return preparedEdit{}, "", fmt.Errorf("tensor %q uses %s; editing supports F16, BF16, F32, and F64", edit.Tensor, info.DType)
	}
	if edit.Index >= info.Elements {
		return preparedEdit{}, "", fmt.Errorf("index %d is outside tensor %q with %d elements", edit.Index, edit.Tensor, info.Elements)
	}
	size, err := dtypeSize(info.DType)
	if err != nil {
		return preparedEdit{}, "", err
	}
	before, err := in.ReadFloat(edit.Tensor, edit.Index)
	if err != nil {
		return preparedEdit{}, "", err
	}
	if edit.ExpectedValue != nil {
		if math.IsNaN(*edit.ExpectedValue) || math.IsInf(*edit.ExpectedValue, 0) {
			return preparedEdit{}, "", errors.New("expected_value must be finite")
		}
		tolerance := 1e-6 * math.Max(1, math.Abs(*edit.ExpectedValue))
		if math.Abs(before-*edit.ExpectedValue) > tolerance {
			return preparedEdit{}, "", fmt.Errorf("stale weight at %s[%d]: expected %.9g, found %.9g", edit.Tensor, edit.Index, *edit.ExpectedValue, before)
		}
	}
	encoded, after, err := encodeFloat(info.DType, before+edit.Delta)
	if err != nil {
		return preparedEdit{}, "", fmt.Errorf("edit %s[%d]: %w", edit.Tensor, edit.Index, err)
	}
	offset, err := checkedEditOffset(in, info, edit.Index, size)
	if err != nil {
		return preparedEdit{}, "", err
	}
	return preparedEdit{edit: AppliedEdit{Tensor: edit.Tensor, Index: edit.Index, Before: before, After: after}, offset: offset, bytes: encoded}, fmt.Sprintf("%s\x00%d", edit.Tensor, edit.Index), nil
}

func checkedEditOffset(in *File, info TensorInfo, index, size uint64) (int64, error) {
	if in.DataStart < 0 || uint64(in.DataStart) > math.MaxInt64 || info.DataOffsets[0] > uint64(math.MaxInt64)-uint64(in.DataStart) || index > (uint64(math.MaxInt64)-uint64(in.DataStart)-info.DataOffsets[0])/size {
		return 0, errors.New("tensor byte offset exceeds supported file range")
	}
	relativeOffset := info.DataOffsets[0] + index*size
	if relativeOffset > uint64(math.MaxInt64)-uint64(in.DataStart) {
		return 0, errors.New("tensor byte offset exceeds supported file range")
	}
	return checkedFileOffset(in.DataStart, relativeOffset)
}

func ensureDistinctPathsWithinRoot(root *xfs.Root, in *File, sourcePath, outputPath string) error {
	if strings.TrimSpace(outputPath) == "" {
		return errors.New("output_path is required")
	}
	if filepath.Clean(sourcePath) == filepath.Clean(outputPath) {
		return errors.New("output_path must differ from input_path; the source checkpoint is never overwritten")
	}
	outputInfo, err := root.Lstat(outputPath)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("inspect workspace-relative output path: %w", err)
	}
	if outputInfo.Mode()&os.ModeSymlink != 0 {
		return errors.New("output_path cannot be a symbolic link")
	}
	sourceInfo, err := in.Stat()
	if err != nil {
		return fmt.Errorf("stat source checkpoint: %w", err)
	}
	if os.SameFile(sourceInfo, outputInfo) {
		return errors.New("output_path refers to the same file as input_path")
	}
	return nil
}

func prepareRootOutput(root *xfs.Root, outputPath string, overwrite bool) (cleanOutput, dir string, returnErr error) {
	var pathErr error
	cleanOutput, pathErr = xfs.Rel(outputPath)
	if pathErr != nil {
		return "", "", fmt.Errorf("validate output path: %w", pathErr)
	}
	dir = filepath.Dir(cleanOutput)
	if dir != "." {
		if mkdirErr := root.MkdirAll(dir, 0o755); mkdirErr != nil {
			return "", "", fmt.Errorf("create output directory: %w", mkdirErr)
		}
	}
	if existing, statErr := root.Lstat(cleanOutput); statErr == nil {
		if existing.Mode()&os.ModeSymlink != 0 {
			return "", "", errors.New("output_path cannot be a symbolic link")
		}
		if !overwrite {
			return "", "", fmt.Errorf("output %q already exists; set overwrite=true only when replacing this output is intended", outputPath)
		}
	} else if !errors.Is(statErr, os.ErrNotExist) {
		return "", "", fmt.Errorf("inspect output path: %w", statErr)
	}
	return cleanOutput, dir, nil
}

func copyAndPatchWithinRoot(ctx context.Context, root *xfs.Root, commitRoot *os.Root, in *File, outputPath string, edits []preparedEdit, overwrite bool, expectedSHA256 string) (inputHash, outputHash string, returnErr error) {
	cleanOutput, dir, setupErr := prepareRootOutput(root, outputPath, overwrite)
	if setupErr != nil {
		return "", "", setupErr
	}
	tmpPath, tmp, createErr := createRootTemp(root, dir)
	if createErr != nil {
		return "", "", createErr
	}
	tmpClosed := false
	defer func() {
		if !tmpClosed {
			if closeErr := tmp.Close(); closeErr != nil {
				returnErr = errors.Join(returnErr, fmt.Errorf("close temporary output: %w", closeErr))
			}
		}
		if removeErr := root.Remove(tmpPath); removeErr != nil && !errors.Is(removeErr, os.ErrNotExist) {
			returnErr = errors.Join(returnErr, fmt.Errorf("remove temporary output %q: %w", tmpPath, removeErr))
		}
	}()
	inputDigest, outputDigest, copyErr := copyAndApplyEdits(ctx, in, tmp, edits, expectedSHA256)
	if copyErr != nil {
		return "", "", copyErr
	}
	sourceInfo, err := in.Stat()
	if err != nil {
		return "", "", fmt.Errorf("stat source checkpoint: %w", err)
	}
	if err := tmp.Chmod(sourceInfo.Mode().Perm()); err != nil {
		return "", "", fmt.Errorf("set output permissions: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		return "", "", fmt.Errorf("sync output checkpoint: %w", err)
	}
	closeErr := tmp.Close()
	tmpClosed = true
	if closeErr != nil {
		return "", "", fmt.Errorf("close output checkpoint: %w", closeErr)
	}
	if overwrite {
		if err := root.Rename(tmpPath, cleanOutput); err != nil {
			return "", "", fmt.Errorf("commit output checkpoint: %w", err)
		}
	} else if err := filecommit.LinkNoReplace(root, commitRoot, tmpPath, cleanOutput); err != nil {
		return "", "", fmt.Errorf("commit output checkpoint without replacement: %w", err)
	}
	return inputDigest, outputDigest, nil
}

func createRootTemp(root *xfs.Root, dir string) (string, *os.File, error) {
	for range 10 {
		var suffix [16]byte
		if _, err := rand.Read(suffix[:]); err != nil {
			return "", nil, fmt.Errorf("generate temporary output name: %w", err)
		}
		name := ".weight-editor-" + hex.EncodeToString(suffix[:]) + ".tmp"
		if dir != "." {
			name = filepath.Join(dir, name)
		}
		file, err := root.OpenFile(name, os.O_CREATE|os.O_EXCL|os.O_RDWR, 0o600)
		if errors.Is(err, os.ErrExist) {
			continue
		}
		if err != nil {
			return "", nil, fmt.Errorf("create temporary output: %w", err)
		}
		return name, file, nil
	}
	return "", nil, errors.New("could not create a unique temporary output after 10 attempts")
}

func ensureDistinctPaths(sourcePath, outputPath string) error {
	if strings.TrimSpace(outputPath) == "" {
		return errors.New("output_path is required")
	}
	sourceAbs, err := filepath.Abs(sourcePath)
	if err != nil {
		return fmt.Errorf("resolve input path: %w", err)
	}
	outputAbs, err := filepath.Abs(outputPath)
	if err != nil {
		return fmt.Errorf("resolve output path: %w", err)
	}
	if filepath.Clean(sourceAbs) == filepath.Clean(outputAbs) {
		return errors.New("output_path must differ from input_path; the source checkpoint is never overwritten")
	}
	if st, err := os.Lstat(outputAbs); err == nil {
		if st.Mode()&os.ModeSymlink != 0 {
			return errors.New("output_path cannot be a symbolic link")
		}
		sourceInfo, sourceErr := os.Stat(sourceAbs)
		if sourceErr != nil {
			return sourceErr
		}
		if os.SameFile(sourceInfo, st) {
			return errors.New("output_path refers to the same file as input_path")
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("inspect output path: %w", err)
	}
	return nil
}

func writeAtomic(path string, overwrite bool, fill func(*os.File) error) error {
	if strings.TrimSpace(path) == "" {
		return errors.New("output path is required")
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return fmt.Errorf("resolve output path: %w", err)
	}
	if st, statErr := os.Lstat(abs); statErr == nil {
		if st.Mode()&os.ModeSymlink != 0 {
			return errors.New("output path cannot be a symbolic link")
		}
		if !overwrite {
			return fmt.Errorf("output %q already exists; set overwrite=true only when replacing this output is intended", path)
		}
	} else if !errors.Is(statErr, os.ErrNotExist) {
		return fmt.Errorf("check output path: %w", statErr)
	}
	dir := filepath.Dir(abs)
	if mkdirErr := os.MkdirAll(dir, 0o755); mkdirErr != nil {
		return fmt.Errorf("create output directory: %w", mkdirErr)
	}
	tmp, err := os.CreateTemp(dir, ".weight-editor-*.tmp")
	if err != nil {
		return fmt.Errorf("create temporary output: %w", err)
	}
	tmpName := tmp.Name()
	defer func() {
		_ = tmp.Close()
		_ = os.Remove(tmpName)
	}()
	if err := fill(tmp); err != nil {
		return fmt.Errorf("write temporary output: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		return fmt.Errorf("sync temporary output: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close temporary output: %w", err)
	}
	if overwrite {
		if err := os.Rename(tmpName, abs); err != nil {
			return fmt.Errorf("commit output: %w", err)
		}
	} else {
		if err := os.Link(tmpName, abs); err != nil {
			return fmt.Errorf("commit new output without replacing an existing file: %w", err)
		}
		if err := os.Remove(tmpName); err != nil {
			return fmt.Errorf("remove temporary output link: %w", err)
		}
	}
	if dirFile, err := os.Open(dir); err == nil {
		syncErr := dirFile.Sync()
		closeErr := dirFile.Close()
		if syncErr != nil {
			return fmt.Errorf("sync output directory: %w", syncErr)
		}
		if closeErr != nil {
			return fmt.Errorf("close output directory: %w", closeErr)
		}
	}
	return nil
}

func prepareOutputPath(outputPath string, overwrite bool) (abs, dir string, returnErr error) {
	var pathErr error
	abs, pathErr = filepath.Abs(outputPath)
	if pathErr != nil {
		return "", "", fmt.Errorf("resolve output path: %w", pathErr)
	}
	dir = filepath.Dir(abs)
	if mkdirErr := os.MkdirAll(dir, 0o755); mkdirErr != nil {
		return "", "", fmt.Errorf("create output directory: %w", mkdirErr)
	}
	if st, statErr := os.Lstat(abs); statErr == nil {
		if st.Mode()&os.ModeSymlink != 0 {
			return "", "", errors.New("output_path cannot be a symbolic link")
		}
		if !overwrite {
			return "", "", fmt.Errorf("output %q already exists; set overwrite=true only when replacing this output is intended", outputPath)
		}
	} else if !errors.Is(statErr, os.ErrNotExist) {
		return "", "", fmt.Errorf("inspect output path: %w", statErr)
	}
	return abs, dir, nil
}

func copyAndPatch(ctx context.Context, in *File, outputPath string, edits []preparedEdit, overwrite bool, expectedSHA256 string) (inputHash, outputHash string, returnErr error) {
	abs, dir, setupErr := prepareOutputPath(outputPath, overwrite)
	if setupErr != nil {
		return "", "", setupErr
	}
	tmp, createErr := os.CreateTemp(dir, ".weight-editor-*.tmp")
	if createErr != nil {
		return "", "", fmt.Errorf("create temporary output: %w", createErr)
	}
	tmpName := tmp.Name()
	defer func() {
		_ = tmp.Close()
		_ = os.Remove(tmpName)
	}()
	inputDigest, outputDigest, copyErr := copyAndApplyEdits(ctx, in, tmp, edits, expectedSHA256)
	if copyErr != nil {
		return "", "", copyErr
	}
	sourceInfo, err := in.file.Stat()
	if err != nil {
		return "", "", fmt.Errorf("stat source checkpoint: %w", err)
	}
	if err := tmp.Chmod(sourceInfo.Mode().Perm()); err != nil {
		return "", "", fmt.Errorf("set output permissions: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		return "", "", fmt.Errorf("sync output checkpoint: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return "", "", fmt.Errorf("close output checkpoint: %w", err)
	}
	if overwrite {
		if err := os.Rename(tmpName, abs); err != nil {
			return "", "", fmt.Errorf("commit output checkpoint: %w", err)
		}
	} else {
		if err := os.Link(tmpName, abs); err != nil {
			return "", "", fmt.Errorf("commit new output without replacing an existing file: %w", err)
		}
		if err := os.Remove(tmpName); err != nil {
			return "", "", fmt.Errorf("remove temporary output link: %w", err)
		}
	}
	if dirFile, err := os.Open(dir); err == nil {
		syncErr := dirFile.Sync()
		closeErr := dirFile.Close()
		if syncErr != nil {
			return "", "", fmt.Errorf("sync output directory: %w", syncErr)
		}
		if closeErr != nil {
			return "", "", fmt.Errorf("close output directory: %w", closeErr)
		}
	}
	return inputDigest, outputDigest, nil
}

func copyAndApplyEdits(ctx context.Context, in *File, tmp *os.File, edits []preparedEdit, expectedSHA256 string) (inputDigest, outputDigest string, returnErr error) {
	if _, seekErr := in.file.Seek(0, io.SeekStart); seekErr != nil {
		return "", "", fmt.Errorf("seek input: %w", seekErr)
	}
	buffer := make([]byte, 4<<20)
	inputHasher := sha256.New()
	if copyErr := copyWithContext(ctx, tmp, in.file, buffer, inputHasher); copyErr != nil {
		return "", "", fmt.Errorf("copy input checkpoint: %w", copyErr)
	}
	inputDigest = hex.EncodeToString(inputHasher.Sum(nil))
	if inputDigest != expectedSHA256 {
		return "", "", fmt.Errorf("source checkpoint SHA-256 changed: expected %s, copied %s", expectedSHA256, inputDigest)
	}
	for _, edit := range edits {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return "", "", ctxErr
		}
		if _, writeErr := tmp.WriteAt(edit.bytes, edit.offset); writeErr != nil {
			return "", "", fmt.Errorf("write edited tensor value: %w", writeErr)
		}
	}
	if _, seekErr := tmp.Seek(0, io.SeekStart); seekErr != nil {
		return "", "", fmt.Errorf("seek patched checkpoint for digest: %w", seekErr)
	}
	outputHasher := sha256.New()
	if hashErr := copyWithContext(ctx, outputHasher, tmp, buffer, nil); hashErr != nil {
		return "", "", fmt.Errorf("hash patched checkpoint: %w", hashErr)
	}
	outputDigest = hex.EncodeToString(outputHasher.Sum(nil))
	return inputDigest, outputDigest, nil
}

func copyWithContext(ctx context.Context, dst io.Writer, src io.Reader, buffer []byte, digest io.Writer) error {
	writer := dst
	if digest != nil {
		writer = io.MultiWriter(dst, digest)
	}
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		n, readErr := src.Read(buffer)
		if n > 0 {
			written, writeErr := writer.Write(buffer[:n])
			if writeErr != nil {
				return writeErr
			}
			if written != n {
				return io.ErrShortWrite
			}
		}
		if errors.Is(readErr, io.EOF) {
			return nil
		}
		if readErr != nil {
			return readErr
		}
	}
}
