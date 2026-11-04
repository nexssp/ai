package weights

import (
	"context"
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
	"slices"
	"sort"
	"strings"

	"github.com/nexssp/kernel/xfs"

	"github.com/nexssp/ai/weights/gguf"
	"github.com/nexssp/ai/weights/internal/filecommit"
	"github.com/nexssp/ai/weights/safetensors"
)

var (
	ErrUnsupportedAdapter  = errors.New("weights: unsupported adapter")
	ErrUnsupportedFormat   = errors.New("weights: unsupported checkpoint format")
	ErrUnsupportedLayout   = errors.New("weights: unsupported checkpoint layout")
	ErrUnsafeSerialization = errors.New("weights: unsafe checkpoint serialization")
)

const (
	maxSafeIndexBytes = 64 << 20
	maxShardFiles     = 4096
	maxBundleTensors  = 1 << 20
)

var safeTensorsAdapter = AdapterInfo{
	ID:                   AdapterSafeTensorsScalarDelta,
	Format:               FormatSafeTensors,
	Scope:                "single SafeTensors file or standard .safetensors.index.json bundle; scalar deltas to floating tensors only; no architecture validation, model loading, inference, or evaluation",
	EditableDTypes:       []string{"F16", "BF16", "F32", "F64"},
	PreservedDTypes:      []string{"BOOL", "U8", "I8", "F8_E4M3", "F8_E4M3FN", "F8_E5M2", "F8_E5M2FNUZ", "U16", "I16", "U32", "I32", "U64", "I64"},
	SupportsShardedFiles: true,
	SupportsInference:    false,
}

var ggufAdapter = AdapterInfo{
	ID:                   AdapterGGUFScalarDelta,
	Format:               FormatGGUF,
	Scope:                "one little-endian GGUF v2/v3 file; F16, BF16, F32, and F64 scalar edits; known quantized blocks are bounds-checked and preserved but never dequantized or edited",
	EditableDTypes:       []string{"F16", "BF16", "F32", "F64"},
	PreservedDTypes:      []string{"I8", "I16", "I32", "I64", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS", "IQ1_M", "Q4_0_4_4", "Q4_0_4_8", "Q4_0_8_8", "TQ1_0", "TQ2_0", "IQ4_NL_4_4", "IQ4_NL_4_8", "IQ4_NL_8_8", "MXFP4", "NVFP4", "Q1_0", "Q2_0"},
	SupportsShardedFiles: false,
	SupportsInference:    false,
}

// Adapters returns a defensive copy of implemented editing adapters. It does not
// advertise PyTorch: general .pt/.pth/.bin deserialization is unsafe without a
// fully constrained, non-executing tensor-only reader.
func Adapters() []AdapterInfo {
	out := []AdapterInfo{safeTensorsAdapter, ggufAdapter}
	for i := range out {
		out[i].EditableDTypes = slices.Clone(out[i].EditableDTypes)
		out[i].PreservedDTypes = slices.Clone(out[i].PreservedDTypes)
	}
	return out
}

type openedCheckpoint struct {
	ctx         context.Context
	root        *xfs.Root
	commitRoot  *os.Root
	model       Model
	format      string
	tensors     map[string]TensorInfo
	tensorPaths map[string]string
	sourcePaths []string
	fileSizes   map[string]int64
	sharded     bool
	index       *safeIndex
	indexPath   string
}

func (c *openedCheckpoint) Close() error {
	if c == nil {
		return nil
	}
	var rootErr, commitErr error
	if c.root != nil {
		rootErr = c.root.Close()
	}
	if c.commitRoot != nil {
		commitErr = c.commitRoot.Close()
	}
	return errors.Join(rootErr, commitErr)
}

func openCheckpoint(ctx context.Context, model Model) (*openedCheckpoint, error) {
	if err := validateCheckpointRequest(ctx, model); err != nil {
		return nil, err
	}
	workspaceRoot, err := filepath.Abs(".")
	if err != nil {
		return nil, fmt.Errorf("weights: resolve workspace root: %w", err)
	}
	commitRoot, err := os.OpenRoot(workspaceRoot)
	if err != nil {
		return nil, fmt.Errorf("weights: open publication root: %w", err)
	}
	root, err := xfs.OpenRoot(workspaceRoot)
	if err != nil {
		_ = commitRoot.Close()
		return nil, fmt.Errorf("weights: open workspace root: %w", err)
	}
	if err := filecommit.VerifyRoots(root, commitRoot); err != nil {
		_ = root.Close()
		_ = commitRoot.Close()
		return nil, fmt.Errorf("weights: verify matching workspace roots: %w", err)
	}
	checkpoint := &openedCheckpoint{
		ctx: ctx, root: root, commitRoot: commitRoot, model: model, format: model.Format,
		tensors: make(map[string]TensorInfo), tensorPaths: make(map[string]string),
		fileSizes: make(map[string]int64),
	}
	if err := checkpoint.openModelFile(); err != nil {
		_ = root.Close()
		_ = commitRoot.Close()
		return nil, err
	}
	return checkpoint, nil
}

func validateCheckpointRequest(ctx context.Context, model Model) error {
	if ctx == nil {
		return errors.New("weights: context is required")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if strings.TrimSpace(model.ID) == "" {
		return errors.New("weights: model_id is required (it is a caller-provided label)")
	}
	if strings.TrimSpace(model.Path) == "" {
		return errors.New("weights: model path is required")
	}
	switch model.Format {
	case FormatSafeTensors:
		if model.Adapter != AdapterSafeTensorsScalarDelta {
			return fmt.Errorf("%w: %q; supported adapter for %s: %s", ErrUnsupportedAdapter, model.Adapter, FormatSafeTensors, AdapterSafeTensorsScalarDelta)
		}
	case FormatGGUF:
		if model.Adapter != AdapterGGUFScalarDelta {
			return fmt.Errorf("%w: %q; supported adapter for %s: %s", ErrUnsupportedAdapter, model.Adapter, FormatGGUF, AdapterGGUFScalarDelta)
		}
	case FormatPyTorch:
		return fmt.Errorf("%w: PyTorch .pt/.pth/.bin checkpoint parsing is disabled because torch.save commonly embeds pickle; no unpickling, Python sidecar, or arbitrary-object loading is performed. Convert a tensor-only state dict to SafeTensors with a trusted tool, then select format %q", ErrUnsafeSerialization, FormatSafeTensors)
	default:
		return fmt.Errorf("%w: %q; implemented formats: %s, %s (PyTorch is rejected unless a safe tensor-only parser is added)", ErrUnsupportedFormat, model.Format, FormatSafeTensors, FormatGGUF)
	}
	if filepath.IsAbs(model.Path) {
		return errors.New("weights: absolute checkpoint paths are not allowed; use a workspace-relative path")
	}
	return nil
}

func (c *openedCheckpoint) openModelFile() error {
	var err error
	switch c.format {
	case FormatSafeTensors:
		if strings.HasSuffix(strings.ToLower(c.model.Path), ".safetensors.index.json") {
			err = c.openShardedSafeTensors(c.model.Path)
		} else {
			err = c.openSingleSafeTensors(c.model.Path)
		}
	case FormatGGUF:
		err = c.openGGUF(c.model.Path)
	}
	return err
}

func (c *openedCheckpoint) openSingleSafeTensors(path string) error {
	file, err := c.root.Open(path)
	if err != nil {
		return fmt.Errorf("weights: open workspace-relative SafeTensors checkpoint %q: %w", path, err)
	}
	stat, err := file.Stat()
	if err != nil {
		_ = file.Close()
		return fmt.Errorf("weights: stat checkpoint %q: %w", path, err)
	}
	if !stat.Mode().IsRegular() {
		_ = file.Close()
		return fmt.Errorf("%w: expected a regular SafeTensors file, got %q", ErrUnsupportedLayout, path)
	}
	var magic [4]byte
	n, readErr := file.Read(magic[:])
	if readErr != nil && !errors.Is(readErr, io.EOF) {
		_ = file.Close()
		return fmt.Errorf("weights: read checkpoint signature: %w", readErr)
	}
	if _, seekErr := file.Seek(0, io.SeekStart); seekErr != nil {
		_ = file.Close()
		return fmt.Errorf("weights: rewind checkpoint: %w", seekErr)
	}
	if n == len(magic) && string(magic[:]) == "GGUF" {
		_ = file.Close()
		return fmt.Errorf("%w: file signature is GGUF but format %q was selected; select format %q and adapter %q", ErrUnsupportedFormat, FormatSafeTensors, FormatGGUF, AdapterGGUFScalarDelta)
	}
	parsed, err := safetensors.OpenFile(file, path)
	if err != nil {
		return fmt.Errorf("weights: open SafeTensors checkpoint: %w", err)
	}
	defer parsed.Close()
	c.sourcePaths = []string{path}
	c.fileSizes[path] = stat.Size()
	for _, name := range parsed.TensorNames() {
		info := parsed.Tensors[name]
		c.tensors[name] = TensorInfo{
			Name: name, DType: info.DType, Shape: slices.Clone(info.Shape), Elements: info.Elements,
			Bytes: info.DataOffsets[1] - info.DataOffsets[0],
		}
		c.tensorPaths[name] = path
	}
	return nil
}

func (c *openedCheckpoint) openGGUF(path string) error {
	file, err := c.root.Open(path)
	if err != nil {
		return fmt.Errorf("weights: open workspace-relative GGUF checkpoint %q: %w", path, err)
	}
	stat, err := file.Stat()
	if err != nil {
		_ = file.Close()
		return fmt.Errorf("weights: stat GGUF checkpoint %q: %w", path, err)
	}
	if !stat.Mode().IsRegular() {
		_ = file.Close()
		return fmt.Errorf("%w: expected a regular GGUF file, got %q", ErrUnsupportedLayout, path)
	}
	parsed, err := gguf.OpenFile(c.ctx, file, path)
	if err != nil {
		return fmt.Errorf("weights: open GGUF checkpoint: %w", err)
	}
	defer parsed.Close()
	if parsed.SplitCount > 1 {
		return fmt.Errorf("%w: GGUF split files are not supported by this adapter; select a complete single GGUF file", ErrUnsupportedLayout)
	}
	c.sourcePaths = []string{path}
	c.fileSizes[path] = stat.Size()
	for _, name := range parsed.TensorNames() {
		info := parsed.Tensors[name]
		c.tensors[name] = TensorInfo{
			Name: name, DType: info.DType, Shape: slices.Clone(info.Shape), Elements: info.Elements,
			Bytes: info.Bytes,
		}
		c.tensorPaths[name] = path
	}
	return nil
}

func (c *openedCheckpoint) readFloatPrefix(name string, count uint64) ([]float64, error) {
	if err := c.ctx.Err(); err != nil {
		return nil, err
	}
	path, exists := c.tensorPaths[name]
	if !exists {
		return nil, fmt.Errorf("tensor %q does not exist", name)
	}
	file, err := c.root.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open tensor source %q: %w", path, err)
	}
	if c.format == FormatGGUF {
		parsed, parseErr := gguf.OpenFile(c.ctx, file, path)
		if parseErr != nil {
			return nil, parseErr
		}
		defer parsed.Close()
		return parsed.ReadFloatPrefix(name, count)
	}
	parsed, err := safetensors.OpenFile(file, path)
	if err != nil {
		return nil, err
	}
	defer parsed.Close()
	return parsed.ReadFloatPrefix(name, count)
}

func (c *openedCheckpoint) provenance(ctx context.Context) (files []FileDigest, aggregate string, total int64, err error) {
	files = make([]FileDigest, 0, len(c.sourcePaths))
	for _, path := range c.sourcePaths {
		if err := ctx.Err(); err != nil {
			return nil, "", 0, err
		}
		digest, err := digestPath(ctx, c.root, path)
		if err != nil {
			return nil, "", 0, fmt.Errorf("hash checkpoint member %q: %w", path, err)
		}
		if c.index != nil && path == c.indexPath {
			expected := sha256.Sum256(c.index.RawBytes)
			if digest.SHA256 != hex.EncodeToString(expected[:]) {
				return nil, "", 0, errors.New("weights: SafeTensors index changed while it was being inspected")
			}
		}
		files = append(files, digest)
		if digest.SizeBytes > math.MaxInt64-total {
			return nil, "", 0, errors.New("weights: total checkpoint size overflows int64")
		}
		total += digest.SizeBytes
	}
	sort.Slice(files, func(i, j int) bool { return files[i].Path < files[j].Path })
	return files, aggregateDigest(files), total, nil
}

func digestPath(ctx context.Context, root *xfs.Root, path string) (FileDigest, error) {
	file, err := root.Open(path)
	if err != nil {
		return FileDigest{}, err
	}
	defer file.Close()
	stat, err := file.Stat()
	if err != nil {
		return FileDigest{}, err
	}
	if !stat.Mode().IsRegular() {
		return FileDigest{}, fmt.Errorf("%q is not a regular file", path)
	}
	hasher := sha256.New()
	buffer := make([]byte, 4<<20)
	for {
		if err := ctx.Err(); err != nil {
			return FileDigest{}, err
		}
		n, readErr := file.Read(buffer)
		if n > 0 {
			if _, err := hasher.Write(buffer[:n]); err != nil {
				return FileDigest{}, err
			}
		}
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return FileDigest{}, readErr
		}
	}
	return FileDigest{Path: path, SizeBytes: stat.Size(), SHA256: hex.EncodeToString(hasher.Sum(nil))}, nil
}

func aggregateDigest(files []FileDigest) string {
	if len(files) == 1 {
		return files[0].SHA256
	}
	ordered := slices.Clone(files)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].Path < ordered[j].Path })
	hasher := sha256.New()
	var number [8]byte
	for _, file := range ordered {
		binary.LittleEndian.PutUint64(number[:], uint64(len(file.Path)))
		_, _ = hasher.Write(number[:])
		if _, err := io.WriteString(hasher, file.Path); err != nil {
			return ""
		}
		if file.SizeBytes < 0 {
			return ""
		}
		binary.LittleEndian.PutUint64(number[:], uint64(file.SizeBytes))
		_, _ = hasher.Write(number[:])
		digest, err := hex.DecodeString(file.SHA256)
		if err != nil {
			return ""
		}
		_, _ = hasher.Write(digest)
	}
	return hex.EncodeToString(hasher.Sum(nil))
}

func editableDType(format, dtype string) bool {
	if format == FormatGGUF {
		return dtype == "F16" || dtype == "BF16" || dtype == "F32" || dtype == "F64"
	}
	return dtype == "F16" || dtype == "BF16" || dtype == "F32" || dtype == "F64"
}

// safeIndex is the standard Hugging Face checkpoint index. Other top-level
// fields are preserved as raw JSON when the output index is rewritten.
type safeIndex struct {
	Document  map[string]json.RawMessage
	WeightMap map[string]string
	RawBytes  []byte
}

func (c *openedCheckpoint) openShardedSafeTensors(indexPath string) error {
	indexPath = filepath.Clean(indexPath)
	indexBytes, indexStat, document, weightMap, err := c.readSafeIndex(indexPath)
	if err != nil {
		return err
	}
	shardForTensor, paths, err := indexedShardPaths(c.ctx, indexPath, weightMap)
	if err != nil {
		return err
	}
	for _, path := range paths {
		if err := c.openIndexedShard(path, shardForTensor); err != nil {
			return err
		}
	}
	if len(c.tensors) != len(weightMap) {
		return fmt.Errorf("weights: index maps %d tensors but shards contain %d; one or more indexed tensor names are missing", len(weightMap), len(c.tensors))
	}
	c.sourcePaths = append([]string{indexPath}, paths...)
	c.fileSizes[indexPath] = indexStat.Size()
	c.sharded = true
	c.indexPath = indexPath
	c.index = &safeIndex{Document: document, WeightMap: weightMap, RawBytes: slices.Clone(indexBytes)}
	return nil
}

func (c *openedCheckpoint) readSafeIndex(indexPath string) (indexBytes []byte, indexStat os.FileInfo, document map[string]json.RawMessage, weightMap map[string]string, err error) {
	indexBytes, err = c.root.ReadFile(indexPath)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("weights: read SafeTensors index %q: %w", indexPath, err)
	}
	if len(indexBytes) == 0 || len(indexBytes) > maxSafeIndexBytes {
		return nil, nil, nil, nil, fmt.Errorf("weights: SafeTensors index size must be between 1 and %d bytes", maxSafeIndexBytes)
	}
	indexStat, err = c.root.Stat(indexPath)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("weights: stat SafeTensors index %q: %w", indexPath, err)
	}
	if !indexStat.Mode().IsRegular() || indexStat.Size() != int64(len(indexBytes)) {
		return nil, nil, nil, nil, fmt.Errorf("%w: SafeTensors index %q is not a stable regular file", ErrUnsupportedLayout, indexPath)
	}
	if err = safetensors.ValidateJSON(indexBytes); err != nil {
		return nil, nil, nil, nil, fmt.Errorf("weights: invalid SafeTensors index JSON: %w", err)
	}
	err = json.Unmarshal(indexBytes, &document)
	if err != nil || document == nil {
		if err == nil {
			err = errors.New("index must be a JSON object")
		}
		return nil, nil, nil, nil, fmt.Errorf("weights: decode SafeTensors index: %w", err)
	}
	weightRaw, exists := document["weight_map"]
	if !exists {
		return nil, nil, nil, nil, errors.New("weights: SafeTensors index is missing weight_map")
	}
	err = json.Unmarshal(weightRaw, &weightMap)
	if err != nil || weightMap == nil {
		if err == nil {
			err = errors.New("weight_map must be an object")
		}
		return nil, nil, nil, nil, fmt.Errorf("weights: decode SafeTensors weight_map: %w", err)
	}
	if len(weightMap) == 0 || len(weightMap) > maxBundleTensors {
		return nil, nil, nil, nil, fmt.Errorf("weights: indexed tensor count %d is outside the supported range 1..%d", len(weightMap), maxBundleTensors)
	}
	return indexBytes, indexStat, document, weightMap, nil
}

func indexedShardPaths(ctx context.Context, indexPath string, weightMap map[string]string) (shardForTensor map[string]string, paths []string, err error) {
	shardForTensor = make(map[string]string, len(weightMap))
	shards := make(map[string]struct{})
	indexDir := filepath.Dir(indexPath)
	for tensorName, shardName := range weightMap {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		if strings.TrimSpace(tensorName) == "" {
			return nil, nil, errors.New("weights: SafeTensors weight_map contains an empty tensor name")
		}
		if strings.TrimSpace(shardName) == "" || shardName == "." || shardName == ".." || filepath.Base(shardName) != shardName || strings.ContainsAny(shardName, `/\:`) || filepath.IsAbs(shardName) {
			return nil, nil, fmt.Errorf("weights: unsafe or non-sibling shard name %q in SafeTensors index", shardName)
		}
		if !strings.HasSuffix(strings.ToLower(shardName), ".safetensors") {
			return nil, nil, fmt.Errorf("weights: indexed shard %q does not use the .safetensors extension", shardName)
		}
		shardPath, err := xfs.Rel(filepath.Join(indexDir, shardName))
		if err != nil {
			return nil, nil, fmt.Errorf("weights: validate shard path %q: %w", shardName, err)
		}
		shardForTensor[tensorName] = shardPath
		shards[shardPath] = struct{}{}
	}
	if len(shards) == 0 || len(shards) > maxShardFiles {
		return nil, nil, fmt.Errorf("weights: referenced shard count %d is outside the supported range 1..%d", len(shards), maxShardFiles)
	}
	paths = make([]string, 0, len(shards))
	for path := range shards {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	return shardForTensor, paths, nil
}

func (c *openedCheckpoint) openIndexedShard(path string, shardForTensor map[string]string) error {
	if err := c.ctx.Err(); err != nil {
		return err
	}
	file, err := c.root.Open(path)
	if err != nil {
		return fmt.Errorf("weights: open indexed SafeTensors shard %q: %w", path, err)
	}
	stat, err := file.Stat()
	if err != nil {
		_ = file.Close()
		return fmt.Errorf("weights: stat indexed shard %q: %w", path, err)
	}
	if !stat.Mode().IsRegular() {
		_ = file.Close()
		return fmt.Errorf("%w: indexed shard %q is not a regular file", ErrUnsupportedLayout, path)
	}
	parsed, err := safetensors.OpenFile(file, path)
	if err != nil {
		return fmt.Errorf("weights: validate SafeTensors shard %q: %w", path, err)
	}
	for _, tensorName := range parsed.TensorNames() {
		if err := c.ctx.Err(); err != nil {
			_ = parsed.Close()
			return err
		}
		if len(c.tensors) >= maxBundleTensors {
			_ = parsed.Close()
			return fmt.Errorf("weights: combined shard tensor count exceeds limit %d", maxBundleTensors)
		}
		mappedPath, indexed := shardForTensor[tensorName]
		if !indexed || mappedPath != path {
			_ = parsed.Close()
			return fmt.Errorf("weights: shard %q contains tensor %q not mapped to that shard by weight_map", path, tensorName)
		}
		if _, duplicate := c.tensors[tensorName]; duplicate {
			_ = parsed.Close()
			return fmt.Errorf("weights: tensor %q appears in more than one SafeTensors shard", tensorName)
		}
		info := parsed.Tensors[tensorName]
		c.tensors[tensorName] = TensorInfo{Name: tensorName, DType: info.DType, Shape: slices.Clone(info.Shape), Elements: info.Elements, Bytes: info.DataOffsets[1] - info.DataOffsets[0]}
		c.tensorPaths[tensorName] = path
	}
	if err := parsed.Close(); err != nil {
		return fmt.Errorf("weights: close SafeTensors shard %q: %w", path, err)
	}
	c.fileSizes[path] = stat.Size()
	return nil
}
