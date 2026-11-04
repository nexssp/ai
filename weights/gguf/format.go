// Package gguf provides bounded metadata parsing and copy-on-write scalar edits for GGUF files.
// It never dequantizes or treats quantized tensor blocks as floating-point values.
package gguf

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
	"unicode/utf8"
)

const (
	MaxHeaderBytes   = 100 << 20
	maxTensorCount   = 1 << 20
	maxMetadataCount = 1 << 20
	maxDimensions    = 4
	maxAlignment     = 1 << 20
	maxReadPrefix    = 128
)

func uint64FromNonNegativeInt64(value int64) (uint64, error) {
	if value < 0 {
		return 0, errors.New("negative file size or offset")
	}
	return uint64(value), nil
}

func int64FromUint64(value uint64) (int64, error) {
	if value > math.MaxInt64 {
		return 0, errors.New("file offset exceeds supported range")
	}
	return int64(value), nil
}

type TensorInfo struct {
	DType    string   `json:"dtype"`
	TypeID   uint32   `json:"-"`
	Shape    []uint64 `json:"shape"`
	Elements uint64   `json:"elements"`
	Bytes    uint64   `json:"bytes"`
	Offset   uint64   `json:"-"`
	DataPos  int64    `json:"-"`
}

type File struct {
	Path       string
	Version    uint32
	Alignment  uint32
	SplitNo    uint64
	SplitCount uint64
	DataStart  int64
	Tensors    map[string]TensorInfo
	file       *os.File
	fileSize   int64
}

type ggufMetadata struct {
	alignment      uint32
	splitNo        uint64
	splitCount     uint64
	splitNoSeen    bool
	splitCountSeen bool
}

type Edit struct {
	Tensor        string
	Index         uint64
	Delta         float64
	ExpectedValue *float64
}

type AppliedEdit struct {
	Tensor string  `json:"tensor"`
	Index  uint64  `json:"index"`
	Before float64 `json:"before"`
	After  float64 `json:"after"`
}

type encoding struct {
	name       string
	blockElems uint64
	blockBytes uint64
	editable   bool
}

// typeEncodings lists supported GGML storage layouts. Quantized types have exact
// block sizes for bounds checking and byte preservation, but are not editable.
// IDs follow the public GGML type enum; unknown future IDs fail closed.
var typeEncodings = map[uint32]encoding{
	0: {"F32", 1, 4, true}, 1: {"F16", 1, 2, true},
	2: {"Q4_0", 32, 18, false}, 3: {"Q4_1", 32, 20, false},
	6: {"Q5_0", 32, 22, false}, 7: {"Q5_1", 32, 24, false},
	8: {"Q8_0", 32, 34, false}, 9: {"Q8_1", 32, 36, false},
	10: {"Q2_K", 256, 84, false}, 11: {"Q3_K", 256, 110, false},
	12: {"Q4_K", 256, 144, false}, 13: {"Q5_K", 256, 176, false},
	14: {"Q6_K", 256, 210, false}, 15: {"Q8_K", 256, 292, false},
	16: {"IQ2_XXS", 256, 66, false}, 17: {"IQ2_XS", 256, 74, false},
	18: {"IQ3_XXS", 256, 98, false}, 19: {"IQ1_S", 256, 50, false},
	20: {"IQ4_NL", 32, 18, false}, 21: {"IQ3_S", 256, 110, false},
	22: {"IQ2_S", 256, 82, false}, 23: {"IQ4_XS", 256, 136, false},
	24: {"I8", 1, 1, false}, 25: {"I16", 1, 2, false},
	26: {"I32", 1, 4, false}, 27: {"I64", 1, 8, false},
	28: {"F64", 1, 8, true}, 29: {"IQ1_M", 256, 56, false},
	30: {"BF16", 1, 2, true},
	31: {"Q4_0_4_4", 32, 18, false}, 32: {"Q4_0_4_8", 32, 18, false},
	33: {"Q4_0_8_8", 32, 18, false}, 34: {"TQ1_0", 256, 54, false},
	35: {"TQ2_0", 256, 66, false}, 36: {"IQ4_NL_4_4", 32, 18, false},
	37: {"IQ4_NL_4_8", 32, 18, false}, 38: {"IQ4_NL_8_8", 32, 18, false},
	39: {"MXFP4", 32, 17, false}, 40: {"NVFP4", 64, 36, false},
	41: {"Q1_0", 128, 18, false}, 42: {"Q2_0", 64, 18, false},
}

// OpenFile takes ownership of f. It supports little-endian GGUF v2 and v3 and
// validates metadata bounds, tensor names, dimensions, encoded byte ranges,
// overlap, alignment, and supported type layouts without loading tensor buffers.
func OpenFile(ctx context.Context, f *os.File, path string) (*File, error) {
	if ctx == nil {
		return nil, errors.New("GGUF parsing context is required")
	}
	if f == nil {
		return nil, errors.New("GGUF file handle is required")
	}
	ok := false
	defer func() {
		if !ok {
			_ = f.Close()
		}
	}()
	st, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat GGUF file: %w", err)
	}
	if !st.Mode().IsRegular() {
		return nil, errors.New("GGUF input must be a regular file")
	}
	if st.Size() < 24 {
		return nil, errors.New("GGUF file is shorter than its fixed header")
	}
	fileSize, err := uint64FromNonNegativeInt64(st.Size())
	if err != nil {
		return nil, fmt.Errorf("invalid GGUF file size: %w", err)
	}
	c := cursor{ctx: ctx, r: f, size: st.Size()}
	version, tensorCount, metadataCount, err := readGGUFHeader(&c)
	if err != nil {
		return nil, err
	}
	c.headerLimit = min(int64(MaxHeaderBytes), st.Size())
	metadata, err := parseGGUFMetadata(ctx, &c, metadataCount)
	if err != nil {
		return nil, err
	}

	infos, err := parseGGUFTensors(ctx, &c, tensorCount)
	if err != nil {
		return nil, err
	}
	dataStart, err := validateGGUFDataRanges(c.pos, fileSize, metadata.alignment, infos)
	if err != nil {
		return nil, err
	}
	if metadata.splitNoSeen != metadata.splitCountSeen || (metadata.splitCountSeen && (metadata.splitCount < 2 || metadata.splitNo >= metadata.splitCount)) {
		return nil, fmt.Errorf("invalid GGUF split metadata split.no=%d split.count=%d", metadata.splitNo, metadata.splitCount)
	}
	dataStartInt, err := int64FromUint64(dataStart)
	if err != nil {
		return nil, errors.New("GGUF tensor data start exceeds platform limit")
	}
	ok = true
	return &File{Path: path, Version: version, Alignment: metadata.alignment, SplitNo: metadata.splitNo, SplitCount: metadata.splitCount, DataStart: dataStartInt, Tensors: infos, file: f, fileSize: st.Size()}, nil
}

func readGGUFHeader(c *cursor) (version uint32, tensorCount, metadataCount uint64, returnErr error) {
	magic, err := c.readBytes(4)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("read GGUF magic: %w", err)
	}
	if string(magic) != "GGUF" {
		if string(magic) == "FUGG" {
			return 0, 0, 0, errors.New("big-endian GGUF is not supported; only little-endian files are accepted")
		}
		return 0, 0, 0, errors.New("invalid GGUF magic")
	}
	version, err = c.u32()
	if err != nil {
		return 0, 0, 0, fmt.Errorf("read GGUF version: %w", err)
	}
	if version != 2 && version != 3 {
		return 0, 0, 0, fmt.Errorf("unsupported GGUF version %d; supported versions are 2 and 3", version)
	}
	tensorCount, err = c.u64()
	if err != nil {
		return 0, 0, 0, fmt.Errorf("read GGUF tensor count: %w", err)
	}
	metadataCount, err = c.u64()
	if err != nil {
		return 0, 0, 0, fmt.Errorf("read GGUF metadata count: %w", err)
	}
	if tensorCount == 0 || tensorCount > maxTensorCount {
		return 0, 0, 0, fmt.Errorf("GGUF tensor count %d is outside the supported range 1..%d", tensorCount, maxTensorCount)
	}
	if metadataCount > maxMetadataCount {
		return 0, 0, 0, fmt.Errorf("GGUF metadata count %d exceeds the limit %d", metadataCount, maxMetadataCount)
	}
	return version, tensorCount, metadataCount, nil
}

func parseGGUFMetadata(ctx context.Context, c *cursor, count uint64) (state ggufMetadata, returnErr error) {
	state.alignment = 32
	seen := make(map[string]struct{}, 4096)
	for i := range count {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ggufMetadata{}, ctxErr
		}
		key, keyErr := c.string(65535)
		if keyErr != nil {
			return ggufMetadata{}, fmt.Errorf("read GGUF metadata key %d: %w", i, keyErr)
		}
		if !utf8.ValidString(key) || key == "" {
			return ggufMetadata{}, fmt.Errorf("GGUF metadata key %d is empty or invalid UTF-8", i)
		}
		if _, exists := seen[key]; exists {
			return ggufMetadata{}, fmt.Errorf("duplicate GGUF metadata key %q", key)
		}
		seen[key] = struct{}{}
		typeID, typeErr := c.u32()
		if typeErr != nil {
			return ggufMetadata{}, fmt.Errorf("read GGUF metadata type for %q: %w", key, typeErr)
		}
		if key == "general.alignment" {
			var value uint64
			var readErr error
			switch typeID {
			case 4:
				var v uint32
				v, readErr = c.u32()
				value = uint64(v)
			case 10:
				value, readErr = c.u64()
			default:
				return ggufMetadata{}, fmt.Errorf("GGUF general.alignment has unsupported metadata type %d", typeID)
			}
			if readErr != nil {
				return ggufMetadata{}, fmt.Errorf("read GGUF alignment: %w", readErr)
			}
			if value < 8 || value > maxAlignment || value%8 != 0 {
				return ggufMetadata{}, fmt.Errorf("GGUF alignment %d must be a multiple of 8 between 8 and %d", value, maxAlignment)
			}
			state.alignment = uint32(value)
			continue
		}
		if key == "split.no" || key == "split.count" {
			value, readErr := c.unsignedMetadata(typeID)
			if readErr != nil {
				return ggufMetadata{}, fmt.Errorf("read GGUF %s: %w", key, readErr)
			}
			if key == "split.no" {
				state.splitNo = value
				state.splitNoSeen = true
			} else {
				state.splitCount = value
				state.splitCountSeen = true
			}
			continue
		}
		if skipErr := c.skipValue(typeID, 0); skipErr != nil {
			return ggufMetadata{}, fmt.Errorf("skip GGUF metadata value %q: %w", key, skipErr)
		}
	}
	return state, nil
}

func parseGGUFTensors(ctx context.Context, c *cursor, count uint64) (infos map[string]TensorInfo, returnErr error) {
	infos = make(map[string]TensorInfo)
	for i := range count {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		name, nameErr := c.string(64)
		if nameErr != nil {
			return nil, fmt.Errorf("read GGUF tensor name %d: %w", i, nameErr)
		}
		if name == "" || !utf8.ValidString(name) {
			return nil, fmt.Errorf("GGUF tensor name %d is empty or invalid UTF-8", i)
		}
		if _, exists := infos[name]; exists {
			return nil, fmt.Errorf("duplicate GGUF tensor name %q", name)
		}
		nDim, dimErr := c.u32()
		if dimErr != nil {
			return nil, fmt.Errorf("read GGUF dimensions for %q: %w", name, dimErr)
		}
		if nDim == 0 || nDim > maxDimensions {
			return nil, fmt.Errorf("GGUF tensor %q has %d dimensions; supported range is 1..%d", name, nDim, maxDimensions)
		}
		shape := make([]uint64, nDim)
		for dim := range shape {
			value, readErr := c.u64()
			if readErr != nil {
				return nil, fmt.Errorf("read GGUF dimension %d for %q: %w", dim, name, readErr)
			}
			shape[dim] = value
		}
		typeID, typeErr := c.u32()
		if typeErr != nil {
			return nil, fmt.Errorf("read GGUF type for %q: %w", name, typeErr)
		}
		offset, offsetErr := c.u64()
		if offsetErr != nil {
			return nil, fmt.Errorf("read GGUF data offset for %q: %w", name, offsetErr)
		}
		typeInfo, exists := typeEncodings[typeID]
		if !exists {
			return nil, fmt.Errorf("GGUF tensor %q uses unknown GGML type id %d; upgrade the format adapter before editing", name, typeID)
		}
		elements, elementsErr := countElements(shape)
		if elementsErr != nil {
			return nil, fmt.Errorf("GGUF tensor %q: %w", name, elementsErr)
		}
		bytes, bytesErr := storageSize(typeInfo, shape)
		if bytesErr != nil {
			return nil, fmt.Errorf("GGUF tensor %q (%s): %w", name, typeInfo.name, bytesErr)
		}
		infos[name] = TensorInfo{
			DType: typeInfo.name, TypeID: typeID, Shape: shape, Elements: elements,
			Bytes: bytes, Offset: offset,
		}
	}
	return infos, nil
}

func validateGGUFDataRanges(headerEnd int64, fileSize uint64, alignment uint32, infos map[string]TensorInfo) (dataStart uint64, returnErr error) {
	if headerEnd > MaxHeaderBytes {
		return 0, fmt.Errorf("GGUF header and tensor directory exceed %d bytes", MaxHeaderBytes)
	}
	dataStart, err := align(uint64(max(headerEnd, 0)), uint64(alignment))
	if err != nil || dataStart > fileSize || dataStart > math.MaxInt64 {
		return 0, errors.New("GGUF tensor data start is outside the file")
	}
	spans := make([]TensorInfo, 0, len(infos))
	for name, info := range infos {
		if info.Offset%uint64(alignment) != 0 {
			return 0, fmt.Errorf("GGUF tensor %q offset %d is not aligned to %d bytes", name, info.Offset, alignment)
		}
		if info.Offset > fileSize-dataStart || info.Bytes > fileSize-dataStart-info.Offset || info.Offset > math.MaxInt64-dataStart {
			return 0, fmt.Errorf("GGUF tensor %q byte range exceeds the file", name)
		}
		dataPos, posErr := int64FromUint64(dataStart + info.Offset)
		if posErr != nil {
			return 0, errors.New("GGUF tensor position exceeds platform limit")
		}
		info.DataPos = dataPos
		infos[name] = info
		spans = append(spans, info)
	}
	sort.Slice(spans, func(i, j int) bool { return spans[i].Offset < spans[j].Offset })
	for i := 1; i < len(spans); i++ {
		previousEnd := spans[i-1].Offset + spans[i-1].Bytes
		if spans[i].Offset < previousEnd {
			return 0, fmt.Errorf("GGUF tensor ranges overlap at offset %d", spans[i].Offset)
		}
	}
	return dataStart, nil
}

func (f *File) Close() error {
	if f == nil || f.file == nil {
		return nil
	}
	err := f.file.Close()
	f.file = nil
	return err
}

func (f *File) Stat() (os.FileInfo, error) {
	if f == nil || f.file == nil {
		return nil, errors.New("GGUF file is closed")
	}
	return f.file.Stat()
}

func (f *File) TensorNames() []string {
	if f == nil {
		return nil
	}
	names := make([]string, 0, len(f.Tensors))
	for name := range f.Tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func (f *File) ReadFloat(name string, index uint64) (float64, error) {
	if f == nil || f.file == nil {
		return 0, errors.New("GGUF file is closed")
	}
	info, exists := f.Tensors[name]
	if !exists {
		return 0, fmt.Errorf("tensor %q does not exist", name)
	}
	if !editable(info.TypeID) {
		return 0, fmt.Errorf("tensor %q uses %s; only F16, BF16, F32, and F64 are editable", name, info.DType)
	}
	if index >= info.Elements {
		return 0, fmt.Errorf("index %d is outside tensor %q with %d elements", index, name, info.Elements)
	}
	size := dtypeBytes(info.TypeID)
	var raw [8]byte
	offset := index * uint64(uint(size))
	if offset > math.MaxInt64 || info.DataPos > math.MaxInt64-int64(offset) {
		return 0, errors.New("GGUF tensor position exceeds platform limit")
	}
	if _, err := f.file.ReadAt(raw[:size], info.DataPos+int64(offset)); err != nil {
		return 0, fmt.Errorf("read GGUF tensor %q index %d: %w", name, index, err)
	}
	value := decodeFloat(info.TypeID, raw[:size])
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return 0, fmt.Errorf("tensor %q index %d contains a non-finite value", name, index)
	}
	return value, nil
}

func (f *File) ReadFloatPrefix(name string, count uint64) ([]float64, error) {
	info, exists := f.Tensors[name]
	if !exists {
		return nil, fmt.Errorf("tensor %q does not exist", name)
	}
	if !editable(info.TypeID) {
		return nil, fmt.Errorf("tensor %q uses %s; samples are available only for F16, BF16, F32, and F64", name, info.DType)
	}
	count = min(count, info.Elements, maxReadPrefix)
	out := make([]float64, 0, int(count))
	for i := range count {
		value, err := f.ReadFloat(name, i)
		if err != nil {
			return nil, err
		}
		out = append(out, value)
	}
	return out, nil
}

// ApplyEdits modifies only the already-open staging file. Callers must copy the
// original to a temporary file, verify its source digest, and atomically commit
// that temporary file only after this method and output validation succeed.
func (f *File) ApplyEdits(ctx context.Context, edits []Edit) ([]AppliedEdit, error) {
	if ctx == nil || f == nil || f.file == nil {
		return nil, errors.New("context and open GGUF file are required")
	}
	if len(edits) == 0 {
		return nil, errors.New("at least one GGUF edit is required")
	}
	seen := make(map[string]struct{}, len(edits))
	prepared := make([]preparedEdit, 0, len(edits))
	for _, edit := range edits {
		editPlan, err := f.prepareEdit(ctx, edit, seen)
		if err != nil {
			return nil, err
		}
		prepared = append(prepared, editPlan)
	}
	for _, edit := range prepared {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if _, err := f.file.WriteAt(edit.bytes, edit.offset); err != nil {
			return nil, fmt.Errorf("write GGUF tensor value: %w", err)
		}
	}
	applied := make([]AppliedEdit, len(prepared))
	for i := range prepared {
		applied[i] = prepared[i].edit
	}
	return applied, nil
}

func (f *File) prepareEdit(ctx context.Context, edit Edit, seen map[string]struct{}) (preparedEdit, error) {
	if err := ctx.Err(); err != nil {
		return preparedEdit{}, err
	}
	if strings.TrimSpace(edit.Tensor) == "" {
		return preparedEdit{}, errors.New("edit tensor name is required")
	}
	if math.IsNaN(edit.Delta) || math.IsInf(edit.Delta, 0) {
		return preparedEdit{}, fmt.Errorf("edit for tensor %q has a non-finite delta", edit.Tensor)
	}
	info, exists := f.Tensors[edit.Tensor]
	if !exists {
		return preparedEdit{}, fmt.Errorf("tensor %q does not exist", edit.Tensor)
	}
	if !editable(info.TypeID) {
		return preparedEdit{}, fmt.Errorf("tensor %q uses %s; quantized and integer GGUF tensors are preserved but not editable", edit.Tensor, info.DType)
	}
	if edit.Index >= info.Elements {
		return preparedEdit{}, fmt.Errorf("index %d is outside tensor %q with %d elements", edit.Index, edit.Tensor, info.Elements)
	}
	key := fmt.Sprintf("%s\x00%d", edit.Tensor, edit.Index)
	if _, exists := seen[key]; exists {
		return preparedEdit{}, fmt.Errorf("duplicate edit for tensor %q index %d", edit.Tensor, edit.Index)
	}
	seen[key] = struct{}{}
	before, err := f.ReadFloat(edit.Tensor, edit.Index)
	if err != nil {
		return preparedEdit{}, err
	}
	if edit.ExpectedValue != nil {
		if math.IsNaN(*edit.ExpectedValue) || math.IsInf(*edit.ExpectedValue, 0) {
			return preparedEdit{}, errors.New("expected_value must be finite")
		}
		tolerance := 1e-6 * math.Max(1, math.Abs(*edit.ExpectedValue))
		if math.Abs(before-*edit.ExpectedValue) > tolerance {
			return preparedEdit{}, fmt.Errorf("stale weight at %s[%d]: expected %.9g, found %.9g", edit.Tensor, edit.Index, *edit.ExpectedValue, before)
		}
	}
	encoded, after, err := encodeFloat(info.TypeID, before+edit.Delta)
	if err != nil {
		return preparedEdit{}, fmt.Errorf("edit %s[%d]: %w", edit.Tensor, edit.Index, err)
	}
	byteIndex := edit.Index * uint64(len(encoded))
	if byteIndex > math.MaxInt64 || info.DataPos > math.MaxInt64-int64(byteIndex) {
		return preparedEdit{}, errors.New("GGUF edit position exceeds platform limit")
	}
	return preparedEdit{AppliedEdit{Tensor: edit.Tensor, Index: edit.Index, Before: before, After: after}, info.DataPos + int64(byteIndex), encoded}, nil
}

type preparedEdit struct {
	edit   AppliedEdit
	offset int64
	bytes  []byte
}

type cursor struct {
	ctx         context.Context
	r           io.ReaderAt
	size        int64
	pos         int64
	headerLimit int64
}

func (c *cursor) ensure(n uint64) error {
	if n > uint64(math.MaxInt64) || n > uint64(int(^uint(0)>>1)) || c.pos < 0 || c.pos > c.size || int64(n) > c.size-c.pos {
		return io.ErrUnexpectedEOF
	}
	if c.headerLimit > 0 && (c.pos > c.headerLimit || int64(n) > c.headerLimit-c.pos) {
		return fmt.Errorf("GGUF header exceeds the %d-byte parsing limit", MaxHeaderBytes)
	}
	return nil
}

func (c *cursor) readBytes(n uint64) ([]byte, error) {
	if err := c.ensure(n); err != nil {
		return nil, err
	}
	if n > uint64(int(^uint(0)>>1)) {
		return nil, errors.New("GGUF read size exceeds platform limit")
	}
	data := make([]byte, int(n))
	if _, err := c.r.ReadAt(data, c.pos); err != nil {
		return nil, err
	}
	c.pos += int64(n)
	return data, nil
}

func (c *cursor) skip(n uint64) error {
	if err := c.ensure(n); err != nil {
		return err
	}
	int64N, err := int64FromUint64(n)
	if err != nil {
		return err
	}
	c.pos += int64N
	return nil
}

func (c *cursor) u8() (byte, error) {
	b, err := c.readBytes(1)
	if err != nil {
		return 0, err
	}
	return b[0], nil
}

func (c *cursor) u16() (uint16, error) {
	b, err := c.readBytes(2)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint16(b), nil
}

func (c *cursor) u32() (uint32, error) {
	b, err := c.readBytes(4)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b), nil
}

func (c *cursor) u64() (uint64, error) {
	b, err := c.readBytes(8)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b), nil
}

func (c *cursor) string(maxBytes uint64) (string, error) {
	n, err := c.u64()
	if err != nil {
		return "", err
	}
	if n > maxBytes {
		return "", fmt.Errorf("string length %d exceeds limit %d", n, maxBytes)
	}
	data, err := c.readBytes(n)
	if err != nil {
		return "", err
	}
	if !utf8.Valid(data) {
		return "", errors.New("string is not valid UTF-8")
	}
	return string(data), nil
}

func (c *cursor) skipString() error {
	n, err := c.u64()
	if err != nil {
		return err
	}
	return c.skip(n)
}

func (c *cursor) skipValue(typeID uint32, depth int) error {
	if depth > 16 {
		return errors.New("nested GGUF metadata arrays exceed depth 16")
	}
	switch typeID {
	case 0, 1:
		return c.skip(1)
	case 2, 3:
		return c.skip(2)
	case 4, 5, 6:
		return c.skip(4)
	case 7:
		value, err := c.u8()
		if err != nil {
			return err
		}
		if value > 1 {
			return fmt.Errorf("invalid GGUF boolean value %d", value)
		}
		return nil
	case 8:
		return c.skipString()
	case 9:
		return c.skipArray(depth)
	case 10, 11, 12:
		return c.skip(8)
	default:
		return fmt.Errorf("unknown GGUF metadata value type %d", typeID)
	}
}

func (c *cursor) skipArray(depth int) error {
	elementType, err := c.u32()
	if err != nil {
		return err
	}
	count, err := c.u64()
	if err != nil {
		return err
	}
	if c.pos < 0 || c.size < c.pos {
		return io.ErrUnexpectedEOF
	}
	remaining, err := uint64FromNonNegativeInt64(c.size - c.pos)
	if err != nil {
		return err
	}
	if elementType > 12 || count > remaining {
		return errors.New("GGUF metadata array type or count is invalid")
	}
	for range count {
		if err := c.ctx.Err(); err != nil {
			return err
		}
		if err := c.skipValue(elementType, depth+1); err != nil {
			return err
		}
	}
	return nil
}

func (c *cursor) unsignedMetadata(typeID uint32) (uint64, error) {
	switch typeID {
	case 0:
		value, err := c.u8()
		return uint64(value), err
	case 2:
		value, err := c.u16()
		return uint64(value), err
	case 4:
		value, err := c.u32()
		return uint64(value), err
	case 10:
		return c.u64()
	default:
		return 0, fmt.Errorf("expected an unsigned integer metadata type, got %d", typeID)
	}
}

func countElements(shape []uint64) (uint64, error) {
	count := uint64(1)
	for _, dim := range shape {
		if dim == 0 {
			return 0, errors.New("zero-sized GGUF dimensions are not supported")
		}
		if count > math.MaxUint64/dim {
			return 0, errors.New("GGUF tensor dimensions overflow uint64")
		}
		count *= dim
	}
	return count, nil
}

func storageSize(info encoding, shape []uint64) (uint64, error) {
	if len(shape) == 0 || info.blockElems == 0 || info.blockBytes == 0 {
		return 0, errors.New("tensor type has no supported storage layout")
	}
	if shape[0]%info.blockElems != 0 {
		return 0, fmt.Errorf("first dimension %d is not divisible by the %d-element block size", shape[0], info.blockElems)
	}
	rows := uint64(1)
	for _, dim := range shape[1:] {
		if rows > math.MaxUint64/dim {
			return 0, errors.New("GGUF tensor byte size overflows uint64")
		}
		rows *= dim
	}
	blocks := shape[0] / info.blockElems
	if blocks > math.MaxUint64/info.blockBytes {
		return 0, errors.New("GGUF tensor byte size overflows uint64")
	}
	rowBytes := blocks * info.blockBytes
	if rowBytes != 0 && rows > math.MaxUint64/rowBytes {
		return 0, errors.New("GGUF tensor byte size overflows uint64")
	}
	return rowBytes * rows, nil
}

func align(n, alignment uint64) (uint64, error) {
	if alignment == 0 || n > math.MaxUint64-(alignment-1) {
		return 0, errors.New("GGUF alignment arithmetic overflow")
	}
	return ((n + alignment - 1) / alignment) * alignment, nil
}

func editable(typeID uint32) bool {
	typeInfo, ok := typeEncodings[typeID]
	return ok && typeInfo.editable
}

func dtypeBytes(typeID uint32) int {
	switch typeID {
	case 0:
		return 4
	case 1, 30:
		return 2
	case 28:
		return 8
	default:
		return 0
	}
}

func decodeFloat(typeID uint32, data []byte) float64 {
	switch typeID {
	case 0:
		return float64(math.Float32frombits(binary.LittleEndian.Uint32(data)))
	case 1:
		return float64(halfToFloat32(binary.LittleEndian.Uint16(data)))
	case 28:
		return math.Float64frombits(binary.LittleEndian.Uint64(data))
	case 30:
		return float64(math.Float32frombits(uint32(binary.LittleEndian.Uint16(data)) << 16))
	default:
		return math.NaN()
	}
}

func encodeFloat(typeID uint32, value float64) (encoded []byte, rounded float64, err error) {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return nil, 0, errors.New("weight values and deltas must remain finite")
	}
	switch typeID {
	case 1:
		half := float32ToHalf(float32(value))
		decoded := halfToFloat32(half)
		if math.IsNaN(float64(decoded)) || math.IsInf(float64(decoded), 0) {
			return nil, 0, errors.New("value overflows F16")
		}
		data := make([]byte, 2)
		binary.LittleEndian.PutUint16(data, half)
		return data, float64(decoded), nil
	case 30:
		f := float32(value)
		bits := math.Float32bits(f)
		bits += 0x7fff + ((bits >> 16) & 1)
		half := uint16(bits >> 16)
		decoded := math.Float32frombits(uint32(half) << 16)
		if math.IsNaN(float64(decoded)) || math.IsInf(float64(decoded), 0) {
			return nil, 0, errors.New("value overflows BF16")
		}
		data := make([]byte, 2)
		binary.LittleEndian.PutUint16(data, half)
		return data, float64(decoded), nil
	case 0:
		f := float32(value)
		if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
			return nil, 0, errors.New("value overflows F32")
		}
		data := make([]byte, 4)
		binary.LittleEndian.PutUint32(data, math.Float32bits(f))
		return data, float64(f), nil
	case 28:
		data := make([]byte, 8)
		binary.LittleEndian.PutUint64(data, math.Float64bits(value))
		return data, value, nil
	default:
		return nil, 0, fmt.Errorf("GGUF type id %d is not editable", typeID)
	}
}

func halfToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exponent := (h >> 10) & 0x1f
	mantissa := uint32(h & 0x03ff)
	var bits uint32
	switch exponent {
	case 0:
		if mantissa == 0 {
			bits = sign
		} else {
			exp := int32(-14)
			for mantissa&0x0400 == 0 {
				mantissa <<= 1
				exp--
			}
			mantissa &= 0x03ff
			bits = sign | uint32(exp+127)<<23 | mantissa<<13
		}
	case 0x1f:
		bits = sign | 0x7f800000 | mantissa<<13
	default:
		bits = sign | uint32(int32(exponent)-15+127)<<23 | mantissa<<13
	}
	return math.Float32frombits(bits)
}

func float32ToHalf(value float32) uint16 {
	bits := math.Float32bits(value)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xff) - 127 + 15
	mantissa := bits & 0x7fffff
	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		mantissa |= 0x800000
		shift := uint(14 - exp)
		base := mantissa >> shift
		remainder := mantissa & ((uint32(1) << shift) - 1)
		halfway := uint32(1) << (shift - 1)
		if remainder > halfway || (remainder == halfway && base&1 != 0) {
			base++
		}
		if base > math.MaxUint16 {
			return sign
		}
		return sign | uint16(base)
	}
	if exp >= 31 {
		return sign | 0x7c00
	}
	base := mantissa >> 13
	remainder := mantissa & 0x1fff
	if remainder > 0x1000 || (remainder == 0x1000 && base&1 != 0) {
		base++
		if base == 0x400 {
			base = 0
			exp++
			if exp >= 31 {
				return sign | 0x7c00
			}
		}
	}
	return sign | uint16(exp<<10) | uint16(base)
}
