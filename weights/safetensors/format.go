// Package safetensors reads and writes SafeTensors files without Python or framework dependencies.
//
// Typical use:
//
//	file, err := safetensors.Open("model.safetensors")
//	if err != nil { return err }
//	defer file.Close()
//	meta := file.Tensors["layer.weight"]
package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
)

const MaxHeaderBytes = 100 << 20

const maxReadFloatPrefix = 128

// ValidateJSON checks that data is one valid JSON value with no duplicate object keys.
// It is useful before decoding untrusted plans, because encoding/json otherwise keeps
// the last occurrence of a duplicate field.
func ValidateJSON(data []byte) error {
	return rejectDuplicateKeys(bytes.TrimSpace(data))
}

type TensorInfo struct {
	DType       string   `json:"dtype"`
	Shape       []uint64 `json:"shape"`
	DataOffsets []uint64 `json:"data_offsets"`
	Elements    uint64   `json:"-"`
}

type File struct {
	Path      string
	Metadata  map[string]string
	Tensors   map[string]TensorInfo
	DataStart int64
	DataBytes uint64
	file      *os.File
	fileSize  int64
}

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

func checkedFileOffset(base int64, relative uint64) (int64, error) {
	if base < 0 {
		return 0, errors.New("file offset exceeds supported range")
	}
	baseUnsigned, err := uint64FromNonNegativeInt64(base)
	if err != nil || relative > uint64(math.MaxInt64)-baseUnsigned {
		return 0, errors.New("file offset exceeds supported range")
	}
	relativeInt, err := int64FromUint64(relative)
	if err != nil {
		return 0, err
	}
	return base + relativeInt, nil
}

// Open parses and validates the SafeTensors header and all tensor ranges. It does not
// load the tensor buffer into memory; weight reads use ReaderAt.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open safetensors file: %w", err)
	}
	return OpenFile(f, path)
}

// OpenFile parses a SafeTensors checkpoint from an already-open file. OpenFile
// takes ownership of f: close the returned File after use, and parse errors
// close f before returning.
func OpenFile(f *os.File, path string) (*File, error) {
	if f == nil {
		return nil, errors.New("SafeTensors file handle is required")
	}
	ok := false
	defer func() {
		if !ok {
			_ = f.Close()
		}
	}()
	st, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat safetensors file: %w", err)
	}
	header, err := readHeader(f, st.Size())
	if err != nil {
		return nil, err
	}
	raw, err := decodeHeader(header)
	if err != nil {
		return nil, err
	}
	fileSize, err := uint64FromNonNegativeInt64(st.Size())
	if err != nil {
		return nil, fmt.Errorf("invalid safetensors file size: %w", err)
	}
	headerBytes := uint64(len(header))
	dataStart, err := checkedFileOffset(8, headerBytes)
	if err != nil {
		return nil, fmt.Errorf("safetensors data start exceeds supported range: %w", err)
	}
	result := &File{
		Path:      path,
		Metadata:  map[string]string{},
		Tensors:   make(map[string]TensorInfo, len(raw)),
		DataStart: dataStart,
		DataBytes: fileSize - 8 - headerBytes,
		file:      f,
		fileSize:  st.Size(),
	}
	for name, value := range raw {
		if name == "__metadata__" {
			if err := json.Unmarshal(value, &result.Metadata); err != nil || result.Metadata == nil {
				if err == nil {
					err = errors.New("metadata must be an object of string values")
				}
				return nil, fmt.Errorf("decode __metadata__: %w", err)
			}
			continue
		}
		info, err := decodeTensorEntry(name, value, result.DataBytes)
		if err != nil {
			return nil, err
		}
		result.Tensors[name] = info
	}
	if err := validateCoverage(result.Tensors, result.DataBytes); err != nil {
		return nil, err
	}
	ok = true
	return result, nil
}

func readHeader(f *os.File, size int64) ([]byte, error) {
	if size < 8 {
		return nil, errors.New("safetensors file is shorter than the 8-byte header length")
	}
	fileSize, err := uint64FromNonNegativeInt64(size)
	if err != nil {
		return nil, fmt.Errorf("invalid safetensors file size: %w", err)
	}
	var prefix [8]byte
	if _, err := io.ReadFull(f, prefix[:]); err != nil {
		return nil, fmt.Errorf("read safetensors header length: %w", err)
	}
	headerBytes := binary.LittleEndian.Uint64(prefix[:])
	if headerBytes < 2 || headerBytes > MaxHeaderBytes {
		return nil, fmt.Errorf("safetensors header length %d is outside the supported range", headerBytes)
	}
	fileRemainder := fileSize - 8
	if headerBytes > fileRemainder {
		return nil, errors.New("safetensors header length exceeds file size")
	}
	if headerBytes > uint64(^uint(0)>>1) {
		return nil, errors.New("safetensors header length exceeds supported memory range")
	}
	header := make([]byte, int(headerBytes))
	if _, err := io.ReadFull(f, header); err != nil {
		return nil, fmt.Errorf("read safetensors header: %w", err)
	}
	return header, nil
}

func decodeHeader(header []byte) (map[string]json.RawMessage, error) {
	if header[0] != '{' {
		return nil, errors.New("safetensors header must begin with '{'")
	}
	trimmed := bytes.TrimRight(header, " \t\r\n")
	if err := rejectDuplicateKeys(trimmed); err != nil {
		return nil, fmt.Errorf("invalid safetensors header: %w", err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(trimmed, &raw); err != nil {
		return nil, fmt.Errorf("decode safetensors header: %w", err)
	}
	if raw == nil {
		return nil, errors.New("safetensors header must be a JSON object")
	}
	return raw, nil
}

func decodeTensorEntry(name string, value json.RawMessage, dataBytes uint64) (TensorInfo, error) {
	if strings.TrimSpace(name) == "" {
		return TensorInfo{}, errors.New("tensor name cannot be empty")
	}
	var info TensorInfo
	dec := json.NewDecoder(bytes.NewReader(value))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&info); err != nil {
		return TensorInfo{}, fmt.Errorf("decode tensor %q metadata: %w", name, err)
	}
	if info.DType == "" || info.DataOffsets == nil || len(info.DataOffsets) != 2 || info.Shape == nil {
		return TensorInfo{}, fmt.Errorf("tensor %q must define dtype, shape, and two data_offsets", name)
	}
	elementBytes, err := dtypeSize(info.DType)
	if err != nil {
		return TensorInfo{}, fmt.Errorf("tensor %q: %w", name, err)
	}
	info.Elements, err = elementCount(info.Shape)
	if err != nil {
		return TensorInfo{}, fmt.Errorf("tensor %q: %w", name, err)
	}
	begin, end := info.DataOffsets[0], info.DataOffsets[1]
	if end < begin || end > dataBytes {
		return TensorInfo{}, fmt.Errorf("tensor %q has invalid data_offsets [%d,%d] for a %d-byte data buffer", name, begin, end, dataBytes)
	}
	wantBytes, err := multiply(info.Elements, elementBytes)
	if err != nil {
		return TensorInfo{}, fmt.Errorf("tensor %q byte size: %w", name, err)
	}
	if end-begin != wantBytes {
		return TensorInfo{}, fmt.Errorf("tensor %q range has %d bytes but shape and dtype require %d", name, end-begin, wantBytes)
	}
	return info, nil
}

func (f *File) Close() error {
	if f == nil || f.file == nil {
		return nil
	}
	err := f.file.Close()
	f.file = nil
	return err
}

// Stat returns metadata for the opened checkpoint file.
func (f *File) Stat() (os.FileInfo, error) {
	if f == nil || f.file == nil {
		return nil, errors.New("SafeTensors file is closed")
	}
	return f.file.Stat()
}

// TensorNames returns tensor names in deterministic lexical order.
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
		return 0, errors.New("SafeTensors file is closed")
	}
	info, ok := f.Tensors[name]
	if !ok {
		return 0, fmt.Errorf("tensor %q does not exist", name)
	}
	if index >= info.Elements {
		return 0, fmt.Errorf("index %d is outside tensor %q with %d elements", index, name, info.Elements)
	}
	size, err := dtypeSize(info.DType)
	if err != nil {
		return 0, err
	}
	if !isEditableFloat(info.DType) {
		return 0, fmt.Errorf("tensor %q uses %s; only F16, BF16, F32, and F64 can be read as editable floats", name, info.DType)
	}
	var buf [8]byte
	if size > uint64(len(buf)) {
		return 0, errors.New("unsupported element byte size")
	}
	n := int(size)
	if f.DataStart < 0 || uint64(f.DataStart) > math.MaxInt64 || info.DataOffsets[0] > uint64(math.MaxInt64)-uint64(f.DataStart) || index > (uint64(math.MaxInt64)-uint64(f.DataStart)-info.DataOffsets[0])/size {
		return 0, errors.New("tensor byte offset exceeds supported file range")
	}
	relativeOffset := info.DataOffsets[0] + index*size
	if relativeOffset > uint64(math.MaxInt64)-uint64(f.DataStart) {
		return 0, errors.New("tensor byte offset exceeds supported file range")
	}
	offset, err := checkedFileOffset(f.DataStart, relativeOffset)
	if err != nil {
		return 0, err
	}
	if _, err := f.file.ReadAt(buf[:n], offset); err != nil {
		return 0, fmt.Errorf("read tensor %q index %d: %w", name, index, err)
	}
	value := decodeFloat(info.DType, buf[:n])
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return 0, fmt.Errorf("tensor %q index %d contains a non-finite value", name, index)
	}
	return value, nil
}

func (f *File) ReadFloatPrefix(name string, count uint64) ([]float64, error) {
	if f == nil || f.file == nil {
		return nil, errors.New("SafeTensors file is closed")
	}
	info, ok := f.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q does not exist", name)
	}
	if !isEditableFloat(info.DType) {
		return nil, fmt.Errorf("tensor %q uses %s; sample display supports F16, BF16, F32, and F64", name, info.DType)
	}
	if count > info.Elements {
		count = info.Elements
	}
	if count > maxReadFloatPrefix {
		count = maxReadFloatPrefix
	}
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

func dtypeSize(dtype string) (uint64, error) {
	switch dtype {
	case "BOOL", "U8", "I8", "F8_E4M3", "F8_E4M3FN", "F8_E5M2", "F8_E5M2FNUZ":
		return 1, nil
	case "F16", "BF16", "U16", "I16":
		return 2, nil
	case "F32", "U32", "I32":
		return 4, nil
	case "F64", "U64", "I64":
		return 8, nil
	default:
		return 0, fmt.Errorf("unsupported SafeTensors dtype %q", dtype)
	}
}

func elementCount(shape []uint64) (uint64, error) {
	count := uint64(1) // A rank-0 shape is a scalar.
	for _, dim := range shape {
		if dim == 0 {
			return 0, nil
		}
		var err error
		count, err = multiply(count, dim)
		if err != nil {
			return 0, err
		}
	}
	return count, nil
}

func multiply(a, b uint64) (uint64, error) {
	if b != 0 && a > math.MaxUint64/b {
		return 0, errors.New("tensor dimensions overflow uint64")
	}
	return a * b, nil
}

func isEditableFloat(dtype string) bool {
	return dtype == "F16" || dtype == "BF16" || dtype == "F32" || dtype == "F64"
}

func validateCoverage(tensors map[string]TensorInfo, bodyBytes uint64) error {
	type span struct{ begin, end uint64 }
	spans := make([]span, 0, len(tensors))
	for _, info := range tensors {
		begin, end := info.DataOffsets[0], info.DataOffsets[1]
		if begin != end {
			spans = append(spans, span{begin, end})
		}
	}
	sort.Slice(spans, func(i, j int) bool {
		if spans[i].begin == spans[j].begin {
			return spans[i].end < spans[j].end
		}
		return spans[i].begin < spans[j].begin
	})
	cursor := uint64(0)
	for _, item := range spans {
		if item.begin != cursor {
			return fmt.Errorf("safetensors data buffer has an overlap or hole at byte %d", cursor)
		}
		cursor = item.end
	}
	if cursor != bodyBytes {
		return fmt.Errorf("safetensors data buffer has %d unindexed trailing bytes", bodyBytes-cursor)
	}
	return nil
}

func rejectDuplicateKeys(raw []byte) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	if err := walkJSON(dec); err != nil {
		return err
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("extra JSON after header")
		}
		return err
	}
	return nil
}

func walkJSON(dec *json.Decoder) error {
	token, err := dec.Token()
	if err != nil {
		return err
	}
	delim, isDelim := token.(json.Delim)
	if !isDelim {
		return nil
	}
	switch delim {
	case '{':
		seen := make(map[string]struct{})
		for dec.More() {
			keyToken, err := dec.Token()
			if err != nil {
				return err
			}
			key, ok := keyToken.(string)
			if !ok {
				return errors.New("object key is not a string")
			}
			if _, exists := seen[key]; exists {
				return fmt.Errorf("duplicate object key %q", key)
			}
			seen[key] = struct{}{}
			if err := walkJSON(dec); err != nil {
				return err
			}
		}
		_, err := dec.Token()
		return err
	case '[':
		for dec.More() {
			if err := walkJSON(dec); err != nil {
				return err
			}
		}
		_, err := dec.Token()
		return err
	default:
		return fmt.Errorf("unexpected JSON delimiter %q", delim)
	}
}

func decodeFloat(dtype string, b []byte) float64 {
	switch dtype {
	case "F16":
		return float64(halfToFloat32(binary.LittleEndian.Uint16(b)))
	case "BF16":
		bits := uint32(binary.LittleEndian.Uint16(b)) << 16
		return float64(math.Float32frombits(bits))
	case "F32":
		return float64(math.Float32frombits(binary.LittleEndian.Uint32(b)))
	case "F64":
		return math.Float64frombits(binary.LittleEndian.Uint64(b))
	default:
		return math.NaN()
	}
}

func encodeFloat(dtype string, value float64) (encoded []byte, decoded float64, err error) {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return nil, 0, errors.New("weight values and deltas must remain finite")
	}
	switch dtype {
	case "F16":
		h := float32ToHalf(float32(value))
		decoded := halfToFloat32(h)
		if math.IsInf(float64(decoded), 0) || math.IsNaN(float64(decoded)) {
			return nil, 0, errors.New("value overflows F16")
		}
		b := make([]byte, 2)
		binary.LittleEndian.PutUint16(b, h)
		return b, float64(decoded), nil
	case "BF16":
		f := float32(value)
		bits := math.Float32bits(f)
		// Round to nearest, ties to even before truncating the low 16 bits.
		bits += 0x7fff + ((bits >> 16) & 1)
		h := uint16(bits >> 16)
		decoded := math.Float32frombits(uint32(h) << 16)
		if math.IsInf(float64(decoded), 0) || math.IsNaN(float64(decoded)) {
			return nil, 0, errors.New("value overflows BF16")
		}
		b := make([]byte, 2)
		binary.LittleEndian.PutUint16(b, h)
		return b, float64(decoded), nil
	case "F32":
		f := float32(value)
		if math.IsInf(float64(f), 0) || math.IsNaN(float64(f)) {
			return nil, 0, errors.New("value overflows F32")
		}
		b := make([]byte, 4)
		binary.LittleEndian.PutUint32(b, math.Float32bits(f))
		return b, float64(f), nil
	case "F64":
		b := make([]byte, 8)
		binary.LittleEndian.PutUint64(b, math.Float64bits(value))
		return b, value, nil
	default:
		return nil, 0, fmt.Errorf("dtype %s is not editable", dtype)
	}
}

func halfToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := (h >> 10) & 0x1f
	mantissa := uint32(h & 0x03ff)
	var bits uint32
	switch exp {
	case 0:
		if mantissa == 0 {
			bits = sign
		} else {
			exponent := int32(-14)
			for mantissa&0x0400 == 0 {
				mantissa <<= 1
				exponent--
			}
			mantissa &= 0x03ff
			bits = sign | uint32(exponent+127)<<23 | mantissa<<13
		}
	case 0x1f:
		bits = sign | 0x7f800000 | mantissa<<13
	default:
		bits = sign | uint32(int32(exp)-15+127)<<23 | mantissa<<13
	}
	return math.Float32frombits(bits)
}

func float32ToHalf(value float32) uint16 {
	bits := math.Float32bits(value)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xff) - 127 + 15
	mant := bits & 0x7fffff
	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint(14 - exp)
		base := mant >> shift
		remainder := mant & ((uint32(1) << shift) - 1)
		halfway := uint32(1) << (shift - 1)
		if remainder > halfway || (remainder == halfway && base&1 != 0) {
			base++
		}
		return sign | uint16(base&0xffff)
	}
	if exp >= 31 {
		return sign | 0x7c00
	}
	base := mant >> 13
	remainder := mant & 0x1fff
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
