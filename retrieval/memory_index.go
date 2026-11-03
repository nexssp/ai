package retrieval

import (
	"context"
	"math"
	"sort"
	"strings"
	"sync"

	"github.com/nexssp/kernel/xerr"
)

// MemoryIndex stores vectors in process memory. It is intended for tests and
// local demonstrations; its contents are lost on restart and it is not a
// persistent database adapter.
type MemoryIndex struct {
	mu      sync.RWMutex
	records map[string]map[string]VectorRecord
}

// NewMemoryIndex creates an empty in-memory vector index.
func NewMemoryIndex() *MemoryIndex {
	return &MemoryIndex{records: make(map[string]map[string]VectorRecord)}
}

// ReplaceDocument atomically replaces every stored chunk for one document.
func (i *MemoryIndex) ReplaceDocument(ctx context.Context, namespace, documentID string, records []VectorRecord) error {
	if err := ctx.Err(); err != nil {
		return xerr.From(err)
	}
	if err := validateIdentifier("namespace", namespace); err != nil {
		return err
	}
	if err := validateIdentifier("document ID", documentID); err != nil {
		return err
	}
	prepared, dimension, err := prepareDocumentRecords(namespace, documentID, records)
	if err != nil {
		return err
	}

	i.mu.Lock()
	defer i.mu.Unlock()
	current := i.records[namespace]
	if err := validateNamespaceDimensions(current, documentID, dimension); err != nil {
		return err
	}
	for chunkID, record := range current {
		if record.DocumentID == documentID {
			delete(current, chunkID)
		}
	}
	if len(prepared) > 0 {
		if current == nil {
			current = make(map[string]VectorRecord)
		}
		for _, record := range prepared {
			current[record.ChunkID] = record
		}
		i.records[namespace] = current
	} else if len(current) == 0 {
		delete(i.records, namespace)
	}
	return nil
}

func prepareDocumentRecords(namespace, documentID string, records []VectorRecord) ([]VectorRecord, int, error) {
	prepared := make([]VectorRecord, len(records))
	seen := make(map[string]struct{}, len(records))
	dimension := 0
	for index, record := range records {
		if record.Namespace != namespace || record.DocumentID != documentID {
			return nil, 0, xerr.Validation("retrieval: replacement record scope does not match the document")
		}
		if err := validateIdentifier("chunk ID", record.ChunkID); err != nil {
			return nil, 0, err
		}
		if strings.TrimSpace(record.Text) == "" {
			return nil, 0, xerr.Validation("retrieval: indexed chunk text is required")
		}
		if err := validateVector(record.Vector); err != nil {
			return nil, 0, err
		}
		if dimension != 0 && len(record.Vector) != dimension {
			return nil, 0, xerr.Validation("retrieval: replacement vectors have inconsistent dimensions")
		}
		dimension = len(record.Vector)
		if _, exists := seen[record.ChunkID]; exists {
			return nil, 0, xerr.Conflict("retrieval: duplicate chunk ID in document replacement")
		}
		seen[record.ChunkID] = struct{}{}
		prepared[index] = cloneRecord(record)
	}
	return prepared, dimension, nil
}

func validateNamespaceDimensions(current map[string]VectorRecord, documentID string, dimension int) error {
	if dimension == 0 {
		return nil
	}
	for _, record := range current {
		if record.DocumentID != documentID && len(record.Vector) != dimension {
			return xerr.Validation("retrieval: vector dimensions differ within a namespace")
		}
	}
	return nil
}

// DeleteDocument removes every chunk for one document in the namespace.
func (i *MemoryIndex) DeleteDocument(ctx context.Context, namespace, documentID string) error {
	if err := ctx.Err(); err != nil {
		return xerr.From(err)
	}
	if err := validateIdentifier("namespace", namespace); err != nil {
		return err
	}
	if err := validateIdentifier("document ID", documentID); err != nil {
		return err
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	for id, record := range i.records[namespace] {
		if record.DocumentID == documentID {
			delete(i.records[namespace], id)
		}
	}
	if len(i.records[namespace]) == 0 {
		delete(i.records, namespace)
	}
	return nil
}

// Search returns normalized cosine-similarity matches in descending order.
func (i *MemoryIndex) Search(ctx context.Context, query VectorQuery) ([]Match, error) {
	if err := ctx.Err(); err != nil {
		return nil, xerr.From(err)
	}
	if err := validateIdentifier("namespace", query.Namespace); err != nil {
		return nil, err
	}
	if err := validateVector(query.Vector); err != nil {
		return nil, err
	}
	if query.Limit <= 0 {
		return nil, xerr.Validation("retrieval: vector query limit must be positive")
	}
	if math.IsNaN(float64(query.MinScore)) || math.IsInf(float64(query.MinScore), 0) || query.MinScore < 0 || query.MinScore > 1 {
		return nil, xerr.Validation("retrieval: minimum score must be in [0, 1]")
	}

	i.mu.RLock()
	defer i.mu.RUnlock()
	records := i.records[query.Namespace]
	matches := make([]Match, 0, len(records))
	for _, record := range records {
		if len(record.Vector) != len(query.Vector) {
			return nil, xerr.Validation("retrieval: query vector dimension does not match indexed vectors")
		}
		score := normalizedCosine(query.Vector, record.Vector)
		if score < query.MinScore {
			continue
		}
		matches = append(matches, Match{
			Namespace:  record.Namespace,
			DocumentID: record.DocumentID,
			ChunkID:    record.ChunkID,
			Text:       record.Text,
			Metadata:   cloneMetadata(record.Metadata),
			Score:      score,
		})
	}
	sort.Slice(matches, func(a, b int) bool {
		if matches[a].Score != matches[b].Score {
			return matches[a].Score > matches[b].Score
		}
		if matches[a].DocumentID != matches[b].DocumentID {
			return matches[a].DocumentID < matches[b].DocumentID
		}
		return matches[a].ChunkID < matches[b].ChunkID
	})
	if len(matches) > query.Limit {
		matches = matches[:query.Limit]
	}
	return matches, nil
}

func normalizedCosine(left, right []float32) float32 {
	dot, leftNorm, rightNorm := 0.0, 0.0, 0.0
	for index := range left {
		a, b := float64(left[index]), float64(right[index])
		dot += a * b
		leftNorm += a * a
		rightNorm += b * b
	}
	cosine := dot / (math.Sqrt(leftNorm) * math.Sqrt(rightNorm))
	if cosine > 1 {
		cosine = 1
	} else if cosine < -1 {
		cosine = -1
	}
	return float32((cosine + 1) / 2)
}

func cloneRecord(record VectorRecord) VectorRecord {
	record.Metadata = cloneMetadata(record.Metadata)
	record.Vector = append([]float32(nil), record.Vector...)
	return record
}

var _ VectorIndex = (*MemoryIndex)(nil)
