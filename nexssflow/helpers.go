package nexssflow

import "maps"

// Micro-utilities shared across the bundle. Every function here is
// small enough that inlining beats any abstraction, and every one is
// used from at least two files. Keeping them in one place prevents
// six copies of `firstNonEmpty` from drifting.

// shallowCopy returns a fresh map with all top-level keys of in.
// Nested values are shared. Nil input yields an empty non-nil map.
func shallowCopy(in map[string]any) map[string]any {
	out := make(map[string]any, len(in)+8)
	maps.Copy(out, in)
	return out
}

// stringOf returns v as string, or "" when the type does not match.
func stringOf(v any) string {
	s, _ := v.(string)
	return s
}

// intOf coerces v to int for JSON-supplied numbers (float64), native
// ints, and int64. Anything else yields 0.
func intOf(v any) int {
	switch x := v.(type) {
	case int:
		return x
	case int64:
		return int(x)
	case float64:
		return int(x)
	}
	return 0
}

// floatOf coerces v to float64 for JSON-supplied numbers (float64) and
// native ints. Anything else yields 0.
func floatOf(v any) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case int:
		return float64(x)
	case int64:
		return float64(x)
	}
	return 0
}

// firstNonEmpty returns the first non-empty string in values, or "".
func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}
