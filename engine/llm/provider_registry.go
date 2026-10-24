package llm

import "sort"

// IsKnownProvider reports whether a provider name is present in the
// preset catalogue. It is used by the @llm directive to reject
// misspelled provider names at compile time.
func IsKnownProvider(name string) bool {
	_, ok := Presets[name]
	if ok {
		return true
	}
	// A few additional canonical names map to the same backends but are
	// not separate presets.
	switch name {
	case "anthropic", "claude", "gemini", "google":
		return true
	}
	return false
}

// KnownProviders returns every provider name accepted by IsKnownProvider,
// sorted for stable error messages.
func KnownProviders() []string {
	seen := map[string]bool{}
	for name := range Presets {
		seen[name] = true
	}
	for _, name := range []string{"anthropic", "claude", "gemini", "google"} {
		seen[name] = true
	}
	out := make([]string, 0, len(seen))
	for name := range seen {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}
