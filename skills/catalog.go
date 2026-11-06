// Package skills loads host-approved prompt instructions and stores proposed
// skills as inert drafts. It never executes skill content.
package skills

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/nexssp/kernel/xerr"
)

const (
	defaultMaxSkillBytes = 64 << 10
	defaultMaxDraftBytes = 64 << 10
	defaultMaxSkills     = 128
)

// Skill is a validated instruction document loaded from the host-approved directory.
type Skill struct {
	Name         string `json:"name"`
	Instructions string `json:"instructions"`
}

// Config points to a host-managed approved directory and, optionally, a separate
// directory for inert proposals. DraftDirectory must not be inside Directory.
type Config struct {
	Directory      string
	DraftDirectory string
	MaxSkillBytes  int64
	MaxDraftBytes  int64
	MaxSkills      int
}

// DraftReceipt identifies a validated proposal written outside the active catalog.
type DraftReceipt struct {
	Name string `json:"name"`
	File string `json:"file"`
}

// Catalog is an immutable snapshot of approved skills plus a write-only draft target.
type Catalog struct {
	approved    []Skill
	byName      map[string]Skill
	draftDir    string
	maxDraftLen int64
}

// New loads all top-level .md files from the configured approved directory.
// Files are treated as plain text; symlinks, nested files, executable hooks, and
// non-Markdown files are not loaded. The draft directory, if set, is separate.
func New(config Config) (*Catalog, error) {
	if config.MaxSkillBytes < 0 || config.MaxDraftBytes < 0 || config.MaxSkills < 0 {
		return nil, xerr.Validation("skills: configured limits cannot be negative")
	}
	if config.MaxSkillBytes == 0 {
		config.MaxSkillBytes = defaultMaxSkillBytes
	}
	if config.MaxDraftBytes == 0 {
		config.MaxDraftBytes = defaultMaxDraftBytes
	}
	if config.MaxSkills == 0 {
		config.MaxSkills = defaultMaxSkills
	}
	activeDir, err := approvedDirectory(config.Directory)
	if err != nil {
		return nil, err
	}

	approved, byName, err := loadApproved(activeDir, config.MaxSkillBytes, config.MaxSkills)
	if err != nil {
		return nil, err
	}
	draftDir, err := prepareDraftDirectory(activeDir, config.DraftDirectory)
	if err != nil {
		return nil, err
	}
	return &Catalog{approved: approved, byName: byName, draftDir: draftDir, maxDraftLen: config.MaxDraftBytes}, nil
}

func approvedDirectory(path string) (string, error) {
	if strings.TrimSpace(path) == "" {
		return "", xerr.Validation("skills: approved directory is required")
	}
	dir, err := filepath.Abs(strings.TrimSpace(path))
	if err != nil {
		return "", xerr.Validation("skills: resolve approved directory", err)
	}
	info, err := os.Lstat(dir)
	if err != nil {
		return "", xerr.Unavailable("skills: inspect approved directory", err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return "", xerr.Validation("skills: approved path must be a directory, not a symlink")
	}
	resolved, err := filepath.EvalSymlinks(dir)
	if err != nil {
		return "", xerr.Unavailable("skills: resolve approved directory", err)
	}
	return resolved, nil
}

func prepareDraftDirectory(activeDir, path string) (string, error) {
	if strings.TrimSpace(path) == "" {
		return "", nil
	}
	dir, err := filepath.Abs(strings.TrimSpace(path))
	if err != nil {
		return "", xerr.Validation("skills: resolve draft directory", err)
	}
	if sameOrWithin(activeDir, dir) || sameOrWithin(dir, activeDir) {
		return "", xerr.Validation("skills: draft and approved directories must be separate")
	}
	if mkdirErr := os.MkdirAll(dir, 0o700); mkdirErr != nil {
		return "", xerr.Unavailable("skills: create draft directory", mkdirErr)
	}
	info, err := os.Lstat(dir)
	if err != nil {
		return "", xerr.Unavailable("skills: inspect draft directory", err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return "", xerr.Validation("skills: draft path must be a directory, not a symlink")
	}
	resolved, err := filepath.EvalSymlinks(dir)
	if err != nil {
		return "", xerr.Unavailable("skills: resolve draft directory", err)
	}
	if sameOrWithin(activeDir, resolved) || sameOrWithin(resolved, activeDir) {
		_ = os.Remove(dir)
		return "", xerr.Validation("skills: draft and approved directories must be separate")
	}
	return resolved, nil
}

// List returns a copy of the approved catalog in deterministic name order.
func (c *Catalog) List() []Skill {
	return append([]Skill(nil), c.approved...)
}

// Get returns one host-approved skill by its exact name.
func (c *Catalog) Get(name string) (Skill, error) {
	if !validName(name) {
		return Skill{}, xerr.Validation("skills: name must be a lowercase identifier")
	}
	skill, exists := c.byName[name]
	if !exists {
		return Skill{}, xerr.NotFound(fmt.Sprintf("skills: approved skill %q not found", name))
	}
	return skill, nil
}

// CreateDraft validates and stores a proposal as JSON in the separate draft
// directory. Drafts are not loaded, activated, or executable through this API.
func (c *Catalog) CreateDraft(skill Skill) (DraftReceipt, error) {
	if c == nil || c.draftDir == "" {
		return DraftReceipt{}, xerr.Validation("skills: draft directory is not configured")
	}
	if err := validateSkill(skill, c.maxDraftLen); err != nil {
		return DraftReceipt{}, err
	}
	data, err := json.MarshalIndent(skill, "", "  ")
	if err != nil {
		return DraftReceipt{}, xerr.Internal("skills: encode draft", err)
	}
	if int64(len(data)) > c.maxDraftLen {
		return DraftReceipt{}, xerr.Validation("skills: encoded draft exceeds the configured size limit")
	}
	var random [12]byte
	if _, randErr := rand.Read(random[:]); randErr != nil {
		return DraftReceipt{}, xerr.Internal("skills: create draft ID", randErr)
	}
	name := skill.Name + "-" + hex.EncodeToString(random[:]) + ".draft.json"
	path := filepath.Join(c.draftDir, name)
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return DraftReceipt{}, xerr.Unavailable("skills: create inert draft", err)
	}
	if _, err := file.Write(data); err != nil {
		_ = file.Close()
		_ = os.Remove(path)
		return DraftReceipt{}, xerr.Unavailable("skills: write inert draft", err)
	}
	if err := file.Close(); err != nil {
		_ = os.Remove(path)
		return DraftReceipt{}, xerr.Unavailable("skills: close inert draft", err)
	}
	return DraftReceipt{Name: skill.Name, File: name}, nil
}

func loadApproved(dir string, maxBytes int64, maxSkills int) ([]Skill, map[string]Skill, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, nil, xerr.Unavailable("skills: read approved directory", err)
	}
	approved := make([]Skill, 0, len(entries))
	byName := make(map[string]Skill, len(entries))
	for _, entry := range entries {
		if !strings.HasSuffix(entry.Name(), ".md") {
			continue
		}
		if len(approved) >= maxSkills {
			return nil, nil, xerr.Validation("skills: approved directory exceeds the configured skill count limit")
		}
		info, err := entry.Info()
		if err != nil {
			return nil, nil, xerr.Unavailable("skills: inspect approved file "+entry.Name(), err)
		}
		if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
			return nil, nil, xerr.Validation("skills: approved Markdown entries must be regular files")
		}
		if info.Size() > maxBytes {
			return nil, nil, xerr.Validation("skills: approved file exceeds the configured size limit")
		}
		name := strings.TrimSuffix(entry.Name(), ".md")
		if !validName(name) {
			return nil, nil, xerr.Validation("skills: approved filename must be a lowercase identifier with .md extension")
		}
		contents, err := os.ReadFile(filepath.Join(dir, entry.Name()))
		if err != nil {
			return nil, nil, xerr.Unavailable("skills: read approved file "+entry.Name(), err)
		}
		skill := Skill{Name: name, Instructions: string(contents)}
		if err := validateSkill(skill, maxBytes); err != nil {
			return nil, nil, fmt.Errorf("skills: validate %s: %w", entry.Name(), err)
		}
		approved = append(approved, skill)
		byName[name] = skill
	}
	sort.Slice(approved, func(i, j int) bool { return approved[i].Name < approved[j].Name })
	return approved, byName, nil
}

func validateSkill(skill Skill, maxBytes int64) error {
	if !validName(skill.Name) {
		return xerr.Validation("skills: name must be a lowercase identifier of at most 64 characters")
	}
	if !utf8.ValidString(skill.Instructions) || strings.ContainsRune(skill.Instructions, '\x00') {
		return xerr.Validation("skills: instructions must be valid UTF-8 text without NUL characters")
	}
	if strings.TrimSpace(skill.Instructions) == "" {
		return xerr.Validation("skills: instructions are required")
	}
	if int64(len(skill.Instructions)) > maxBytes {
		return xerr.Validation("skills: instructions exceed the configured size limit")
	}
	return nil
}

func validName(name string) bool {
	if name == "" || len(name) > 64 || name[0] < 'a' || name[0] > 'z' {
		return false
	}
	for i := 1; i < len(name); i++ {
		char := name[i]
		if char != '-' && (char < 'a' || char > 'z') && (char < '0' || char > '9') {
			return false
		}
	}
	return true
}

func sameOrWithin(root, path string) bool {
	rel, err := filepath.Rel(root, path)
	if err != nil {
		return true
	}
	return rel == "." || rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}
