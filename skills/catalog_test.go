package skills

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/nexssp/kernel/xtest/ktest"
)

func TestCatalogLoadsApprovedMarkdownAndWritesInertDraftOutsideIt(t *testing.T) {
	activeDir := t.TempDir()
	draftDir := filepath.Join(t.TempDir(), "drafts")
	for name, content := range map[string]string{
		"zeta.md":   "Use short paragraphs.\n",
		"alpha.md":  "Prefer concrete examples.\n",
		"notes.txt": "not a skill",
	} {
		if err := os.WriteFile(filepath.Join(activeDir, name), []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}

	catalog, err := New(Config{Directory: activeDir, DraftDirectory: draftDir})
	ktest.RequireNoError(t, err)
	listed := catalog.List()
	ktest.RequireEqual(t, len(listed), 2)
	ktest.RequireEqual(t, listed[0].Name, "alpha")
	ktest.RequireEqual(t, listed[1].Name, "zeta")

	got, err := catalog.Get("alpha")
	ktest.RequireNoError(t, err)
	ktest.RequireEqual(t, got.Instructions, "Prefer concrete examples.\n")

	receipt, err := catalog.CreateDraft(Skill{Name: "new-helper", Instructions: "Explain the steps, then stop."})
	ktest.RequireNoError(t, err)
	ktest.RequireEqual(t, receipt.Name, "new-helper")
	ktest.RequireCondition(t, filepath.Base(receipt.File) == receipt.File && filepath.Ext(receipt.File) == ".json", "draft receipt must contain only a JSON filename")

	draftPath := filepath.Join(draftDir, receipt.File)
	data, err := os.ReadFile(draftPath)
	ktest.RequireNoError(t, err)
	var draft Skill
	ktest.RequireNoError(t, json.Unmarshal(data, &draft))
	ktest.RequireEqual(t, draft, Skill{Name: "new-helper", Instructions: "Explain the steps, then stop."})
	info, err := os.Stat(draftPath)
	ktest.RequireNoError(t, err)
	ktest.RequireEqual(t, info.Mode().Perm(), os.FileMode(0o600))
	ktest.RequireEqual(t, len(catalog.List()), 2)
	if _, err := catalog.Get("new-helper"); err == nil {
		t.Fatal("draft became active without host review")
	}
}

func TestCatalogRejectsUnsafeOrInvalidSkillEntries(t *testing.T) {
	t.Run("symlink", func(t *testing.T) {
		activeDir := t.TempDir()
		target := filepath.Join(t.TempDir(), "outside.md")
		if err := os.WriteFile(target, []byte("outside instructions"), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(target, filepath.Join(activeDir, "linked.md")); err != nil {
			t.Skipf("symlink unavailable: %v", err)
		}
		if _, err := New(Config{Directory: activeDir}); err == nil {
			t.Fatal("catalog followed a Markdown symlink")
		}
	})

	for _, test := range []struct {
		name string
		file string
		data []byte
	}{
		{name: "invalid filename", file: "Bad.md", data: []byte("text")},
		{name: "NUL", file: "bad.md", data: []byte("hello\x00world")},
		{name: "empty", file: "empty.md", data: []byte(" \n")},
	} {
		t.Run(test.name, func(t *testing.T) {
			activeDir := t.TempDir()
			if err := os.WriteFile(filepath.Join(activeDir, test.file), test.data, 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := New(Config{Directory: activeDir}); err == nil {
				t.Fatal("invalid approved entry was accepted")
			}
		})
	}
}

func TestCatalogRequiresSeparateDraftDirectoryAndValidatesDrafts(t *testing.T) {
	activeDir := t.TempDir()
	if _, err := New(Config{Directory: activeDir, DraftDirectory: filepath.Join(activeDir, "drafts")}); err == nil {
		t.Fatal("draft directory inside approved directory was accepted")
	}
	t.Run("symlinked parent", func(t *testing.T) {
		active := t.TempDir()
		aliasParent := t.TempDir()
		alias := filepath.Join(aliasParent, "active")
		if err := os.Symlink(active, alias); err != nil {
			t.Skipf("symlink unavailable: %v", err)
		}
		if _, err := New(Config{Directory: active, DraftDirectory: filepath.Join(alias, "drafts")}); err == nil {
			t.Fatal("draft directory inside approved directory through a symlink was accepted")
		}
	})

	catalog, err := New(Config{Directory: activeDir, DraftDirectory: t.TempDir(), MaxDraftBytes: 32})
	ktest.RequireNoError(t, err)
	for _, invalid := range []Skill{
		{Name: "../escape", Instructions: "text"},
		{Name: "empty", Instructions: " \n"},
		{Name: "nul", Instructions: "bad\x00text"},
		{Name: "too-large", Instructions: "This text is larger than the configured limit."},
	} {
		if _, err := catalog.CreateDraft(invalid); err == nil {
			t.Fatalf("invalid draft %q was accepted", invalid.Name)
		}
	}

	withoutDrafts, err := New(Config{Directory: activeDir})
	ktest.RequireNoError(t, err)
	if _, err := withoutDrafts.CreateDraft(Skill{Name: "draft", Instructions: "text"}); err == nil {
		t.Fatal("draft was allowed without a configured draft directory")
	}
}

func TestCatalogBoundsApprovedFileSizeAndCount(t *testing.T) {
	t.Run("file size", func(t *testing.T) {
		activeDir := t.TempDir()
		if err := os.WriteFile(filepath.Join(activeDir, "large.md"), []byte("long instructions"), 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := New(Config{Directory: activeDir, MaxSkillBytes: 4}); err == nil {
			t.Fatal("oversized approved skill was accepted")
		}
	})

	t.Run("skill count", func(t *testing.T) {
		activeDir := t.TempDir()
		for _, name := range []string{"one.md", "two.md"} {
			if err := os.WriteFile(filepath.Join(activeDir, name), []byte("text"), 0o600); err != nil {
				t.Fatal(err)
			}
		}
		if _, err := New(Config{Directory: activeDir, MaxSkills: 1}); err == nil {
			t.Fatal("catalog exceeding its skill-count limit was accepted")
		}
	})
}
