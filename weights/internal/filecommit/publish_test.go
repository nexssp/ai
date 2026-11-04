package filecommit

import (
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/nexssp/kernel/xfs"
)

func TestLinkNoReplaceConcurrentPublishers(t *testing.T) {
	dir := t.TempDir()
	workspaceRoot, err := xfs.OpenRoot(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer workspaceRoot.Close()
	commitRoot, err := os.OpenRoot(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer commitRoot.Close()

	for name, value := range map[string]string{"first.tmp": "first", "second.tmp": "second"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(value), 0o600); err != nil {
			t.Fatal(err)
		}
	}

	start := make(chan struct{})
	results := make(chan error, 2)
	var wait sync.WaitGroup
	for _, stage := range []string{"first.tmp", "second.tmp"} {
		wait.Add(1)
		go func(stage string) {
			defer wait.Done()
			<-start
			results <- LinkNoReplace(workspaceRoot, commitRoot, stage, "published.safetensors")
		}(stage)
	}
	close(start)
	wait.Wait()
	close(results)

	successes := 0
	failures := 0
	for err := range results {
		if err == nil {
			successes++
			continue
		}
		if !errors.Is(err, os.ErrExist) {
			t.Fatalf("LinkNoReplace() error = %v, want an existing-destination error", err)
		}
		failures++
	}
	if successes != 1 || failures != 1 {
		t.Fatalf("concurrent publication results: successes=%d failures=%d, want 1/1", successes, failures)
	}

	published, err := os.ReadFile(filepath.Join(dir, "published.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	if got := string(published); got != "first" && got != "second" {
		t.Fatalf("published output = %q, want one complete staged value", got)
	}
}

func TestLinkNoReplaceRejectsMismatchedRoots(t *testing.T) {
	workspaceDir := t.TempDir()
	publicationDir := t.TempDir()
	workspaceRoot, err := xfs.OpenRoot(workspaceDir)
	if err != nil {
		t.Fatal(err)
	}
	defer workspaceRoot.Close()
	commitRoot, err := os.OpenRoot(publicationDir)
	if err != nil {
		t.Fatal(err)
	}
	defer commitRoot.Close()

	if err := os.WriteFile(filepath.Join(workspaceDir, "stage.tmp"), []byte("stage"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := LinkNoReplace(workspaceRoot, commitRoot, "stage.tmp", "published.safetensors"); err == nil {
		t.Fatal("LinkNoReplace() accepted different workspace and publication roots")
	}
	if _, err := os.Stat(filepath.Join(publicationDir, "published.safetensors")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("mismatched-root publication created an output: %v", err)
	}
}
