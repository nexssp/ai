// Package filecommit contains safe publication helpers for checkpoint outputs.
package filecommit

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"

	"github.com/nexssp/kernel/xfs"
)

// VerifyRoots checks that the xfs and os root handles anchor the same directory.
func VerifyRoots(workspaceRoot *xfs.Root, commitRoot *os.Root) error {
	if workspaceRoot == nil || commitRoot == nil {
		return errors.New("filecommit: matching workspace roots are required")
	}
	workspaceDir, err := workspaceRoot.FS().Open(".")
	if err != nil {
		return fmt.Errorf("filecommit: inspect workspace root: %w", err)
	}
	workspaceInfo, statErr := workspaceDir.Stat()
	_ = workspaceDir.Close()
	if statErr != nil {
		return fmt.Errorf("filecommit: stat workspace root: %w", statErr)
	}
	commitDir, err := commitRoot.Open(".")
	if err != nil {
		return fmt.Errorf("filecommit: inspect publication root: %w", err)
	}
	commitInfo, statErr := commitDir.Stat()
	_ = commitDir.Close()
	if statErr != nil {
		return fmt.Errorf("filecommit: stat publication root: %w", statErr)
	}
	if !os.SameFile(workspaceInfo, commitInfo) {
		return errors.New("filecommit: workspace and publication roots do not identify the same directory")
	}
	return nil
}

// LinkNoReplace atomically publishes a staged file without replacing an existing
// destination. The stage file must be in the destination directory (and therefore
// on the same filesystem). Both paths are relative to the matching roots. Callers
// should remove the staging name after publication; the destination remains a
// hard link to it.
func LinkNoReplace(workspaceRoot *xfs.Root, commitRoot *os.Root, stagePath, destinationPath string) error {
	if err := VerifyRoots(workspaceRoot, commitRoot); err != nil {
		return err
	}
	stagePath, err := xfs.Rel(stagePath)
	if err != nil {
		return fmt.Errorf("filecommit: invalid stage path: %w", err)
	}
	destinationPath, err = xfs.Rel(destinationPath)
	if err != nil {
		return fmt.Errorf("filecommit: invalid destination path: %w", err)
	}
	if filepath.Dir(stagePath) != filepath.Dir(destinationPath) {
		return errors.New("filecommit: stage file must be in the destination directory")
	}
	if err := commitRoot.Link(stagePath, destinationPath); err != nil {
		return fmt.Errorf("filecommit: publish %q without replacing an existing file: %w", destinationPath, err)
	}
	if runtime.GOOS != "windows" {
		if dir, err := commitRoot.Open(filepath.Dir(destinationPath)); err == nil {
			if err := dir.Sync(); err != nil {
				slog.Warn("filecommit_dir_fsync_failed", "dir", filepath.Dir(destinationPath), "error", err)
			}
			_ = dir.Close()
		}
	}
	return nil
}
