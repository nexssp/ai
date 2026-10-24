// Package siaimodules provides the embedded public NFlow modules shipped with ai.
package siaimodules

import "embed"

// SourceModules is the compile-time source filesystem for the public NFlow modules.
// Exporting remains an explicit compiler.Bundle policy in nexssflow.Bundle.
//
//go:embed nflows/lib/*.nflow
var SourceModules embed.FS
