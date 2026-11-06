# Host-approved skills

The `skills` package loads a host-approved set of plain Markdown instructions and keeps generated proposals separate. It does not execute skill text, grant tool access, edit the active catalog, or activate proposals.

## Configure and inject at startup

```go
catalog, err := skills.New(skills.Config{
    Directory:      "./skills/approved",
    DraftDirectory: "./skills/drafts",
})
if err != nil {
    return err
}

bundle, err := nexssflow.NewBundleWithHostServices(globalOptions, providerBuilder, nexssflow.HostServices{
    Memory: memoryService,
    Skills: catalog,
})
if err != nil {
    return err
}
```

`Directory` must already exist and be a real directory, not a symlink. Each approved skill is one top-level `<name>.md` file. Names are lowercase identifiers (letters, digits, and hyphens; first character must be a letter). The file body is treated as instruction text; there is no metadata parser, command execution, hook loader, or nested directory traversal. Startup fails closed for malformed names/content, symlinks, invalid UTF-8/NUL, oversized files, and excess skill count. Defaults are 64 KiB per file and 128 skills; `MaxSkillBytes`, `MaxSkills`, and `MaxDraftBytes` can tighten or raise those bounds.

The catalog is an immutable startup snapshot. Adding or changing a Markdown file does not alter a running catalog; create a new catalog after the host has reviewed and installed an approved change.

## NFlow actions

`nexssflow.NewBundleWithHostServices` exposes these actions when a catalog is injected:

- `ai.skills.list` returns the approved startup snapshot.
- `ai.skills.get` takes `{name}` and returns one approved skill. It does not accept a path or read arbitrary files.
- `ai.skills.draft` takes `{name, instructions}`, validates the content, and creates a unique JSON file with restrictive permissions in `DraftDirectory`.

The draft directory is created with private permissions if needed; permissions on an existing directory remain host-managed. It must be separate from the active directory (not equal to it, inside it, or a parent of it). Draft names do not become active names automatically. There is intentionally no activation API. The host operator must inspect the JSON proposal, decide whether it is acceptable, install the approved text as `<name>.md` in the active directory, and recreate the catalog. Draft creation does not execute code or change any active skill.

Treat the resulting instruction text as trusted only after review. A workflow may pass it to an LLM as a system prompt, but the skills package does not independently authorize the model's tools or actions; enforce those capabilities in the host/runtime.
