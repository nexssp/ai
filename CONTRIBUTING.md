# Contributing to Nexss Ecosystem

This repository is part of the **Nexss Ecosystem**.

We keep packages small, focused, and composable. Contributions should improve
correctness, clarity, portability, or measured performance without expanding the
public API unnecessarily.

## Toolchain

Every workflow in this repo goes through [Task](https://taskfile.dev). Do not
invoke `go build`, `go test`, `golangci-lint`, or `govulncheck` directly — the
`task` targets keep flags, ordering, and cache behavior identical to CI.

Install Task once:

```bash
brew install go-task/tap/go-task                       # macOS / Linuxbrew
scoop install task                                     # Windows (Scoop)
go install github.com/go-task/task/v3/cmd/task@latest  # any platform
```

Bootstrap the repo:

```bash
task           # list every available target
task setup     # fetch repo templates + install tool deps (gotestsum, govulncheck)
```

## Before opening a pull request

Run the full gate:

```bash
task tidy       # go mod tidy — keep go.mod / go.sum clean
task fmt        # gofumpt + gci (auto-fix)
task pre-push   # [lint, test:race]
task ci         # [lint, test:race:nocache] — CI verification gate
```

Optional extra checks when relevant:

```bash
task vuln       # govulncheck
task bench      # benchmark smoke test
task test:cover # short tests with coverage
```

For focused checks, pass package paths and Go flags after `--`, for example
`task test -- ./tcli/... -run TestName` or `task lint -- ./message`.
Package-taking tasks use `./...` when no arguments are supplied.
Tests use `go test` by default. To use the optionally installed gotestsum,
run `task test GOTEST='gotestsum --' -- ./message`.
Transport is a library: Flow compiler self-tests, catalog generation, spec linting,
and Flow CLI/bundle tasks belong in the Flow repository, not this Taskfile.

## Requirements

- A change to a **public interface** requires a compatibility explanation and tests.
- A performance claim requires a benchmark on a documented Go version and hardware.
- New dependencies require a clear reason, license review, and evidence that the
  dependency is not better placed in an adapter module.
- Keep business rules, hosted services, and provider-specific
  behavior outside this repository.
- Flow integration changes should include focused tests in the relevant
  transport adapter package; compiler and DSL changes belong in Flow.
- Do not add `testify` or other assertion helpers; use `xtest` / `xtest/ktest`.

## Local pre-commit (optional)

Install the hooks:

```bash
pre-commit install
# or
prek install
```

Commit hooks run `task lint:new` and `task test`; push hooks run `task pre-push`,
so **Task must be installed first** — see the Toolchain section above.

## Getting help

If a core or Kernel change appears necessary, write a short RFC before opening
the PR: describe the missing contract, alternatives considered, and why an
existing extension or adapter cannot solve the problem.

Thank you for contributing to Nexss.
