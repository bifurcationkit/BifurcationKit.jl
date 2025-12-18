# Instructions for Agents and Developers (BifurcationKit.jl)

This file documents the specifics of the `BifurcationKit.jl` project to assist AI agents and new contributors.

## 1. Environment and Compilation

### Precompilation

The project uses `PrecompileTools.jl` to improve "Time-To-First-Plot".
If you modify the source code (especially `src/periodicorbit/` or `src/BifurcationKit.jl`), you must regenerate the caches locally to see the performance gains:

```bash
julia --project=. -e 'using Pkg; Pkg.precompile()'
```

### Example Scripts

Scripts in `examples/` should be executed by activating the parent project environment to ensure dependencies (and local fixes) are correctly loaded:

```julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
```

## 2. Tests and Quality

### Running Tests

Do not run `include("test/runtests.jl")` directly without precautions. Instead, use:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Or for a specific test (via `Pkg` which handles the test environment correctly):

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["newton"])'
```

### CI (GitHub Actions)

The CI workflow (`.github/workflows/ci.yml`) parses PR labels to filter which tests to run.
Be aware of secrets: `CODECOV_TOKEN` is required for coverage upload (Codecov v4).

## 3. Architecture and Critical Points

### Automatic Differentiation (AD)

The package relies heavily on `ForwardDiff.jl`.
*   **Chunk Sizes Warning:** For large-scale problems (Collocation), the default chunk size of `DiffCache` may be insufficient. Always use `ForwardDiff.pickchunksize(N_total)` when creating caches to avoid dynamic allocations.
*   **Compilation Times:** Functions like `continuation` involving `MinAugFold` and `PeriodicOrbitOCollProblem` generate very heavy compilation graphs. The precompilation block in `src/BifurcationKit.jl` is crucial to mitigate this.

## Instructions for coding agents

### Code Style

* Prefer `using` over `import` for standard libraries.
* Core modules are located in `src/`.
* all comments, strins, instructions, variable names, etc.. must be in english (no french allowed)
* do not use any emojis (in comments/messages/files/etc..)

### Reporting

* always report analysis and action in the `compilation_analysis_report.md` file
* for now, the content of `compilation_analysis_report.md` remains in french

### Commands

* ask for autorisation then running a command, specially if the command is dangerous or may delete a file
* never work on master or main branch of a git repository
* to not use git commands
