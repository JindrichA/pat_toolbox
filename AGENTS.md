# AGENTS.md

This file gives coding agents repository-specific guidance for `pat_toolbox`.
It is based on the current codebase layout, configuration files, and source conventions.

## Scope

- Main entry point: `main.py`
- Core package: `pat_toolbox/`
- Analysis scripts: `analysis/`
- Experimental scripts: `experiments/`
- Dependency source: `requirements.txt`
- There is currently no existing `AGENTS.md` in this repository.

## Rule Files

- No `.cursor/rules/` directory was found.
- No `.cursorrules` file was found.
- No `.github/copilot-instructions.md` file was found.
- Do not assume any hidden editor-specific rules beyond this file.

## Environment And Setup

- The repo is Python-based and uses a plain `requirements.txt` workflow.
- There is no `pyproject.toml`, `setup.cfg`, `pytest.ini`, `ruff.toml`, or `mypy.ini`.
- A local virtual environment at `.venv/` exists in the workspace, but agents should not depend on it being pre-activated.
- Install dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build, Lint, And Test Commands

The repository does not define a formal build system or automated lint/test tooling yet.
Use the commands below as the current operational workflow.

### Run The Main Pipeline

```bash
python main.py
```

- This is the primary way to execute the EDF processing pipeline.
- It reads paths and runtime toggles from `pat_toolbox/config.py`.
- It writes outputs under `config.BASE_OUTPUT_DIR`.

### Run A Specific Analysis Script

```bash
python analysis/boxplots_AHI.py
python analysis/boxplots_AHI_groups.py
python experiments/hypnogram_diego.py
```

- These are standalone scripts, not package entry points.
- Prefer running them from the repository root so relative paths resolve consistently.

### Linting

- No linter is configured in-repo.
- If you need a one-off syntax/smoke check without introducing new tooling, use:

```bash
python -m compileall main.py pat_toolbox analysis experiments
```

- Do not add or assume `ruff`, `flake8`, `black`, or `pylint` unless the user asks.

### Tests

- No automated test suite was found.
- No `tests/` directory or `test_*.py` files are currently present.
- Do not claim that `pytest` is configured unless you add tests and the user asks for that work.

### Running A Single Test

- There is currently no in-repo single-test command because no automated tests exist.
- If tests are added later and use `pytest`, the conventional single-test form would be:

```bash
python -m pytest path/to/test_file.py::test_name
```

- Until such tests exist, prefer targeted function-level smoke checks or script execution.

## High-Level Architecture

- `main.py` orchestrates batch processing over EDF files.
- `pat_toolbox/config.py` is the control center for paths, toggles, and algorithm parameters.
- `pat_toolbox/workflows.py` wires together loading, filtering, HR, HRV, burden, plotting, and summary output.
- `pat_toolbox/context.py` uses a dataclass to hold per-recording state across workflow steps.
- `pat_toolbox/io_edf.py` and `pat_toolbox/io_aux_csv.py` handle file discovery and normalization.
- `pat_toolbox/metrics/` contains computational logic.
- `pat_toolbox/plotting/` contains PDF/report generation.

## Repository-Specific Coding Principles

- Prefer extending the existing config-driven design over adding hard-coded constants.
- Keep data-processing logic in `metrics/`, orchestration in `workflows.py`, and file/path concerns in `io_*` or `paths.py`.
- Preserve the current pattern of best-effort batch processing: failures for one EDF should log a warning and not abort the full run.
- Favor reproducible outputs and deterministic naming through `config.RUN_ID`, `config.RUN_TAG`, and `paths.get_output_folder()`.
- Be careful with machine-specific absolute paths already present in `config.py`; do not silently normalize or rewrite them unless requested.

## Style Guidelines

These conventions are inferred from the existing codebase and should be preserved unless the user requests a broader refactor.

### Imports

- Use standard library imports first, then third-party imports, then local package imports.
- Within local imports, prefer explicit package-relative imports inside the package, for example `from . import config` or `from .. import paths`.
- Prefer one import per logical line instead of compressed star imports.
- Avoid unused imports; the codebase is mostly explicit.

### Formatting

- Follow PEP 8 style with 4-space indentation.
- Use readable vertical spacing between helper sections.
- Keep long calls split across multiple lines with trailing commas.
- Existing code mixes compact and spaced styles; for touched code, prefer the cleaner multi-line style already used in `workflows.py` and `report.py`.
- Keep comments practical and section-oriented; do not add commentary for obvious code.

### Types

- Type hints are used broadly and should be kept when modifying typed code.
- Prefer builtin generics on modern Python where already used, e.g. `set[int]`, while tolerating older `typing` imports where they already exist.
- Use `Path` instead of raw strings for filesystem paths.
- Use `Optional[...]` or `X | None` consistently with the surrounding file rather than mixing styles unnecessarily.
- For pandas-only imports used for typing, guard them with `TYPE_CHECKING` when practical.

### Naming

- Modules use snake_case.
- Functions use snake_case.
- Config constants use ALL_CAPS.
- Internal helper functions often use a leading underscore, especially in `workflows.py`, `io_aux_csv.py`, and metrics modules.
- Dataclasses and similar types use CapWords, e.g. `RecordingContext`.
- Time axis names typically use `t_*` and units are encoded in names like `*_sec`, `*_hz`, `*_ms`, `sfreq`, and `fs`.

### Numeric And Array Conventions

- Use NumPy arrays for signal and time-series data.
- Convert arrays to float explicitly when needed for NaN-safe operations.
- Prefer vectorized operations, but small explicit loops are acceptable for sliding windows, median filters, and stepwise cleanup logic.
- Validate signal shape and sampling frequency early, then raise a clear `ValueError` for invalid inputs.
- Preserve unit clarity in variable names and docstrings.

### Error Handling

- Fail fast inside low-level computation helpers when inputs are invalid.
- At workflow or batch boundaries, catch exceptions, print a warning or error, and continue with the next file.
- Use informative messages that include the EDF filename or channel/context when possible.
- When a feature is optional, prefer returning `None`, empty arrays, or NaN-filled outputs rather than crashing the whole pipeline.
- Do not swallow exceptions silently unless cleanup requires it.

### Configuration And Parameters

- New algorithm knobs should usually live in `pat_toolbox/config.py`.
- Read optional config values via `getattr(config, "NAME", default)` when backward compatibility matters.
- Reuse existing config naming patterns such as `*_SEC`, `*_HZ`, `*_BPM`, `ENABLE_*`, and `*_SUBFOLDER`.

### Paths And Outputs

- Use `pathlib.Path` everywhere for path composition.
- Use helpers in `pat_toolbox/paths.py` instead of manually creating output directories.
- Keep output filenames descriptive and consistent with the existing `<edf_base>__...` convention.

### Documentation

- Keep docstrings concise and practical.
- Document units, array semantics, and return values when they are not obvious.
- Prefer technical comments for algorithmic rationale, not narrative prose.

## Agent Do And Don't List

- Do inspect `config.py` before changing pipeline behavior.
- Do preserve current public function names unless the user asked for a rename.
- Do keep plotting/report code separate from metric computation.
- Do not introduce a packaging system, formatter, or linter as incidental cleanup.
- Do not add tests, CI, or new developer tooling unless requested.
- Do not assume reference HR is active; current plotting code intentionally hides proprietary/device HR.

## Validation Expectations For Changes

- For pipeline changes, run `python main.py` only if the configured data paths are valid in the current environment.
- For script-only changes, run the relevant script directly.
- If real data is unavailable, fall back to `python -m compileall main.py pat_toolbox analysis experiments` and clearly state that runtime validation was limited.

## Practical Note For Future Test Additions

- If the repository later gains `pytest`, add a dedicated test section here.
- Include both full-suite and single-test commands.
- Until then, describe validation as smoke testing rather than automated testing.

<!-- BEGIN BEADS INTEGRATION v:1 profile:full hash:d4f96305 -->
## Issue Tracking with bd (beads)

**IMPORTANT**: This project uses **bd (beads)** for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods.

### Why bd?

- Dependency-aware: Track blockers and relationships between issues
- Git-friendly: Dolt-powered version control with native sync
- Agent-optimized: JSON output, ready work detection, discovered-from links
- Prevents duplicate tracking systems and confusion

### Quick Start

**Check for ready work:**

```bash
bd ready --json
```

**Create new issues:**

```bash
bd create "Issue title" --description="Detailed context" -t bug|feature|task -p 0-4 --json
bd create "Issue title" --description="What this issue is about" -p 1 --deps discovered-from:bd-123 --json
```

**Claim and update:**

```bash
bd update <id> --claim --json
bd update bd-42 --priority 1 --json
```

**Complete work:**

```bash
bd close bd-42 --reason "Completed" --json
```

### Issue Types

- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Workflow for AI Agents

1. **Check ready work**: `bd ready` shows unblocked issues
2. **Claim your task atomically**: `bd update <id> --claim`
3. **Work on it**: Implement, test, document
4. **Discover new work?** Create linked issue:
   - `bd create "Found bug" --description="Details about what was found" -p 1 --deps discovered-from:<parent-id>`
5. **Complete**: `bd close <id> --reason "Done"`

### Auto-Sync

bd automatically syncs via Dolt:

- Each write auto-commits to Dolt history
- Use `bd dolt push`/`bd dolt pull` for remote sync
- No manual export/import needed!

### Important Rules

- ✅ Use bd for ALL task tracking
- ✅ Always use `--json` flag for programmatic use
- ✅ Link discovered work with `discovered-from` dependencies
- ✅ Check `bd ready` before asking "what should I work on?"
- ❌ Do NOT create markdown TODO lists
- ❌ Do NOT use external issue trackers
- ❌ Do NOT duplicate tracking systems

For more details, see README.md and docs/QUICKSTART.md.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

<!-- END BEADS INTEGRATION -->
