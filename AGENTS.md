# AGENTS.md

Repository guidance for coding agents working in `pat_toolbox`.

## Scope

- Primary entry point: `main.py`
- Main package: `pat_toolbox/`
- Standalone analysis scripts: `analysis/`
- One-off experiments: `experiments/`
- Dependencies are pinned in `requirements.txt`

## Editor Rule Files

- No `.cursor/rules/` directory was found.
- No `.cursorrules` file was found.
- No `.github/copilot-instructions.md` file was found.
- Do not assume any hidden Cursor or Copilot rules beyond this file.

## Environment And Setup

- This is a plain Python repository, not a packaged project.
- There is no `pyproject.toml`, `setup.cfg`, `pytest.ini`, `ruff.toml`, or `mypy.ini`.
- Run commands from the repository root so relative paths resolve consistently.
- Typical setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build, Lint, And Test Commands

### Main pipeline

```bash
python main.py
```

- This is the closest thing to the canonical app run.
- It uses `pat_toolbox/config.py` for input paths, output folders, toggles, and thresholds.

### Standalone scripts

```bash
python analysis/boxplots_AHI.py
python analysis/boxplots_AHI_groups.py
python experiments/hypnogram_diego.py
```

- These are script entry points, not tests.
- Use them only when your change clearly affects that workflow.

### Lint / smoke check

- No formatter or linter is configured in-repo.
- Do not assume `ruff`, `black`, `flake8`, `pylint`, or `mypy` are available.
- For a safe syntax check, use:

```bash
python -m compileall main.py pat_toolbox analysis experiments
```

### Tests

- No automated test suite exists today.
- No `tests/` directory or `test_*.py` files are present.
- Do not claim `pytest` support as an existing project contract.

### Running a single test

- There is currently no real single-test command because the repository has no tests.
- If a future change adds `pytest`, the conventional single-test form would be:

```bash
python -m pytest path/to/test_file.py::test_name
```

- Until then, validate with targeted script runs or `compileall`.

## High-Level Architecture

- `main.py` enumerates EDF files and drives batch processing.
- `pat_toolbox/config.py` is the central control surface for paths, feature flags, masking policy, and numeric parameters.
- `pat_toolbox/workflows.py` orchestrates loading, filtering, HR, HRV, PSD, PAT burden, and report generation.
- `pat_toolbox/context.py` defines `RecordingContext`, the per-recording state container.
- `pat_toolbox/io_edf.py` reads EDF channels and lists EDF inputs.
- `pat_toolbox/io_aux_csv.py` and `pat_toolbox/io/` normalize auxiliary CSV data and event masks.
- `pat_toolbox/core/` holds shared low-level logic such as RR cleaning and interpolation helpers.
- `pat_toolbox/metrics/` contains signal-derived computations.
- `pat_toolbox/plotting/` contains PDF/report rendering.

## Repository-Specific Design Rules

- Prefer config-driven changes over hard-coded constants.
- Keep orchestration in `workflows.py`, not inside metric helpers.
- Keep file discovery and path logic in `io_*` modules or `pat_toolbox/paths.py`.
- Keep plotting concerns in `pat_toolbox/plotting/`, not in metric computation.
- Preserve current batch behavior: a bad file should log an error and not abort the full run.
- Be careful with machine-specific absolute paths in `pat_toolbox/config.py`; do not rewrite them unless requested.
- Preserve current public entry points unless the user asked for a broader refactor.

## Code Style Guidelines

### Imports

- Group imports as standard library, third-party, then local package imports.
- Within package code, prefer explicit relative imports such as `from . import config` or `from ..metrics import hrv`.
- Avoid wildcard imports.
- Remove unused imports when touching a file.

### Formatting

- Follow PEP 8 with 4-space indentation.
- Prefer readable multi-line calls with trailing commas.
- Keep helper blocks separated by a single blank line or short section headers when useful.
- Keep comments practical; avoid narrating obvious code.

### Types

- Preserve existing type hints and add them to new non-trivial functions.
- Match the local file style: some files use `Optional[...]`, others use `Path | None`.
- Use `TYPE_CHECKING` guards for heavy typing-only imports such as pandas.
- Use `Path` for filesystem paths instead of raw strings when writing new code.
- Use NumPy arrays as the default representation for signal and time-series data.

### Naming

- Modules, functions, and variables use `snake_case`.
- Classes and dataclasses use `CapWords`.
- Config constants use `ALL_CAPS`.
- Internal helpers often use a leading underscore.
- Keep units visible in names when practical: `*_sec`, `*_hz`, `*_ms`, `fs`, `sfreq`, `t_*`.

### Numeric And Signal Conventions

- Validate sampling frequency and array shape early.
- Raise `ValueError` for invalid low-level inputs.
- Convert arrays to float when downstream NaN handling or interpolation requires it.
- Prefer vectorized NumPy operations, but small explicit loops are acceptable for cleanup and window logic.
- Preserve unit clarity in variable names, docstrings, and summary fields.

### Error Handling

- Fail fast in low-level helpers when inputs are malformed.
- At workflow boundaries, catch exceptions, print a useful warning, and continue to the next EDF.
- Include file or channel context in error messages when possible.
- For optional data sources, prefer `None`, empty outputs, or `np.nan` over crashing the full pipeline.
- Do not silently swallow exceptions unless cleanup requires it.

### Configuration

- New algorithm knobs usually belong in `pat_toolbox/config.py`.
- Use `getattr(config, "NAME", default)` when backward compatibility matters.
- Reuse existing config naming patterns such as `ENABLE_*`, `*_SEC`, `*_HZ`, `*_BPM`, and `*_SUBFOLDER`.

### Paths And Outputs

- Use `pathlib.Path` for path composition.
- Prefer helpers in `pat_toolbox/paths.py` over ad hoc directory creation.
- Keep output filenames descriptive and consistent with current `<edf_base>__...` conventions.

## Validation Expectations

- For non-runtime-safe edits, run `python -m compileall main.py pat_toolbox analysis experiments`.
- Run `python main.py` only when the configured EDF and output paths are valid in the current environment.
- If you change a standalone script, run that script directly when practical.
- If runtime validation is impossible because local data paths are machine-specific, state that clearly.

## Agent Workflow Notes

- Check `pat_toolbox/config.py` before changing behavior or claiming how the pipeline runs.
- Keep changes narrow; avoid incidental tooling or packaging cleanup.
- Do not add a formatter, linter, CI pipeline, or test framework unless the user asked for it.
- This repository uses `bd` for issue tracking; prefer `bd` over markdown TODOs if task tracking is needed.
