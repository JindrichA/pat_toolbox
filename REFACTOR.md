# Refactor Plan For PAT Toolbox

## Goal

Turn this repository into a clear, extensible EDF-based physiology toolbox where new features can be added without rewriting the pipeline each time.

Critical constraint:

- refactoring must not change algorithms or numerical results
- the same inputs and config should produce the same outputs after refactor
- behavior changes should only happen when a user explicitly requests a new feature or new algorithm option

The toolbox should support the current HR/HRV workflow now, and make it easy to add more EDF-derived features later, such as:

- delta HR
- PAT burden
- PSD-derived features
- future event features
- future signal quality metrics
- future summary/report modules

The main design target is:

- simple to read
- easy to debug
- config-driven
- modular enough that new metrics can be plugged in
- stable enough that other users can adopt it as a general toolbox

## Current Problems

The main issues in the current codebase are structural rather than algorithmic.

### 1. Large files mix multiple responsibilities

The biggest examples are:

- `pat_toolbox/metrics/hr.py`
- `pat_toolbox/metrics/hrv.py`
- `pat_toolbox/metrics/psd.py`
- `pat_toolbox/io_aux_csv.py`
- `pat_toolbox/workflows.py`
- `pat_toolbox/plotting/figures_hrv.py`
- `pat_toolbox/plotting/segments.py`
- `pat_toolbox/plotting/figures_summary.py`

These modules currently mix:

- low-level signal math
- windowing logic
- masking logic
- EDF wrappers
- summary generation
- plotting
- report assembly

That makes it hard to add new features without touching many unrelated places.

### 2. Some logic exists in the wrong layer

Examples:

- metric-specific window rules are partly embedded in plotting expectations
- summary CSV concerns live inside metric modules
- aux CSV parsing and exclusion logic are bundled together
- HR-derived reusable RR logic is buried inside `hr.py`

This increases coupling and makes reuse harder.

### 3. Similar logic is duplicated

Examples of duplication or near-duplication:

- window acceptance logic for HRV time-domain vs spectral metrics
- summary/stat formatting helpers
- exclusion/mask-related handling across HRV, PSD, burden, and plotting
- repeated “collect data -> compute metric -> package outputs” patterns

### 4. The pipeline is feature-specific instead of feature-oriented

Right now, adding a new metric often means editing several existing files directly.

The future toolbox should instead allow a new feature to be added as a self-contained computation module with:

- inputs
- config options
- outputs
- plotting hooks
- summary hooks

## Refactor Principles

These principles should guide every change.

### Preserve algorithm behavior exactly

This is the most important non-structural rule.

- refactor structure, not science
- do not silently change thresholds, masks, interpolation, or window semantics
- do not merge similar logic if the current behavior is intentionally different
- when moving code, keep results byte-for-byte or numerically identical whenever practical
- if a future cleanup would change results, it should be introduced as a new explicitly named option, not as part of refactoring

Refactor success means:

- same computation
- better structure
- easier extension
- easier debugging

### Keep the API stable while refactoring

Refactor in layers, but keep current external entry points working:

- `main.py`
- `pat_toolbox/workflows.py`
- report generation entry points

This avoids breaking the current workflow while the internal structure improves.

### Separate core computation from orchestration

Computation modules should not decide file paths, PDF pages, or summary-file layout.

- core modules compute data
- workflow modules decide what runs and in what order
- plotting modules only render already-computed data
- summary/output modules serialize results

### Create one canonical place for shared logic

There should be one clear home for each category of shared behavior:

- RR extraction and cleaning
- masking and exclusion
- sliding-window acceptance rules
- summary formatting helpers
- aux normalization

Important exception:

- only centralize shared logic when behavior is truly shared
- if two features intentionally use different masking or window rules, keep those differences explicit rather than forcing them into one common rule

### Prefer small modules with explicit names

Files should be organized by responsibility, not by historical growth.

Good:

- `rr_cleaning.py`
- `hrv_windows.py`
- `aux_events.py`
- `summary_writer.py`

Avoid adding more generic “misc” files.

### Preserve config-driven behavior

New features should be controlled from `pat_toolbox/config.py` or a future split config package, not with hard-coded switches spread through the code.

## Target Architecture

Recommended future structure:

```text
pat_toolbox/
|- config.py
|- context.py
|- paths.py
|- workflows/
|  |- __init__.py
|  |- runner.py
|  |- load_steps.py
|  |- metric_steps.py
|  `- output_steps.py
|- io/
|  |- __init__.py
|  |- edf_reader.py
|  |- aux_reader.py
|  |- aux_normalize.py
|  `- aux_events.py
|- core/
|  |- __init__.py
|  |- rr_cleaning.py
|  |- windows.py
|  |- masking.py
|  `- signal_quality.py
|- features/
|  |- __init__.py
|  |- hr/
|  |  |- __init__.py
|  |  |- compute.py
|  |  |- summary.py
|  |  `- debug.py
|  |- hrv/
|  |  |- __init__.py
|  |  |- time_domain.py
|  |  |- freq_domain.py
|  |  |- windows.py
|  |  |- pipeline.py
|  |  `- summary.py
|  |- delta_hr/
|  |  |- __init__.py
|  |  |- compute.py
|  |  `- summary.py
|  |- pat_burden/
|  |  |- __init__.py
|  |  |- compute.py
|  |  `- summary.py
|  `- psd/
|     |- __init__.py
|     |- compute.py
|     |- peaks.py
|     `- summary.py
|- plotting/
|  |- __init__.py
|  |- report.py
|  |- summary_tables.py
|  |- stagegram.py
|  |- segments/
|  |  |- __init__.py
|  |  |- pages.py
|  |  |- hr_panel.py
|  |  |- hrv_panel.py
|  |  |- delta_panel.py
|  |  `- event_panel.py
|  `- hrv/
|     |- __init__.py
|     |- overview.py
|     |- tv.py
|     `- combined.py
`- outputs/
   |- __init__.py
   |- summary_writer.py
   `- csv_export.py
```

This structure separates:

- I/O
- reusable core logic
- feature computations
- plotting
- output serialization
- workflow orchestration

## Core Design Concept: Feature Modules

Each feature should become a small package with a predictable structure.

For example, a feature should define:

- what inputs it needs
- what config enables it
- what outputs it produces
- whether it contributes summary values
- whether it contributes plots

Example feature categories:

- `hr`
- `hrv`
- `delta_hr`
- `pat_burden`
- `psd`

Each feature should return structured outputs instead of loose dicts where possible.

Each feature should also own its own analysis policy where needed, especially for masking and exclusions.

That means a future feature may define:

- whether sleep-stage masking applies
- whether event masking applies
- which event columns matter
- whether desaturation-linked exclusion applies
- whether it uses raw, sleep-only, or fully excluded data

This should be configurable per feature rather than globally forced when the scientific intent differs.

In addition, each feature should declare its output surfaces explicitly.

That means one place should decide whether a feature contributes to:

- computation
- PDF plots
- summary tables
- batch summary CSV
- debug plots

This is important so that disabling a feature removes it consistently everywhere, instead of leaving partial traces in tables or plots.

Recommended concept:

- a feature registry or feature manifest
- each feature has explicit enable flags for the outputs it owns

Recommended pattern:

```python
@dataclass
class HRVResult:
    t_sec: np.ndarray
    rmssd_clean: np.ndarray
    rmssd_raw: np.ndarray
    tv_metrics: dict[str, np.ndarray]
    summary: dict[str, float]
    mask_info: dict[str, np.ndarray] | None
```

Later, these dicts can also be replaced with more specific dataclasses.

## Main Refactor Stages

### Stage 1. Stabilize shared core logic

This is the most important first step.

#### 1.1 Extract RR cleaning from `pat_toolbox/metrics/hr.py`

Create a reusable core RR module:

- `pat_toolbox/core/rr_cleaning.py`

Move there:

- PAT peak-to-RR extraction helpers
- RR physiologic filtering
- median/gap/jump/alternans cleanup
- contiguous-run filtering

Reason:

- HR, HRV, PSD, and future features all depend on the same RR base.

#### 1.2 Extract shared window logic

Create:

- `pat_toolbox/core/windows.py`

Move there:

- sliding-window traversal helpers
- shared time-domain window acceptance logic
- contiguous-run splitting helpers
- fixed-window iteration helpers

Reason:

- window logic is currently scattered across HRV and PSD code.

#### 1.3 Keep masking centralized

The new `pat_toolbox/masking.py` is already a good direction.

Long term, it should become:

- `pat_toolbox/core/masking.py`

The rule should be:

- all features use the same mask API
- plotting never reimplements mask semantics

But not necessarily the same mask policy.

Recommended direction:

- one shared masking engine
- feature-specific mask policies
- explicit per-feature config switches

Example:

- HRV may use sleep + event exclusion
- PAT burden may use event-focused windows with a different exclusion meaning
- future features may use sleep-only, event-only, or no masking

So the shared part should be the machinery, not a single forced scientific policy.

### Stage 2. Clean up aux CSV handling

Split `pat_toolbox/io_aux_csv.py` into clearer layers.

Suggested split:

- `pat_toolbox/io/aux_reader.py`
- `pat_toolbox/io/aux_normalize.py`
- `pat_toolbox/io/aux_events.py`

Responsibilities:

- reader: locate and read CSV
- normalize: canonical columns and time parsing
- events: event windows, desat windows, aux-derived helpers

Reason:

- aux parsing is a foundational dependency for hypnogram, masking, burden, and reports.

### Stage 3. Rebuild HR and HRV around shared core

#### 3.1 Split HR

Current `pat_toolbox/metrics/hr.py` should become:

- `pat_toolbox/features/hr/compute.py`
- `pat_toolbox/features/hr/summary.py`
- `pat_toolbox/features/hr/debug.py`

Keep summary CSV writing out of the compute module.

#### 3.2 Split HRV

Current `pat_toolbox/metrics/hrv.py` should become:

- `pat_toolbox/features/hrv/time_domain.py`
- `pat_toolbox/features/hrv/freq_domain.py`
- `pat_toolbox/features/hrv/windows.py`
- `pat_toolbox/features/hrv/pipeline.py`
- `pat_toolbox/features/hrv/summary.py`

Recommended behavior:

- RMSSD and SDNN share the same time-domain window gate
- LF/HF use an additional spectral gate
- global summary calculations are separated from TV-series generation

Reason:

- HRV is currently one of the most overloaded modules in the repository.

### Stage 4. Make each optional feature pluggable

Future feature additions should require minimal edits outside the feature package.

Recommended rule:

- every feature has one compute entry point
- every feature declares whether it is enabled in config
- every feature returns a structured result or `None`
- every feature may define its own masking policy/config block
- every feature declares whether it is shown in report outputs

Examples:

- `ENABLE_HR`
- `ENABLE_HRV`
- `ENABLE_DELTA_HR`
- `ENABLE_PAT_BURDEN`
- `ENABLE_PSD`

And for future per-feature masking behavior, prefer explicit knobs such as:

- `HRV_ENABLE_SLEEP_MASKING`
- `HRV_ENABLE_EVENT_EXCLUSION`
- `PAT_BURDEN_ENABLE_SLEEP_MASKING`
- `PAT_BURDEN_ENABLE_EVENT_EXCLUSION`
- `DELTA_HR_ENABLE_SLEEP_MASKING`

The exact names can evolve, but the design principle should remain:

- feature toggles are local
- masking intent is explicit
- no feature should inherit a scientific rule accidentally

Recommended second layer of output toggles:

- `REPORT_INCLUDE_HR`
- `REPORT_INCLUDE_HRV`
- `REPORT_INCLUDE_DELTA_HR`
- `REPORT_INCLUDE_PAT_BURDEN`
- `REPORT_INCLUDE_PSD`

Or, even better, keep a single feature definition object per feature that controls both computation and presentation.

Example direction:

```python
FEATURES = {
    "hrv": {
        "enabled": True,
        "include_in_report": True,
        "include_in_summary_tables": True,
        "include_in_batch_summary": True,
        "include_in_debug_plots": False,
    },
    "pat_burden": {
        "enabled": True,
        "include_in_report": True,
        "include_in_summary_tables": True,
        "include_in_batch_summary": True,
        "include_in_debug_plots": False,
    },
}
```

The exact implementation can be simpler, but the architectural rule should be:

- a feature is declared once
- report builders read that declaration
- summary builders read that declaration
- plotting code does not independently decide visibility

This prevents the common problem where a feature is disabled in computation but still appears as empty rows, empty panels, or placeholder labels.

At workflow level, the code should look like:

```python
if config.ENABLE_HRV:
    ctx.hrv = compute_hrv_feature(ctx)

if config.ENABLE_DELTA_HR:
    ctx.delta_hr = compute_delta_hr_feature(ctx)
```

This makes the toolbox easier to extend.

### Stage 5. Simplify plotting and report assembly

Plotting should consume already-computed results, not recompute logic.

Plotting and summary code should also be feature-aware through one centralized feature visibility layer.

Recommended rule:

- report pages ask "is this feature enabled and report-visible?"
- summary tables ask "is this feature enabled and summary-visible?"
- no ad hoc checks scattered across page builders

#### 5.1 Split HRV plotting

Break up `pat_toolbox/plotting/figures_hrv.py` into:

- `pat_toolbox/plotting/hrv/overview.py`
- `pat_toolbox/plotting/hrv/tv.py`
- `pat_toolbox/plotting/hrv/combined.py`
- `pat_toolbox/plotting/stagegram.py`

#### 5.2 Split segment plotting

Break up `pat_toolbox/plotting/segments.py` into:

- `pat_toolbox/plotting/segments/pages.py`
- `pat_toolbox/plotting/segments/hr_panel.py`
- `pat_toolbox/plotting/segments/hrv_panel.py`
- `pat_toolbox/plotting/segments/delta_panel.py`
- `pat_toolbox/plotting/segments/event_panel.py`

#### 5.3 Split summary plotting from summary data assembly

Current `pat_toolbox/plotting/figures_summary.py` mixes:

- metric summarization
- formatting helpers
- table rendering
- stagegram rendering

Recommended split:

- `pat_toolbox/plotting/summary_tables.py`
- `pat_toolbox/plotting/stagegram.py`
- `pat_toolbox/outputs/summary_writer.py`

Reason:

- plotting should format results, not own business logic.

Also add a small report/summary selection layer, for example:

- `pat_toolbox/outputs/feature_visibility.py`
- or `pat_toolbox/features/registry.py`

Its job would be to answer questions like:

- should this feature be computed?
- should it appear on the report?
- should it appear in the summary table?
- should it be written to batch CSV?

### Stage 6. Refactor workflows into thin orchestration

Current `pat_toolbox/workflows.py` should become a thin runner over explicit steps.

Suggested split:

- `pat_toolbox/workflows/load_steps.py`
- `pat_toolbox/workflows/metric_steps.py`
- `pat_toolbox/workflows/output_steps.py`
- `pat_toolbox/workflows/runner.py`

Workflow code should mainly do:

- load EDF and aux inputs
- populate context
- run enabled features
- generate configured outputs
- catch per-file failures and continue

This keeps orchestration readable.

## Configuration Strategy

`config.py` should remain the control center, but its internal organization can improve later.

Short term:

- keep one `config.py`
- group settings by feature and responsibility
- document all feature toggles clearly
- document both compute toggles and presentation toggles clearly

Recommended rule for future maintenance:

- computation enablement and report visibility must not be hidden in plotting code
- if a feature is turned off, it should disappear consistently from plots, tables, and summaries according to config

Long term optional split:

- `config_paths.py`
- `config_hr.py`
- `config_hrv.py`
- `config_plotting.py`

This should only happen after the computation modules are cleaner.

## Data Model Improvements

The project currently passes many loose dicts and many parallel arrays.

Recommended gradual improvement:

- keep `RecordingContext` as the session container
- add small dataclasses for important outputs
- avoid many unnamed dict payloads

Good candidates for dataclasses:

- feature results
- mask bundles
- PSD diagnostics
- summary row payloads

This will improve readability and debugging.

## Debugging And Readability Rules

To keep the toolbox simple and usable by others, adopt these rules.

### 1. Pure compute functions should be easy to test mentally

Prefer functions that:

- accept arrays and numeric parameters
- return arrays and summaries
- avoid file access and side effects

### 2. Plotting should never hide computation rules

All metric validity rules should live in compute modules, not in plot modules.

### 3. One source of truth per rule

Examples:

- one RR cleaning implementation
- one masking engine implementation
- one shared time-domain window gate
- one spectral acceptance gate

But where scientific behavior differs, keep separate policy definitions instead of flattening them into one universal rule.

### 4. Keep function names honest

Use explicit names that describe exactly what happens.

Examples:

- `compute_hrv_time_domain_series`
- `compute_hrv_spectral_tv_series`
- `build_rr_mask_bundle`
- `write_batch_summary_row`

## Suggested Implementation Order

This order minimizes risk.

### Phase A: low-risk extraction

- extract shared helpers from large modules
- extract RR/window helpers into `core/`
- do not change behavior yet

### Phase B: reorganize feature computation

- split `hr.py`
- split `hrv.py`
- split `psd.py`
- keep old public entry points delegating to new modules

### Phase C: reorganize plotting

- split HRV figures
- split segment pages
- split summary pages

### Phase D: thin workflow runner

- move loading, metric, and output steps into `workflows/`
- keep `main.py` very small

### Phase E: structured outputs

- replace fragile dict bundles with dataclasses where it helps most

## Definition Of A Good End State

The refactor is successful when:

- adding a new feature needs only a small feature package plus one workflow hook
- toggling a feature is done in config only
- shared logic is not duplicated
- plotting does not contain core computation logic
- the codebase is understandable by a new collaborator without tribal knowledge
- debugging a metric means looking in one obvious place
- refactoring preserves the current numerical outputs unless a new option is explicitly enabled
- feature-specific masking/exclusion behavior is configurable and explicit
- feature visibility in plots, report pages, summary tables, and batch summaries is driven from one central declaration

## Immediate Practical Recommendation

If starting now, the best first refactor sequence is:

1. extract `rr_cleaning.py` and `windows.py`
2. split `io_aux_csv.py`
3. split `hrv.py` into time-domain and freq-domain pieces
4. split `hr.py` into compute vs summary/debug pieces
5. split plotting modules
6. thin out `workflows.py`

This path builds a reusable core toolbox without overengineering too early.
