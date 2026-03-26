# PAT Toolbox

PAT Toolbox processes EDF recordings with VIEW PAT data and produces heart-rate, HRV, PSD, PAT burden, and PDF report outputs. The repository is config-driven: `pat_toolbox/config.py` controls input paths, sleep-stage policy, masking behavior, output folders, and algorithm parameters.

## What it does

- Reads `VIEW_PAT` from EDF and derives beat-to-beat RR intervals.
- Computes PAT-derived HR on a regular time grid.
- Computes HRV metrics from cleaned RR intervals, including RMSSD, SDNN, LF/HF, and time-varying HRV series.
- Applies shared sleep, event, and desaturation masking through `pat_toolbox/masking.py`.
- Generates multi-page PDF reports with signal overlays, event markers, stage context, and summary tables.
- Optionally computes PAT burden from PAT amplitude during masked event regions.

## Repository layout

```text
.
|- main.py
|- requirements.txt
|- analysis/
|  |- boxplots_AHI.py
|  `- boxplots_AHI_groups.py
|- experiments/
|  `- hypnogram_diego.py
`- pat_toolbox/
   |- config.py
   |- context.py
   |- filters.py
   |- io_aux_csv.py
   |- io_edf.py
   |- masking.py
   |- paths.py
   |- sleep_mask.py
   |- workflows.py
   |- metrics/
   |  |- hr.py
   |  |- hrv.py
   |  |- pat_burden.py
   |  |- psd.py
   |  `- hr_delta.py
   `- plotting/
      |- figures_hrv.py
      |- figures_summary.py
      |- peaks_debug.py
      |- report.py
      |- segments.py
      |- specs.py
      `- utils.py
```

## Main workflow

The primary entry point is `main.py`.

At a high level, each EDF goes through:

1. PAT loading from `VIEW_PAT`.
2. PAT filtering with the configured band-pass filter.
3. Optional auxiliary CSV loading and sleep-stage normalization.
4. PAT-derived HR computation.
5. HRV computation from cleaned RR intervals.
6. Optional PAT burden and PSD feature computation.
7. PDF report generation and summary CSV updates.

The orchestration lives in `pat_toolbox/workflows.py` and state is carried in `pat_toolbox/context.py`.

## Inputs

### EDF

Required channel:

- `VIEW_PAT`

Optional channels used when available:

- `DERIVED_HR`
- `DERIVED_PAT_AMP`
- `ACTIGRAPH`

Channel names are configured in `pat_toolbox/config.py`.

### Auxiliary CSV

When present, the per-recording aux CSV is used for:

- sleep stage labels
- SpO2 and desaturation flags
- HR/PAT exclusion flags
- scored event flags such as central, obstructive, and unclassified A/H events

Internal canonical names are mapped through `config.COL_NAMES`.

## Shared masking model

Masking is now centralized in `pat_toolbox/masking.py`.

`MaskPolicy` is built from `config.py` and controls:

- whether sleep-stage masking is active
- which sleep stages are included
- which aux columns are treated as exclusion events
- whether desaturation windows are enabled
- event pre/post padding and desaturation gating parameters

`MaskBundle` exposes aligned boolean masks for a given time base:

- `sleep_keep`
- `event_keep`
- `desat_keep`
- `combined_keep`

This same shared masking logic is used by:

- HR masking in `pat_toolbox/workflows.py`
- RR-level HRV masking in `pat_toolbox/metrics/hrv.py`
- PSD feature computation in `pat_toolbox/metrics/psd.py`
- PAT burden computation in `pat_toolbox/metrics/pat_burden.py`
- plotting/report overlays in `pat_toolbox/plotting/`

## Core computations

### HR

PAT-derived HR is computed from detected PAT peaks after band-pass filtering. RR intervals are cleaned with physiologic bounds and several robustness steps before interpolation to the configured output grid.

Main implementation:

- `pat_toolbox/metrics/hr.py`

### HRV

HRV reuses the cleaned RR intervals and computes:

- RMSSD series on a 1 Hz grid
- global HRV summary values
- time-varying SDNN, LF, HF, and LF/HF
- clean vs sleep-only views for reporting

Main implementation:

- `pat_toolbox/metrics/hrv.py`

### PSD

PSD features are computed from an HRV-matched RR/tachogram view when possible, with fallback whole-tachogram PSD when matched windows are unavailable. The summary output records both the PSD mode and any diagnostic reason.

Main implementation:

- `pat_toolbox/metrics/psd.py`

### PAT burden

PAT burden is estimated from PAT amplitude inside event/desaturation exclusion regions, optionally under alternate sleep-stage policies.

Main implementation:

- `pat_toolbox/metrics/pat_burden.py`

## Reporting outputs

The plotting/report pipeline lives under `pat_toolbox/plotting/` and produces:

- multi-page overlay PDFs
- HRV overview pages
- stagegram plus time-varying HRV pages
- summary tables with HR, HRV, PSD, burden, and aux-event counts
- optional peak-debug PDFs

Recent report behavior includes:

- event keys that reflect active exclusion columns
- HRV mask shading for sleep-only vs fully excluded regions
- PSD summary fields that show matched vs fallback mode

## Important configuration areas

Most day-to-day changes happen in `pat_toolbox/config.py`.

Common sections:

- input/output paths: `EDF_FOLDER`, `BASE_OUTPUT_DIR`
- run naming: `RUN_ID`, `RUN_TAG`
- feature toggles: `ENABLE_VIEW_PAT_OVERLAY_PLOTS`, `ENABLE_HR`, `ENABLE_PAT_PEAK_DEBUG_PLOTS`
- sleep masking: `ENABLE_SLEEP_STAGE_MASKING`, `SLEEP_STAGE_POLICY`
- aux mapping: `COL_NAMES`
- shared exclusion settings: `HRV_EXCLUSION_EVENT_COLUMNS`, `HRV_EXCLUSION_PRE_SEC`, `HRV_EXCLUSION_POST_SEC`, `HRV_EXCLUSION_USE_DESAT_WINDOWS`
- HR parameters: `HR_*`
- HRV parameters: `HRV_*`

Outputs are organized under subfolders derived from the run suffix, for example:

- `ViewPatPlotsOverlay__<run_suffix>`
- `HR__<run_suffix>`
- `PSD__<run_suffix>`

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

### Full pipeline

```bash
python main.py
```

This is the standard batch workflow and uses the settings in `pat_toolbox/config.py`.

### Analysis scripts

```bash
python analysis/boxplots_AHI.py
python analysis/boxplots_AHI_groups.py
python experiments/hypnogram_diego.py
```

Run these from the repository root so relative paths resolve consistently.

## Validation

The repository does not currently include an automated test suite. For a lightweight syntax check, run:

```bash
python -m compileall main.py pat_toolbox analysis experiments
```

If the configured dataset paths are valid on the current machine, `python main.py` is the main smoke test.

## Troubleshooting

- No EDF files found: verify `config.EDF_FOLDER`.
- Empty HR or HRV outputs: check PAT signal quality, RR cleaning thresholds, sleep policy, and exclusion windows.
- PSD values missing: the RR series may be too sparse after masking; inspect PSD diagnostics in the summary output.
- Missing aux-driven overlays: verify aux CSV naming, time column parsing, and `config.COL_NAMES`.

## Notes

- This repository uses machine-specific absolute paths in `pat_toolbox/config.py`; update them deliberately for your environment.
- The run identifier is generated at import time, so output folders change from run to run unless you keep the same generated folder externally.
- The README intentionally avoids embedding a hard-coded config snapshot so it stays accurate as defaults evolve.
