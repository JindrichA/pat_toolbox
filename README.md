# Technical README — PAT/EDF Signal Processing Pipeline

This document describes the **end-to-end processing pipeline** implemented in this repository for extracting **heart rate (HR)** and **heart rate variability (HRV)** metrics from **Peripheral Arterial Tone (PAT)** signals stored in **EDF** files, optionally synchronized with an **auxiliary CSV** (sleep stages + event flags). It is written as a *technical/Methods-style* description suitable for internal documentation and for reproducible analysis.

---

## 1) Repository layout

```
.
├── main.py
└── pat_toolbox/
    ├── config.py
    ├── context.py
    ├── filters.py
    ├── io_edf.py
    ├── io_aux_csv.py
    ├── paths.py
    ├── sleep_mask.py
    ├── workflows.py
    ├── metrics/
    │   ├── hr.py
    │   ├── hrv.py
    │   └── psd.py
    └── plotting/
        ├── report.py
        ├── figures_summary.py
        ├── figures_hrv.py
        ├── peaks_debug.py
        ├── segments.py
        └── ...
```

**Primary entry point**: `main.py`
**Primary pipeline implementation**: `pat_toolbox/workflows.py`
**Signal processing + metrics**: `pat_toolbox/metrics/hr.py`, `pat_toolbox/metrics/hrv.py`
**I/O**: `pat_toolbox/io_edf.py`, `pat_toolbox/io_aux_csv.py`
**Filtering**: `pat_toolbox/filters.py`
**Masking (sleep staging)**: `pat_toolbox/sleep_mask.py`

---

## 2) Data model

### 2.1 EDF inputs

The pipeline expects EDF recordings containing (at minimum) a PAT channel:

* **PAT**: `VIEW_PAT` (configured by `config.VIEW_PAT_CHANNEL_NAME`)

Optional additional EDF channels:

* **EDF-provided HR**: `DERIVED_HR` (configured by `config.HR_CHANNEL_NAME`) — used for *correlation/validation* against PAT-derived HR.
* **PAT amplitude**: `DERIVED_PAT_AMP` (configured by `config.PAT_AMP_CHANNEL_NAME`) — used for overlay in reporting.
* **Actigraphy**: `ACTIGRAPH` (configured by `config.ACTIGRAPH_CHANNEL_NAME`) — used only for peak-debug visualization when enabled.

### 2.2 Auxiliary CSV (optional)

An optional CSV can be associated with each EDF (via `io_aux_csv.read_aux_csv_for_edf`). It is treated as **time-synchronized metadata**, typically including:

* Sleep stage labels (e.g., `WP Stages`) → standardized stage codes (Wake/Light/Deep/REM)
* SpO2 and desaturation flags
* Exclusion flags (e.g., “Exclude PAT”)
* Event markers (e.g., central/obstructive/unclassified A/H)

The canonical internal column mapping is defined in `config.COL_NAMES`.

---

## 3) Configuration and run identity

All run behavior is controlled in `pat_toolbox/config.py`. The design is **config-driven** to support reproducibility.

### 3.1 Input/output paths

* `config.EDF_FOLDER`: folder containing `.edf` files
* `config.BASE_OUTPUT_DIR`: global output root; all run outputs are placed beneath this directory

### 3.2 Run identifier and output subfolders

A unique run identifier is created at import time:

* `config.RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")`
* `config.RUN_TAG`: optional human-readable tag

Output folder names are suffixed using:

* `run_suffix() = RUN_ID + sleep_stage_suffix() + RUN_TAG`

Key output subfolders:

* `config.OUTPUT_SUBFOLDER`: PDF overlays
* `config.HR_OUTPUT_SUBFOLDER`: HR/summary outputs
* `config.PSD_OUTPUT_SUBFOLDER`: PSD outputs (if used)

### 3.3 Feature toggles

The main runtime toggles used by `main.py`:

* `ENABLE_VIEW_PAT_OVERLAY_PLOTS` (PDF reporting pipeline)
* `ENABLE_HR` (HR CSV-only mode when plots disabled)
* `ENABLE_PAT_PEAK_DEBUG_PLOTS` (debug PDF with peak markers)

### 3.4 Sleep-stage masking policy

When auxiliary CSV is available and masking is enabled:

* `ENABLE_SLEEP_STAGE_MASKING = True/False`
* `SLEEP_STAGE_POLICY ∈ {all_sleep, rem_only, nrem_only, deep_only, nrem_light_only, custom}`

Internally, the pipeline converts stage labels to numeric stage codes:

* `0 = Wake`, `1 = Light`, `2 = Deep`, `3 = REM`

The include set is resolved by `config.sleep_include_numeric()`.

---

## 4) Execution flow (top-level)

### 4.1 main.py orchestration

`main.py` does the following:

1. Reads configuration (EDF folder, enabled features, segment length, filter settings).
2. Optionally prevents macOS sleep using `caffeinate`.
3. Lists EDF files using `io_edf.list_edf_files(EDF_FOLDER)`.
4. For each EDF file, runs one or more actions:

* If `ENABLE_VIEW_PAT_OVERLAY_PLOTS`: call
  `workflows.process_view_pat_overlay_for_file(edf_path)`
* Else if `ENABLE_HR`: compute PAT HR and save CSV via
  `hr.compute_hr_for_edf_file(edf_path, save_csv=True)`
* If `ENABLE_PAT_PEAK_DEBUG_PLOTS` and workflow didn’t already run:
  `hr.create_peaks_debug_pdf_for_edf(edf_path)`

The workflow path is preferred because it produces the full scientific report and summary outputs.

---

## 5) Detailed signal-processing pipeline

The core end-to-end workflow is implemented in:

* `pat_toolbox/workflows.py::process_view_pat_overlay_for_file(edf_path)`

The steps below describe what happens for each EDF recording.

### Step A — Load PAT (`VIEW_PAT`) from EDF

**Function**: `_load_pat(ctx)`
**I/O**: `io_edf.read_edf_channel(edf_path, VIEW_PAT_CHANNEL_NAME)`

* Reads PAT samples and sampling frequency `fs`.
* Validates: non-empty signal and `fs > 0`.

### Step B — Band-pass filter PAT

**Function**: `_filter_pat(ctx)`
**I/O**: `filters.bandpass_filter(view_pat, fs)`

Band-pass parameters (config-driven):

* Low cut: `PAT_BANDPASS_LOWCUT_HZ` (default 0.5 Hz)
* High cut: `PAT_BANDPASS_HIGHCUT_HZ` (default 8.0 Hz)
* Filter order: `PAT_BANDPASS_ORDER` (default 4)

**Rationale**: isolate pulsatile content while suppressing baseline drift and high-frequency noise.

### Step C — Load optional channels (PAT amplitude)

**Function**: `_load_pat_amp(ctx)`
**I/O**: `io_edf.read_edf_channel(edf_path, PAT_AMP_CHANNEL_NAME)`

* If present and valid, creates a time axis `t_pat_amp = np.arange(n)/fs_amp`.
* Used for contextual overlay/quality review.

### Step D — Load optional auxiliary CSV and derive sleep stage code

**Function**: `_load_aux_csv(ctx)`
**I/O**: `io_aux_csv.read_aux_csv_for_edf(edf_path)`

* Loads time-synchronized metadata.
* Ensures numeric stage column exists using:
  `sleep_mask.ensure_stage_code_column(aux_df)`.

### Step E — Compute PAT-derived HR (1 Hz grid)

**Function**: `_compute_hr_from_pat(ctx)`
**I/O**: `hr_metrics.compute_hr_from_pat_signal(view_pat, fs)`

This step is scientifically central; it converts a pulsatile PAT waveform into heart rate.

#### E1) Peak detection on band-passed PAT

**Function**: `hr._detect_pat_peaks(pat_signal, fs)`

1. Band-pass filters PAT (same filtering concept as Step B).
2. Uses `scipy.signal.find_peaks` with:

   * **minimum peak distance** derived from physiologic minimum RR:
     `min_rr_samples = HR_MIN_RR_SEC * fs`
   * **prominence** scaled to signal dispersion:
     `prom = HR_PEAK_PROMINENCE_FACTOR * std(pat_filt)`
3. Performs a lightweight sanity filter on RR intervals.

#### E2) RR extraction and multi-stage RR cleaning

**Function**: `hr.extract_clean_rr_from_pat(pat_signal, fs)`

From detected peaks:

* Peak times: `t_peak = peaks / fs`
* RR intervals: `rr_sec = diff(t_peak)`
* RR mid-times (for time-tagging intervals):
  `rr_mid = 0.5 * (t_peak[i] + t_peak[i+1])`

Cleaning stages:

1. **Physiologic RR limits**: keep only
   `HR_MIN_RR_SEC ≤ RR ≤ HR_MAX_RR_SEC`
2. **Local median outlier rejection**:

   * rolling median (kernel `HR_RR_MEDFILT_KERNEL`)
   * reject if `|RR - median| / median > HR_RR_OUTLIER_REL_THR`
3. **Gap masking**:

   * reject “very long” RR relative to median: `RR ≤ HR_RR_GAP_FACTOR * median`
4. **Abrupt RR jump masking**:

   * rejects transient mis-detections via relative jump threshold `HR_RR_JUMP_REL_THR`
5. **Alternans pair rejection** (double-peak + missed-peak signature):

   * marks short+long or long+short adjacent RR pairs using
     `HR_RR_ALT_SHORT_REL`, `HR_RR_ALT_LONG_REL`
6. **Minimum contiguous run requirement**:

   * keeps only runs of at least `HR_RR_MIN_GOOD_RUN` intervals

**Outputs**:

* `rr_sec_clean`, `rr_mid_clean`, and total duration

#### E3) Convert to instantaneous HR and resample (gap-aware)

**Function**: `hr.compute_hr_from_pat_signal(...)`

* Instantaneous HR (bpm): `inst_hr = 60 / rr_sec_clean`
* Defines a regular time grid:
  `t_hr = np.arange(0, duration, 1/HR_TARGET_FS_HZ)` (default 1 Hz)
* Interpolates **without bridging gaps** via `_interp_with_gaps(...)`:

  * splits at gaps where `diff(rr_mid) > HR_MAX_RR_GAP_SEC`
  * interpolates only within contiguous runs; leaves gaps as NaN

#### E4) HR post-processing

* Optional NaN-aware moving average smoothing:
  `HR_SMOOTHING_WINDOW_SEC`
* Hard HR clamps:
  `HR_MIN_BPM ≤ HR ≤ HR_MAX_BPM` (finite-only)
* Optional Hampel despiking:

  * `HR_HAMPEL_WINDOW_SEC`, `HR_HAMPEL_SIGMA`
  * only applied if NaN fraction is small
* Optional slope limiter:
  `HR_MAX_DELTA_BPM_PER_SEC`

#### E5) Sleep-stage masking (time-domain)

If aux CSV exists and masking is enabled:

* Build mask on HR time axis: `sleep_mask.build_sleep_include_mask(t_hr, aux_df)`
* Apply mask in-place: `sleep_mask.apply_sleep_mask_inplace(hr, mask)`

Masked samples become NaN (excluded from analysis and plotting).

### Step F — Load EDF-derived HR (optional) and apply masking

**Function**: `_load_hr_from_edf(ctx)`

* Reads EDF `DERIVED_HR` and scales to bpm using
  `HR_EDF_SCALE_FACTOR`.
* Applies sleep-stage masking in the same way as PAT HR.

### Step G — HR correlation (validation vs EDF HR)

**Function**: `_compute_hr_correlation(ctx)` → `hr.compute_hr_correlation(...)`

* Builds a common 1 Hz time grid over overlapping time support.
* Interpolates both series onto this grid.
* Computes:

  * Pearson correlation (linear agreement)
  * Spearman correlation (rank agreement)
  * RMSE (bpm)

### Step H — HRV computation from PAT (RMSSD + LF/HF)

**Function**: `_compute_hrv(ctx)` → `hrv.compute_hrv_from_pat_signal_with_tv_metrics(...)`

This step reuses **the same RR extraction/cleaning** as HR.

#### H1) RR extraction (shared)

* Calls `hr.extract_clean_rr_from_pat(pat_signal, fs)`

#### H2) RR-level sleep-stage masking

Sleep masking is applied at the **RR mid-time level**, which is methodologically cleaner than masking the final metric series.

* `sleep_mask.build_sleep_include_mask_for_times(rr_mid, aux_df)`
* filters `rr_mid` and `rr_ms` directly

#### H3) RMSSD series (sliding window, gap-aware)

**Function**: `_calculate_rmssd_series(t_hrv, rr_mid, rr_ms, window_sec, ...)`

* Time grid: `t_hrv = np.arange(0, duration, 1/HRV_TARGET_FS_HZ)` (default 1 Hz)
* Uses a **two-pointer sliding window** (O(N)) to select RR intervals inside each window.
* Window validity constraints (config-driven):

  * minimum RR count: `HRV_MIN_INTERVALS_PER_WINDOW`
  * reject if window spans RR gaps: `HRV_MAX_RR_GAP_SEC`
  * minimum temporal span: `HRV_RMSSD_MIN_SPAN_SEC`
  * optional minimum coverage fraction: `HRV_MIN_WINDOW_COVERAGE`
  * optional veto if too many huge diffs: `HRV_RMSSD_BIGDIFF_*`

RMSSD itself is computed robustly:

* Hard cap on |ΔRR|: `HRV_RMSSD_DIFF_HARD_CAP_MS`
* MAD-based rejection on ΔRR: `HRV_RMSSD_DIFF_MAD_SIGMAS`
* Minimum diffs: `HRV_RMSSD_MIN_DIFFS`
* Floor: RMSSD < `HRV_RMSSD_FLOOR_MS` → NaN

Two RMSSD series are produced:

* **RAW**: sleep-masked RR only (before event exclusion)
* **CLEAN**: sleep-masked RR plus event-based exclusion

#### H4) Event-based exclusion on RR intervals (aux CSV)

If aux CSV exists, RR mid-times are excluded using:

* `io_aux_csv.get_rr_exclusion_mask(rr_mid, aux_df)`

This applies pre/post exclusion around configured events and/or desaturation windows:

* `HRV_EXCLUSION_EVENT_COLUMNS`
* Fixed windows: `HRV_EXCLUSION_PRE_SEC`, `HRV_EXCLUSION_POST_SEC`
* Desat-window mode: `HRV_EXCLUSION_USE_DESAT_WINDOWS` and pads

#### H5) Global HRV summary on CLEAN RR

Computed on RR after **sleep masking + event exclusion**:

* SDNN: sample std of RR (ms)
* LF/HF using Welch PSD of a **resampled tachogram**:

  * base method: `_lf_hf_from_rr(rr_ms, rr_mid, fs_resample)`
  * *gap-robust* method: `_lf_hf_from_rr_segmented(...)`

    * splits into contiguous RR runs (no interpolation across gaps)
    * duration-weighted averaging across runs

Bands:

* LF: 0.04–0.15 Hz
* HF: 0.15–0.40 Hz

Units:

* LF and HF are returned in **ms²** (by converting from s² → ms²).

Diagnostics:

* number of contiguous segments used for LF/HF

#### H6) “Publication-style” fixed-window LF/HF (Option A)

Additionally computed:

* `_calculate_lfhf_fixed_windows(...)` produces LF/HF in fixed windows
  (default 120 s windows, non-overlapping).

Summary includes:

* median and mean LF/HF across valid windows
* number of valid windows

#### H7) Time-varying HRV metrics (TV)

A second set of windowed metrics is computed on the CLEAN RR set:

* RMSSD(t), SDNN(t), LF(t), HF(t), LF/HF(t)

These are evaluated on the same 1 Hz grid using `_calculate_hrv_windowed_series(...)`.

#### H8) Time-domain masking for plotting

After RR-level computations, time-series are additionally masked for plotting using:

* `io_aux_csv.build_time_exclusion_mask(t_hrv, aux_df)`
* `sleep_mask.build_sleep_include_mask(t_hrv, aux_df)`

This ensures the plotted HRV series respect both sleep stage policy and event exclusion windows.

#### H9) HRV CSV output

Saved via:

* `save_hrv_series_to_csv(edf_path, t_hrv, rmssd_clean)`

Output columns:

* `time_sec, rmssd_ms`

### Step I — PDF reporting and spectral features

**Function**: `_build_pdf(ctx)` → `plotting.plot_pat_and_hr_segments_to_pdf(...)`

The report produces **multi-page PDFs** containing segmented overlays (default `SEGMENT_MINUTES = 15`).

Inputs plotted (when available):

* raw PAT and filtered PAT
* PAT-derived HR and EDF HR
* HRV series (raw + clean), and time-varying metrics
* auxiliary signals/events (from aux CSV)
* PAT amplitude overlay (if present)

The plotting function also returns a dictionary of spectral features (PSD-related), which are stored in:

* `ctx.psd_features`

Common outputs captured in context:

* `mayer_peak_hz`
* `resp_peak_hz`
* band powers and normalized band powers (if computed)

### Step J — Optional peaks debug PDF

**Function**: `_build_peaks_debug_pdf(ctx)` → `hr.create_peaks_debug_pdf_for_edf(edf_path)`

Produces a 1-minute-per-page PDF showing:

* PAT raw and filtered
* detected peaks (markers)
* optional ACTIGRAPH motion envelope

Actigraphy is preprocessed as a motion envelope:

1. high-pass
2. absolute value (rectify)
3. low-pass smoothing
4. optional z-score normalization

### Step K — Append per-file summary row

**Function**: `_append_summary(ctx)` → `hr.append_hr_correlation_to_summary(...)`

Appends one row per EDF recording to a summary CSV including:

* HR correlation metrics (Pearson, Spearman, RMSE)
* HRV summary metrics (RMSSD mean/median, SDNN, LF, HF, LF/HF)
* LF/HF diagnostics (segments used) and fixed-window LF/HF summary
* spectral peaks and PSD features (if available)
* quality indicators (NaN% for HR/HRV series)
* aux flag counts (events, desats, exclusions)
* sleep stage distribution and included/excluded percentages

The summary schema can auto-upgrade when new columns appear.

---

## 6) Outputs

All results are written under:

* `BASE_OUTPUT_DIR / <subfolder>`

Typical artifacts:

1. **PDF report** (per EDF):

   * `.../ViewPatPlotsOverlay__<run_suffix>/<edf_base>__VIEW_PAT_HR_HRV_<seg>min_overlay_<policy>.pdf`
2. **HR CSV** (optional per EDF):

   * `.../HR__<run_suffix>/<edf_base>__HR_1Hz.csv`
3. **HRV RMSSD CSV** (per EDF):

   * `.../<HRV_subfolder>/<edf_base>__HRV_RMSSD_1Hz_Clean.csv`
4. **Peaks debug PDF** (optional per EDF):

   * `.../<run>/<edf_base>__PAT_Peaks_<mins>min.pdf`
5. **Batch summary CSV** (append-only):

   * `.../HR__<run_suffix>/HR_HRV_summary__<policy>__<run_id>.csv`

---

## 7) How to run

### 7.1 Configure paths

Edit `pat_toolbox/config.py`:

* `EDF_FOLDER = Path("...")`
* `BASE_OUTPUT_DIR = Path("...")`

Optionally set:

* `RUN_TAG` (short description)
* `SLEEP_STAGE_POLICY` (e.g., `rem_only`)

### 7.2 Run batch processing

```bash
python main.py
```

### 7.3 Typical runtime modes

* **Full pipeline (recommended)**:

  * `ENABLE_VIEW_PAT_OVERLAY_PLOTS = True`
  * Produces PDFs + HR/HRV + summary CSV

* **HR-only CSV mode**:

  * `ENABLE_VIEW_PAT_OVERLAY_PLOTS = False`
  * `ENABLE_HR = True`
  * Produces per-file HR CSV (no PDF workflow)

* **Peak debug mode**:

  * `ENABLE_PAT_PEAK_DEBUG_PLOTS = True`
  * Produces per-file peak debug PDFs

---

## 8) Scientific notes and design choices

### 8.1 Gap-aware processing

PAT peak detection can fail in motion/artifact segments. To avoid fabricating physiology:

* HR interpolation does **not** bridge gaps larger than `HR_MAX_RR_GAP_SEC`.
* HRV LF/HF uses segmented PSD to avoid interpolating across gaps.
* Sliding-window RMSSD rejects windows spanning RR gaps.

### 8.2 RR-level masking

Sleep stages are applied at the RR-level (mid-times), which:

* avoids window contamination
* preserves correct time-tagging for HRV windows

### 8.3 Robustification against PAT peak glitches

The RR pipeline is hardened via:

* physiologic RR bounds
* local median outlier rejection
* gap masking vs local median
* abrupt jump rejection
* alternans signature rejection
* minimum contiguous run requirement

The RMSSD pipeline further rejects implausible RR-diff bursts.

---

## 9) Key parameters (quick reference)

### PAT filtering

* `PAT_BANDPASS_LOWCUT_HZ`, `PAT_BANDPASS_HIGHCUT_HZ`, `PAT_BANDPASS_ORDER`

### Peak / RR cleaning

* `HR_MIN_RR_SEC`, `HR_MAX_RR_SEC`
* `HR_PEAK_PROMINENCE_FACTOR`
* `HR_RR_MEDFILT_KERNEL`, `HR_RR_OUTLIER_REL_THR`
* `HR_RR_GAP_FACTOR`, `HR_RR_JUMP_REL_THR`
* `HR_RR_ALT_SHORT_REL`, `HR_RR_ALT_LONG_REL`
* `HR_RR_MIN_GOOD_RUN`

### HR grid + post-processing

* `HR_TARGET_FS_HZ`
* `HR_MAX_RR_GAP_SEC`
* `HR_SMOOTHING_WINDOW_SEC`
* `HR_HAMPEL_WINDOW_SEC`, `HR_HAMPEL_SIGMA`
* `HR_MAX_DELTA_BPM_PER_SEC`
* `HR_MIN_BPM`, `HR_MAX_BPM`

### HRV time-domain

* `HRV_TARGET_FS_HZ`
* `HRV_WINDOW_SEC`
* `HRV_MIN_INTERVALS_PER_WINDOW`
* `HRV_MAX_RR_GAP_SEC`
* `HRV_RMSSD_*` parameters (hard cap, MAD, floors)

### HRV frequency-domain

* `HRV_TACHO_RESAMPLE_HZ`
* `HRV_MIN_FREQ_DOMAIN_SEC`

### Sleep masking

* `ENABLE_SLEEP_STAGE_MASKING`
* `SLEEP_STAGE_POLICY`

### Event exclusion

* `HRV_EXCLUSION_EVENT_COLUMNS`
* `HRV_EXCLUSION_PRE_SEC`, `HRV_EXCLUSION_POST_SEC`
* `HRV_EXCLUSION_USE_DESAT_WINDOWS` and related padding parameters

---

## 10) Troubleshooting

* **No EDF files found**: confirm `config.EDF_FOLDER` exists and contains `.edf` files.
* **Empty PAT signal**: check the EDF channel name (`VIEW_PAT_CHANNEL_NAME`).
* **All NaN HR/HRV**:

  * peak detection may be failing (motion/noise)
  * physiologic bounds may be too strict
  * sleep-stage masking policy may be excluding most samples
  * event exclusion may be excluding everything
* **LF/HF is NaN**:

  * insufficient contiguous RR span (`HRV_MIN_FREQ_DOMAIN_SEC`)
  * too many RR gaps
* **Peak debug PDF**: enable `ENABLE_PAT_PEAK_DEBUG_PLOTS` and inspect pages where HR collapses.

---

## 11) Reproducibility checklist

For a fully reproducible run, record:

* `RUN_ID`, `RUN_TAG`
* `SLEEP_STAGE_POLICY` and whether masking is enabled
* all filter and RR cleaning parameters
* code commit hash
* EDF source dataset version

---

## 12) Citation-ready method summary (copy/paste)

**PAT-derived HR** was estimated by band-pass filtering the PAT waveform (0.5–8 Hz, 4th order) and detecting pulse peaks via prominence-constrained peak finding with physiologically constrained minimum inter-peak distance. RR intervals were computed from successive peaks, time-tagged by RR mid-times, and cleaned using a multi-stage procedure including physiologic bounds, local-median outlier rejection, gap and jump rejection, alternans-pair rejection, and a minimum contiguous run requirement. Instantaneous HR (60/RR) was interpolated onto a 1 Hz grid without bridging large RR gaps, then optionally smoothed and despiked.

**HRV** was computed on the same cleaned RR intervals. Sleep stages from an auxiliary time-synchronized CSV were applied at the RR level. Sliding-window RMSSD was computed on a 1 Hz grid using gap-aware windows and robust outlier rejection on successive RR differences. Frequency-domain HRV (LF/HF) was obtained from Welch PSD of a resampled tachogram, with contiguous-run segmentation to avoid interpolation across RR gaps; LF and HF power were integrated over 0.04–0.15 Hz and 0.15–0.40 Hz, respectively.
