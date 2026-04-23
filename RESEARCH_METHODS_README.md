# PAT Toolbox Research Methods README

This document describes the current signal-processing workflow used in this repository, using the parameters from the present configuration in `pat_toolbox/config.py`.

It is written as a methods-style reference for research reporting. It explains, step by step, how features are derived from the EDF and auxiliary CSV inputs, how exclusion is applied, how sleep-stage policies are handled, and which calculations are active in the current setup.

## Current Active Setup

The current `FEATURES` configuration is:

- `hr = True`
- `hrv = True`
- `psd = False`
- `delta_hr = False`
- `pat_burden = False`
- `sleep_combo_summary = True`
- `report_pdf = True`
- `peaks_debug_pdf = False`

This means the current run produces:

- PAT-derived HR
- HRV time-domain features
- HRV frequency-domain features based on RR intervals
- sleep-subset comparison summaries
- the main PDF report

The current sleep-stage policy is:

- `SLEEP_STAGE_POLICY = "nrem_only"`
- included sleep stages = `{1, 2}`

Under the current stage mapping:

- `0 = Wake`
- `1 = Light sleep`
- `2 = Deep sleep`
- `3 = REM`

Therefore, in the main selected-policy analysis, only light sleep and deep sleep are included. Wake and REM are excluded from the main selected-policy HR and HRV analysis.

## Input Data

### EDF input

For each recording, the workflow reads the EDF channel:

- `VIEW_PAT`

This is the PAT waveform used for peak detection, RR extraction, HR, and HRV.

The code uses the sampling frequency stored in the EDF for that channel. In practice, your recordings may use a 40 Hz PAT signal, but the algorithm itself uses the EDF-native sampling frequency rather than a hard-coded value.

### Auxiliary CSV input

If available, a matching auxiliary CSV is loaded. The current mapped columns are:

- stage column: `WP Stages`
- desaturation flag: `Desaturation`
- HR quality flag: `Exclude HR`
- PAT quality flag: `Exclude PAT`
- central respiratory event flag: `A/H central-3% (Last second)`
- obstructive respiratory event flag: `A/H obstructive-3% (Last second)`
- unclassified respiratory event flag: `A/H unclassified-3% (Last second)`

The auxiliary time column is parsed into:

- `time_sec`

The sleep stage labels are normalized into numeric stage codes:

- `WK -> 0`
- `L. Sleep -> 1`
- `D. Sleep -> 2`
- `REM -> 3`

## Overall Workflow

For each EDF file, the processing order is:

1. Read the PAT channel from EDF.
2. Band-pass filter the PAT waveform.
3. Optionally load PAT amplitude if burden analysis is enabled.
4. Load and normalize the auxiliary CSV.
5. Compute sleep-subset summaries.
6. Compute PAT burden if enabled.
7. Compute PAT-derived HR.
8. Compute HRV and RR-based summaries.
9. Compute separate PSD features if enabled.
10. Export per-feature CSVs.
11. Build the PDF report.
12. Append one row to the grouped summary CSV.

## PAT Preprocessing

Before peak detection, the PAT waveform is band-pass filtered with a Butterworth band-pass filter using:

- low cut: `0.5 Hz`
- high cut: `8.0 Hz`
- filter order: `4`

Filtering is done with zero-phase forward-backward filtering (`filtfilt`), so no intentional phase shift is introduced.

## Peak Detection And RR Extraction

PAT peak detection is performed on the filtered PAT waveform.

### Peak detection settings

- minimum RR interval: `0.30 s`
- maximum RR interval: `2.50 s`
- peak prominence factor: `0.30 x SD(filtered PAT)`

Operationally, the detector:

1. Finds local PAT peaks with a minimum peak-to-peak distance corresponding to `0.30 s`.
2. Uses a prominence threshold equal to `0.30 x` the standard deviation of the filtered PAT waveform when the filtered signal has nonzero variance.
3. Converts detected peak times into beat-to-beat RR intervals.
4. Keeps only RR intervals within the physiologic range `0.30 to 2.50 s`.

RR mid-times are defined as the midpoint between consecutive peaks.

## Low-Level RR Cleaning

After initial RR extraction, the repository applies an additional low-level RR cleaning pass before HR and HRV are derived.

### RR cleaning parameters

- median filter kernel: `5`
- RR relative outlier threshold: `0.30`
- RR gap factor: `2.4`
- RR jump relative threshold: `0.6`
- alternans short relative threshold: `0.25`
- alternans long relative threshold: `0.35`
- minimum retained contiguous good run length: `3 RR intervals`

### RR cleaning procedure

The cleaned RR stream is produced as follows:

1. Compute the raw RR sequence from consecutive PAT peaks.
2. Remove RR intervals outside the physiologic range `0.30 to 2.50 s`.
3. Compute a local median RR using a median filter of length `5`.
4. Mark an RR interval as bad if its relative deviation from the local median exceeds `30%`.
5. Mark an RR interval as bad if it exceeds `2.4 x` the local median RR.
6. Mark abrupt jumps as bad when the relative jump between consecutive RR intervals exceeds `0.6`.
7. Reject short-long or long-short alternans-like pairs using:
   - short threshold: `< (1 - 0.25) x local median`
   - long threshold: `> (1 + 0.35) x local median`
8. After all point-wise rejection, keep only contiguous runs of good RR intervals of length at least `3`.

This produces the shared physiologically cleaned RR series used by HR and HRV.

## Sleep-Stage Policy

The main selected-policy analysis uses:

- `nrem_only = {1, 2}`

This means:

- included: light sleep and deep sleep
- excluded: wake and REM

### How sleep-stage masking is applied

The code maps each analysis time point or RR midpoint to the nearest auxiliary stage sample in time.

For the selected-policy analysis:

- if the mapped stage is light or deep, the sample is kept by the sleep mask
- if the mapped stage is wake or REM, the sample is rejected by the sleep mask

### Sleep subsets used in sleep-combo summaries

In addition to the selected-policy analysis, the repository computes fixed subset summaries for:

- `pre_sleep_wake = {0}` before first sleep onset only
- `all_sleep = {1, 2, 3}`
- `wake_sleep = {0, 1, 2, 3}`
- `nrem = {1, 2}`
- `deep = {2}`
- `rem = {3}`

These are separate summaries, not alternative masks applied to the main selected-policy outputs.

## Event And Quality Exclusion Logic

After sleep-stage selection, the code applies event and quality exclusions using the auxiliary CSV.

### Active exclusion columns in the current configuration

- `evt_central_3`
- `evt_obstructive_3`
- `evt_unclassified_3`
- `exclude_hr_flag`
- `exclude_pat_flag`

The desaturation flag is not used as a standalone fixed exclusion column. Instead, desaturations are used through gated desaturation windows, described below.

### Fixed event padding

For active exclusion events and quality flags, the current fixed exclusion window is:

- pre-event padding: `15 s`
- post-event padding: `30 s`

This means that each flagged event time removes the interval:

- `[event time - 15 s, event time + 30 s]`

### Separation of event types inside the mask

Internally, the exclusion columns are split into:

- apnea/event columns: all `evt_*`
- quality columns: all `exclude_*`

The code builds:

- `apnea_keep`
- `quality_keep`
- `event_keep = apnea_keep AND quality_keep`

### Desaturation-gated exclusion windows

The current configuration also enables desaturation-gated masking:

- `HRV_EXCLUSION_USE_DESAT_WINDOWS = True`
- desaturation column: `desat_flag`
- desaturation start padding: `15 s`
- desaturation end padding: `0 s`
- minimum desaturation run: `5.0 s`
- event lookback for gating: `0.0 s`
- event lookahead for gating: `0.0 s`

Operationally:

1. Consecutive desaturation samples are grouped into runs.
2. If a desaturation run lasts at least `5 s`, it becomes one exclusion window.
3. Shorter runs are converted into per-sample exclusion windows.
4. Each desaturation window is padded by `15 s` at the start and `0 s` at the end.
5. A desaturation window is only activated if at least one active exclusion event falls inside the exact desaturation window because both lookback and lookahead are set to `0 s`.

This is a conservative event-gated desaturation logic. Desaturation alone does not exclude data unless it coincides with an active event window under this gating rule.

### Final combined mask

For time-grid or RR-midpoint analyses, the final selected-policy keep mask is:

- `combined_keep = sleep_keep AND event_keep AND desat_keep`

Therefore a sample is kept only if:

1. it belongs to the selected sleep stages
2. it is outside the padded event windows from respiratory events and quality flags
3. it is outside any active gated desaturation windows

## PAT-Derived HR Calculation

Heart rate is derived from the cleaned RR intervals.

### HR parameters

- output sampling rate: `1.0 Hz`
- HR clamp: `30 to 220 bpm`
- HR interpolation maximum gap: `2.5 s`
- HR smoothing window: `4.0 s`
- HR Hampel window: `8.0 s`
- HR Hampel sigma: `2.0`
- maximum HR slope limit: `10 bpm/s`

### HR derivation procedure

1. Convert the cleaned RR intervals to instantaneous HR using `HR = 60 / RR`.
2. Interpolate instantaneous HR to a regular `1 Hz` grid.
3. Do not interpolate across RR gaps larger than `2.5 s`.
4. Smooth the interpolated HR with a moving average over `4 s`.
5. Clamp the smoothed HR to `30 to 220 bpm`.
6. Apply a Hampel despiking filter with an `8 s` window and `2 sigma`, but only when the NaN fraction is below `5%`.
7. Clamp the despiked HR again to `30 to 220 bpm`.
8. Apply a slope limiter so that adjacent 1 Hz samples do not change by more than `10 bpm/s`.
9. Apply the selected-policy `combined_keep` mask, setting excluded samples to `NaN`.

The final selected-policy HR summary statistics are computed from the finite HR samples remaining after this masking.

## HRV Calculation Overview

The HRV feature uses the shared cleaned RR intervals and then creates two distinct families of outputs:

1. time-domain HRV outputs
2. frequency-domain HRV outputs

These two families do not use exactly the same window definition in the current setup.

## Time-Domain HRV: RMSSD And SDNN

### Core time-domain settings

- HRV output grid: `1.0 Hz`
- main HRV window: `300 s` (`5.0 min`)
- minimum RR intervals per window: `6`
- maximum RR gap inside HRV windows: `8.0 s`
- minimum time span inside an HRV window: `5.0 s`
- minimum window coverage fraction: `0.2`
- RMSSD smoothing window: `5.0 s`

### RMSSD robustness settings

- hard cap on successive RR differences: `400 ms`
- MAD cutoff: `4.0 sigma`
- minimum surviving RR differences: `3`
- RMSSD floor: `2.0 ms`
- large-difference veto: `False`
- if enabled, the unused veto parameters would be:
  - big difference threshold: `300 ms`
  - max fraction of large differences: `0.35`

### RMSSD derivation procedure

For each `1 Hz` analysis time point:

1. Center a `300 s` window on that time.
2. Gather RR intervals whose midpoints fall inside the window.
3. Require the shared time-domain window gate to pass:
   - at least `6` RR intervals
   - no RR gap larger than `8.0 s`
   - RR midpoint span at least `5.0 s`
   - span at least `20%` of the `300 s` window because `HRV_MIN_WINDOW_COVERAGE = 0.2`
4. Compute successive RR differences.
5. Remove RR differences whose absolute value exceeds `400 ms`.
6. Compute the median and MAD of the remaining differences.
7. Remove differences farther than `4.0 x robust sigma` from the median.
8. Require at least `3` remaining RR differences.
9. Compute RMSSD as the square root of the mean squared successive difference.
10. Reject the value if RMSSD is below `2.0 ms`.

This is done twice:

- `rmssd_raw`: after sleep-stage masking only
- `rmssd_clean`: after the full selected-policy combined mask

The main report and summary tables use the final clean signal.

### SDNN derivation procedure

The time-varying SDNN series is computed on the same `300 s` sliding windows used for RMSSD.

For each valid window:

- `SDNN = standard deviation of RR intervals in ms`

In addition to the time-varying series, the summary tables report:

- `SDNN mean`: mean of valid sliding-window SDNN values
- `SDNN median`: median of valid sliding-window SDNN values

## Frequency-Domain HRV: LF, HF, LF/HF

### Fundamental spectral settings

- tachogram resampling frequency: `4.0 Hz`
- LF band: `0.04 to 0.15 Hz`
- HF band: `0.15 to 0.40 Hz` for LF/HF computation

### Main reported spectral summary settings

The current selected-policy spectral summary uses fixed windows, not the same `5 min` time-domain windows.

- fixed window length: `120 s` (`2.0 min`)
- fixed hop: `120 s`
- minimum RR intervals per fixed spectral window: `0` additional requirement beyond the built-in minimum of `4`
- maximum RR gap in fixed spectral windows: `3.0 s`
- minimum retained span in a fixed spectral window: at least `80%` of the window, therefore at least `96 s`

### Spectral computation within one valid fixed window

For each valid fixed window:

1. Extract RR intervals whose midpoints fall inside the fixed window.
2. Require at least `4` RR intervals.
3. Reject the window if any RR gap exceeds `3.0 s`.
4. Reject the window if the RR midpoint span is below `96 s`.
5. Resample the tachogram to `4.0 Hz`.
6. Compute the PSD using Welch spectral estimation.
7. Integrate LF power over `0.04 to 0.15 Hz`.
8. Integrate HF power over `0.15 to 0.40 Hz`.
9. Convert both to `ms^2`.
10. Compute `LF/HF = LF / HF` when HF is positive.

### Summary spectral values reported in the current setup

The selected-policy spectral summary reports:

- `LF mean [ms^2]`: arithmetic mean of valid fixed-window LF values
- `LF median [ms^2]`: median of valid fixed-window LF values
- `HF mean [ms^2]`: arithmetic mean of valid fixed-window HF values
- `HF median [ms^2]`: median of valid fixed-window HF values
- `LF/HF mean [-]:` arithmetic mean of valid fixed-window LF/HF ratios
- `LF/HF median [-]:` median of valid fixed-window LF/HF ratios

Importantly:

- `LF/HF mean` is the mean of the per-window ratio values
- it is not forced to equal `LF mean / HF mean`

### Legacy segmented spectral summary

The code still contains a segmented spectral helper that pools contiguous clean RR runs of at least `120 s` and weights them by duration.

However, the main exported `lf`, `hf`, and `lf_hf` summary values are currently overwritten to follow the fixed-window analysis, not the legacy segmented summary.

## Time-Varying Spectral Plots In The Current Setup

The current report plotting was recently aligned with the actual reported spectral summary.

Therefore:

- RMSSD and SDNN plots use the `5 min` sliding-window time-domain analysis
- plotted `LF`, `HF`, and `LF/HF` traces use the same `2 min` fixed-window spectral analysis used by the selected-policy summary tables

This means the spectral plot pages and the spectral summary table now correspond to the same window definition.

## Separate PSD Feature Block

The dedicated `psd` feature is currently disabled:

- `psd = False`

So in the present setup:

- no dedicated PAT PSD pages are generated
- no separate PSD summary fields are expected in the main output as a run feature

If enabled, that feature would compute averaged RR-based PSDs over the same fixed `120 s` windows and report Mayer-band and respiratory-band power summaries.

## Sleep Timing And Sleep-Half Analysis

Sleep timing is computed from the auxiliary sleep-stage timeline.

Definitions:

- sleep onset: first time with stage in `{1, 2, 3}`
- sleep end: last time with stage in `{1, 2, 3}`, plus one auxiliary sample interval
- sleep midpoint: midpoint between sleep onset and sleep end

In the current midpoint-half HRV comparison:

- the half analysis is restricted to NREM only, using stages `{1, 2}`
- the NREM RR stream is cut into first and second halves relative to the sleep midpoint
- each half is then summarized with the same selected-policy HRV summary engine

This is why the comparison table is labeled:

- `NREM first half`
- `NREM second half`

## Sleep-Subset Comparison Summaries

Because `sleep_combo_summary = True`, the pipeline recomputes metrics separately for predefined subsets.

For each subset, the code can produce:

- sleep hours
- HR mean, median, standard deviation
- RMSSD mean and median
- SDNN mean and median
- LF mean
- HF mean
- LF/HF mean

In the current setup, because `delta_hr`, `pat_burden`, and `psd` are disabled:

- event-response columns are absent
- PAT burden columns are absent
- PSD-window-count columns are absent

## What “Pre-Final Exclusion” Means

Some plots and coverage tables distinguish:

- `pre-final exclusion`
- `final-analysis`

In this repository, the distinction is:

- `pre-final exclusion` = after sleep-stage selection, before the full event/quality/desaturation exclusion
- `final-analysis` = after the full selected-policy `combined_keep` mask

This distinction is especially important for:

- RMSSD coverage
- SDNN coverage
- LF/HF coverage traces

## Current Selected-Policy Exclusion Interpretation

Under the current setup, the exclusion breakdown in the report is mask-based only.

That means:

- the table describes sleep-stage, event, quality, and gated-desaturation masking applied on the final time grid
- it does not include upstream RR loss from PAT peak-detection failure or low-level RR cleaning rejection

Therefore, the difference between:

- total selected-policy time
- final clean kept time

reflects the explicit mask logic, not the full end-to-end loss of valid physiology.

## Summary Of The Current Method In One Sequence

For the current configuration, the main selected-policy HRV workflow can be summarized as:

1. Read the PAT waveform from EDF.
2. Band-pass filter the PAT waveform from `0.5 to 8.0 Hz` with order `4`.
3. Detect PAT peaks using a minimum peak distance corresponding to `0.30 s` and a prominence threshold of `0.30 x signal SD`.
4. Convert consecutive peaks into RR intervals and RR midpoint times.
5. Keep only physiologic RR intervals between `0.30 and 2.50 s`.
6. Apply low-level RR cleaning using median-based outlier rejection, gap rejection, jump rejection, alternans rejection, and a minimum good-run length of `3`.
7. Read the auxiliary CSV and convert sleep stages into numeric stage codes.
8. Build the selected sleep-stage mask using `nrem_only = {1, 2}`.
9. Build event and quality exclusion windows from respiratory event flags and `Exclude HR` / `Exclude PAT` flags using `15 s` pre-padding and `30 s` post-padding.
10. Build event-gated desaturation windows using the desaturation flag, `15 s` start padding, `0 s` end padding, and minimum run length `5 s`.
11. Combine sleep, event, quality, and gated-desaturation masks into the final selected-policy keep mask.
12. Derive PAT HR from the cleaned RR stream, interpolate to `1 Hz`, smooth, despike, and apply the final selected-policy mask.
13. Derive RMSSD and SDNN on `5 min` sliding windows evaluated on a `1 Hz` grid.
14. Derive LF, HF, and LF/HF on non-overlapping fixed `2 min` windows using RR-tachogram Welch PSD.
15. Summarize the surviving values over the selected-policy valid windows.
16. Recompute the same family of metrics for fixed subsets such as all sleep, NREM, deep, REM, wake+sleep, and pre-sleep wake.

## Recommended Citation Style For Methods Sections

If you want to convert this into manuscript text, the cleanest wording is usually:

1. Describe PAT preprocessing and RR extraction.
2. Describe low-level RR cleaning.
3. Describe sleep-stage selection and event/desaturation masking.
4. Describe time-domain HRV windows.
5. Describe fixed-window spectral HRV analysis.
6. Describe subset-specific summaries such as NREM halves or sleep-combo analyses.

If you want, this file can be turned into a manuscript-style Methods section next, with full prose instead of the current engineering-style step-by-step format.
