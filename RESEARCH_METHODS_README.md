# PAT Toolbox Research Methods README

This document describes the current signal-processing workflow used in this repository, using the parameters from the present configuration in `pat_toolbox/config.py`.

It is written as a methods-style reference for research reporting. It explains, step by step, how features are derived from the EDF and auxiliary CSV inputs, how exclusion is applied, how sleep-stage policies are handled, and which calculations are active in the current setup.

Terminology used below:

- `PAT` = Peripheral Arterial Tone.
- `PR` = pulse-to-pulse interval (PRI) between adjacent detected PAT peaks.
- `PRV` = pulse rate variability derived from the PAT pulse-interval stream.

These `PR` intervals are PAT-derived pulse intervals (PRI values). They are not ECG `R-R` intervals, and the repository uses `PR` terminology as a short repository label for that pulse-interval stream.

Conceptually, the method can be read in three physiological layers:

1. the recorded PAT waveform, which reflects pulsatile peripheral vascular tone
2. the derived PR interval stream, which captures pulse-to-pulse timing
3. the downstream summaries, which describe pulse timing variability, spectral organization, and optional amplitude-related burden

Accordingly, the reported HR and PRV measures in this repository should be interpreted as PAT-derived vascular pulse metrics rather than ECG-derived cardiac interval metrics.

## Current Active Setup

The current `FEATURES` configuration is:

- `hr = True`
- `prv = True`
- `psd = False`
- `delta_hr = True`
- `pat_burden = True`
- `pwa_drop = True`
- `pat_harmonics = True`
- `pat_paper_harmonics = True`
- `sleep_combo_summary = True`
- `report_pdf = True`
- `peaks_debug_pdf = False`

This means the current run produces:

- PAT-derived HR
- PRV time-domain features
- PRV frequency-domain features based on PR intervals
- event-response HR summaries and plots
- PAT burden summaries and plots
- PWA-drop summaries and plots for the 30% and 50% threshold variants
- raw Welch PAT harmonic summaries and plots
- paper-style beat-synchronous PAT harmonic summaries and plots
- sleep-subset comparison summaries
- the main PDF report
- a publication-style PRV PNG export

The current sleep-stage policy is:

- `SLEEP_STAGE_POLICY = "all_sleep_incluidng_wake"`
- included stages = `{0, 1, 2, 3}`

Under the current stage mapping:

- `0 = Wake`
- `1 = Light sleep`
- `2 = Deep sleep`
- `3 = REM`

Therefore, in the main selected-policy analysis, wake, light sleep, deep sleep, and REM are all included. The misspelling in `all_sleep_incluidng_wake` is the current config token used by the code.

## Input Data

### EDF input

For each recording, the workflow reads the EDF channel:

- `VIEW_PAT`

This is the PAT waveform used for peak detection, PR extraction, HR, and PRV.

The code uses the sampling frequency stored in the EDF for that channel. In practice, your recordings may use a 40 Hz PAT signal, but the algorithm itself uses the EDF-native sampling frequency rather than a hard-coded value.

From a physiological perspective, this PAT channel is the primary source signal for the analysis. All timing-derived outputs in the repository begin with this vascular pulse waveform.

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
4. Optionally load SpO2 for validation plots.
5. Load and normalize the auxiliary CSV.
6. Compute sleep-subset summaries.
7. Compute PAT burden if enabled.
8. Compute PWA-drop threshold variants if enabled.
9. Compute raw Welch PAT harmonics if enabled.
10. Compute paper-style beat-synchronous PAT harmonics if enabled.
11. Compute PAT-derived HR.
12. Compute event-response DHR if enabled.
13. Compute PRV and PR-based summaries.
14. Compute separate PSD features if enabled.
15. Export per-feature CSVs.
16. Build the PDF report.
17. Optionally export a publication-style PRV PNG for an automatically selected NREM segment.
18. Append one row to the grouped summary CSV.

In compact form, the physiological flow is:

- PAT waveform -> PAT peaks -> PR intervals -> HR / PRV / PR-tachogram summaries

The auxiliary CSV is then used to determine which parts of the night should contribute to the final reported values.

## PAT Preprocessing

Before peak detection, the PAT waveform is band-pass filtered with a Butterworth band-pass filter using:

- low cut: `0.5 Hz`
- high cut: `8.0 Hz`
- filter order: `4`

Filtering is done with zero-phase forward-backward filtering (`filtfilt`), so no intentional phase shift is introduced.

## Peak Detection And PR Extraction

PAT peak detection is performed on the filtered PAT waveform.

### Peak detection settings

- minimum PR interval: `0.30 s`
- maximum PR interval: `2.50 s`
- peak prominence factor: `0.30 x SD(filtered PAT)`

Operationally, the detector:

1. Finds local PAT peaks with a minimum peak-to-peak distance corresponding to `0.30 s`.
2. Uses a prominence threshold equal to `0.30 x` the standard deviation of the filtered PAT waveform when the filtered signal has nonzero variance.
3. Converts detected peak times into beat-to-beat PR intervals.
4. Keeps only PR intervals within the physiologic range `0.30 to 2.50 s`.

PR mid-times are defined as the midpoint between consecutive peaks.

This conversion step is the key bridge between the vascular waveform and all later timing-derived outputs. Once the PAT signal has been reduced to a cleaned PR interval stream, the downstream HR, PRV, and spectral analyses all use that same shared interval representation.

## Low-Level PR Cleaning

After initial PR extraction, the repository applies an additional low-level PR cleaning pass before HR and PRV are derived.

### PR cleaning parameters

- median filter kernel: `5`
- PR relative outlier threshold: `0.30`
- PR gap factor: `2.4`
- PR jump relative threshold: `0.6`
- alternans short relative threshold: `0.25`
- alternans long relative threshold: `0.35`
- minimum retained contiguous artifact-free run length: `3 PR intervals`

### PR cleaning procedure

The cleaned PR stream is produced as follows:

1. Compute the raw PR sequence from consecutive PAT peaks.
2. Remove PR intervals outside the physiologic range `0.30 to 2.50 s`.
3. Compute a local median PR using a median filter of length `5`.
4. Mark a PR interval as artifactual if its relative deviation from the local median exceeds `30%`.
5. Mark a PR interval as artifactual if it exceeds `2.4 x` the local median PR.
6. Mark abrupt jumps as artifactual when the relative jump between consecutive PR intervals exceeds `0.6`.
7. Reject short-long or long-short alternans-like pairs using:
   - short threshold: `< (1 - 0.25) x local median`
   - long threshold: `> (1 + 0.35) x local median`
8. After all point-wise rejection, keep only contiguous runs of artifact-free PR intervals of length at least `3`.

This produces the shared physiologically cleaned PR series used by HR and PRV.

## Sleep-Stage Policy

The main selected-policy analysis currently uses:

- `all_sleep_incluidng_wake = {0, 1, 2, 3}`

This means:

- included: wake, light sleep, deep sleep, and REM
- excluded by sleep-stage policy: no mapped stage among the canonical four stages

### How sleep-stage masking is applied

The code maps each analysis time point or PR midpoint to the nearest auxiliary stage sample in time.

For the selected-policy analysis:

- if the mapped stage is wake, light sleep, deep sleep, or REM, the sample is kept by the sleep mask
- if the mapped stage is missing or outside the include set, the sample is rejected by the sleep mask

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

The signal-quality component of this exclusion logic is important because the downstream timing and spectral metrics are only meaningful when the underlying PAT pulse train is interpretable. When quality-style auxiliary flags are included in the active exclusion list, the surrounding interval is removed with the same padded window logic used for respiratory event exclusions. This is done to reduce contamination from motion, signal dropout, poor pulse definition, or other waveform conditions that can distort peak detection, corrupt PR intervals, and propagate implausible values into HR, PRV, or spectral estimates.

### Active exclusion columns in the current configuration

- `evt_central_3`
- `evt_obstructive_3`
- `evt_unclassified_3`

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

- `PRV_EXCLUSION_USE_DESAT_WINDOWS = True`
- desaturation column: `desat_flag`
- desaturation start padding: `15 s`
- desaturation end padding: `30 s`
- minimum desaturation run: `5.0 s`
- event lookback for gating: `0.0 s`
- event lookahead for gating: `0.0 s`

Operationally:

1. Consecutive desaturation samples are grouped into runs.
2. If a desaturation run lasts at least `5 s`, it becomes one exclusion window.
3. Shorter runs are converted into per-sample exclusion windows.
4. Each desaturation window is padded by `15 s` at the start and `30 s` at the end.
5. A desaturation window is only activated if at least one active exclusion event falls inside the exact desaturation window because both lookback and lookahead are set to `0 s`.

This is a conservative event-gated desaturation logic. Desaturation alone does not exclude data unless it coincides with an active event window under this gating rule.

### Final combined mask

For time-grid or PR-midpoint analyses, the final selected-policy keep mask is:

- `combined_keep = sleep_keep AND event_keep AND desat_keep`

Therefore a sample is kept only if:

1. it belongs to the selected sleep stages
2. it is outside the padded event windows from respiratory events and quality flags
3. it is outside any active gated desaturation windows

## PAT-Derived HR Calculation

Heart rate is derived from the cleaned PR intervals.

### HR parameters

- output sampling rate: `1.0 Hz`
- HR clamp: `30 to 220 bpm`
- HR interpolation maximum gap: `2.5 s`
- HR smoothing window: `4.0 s`
- HR Hampel window: `8.0 s`
- HR Hampel sigma: `2.0`
- maximum HR slope limit: `10 bpm/s`

### HR derivation procedure

1. Convert the cleaned PR intervals to instantaneous HR using `HR = 60 / PR`.
2. Interpolate instantaneous HR to a regular `1 Hz` grid.
3. Do not interpolate across PR gaps larger than `2.5 s`.
4. Smooth the interpolated HR with a moving average over `4 s`.
5. Clamp the smoothed HR to `30 to 220 bpm`.
6. Apply a Hampel despiking filter with an `8 s` window and `2 sigma`, but only when the NaN fraction is below `5%`.
7. Clamp the despiked HR again to `30 to 220 bpm`.
8. Apply a slope limiter so that adjacent 1 Hz samples do not change by more than `10 bpm/s`.
9. Apply the selected-policy `combined_keep` mask, setting excluded samples to `NaN`.

The final selected-policy HR summary statistics are computed from the finite HR samples remaining after this masking.

## Event-Response HR (DHR)

The `delta_hr` feature is an event-centered DHR analysis derived from the PAT-based HR signal. It is intended to quantify how strongly HR rises from a respiratory-event window minimum to a post-event maximum.

### Event-response HR settings

- HR smoothing before event analysis: `5.0 s`
- nominal event window length: `15.0 s`
- fallback post-event search window end: `45.0 s` from event onset
- ensemble pre-event support: `20.0 s`
- ensemble grid: `1.0 s`
- minimum events for ensemble-derived search window: `5`
- ensemble peak margin: `10.0 s`
- minimum valid samples in both event and post-event search windows: `3`
- desaturation-aware extension: `True`

### Event-response HR procedure

For each excluded event run in the selected sleep policy:

1. Define the event window starting at the beginning of the run.
2. Set the nominal event end to `start + 15 s`.
3. If desaturation-aware extension is enabled, extend the event window to the end of any overlapping gated desaturation window that begins before the nominal recovery endpoint.
4. Build an ensemble-average event response for the recording when at least `5` valid events are available.
5. Define a post-event DHR search window around the ensemble post-event peak, with the configured margin.
6. If the ensemble search window is unavailable, use the fixed fallback interval from `15 s` to `45 s` after event onset.
7. Skip the event if the next event begins before the DHR search window ends.
8. Smooth HR first using a `5 s` moving average.
9. Require at least `3` valid smoothed HR samples in both the event and post-event search windows.
10. Compute:
   - event-window minimum HR
   - post-event search-window maximum HR
   - `DHR = post-event maximum HR - event-window minimum HR`

### Delta-HR outputs and interpretation

The selected-policy summary reports:

- number of event windows detected
- number of event windows used
- mean, median, p25, and p75 DHR
- DHR search-window source and offsets
- ensemble events used and ensemble peak offset when available

Physiologically, larger values indicate a stronger HR rebound around event-linked disturbed intervals. This is an event-response metric rather than a whole-night HR variability metric.

## PRV Calculation Overview

The PRV feature uses the shared cleaned PR intervals and then creates two distinct families of outputs:

1. time-domain PRV outputs
2. frequency-domain PRV outputs

These two families do not use exactly the same window definition in the current setup. This is intentional: time-domain metrics use overlapping windows to track gradual temporal variation in pulse timing, whereas the main reported frequency-domain metrics use non-overlapping fixed windows so that spectral values are not interpreted as having the same effective temporal resolution.

The exclusion logic also interacts differently with these two domains. Time-domain RMSSD and SDNN are now evaluated on the same accepted-window set: a window must pass the shared gap/coverage rules and also survive the RMSSD-oriented robustness checks before either metric is retained. Frequency-domain metrics are more restrictive in a different way because spectral estimation requires a more continuous interval sequence over a longer effective support window. Consequently, a short quality-related interruption may still invalidate a spectral window even when enough valid PR intervals remain for time-domain estimation. This difference is expected and reflects the different physiological and statistical requirements of beat-to-beat variability summaries versus windowed spectral decomposition.

## Time-Domain PRV: RMSSD And SDNN

### Core time-domain settings

- PRV output grid: `1.0 Hz`
- main PRV window: `300 s` (`5.0 min`)
- minimum PR intervals per window: `6`
- maximum PR gap inside PRV windows: `8.0 s`
- minimum time span inside a PRV window: `5.0 s`
- minimum window coverage fraction: `0.2`
- RMSSD smoothing window: `5.0 s`

### RMSSD robustness settings

- hard cap on successive PR differences: `400 ms`
- MAD cutoff: `4.0 sigma`
- minimum surviving PR differences: `3`
- RMSSD floor: `2.0 ms`
- large-difference veto: `False`
- if enabled, the unused veto parameters would be:
  - large-difference threshold: `300 ms`
  - max fraction of large differences: `0.35`

### RMSSD derivation procedure

For each `1 Hz` analysis time point:

1. Center a `300 s` window on that time.
2. Gather PR intervals whose midpoints fall inside the window.
3. Require the shared time-domain window gate to pass:
   - at least `6` PR intervals
   - no PR gap larger than `8.0 s`
   - PR midpoint span at least `5.0 s`
   - span at least `20%` of the `300 s` window because `PRV_MIN_WINDOW_COVERAGE = 0.2`
4. Compute successive PR differences.
5. Remove PR differences whose absolute value exceeds `400 ms`.
6. Compute the median and MAD of the remaining differences.
7. Remove differences farther than `4.0 x robust sigma` from the median.
8. Require at least `3` remaining PR differences.
9. Compute RMSSD as the square root of the mean squared successive difference.
10. Reject the value if RMSSD is below `2.0 ms`.

This is done twice:

- `rmssd_raw`: after sleep-stage masking only
- `rmssd_clean`: after the full selected-policy combined mask

The main report and summary tables use the final analysis signal.

### SDNN derivation procedure

The time-varying SDNN series is computed on the same `300 s` sliding windows used for RMSSD.

For each valid window:

- `SDNN = standard deviation of PR intervals in ms`

Importantly, the accepted-window logic is shared with RMSSD. In other words, SDNN is reported only for windows that satisfy the common time-domain gate and also survive the RMSSD-oriented robustness rejection steps. This keeps the displayed and summarized RMSSD and SDNN series aligned to the same retained physiological support.

In addition to the time-varying series, the summary tables report:

- `SDNN mean`: mean of valid sliding-window SDNN values
- `SDNN median`: median of valid sliding-window SDNN values

## Frequency-Domain PRV: LF, HF, LF/HF

### Fundamental spectral settings

- tachogram resampling frequency: `4.0 Hz`
- LF band: `0.04 to 0.15 Hz`
- HF band: `0.15 to 0.40 Hz` for LF/HF computation

### Main reported spectral summary settings

The current selected-policy spectral summary uses fixed windows, not the same `5 min` time-domain windows.

- fixed window length: `120 s` (`2.0 min`)
- fixed hop: `120 s`
- minimum PR intervals per fixed spectral window: `0` additional requirement beyond the built-in minimum of `4`
- maximum PR gap in fixed spectral windows: `3.0 s`
- minimum retained span in a fixed spectral window: at least `80%` of the window, therefore at least `96 s`

### Spectral computation within one valid fixed window

For each valid fixed window:

1. Extract PR intervals whose midpoints fall inside the fixed window.
2. Require at least `4` PR intervals.
3. Reject the window if any PR gap exceeds `3.0 s`.
4. Reject the window if the PR midpoint span is below `96 s`.
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

The code still contains a segmented spectral helper that pools contiguous clean PR runs of at least `120 s` and weights them by duration.

However, the main exported `lf`, `hf`, and `lf_hf` summary values are currently overwritten to follow the fixed-window analysis, not the legacy segmented summary.

## Time-Varying Spectral Plots In The Current Setup

The current report plotting is aligned with the actual reported spectral summary.

Therefore:

- RMSSD and SDNN plots use the `5 min` sliding-window time-domain analysis
- plotted `LF`, `HF`, and `LF/HF` traces use the same `2 min` fixed-window spectral analysis used by the selected-policy summary tables

For whole-night summary-style plots, these traces are then displayed as binned summaries. The current default bin size is `10 min`, with a lower valid-count threshold for spectral bins than for dense `1 Hz` RMSSD/SDNN traces because the spectral estimates are much sparser in time.

This plotting choice is meant to preserve the physiological meaning of the estimates: RMSSD and SDNN behave like densely re-estimated windowed trends, whereas LF, HF, and LF/HF remain lower-resolution summaries of longer spectral support windows.

This means the spectral plot pages and the spectral summary table now correspond to the same window definition.

## Separate PSD Feature Block

The dedicated `psd` feature is currently disabled:

- `psd = False`

So in the present setup:

- no dedicated PAT PSD pages are generated
- no separate PSD summary fields are expected in the main output as a run feature

If enabled, that feature would compute averaged PR-based PSDs over the same fixed `120 s` windows and report Mayer-band and respiratory-band power summaries.

## PAT Burden

The `pat_burden` feature quantifies cumulative PAT amplitude suppression during excluded event/desaturation periods within the selected sleep policy.

### PAT burden settings

- baseline lookback: `30 s`
- baseline minimum samples: `5`
- baseline percentile: `95th percentile`
- minimum episode length: `5 s`
- relative burden mode: `False`

### PAT burden procedure

1. Load the PAT amplitude channel `DERIVED_PAT_AMP` when available.
2. Build the selected-policy sleep mask and the event/desaturation exclusion mask.
3. Define burden episodes as contiguous time spent inside excluded event/desaturation regions while still inside the selected sleep policy.
4. Skip episodes shorter than `5 s`.
5. For each episode, gather valid baseline PAT amplitude samples from the `30 s` interval immediately before the episode, restricted to selected sleep and outside the excluded event/desaturation region.
6. Require at least `5` valid baseline samples.
7. Define the episode baseline as the `95th percentile` of those baseline samples.
8. Within the episode, compute the non-negative drop below baseline.
9. Integrate the drop over time to produce episode area.
10. Sum episode areas across the night and divide by sleep hours.

In the current setup, burden is reported in `amp·min/h` because relative normalization is disabled. If relative mode is enabled, the drop is normalized by the local baseline first and the unit becomes `rel·min/h`.

### PAT burden diagnostics and interpretation

The burden summary now reports:

- burden per sleep hour
- sleep hours
- total burden area
- episodes total, used, and skipped
- finite PAT AMP coverage
- inside-event/desaturation minutes
- invalid PAT AMP minutes inside burden regions
- skipped episode reasons

Physiologically, higher PAT burden means deeper and/or longer PAT amplitude attenuation during respiratory-event-related disturbed periods. It is therefore closer to a cumulative vascular/autonomic burden measure than to a simple event count.

## PWA-Drop

The `pwa_drop` feature is a waveform-derived pulse-wave-amplitude drop detector modeled after the external MATLAB `PWA_drop` project. It is designed to identify discrete transient decreases in pulse-wave amplitude rather than to compute cumulative burden inside pre-defined exclusion intervals.

### PWA-drop settings

- 30% variant primary amplitude threshold: `30%` decrease from baseline
- 30% variant secondary amplitude threshold: `20%` decrease from baseline
- 50% variant primary amplitude threshold: `50%` decrease from baseline
- 50% variant secondary amplitude threshold: `40%` decrease from baseline
- minimum primary-threshold points: `2`
- minimum secondary-threshold points: `4`
- baseline length: `5` cardiac cycles
- sensor-loss threshold: `5`
- maximum accepted heart rate for the beat-to-beat PWA stream: `250 bpm`

### PWA-drop procedure

1. Start from the filtered PAT waveform `VIEW_PAT` used by the workflow.
2. Detrend and smooth the waveform.
3. Identify local maxima and minima and derive a beat-to-beat `PWA` series from their amplitude difference.
4. Reject implausible beat timing using the maximum accepted heart-rate constraint.
5. Mark sensor-loss and abrupt PWA discontinuity artefacts and remove those PWA samples.
6. Detrend and smooth the PWA time series itself.
7. Estimate local variance and derivative on the PWA series.
8. Build a baseline mask from low-variance stable tracts.
9. Select candidate drops where local variance peaks coincide with negative local derivative.
10. For each candidate, define a local observation interval using neighboring local maxima and minima.
11. Estimate baseline from preceding stable PWA beats.
12. Express the candidate interval as percentage decrease from baseline.
13. Keep the event only if the decrease satisfies both the primary and secondary threshold-duration criteria for the tested variant.
14. Extract per-drop parameters such as start, end, and center time, duration, amplitude percentage, AUC, and descending / ascending slopes.
15. Repeat the detection for both configured variants using the same beat-to-beat PWA series.
16. Restrict the final reported events to the selected sleep policy and summarize them per sleep hour.

### PWA-drop outputs and interpretation

The selected-policy summary reports:

- number of detected drops for each variant
- drop rate per sleep hour for each variant
- mean amplitude percentage for each variant
- mean duration for each variant
- mean AUC for each variant
- number and percentage of drops overlapping excluded event/desaturation regions for each variant

Physiologically, this feature is intended to capture discrete transient PWA suppression events, which may reflect peripheral vasoconstrictive or autonomic responses. It is complementary to PAT burden:

- `PAT burden` quantifies cumulative suppression within event-linked regions
- `PWA-drop` counts and characterizes individual waveform-derived suppression events

## PAT Harmonics

The current setup enables two separate PAT harmonic feature families. These are computed from the full raw `VIEW_PAT` signal by default. Respiratory-event, desaturation, and sleep-stage masks are not applied to the harmonic calculation unless `PAT_HARMONICS_USE_MASK` or `PAT_PAPER_HARMONICS_USE_MASK` is explicitly set to `True`. Sleep stages are still assigned afterward for reporting and grouping.

### Raw Welch PAT harmonics

Settings:

- window length: `120 s`
- hop: `60 s`
- fundamental search band: `0.5 to 2.5 Hz`
- maximum harmonic: `H5`
- Welch segment length: `16 s`
- FFT length: `4096`

Procedure:

1. Divide the full PAT signal into overlapping `120 s` windows with `60 s` hop.
2. Fill non-finite samples inside an accepted window using the window median.
3. Estimate a Welch PSD.
4. Identify the fundamental frequency in the pulse band.
5. Integrate power around harmonic bands `H1` through `H5`.
6. Report harmonic powers, harmonic ratios, total harmonic power, and a harmonic distortion index.

### Paper-style beat-synchronous PAT harmonics

Settings:

- window length: `120 s`
- hop: `60 s`
- maximum coefficient: `C10`
- pulse resampling length: `256` samples per pulse cycle
- minimum beats per window: `20`
- minimum pulse valid fraction: `0.90`

Procedure:

1. Detect PAT peaks in the full PAT waveform.
2. Use adjacent peaks to define individual pulse cycles.
3. Reject implausible pulse durations using the same physiologic PR bounds used elsewhere.
4. Resample each accepted pulse to a common phase grid.
5. Compute FFT amplitude coefficients `C0` through `C10` for each pulse.
6. Normalize harmonic amplitudes as `C1/C0` through `C10/C0`.
7. Average pulse-level coefficients within each `120 s` window.
8. Compute subharmonic VLF/LF/HF powers from the raw windowed PAT signal.
9. Assign a sleep-stage code and label to each harmonic window from the sleep stage nearest the window center.
10. Summarize harmonic metrics overall and separately for wake, light sleep, deep sleep, and REM.

The reported `hf_ratio` is not HRV HF power. It is a pulse-shape index defined as:

```text
hf_ratio = sum(C6/C0 ... C10/C0) / sum(C1/C0 ... C10/C0)
```

It represents the fraction of normalized pulse-shape harmonic amplitude contained in higher-order harmonics.

### Harmonics outputs and visualization

The paper-style harmonics window CSV contains per-window timestamps, center-stage labels, `C0`, `C1/C0` through `C10/C0`, `hf_ratio`, subharmonic powers, and beat counts. The summary CSV includes overall and per-stage means/medians. The grouped summary CSV includes a compact per-stage subset for C0, selected normalized coefficients, high-order ratio, and subharmonic powers.

The PDF report places overnight overview pages before detailed segment pages. The harmonics overview pages are placed early after the hypnogram/stage page. Event and exclusion overlays are intentionally not shown on harmonics overview panels because the harmonics are computed from the full signal by default. The paper-style C/C0 heatmap uses a horizontal colorbar above the heatmap to preserve subplot alignment, and normalized coefficient plots use robust y-limits with clipped outlier markers for readability.

## Actigraphy Visualization

When the EDF contains the configured `ACTIGRAPH` channel, the report can include an actigraphy segment subplot for visual motion sanity checks. This subplot is visual-only: actigraphy is not currently used to reject samples or change any metric calculation. The plotted actigraphy trace uses absolute amplitude, starts at zero, and uses a consistent scale across the recording.

## Sleep Timing And Sleep-Half Analysis

Sleep timing is computed from the auxiliary sleep-stage timeline.

Definitions:

- sleep onset: first time with stage in `{1, 2, 3}`
- sleep end: last time with stage in `{1, 2, 3}`, plus one auxiliary sample interval
- sleep midpoint: midpoint between sleep onset and sleep end

In the current midpoint-half PRV comparison:

- the half analysis is restricted to NREM only, using stages `{1, 2}`
- the NREM PR stream is cut into first and second halves relative to the sleep midpoint
- each half is then summarized with the same selected-policy PRV summary engine

This NREM-restricted midpoint analysis is a secondary comparison only. It does not mean that the main selected-policy analysis is currently NREM-only.

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

In the current setup, because `delta_hr`, `pat_burden`, and `pwa_drop` are enabled while `psd` is disabled:

- event-response columns are present
- PAT burden columns are present
- PWA-drop columns are present separately for 30% and 50% variants
- raw and paper-style PAT harmonic selected-policy columns are present
- paper-style PAT harmonic per-stage columns are present in dedicated harmonics outputs and compact grouped summary fields
- PSD-window-count columns are absent

For `delta_hr`, `pat_burden`, and `pwa_drop`, the fixed sleep-subset summaries are currently available in summary/comparison outputs. Detailed event-level or episode-level CSV exports are generated for the main selected-policy analysis only and are not written as separate files for each sleep subset.

The PDF sleep-subset comparison tables are split across multiple pages when needed so wide metric groups remain readable. The explanatory notes are shortened and placed above the relevant full-page table.

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
- it does not include upstream PR loss from PAT peak-detection failure or low-level PR cleaning rejection

Therefore, the difference between:

- total selected-policy time
- final clean kept time

reflects the explicit mask logic, not the full end-to-end loss of valid physiology.

## Summary Of The Current Method In One Sequence

For the current configuration, the main selected-policy PRV workflow can be summarized as:

1. Read the PAT waveform from EDF.
2. Band-pass filter the PAT waveform from `0.5 to 8.0 Hz` with order `4`.
3. Detect PAT peaks using a minimum peak distance corresponding to `0.30 s` and a prominence threshold of `0.30 x signal SD`.
4. Convert consecutive peaks into PR intervals and PR midpoint times.
5. Keep only physiologic PR intervals between `0.30 and 2.50 s`.
6. Apply low-level PR cleaning using median-based outlier rejection, gap rejection, jump rejection, alternans rejection, and a minimum artifact-free run length of `3`.
7. Read the auxiliary CSV and convert sleep stages into numeric stage codes.
8. Build the selected sleep-stage mask using `all_sleep_incluidng_wake = {0, 1, 2, 3}`.
9. Build event and quality exclusion windows from the currently active exclusion columns using `15 s` pre-padding and `30 s` post-padding.
10. Build event-gated desaturation windows using the desaturation flag, `15 s` start padding, `30 s` end padding, and minimum run length `5 s`.
11. Combine sleep, event, quality, and gated-desaturation masks into the final selected-policy keep mask.
12. Derive PAT HR from the cleaned PR stream, interpolate to `1 Hz`, smooth, despike, and apply the final selected-policy mask.
13. Derive event-response DHR windows and summarize post-event maximum HR minus event-window minimum HR across valid events.
14. Derive RMSSD and SDNN on `5 min` sliding windows evaluated on a `1 Hz` grid.
15. Derive LF, HF, and LF/HF on non-overlapping fixed `2 min` windows using PR-tachogram Welch PSD.
16. Derive PAT burden from PAT amplitude inside excluded event/desaturation regions relative to a local pre-episode baseline.
17. Derive discrete 30% and 50% PWA-drop events from the waveform-derived PWA series and summarize their count, rate, and morphology.
18. Derive full-signal raw Welch PAT harmonics and full-signal paper-style beat-synchronous PAT harmonics.
19. Assign paper-style harmonic windows to sleep stages by window-center stage for per-stage summaries.
20. Summarize the surviving values over the selected-policy valid windows.
21. Recompute the same family of selected subset metrics for fixed subsets such as all sleep, NREM, deep, REM, wake+sleep, and pre-sleep wake.
