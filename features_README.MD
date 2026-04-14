# Feature And Export Reference

This file explains the main features in the pipeline and the meaning of the most important exported parameters.

It is meant as a practical guide for anyone reading the PDF reports, per-feature CSV exports, or the final grouped summary CSV.

## Feature Toggles

The main feature switches live in `pat_toolbox/config.py` under `FEATURES`.

- `hr`
  - PAT-derived heart-rate feature
  - computes HR from cleaned RR intervals
  - contributes HR plots, HR CSV, and HR summary fields

- `hrv`
  - heart-rate variability feature
  - computes RMSSD, SDNN, LF, HF, LF/HF and time-varying HRV series
  - contributes HRV plots, HRV CSV, and HRV summary fields

- `psd`
  - spectral feature set derived from the RR/tachogram signal
  - contributes PSD plots and PSD summary fields

- `delta_hr`
  - despite the legacy name, this now means event-response HR analysis
  - contributes Event-Response HR plots and event-response summary fields

- `pat_burden`
  - PAT amplitude burden feature
  - contributes PAT burden plots, burden episode CSV, and burden summary fields

- `sleep_combo_summary`
  - enables repeated summaries across predefined sleep subsets
  - such as all sleep, wake+sleep, NREM, deep, and REM

- `report_pdf`
  - enables the main multi-page PDF report

- `peaks_debug_pdf`
  - enables the PAT peak-debug PDF

## Core Signal Flow

The main workflow is:

1. Read `VIEW_PAT`
2. Band-pass filter PAT
3. Detect PAT peaks
4. Convert peaks to RR intervals
5. Clean RR intervals
6. Compute enabled features from the cleaned RR stream and auxiliary masks

Auxiliary CSV information is used to define:

- sleep-stage filtering
- event exclusion windows
- desaturation-related logic

## Sleep And Exclusion Terms

You will see several kinds of masking/exclusion in the report.

- `Stage-policy excluded`
  - data excluded because it is outside the currently selected sleep-stage policy

- `Event-excluded`
  - data additionally excluded because it falls in configured event/desaturation exclusion windows

- `Metric invalid`
  - the region was allowed by stage and event rules, but the metric still could not be computed there
  - examples: too few RR intervals, too large RR gaps, or insufficient valid data for spectral estimation

- `Pre-event-exclusion`
  - a gray line shown in some plots before the final event exclusion is applied
  - this helps visualize what existed before the clean analysis signal was formed

## HR Feature

The HR feature computes PAT-derived heart rate from the cleaned RR intervals.

### Main idea

- RR intervals are converted to HR in beats per minute
- HR is interpolated to a regular grid
- smoothing and plausibility limits may be applied
- the final clean HR signal uses the same shared mask logic style as other features

### Important exported HR fields

- `hr_pat_nan_pct`
  - percentage of missing/invalid values in the final HR signal

### Sleep-combo HR fields

For each subset you may see:

- `combo_*_hr_mean`
  - mean HR in that subset after the clean mask is applied

- `combo_*_hr_median`
  - median HR in that subset after the clean mask is applied

- `combo_*_hr_std`
  - standard deviation of HR in that subset after the clean mask is applied

- `combo_*_hr_n_used`
  - number of finite HR samples contributing to that subset summary

### HR CSV export

- `__HR_1Hz.csv`
  - `time_sec`
  - `hr_bpm`

## HRV Feature

The HRV feature computes both scalar HRV metrics and sliding-window HRV series.

### Scalar HRV summary metrics

- `rmssd_mean_ms`
  - mean RMSSD over valid clean windows

- `rmssd_median_ms`
  - median RMSSD over valid clean windows

- `sdnn_ms`
  - SDNN from the clean RR series

- `lf`
  - low-frequency HRV power from the clean RR series

- `hf`
  - high-frequency HRV power from the clean RR series

- `lf_hf`
  - LF/HF ratio from the clean RR series

- `lf_n_segments_used`
  - number of contiguous clean RR segments contributing to the LF/HF summary

### Fixed-window LF/HF summary fields

- `lf_hf_fixed_median`
  - median LF/HF across valid fixed windows

- `lf_hf_fixed_mean`
  - mean LF/HF across valid fixed windows

- `lf_hf_fixed_n_windows_valid`
  - number of valid fixed windows used for LF/HF

- `lf_hf_fixed_n_windows_total`
  - total number of fixed windows considered

- `lf_hf_fixed_window_sec`
  - fixed-window duration in seconds

- `lf_hf_fixed_hop_sec`
  - hop size between fixed windows in seconds

### HRV NaN / coverage fields

- `hrv_rmssd_clean_nan_pct`
  - percentage of invalid points in the clean RMSSD series

- `hrv_rmssd_raw_nan_pct`
  - percentage of invalid points in the pre-event-exclusion RMSSD series

- `hrv_tv_*_nan_pct`
  - missing fraction for each sliding-window HRV trace

### Sliding-window HRV series

These are the time-varying HRV features:

- `rmssd_ms`
- `sdnn_ms`
- `lf`
- `hf`
- `lf_hf`

Some exports also include raw/pre-event-exclusion versions:

- `sdnn_ms_raw`
- `lf_raw`
- `hf_raw`
- `lf_hf_raw`

### HRV CSV export

- `__HRV_Features_1Hz.csv`
  - `time_sec`
  - `rmssd_clean_ms`
  - `rmssd_raw_ms`
  - optional sliding-window HRV columns

### Sleep-combo HRV fields

For each subset you may see:

- `combo_*_rmssd_mean_ms`
- `combo_*_sdnn_ms`
- `combo_*_lf_hf`
- `combo_*_lf_n_segments_used`
- `combo_*_lf_hf_fixed_n_windows_valid`

These are the subset-specific HRV summaries after applying the subset sleep policy and event logic.

## PSD Feature

The PSD feature computes spectral summaries from the RR/tachogram signal.

### Main PSD fields

- `mayer_peak_hz`
  - dominant peak frequency in the Mayer-wave band

- `resp_peak_hz`
  - dominant peak frequency in the respiratory band

- `psd_pow_vlf`
  - very-low-frequency power

- `psd_pow_mayer`
  - power in the Mayer-wave band

- `psd_pow_resp`
  - power in the respiratory band

- `psd_norm_mayer`
  - Mayer-band power normalized to total considered power

- `psd_norm_resp`
  - respiratory-band power normalized to total considered power

- `psd_valid_windows`
  - number of valid windows contributing to the PSD summary

### Sleep-combo PSD fields

- `combo_*_psd_valid_windows`
  - number of valid PSD windows in that sleep subset

## Event-Response HR Feature

This feature is currently controlled by `FEATURES["delta_hr"]`, but its meaning has changed.

It no longer exports a lag-based delta-HR signal as the main analysis output.

Instead, it computes event-centered HR response metrics from a smoothed HR signal.

### Definition

The smoothed HR signal is built first.

For each event start time `ts`:

- event window
  - `[ts, ts + HR_EVENT_WINDOW_SEC]`

- recovery window
  - `[ts + HR_EVENT_WINDOW_SEC, ts + HR_EVENT_RECOVERY_END_SEC]`

From these windows, the code computes:

- event trough
  - minimum HR inside the event window

- event mean
  - mean HR inside the event window

- recovery peak
  - maximum HR inside the recovery window

The event is skipped if:

- there is insufficient valid HR in either window
- another event starts before the recovery window ends

### Main response metrics

- `trough_to_peak_response_mean`
  - mean of:
    - `recovery-window max HR - event-window min HR`

- `mean_to_peak_response_mean`
  - mean of:
    - `recovery-window max HR - event-window mean HR`

- `event_windows_total`
  - total detected event windows considered before the final filtering

- `event_windows_used`
  - event windows that remained valid after the missing-data and overlap rules

### Sleep-combo event-response fields

For each subset you may see:

- `combo_*_trough_to_peak_response_mean`
- `combo_*_mean_to_peak_response_mean`
- `combo_*_event_windows_total`
- `combo_*_event_windows_used`

These are subset-specific event-response HR metrics.

### Event-response plots

The Event-Response HR plots show:

- raw HR trace
- event window shading
- recovery window shading
- dashed event-window mean line
- event minimum marker
- recovery maximum marker

## PAT Burden Feature

The PAT burden feature works on PAT amplitude around events/desaturations.

### Main PAT burden summary fields

- `pat_burden`
  - overall burden summary value

- `pat_burden_sleep_hours`
  - total included sleep time used for burden normalization

- `pat_burden_total_area_min`
  - total integrated burden area

- `pat_burden_n_episodes`
  - number of detected episodes

- `pat_burden_n_episodes_used`
  - number of valid episodes contributing to the final burden

- `pat_burden_relative`
  - indicates whether burden used relative normalization

- `pat_burden_nan_pct`
  - invalid fraction inside the burden-related signal region

### Sleep-combo PAT burden fields

- `combo_*_pat_burden`
  - PAT burden value in that subset

### PAT burden episode CSV

- `__PAT_Burden_Episodes.csv`
  - one row per episode
  - contains timing, area, baseline, and validity/skip information

## Sleep-Combo Summaries

When `sleep_combo_summary` is enabled, the pipeline repeats summaries across predefined subsets.

Common subsets are:

- `all_sleep`
- `wake_sleep`
- `nrem`
- `deep`
- `rem`

The prefix used in the grouped summary CSV is:

- `combo_all_sleep_*`
- `combo_wake_sleep_*`
- `combo_nrem_*`
- `combo_deep_*`
- `combo_rem_*`

### Sleep-combo shared fields

- `combo_*_sleep_hours`
  - included duration of that subset in hours

Depending on enabled features, the subset can also export:

- HR summaries
- HRV summaries
- PSD valid window counts
- event-response HR summaries
- PAT burden summaries

## Auxiliary Event And Sleep Fields

The final grouped summary CSV also contains several descriptive fields from the auxiliary CSV.

### Event count fields

- `desat_n`, `desat_pct`
- `exclude_hr_n`, `exclude_hr_pct`
- `exclude_pat_n`, `exclude_pat_pct`
- `evt_central_3_n`, `evt_central_3_pct`
- `evt_obstructive_3_n`, `evt_obstructive_3_pct`
- `evt_unclassified_3_n`, `evt_unclassified_3_pct`
- optional `*_4_*` fields if those columns are present

### Sleep-stage mask fields

- `sleep_mask_enabled`
- `sleep_policy`
- `sleep_included_n`, `sleep_included_pct`
- `sleep_excluded_n`, `sleep_excluded_pct`
- `sleep_wake_n`, `sleep_wake_pct`
- `sleep_light_n`, `sleep_light_pct`
- `sleep_deep_n`, `sleep_deep_pct`
- `sleep_rem_n`, `sleep_rem_pct`

## Common Output Files

Typical outputs for one recording are:

- HR CSV
  - `__HR_1Hz.csv`

- HRV CSV
  - `__HRV_Features_1Hz.csv`

- PAT burden episode CSV
  - `__PAT_Burden_Episodes.csv`

- multi-page PDF report

- grouped final summary CSV
  - one row per recording

## Practical Reading Guide

If you only want the core summary meaning:

- `HR` fields
  - average and variability of the accepted heart-rate signal

- `HRV` fields
  - variability/autonomic-style summaries derived from clean RR intervals

- `PSD` fields
  - spectral organization of the RR/tachogram signal

- `Event-response HR` fields
  - how much HR rises from the event period into the recovery period

- `PAT burden` fields
  - cumulative PAT-amplitude burden around relevant event/desaturation episodes

If you only want the per-subset interpretation:

- `combo_*_...`
  - the same type of summary, but restricted to that sleep subset

## Important Note About Legacy Naming

The config key `delta_hr` is still present for backward compatibility.

In the current codebase it refers to:

- Event-Response HR metrics and plots

not to the older lag-based delta-HR signal.
