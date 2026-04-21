# Feature And Export Reference

This file explains the main features in the pipeline and the meaning of the most important exported parameters.

It is meant as a practical guide for anyone reading the PDF reports, per-feature CSV exports, or the final grouped summary CSV.

## Selected-Policy Naming

In the final grouped summary CSV, global policy-dependent fields now use the prefix `selected_`.

This means:

- `selected_*`
  - summary computed under the currently active sleep-stage policy from `pat_toolbox/config.py`
  - for example the current `SLEEP_STAGE_POLICY` and shared event exclusion settings

- `combo_*`
  - fixed sleep-subset summaries computed separately for predefined subsets such as all sleep, NREM, deep, or REM

- raw aux/event count fields without `selected_`
  - descriptive whole-recording counts from the auxiliary CSV
  - not filtered by the currently selected sleep policy

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

## How To Read Provenance

For most exported values, the most important question is not only "what does this metric mean?" but also:

- which signal was used
- which masks were active
- whether the value is point-wise, windowed, or whole-interval pooled

In this repository, the main provenance terms are:

- `physio-clean RR`
  - RR intervals obtained from PAT after beat detection and low-level RR cleaning
  - this happens before sleep/event/desaturation masking

- `sleep_keep`
  - time or RR samples retained by the selected sleep-stage policy

- `combined_keep`
  - final clean analysis mask after sleep policy plus event/quality/desaturation exclusion

- `pre-exclusion`
  - after sleep-stage selection, before the final event/quality/desaturation exclusion

- `clean`
  - after the final selected-policy combined mask

- `fixed-window`
  - value computed in repeated fixed-duration windows, then summarized across valid windows

- `selected-policy`
  - value computed under the currently active sleep policy from `config.py`

- `combo_*`
  - same kind of value, recomputed separately for a predefined subset such as all sleep, NREM, deep, REM, or wake+sleep

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

### What signal is summarized

The main HR summaries in the grouped CSV and PDF are based on:

- PAT-derived HR
- interpolated to a regular grid
- then masked by the selected-policy `combined_keep`

So `HR mean`, `HR median`, and `HR std` are not based on a residual signal. They are summaries of the final PAT-derived HR signal that remains after the configured sleep/event/quality/desaturation mask.

### Important exported HR fields

- `selected_hr_valid_pct`
  - percentage of valid PAT-derived HR samples after the selected-policy combined mask

- `selected_hr_valid_min`
  - valid PAT-derived HR coverage in minutes after the selected-policy combined mask

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

Important change:

- the main exported `LF`, `HF`, and `LF/HF` values now follow the fixed-window analysis
- they are no longer the legacy global segmented spectral values
- in other words:
  - `selected_lf`, `selected_hf`, `selected_lf_hf`
  - `combo_*_lf`, `combo_*_hf`, `combo_*_lf_hf`
  now represent fixed-window means across valid windows

- `rmssd_mean_ms`
  - mean RMSSD over valid clean windows

- `rmssd_median_ms`
  - median RMSSD over valid clean windows

- `sdnn_ms`
  - SDNN from the clean RR series
  - this is still an overall clean-RR time-domain summary, not a fixed-window mean

- `lf`
  - low-frequency HRV power represented by the mean across valid fixed windows
  - units are `ms^2`

- `hf`
  - high-frequency HRV power represented by the mean across valid fixed windows
  - units are `ms^2`

- `lf_hf`
  - LF/HF ratio represented by the mean across valid fixed windows
  - unitless

- `lf_n_segments_used`
  - legacy count from the older segmented frequency-domain calculation
  - kept as reference only; it is no longer the primary source of the reported LF/HF values

### Time-domain versus frequency-domain summary style

The main selected-policy HRV summary now mixes two styles by design:

- time-domain
  - `RMSSD mean`, `RMSSD median`
    - derived from valid sliding RMSSD windows on the clean RR signal
  - `SDNN`
    - derived from the clean RR series as a whole

- frequency-domain
  - `LF`, `HF`, `LF/HF`
    - derived from repeated fixed windows on the clean RR signal
    - then summarized across valid windows

So if you compare `SDNN` and `LF`, they are not coming from exactly the same style of aggregation.

### Fixed-window LF/HF summary fields

These are computed on the selected-policy clean RR signal using:

- `window_sec = HRV_LFHF_FIXED_WINDOW_SEC`
- `hop_sec = HRV_LFHF_FIXED_HOP_SEC`

If `hop_sec == window_sec`, windows do not overlap.

The current main exported frequency-domain values use this same fixed-window engine.

- `lf_hf_fixed_median`
  - median LF/HF across valid fixed windows

- `lf_hf_fixed_mean`
  - mean LF/HF across valid fixed windows

- `lf_hf_fixed_n_windows_valid`
  - number of valid fixed windows used for LF/HF

- `lf_hf_fixed_n_windows_total`
  - total number of fixed windows considered

- `lf_hf_fixed_valid_pct`
  - percentage of valid fixed windows

- `lf_hf_fixed_valid_min`
  - total valid fixed-window coverage in minutes

- `lf_hf_fixed_total_min`
  - total fixed-window span considered in minutes

- `lf_hf_fixed_window_sec`
  - fixed-window duration in seconds

- `lf_hf_fixed_hop_sec`
  - hop size between fixed windows in seconds

Internally the fixed-window engine also computes per-window:

- `lf_ms2`
- `hf_ms2`
- `lf_hf`

The main exported `lf`, `hf`, and `lf_hf` values now correspond to the mean of those valid per-window values.

### HRV coverage fields

Coverage fields answer a different question than the summary metrics themselves:

- not "what is the value?"
- but "how much valid data contributed to this value?"

This is important because a metric can have a plausible value but still be based on limited valid coverage.

- `selected_hrv_rmssd_clean_valid_pct`
  - percentage of valid points in the final clean RMSSD series

- `selected_hrv_rmssd_clean_valid_min`
  - valid coverage in minutes for the final clean RMSSD series

- `selected_hrv_rmssd_raw_valid_pct`
  - percentage of valid points in the pre-exclusion RMSSD series

- `selected_hrv_rmssd_raw_valid_min`
  - valid coverage in minutes for the pre-exclusion RMSSD series

- `hrv_tv_*_valid_pct`
  - valid coverage percentage for each sliding-window HRV trace

- `hrv_tv_*_valid_min`
  - valid coverage in minutes for each sliding-window HRV trace

Current sliding-window naming uses explicit stage wording, for example:

- `hrv_tv_sdnn_pre_exclusion_valid_pct`
- `hrv_tv_sdnn_clean_valid_pct`
- `hrv_tv_hf_pre_exclusion_valid_min`
- `hrv_tv_hf_clean_valid_min`

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

Interpretation:

- `*_pre_exclusion_*`
  - after sleep-stage selection only

- `*_clean_*`
  - after the final selected-policy combined mask

### HRV CSV export

- `__HRV_Features_1Hz.csv`
  - `time_sec`
  - `rmssd_clean_ms`
  - `rmssd_raw_ms`
  - optional sliding-window HRV columns

- `__HRV_Mask_1Hz.csv`
  - `time_sec`
  - `sleep_keep`
  - `apnea_keep`
  - `quality_keep`
  - `event_keep`
  - `desat_keep`
  - `combined_keep`

- `__Sleep_Timing.csv`
  - `sleep_onset_rel_sec`
  - `sleep_onset_rel_h`
  - `sleep_onset_rel_hhmm`
  - `sleep_midpoint_rel_sec`
  - `sleep_midpoint_rel_h`
  - `sleep_midpoint_rel_hhmm`
  - `sleep_end_rel_sec`
  - `sleep_end_rel_h`
  - `sleep_end_rel_hhmm`

### Sleep-combo HRV fields

For each subset you may see:

- `combo_*_rmssd_mean_ms`
- `combo_*_sdnn_ms`
- `combo_*_lf_hf`
- `combo_*_lf_n_segments_used`
- `combo_*_lf_hf_fixed_n_windows_valid`

- `combo_*_lf_hf_fixed_n_windows_total`

- `combo_*_lf_hf_fixed_valid_pct`

- `combo_*_lf_hf_fixed_valid_min`

- `combo_*_lf_hf_fixed_total_min`

These are the subset-specific HRV summaries after applying the subset sleep policy and event logic.

Important note:

- the subset-table `LF`, `HF`, and `LF/HF` values are now also fixed-window means
- they are not the legacy global segmented subset values anymore

### Sleep timing and midpoint-half HRV fields

- `sleep_onset_h_from_start`
  - sleep onset in hours from recording start

- `sleep_onset_hhmm_from_start`
  - sleep onset as `HH:MM` from recording start

- `sleep_midpoint_h_from_start`
  - sleep midpoint in hours from recording start

- `sleep_midpoint_hhmm_from_start`
  - sleep midpoint as `HH:MM` from recording start

- `sleep_end_h_from_start`
  - sleep end in hours from recording start

- `sleep_end_hhmm_from_start`
  - sleep end as `HH:MM` from recording start

The grouped summary CSV also now exports selected-policy HRV summaries split by sleep midpoint:

- `selected_first_half_*`
  - selected-policy clean HRV summary for `sleep onset -> sleep midpoint`

- `selected_second_half_*`
  - selected-policy clean HRV summary for `sleep midpoint -> sleep end`

These midpoint-half summaries use the same conventions as the main selected-policy HRV summary:

- `RMSSD mean`
  - mean over valid clean RMSSD windows in that half

- `SDNN`
  - overall clean-RR SDNN in that half

- `LF`, `HF`, `LF/HF`
  - fixed-window means across valid windows inside that half

Current midpoint-half fields are:

- `selected_first_half_rmssd_mean_ms`
- `selected_first_half_sdnn_ms`
- `selected_first_half_lf_ms2`
- `selected_first_half_hf_ms2`
- `selected_first_half_lf_hf_ratio`
- `selected_first_half_lf_hf_fixed_mean`
- `selected_first_half_lf_hf_fixed_median`
- `selected_first_half_lf_hf_fixed_n_windows_valid`
- `selected_second_half_rmssd_mean_ms`
- `selected_second_half_sdnn_ms`
- `selected_second_half_lf_ms2`
- `selected_second_half_hf_ms2`
- `selected_second_half_lf_hf_ratio`
- `selected_second_half_lf_hf_fixed_mean`
- `selected_second_half_lf_hf_fixed_median`
- `selected_second_half_lf_hf_fixed_n_windows_valid`

## PSD Feature

The PSD feature computes spectral summaries from the RR/tachogram signal.

### Main PSD fields

These PSD features are separate from the HRV fixed-window LF/HF summary. They come from the PSD pipeline, not from the sliding HRV TV series.

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

Optional desaturation-aware extension:

- if `HR_EVENT_USE_DESAT_EXTENSION = True`
  - the event window is extended to the end of any overlapping EVENT-gated desaturation window
  - the recovery window then starts after that extended event end

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

Interpretation detail:

- `event_windows_total`
  - number of candidate event-centered windows

- `event_windows_used`
  - number of windows that survived overlap and data-availability checks

- this is a count of windows, not minutes of signal

The grouped summary CSV uses the selected-policy prefix for the current analysis setting:

- `selected_trough_to_peak_response_mean`
- `selected_mean_to_peak_response_mean`
- `selected_event_windows_total`
- `selected_event_windows_used`

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

In the grouped summary CSV these appear as selected-policy fields such as:

- `selected_pat_burden`
- `selected_pat_burden_sleep_hours`
- `selected_pat_burden_total_area_min`
- `selected_pat_burden_n_episodes`
- `selected_pat_burden_n_episodes_used`
- `selected_pat_burden_relative`
- `selected_pat_burden_nan_pct`

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

### Selected-policy exclusion breakdown fields

These quantify mask-based loss within the selected-policy interval:

- `selected_hrv_selected_policy_min`
- `selected_hrv_clean_kept_min`
- `selected_hrv_clean_kept_pct_of_selected`
- `selected_hrv_mask_excluded_total_min`
- `selected_hrv_mask_excluded_total_pct_of_selected`
- `selected_hrv_excluded_apnea_only_min`
- `selected_hrv_excluded_apnea_only_pct_of_selected`
- `selected_hrv_excluded_quality_only_min`
- `selected_hrv_excluded_quality_only_pct_of_selected`
- `selected_hrv_excluded_desat_only_min`
- `selected_hrv_excluded_desat_only_pct_of_selected`
- `selected_hrv_excluded_overlap_min`
- `selected_hrv_excluded_overlap_pct_of_selected`

Important note:

- these are mask-based exclusions only
- they do not include upstream RR-cleaning / PAT-artifact loss before the final mask is applied

So if a recording has poor PAT peak detection or poor physiologic RR quality before masking, that loss is not fully represented by these selected-policy exclusion rows.

## Common Output Files

Typical outputs for one recording are:

- HR CSV
  - `__HR_1Hz.csv`

- HRV CSV
  - `__HRV_Features_1Hz.csv`

- HRV mask CSV
  - `__HRV_Mask_1Hz.csv`

- sleep timing CSV
  - `__Sleep_Timing.csv`

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
  - for `LF`, `HF`, and `LF/HF`, the main reported values now come from fixed-window means

- `PSD` fields
  - spectral organization of the RR/tachogram signal

- `Event-response HR` fields
  - how much HR rises from the event period into the recovery period

- `PAT burden` fields
  - cumulative PAT-amplitude burden around relevant event/desaturation episodes

If you only want the per-subset interpretation:

- `combo_*_...`
  - the same type of summary, but restricted to that sleep subset

## Current Main CSV Interpretation

For the grouped summary CSV, the most common selected-policy fields should currently be read as follows:

- `selected_rmssd_mean_ms`
  - mean clean RMSSD over valid sliding RMSSD windows

- `selected_rmssd_median_ms`
  - median clean RMSSD over valid sliding RMSSD windows

- `selected_sdnn_ms`
  - overall clean-RR SDNN

- `selected_lf`
  - fixed-window LF mean across valid windows

- `selected_hf`
  - fixed-window HF mean across valid windows

- `selected_lf_hf`
  - fixed-window LF/HF mean across valid windows

- `selected_lf_hf_fixed_median`
  - fixed-window LF/HF median across valid windows

- `selected_lf_hf_fixed_n_windows_valid`
  - number of valid fixed windows contributing to the fixed-window frequency summary

- `selected_hr_valid_pct`
  - valid PAT-derived HR coverage after the selected-policy combined mask

- `selected_hrv_rmssd_clean_valid_pct`
  - valid clean RMSSD coverage after the selected-policy combined mask

## Important Note About Legacy Naming

The config key `delta_hr` is still present for backward compatibility.

In the current codebase it refers to:

- Event-Response HR metrics and plots

not to the older lag-based delta-HR signal.
