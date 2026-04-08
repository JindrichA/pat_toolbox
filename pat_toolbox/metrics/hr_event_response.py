from __future__ import annotations

from typing import Dict

import numpy as np

from .. import config, masking


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return []
    d = np.diff(mask.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]


def summarize_event_hr_response(
    t_hr: np.ndarray,
    hr_bpm: np.ndarray,
    aux_df,
    *,
    include_set: set[int],
) -> Dict[str, float]:
    t_hr = np.asarray(t_hr, dtype=float)
    hr = np.asarray(hr_bpm, dtype=float)
    out = {
        "n_event_windows": 0.0,
        "n_used_windows": 0.0,
        "peak_minus_baseline": np.nan,
        "peak_to_trough": np.nan,
        "post_peak_minus_pre_mean": np.nan,
    }
    if t_hr.size == 0 or hr.size == 0 or hr.size != t_hr.size or aux_df is None:
        return out

    baseline_sec = float(getattr(config, "HR_EVENT_BASELINE_SEC", 30.0))
    response_sec = float(getattr(config, "HR_EVENT_RESPONSE_SEC", 45.0))
    post_sec = float(getattr(config, "HR_EVENT_POST_SEC", 45.0))
    min_samples = int(getattr(config, "HR_EVENT_MIN_SAMPLES", 3))

    policy = masking.policy_from_config(include_stages=include_set, force_sleep=True)
    bundle = masking.build_mask_bundle(t_hr, aux_df, policy=policy)

    inside_event = np.asarray(bundle.sleep_keep, dtype=bool) & (
        (~np.asarray(bundle.event_keep, dtype=bool))
        | (~np.asarray(bundle.desat_keep, dtype=bool))
    )
    runs = _contiguous_true_runs(inside_event)
    out["n_event_windows"] = float(len(runs))
    if not runs:
        return out

    peak_minus_baseline_vals: list[float] = []
    peak_to_trough_vals: list[float] = []
    post_peak_minus_pre_mean_vals: list[float] = []

    sleep_keep = np.asarray(bundle.sleep_keep, dtype=bool)
    quiet_keep = sleep_keep & np.asarray(bundle.event_keep, dtype=bool) & np.asarray(bundle.desat_keep, dtype=bool)

    for start_idx, end_idx in runs:
        start_t = float(t_hr[start_idx])
        end_t = float(t_hr[end_idx - 1])

        baseline_mask = (
            (t_hr >= (start_t - baseline_sec))
            & (t_hr < start_t)
            & quiet_keep
            & np.isfinite(hr)
        )
        response_mask = (
            (t_hr >= start_t)
            & (t_hr <= (end_t + response_sec))
            & sleep_keep
            & np.isfinite(hr)
        )
        centered_mask = (
            (t_hr >= (start_t - baseline_sec))
            & (t_hr <= (end_t + response_sec))
            & sleep_keep
            & np.isfinite(hr)
        )
        post_mask = (
            (t_hr >= end_t)
            & (t_hr <= (end_t + post_sec))
            & sleep_keep
            & np.isfinite(hr)
        )

        if np.count_nonzero(baseline_mask) < min_samples or np.count_nonzero(response_mask) < min_samples:
            continue

        baseline_vals = hr[baseline_mask]
        response_vals = hr[response_mask]
        centered_vals = hr[centered_mask]
        post_vals = hr[post_mask]

        baseline_median = float(np.nanmedian(baseline_vals))
        baseline_mean = float(np.nanmean(baseline_vals))
        response_peak = float(np.nanmax(response_vals))

        peak_minus_baseline_vals.append(response_peak - baseline_median)

        if centered_vals.size >= min_samples:
            peak_to_trough_vals.append(float(np.nanmax(centered_vals) - np.nanmin(centered_vals)))

        if post_vals.size >= min_samples:
            post_peak_minus_pre_mean_vals.append(float(np.nanmax(post_vals) - baseline_mean))

    out["n_used_windows"] = float(len(peak_minus_baseline_vals))
    if peak_minus_baseline_vals:
        out["peak_minus_baseline"] = float(np.nanmean(np.asarray(peak_minus_baseline_vals, dtype=float)))
    if peak_to_trough_vals:
        out["peak_to_trough"] = float(np.nanmean(np.asarray(peak_to_trough_vals, dtype=float)))
    if post_peak_minus_pre_mean_vals:
        out["post_peak_minus_pre_mean"] = float(np.nanmean(np.asarray(post_peak_minus_pre_mean_vals, dtype=float)))
    return out
