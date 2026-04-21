from __future__ import annotations

from typing import Dict, Optional

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


def _smooth_hr_signal(t_hr: np.ndarray, hr_bpm: np.ndarray, smooth_sec: float) -> np.ndarray:
    hr = np.asarray(hr_bpm, dtype=float)
    if hr.size == 0 or smooth_sec <= 0:
        return hr.copy()

    dt = np.diff(np.asarray(t_hr, dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return hr.copy()

    fs = 1.0 / float(np.median(dt))
    win = max(1, int(round(float(smooth_sec) * fs)))
    if win <= 1:
        return hr.copy()

    kernel = np.ones(win, dtype=float)
    valid = np.isfinite(hr).astype(float)
    filled = np.where(np.isfinite(hr), hr, 0.0)
    num = np.convolve(filled, kernel, mode="same")
    den = np.convolve(valid, kernel, mode="same")
    out = np.full_like(hr, np.nan, dtype=float)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def _extend_event_end_with_desat(
    start_t: float,
    fixed_event_end_t: float,
    gated_desat_windows: tuple[tuple[float, float], ...],
    *,
    max_desat_start_t: float,
) -> tuple[float, bool]:
    if not gated_desat_windows:
        return fixed_event_end_t, False

    event_end_t = float(fixed_event_end_t)
    used_extension = False
    changed = True
    while changed:
        changed = False
        for a, b in gated_desat_windows:
            if not (np.isfinite(a) and np.isfinite(b) and b > a):
                continue
            starts_close_enough = a <= max_desat_start_t
            overlaps = starts_close_enough and (b >= start_t)
            if overlaps and b > event_end_t:
                event_end_t = float(b)
                used_extension = True
                changed = True
    return event_end_t, used_extension


def extract_event_hr_windows(
    t_hr: np.ndarray,
    hr_bpm: np.ndarray,
    aux_df,
    *,
    include_set: Optional[set[int]] = None,
) -> list[dict[str, float]]:
    t_hr = np.asarray(t_hr, dtype=float)
    hr = np.asarray(hr_bpm, dtype=float)
    if t_hr.size == 0 or hr.size == 0 or hr.size != t_hr.size or aux_df is None:
        return []

    smooth_sec = float(getattr(config, "HR_EVENT_SMOOTH_SEC", 5.0))
    event_window_sec = float(getattr(config, "HR_EVENT_WINDOW_SEC", 15.0))
    recovery_end_sec = float(getattr(config, "HR_EVENT_RECOVERY_END_SEC", 45.0))
    min_samples = int(getattr(config, "HR_EVENT_MIN_SAMPLES", 3))
    hr_smooth = _smooth_hr_signal(t_hr, hr, smooth_sec)
    use_desat_extension = bool(getattr(config, "HR_EVENT_USE_DESAT_EXTENSION", False))

    if include_set is None:
        policy = masking.policy_from_config()
    else:
        policy = masking.policy_from_config(include_stages=include_set, force_sleep=True)
    bundle = masking.build_mask_bundle(t_hr, aux_df, policy=policy)

    inside_event = np.asarray(bundle.sleep_keep, dtype=bool) & (
        (~np.asarray(bundle.event_keep, dtype=bool))
        | (~np.asarray(bundle.desat_keep, dtype=bool))
    )
    runs = _contiguous_true_runs(inside_event)
    if not runs and bundle.active_event_times_sec.size == 0:
        return []

    windows: list[dict[str, float]] = []
    event_starts = [float(t_hr[start_idx]) for start_idx, _end_idx in runs]
    recovery_duration_sec = max(0.0, float(recovery_end_sec - event_window_sec))
    sleep_keep = np.asarray(bundle.sleep_keep, dtype=bool)

    for i, start_t in enumerate(event_starts):
        next_event_t = event_starts[i + 1] if i + 1 < len(event_starts) else np.inf
        fixed_event_end_t = float(start_t + event_window_sec)
        nominal_recovery_end_t = float(start_t + recovery_end_sec)
        event_end_t = fixed_event_end_t
        used_desat_extension = False
        if use_desat_extension:
            event_end_t, used_desat_extension = _extend_event_end_with_desat(
                start_t,
                fixed_event_end_t,
                bundle.gated_desat_windows,
                max_desat_start_t=nominal_recovery_end_t,
            )
        recovery_start_t = float(event_end_t)
        recovery_end_t = float(recovery_start_t + recovery_duration_sec)

        if next_event_t <= recovery_end_t:
            continue

        event_mask = (t_hr >= start_t) & (t_hr <= event_end_t) & sleep_keep & np.isfinite(hr_smooth)
        recovery_mask = (t_hr >= recovery_start_t) & (t_hr <= recovery_end_t) & sleep_keep & np.isfinite(hr_smooth)
        if np.count_nonzero(event_mask) < min_samples or np.count_nonzero(recovery_mask) < min_samples:
            continue

        event_vals = hr_smooth[event_mask]
        event_times = t_hr[event_mask]
        recovery_vals = hr_smooth[recovery_mask]
        recovery_times = t_hr[recovery_mask]

        event_min_idx = int(np.nanargmin(event_vals))
        event_min = float(event_vals[event_min_idx])
        event_min_t = float(event_times[event_min_idx])
        event_mean = float(np.nanmean(event_vals))
        recovery_max_idx = int(np.nanargmax(recovery_vals))
        recovery_max = float(recovery_vals[recovery_max_idx])
        recovery_max_t = float(recovery_times[recovery_max_idx])
        trough_to_peak_response = float(recovery_max - event_min)
        mean_to_peak_response = float(recovery_max - event_mean)

        windows.append(
            {
                "event_start_t": start_t,
                "fixed_event_end_t": fixed_event_end_t,
                "event_end_t": event_end_t,
                "recovery_start_t": recovery_start_t,
                "recovery_end_t": recovery_end_t,
                "used_desat_extension": bool(used_desat_extension),
                "event_min_t": event_min_t,
                "event_min_hr": event_min,
                "event_mean_hr": event_mean,
                "recovery_max_t": recovery_max_t,
                "recovery_max_hr": recovery_max,
                "trough_to_peak_response": trough_to_peak_response,
                "mean_to_peak_response": mean_to_peak_response,
            }
        )

    return windows


def summarize_event_hr_response(
    t_hr: np.ndarray,
    hr_bpm: np.ndarray,
    aux_df,
    *,
    include_set: set[int],
) -> Dict[str, float]:
    out = {
        "n_event_windows": 0.0,
        "n_used_windows": 0.0,
        "trough_to_peak_response_mean": np.nan,
        "mean_to_peak_response_mean": np.nan,
    }
    windows = extract_event_hr_windows(t_hr, hr_bpm, aux_df, include_set=include_set)
    out["n_event_windows"] = float(len(windows))
    if not windows:
        return out

    out["n_used_windows"] = float(len(windows))
    out["trough_to_peak_response_mean"] = float(np.nanmean([w["trough_to_peak_response"] for w in windows]))
    out["mean_to_peak_response_mean"] = float(np.nanmean([w["mean_to_peak_response"] for w in windows]))
    return out
