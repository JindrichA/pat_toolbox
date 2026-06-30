from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .. import config, masking, paths


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


def _event_starts_from_bundle(bundle: masking.MaskBundle, t_hr: np.ndarray) -> list[float]:
    inside_event = np.asarray(bundle.sleep_keep, dtype=bool) & (
        (~np.asarray(bundle.event_keep, dtype=bool))
        | (~np.asarray(bundle.desat_keep, dtype=bool))
    )
    runs = _contiguous_true_runs(inside_event)
    if runs:
        starts = [float(t_hr[start_idx]) for start_idx, _end_idx in runs]
    else:
        starts = [float(t) for t in np.asarray(bundle.active_event_times_sec, dtype=float) if np.isfinite(t)]
    return sorted(set(starts))


def _derive_ensemble_search_window(
    t_hr: np.ndarray,
    hr_smooth: np.ndarray,
    event_starts: list[float],
    sleep_keep: np.ndarray,
    *,
    event_window_sec: float,
    recovery_end_sec: float,
    min_samples: int,
) -> dict[str, float | str]:
    fallback = {
        "dhr_search_window_source": "fallback",
        "dhr_search_start_offset_sec": float(event_window_sec),
        "dhr_search_end_offset_sec": float(recovery_end_sec),
        "dhr_ensemble_events_used": 0.0,
        "dhr_ensemble_peak_offset_sec": np.nan,
    }
    if not event_starts:
        return fallback

    min_events = int(getattr(config, "HR_EVENT_ENSEMBLE_MIN_EVENTS", 5))
    pre_sec = float(getattr(config, "HR_EVENT_ENSEMBLE_PRE_SEC", 20.0))
    step_sec = float(getattr(config, "HR_EVENT_ENSEMBLE_GRID_SEC", 1.0))
    peak_margin_sec = float(getattr(config, "HR_EVENT_ENSEMBLE_PEAK_MARGIN_SEC", 10.0))
    step_sec = max(step_sec, 0.1)
    grid = np.arange(-pre_sec, recovery_end_sec + 0.5 * step_sec, step_sec, dtype=float)
    if grid.size < 3:
        return fallback

    curves: list[np.ndarray] = []
    for start_t in event_starts:
        abs_t = start_t + grid
        in_range = (abs_t >= t_hr[0]) & (abs_t <= t_hr[-1])
        if np.count_nonzero(in_range) < min_samples:
            continue
        sleep_interp = np.interp(abs_t, t_hr, sleep_keep.astype(float), left=0.0, right=0.0) >= 0.5
        y = np.interp(abs_t, t_hr, hr_smooth, left=np.nan, right=np.nan)
        y[~in_range | ~sleep_interp] = np.nan
        if np.count_nonzero(np.isfinite(y[(grid >= 0.0) & (grid <= recovery_end_sec)])) >= min_samples:
            curves.append(y)

    if len(curves) < min_events:
        return fallback | {"dhr_ensemble_events_used": float(len(curves))}

    with np.errstate(invalid="ignore", divide="ignore"):
        ensemble = np.nanmean(np.vstack(curves), axis=0)

    search_mask = (grid >= event_window_sec) & (grid <= recovery_end_sec) & np.isfinite(ensemble)
    if np.count_nonzero(search_mask) < min_samples:
        return fallback | {"dhr_ensemble_events_used": float(len(curves))}

    search_grid = grid[search_mask]
    search_vals = ensemble[search_mask]
    peak_idx = int(np.nanargmax(search_vals))
    peak_offset = float(search_grid[peak_idx])
    search_start = max(event_window_sec, peak_offset - peak_margin_sec)
    search_end = min(recovery_end_sec, peak_offset + peak_margin_sec)
    if not (np.isfinite(search_start) and np.isfinite(search_end) and search_end > search_start):
        return fallback | {"dhr_ensemble_events_used": float(len(curves))}

    return {
        "dhr_search_window_source": "ensemble",
        "dhr_search_start_offset_sec": float(search_start),
        "dhr_search_end_offset_sec": float(search_end),
        "dhr_ensemble_events_used": float(len(curves)),
        "dhr_ensemble_peak_offset_sec": peak_offset,
    }


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

    event_starts = _event_starts_from_bundle(bundle, t_hr)
    if not event_starts:
        return []

    windows: list[dict[str, float]] = []
    sleep_keep = np.asarray(bundle.sleep_keep, dtype=bool)
    ensemble_info = _derive_ensemble_search_window(
        t_hr,
        hr_smooth,
        event_starts,
        sleep_keep,
        event_window_sec=event_window_sec,
        recovery_end_sec=recovery_end_sec,
        min_samples=min_samples,
    )
    search_start_offset_sec = float(ensemble_info["dhr_search_start_offset_sec"])
    search_end_offset_sec = float(ensemble_info["dhr_search_end_offset_sec"])

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
        recovery_start_t = float(start_t + search_start_offset_sec)
        recovery_end_t = float(start_t + search_end_offset_sec)
        if event_end_t > recovery_start_t:
            recovery_start_t = float(event_end_t)

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
        recovery_max_idx = int(np.nanargmax(recovery_vals))
        recovery_max = float(recovery_vals[recovery_max_idx])
        recovery_max_t = float(recovery_times[recovery_max_idx])
        dhr_bpm = float(recovery_max - event_min)

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
                "recovery_max_t": recovery_max_t,
                "recovery_max_hr": recovery_max,
                "dhr_bpm": dhr_bpm,
                **ensemble_info,
            }
        )

    return windows


def summarize_event_hr_response(
    t_hr: np.ndarray,
    hr_bpm: np.ndarray,
    aux_df,
    *,
    include_set: Optional[set[int]] = None,
) -> Dict[str, float]:
    windows = extract_event_hr_windows(t_hr, hr_bpm, aux_df, include_set=include_set)
    return summarize_event_hr_response_from_windows(windows)


def summarize_event_hr_response_from_windows(
    windows: list[dict[str, float]],
) -> Dict[str, float]:
    out = {
        "n_event_windows": 0.0,
        "n_used_windows": 0.0,
        "dhr_mean_bpm": np.nan,
        "dhr_median_bpm": np.nan,
        "dhr_p25_bpm": np.nan,
        "dhr_p75_bpm": np.nan,
        "dhr_search_window_source": "none",
        "dhr_search_start_offset_sec": np.nan,
        "dhr_search_end_offset_sec": np.nan,
        "dhr_ensemble_events_used": np.nan,
        "dhr_ensemble_peak_offset_sec": np.nan,
    }
    out["n_event_windows"] = float(len(windows))
    if not windows:
        return out

    out["n_used_windows"] = float(len(windows))
    vals = np.asarray([w["dhr_bpm"] for w in windows], dtype=float)
    out["dhr_mean_bpm"] = float(np.nanmean(vals))
    out["dhr_median_bpm"] = float(np.nanmedian(vals))
    out["dhr_p25_bpm"] = float(np.nanpercentile(vals, 25))
    out["dhr_p75_bpm"] = float(np.nanpercentile(vals, 75))
    first = windows[0]
    for key in (
        "dhr_search_window_source",
        "dhr_search_start_offset_sec",
        "dhr_search_end_offset_sec",
        "dhr_ensemble_events_used",
        "dhr_ensemble_peak_offset_sec",
    ):
        out[key] = first.get(key, out[key])
    return out


def save_event_hr_windows_to_csv(
    edf_path: Path,
    windows: list[dict[str, float]],
) -> Path | None:
    if not windows:
        return None

    out_folder = paths.get_output_folder(
        getattr(config, "DELTA_HR_OUTPUT_SUBFOLDER", getattr(config, "HR_OUTPUT_SUBFOLDER", config.OUTPUT_SUBFOLDER)),
    )
    out_csv = out_folder / f"{edf_path.stem}__Event_HR_Windows.csv"
    fieldnames = list(windows[0].keys())

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in windows:
            writer.writerow(row)

    return out_csv
