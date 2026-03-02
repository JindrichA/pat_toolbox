from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

from .. import config, sleep_mask, io_aux_csv


def _contiguous_true_runs(m: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (start_idx, end_idx_exclusive) for contiguous True runs."""
    m = np.asarray(m, dtype=bool)
    if m.size == 0 or not np.any(m):
        return []
    d = np.diff(m.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if m[0]:
        starts = np.r_[0, starts]
    if m[-1]:
        ends = np.r_[ends, m.size]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]


def _sleep_hours_from_mask(m_sleep_keep: np.ndarray, t_sec: np.ndarray) -> float:
    """Compute sleep hours from a boolean keep mask on an arbitrary timebase."""
    ok = np.asarray(m_sleep_keep, dtype=bool) & np.isfinite(t_sec)
    if ok.size < 2 or np.count_nonzero(ok) < 2:
        return 0.0

    # integrate time where ok is True using dt between consecutive samples
    t = np.asarray(t_sec, dtype=float)
    dt = np.diff(t)
    dt = np.clip(dt, 0.0, None)
    # count an interval if BOTH endpoints are in sleep_keep (conservative)
    keep_interval = ok[:-1] & ok[1:]
    sleep_sec = float(np.sum(dt[keep_interval]))
    return sleep_sec / 3600.0


def compute_pat_burden_from_pat_amp(
    *,
    t_sec: np.ndarray,
    pat_amp: np.ndarray,
    aux_df,
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Compute PAT burden within the *excluded* region defined by io_aux_csv.build_time_exclusion_mask
    (typically event+desat), restricted to included sleep stages.

    Returns:
      pat_burden (amp·min per sleep hour) OR (relative·min per sleep hour if PAT_BURDEN_RELATIVE)
      diag dict
      episodes list with per-episode contributions
    """
    t_sec = np.asarray(t_sec, dtype=float)
    y = np.asarray(pat_amp, dtype=float)

    if t_sec.size == 0 or y.size == 0 or t_sec.size != y.size:
        return np.nan, {"reason": "empty_or_mismatched"}, []

    if aux_df is None:
        return np.nan, {"reason": "no_aux_df"}, []

    # --- masks on this timebase ---
    m_sleep_keep = sleep_mask.build_sleep_include_mask_for_times(t_sec, aux_df)
    if m_sleep_keep is None:
        # treat as "all sleep allowed" if masking disabled/unavailable
        m_sleep_keep = np.ones_like(t_sec, dtype=bool)

    m_evt_keep = io_aux_csv.build_time_exclusion_mask(t_sec, aux_df)  # True = keep (outside excluded region)
    if m_evt_keep is None:
        return np.nan, {"reason": "no_event_mask"}, []

    # "Inside event+desat region" = NOT keep
    m_inside = np.asarray(m_sleep_keep, dtype=bool) & (~np.asarray(m_evt_keep, dtype=bool))

    sleep_hours = _sleep_hours_from_mask(np.asarray(m_sleep_keep, dtype=bool), t_sec)
    if sleep_hours <= 0:
        return np.nan, {"reason": "sleep_hours<=0", "sleep_hours": sleep_hours}, []

    runs = _contiguous_true_runs(m_inside)

    min_ep_sec = float(getattr(config, "PAT_BURDEN_MIN_EPISODE_SEC", 5.0))
    lookback = float(getattr(config, "PAT_BURDEN_BASELINE_LOOKBACK_SEC", 30.0))
    pctl = float(getattr(config, "PAT_BURDEN_BASELINE_PCTL", 95.0))
    min_base_n = int(getattr(config, "PAT_BURDEN_BASELINE_MIN_SAMPLES", 5))
    use_rel = bool(getattr(config, "PAT_BURDEN_RELATIVE", False))

    total_area = 0.0
    episodes: List[Dict[str, Any]] = []

    # helper mask for "eligible baseline samples": sleep_keep AND outside excluded region
    m_baseline_ok = np.asarray(m_sleep_keep, dtype=bool) & np.asarray(m_evt_keep, dtype=bool)

    for (s, e) in runs:
        t0 = float(t_sec[s])
        t1 = float(t_sec[e - 1])
        if not (np.isfinite(t0) and np.isfinite(t1)):
            continue
        dur = t1 - t0
        if dur < min_ep_sec:
            continue

        # baseline window: [t0 - lookback, t0)
        w0 = t0 - lookback
        w1 = t0
        m_pre = (t_sec >= w0) & (t_sec < w1) & m_baseline_ok & np.isfinite(y)

        if np.count_nonzero(m_pre) < min_base_n:
            # if baseline is not reliable, skip this episode (HB would skip)
            episodes.append({
                "t_start": t0,
                "t_end": t1,
                "dur_sec": dur,
                "used": False,
                "reason": "insufficient_baseline",
                "baseline_n": int(np.count_nonzero(m_pre)),
            })
            continue

        baseline = float(np.nanpercentile(y[m_pre], pctl))
        if not np.isfinite(baseline):
            episodes.append({
                "t_start": t0,
                "t_end": t1,
                "dur_sec": dur,
                "used": False,
                "reason": "baseline_nonfinite",
            })
            continue

        # episode samples
        tt = t_sec[s:e]
        yy = y[s:e]

        good = np.isfinite(tt) & np.isfinite(yy)
        if np.count_nonzero(good) < 2:
            episodes.append({
                "t_start": t0,
                "t_end": t1,
                "dur_sec": dur,
                "used": False,
                "reason": "no_finite_episode",
                "baseline": baseline,
            })
            continue

        tt = tt[good]
        yy = yy[good]

        drop = baseline - yy
        drop = np.maximum(drop, 0.0)

        if use_rel:
            denom = baseline if baseline > 0 else np.nan
            drop = drop / denom

        # integrate (trapezoid) in seconds -> convert to minutes
        area_sec = float(np.trapz(drop, tt))
        area_min = area_sec / 60.0

        total_area += area_min

        episodes.append({
            "t_start": t0,
            "t_end": t1,
            "dur_sec": float(tt[-1] - tt[0]),
            "used": True,
            "baseline": baseline,
            "area_min": area_min,
            "relative": use_rel,
            "baseline_n": int(np.count_nonzero(m_pre)),
        })

    burden = total_area / sleep_hours if sleep_hours > 0 else np.nan

    diag: Dict[str, Any] = {
        "sleep_hours": float(sleep_hours),
        "n_episodes": int(len(runs)),
        "n_episodes_used": int(sum(1 for ep in episodes if ep.get("used"))),
        "total_area_min": float(total_area),
        "burden_per_sleep_hour": float(burden) if np.isfinite(burden) else np.nan,
        "relative": bool(use_rel),
        "baseline_lookback_sec": float(lookback),
        "baseline_pctl": float(pctl),
        "min_episode_sec": float(min_ep_sec),
    }

    return (float(burden) if np.isfinite(burden) else np.nan), diag, episodes
