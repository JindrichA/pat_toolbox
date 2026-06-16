from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple

import numpy as np

from .. import config, masking


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


def _estimate_dt_sec(t_sec: np.ndarray) -> float:
    t = np.asarray(t_sec, dtype=float)
    d = np.diff(t[np.isfinite(t)])
    d = d[d > 0]
    return float(np.median(d)) if d.size else 1.0


def _minutes_from_mask(mask: np.ndarray, t_sec: np.ndarray) -> float:
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return 0.0
    dt_sec = _estimate_dt_sec(t_sec)
    return float(np.count_nonzero(m) * dt_sec / 60.0)


def compute_pat_burden_from_pat_amp(
    *,
    t_sec: np.ndarray,
    pat_amp: np.ndarray,
    aux_df,
    include_set: Optional[set[int]] = None,
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

    policy = masking.policy_from_config(include_stages=include_set, force_sleep=(include_set is not None))
    bundle = masking.build_mask_bundle(t_sec, aux_df, policy=policy)

    m_sleep_keep = np.asarray(bundle.sleep_keep, dtype=bool)
    m_evt_keep = np.asarray(bundle.event_keep & bundle.desat_keep, dtype=bool)
    finite_pat = np.isfinite(y)

    # "Inside event+desat region" = NOT keep
    m_inside = np.asarray(m_sleep_keep, dtype=bool) & (~np.asarray(m_evt_keep, dtype=bool))
    m_inside_finite = m_inside & finite_pat

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

    skipped_reason_counts: Dict[str, int] = {}
    n_used = 0
    for ep in episodes:
        if ep.get("used"):
            n_used += 1
            continue
        reason = str(ep.get("reason", "unknown"))
        skipped_reason_counts[reason] = skipped_reason_counts.get(reason, 0) + 1

    diag: Dict[str, Any] = {
        "sleep_hours": float(sleep_hours),
        "n_episodes": int(len(runs)),
        "n_episodes_used": int(n_used),
        "n_episodes_skipped": int(len(episodes) - n_used),
        "total_area_min": float(total_area),
        "burden_per_sleep_hour": float(burden) if np.isfinite(burden) else np.nan,
        "relative": bool(use_rel),
        "baseline_lookback_sec": float(lookback),
        "baseline_pctl": float(pctl),
        "baseline_min_samples": int(min_base_n),
        "min_episode_sec": float(min_ep_sec),
        "pat_amp_total_min": _minutes_from_mask(np.isfinite(t_sec), t_sec),
        "pat_amp_finite_min": _minutes_from_mask(finite_pat, t_sec),
        "sleep_selected_min": _minutes_from_mask(m_sleep_keep, t_sec),
        "inside_event_desat_min": _minutes_from_mask(m_inside, t_sec),
        "inside_event_desat_finite_min": _minutes_from_mask(m_inside_finite, t_sec),
        "outside_event_desat_min": _minutes_from_mask(m_sleep_keep & m_evt_keep, t_sec),
        "pat_amp_invalid_inside_min": _minutes_from_mask(m_inside & (~finite_pat), t_sec),
        "pat_amp_invalid_selected_min": _minutes_from_mask(m_sleep_keep & (~finite_pat), t_sec),
        "nan_pct_inside": float(100.0 * np.mean(~finite_pat[m_inside])) if np.any(m_inside) else np.nan,
        "skipped_reason_counts": skipped_reason_counts,
    }

    return (float(burden) if np.isfinite(burden) else np.nan), diag, episodes
