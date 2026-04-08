from __future__ import annotations

import numpy as np


def split_into_contiguous_runs(t: np.ndarray, max_gap_sec: float) -> list[np.ndarray]:
    """
    Return list of index arrays for contiguous runs where diff(t) <= max_gap_sec.
    Assumes t is sorted.
    """
    if t.size == 0:
        return []
    if t.size == 1:
        return [np.array([0], dtype=int)]

    cut = np.where(np.diff(t) > float(max_gap_sec))[0] + 1
    idx = np.arange(t.size)
    runs = [seg for seg in np.split(idx, cut) if seg.size > 0]
    return runs


def passes_time_domain_window_gate(
    rr_mid_win: np.ndarray,
    *,
    window_sec: float,
    min_intervals: int,
    max_gap_sec: float,
    min_span_sec: float,
    min_cov: float,
) -> bool:
    """Return True when an RR window passes the shared time-domain gate."""
    k = int(np.size(rr_mid_win))
    if k < int(min_intervals):
        return False

    if k < 2:
        return False

    rr_mid_win = np.asarray(rr_mid_win, dtype=float)
    gaps = np.diff(rr_mid_win)
    if gaps.size > 0 and np.any(gaps > float(max_gap_sec)):
        return False

    span = float(rr_mid_win[-1] - rr_mid_win[0])
    if span < float(min_span_sec):
        return False

    if float(min_cov) > 0.0 and span < (float(min_cov) * float(window_sec)):
        return False

    return True


def interp_with_gaps(
    t_grid: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    max_gap_sec: float,
) -> np.ndarray:
    """
    Interpolate y(t) onto t_grid, but do NOT bridge across gaps.
    Returns NaN in gap regions.
    """
    out = np.full_like(t_grid, np.nan, dtype=float)
    if t.size < 2:
        return out

    cuts = np.where(np.diff(t) > float(max_gap_sec))[0] + 1
    for idx in np.split(np.arange(t.size), cuts):
        if idx.size < 2:
            continue
        t0, t1 = t[idx[0]], t[idx[-1]]
        mask = (t_grid >= t0) & (t_grid <= t1)
        if np.any(mask):
            out[mask] = np.interp(t_grid[mask], t[idx], y[idx])
    return out
