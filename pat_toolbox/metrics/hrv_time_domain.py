from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .. import config
from ..core.windows import passes_time_domain_window_gate


def _rmssd(rr_ms: np.ndarray) -> float:
    """Robust RMSSD (ms) from RR in ms: reject outlier successive diffs before RMSSD."""
    if rr_ms.size < 3:
        return np.nan

    diffs = np.diff(rr_ms).astype(float)
    if diffs.size < 2:
        return np.nan

    hard_cap_ms = float(getattr(config, "HRV_RMSSD_DIFF_HARD_CAP_MS"))
    diffs = diffs[np.abs(diffs) <= hard_cap_ms]
    if diffs.size < 2:
        return np.nan

    med = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - med)))
    if mad > 0:
        sigma = 1.4826 * mad
        k = float(getattr(config, "HRV_RMSSD_DIFF_MAD_SIGMAS"))
        keep = np.abs(diffs - med) <= (k * sigma)
        diffs = diffs[keep]

    min_diffs = int(getattr(config, "HRV_RMSSD_MIN_DIFFS"))
    if diffs.size < min_diffs:
        return np.nan

    rmssd = float(np.sqrt(np.mean(diffs ** 2)))

    rmssd_floor_ms = float(getattr(config, "HRV_RMSSD_FLOOR_MS", 2.0))
    if not np.isfinite(rmssd) or rmssd < rmssd_floor_ms:
        return np.nan

    return rmssd


def _sdnn(rr_ms: np.ndarray) -> float:
    """Compute SDNN (ms) from RR intervals in ms (sample std)."""
    if rr_ms.size < 2:
        return np.nan
    return float(np.std(rr_ms, ddof=1))


def _calculate_rmssd_series(
    t_hrv: np.ndarray,
    rr_mid: np.ndarray,
    rr_ms: np.ndarray,
    window_sec: float,
    *,
    max_gap_sec: float = 3.0,
    min_span_sec: float = 10.0,
) -> Tuple[np.ndarray, List[float]]:
    """
    Calculate RMSSD series over a 1 Hz grid.
    """
    target_fs = float(getattr(config, "HRV_TARGET_FS_HZ", 1.0))
    half_win = 0.5 * float(window_sec)

    min_intervals = int(getattr(config, "HRV_MIN_INTERVALS_PER_WINDOW", 4))
    min_cov = float(getattr(config, "HRV_MIN_WINDOW_COVERAGE", 0.0))

    veto_bigdiff = bool(getattr(config, "HRV_RMSSD_VETO_BIGDIFF", True))
    bigdiff_thr_ms = float(getattr(config, "HRV_RMSSD_BIGDIFF_THR_MS", 250.0))
    bigdiff_max_frac = float(getattr(config, "HRV_RMSSD_BIGDIFF_MAX_FRAC", 0.20))

    rmssd_floor_ms = float(getattr(config, "HRV_RMSSD_FLOOR_MS", 2.0))

    rmssd_1hz = np.full_like(t_hrv, fill_value=np.nan, dtype=float)
    rmssd_windows_list: List[float] = []

    if t_hrv.size == 0 or rr_mid.size == 0 or rr_ms.size == 0:
        return rmssd_1hz, rmssd_windows_list

    n = rr_mid.size
    left = 0
    right = 0

    for i, t in enumerate(t_hrv):
        start = t - half_win
        end = t + half_win

        while left < n and rr_mid[left] < start:
            left += 1
        if right < left:
            right = left
        while right < n and rr_mid[right] < end:
            right += 1

        rr_win_ms = rr_ms[left:right]
        rr_mid_win = rr_mid[left:right]

        if not passes_time_domain_window_gate(
            rr_mid_win,
            window_sec=float(window_sec),
            min_intervals=min_intervals,
            max_gap_sec=float(max_gap_sec),
            min_span_sec=float(min_span_sec),
            min_cov=min_cov,
        ):
            continue

        if veto_bigdiff and rr_win_ms.size >= 3:
            diffs = np.abs(np.diff(rr_win_ms.astype(float)))
            if diffs.size > 0:
                frac_big = float(np.mean(diffs > bigdiff_thr_ms))
                if frac_big > bigdiff_max_frac:
                    continue

        rmssd_win = _rmssd(rr_win_ms)
        if not np.isfinite(rmssd_win) or rmssd_win < rmssd_floor_ms:
            continue

        rmssd_1hz[i] = rmssd_win
        rmssd_windows_list.append(rmssd_win)

    if not np.any(np.isnan(rmssd_1hz)):
        smooth_sec = float(getattr(config, "HRV_SMOOTHING_WINDOW_SEC", 0.0))
        smooth_samples = int(round(smooth_sec * target_fs))
        if smooth_samples > 1:
            kernel = np.ones(smooth_samples, dtype=float) / float(smooth_samples)
            rmssd_1hz = np.convolve(rmssd_1hz, kernel, mode="same")

    return rmssd_1hz, rmssd_windows_list
