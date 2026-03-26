# pat_toolbox/metrics/hrv.py

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Optional, TYPE_CHECKING, List

import numpy as np
import pandas as pd
from scipy.signal import welch

from .. import config, paths, masking
from . import hr as hr_metrics

if TYPE_CHECKING:
    from pandas import DataFrame as pd_DataFrame


def _calculate_lfhf_fixed_windows(
    rr_mid: np.ndarray,
    rr_ms: np.ndarray,
    duration_sec: float,
    *,
    window_sec: float = 300.0,
    hop_sec: float = 300.0,
    fs_resample: float = 4.0,
    max_gap_sec: float = 3.0,
    min_rr: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Compute LF/HF on fixed-length windows (publication-style).
    Non-overlapping by default (hop_sec == window_sec).
    """
    out = {
        "t_win_center_sec": np.array([], dtype=float),
        "lf_ms2": np.array([], dtype=float),
        "hf_ms2": np.array([], dtype=float),
        "lf_hf": np.array([], dtype=float),
        "valid_mask": np.array([], dtype=bool),
    }

    if rr_mid.size == 0 or rr_ms.size == 0 or duration_sec <= 0:
        return out

    window_sec = float(window_sec)
    hop_sec = float(hop_sec)
    half = 0.5 * window_sec

    centers = np.arange(half, max(half, duration_sec - half) + 1e-9, hop_sec)
    if centers.size == 0:
        return out

    lf = np.full(centers.shape, np.nan, dtype=float)
    hf = np.full(centers.shape, np.nan, dtype=float)
    lf_hf = np.full(centers.shape, np.nan, dtype=float)
    valid = np.zeros(centers.shape, dtype=bool)

    n = rr_mid.size
    left = 0
    right = 0

    for i, c in enumerate(centers):
        start = c - half
        end = c + half

        while left < n and rr_mid[left] < start:
            left += 1
        if right < left:
            right = left
        while right < n and rr_mid[right] < end:
            right += 1

        k = right - left
        if k < 4:
            continue
        if min_rr and k < int(min_rr):
            continue

        rr_win_ms = rr_ms[left:right]
        rr_mid_win = rr_mid[left:right]

        if rr_mid_win.size >= 2:
            if np.any(np.diff(rr_mid_win) > float(max_gap_sec)):
                continue
            span = float(rr_mid_win[-1] - rr_mid_win[0])
            if span < 0.8 * window_sec:
                continue
        else:
            continue

        lfi, hfi, r = _lf_hf_from_rr(rr_win_ms, rr_mid_win, fs_resample=float(fs_resample))
        if np.isfinite(lfi) and np.isfinite(hfi):
            lf[i] = lfi
            hf[i] = hfi
            lf_hf[i] = r
            valid[i] = True

    out["t_win_center_sec"] = centers
    out["lf_ms2"] = lf
    out["hf_ms2"] = hf
    out["lf_hf"] = lf_hf
    out["valid_mask"] = valid
    return out


# ---------------------------------------------------------------------
# Basic HRV metrics on RR intervals
# ---------------------------------------------------------------------

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


def _lf_hf_from_rr(
    rr_ms: np.ndarray,
    rr_mid_times_sec: np.ndarray,
    fs_resample: float = 4.0,
) -> Tuple[float, float, float]:
    """
    Compute LF, HF, LF/HF from RR intervals using Welch PSD on a resampled tachogram.
    """
    if rr_ms.size < 4 or rr_mid_times_sec.size < 4:
        return np.nan, np.nan, np.nan

    rr_sec = rr_ms / 1000.0
    rr_detr = rr_sec - np.mean(rr_sec)

    t = rr_mid_times_sec
    if t.size < 2 or t[-1] <= t[0]:
        return np.nan, np.nan, np.nan

    t_grid = np.arange(t[0], t[-1], 1.0 / fs_resample)
    if t_grid.size < 8:
        return np.nan, np.nan, np.nan

    rr_interp = np.interp(t_grid, t, rr_detr)

    nperseg = min(len(rr_interp), 256)
    if nperseg < 8:
        return np.nan, np.nan, np.nan

    f, pxx = welch(rr_interp, fs=fs_resample, nperseg=nperseg)

    lf_band = (f >= 0.04) & (f < 0.15)
    hf_band = (f >= 0.15) & (f < 0.40)

    if not np.any(lf_band) or not np.any(hf_band):
        return np.nan, np.nan, np.nan

    lf = float(np.trapz(pxx[lf_band], f[lf_band]))
    hf = float(np.trapz(pxx[hf_band], f[hf_band]))

    lf_ms2 = lf * 1e6
    hf_ms2 = hf * 1e6
    lf_hf = float(lf_ms2 / hf_ms2) if hf_ms2 > 0 else np.nan
    return lf_ms2, hf_ms2, lf_hf


# ---------------------------------------------------------------------
# Gap-robust helpers
# ---------------------------------------------------------------------

def _split_into_contiguous_runs(t: np.ndarray, max_gap_sec: float) -> List[np.ndarray]:
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


def _lf_hf_from_rr_segmented(
    rr_ms: np.ndarray,
    rr_mid_times_sec: np.ndarray,
    fs_resample: float = 4.0,
    *,
    max_gap_sec: float = 3.0,
    min_span_sec: float = 120.0,
    return_info: bool = False,
) -> Tuple[float, float, float] | Tuple[float, float, float, Dict[str, float]]:
    """
    Compute LF, HF, LF/HF from RR intervals WITHOUT interpolating across big gaps.
    """
    if rr_ms.size < 4 or rr_mid_times_sec.size < 4:
        if return_info:
            return np.nan, np.nan, np.nan, {
                "n_segments_total": 0.0,
                "n_segments_used": 0.0,
                "dur_used_sec": 0.0,
            }
        return np.nan, np.nan, np.nan

    runs = _split_into_contiguous_runs(rr_mid_times_sec, float(max_gap_sec))
    n_segments_total = int(len(runs))

    if not runs:
        if return_info:
            return np.nan, np.nan, np.nan, {
                "n_segments_total": float(n_segments_total),
                "n_segments_used": 0.0,
                "dur_used_sec": 0.0,
            }
        return np.nan, np.nan, np.nan

    lf_list: List[float] = []
    hf_list: List[float] = []
    dur_list: List[float] = []

    for r in runs:
        if r.size < 4:
            continue

        t = rr_mid_times_sec[r]
        span = float(t[-1] - t[0])
        if span < float(min_span_sec):
            continue

        lf, hf, _ = _lf_hf_from_rr(rr_ms[r], t, fs_resample=float(fs_resample))
        if np.isfinite(lf) and np.isfinite(hf):
            lf_list.append(float(lf))
            hf_list.append(float(hf))
            dur_list.append(span)

    n_segments_used = int(len(dur_list))
    dur_used_sec = float(np.sum(dur_list)) if n_segments_used > 0 else 0.0

    if n_segments_used == 0:
        if return_info:
            return np.nan, np.nan, np.nan, {
                "n_segments_total": float(n_segments_total),
                "n_segments_used": float(n_segments_used),
                "dur_used_sec": float(dur_used_sec),
            }
        return np.nan, np.nan, np.nan

    w = np.asarray(dur_list, dtype=float)
    w = w / float(np.sum(w))

    lf = float(np.sum(w * np.asarray(lf_list, dtype=float)))
    hf = float(np.sum(w * np.asarray(hf_list, dtype=float)))
    lf_hf = float(lf / hf) if hf > 0 else np.nan

    if return_info:
        return lf, hf, lf_hf, {
            "n_segments_total": float(n_segments_total),
            "n_segments_used": float(n_segments_used),
            "dur_used_sec": float(dur_used_sec),
        }

    return lf, hf, lf_hf


# ---------------------------------------------------------------------
# Time-varying RMSSD helper
# ---------------------------------------------------------------------

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

        k = right - left
        if k < min_intervals:
            continue

        rr_win_ms = rr_ms[left:right]
        rr_mid_win = rr_mid[left:right]

        if rr_mid_win.size >= 2:
            gaps = np.diff(rr_mid_win)
            if gaps.size > 0 and np.any(gaps > float(max_gap_sec)):
                continue

            span = float(rr_mid_win[-1] - rr_mid_win[0])
            if span < float(min_span_sec):
                continue

            if min_cov > 0.0 and span < (min_cov * float(window_sec)):
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


# ---------------------------------------------------------------------
# Windowed HRV metrics for TV plots
# ---------------------------------------------------------------------

def _calculate_hrv_windowed_series(
    t_grid: np.ndarray,
    rr_mid: np.ndarray,
    rr_ms: np.ndarray,
    window_sec: float,
    fs_resample: float,
    min_rr: int,
    *,
    min_span_sec: float = 120.0,
    max_gap_sec: float = 3.0,
) -> Dict[str, np.ndarray]:
    """
    Compute windowed HRV metrics on a time grid:
      rmssd_ms, sdnn_ms, lf, hf, lf_hf

    Includes debug counters for why spectral windows are rejected.
    """
    out: Dict[str, np.ndarray] = {
        "rmssd_ms": np.full_like(t_grid, np.nan, dtype=float),
        "sdnn_ms": np.full_like(t_grid, np.nan, dtype=float),
        "lf": np.full_like(t_grid, np.nan, dtype=float),
        "hf": np.full_like(t_grid, np.nan, dtype=float),
        "lf_hf": np.full_like(t_grid, np.nan, dtype=float),
    }

    if rr_mid.size == 0 or rr_ms.size == 0 or t_grid.size == 0:
        return out

    half = 0.5 * float(window_sec)

    n = rr_mid.size
    left = 0
    right = 0

    n_total = 0
    n_fail_min_rr = 0
    n_fail_span = 0
    n_fail_gap = 0
    n_fail_psd = 0
    n_ok = 0

    for i, t in enumerate(t_grid):
        n_total += 1
        start = t - half
        end = t + half

        while left < n and rr_mid[left] < start:
            left += 1
        if right < left:
            right = left
        while right < n and rr_mid[right] < end:
            right += 1

        k = right - left
        if k < min_rr:
            n_fail_min_rr += 1
            continue

        rr_win_ms = rr_ms[left:right]
        rr_mid_win = rr_mid[left:right]

        rmssd_i = _rmssd(rr_win_ms)
        sdnn_i = _sdnn(rr_win_ms)

        span = float(rr_mid_win[-1] - rr_mid_win[0]) if rr_mid_win.size >= 2 else 0.0
        if span < float(min_span_sec):
            n_fail_span += 1
            continue

        gaps = np.diff(rr_mid_win) if rr_mid_win.size >= 2 else np.array([])
        if gaps.size > 0 and np.any(gaps > float(max_gap_sec)):
            n_fail_gap += 1
            continue

        out["rmssd_ms"][i] = rmssd_i
        out["sdnn_ms"][i] = sdnn_i

        lf, hf, lf_hf = _lf_hf_from_rr(rr_win_ms, rr_mid_win, fs_resample=float(fs_resample))
        if not (np.isfinite(lf) and np.isfinite(hf) and np.isfinite(lf_hf)):
            n_fail_psd += 1
            continue

        out["lf"][i] = lf
        out["hf"][i] = hf
        out["lf_hf"][i] = lf_hf
        n_ok += 1

    print(
        "  TV spectral debug: "
        f"total={n_total}, ok={n_ok}, "
        f"fail_min_rr={n_fail_min_rr}, "
        f"fail_span={n_fail_span}, "
        f"fail_gap={n_fail_gap}, "
        f"fail_psd={n_fail_psd}"
    )

    return out


# ---------------------------------------------------------------------
# Main HRV computation (PAT-only)
# ---------------------------------------------------------------------

def compute_hrv_from_pat_signal(
    pat_signal: np.ndarray,
    fs: float,
    aux_df: Optional[pd.DataFrame] = None,
    target_fs: float = None,
    window_sec: float = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Dict[str, float]]]:
    """
    Compute HRV from PAT signal using the same RR extraction/cleaning as HR.

    Outputs:
      - t_hrv
      - rmssd_1hz_raw: after sleep masking, before event exclusion
      - rmssd_1hz_clean: after sleep masking + event exclusion
      - summary: global HRV metrics on clean RR
    """
    if target_fs is None:
        target_fs = float(getattr(config, "HRV_TARGET_FS_HZ"))
    if window_sec is None:
        window_sec = float(getattr(config, "HRV_WINDOW_SEC"))

    if fs <= 0:
        raise ValueError("Sampling frequency fs must be positive.")

    n_samples = len(pat_signal)
    if pat_signal.ndim != 1 or n_samples == 0:
        raise ValueError("PAT signal must be a non-empty 1D array.")

    max_gap_sec = float(getattr(config, "HRV_MAX_RR_GAP_SEC"))
    rmssd_min_span_sec = float(getattr(config, "HRV_RMSSD_MIN_SPAN_SEC"))
    fs_resample = float(getattr(config, "HRV_TACHO_RESAMPLE_HZ"))
    min_freq_span_sec = float(getattr(config, "HRV_MIN_FREQ_DOMAIN_SEC"))

    rr_sec_physio_clean, rr_mid_physio_clean, duration_sec = hr_metrics.extract_clean_rr_from_pat(
        pat_signal, fs
    )

    t_hrv = np.arange(0, duration_sec, 1.0 / float(target_fs))

    if rr_sec_physio_clean.size < 1:
        nan_array = np.full_like(t_hrv, fill_value=np.nan, dtype=float)
        return t_hrv, nan_array, nan_array, None

    rr_ms_physio_clean = rr_sec_physio_clean * 1000.0

    bundle = masking.build_rr_mask_bundle(rr_mid_physio_clean, aux_df)
    rr_mid_sleep = rr_mid_physio_clean[bundle.sleep_keep]
    rr_ms_sleep = rr_ms_physio_clean[bundle.sleep_keep]

    rmssd_1hz_raw, _rmssd_windows_list_raw = _calculate_rmssd_series(
        t_hrv,
        rr_mid_sleep,
        rr_ms_sleep,
        float(window_sec),
        max_gap_sec=max_gap_sec,
        min_span_sec=rmssd_min_span_sec,
    )

    if rr_ms_sleep.size < 1:
        nan_array = np.full_like(t_hrv, fill_value=np.nan, dtype=float)
        return t_hrv, rmssd_1hz_raw, nan_array, None

    rr_mid_for_calc = rr_mid_physio_clean[bundle.combined_keep]
    rr_ms_for_calc = rr_ms_physio_clean[bundle.combined_keep]

    if rr_ms_for_calc.size < 1:
        nan_array = np.full_like(t_hrv, fill_value=np.nan, dtype=float)
        return t_hrv, rmssd_1hz_raw, nan_array, None

    rmssd_1hz_clean, rmssd_windows_list_clean = _calculate_rmssd_series(
        t_hrv,
        rr_mid_for_calc,
        rr_ms_for_calc,
        float(window_sec),
        max_gap_sec=max_gap_sec,
        min_span_sec=rmssd_min_span_sec,
    )

    sdnn_ms = _sdnn(rr_ms_for_calc)

    lf, hf, lf_hf, lf_info = _lf_hf_from_rr_segmented(
        rr_ms_for_calc,
        rr_mid_for_calc,
        fs_resample=fs_resample,
        max_gap_sec=max_gap_sec,
        min_span_sec=min_freq_span_sec,
        return_info=True,
    )

    if len(rmssd_windows_list_clean) > 0:
        rmssd_arr = np.asarray(rmssd_windows_list_clean, dtype=float)
        summary: Dict[str, float] = {
            "rmssd_mean": float(np.nanmean(rmssd_arr)),
            "rmssd_median": float(np.nanmedian(rmssd_arr)),
            "sdnn": float(sdnn_ms),
            "lf": float(lf),
            "hf": float(hf),
            "lf_hf": float(lf_hf),
            "lf_n_segments_used": int(lf_info["n_segments_used"]),
        }
    else:
        summary = {
            "rmssd_mean": float(_rmssd(rr_ms_for_calc)),
            "rmssd_median": float(np.nan),
            "sdnn": float(sdnn_ms),
            "lf": float(lf),
            "hf": float(hf),
            "lf_hf": float(lf_hf),
            "lf_n_segments_used": int(lf_info["n_segments_used"]),
        }

    return t_hrv, rmssd_1hz_raw, rmssd_1hz_clean, summary


# ---------------------------------------------------------------------
# Post-hoc sleep-policy summaries from base RR
# ---------------------------------------------------------------------

def _subset_rr_by_sleep_and_events(
    rr_mid_times_sec: np.ndarray,
    rr_ms: np.ndarray,
    aux_df: Optional[pd.DataFrame],
    *,
    include_set: Optional[set[int]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rr_mid_times_sec = np.asarray(rr_mid_times_sec, dtype=float)
    rr_ms = np.asarray(rr_ms, dtype=float)

    policy = masking.policy_from_config(include_stages=include_set, force_sleep=(include_set is not None))
    bundle = masking.build_rr_mask_bundle(rr_mid_times_sec, aux_df, policy=policy)

    rr_mid_sleep = rr_mid_times_sec[bundle.sleep_keep]
    rr_ms_sleep = rr_ms[bundle.sleep_keep]

    rr_mid_clean = rr_mid_times_sec[bundle.combined_keep]
    rr_ms_clean = rr_ms[bundle.combined_keep]
    return rr_mid_sleep, rr_ms_sleep, rr_mid_clean, rr_ms_clean


def summarize_hrv_from_rr(
    rr_mid_times_sec: np.ndarray,
    rr_ms: np.ndarray,
    duration_sec: float,
    aux_df: Optional[pd.DataFrame],
    *,
    include_set: Optional[set[int]] = None,
    target_fs: Optional[float] = None,
    window_sec: Optional[float] = None,
) -> Dict[str, float]:
    if target_fs is None:
        target_fs = float(getattr(config, "HRV_TARGET_FS_HZ"))
    if window_sec is None:
        window_sec = float(getattr(config, "HRV_WINDOW_SEC"))

    max_gap_sec = float(getattr(config, "HRV_MAX_RR_GAP_SEC"))
    rmssd_min_span_sec = float(getattr(config, "HRV_RMSSD_MIN_SPAN_SEC"))
    fs_resample_global = float(getattr(config, "HRV_TACHO_RESAMPLE_HZ"))
    min_freq_span_sec = float(getattr(config, "HRV_MIN_FREQ_DOMAIN_SEC"))
    fixed_win_sec = float(getattr(config, "HRV_LFHF_FIXED_WINDOW_SEC", 300.0))
    fixed_hop_sec = float(getattr(config, "HRV_LFHF_FIXED_HOP_SEC", fixed_win_sec))
    fixed_min_rr = int(getattr(config, "HRV_LFHF_FIXED_MIN_RR", 0))

    rr_mid_sleep, rr_ms_sleep, rr_mid_clean, rr_ms_clean = _subset_rr_by_sleep_and_events(
        rr_mid_times_sec,
        rr_ms,
        aux_df,
        include_set=include_set,
    )

    t_hrv = np.arange(0, float(duration_sec), 1.0 / float(target_fs))
    rmssd_raw, _ = _calculate_rmssd_series(
        t_hrv,
        rr_mid_sleep,
        rr_ms_sleep,
        float(window_sec),
        max_gap_sec=max_gap_sec,
        min_span_sec=rmssd_min_span_sec,
    )
    rmssd_clean, rmssd_windows_list_clean = _calculate_rmssd_series(
        t_hrv,
        rr_mid_clean,
        rr_ms_clean,
        float(window_sec),
        max_gap_sec=max_gap_sec,
        min_span_sec=rmssd_min_span_sec,
    )

    summary: Dict[str, float] = {
        "rr_n_sleep": int(rr_ms_sleep.size),
        "rr_n_clean": int(rr_ms_clean.size),
        "rmssd_nan_pct_raw": float(100.0 * np.mean(~np.isfinite(rmssd_raw))) if rmssd_raw.size else np.nan,
        "rmssd_nan_pct_clean": float(100.0 * np.mean(~np.isfinite(rmssd_clean))) if rmssd_clean.size else np.nan,
    }

    if rr_ms_clean.size < 1:
        summary.update(
            {
                "rmssd_mean": np.nan,
                "rmssd_median": np.nan,
                "sdnn": np.nan,
                "lf": np.nan,
                "hf": np.nan,
                "lf_hf": np.nan,
                "lf_n_segments_used": 0,
                "lf_hf_fixed_median": np.nan,
                "lf_hf_fixed_mean": np.nan,
                "lf_hf_fixed_n_windows_valid": 0,
                "lf_hf_fixed_n_windows_total": 0,
                "lf_hf_fixed_window_sec": float(fixed_win_sec),
                "lf_hf_fixed_hop_sec": float(fixed_hop_sec),
            }
        )
        return summary

    sdnn_ms = _sdnn(rr_ms_clean)
    lf, hf, lf_hf, lf_info = _lf_hf_from_rr_segmented(
        rr_ms_clean,
        rr_mid_clean,
        fs_resample=fs_resample_global,
        max_gap_sec=max_gap_sec,
        min_span_sec=min_freq_span_sec,
        return_info=True,
    )

    if len(rmssd_windows_list_clean) > 0:
        rmssd_arr = np.asarray(rmssd_windows_list_clean, dtype=float)
        summary.update(
            {
                "rmssd_mean": float(np.nanmean(rmssd_arr)),
                "rmssd_median": float(np.nanmedian(rmssd_arr)),
            }
        )
    else:
        summary.update(
            {
                "rmssd_mean": float(_rmssd(rr_ms_clean)),
                "rmssd_median": np.nan,
            }
        )

    summary.update(
        {
            "sdnn": float(sdnn_ms),
            "lf": float(lf),
            "hf": float(hf),
            "lf_hf": float(lf_hf),
            "lf_n_segments_used": int(lf_info["n_segments_used"]),
        }
    )

    lfhf_fixed = _calculate_lfhf_fixed_windows(
        rr_mid=rr_mid_clean,
        rr_ms=rr_ms_clean,
        duration_sec=float(duration_sec),
        window_sec=fixed_win_sec,
        hop_sec=fixed_hop_sec,
        fs_resample=fs_resample_global,
        max_gap_sec=max_gap_sec,
        min_rr=fixed_min_rr,
    )
    valid = lfhf_fixed.get("valid_mask", np.array([], dtype=bool))
    n_total = int(valid.size)
    n_valid = int(np.sum(valid)) if n_total > 0 else 0
    if n_valid > 0:
        lfhf_vals = np.asarray(lfhf_fixed["lf_hf"], dtype=float)[valid]
        summary["lf_hf_fixed_median"] = float(np.nanmedian(lfhf_vals))
        summary["lf_hf_fixed_mean"] = float(np.nanmean(lfhf_vals))
    else:
        summary["lf_hf_fixed_median"] = np.nan
        summary["lf_hf_fixed_mean"] = np.nan

    summary["lf_hf_fixed_n_windows_valid"] = n_valid
    summary["lf_hf_fixed_n_windows_total"] = n_total
    summary["lf_hf_fixed_window_sec"] = float(fixed_win_sec)
    summary["lf_hf_fixed_hop_sec"] = float(fixed_hop_sec)
    return summary


# ---------------------------------------------------------------------
# HRV computation with TV metrics (PAT-only)
# ---------------------------------------------------------------------

def compute_hrv_from_pat_signal_with_tv_metrics(
    pat_signal: np.ndarray,
    fs: float,
    aux_df: Optional[pd.DataFrame] = None,
    target_fs: float = None,
    window_sec: float = None,
    tv_window_sec: float = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[Dict[str, float]],
    Optional[Dict[str, np.ndarray]],
]:
    """
    Same as compute_hrv_from_pat_signal, but also returns time-varying HRV metrics.
    """
    if target_fs is None:
        target_fs = float(getattr(config, "HRV_TARGET_FS_HZ"))
    if window_sec is None:
        window_sec = float(getattr(config, "HRV_WINDOW_SEC"))
    if tv_window_sec is None:
        tv_window_sec = float(getattr(config, "HRV_TV_WINDOW_SEC"))

    if fs <= 0:
        raise ValueError("Sampling frequency fs must be positive.")
    n_samples = len(pat_signal)
    if pat_signal.ndim != 1 or n_samples == 0:
        raise ValueError("PAT signal must be a non-empty 1D array.")

    max_gap_sec = float(getattr(config, "HRV_MAX_RR_GAP_SEC"))
    rmssd_min_span_sec = float(getattr(config, "HRV_RMSSD_MIN_SPAN_SEC"))

    fs_resample_global = float(getattr(config, "HRV_TACHO_RESAMPLE_HZ"))
    min_freq_span_sec = float(getattr(config, "HRV_MIN_FREQ_DOMAIN_SEC"))

    fs_resample_tv = float(getattr(config, "HRV_TV_TACHO_RESAMPLE_HZ"))
    min_rr_tv = int(getattr(config, "HRV_TV_MIN_RR_PER_WINDOW"))

    # TV-specific thresholds
    min_freq_span_sec_tv = float(
        getattr(config, "HRV_TV_MIN_FREQ_DOMAIN_SEC", getattr(config, "HRV_MIN_FREQ_DOMAIN_SEC"))
    )
    max_gap_sec_tv = float(
        getattr(config, "HRV_TV_MAX_TACHO_GAP_SEC", getattr(config, "HRV_MAX_RR_GAP_SEC"))
    )

    rr_sec_physio_clean, rr_mid_physio_clean, duration_sec = hr_metrics.extract_clean_rr_from_pat(
        pat_signal, fs
    )
    t_hrv = np.arange(0, duration_sec, 1.0 / float(target_fs))

    if rr_sec_physio_clean.size < 1:
        nan_array = np.full_like(t_hrv, fill_value=np.nan, dtype=float)
        return t_hrv, nan_array, nan_array, None, None

    rr_ms_physio_clean = rr_sec_physio_clean * 1000.0

    bundle = masking.build_rr_mask_bundle(rr_mid_physio_clean, aux_df)
    rr_mid_sleep = rr_mid_physio_clean[bundle.sleep_keep]
    rr_ms_sleep = rr_ms_physio_clean[bundle.sleep_keep]

    if rr_ms_sleep.size < 1:
        nan_array = np.full_like(t_hrv, fill_value=np.nan, dtype=float)
        return t_hrv, nan_array, nan_array, None, None

    # RAW RMSSD = sleep-masked, before event exclusion
    rmssd_1hz_raw, _rmssd_windows_list_raw = _calculate_rmssd_series(
        t_hrv,
        rr_mid_sleep,
        rr_ms_sleep,
        float(window_sec),
        max_gap_sec=max_gap_sec,
        min_span_sec=rmssd_min_span_sec,
    )

    # CLEAN RR = sleep-masked + event-excluded
    rr_mid_for_calc = rr_mid_physio_clean[bundle.combined_keep]
    rr_ms_for_calc = rr_ms_physio_clean[bundle.combined_keep]

    if rr_ms_for_calc.size < 1:
        nan_array = np.full_like(t_hrv, fill_value=np.nan, dtype=float)
        return t_hrv, rmssd_1hz_raw, nan_array, None, None

    rmssd_1hz_clean, rmssd_windows_list_clean = _calculate_rmssd_series(
        t_hrv,
        rr_mid_for_calc,
        rr_ms_for_calc,
        float(window_sec),
        max_gap_sec=max_gap_sec,
        min_span_sec=rmssd_min_span_sec,
    )

    sdnn_ms = _sdnn(rr_ms_for_calc)

    lf, hf, lf_hf, lf_info = _lf_hf_from_rr_segmented(
        rr_ms_for_calc,
        rr_mid_for_calc,
        fs_resample=fs_resample_global,
        max_gap_sec=max_gap_sec,
        min_span_sec=min_freq_span_sec,
        return_info=True,
    )

    if len(rmssd_windows_list_clean) > 0:
        rmssd_arr = np.asarray(rmssd_windows_list_clean, dtype=float)
        summary: Dict[str, float] = {
            "rmssd_mean": float(np.nanmean(rmssd_arr)),
            "rmssd_median": float(np.nanmedian(rmssd_arr)),
            "sdnn": float(sdnn_ms),
            "lf": float(lf),
            "hf": float(hf),
            "lf_hf": float(lf_hf),
            "lf_n_segments_used": int(lf_info["n_segments_used"]),
        }
    else:
        summary = {
            "rmssd_mean": float(_rmssd(rr_ms_for_calc)),
            "rmssd_median": float(np.nan),
            "sdnn": float(sdnn_ms),
            "lf": float(lf),
            "hf": float(hf),
            "lf_hf": float(lf_hf),
            "lf_n_segments_used": int(lf_info["n_segments_used"]),
        }

    # Fixed-window LF/HF
    fixed_win_sec = float(getattr(config, "HRV_LFHF_FIXED_WINDOW_SEC", 300.0))
    fixed_hop_sec = float(getattr(config, "HRV_LFHF_FIXED_HOP_SEC", fixed_win_sec))
    fs_resample_fixed = float(getattr(config, "HRV_TACHO_RESAMPLE_HZ", 4.0))
    fixed_min_rr = int(getattr(config, "HRV_LFHF_FIXED_MIN_RR", 0))

    lfhf_fixed = _calculate_lfhf_fixed_windows(
        rr_mid=rr_mid_for_calc,
        rr_ms=rr_ms_for_calc,
        duration_sec=duration_sec,
        window_sec=fixed_win_sec,
        hop_sec=fixed_hop_sec,
        fs_resample=fs_resample_fixed,
        max_gap_sec=max_gap_sec,
        min_rr=fixed_min_rr,
    )

    valid = lfhf_fixed.get("valid_mask", np.array([], dtype=bool))
    n_total = int(valid.size)
    n_valid = int(np.sum(valid)) if n_total > 0 else 0

    if n_valid > 0:
        lfhf_vals = np.asarray(lfhf_fixed["lf_hf"], dtype=float)[valid]
        summary["lf_hf_fixed_median"] = float(np.nanmedian(lfhf_vals))
        summary["lf_hf_fixed_mean"] = float(np.nanmean(lfhf_vals))
    else:
        summary["lf_hf_fixed_median"] = float(np.nan)
        summary["lf_hf_fixed_mean"] = float(np.nan)

    summary["lf_hf_fixed_n_windows_valid"] = n_valid
    summary["lf_hf_fixed_n_windows_total"] = n_total
    summary["lf_hf_fixed_window_sec"] = float(fixed_win_sec)
    summary["lf_hf_fixed_hop_sec"] = float(fixed_hop_sec)

    if rr_mid_for_calc.size == 0:
        tv = {
            "rmssd_ms": np.full_like(t_hrv, np.nan, dtype=float),
            "sdnn_ms": np.full_like(t_hrv, np.nan, dtype=float),
            "lf": np.full_like(t_hrv, np.nan, dtype=float),
            "hf": np.full_like(t_hrv, np.nan, dtype=float),
            "lf_hf": np.full_like(t_hrv, np.nan, dtype=float),
            "tv_window_sec": np.array([float(tv_window_sec)], dtype=float),
        }
        return t_hrv, rmssd_1hz_raw, rmssd_1hz_clean, summary, tv

    tv = _calculate_hrv_windowed_series(
        t_grid=t_hrv,
        rr_mid=rr_mid_for_calc,
        rr_ms=rr_ms_for_calc,
        window_sec=float(tv_window_sec),
        fs_resample=float(fs_resample_tv),
        min_rr=int(min_rr_tv),
        min_span_sec=float(min_freq_span_sec_tv),
        max_gap_sec=float(max_gap_sec_tv),
    )
    tv["tv_window_sec"] = np.array([float(tv_window_sec)], dtype=float)

    return t_hrv, rmssd_1hz_raw, rmssd_1hz_clean, summary, tv


# ---------------------------------------------------------------------
# CSV saving
# ---------------------------------------------------------------------

def save_hrv_series_to_csv(
    edf_path: Path,
    t_hrv: np.ndarray,
    rmssd_1hz: np.ndarray,
) -> Optional[Path]:
    """
    Save HRV 1 Hz series (RMSSD sliding) to CSV: time_sec, rmssd_ms
    """
    if t_hrv.size == 0 or rmssd_1hz.size == 0:
        return None

    hrv_sub = getattr(config, "HRV_OUTPUT_SUBFOLDER", getattr(config, "HR_OUTPUT_SUBFOLDER", "HRV"))
    hrv_folder = paths.get_output_folder(hrv_sub)

    edf_base = edf_path.stem
    out_csv = hrv_folder / f"{edf_base}__HRV_RMSSD_1Hz_Clean.csv"

    data = np.column_stack([t_hrv, rmssd_1hz])
    header = "time_sec,rmssd_ms"
    np.savetxt(out_csv, data, delimiter=",", header=header, comments="")
    return out_csv
