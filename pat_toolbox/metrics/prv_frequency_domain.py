from __future__ import annotations

from typing import Dict, List, Literal, Tuple, overload

import numpy as np
from ..core.windows import split_into_contiguous_runs
from .spectral_utils import tachogram_psd_from_rr


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

    f, pxx = tachogram_psd_from_rr(rr_ms, rr_mid_times_sec, fs_resample=fs_resample)
    if f.size == 0 or pxx.size == 0:
        return np.nan, np.nan, np.nan

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


@overload
def _lf_hf_from_rr_segmented(
    rr_ms: np.ndarray,
    rr_mid_times_sec: np.ndarray,
    fs_resample: float = 4.0,
    *,
    max_gap_sec: float = 3.0,
    min_span_sec: float = 120.0,
    return_info: Literal[False] = False,
) -> Tuple[float, float, float]: ...


@overload
def _lf_hf_from_rr_segmented(
    rr_ms: np.ndarray,
    rr_mid_times_sec: np.ndarray,
    fs_resample: float = 4.0,
    *,
    max_gap_sec: float = 3.0,
    min_span_sec: float = 120.0,
    return_info: Literal[True],
) -> Tuple[float, float, float, Dict[str, float]]: ...


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

    runs = split_into_contiguous_runs(rr_mid_times_sec, float(max_gap_sec))
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
