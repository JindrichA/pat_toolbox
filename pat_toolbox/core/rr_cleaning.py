from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import find_peaks

from .. import config, filters


def detect_pat_peaks(
    pat_signal: np.ndarray,
    fs: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter PAT and detect peaks.

    Returns:
        pat_filt: filtered PAT signal (same length as input)
        peak_indices: np.ndarray of integer indices of detected peaks
    """
    if fs <= 0:
        raise ValueError("Sampling frequency fs must be positive.")

    n_samples = len(pat_signal)
    if pat_signal.ndim != 1 or n_samples == 0:
        raise ValueError("PAT signal must be a non-empty 1D array.")

    pat_filt = filters.bandpass_filter(pat_signal, fs=fs)

    min_rr_samples = int(config.HR_MIN_RR_SEC * fs)
    if min_rr_samples < 1:
        min_rr_samples = 1

    std_sig = np.std(pat_filt)
    prom = None
    if std_sig > 0:
        prom = config.HR_PEAK_PROMINENCE_FACTOR * std_sig

    if prom is not None and prom > 0:
        peaks, _props = find_peaks(
            pat_filt,
            distance=min_rr_samples,
            prominence=prom,
        )
    else:
        peaks, _props = find_peaks(
            pat_filt,
            distance=min_rr_samples,
        )

    if len(peaks) >= 2:
        peak_times_sec = peaks / fs
        rr_intervals_sec = np.diff(peak_times_sec)
        valid_rr_mask = (
            (rr_intervals_sec >= config.HR_MIN_RR_SEC)
            & (rr_intervals_sec <= config.HR_MAX_RR_SEC)
        )
        if np.any(valid_rr_mask):
            valid_peak_mask = np.zeros_like(peak_times_sec, dtype=bool)
            valid_peak_mask[:-1] |= valid_rr_mask
            valid_peak_mask[1:] |= valid_rr_mask
            peaks = peaks[valid_peak_mask]

    return pat_filt, peaks


def extract_clean_rr_from_pat(
    pat_signal: np.ndarray,
    fs: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Shared helper for HR + HRV.

    Steps:
      1) Detect peaks on filtered PAT (detect_pat_peaks)
      2) Compute RR intervals and mid-times
      3) Apply physiologic RR limits
      4) Apply median-based RR outlier rejection
      5) Reject gaps, abrupt jumps, and short/long alternans pairs
      6) Keep only contiguous "good" runs

    Returns:
        rr_sec_clean: np.ndarray of RR intervals [s]
        rr_mid_clean: np.ndarray of RR mid-times [s]
        duration_sec: float, total signal duration [s]
    """
    if fs <= 0:
        raise ValueError("Sampling frequency fs must be positive.")

    n_samples = len(pat_signal)
    if pat_signal.ndim != 1 or n_samples == 0:
        raise ValueError("PAT signal must be a non-empty 1D array.")

    duration_sec = n_samples / fs

    _pat_filt, peaks = detect_pat_peaks(pat_signal, fs)
    if len(peaks) < 2:
        return np.array([]), np.array([]), duration_sec

    peak_times_sec = peaks / fs

    rr_sec = np.diff(peak_times_sec)
    rr_mid_times = 0.5 * (peak_times_sec[1:] + peak_times_sec[:-1])

    valid_rr = (rr_sec >= config.HR_MIN_RR_SEC) & (rr_sec <= config.HR_MAX_RR_SEC)
    rr_sec_valid = rr_sec[valid_rr]
    rr_mid_valid = rr_mid_times[valid_rr]

    if rr_sec_valid.size < 1:
        return np.array([]), np.array([]), duration_sec

    kernel_len = int(getattr(config, "HR_RR_MEDFILT_KERNEL", 5))
    if kernel_len < 1:
        kernel_len = 1
    if kernel_len % 2 == 0:
        kernel_len += 1

    pad = kernel_len // 2
    rr_padded = np.pad(rr_sec_valid, pad_width=pad, mode="edge")
    rr_med = np.empty_like(rr_sec_valid)
    for i in range(rr_sec_valid.size):
        rr_med[i] = np.median(rr_padded[i : i + kernel_len])

    rel_thr = float(getattr(config, "HR_RR_OUTLIER_REL_THR", 0.25))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_dev = np.abs(rr_sec_valid - rr_med) / rr_med
    rel_dev[~np.isfinite(rel_dev)] = 0.0

    good = rel_dev <= rel_thr

    gap_factor = float(getattr(config, "HR_RR_GAP_FACTOR", 2.2))
    gap_ok = rr_sec_valid <= (gap_factor * rr_med)
    good &= gap_ok

    jump_thr = float(getattr(config, "HR_RR_JUMP_REL_THR", 0.5))
    if rr_sec_valid.size >= 2:
        rr_prev = rr_sec_valid[:-1]
        rr_next = rr_sec_valid[1:]
        denom = np.maximum(0.30, 0.5 * (rr_prev + rr_next))
        rel_jump = np.abs(rr_next - rr_prev) / denom
        jump_bad = np.zeros_like(rr_sec_valid, dtype=bool)
        jump_bad[1:] = rel_jump > jump_thr
        good &= ~jump_bad

    alt_short_rel = float(getattr(config, "HR_RR_ALT_SHORT_REL", 0.25))
    alt_long_rel = float(getattr(config, "HR_RR_ALT_LONG_REL", 0.35))

    if rr_sec_valid.size >= 2:
        short = rr_sec_valid < (1.0 - alt_short_rel) * rr_med
        long = rr_sec_valid > (1.0 + alt_long_rel) * rr_med

        pair1 = short[:-1] & long[1:]
        pair2 = long[:-1] & short[1:]

        alt_bad = np.zeros_like(rr_sec_valid, dtype=bool)
        alt_bad[:-1] |= (pair1 | pair2)
        alt_bad[1:] |= (pair1 | pair2)
        good &= ~alt_bad

    rr_sec_good = rr_sec_valid[good]
    rr_mid_good = rr_mid_valid[good]

    if rr_sec_good.size < 1:
        return np.array([]), np.array([]), duration_sec

    min_keep_run = int(getattr(config, "HR_RR_MIN_GOOD_RUN", 3))
    if min_keep_run > 1:
        idx = np.flatnonzero(good)
        if idx.size == 0:
            return np.array([]), np.array([]), duration_sec

        splits = np.where(np.diff(idx) > 1)[0] + 1
        runs = np.split(idx, splits)

        keep = np.zeros_like(good, dtype=bool)
        for r in runs:
            if r.size >= min_keep_run:
                keep[r] = True

        rr_sec_good = rr_sec_valid[keep]
        rr_mid_good = rr_mid_valid[keep]

    if rr_sec_good.size < 1:
        return np.array([]), np.array([]), duration_sec

    return rr_sec_good, rr_mid_good, duration_sec
