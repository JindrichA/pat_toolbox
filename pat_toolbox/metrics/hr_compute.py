from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .. import config
from ..core.rr_cleaning import detect_pat_peaks, extract_clean_rr_from_pat as core_extract_clean_rr_from_pat
from ..core.windows import interp_with_gaps


def _detect_pat_peaks(
    pat_signal: np.ndarray,
    fs: float,
) -> Tuple[np.ndarray, np.ndarray]:
    return detect_pat_peaks(pat_signal, fs)


def extract_clean_rr_from_pat(
    pat_signal: np.ndarray,
    fs: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    return core_extract_clean_rr_from_pat(pat_signal, fs)


def _hampel_filter_1d(
    x: np.ndarray,
    window_size: int,
    n_sigmas: float = 3.0,
) -> np.ndarray:
    if x.ndim != 1 or x.size == 0:
        return x
    if window_size < 1:
        return x
    if window_size % 2 == 0:
        window_size += 1

    k = window_size // 2
    n = x.size
    y = x.copy()

    for i in range(n):
        i0 = max(0, i - k)
        i1 = min(n, i + k + 1)
        window = x[i0:i1]
        med = np.median(window)
        mad = np.median(np.abs(window - med))
        if mad == 0:
            continue
        sigma = 1.4826 * mad
        if np.abs(x[i] - med) > n_sigmas * sigma:
            y[i] = med

    return y


def _interp_with_gaps(
    t_grid: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    max_gap_sec: float,
) -> np.ndarray:
    return interp_with_gaps(t_grid, t, y, max_gap_sec)


def compute_hr_from_pat_signal(
    pat_signal: np.ndarray,
    fs: float,
    target_fs: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute HR (1 Hz by default) from PAT using shared RR extraction.
    """
    if target_fs is None:
        target_fs = float(getattr(config, "HR_TARGET_FS_HZ", 1.0))

    rr_sec_clean, rr_mid_clean, duration_sec = extract_clean_rr_from_pat(pat_signal, fs)

    t_hr = np.arange(0, duration_sec, 1.0 / float(target_fs))
    if rr_sec_clean.size < 1:
        hr_1hz = np.full_like(t_hr, fill_value=np.nan, dtype=float)
        return t_hr, hr_1hz

    inst_hr = 60.0 / rr_sec_clean

    max_gap_sec = float(getattr(config, "HR_MAX_RR_GAP_SEC", 2.5))
    hr_grid = _interp_with_gaps(t_hr, rr_mid_clean, inst_hr, max_gap_sec=max_gap_sec)

    smooth_sec = float(getattr(config, "HR_SMOOTHING_WINDOW_SEC", 0.0))
    smooth_samples = int(round(smooth_sec * target_fs))

    if smooth_samples > 1:
        kernel = np.ones(smooth_samples, dtype=float)
        x = hr_grid.astype(float)

        valid = np.isfinite(x).astype(float)
        x0 = np.where(np.isfinite(x), x, 0.0)

        num = np.convolve(x0, kernel, mode="same")
        den = np.convolve(valid, kernel, mode="same")
        with np.errstate(divide="ignore", invalid="ignore"):
            hr_smooth = num / den
        hr_smooth[den <= 0] = np.nan
    else:
        hr_smooth = hr_grid

    hr_smooth = np.where(
        np.isfinite(hr_smooth),
        np.clip(hr_smooth, config.HR_MIN_BPM, config.HR_MAX_BPM),
        np.nan,
    )

    hampel_win_sec = float(getattr(config, "HR_HAMPEL_WINDOW_SEC", 10.0))
    hampel_sigmas = float(getattr(config, "HR_HAMPEL_SIGMA", 3.0))
    hampel_win_samples = int(round(hampel_win_sec * target_fs))

    hr_despiked = hr_smooth
    nan_frac = float(np.mean(~np.isfinite(hr_smooth))) if hr_smooth.size > 0 else 1.0
    if hampel_win_samples > 1 and nan_frac < 0.05:
        hr_despiked = _hampel_filter_1d(
            hr_smooth,
            window_size=hampel_win_samples,
            n_sigmas=hampel_sigmas,
        )

    hr_despiked = np.where(
        np.isfinite(hr_despiked),
        np.clip(hr_despiked, config.HR_MIN_BPM, config.HR_MAX_BPM),
        np.nan,
    )

    max_delta_per_sec = float(getattr(config, "HR_MAX_DELTA_BPM_PER_SEC", 0.0))
    if max_delta_per_sec > 0:
        hr_limited = hr_despiked.copy()
        max_step = max_delta_per_sec / float(target_fs)
        for i in range(1, len(hr_limited)):
            if not np.isfinite(hr_limited[i]) or not np.isfinite(hr_limited[i - 1]):
                continue
            delta = hr_limited[i] - hr_limited[i - 1]
            if np.abs(delta) > max_step:
                hr_limited[i] = hr_limited[i - 1] + np.sign(delta) * max_step
    else:
        hr_limited = hr_despiked

    hr_limited = np.where(
        np.isfinite(hr_limited),
        np.clip(hr_limited, config.HR_MIN_BPM, config.HR_MAX_BPM),
        np.nan,
    )

    return t_hr, hr_limited
