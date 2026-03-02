# pat_toolbox/metrics/hr_delta.py

from __future__ import annotations

import numpy as np
from typing import Optional


def compute_delta_hr(
    hr_bpm: np.ndarray,
    *,
    lag_sec: float,
    fs: float = 1.0,
    pre_smooth_sec: float = 0.0,
    use_abs: bool = False,
) -> np.ndarray:
    """
    ΔHR(t) = HR(t) - HR(t - lag_sec) computed on the same regular grid.

    - NaN-safe: if either sample in the pair is NaN -> NaN.
    - Optional NaN-aware pre-smoothing (moving average).

    Args:
        hr_bpm: HR samples on a regular grid (typically 1 Hz).
        lag_sec: lag in seconds (e.g. 10 for ΔHR over 10 seconds).
        fs: sampling rate of hr_bpm (Hz). Default 1.0.
        pre_smooth_sec: optional smoothing window (seconds) applied before delta.
        use_abs: if True, return |ΔHR|.

    Returns:
        np.ndarray same shape as hr_bpm
    """
    if hr_bpm is None:
        return np.array([], dtype=float)

    hr = np.asarray(hr_bpm, dtype=float)
    if hr.size == 0:
        return hr.copy()

    # Optional pre-smoothing (NaN-aware moving average)
    hr_use = hr
    if pre_smooth_sec and pre_smooth_sec > 0:
        win = int(round(float(pre_smooth_sec) * float(fs)))
        if win > 1:
            kernel = np.ones(win, dtype=float)
            valid = np.isfinite(hr).astype(float)
            hr0 = np.where(np.isfinite(hr), hr, 0.0)
            num = np.convolve(hr0, kernel, mode="same")
            den = np.convolve(valid, kernel, mode="same")
            with np.errstate(divide="ignore", invalid="ignore"):
                hr_use = num / den
            hr_use[den <= 0] = np.nan

    lag_samples = int(round(float(lag_sec) * float(fs)))
    lag_samples = max(1, lag_samples)

    d = np.full_like(hr_use, np.nan, dtype=float)
    a = hr_use[lag_samples:]
    b = hr_use[:-lag_samples]

    ok = np.isfinite(a) & np.isfinite(b)
    dd = np.full_like(a, np.nan, dtype=float)
    dd[ok] = a[ok] - b[ok]

    if use_abs:
        dd = np.abs(dd)

    d[lag_samples:] = dd
    return d
