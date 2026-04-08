from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.signal import welch

from .. import config


def tachogram_psd_from_rr(
    rr_ms: np.ndarray,
    rr_mid_times_sec: np.ndarray,
    *,
    fs_resample: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Welch PSD on a resampled RR tachogram.
    Returns frequency and linear PSD in sec^2/Hz.
    """
    if rr_ms.size < 4 or rr_mid_times_sec.size < 4:
        return np.array([]), np.array([])

    t = rr_mid_times_sec
    if t.size < 2 or t[-1] <= t[0]:
        return np.array([]), np.array([])

    rr_sec = rr_ms / 1000.0
    rr_detr = rr_sec - np.mean(rr_sec)

    t_grid = np.arange(t[0], t[-1], 1.0 / float(fs_resample))
    if t_grid.size < 8:
        return np.array([]), np.array([])

    rr_interp = np.interp(t_grid, t, rr_detr)

    nperseg = min(len(rr_interp), 256)
    if nperseg < 8:
        return np.array([]), np.array([])

    nfft = int(getattr(config, "PSD_TACHO_NFFT", 2048))
    nfft = max(nfft, int(nperseg))

    f, pxx = welch(
        rr_interp,
        fs=float(fs_resample),
        nperseg=int(nperseg),
        nfft=int(nfft),
    )
    return f, pxx


def integrate_band_power(
    f_arr: np.ndarray,
    p_arr_lin: np.ndarray,
    band: tuple[float, float],
) -> float:
    low, high = band
    mask = (f_arr >= low) & (f_arr <= high)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(p_arr_lin[mask], f_arr[mask]))


def find_peak_in_band(
    f_arr: np.ndarray,
    p_arr_db: np.ndarray,
    band: tuple[float, float],
    f_max_limit: float,
) -> Tuple[Optional[float], Optional[float]]:
    low, high = band
    high = min(high, f_max_limit)
    mask = (f_arr >= low) & (f_arr <= high)
    if not np.any(mask):
        return None, None
    f_sub = f_arr[mask]
    p_sub = p_arr_db[mask]
    idx = int(np.argmax(p_sub))
    return float(f_sub[idx]), float(p_sub[idx])
