from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.signal import welch

from .. import config, masking


def _empty_summary(reason: str) -> dict[str, float]:
    out: dict[str, float] = {
        "n_windows_total": 0.0,
        "n_windows_valid": 0.0,
        "valid_pct": np.nan,
        "window_sec": float(getattr(config, "PAT_HARMONICS_WINDOW_SEC", 120.0)),
        "hop_sec": float(getattr(config, "PAT_HARMONICS_HOP_SEC", 120.0)),
        "diag_reason": reason,
    }
    for key in ["f0_hz", "h1_power", "h2_power", "h3_power", "h4_power", "h5_power", "h2_h1", "h3_h1", "harmonic_total_power", "harmonic_distortion_index"]:
        out[f"{key}_mean"] = np.nan
        out[f"{key}_median"] = np.nan
    return out


def _finite_summary(values: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    return float(np.nanmean(vals)), float(np.nanmedian(vals))


def _integrate_band(f: np.ndarray, pxx: np.ndarray, center_hz: float, bandwidth_hz: float) -> float:
    half = 0.5 * float(bandwidth_hz)
    lo = max(0.0, float(center_hz) - half)
    hi = float(center_hz) + half
    mask = (f >= lo) & (f <= hi)
    if np.count_nonzero(mask) < 2:
        return np.nan
    return float(np.trapz(pxx[mask], f[mask]))


def _window_harmonics(y: np.ndarray, fs: float) -> dict[str, float] | None:
    yy = np.asarray(y, dtype=float)
    ok = np.isfinite(yy)
    if np.count_nonzero(ok) < 8:
        return None
    fill = float(np.nanmedian(yy[ok]))
    yy = yy.copy()
    yy[~ok] = fill
    yy = yy - float(np.nanmean(yy))

    nperseg_sec = float(getattr(config, "PAT_HARMONICS_WELCH_NPERSEG_SEC", 16.0))
    nperseg = min(yy.size, max(8, int(round(nperseg_sec * float(fs)))))
    nfft = max(int(getattr(config, "PAT_HARMONICS_NFFT", 4096)), nperseg)
    if nperseg < 8:
        return None

    f, pxx = welch(yy, fs=float(fs), nperseg=nperseg, nfft=nfft)
    if f.size == 0 or pxx.size == 0:
        return None

    f0_band = tuple(getattr(config, "PAT_HARMONICS_F0_BAND_HZ", (0.5, 2.5)))
    f0_mask = (f >= float(f0_band[0])) & (f <= float(f0_band[1]))
    if not np.any(f0_mask):
        return None
    f0_candidates = f[f0_mask]
    p0 = pxx[f0_mask]
    if p0.size == 0 or not np.any(np.isfinite(p0)):
        return None
    f0_hz = float(f0_candidates[int(np.nanargmax(p0))])

    max_n = max(1, int(getattr(config, "PAT_HARMONICS_MAX_N", 5)))
    bandwidth = float(getattr(config, "PAT_HARMONICS_BANDWIDTH_HZ", 0.08))
    nyquist = 0.5 * float(fs)
    out: dict[str, float] = {"f0_hz": f0_hz}
    powers: list[float] = []
    for n in range(1, max_n + 1):
        center = n * f0_hz
        val = _integrate_band(f, pxx, center, bandwidth) if center < nyquist else np.nan
        out[f"h{n}_power"] = val
        powers.append(val)

    h1 = out.get("h1_power", np.nan)
    out["h2_h1"] = float(out.get("h2_power", np.nan) / h1) if np.isfinite(h1) and h1 > 0 else np.nan
    out["h3_h1"] = float(out.get("h3_power", np.nan) / h1) if np.isfinite(h1) and h1 > 0 else np.nan
    finite_powers = np.asarray([p for p in powers if np.isfinite(p)], dtype=float)
    out["harmonic_total_power"] = float(np.sum(finite_powers)) if finite_powers.size else np.nan
    higher = np.asarray([out.get(f"h{n}_power", np.nan) for n in range(2, max_n + 1)], dtype=float)
    higher = higher[np.isfinite(higher)]
    out["harmonic_distortion_index"] = float(np.sum(higher) / h1) if np.isfinite(h1) and h1 > 0 and higher.size else np.nan
    return out


def compute_pat_harmonics_from_raw_pat(
    pat_signal: np.ndarray,
    fs: float,
    *,
    aux_df=None,
    include_set: Optional[set[int]] = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    signal = np.asarray(pat_signal, dtype=float)
    if signal.ndim != 1 or signal.size == 0 or fs <= 0:
        return _empty_summary("invalid_signal"), []

    duration_sec = signal.size / float(fs)
    window_sec = float(getattr(config, "PAT_HARMONICS_WINDOW_SEC", 120.0))
    hop_sec = float(getattr(config, "PAT_HARMONICS_HOP_SEC", window_sec))
    min_valid_fraction = float(getattr(config, "PAT_HARMONICS_MIN_VALID_FRACTION", 0.80))
    half = 0.5 * window_sec
    centers = np.arange(half, max(half, duration_sec - half) + 1e-9, hop_sec)
    if centers.size == 0:
        return _empty_summary("no_windows_defined"), []

    policy = masking.policy_from_config(include_stages=include_set, force_sleep=(include_set is not None))
    windows: list[dict[str, float]] = []
    n_valid = 0
    for center in centers:
        start_sec = float(center - half)
        end_sec = float(center + half)
        i0 = max(0, int(np.floor(start_sec * float(fs))))
        i1 = min(signal.size, int(np.ceil(end_sec * float(fs))))
        if i1 <= i0:
            continue
        t_win = np.arange(i0, i1, dtype=float) / float(fs)
        y_win = signal[i0:i1].astype(float)
        bundle = masking.build_mask_bundle(t_win, aux_df, policy=policy)
        keep = np.asarray(bundle.combined_keep, dtype=bool) & np.isfinite(y_win)
        valid_fraction = float(np.mean(keep)) if keep.size else 0.0
        row: dict[str, float] = {
            "t_start_sec": start_sec,
            "t_end_sec": end_sec,
            "t_center_sec": float(center),
            "valid_fraction": valid_fraction,
            "valid": 0.0,
        }
        if valid_fraction >= min_valid_fraction:
            y_used = y_win.copy()
            if np.any(keep):
                fill = float(np.nanmedian(y_used[keep]))
                y_used[~keep] = fill
            metrics = _window_harmonics(y_used, fs)
            if metrics is not None and np.isfinite(metrics.get("f0_hz", np.nan)):
                row.update(metrics)
                row["valid"] = 1.0
                n_valid += 1
        windows.append(row)

    if n_valid == 0:
        summary = _empty_summary("no_valid_windows")
        summary["n_windows_total"] = float(len(windows))
        summary["valid_pct"] = 0.0 if windows else np.nan
        return summary, windows

    summary = _empty_summary("")
    summary["n_windows_total"] = float(len(windows))
    summary["n_windows_valid"] = float(n_valid)
    summary["valid_pct"] = float(100.0 * n_valid / len(windows)) if windows else np.nan
    summary["diag_reason"] = ""
    keys = ["f0_hz", "h1_power", "h2_power", "h3_power", "h4_power", "h5_power", "h2_h1", "h3_h1", "harmonic_total_power", "harmonic_distortion_index"]
    for key in keys:
        vals = np.asarray([float(w.get(key, np.nan)) for w in windows if float(w.get("valid", 0.0)) == 1.0], dtype=float)
        mean, median = _finite_summary(vals)
        summary[f"{key}_mean"] = mean
        summary[f"{key}_median"] = median
    return summary, windows
