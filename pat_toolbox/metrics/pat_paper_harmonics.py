from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import welch

from .. import config, masking, sleep_mask
from ..core.pr_cleaning import detect_pat_peaks


def _empty_summary(reason: str) -> dict[str, float]:
    out: dict[str, float] = {
        "n_windows_total": 0.0,
        "n_windows_valid": 0.0,
        "valid_pct": np.nan,
        "window_sec": float(getattr(config, "PAT_PAPER_HARMONICS_WINDOW_SEC", 120.0)),
        "hop_sec": float(getattr(config, "PAT_PAPER_HARMONICS_HOP_SEC", 60.0)),
        "diag_reason": reason,
    }
    for key in _summary_metric_keys():
        out[f"{key}_mean"] = np.nan
        out[f"{key}_median"] = np.nan
    return out


def _summary_metric_keys() -> list[str]:
    keys = ["c0", "hf_ratio", "sub_vlf_power", "sub_lf_power", "sub_hf_power"]
    keys.extend([f"c{n}" for n in range(1, 11)])
    keys.extend([f"c{n}_c0" for n in range(1, 11)])
    return keys


def _finite_summary(values: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    return float(np.nanmean(vals)), float(np.nanmedian(vals))


def _stage_label(stage_code: float) -> str:
    labels = {
        0: "wake",
        1: "light",
        2: "deep",
        3: "rem",
    }
    if not np.isfinite(stage_code):
        return "unknown"
    return labels.get(int(round(stage_code)), "unknown")


def _stage_codes_at_times(aux_df, t_sec: np.ndarray) -> np.ndarray:
    tt = np.asarray(t_sec, dtype=float)
    out = np.full(tt.size, np.nan, dtype=float)
    if aux_df is None or tt.size == 0:
        return out
    time_col = getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec")
    stage_col = getattr(config, "AUX_CSV_STAGE_CODE_COLUMN", "stage_code")
    try:
        aux_stage = sleep_mask.ensure_stage_code_column(aux_df)
        if time_col not in aux_stage.columns or stage_col not in aux_stage.columns:
            return out
        aux_t = np.asarray(aux_stage[time_col].to_numpy(dtype=float), dtype=float)
        aux_s = np.asarray(aux_stage[stage_col].to_numpy(dtype=float), dtype=float)
    except Exception:
        return out
    ok = np.isfinite(aux_t) & np.isfinite(aux_s)
    if not np.any(ok):
        return out
    aux_t = aux_t[ok]
    aux_s = aux_s[ok]
    order = np.argsort(aux_t)
    aux_t = aux_t[order]
    aux_s = aux_s[order]
    idx = np.searchsorted(aux_t, tt, side="left")
    idx0 = np.clip(idx - 1, 0, aux_t.size - 1)
    idx1 = np.clip(idx, 0, aux_t.size - 1)
    d0 = np.abs(tt - aux_t[idx0])
    d1 = np.abs(tt - aux_t[idx1])
    pick = np.where(d1 < d0, idx1, idx0)
    valid_t = np.isfinite(tt)
    out[valid_t] = np.round(aux_s[pick[valid_t]])
    return out


def _add_stage_summaries(summary: dict[str, float], rows: list[dict[str, float]]) -> None:
    valid_rows = [row for row in rows if float(row.get("valid", 0.0)) == 1.0]
    for stage in ["wake", "light", "deep", "rem"]:
        stage_rows = [row for row in valid_rows if str(row.get("sleep_stage_label", "")) == stage]
        prefix = f"{stage}_"
        summary[f"{prefix}n_windows_valid"] = float(len(stage_rows))
        for key in _summary_metric_keys():
            vals = np.asarray([float(row.get(key, np.nan)) for row in stage_rows], dtype=float)
            mean, median = _finite_summary(vals)
            summary[f"{prefix}{key}_mean"] = mean
            summary[f"{prefix}{key}_median"] = median


def _resample_pulse(y: np.ndarray, n_samples: int) -> np.ndarray:
    if y.size < 3:
        return np.full(n_samples, np.nan, dtype=float)
    x_old = np.linspace(0.0, 1.0, y.size)
    x_new = np.linspace(0.0, 1.0, n_samples)
    return np.interp(x_new, x_old, y.astype(float))


def _pulse_coefficients(pulse: np.ndarray, max_n: int) -> dict[str, float]:
    y = np.asarray(pulse, dtype=float)
    if y.size < 8 or not np.all(np.isfinite(y)):
        return {}
    coeff = np.fft.rfft(y)
    n = float(y.size)
    c0 = float(abs(coeff[0]) / n)
    out: dict[str, float] = {"c0": c0}
    for k in range(1, max_n + 1):
        amp = float(2.0 * abs(coeff[k]) / n) if k < coeff.size else np.nan
        out[f"c{k}"] = amp
        out[f"c{k}_c0"] = float(amp / c0) if np.isfinite(c0) and c0 > 0 else np.nan
    return out


def _subharmonic_powers(y: np.ndarray, fs: float) -> dict[str, float]:
    yy = np.asarray(y, dtype=float)
    ok = np.isfinite(yy)
    out = {"sub_vlf_power": np.nan, "sub_lf_power": np.nan, "sub_hf_power": np.nan}
    if np.count_nonzero(ok) < max(16, int(round(fs * 10.0))):
        return out
    fill = float(np.nanmedian(yy[ok]))
    yy = yy.copy()
    yy[~ok] = fill
    yy -= float(np.nanmean(yy))
    nperseg = min(yy.size, max(16, int(round(float(getattr(config, "PAT_PAPER_HARMONICS_SUB_NPERSEG_SEC", 64.0)) * fs))))
    f, pxx = welch(yy, fs=float(fs), nperseg=nperseg)
    bands = {
        "sub_vlf_power": (0.005, 0.04),
        "sub_lf_power": (0.04, 0.15),
        "sub_hf_power": (0.15, 0.50),
    }
    for key, (lo, hi) in bands.items():
        m = (f >= lo) & (f < hi)
        if np.count_nonzero(m) >= 2:
            out[key] = float(np.trapz(pxx[m], f[m]))
    return out


def _window_metrics(
    signal: np.ndarray,
    peak_indices: np.ndarray,
    keep_mask: np.ndarray,
    fs: float,
    i0: int,
    i1: int,
) -> dict[str, float] | None:
    max_n = int(getattr(config, "PAT_PAPER_HARMONICS_MAX_N", 10))
    resample_n = int(getattr(config, "PAT_PAPER_HARMONICS_RESAMPLE_N", 256))
    min_beats = int(getattr(config, "PAT_PAPER_HARMONICS_MIN_BEATS", 20))
    pulse_rows: list[dict[str, float]] = []
    pulses: list[np.ndarray] = []

    peaks = peak_indices[(peak_indices >= i0) & (peak_indices < i1)]
    for a, b in zip(peaks[:-1], peaks[1:]):
        if b <= a + 2:
            continue
        pr_sec = (b - a) / float(fs)
        if pr_sec < float(getattr(config, "HR_MIN_PR_SEC", 0.3)) or pr_sec > float(getattr(config, "HR_MAX_PR_SEC", 2.5)):
            continue
        if np.mean(keep_mask[a:b]) < float(getattr(config, "PAT_PAPER_HARMONICS_MIN_PULSE_VALID_FRACTION", 0.90)):
            continue
        pulse = _resample_pulse(signal[a:b], resample_n)
        coeffs = _pulse_coefficients(pulse, max_n)
        if coeffs and np.isfinite(coeffs.get("c0", np.nan)):
            pulse_rows.append(coeffs)
            pulses.append(pulse)

    if len(pulse_rows) < min_beats:
        return None

    out: dict[str, float] = {"n_beats": float(len(pulse_rows))}
    for key in ["c0", *[f"c{n}" for n in range(1, max_n + 1)], *[f"c{n}_c0" for n in range(1, max_n + 1)]]:
        vals = np.asarray([row.get(key, np.nan) for row in pulse_rows], dtype=float)
        out[key] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan

    denom = np.nansum([out.get(f"c{n}_c0", np.nan) for n in range(1, max_n + 1)])
    num = np.nansum([out.get(f"c{n}_c0", np.nan) for n in range(6, max_n + 1)])
    out["hf_ratio"] = float(num / denom) if np.isfinite(denom) and denom > 0 else np.nan
    out.update(_subharmonic_powers(signal[i0:i1], fs))

    ensemble = np.nanmean(np.vstack(pulses), axis=0)
    for idx, val in enumerate(ensemble[: min(32, ensemble.size)]):
        out[f"ensemble_{idx:02d}"] = float(val)
    return out


def compute_pat_paper_harmonics_from_raw_pat(
    pat_signal: np.ndarray,
    fs: float,
    *,
    aux_df=None,
    include_set: Optional[set[int]] = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    signal = np.asarray(pat_signal, dtype=float)
    if signal.ndim != 1 or signal.size == 0 or fs <= 0:
        return _empty_summary("invalid_signal"), []

    try:
        _pat_filt, peaks = detect_pat_peaks(signal, fs)
    except Exception:
        return _empty_summary("peak_detection_failed"), []
    if peaks.size < 2:
        return _empty_summary("insufficient_peaks"), []

    duration_sec = signal.size / float(fs)
    window_sec = float(getattr(config, "PAT_PAPER_HARMONICS_WINDOW_SEC", 120.0))
    hop_sec = float(getattr(config, "PAT_PAPER_HARMONICS_HOP_SEC", 60.0))
    min_valid_fraction = float(getattr(config, "PAT_PAPER_HARMONICS_MIN_VALID_FRACTION", 0.80))
    half = 0.5 * window_sec
    centers = np.arange(half, max(half, duration_sec - half) + 1e-9, hop_sec)
    if centers.size == 0:
        return _empty_summary("no_windows_defined"), []

    use_mask = bool(getattr(config, "PAT_PAPER_HARMONICS_USE_MASK", False))
    policy = masking.policy_from_config(include_stages=include_set, force_sleep=(include_set is not None)) if use_mask else None
    stage_codes = _stage_codes_at_times(aux_df, centers)
    rows: list[dict[str, float]] = []
    n_valid = 0
    for idx_center, center in enumerate(centers):
        start_sec = float(center - half)
        end_sec = float(center + half)
        i0 = max(0, int(np.floor(start_sec * float(fs))))
        i1 = min(signal.size, int(np.ceil(end_sec * float(fs))))
        if i1 <= i0:
            continue
        t_win = np.arange(i0, i1, dtype=float) / float(fs)
        if use_mask:
            bundle = masking.build_mask_bundle(t_win, aux_df, policy=policy)
            keep_win = np.asarray(bundle.combined_keep, dtype=bool) & np.isfinite(signal[i0:i1])
        else:
            keep_win = np.isfinite(signal[i0:i1])
        valid_fraction = float(np.mean(keep_win)) if keep_win.size else 0.0
        row: dict[str, float] = {
            "t_start_sec": start_sec,
            "t_end_sec": end_sec,
            "t_center_sec": float(center),
            "sleep_stage_code": float(stage_codes[idx_center]) if idx_center < stage_codes.size else np.nan,
            "sleep_stage_label": _stage_label(float(stage_codes[idx_center])) if idx_center < stage_codes.size else "unknown",
            "valid_fraction": valid_fraction,
            "valid": 0.0,
        }
        if valid_fraction >= min_valid_fraction:
            keep_full = np.zeros(signal.size, dtype=bool)
            keep_full[i0:i1] = keep_win
            metrics = _window_metrics(signal, peaks, keep_full, fs, i0, i1)
            if metrics is not None:
                row.update(metrics)
                row["valid"] = 1.0
                n_valid += 1
        rows.append(row)

    if n_valid == 0:
        summary = _empty_summary("no_valid_windows")
        summary["n_windows_total"] = float(len(rows))
        summary["valid_pct"] = 0.0 if rows else np.nan
        _add_stage_summaries(summary, rows)
        return summary, rows

    summary = _empty_summary("")
    summary["n_windows_total"] = float(len(rows))
    summary["n_windows_valid"] = float(n_valid)
    summary["valid_pct"] = float(100.0 * n_valid / len(rows)) if rows else np.nan
    summary["diag_reason"] = ""
    for key in _summary_metric_keys():
        vals = np.asarray([float(w.get(key, np.nan)) for w in rows if float(w.get("valid", 0.0)) == 1.0], dtype=float)
        mean, median = _finite_summary(vals)
        summary[f"{key}_mean"] = mean
        summary[f"{key}_median"] = median
    _add_stage_summaries(summary, rows)
    return summary, rows
