from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .. import config, masking


def _odd_window(n: int, minimum: int = 5) -> int:
    n = max(int(n), int(minimum))
    if n % 2 == 0:
        n += 1
    return n


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if x.size == 0:
        return x.copy()
    window = max(1, int(window))
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def _linear_detrend_safe(x: np.ndarray) -> np.ndarray:
    xx = np.asarray(x, dtype=float).copy()
    if xx.size < 2:
        return xx
    ok = np.isfinite(xx)
    if np.count_nonzero(ok) < 2:
        fill = float(np.nanmedian(xx[ok])) if np.any(ok) else 0.0
        xx[~ok] = fill
        return xx - fill
    idx = np.arange(xx.size, dtype=float)
    fill = float(np.nanmedian(xx[ok]))
    xx[~ok] = fill
    coeff = np.polyfit(idx, xx, 1)
    trend = coeff[0] * idx + coeff[1]
    return xx - trend


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.asarray(mask, dtype=bool)
    if m.size == 0 or not np.any(m):
        return []
    d = np.diff(m.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if m[0]:
        starts = np.r_[0, starts]
    if m[-1]:
        ends = np.r_[ends, m.size]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]


def _robust_high_outliers(x: np.ndarray, z: float = 3.0) -> np.ndarray:
    xx = np.asarray(x, dtype=float)
    out = np.zeros(xx.shape, dtype=bool)
    ok = np.isfinite(xx)
    if np.count_nonzero(ok) < 4:
        return out
    med = float(np.nanmedian(xx[ok]))
    mad = float(np.nanmedian(np.abs(xx[ok] - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        sd = float(np.nanstd(xx[ok]))
        scale = sd if np.isfinite(sd) and sd > 0 else 1.0
    out[ok] = xx[ok] > (med + float(z) * scale)
    return out


def _extract_pwa_series(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if signal.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    sig = np.asarray(signal, dtype=float).copy()
    if not np.all(np.isfinite(sig)):
        finite = sig[np.isfinite(sig)]
        fill = float(np.nanmedian(finite)) if finite.size else 0.0
        sig[~np.isfinite(sig)] = fill
    sig = _linear_detrend_safe(sig)
    smooth_win = _odd_window(int(round(0.2 * float(fs))), minimum=5)
    if sig.size >= smooth_win:
        sig_f = savgol_filter(sig, window_length=smooth_win, polyorder=2, mode="interp")
    else:
        sig_f = sig.copy()

    signal_mean = float(np.nanmean(np.abs(sig))) if np.any(np.isfinite(sig)) else 0.0
    prominence = max(signal_mean / 100.0, 1e-9)

    peaks_max, props_max = find_peaks(sig_f, prominence=prominence)
    peaks_min, props_min = find_peaks(-sig_f, prominence=prominence)
    max_vals = sig_f[peaks_max]
    min_vals = sig_f[peaks_min]

    values = np.concatenate([peaks_max, peaks_min])
    amps = np.concatenate([max_vals, min_vals])
    types = np.concatenate([np.ones(peaks_max.size, dtype=int), -np.ones(peaks_min.size, dtype=int)])
    if values.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    order = np.argsort(values)
    values = values[order]
    amps = amps[order]
    types = types[order]

    keep = np.ones(values.size, dtype=bool)
    i = 0
    while i < values.size - 1:
        if types[i] == types[i + 1]:
            if types[i] == 1:
                drop_idx = i if amps[i] < amps[i + 1] else i + 1
            else:
                drop_idx = i if amps[i] > amps[i + 1] else i + 1
            keep[drop_idx] = False
        i += 1
    values = values[keep]
    amps = amps[keep]
    types = types[keep]

    pwa: list[float] = []
    points: list[float] = []
    d_cc: list[float] = []
    max_hr = float(getattr(config, "PWA_DROP_MAX_HR_BPM", 250.0))
    min_d = 60.0 / max(1.0, max_hr)
    prev_point = None
    for i in range(values.size - 1):
        if types[i] != 1 or types[i + 1] != -1:
            continue
        pwa_i = float(amps[i] - amps[i + 1])
        point_idx = int(values[i])
        if prev_point is None:
            d_i = np.inf
        else:
            d_i = (point_idx - prev_point) / float(fs)
        prev_point = point_idx
        if d_i < min_d:
            continue
        pwa.append(pwa_i)
        points.append(point_idx / float(fs))
        d_cc.append(d_i if np.isfinite(d_i) else 1.0)

    return np.asarray(pwa, dtype=float), np.asarray(points, dtype=float), np.asarray(d_cc, dtype=float)


def extract_pwa_debug_from_pat_signal(signal: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    """
    Return intermediate maxima/minima used to derive beat-to-beat PWA.

    This is intentionally separate from the HR/PRV peak detector so the debug
    PDF can compare the two peak-picking paths without changing calculations.
    """
    if signal.size == 0 or fs <= 0:
        return {}

    sig = np.asarray(signal, dtype=float).copy()
    if not np.all(np.isfinite(sig)):
        finite = sig[np.isfinite(sig)]
        fill = float(np.nanmedian(finite)) if finite.size else 0.0
        sig[~np.isfinite(sig)] = fill
    sig = _linear_detrend_safe(sig)
    smooth_win = _odd_window(int(round(0.2 * float(fs))), minimum=5)
    if sig.size >= smooth_win:
        sig_f = savgol_filter(sig, window_length=smooth_win, polyorder=2, mode="interp")
    else:
        sig_f = sig.copy()

    signal_mean = float(np.nanmean(np.abs(sig))) if np.any(np.isfinite(sig)) else 0.0
    prominence = max(signal_mean / 100.0, 1e-9)
    peaks_max, _props_max = find_peaks(sig_f, prominence=prominence)
    peaks_min, _props_min = find_peaks(-sig_f, prominence=prominence)

    values = np.concatenate([peaks_max, peaks_min])
    amps = np.concatenate([sig_f[peaks_max], sig_f[peaks_min]])
    types = np.concatenate([np.ones(peaks_max.size, dtype=int), -np.ones(peaks_min.size, dtype=int)])
    if values.size == 0:
        return {
            "signal_smooth": sig_f,
            "max_indices": peaks_max,
            "min_indices": peaks_min,
            "pair_max_indices": np.array([], dtype=int),
            "pair_min_indices": np.array([], dtype=int),
            "pwa_values": np.array([], dtype=float),
        }

    order = np.argsort(values)
    values = values[order]
    amps = amps[order]
    types = types[order]

    keep = np.ones(values.size, dtype=bool)
    i = 0
    while i < values.size - 1:
        if types[i] == types[i + 1]:
            if types[i] == 1:
                drop_idx = i if amps[i] < amps[i + 1] else i + 1
            else:
                drop_idx = i if amps[i] > amps[i + 1] else i + 1
            keep[drop_idx] = False
        i += 1
    values = values[keep]
    amps = amps[keep]
    types = types[keep]

    pair_max: list[int] = []
    pair_min: list[int] = []
    pwa_vals: list[float] = []
    max_hr = float(getattr(config, "PWA_DROP_MAX_HR_BPM", 250.0))
    min_d = 60.0 / max(1.0, max_hr)
    prev_point = None
    for i in range(values.size - 1):
        if types[i] != 1 or types[i + 1] != -1:
            continue
        point_idx = int(values[i])
        if prev_point is None:
            d_i = np.inf
        else:
            d_i = (point_idx - prev_point) / float(fs)
        prev_point = point_idx
        if d_i < min_d:
            continue
        pair_max.append(point_idx)
        pair_min.append(int(values[i + 1]))
        pwa_vals.append(float(amps[i] - amps[i + 1]))

    return {
        "signal_smooth": sig_f,
        "max_indices": peaks_max,
        "min_indices": peaks_min,
        "pair_max_indices": np.asarray(pair_max, dtype=int),
        "pair_min_indices": np.asarray(pair_min, dtype=int),
        "pwa_values": np.asarray(pwa_vals, dtype=float),
    }


def _pwa_sensorloss_mask(signal: np.ndarray, fs: float) -> np.ndarray:
    sig = np.asarray(signal, dtype=float)
    if sig.size == 0:
        return np.array([], dtype=bool)
    win = max(3, int(round(100.0 * (float(fs) / 32.0))))
    power = np.sqrt(_moving_average(sig * sig, win))
    thr_cfg = float(getattr(config, "PWA_DROP_SENSORLOSS_THR", 5.0))
    thr = min(thr_cfg, abs(float(np.nanmean(power) - 3.0 * np.nanstd(power)))) if np.any(np.isfinite(power)) else thr_cfg
    return power < thr


def _drop_events_from_pwa(
    pwa: np.ndarray,
    points_sec: np.ndarray,
    d_cc_sec: np.ndarray,
) -> list[dict[str, float]]:
    if pwa.size < 8 or points_sec.size != pwa.size:
        return []

    pwa = np.asarray(pwa, dtype=float).copy()
    if not np.all(np.isfinite(pwa)):
        finite = pwa[np.isfinite(pwa)]
        fill = float(np.nanmedian(finite)) if finite.size else 0.0
        pwa[~np.isfinite(pwa)] = fill
    pwa = _linear_detrend_safe(pwa) + float(np.nanmean(pwa))
    win_hi = _odd_window(min(11, max(5, pwa.size // 10)), minimum=5)
    win_lo = _odd_window(min(5, max(5, pwa.size // 20)), minimum=5)
    pwa_f = savgol_filter(pwa, window_length=min(win_hi, pwa.size - (1 - pwa.size % 2)), polyorder=2, mode="interp") if pwa.size >= 5 else pwa.copy()
    pwa_ff = savgol_filter(pwa, window_length=min(win_lo, pwa.size - (1 - pwa.size % 2)), polyorder=2, mode="interp") if pwa.size >= 5 else pwa.copy()

    sl_w = 5
    derivative = np.zeros_like(pwa, dtype=float)
    var_var = np.zeros_like(pwa, dtype=float)
    for i in range(2, pwa.size - 2):
        dt = float(np.nanmean(d_cc_sec[max(0, i - 1): min(d_cc_sec.size, i + 3)]))
        dt = dt if np.isfinite(dt) and dt > 0 else 1.0
        derivative[i] = (pwa_ff[i - 2] - 8 * pwa_ff[i - 1] + 8 * pwa_ff[i + 1] - pwa_ff[i + 2]) / (12.0 * dt)
        var_var[i] = float(np.nanstd(pwa_ff[i - 2:i + 3]))

    unstable = _robust_high_outliers(var_var, z=3.0)
    mask_baseline = np.ones_like(pwa, dtype=bool)
    mask_baseline[unstable] = False
    for s, e in _contiguous_true_runs(mask_baseline):
        if (e - s) <= 3:
            mask_baseline[s:e] = False

    temp = var_var.copy()
    temp[derivative > 0] = np.nan
    finite_temp = np.where(np.isfinite(temp), temp, -np.inf)
    prominence = max(float(np.nanstd(var_var)) / 10.0, 1e-9)
    candidates, _ = find_peaks(finite_temp, prominence=prominence)
    if candidates.size == 0:
        return []

    merged: list[int] = []
    for cand in candidates:
        if not merged:
            merged.append(int(cand))
            continue
        prev = merged[-1]
        if (points_sec[cand] - points_sec[prev]) < 1.0:
            if var_var[cand] > var_var[prev]:
                merged[-1] = int(cand)
        else:
            merged.append(int(cand))
    candidates = np.asarray(merged, dtype=int)

    max_idx, _ = find_peaks(pwa_f)
    min_idx, _ = find_peaks(-pwa_f)
    primary_thr = -abs(float(getattr(config, "PWA_DROP_PRIMARY_THR_PCT", 40.0)))
    secondary_thr = -abs(float(getattr(config, "PWA_DROP_SECONDARY_THR_PCT", 30.0)))
    n_primary = int(getattr(config, "PWA_DROP_MIN_POINTS_PRIMARY", 2))
    n_secondary = int(getattr(config, "PWA_DROP_MIN_POINTS_SECONDARY", 4))
    baseline_cycles = int(getattr(config, "PWA_DROP_BASELINE_CYCLES", 5))
    min_baseline = int(getattr(config, "PWA_DROP_SUMMARY_MIN_BASELINE_POINTS", 3))

    events: list[dict[str, float]] = []
    for cand in candidates:
        prev_max = max_idx[max_idx <= cand]
        next_min = min_idx[min_idx >= cand]
        if prev_max.size == 0 or next_min.size == 0:
            continue
        p_start = int(prev_max[-1])
        p_min = int(next_min[0])
        next_max = max_idx[max_idx >= p_min]
        if next_max.size == 0:
            p_stop = pwa.size - 1
        else:
            p_stop = int(next_max[0])
        if (points_sec[p_stop] - points_sec[p_min]) < 5.0:
            later_max = max_idx[max_idx > p_stop]
            if later_max.size > 0:
                p_stop = int(later_max[0])
        p_stop = min(p_stop, p_min + 30, pwa.size - 1)

        base_idx = np.where(mask_baseline[:p_start])[0]
        if base_idx.size < min_baseline:
            continue
        l_b = min(base_idx.size, baseline_cycles)
        baseline = float(np.nanmedian(pwa[base_idx[-l_b:]]))
        if not np.isfinite(baseline) or baseline == 0:
            continue

        p_all = 100.0 * ((pwa[p_start:p_stop + 1] - baseline) / baseline)
        desc = p_all[:(p_min - p_start + 1)]
        asc = p_all[(p_min - p_start):]

        below_primary = np.where(p_all <= primary_thr)[0]
        below_secondary = np.where(p_all <= secondary_thr)[0]
        consecutive_primary = np.where(np.diff(below_primary) <= 2)[0] if below_primary.size >= 2 else np.array([], dtype=int)
        consecutive_secondary = np.where(np.diff(below_secondary) <= 2)[0] if below_secondary.size >= 2 else np.array([], dtype=int)
        if consecutive_primary.size < n_primary or consecutive_secondary.size < n_secondary:
            continue

        i_var = int(np.argmin(desc))
        sign_10 = desc < -10.0
        starts = np.where(np.diff(np.r_[0, sign_10.astype(int)]) == 1)[0]
        if starts.size == 0:
            p_in = max(p_min - 1, p_start)
            mask_drop_start = p_min - p_start
        else:
            valid_starts = starts[starts <= i_var]
            mask_drop_start = int(valid_starts[-1] if valid_starts.size else starts[0])
            p_in = p_start + mask_drop_start

        asc_der = derivative[p_min:p_stop + 1].copy()
        if asc_der.size == 0:
            continue
        idm = int(np.argmax(asc_der))
        asc_der[:idm + 1] = 10.0
        return_candidates = np.where((asc > -10.0) | (asc_der < 1.0))[0]
        p_end = p_min + int(return_candidates[0]) if return_candidates.size else p_stop
        p_end = min(max(p_end, p_min), p_stop)

        t0 = float(points_sec[p_in])
        t1 = float(points_sec[p_end])
        tc = float(points_sec[p_min])
        if not (np.isfinite(t0) and np.isfinite(t1) and np.isfinite(tc) and t1 > t0):
            continue

        p_segment = 100.0 * ((pwa[p_in:p_end + 1] - baseline) / baseline)
        p_segment[p_segment > 0] = 0.0
        tt = points_sec[p_in:p_end + 1]
        auc = float(np.trapz(-p_segment, tt))
        amp = float(np.max(np.abs(p_all)))
        slope1 = float((100.0 * (pwa_f[p_min] - baseline) / baseline - 100.0 * (pwa_f[p_in] - baseline) / baseline) / max(1e-6, points_sec[p_min] - points_sec[p_in]))
        slope2 = float((100.0 * (pwa_f[p_end] - baseline) / baseline - 100.0 * (pwa_f[p_min] - baseline) / baseline) / max(1e-6, points_sec[p_end] - points_sec[p_min]))
        events.append(
            {
                "t_start": t0,
                "t_end": t1,
                "t_center": tc,
                "duration_sec": float(t1 - t0),
                "duration_beats": float(p_end - p_in + 1),
                "amplitude_pct": amp,
                "auc_pct_sec": auc,
                "slope_down_pct_per_sec": slope1,
                "slope_up_pct_per_sec": slope2,
                "baseline": baseline,
            }
        )

    return events


def _sleep_hours_from_policy(t_sec: np.ndarray, aux_df, include_set: Optional[set[int]]) -> float:
    if aux_df is None or t_sec.size == 0:
        return float((t_sec[-1] - t_sec[0]) / 3600.0) if t_sec.size >= 2 else 0.0
    policy = masking.policy_from_config(include_stages=include_set, force_sleep=(include_set is not None))
    bundle = masking.build_mask_bundle(t_sec, aux_df, policy=policy)
    keep = np.asarray(bundle.sleep_keep, dtype=bool)
    if keep.size < 2:
        return 0.0
    dt = np.diff(t_sec)
    dt = np.clip(dt, 0.0, None)
    intervals = keep[:-1] & keep[1:]
    return float(np.sum(dt[intervals]) / 3600.0)


def _summarize_events(events: list[dict[str, float]], sleep_hours: float) -> dict[str, float]:
    out: dict[str, float] = {
        "n_drops": 0.0,
        "drop_rate_per_sleep_hour": np.nan,
        "mean_amplitude_pct": np.nan,
        "mean_duration_sec": np.nan,
        "mean_auc_pct_sec": np.nan,
        "mean_slope_down_pct_per_sec": np.nan,
        "mean_slope_up_pct_per_sec": np.nan,
        "sleep_hours": float(sleep_hours),
        "n_drops_event_overlap": 0.0,
        "event_overlap_pct": np.nan,
    }
    if not events:
        out["drop_rate_per_sleep_hour"] = 0.0 if sleep_hours > 0 else np.nan
        return out
    out["n_drops"] = float(len(events))
    out["drop_rate_per_sleep_hour"] = float(len(events) / sleep_hours) if sleep_hours > 0 else np.nan
    for key, dst in [
        ("amplitude_pct", "mean_amplitude_pct"),
        ("duration_sec", "mean_duration_sec"),
        ("auc_pct_sec", "mean_auc_pct_sec"),
        ("slope_down_pct_per_sec", "mean_slope_down_pct_per_sec"),
        ("slope_up_pct_per_sec", "mean_slope_up_pct_per_sec"),
    ]:
        vals = [float(ev[key]) for ev in events if key in ev and np.isfinite(ev[key])]
        if vals:
            out[dst] = float(np.nanmean(vals))
    overlap = [bool(ev.get("event_overlap", False)) for ev in events]
    out["n_drops_event_overlap"] = float(sum(overlap))
    out["event_overlap_pct"] = float(100.0 * sum(overlap) / len(overlap)) if overlap else np.nan
    return out


def compute_pwa_drop_from_pat_signal(
    pat_signal: np.ndarray,
    fs: float,
    *,
    aux_df=None,
    include_set: Optional[set[int]] = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], list[dict[str, float]]]:
    signal = np.asarray(pat_signal, dtype=float)
    if signal.ndim != 1 or signal.size == 0 or fs <= 0:
        return np.array([], dtype=float), np.array([], dtype=float), _summarize_events([], 0.0), []

    pwa, points_sec, d_cc_sec = _extract_pwa_series(signal, fs)
    if pwa.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float), _summarize_events([], 0.0), []

    sensorloss_sf = _pwa_sensorloss_mask(signal, fs)
    artefacts = np.zeros(pwa.size, dtype=bool)
    if pwa.size >= 3:
        delta_c = np.abs(np.diff(pwa))
        if delta_c.size >= 2:
            thr = 3.0 * float(np.nanstd(np.abs(delta_c[:-1])))
            bad = np.where((delta_c[:-1] > thr) & (delta_c[1:] > thr))[0] + 1
            artefacts[bad] = True
    if sensorloss_sf.size and points_sec.size:
        point_idx = np.clip(np.round(points_sec * float(fs)).astype(int), 0, sensorloss_sf.size - 1)
        artefacts |= sensorloss_sf[point_idx]

    pwa_clean = pwa[~artefacts]
    points_clean = points_sec[~artefacts]
    d_cc_clean = d_cc_sec[~artefacts]
    events = _drop_events_from_pwa(pwa_clean, points_clean, d_cc_clean)

    if aux_df is not None and points_clean.size > 0:
        policy = masking.policy_from_config(include_stages=include_set, force_sleep=(include_set is not None))
        bundle = masking.build_mask_bundle(points_clean, aux_df, policy=policy)
        sleep_keep = np.asarray(bundle.sleep_keep, dtype=bool)
        event_overlap_keep = np.asarray(bundle.event_keep & bundle.desat_keep, dtype=bool)
        center_lookup = np.searchsorted(points_clean, [ev["t_center"] for ev in events], side="left")
        selected_events: list[dict[str, float]] = []
        for ev, idx in zip(events, center_lookup):
            idx = min(max(int(idx), 0), max(0, points_clean.size - 1))
            if not sleep_keep[idx]:
                continue
            ev2 = dict(ev)
            ev2["event_overlap"] = bool(not event_overlap_keep[idx])
            selected_events.append(ev2)
        events = selected_events

    sleep_hours = _sleep_hours_from_policy(points_clean, aux_df, include_set)
    summary = _summarize_events(events, sleep_hours)
    summary["n_pwa_points"] = float(points_clean.size)
    summary["valid_pwa_min"] = float((points_clean[-1] - points_clean[0]) / 60.0) if points_clean.size >= 2 else 0.0
    return points_clean, pwa_clean, summary, events
