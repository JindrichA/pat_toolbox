from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np

from .. import config, masking
from ..core.windows import passes_time_domain_window_gate
from . import hr as hr_metrics
from .hrv_frequency_domain import (
    _calculate_lfhf_fixed_windows,
    _lf_hf_from_rr,
    _lf_hf_from_rr_segmented,
)
from .hrv_time_domain import _calculate_rmssd_series, _rmssd, _sdnn

if TYPE_CHECKING:
    import pandas as pd


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
    n_fail_time_domain = 0
    n_fail_psd = 0
    n_ok = 0

    min_intervals_td = int(getattr(config, "HRV_MIN_INTERVALS_PER_WINDOW", min_rr))
    min_cov_td = float(getattr(config, "HRV_MIN_WINDOW_COVERAGE", 0.0))
    min_span_td = float(getattr(config, "HRV_RMSSD_MIN_SPAN_SEC", min_span_sec))
    max_gap_td = float(getattr(config, "HRV_MAX_RR_GAP_SEC", max_gap_sec))

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

        rr_win_ms = rr_ms[left:right]
        rr_mid_win = rr_mid[left:right]

        td_ok = passes_time_domain_window_gate(
            rr_mid_win,
            window_sec=float(window_sec),
            min_intervals=min_intervals_td,
            max_gap_sec=max_gap_td,
            min_span_sec=min_span_td,
            min_cov=min_cov_td,
        )
        if not td_ok:
            n_fail_time_domain += 1
            continue

        rmssd_i = _rmssd(rr_win_ms)
        sdnn_i = _sdnn(rr_win_ms)

        out["rmssd_ms"][i] = rmssd_i
        out["sdnn_ms"][i] = sdnn_i

        k = right - left
        if k < min_rr:
            continue
        span = float(rr_mid_win[-1] - rr_mid_win[0]) if rr_mid_win.size >= 2 else 0.0
        if span < float(min_span_sec):
            continue

        gaps = np.diff(rr_mid_win) if rr_mid_win.size >= 2 else np.array([])
        if gaps.size > 0 and np.any(gaps > float(max_gap_sec)):
            continue

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
        f"fail_time_domain={n_fail_time_domain}, "
        f"fail_psd={n_fail_psd}"
    )

    return out


def _subset_rr_by_sleep_and_events(
    rr_mid_times_sec: np.ndarray,
    rr_ms: np.ndarray,
    aux_df: Optional["pd.DataFrame"],
    *,
    include_set: Optional[set[int]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rr_mid_times_sec = np.asarray(rr_mid_times_sec, dtype=float)
    rr_ms = np.asarray(rr_ms, dtype=float)

    policy = masking.policy_from_config(
        include_stages=include_set,
        force_sleep=(include_set is not None),
    )
    bundle = masking.build_rr_mask_bundle(rr_mid_times_sec, aux_df, policy=policy)

    rr_mid_sleep = rr_mid_times_sec[bundle.sleep_keep]
    rr_ms_sleep = rr_ms[bundle.sleep_keep]

    rr_mid_clean = rr_mid_times_sec[bundle.combined_keep]
    rr_ms_clean = rr_ms[bundle.combined_keep]
    return rr_mid_sleep, rr_ms_sleep, rr_mid_clean, rr_ms_clean


def _calculate_sdnn_series(
    t_grid: np.ndarray,
    rr_mid: np.ndarray,
    rr_ms: np.ndarray,
    window_sec: float,
    *,
    min_span_sec: float,
    max_gap_sec: float,
) -> np.ndarray:
    out = np.full_like(t_grid, np.nan, dtype=float)
    if rr_mid.size == 0 or rr_ms.size == 0 or t_grid.size == 0:
        return out

    half = 0.5 * float(window_sec)
    left = 0
    right = 0
    n = rr_mid.size
    min_intervals_td = int(getattr(config, "HRV_MIN_INTERVALS_PER_WINDOW", 0))
    min_cov_td = float(getattr(config, "HRV_MIN_WINDOW_COVERAGE", 0.0))
    min_span_td = float(getattr(config, "HRV_RMSSD_MIN_SPAN_SEC", min_span_sec))
    max_gap_td = float(getattr(config, "HRV_MAX_RR_GAP_SEC", max_gap_sec))

    for i, t in enumerate(t_grid):
        start = t - half
        end = t + half
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
            min_intervals=min_intervals_td,
            max_gap_sec=max_gap_td,
            min_span_sec=min_span_td,
            min_cov=min_cov_td,
        ):
            continue
        out[i] = _sdnn(rr_win_ms)
    return out


def _build_summary_from_clean_rr_and_times(
    rr_mid_clean: np.ndarray,
    rr_ms_clean: np.ndarray,
    rmssd_windows_list_clean: list[float],
    sdnn_series_clean: np.ndarray,
    *,
    fs_resample: float,
    max_gap_sec: float,
    min_freq_span_sec: float,
) -> Dict[str, float]:
    sdnn_ms = _sdnn(rr_ms_clean)
    lf, hf, lf_hf, lf_info = _lf_hf_from_rr_segmented(
        rr_ms_clean,
        rr_mid_clean,
        fs_resample=fs_resample,
        max_gap_sec=max_gap_sec,
        min_span_sec=min_freq_span_sec,
        return_info=True,
    )

    if len(rmssd_windows_list_clean) > 0:
        rmssd_arr = np.asarray(rmssd_windows_list_clean, dtype=float)
        sdnn_arr = np.asarray(sdnn_series_clean, dtype=float)
        sdnn_arr = sdnn_arr[np.isfinite(sdnn_arr)]
        summary: Dict[str, float] = {
            "rmssd_mean": float(np.nanmean(rmssd_arr)),
            "rmssd_median": float(np.nanmedian(rmssd_arr)),
            "sdnn_mean": float(np.nanmean(sdnn_arr)) if sdnn_arr.size else float(sdnn_ms),
            "sdnn_median": float(np.nanmedian(sdnn_arr)) if sdnn_arr.size else float(sdnn_ms),
            "sdnn": float(sdnn_ms),
            "lf": float(lf),
            "hf": float(hf),
            "lf_hf": float(lf_hf),
            "lf_n_segments_used": int(lf_info["n_segments_used"]),
        }
    else:
        sdnn_arr = np.asarray(sdnn_series_clean, dtype=float)
        sdnn_arr = sdnn_arr[np.isfinite(sdnn_arr)]
        summary = {
            "rmssd_mean": float(_rmssd(rr_ms_clean)),
            "rmssd_median": float(np.nan),
            "sdnn_mean": float(np.nanmean(sdnn_arr)) if sdnn_arr.size else float(sdnn_ms),
            "sdnn_median": float(np.nanmedian(sdnn_arr)) if sdnn_arr.size else float(sdnn_ms),
            "sdnn": float(sdnn_ms),
            "lf": float(lf),
            "hf": float(hf),
            "lf_hf": float(lf_hf),
            "lf_n_segments_used": int(lf_info["n_segments_used"]),
        }

    return summary


def _apply_fixed_window_frequency_summary(
    summary: Dict[str, float],
    lfhf_fixed: Dict[str, np.ndarray],
    *,
    fixed_win_sec: float,
    fixed_hop_sec: float,
) -> Dict[str, float]:
    valid = lfhf_fixed.get("valid_mask", np.array([], dtype=bool))
    n_total = int(valid.size)
    n_valid = int(np.sum(valid)) if n_total > 0 else 0

    summary["lf_hf_fixed_n_windows_valid"] = n_valid
    summary["lf_hf_fixed_n_windows_total"] = n_total
    summary["lf_hf_fixed_valid_pct"] = float(100.0 * n_valid / n_total) if n_total > 0 else np.nan
    summary["lf_hf_fixed_valid_min"] = float(n_valid * fixed_win_sec / 60.0)
    summary["lf_hf_fixed_total_min"] = float(n_total * fixed_win_sec / 60.0)
    summary["lf_hf_fixed_window_sec"] = float(fixed_win_sec)
    summary["lf_hf_fixed_hop_sec"] = float(fixed_hop_sec)

    if n_valid <= 0:
        summary["lf_fixed_mean"] = np.nan
        summary["lf_fixed_median"] = np.nan
        summary["hf_fixed_mean"] = np.nan
        summary["hf_fixed_median"] = np.nan
        summary["lf_hf_fixed_median"] = np.nan
        summary["lf_hf_fixed_mean"] = np.nan
        summary["lf"] = np.nan
        summary["hf"] = np.nan
        summary["lf_hf"] = np.nan
        return summary

    lf_vals = np.asarray(lfhf_fixed["lf_ms2"], dtype=float)[valid]
    hf_vals = np.asarray(lfhf_fixed["hf_ms2"], dtype=float)[valid]
    lfhf_vals = np.asarray(lfhf_fixed["lf_hf"], dtype=float)[valid]

    summary["lf_fixed_mean"] = float(np.nanmean(lf_vals))
    summary["lf_fixed_median"] = float(np.nanmedian(lf_vals))
    summary["hf_fixed_mean"] = float(np.nanmean(hf_vals))
    summary["hf_fixed_median"] = float(np.nanmedian(hf_vals))
    summary["lf_hf_fixed_median"] = float(np.nanmedian(lfhf_vals))
    summary["lf_hf_fixed_mean"] = float(np.nanmean(lfhf_vals))

    # Main exported LF/HF metrics now follow the fixed-window analysis.
    summary["lf"] = summary["lf_fixed_mean"]
    summary["hf"] = summary["hf_fixed_mean"]
    summary["lf_hf"] = summary["lf_hf_fixed_mean"]
    return summary


def compute_hrv_from_pat_signal(
    pat_signal: np.ndarray,
    fs: float,
    aux_df: Optional["pd.DataFrame"] = None,
    target_fs: Optional[float] = None,
    window_sec: Optional[float] = None,
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
        pat_signal,
        fs,
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

    summary = _build_summary_from_clean_rr_and_times(
        rr_mid_for_calc,
        rr_ms_for_calc,
        rmssd_windows_list_clean,
        fs_resample=fs_resample,
        max_gap_sec=max_gap_sec,
        min_freq_span_sec=min_freq_span_sec,
    )

    return t_hrv, rmssd_1hz_raw, rmssd_1hz_clean, summary


def summarize_hrv_from_rr(
    rr_mid_times_sec: np.ndarray,
    rr_ms: np.ndarray,
    duration_sec: float,
    aux_df: Optional["pd.DataFrame"],
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
    sdnn_clean = _calculate_sdnn_series(
        t_hrv,
        rr_mid_clean,
        rr_ms_clean,
        float(window_sec),
        min_span_sec=rmssd_min_span_sec,
        max_gap_sec=max_gap_sec,
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
                "sdnn_mean": np.nan,
                "sdnn_median": np.nan,
                "sdnn": np.nan,
                "lf": np.nan,
                "hf": np.nan,
                "lf_hf": np.nan,
                "lf_fixed_mean": np.nan,
                "lf_fixed_median": np.nan,
                "hf_fixed_mean": np.nan,
                "hf_fixed_median": np.nan,
                "lf_n_segments_used": 0,
                "lf_hf_fixed_median": np.nan,
                "lf_hf_fixed_mean": np.nan,
                "lf_hf_fixed_n_windows_valid": 0,
                "lf_hf_fixed_n_windows_total": 0,
                "lf_hf_fixed_valid_pct": np.nan,
                "lf_hf_fixed_valid_min": 0.0,
                "lf_hf_fixed_total_min": 0.0,
                "lf_hf_fixed_window_sec": float(fixed_win_sec),
                "lf_hf_fixed_hop_sec": float(fixed_hop_sec),
            }
        )
        return summary

    summary.update(
        _build_summary_from_clean_rr_and_times(
            rr_mid_clean,
            rr_ms_clean,
            rmssd_windows_list_clean,
            sdnn_clean,
            fs_resample=fs_resample_global,
            max_gap_sec=max_gap_sec,
            min_freq_span_sec=min_freq_span_sec,
        )
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
    return _apply_fixed_window_frequency_summary(
        summary,
        lfhf_fixed,
        fixed_win_sec=fixed_win_sec,
        fixed_hop_sec=fixed_hop_sec,
    )


def summarize_hrv_from_clean_rr(
    rr_mid_times_sec: np.ndarray,
    rr_ms: np.ndarray,
    duration_sec: float,
    *,
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

    rr_mid_times_sec = np.asarray(rr_mid_times_sec, dtype=float)
    rr_ms = np.asarray(rr_ms, dtype=float)
    duration_sec = float(duration_sec)

    t_hrv = np.arange(0, duration_sec, 1.0 / float(target_fs)) if duration_sec > 0 else np.array([], dtype=float)
    rmssd_clean, rmssd_windows_list_clean = _calculate_rmssd_series(
        t_hrv,
        rr_mid_times_sec,
        rr_ms,
        float(window_sec),
        max_gap_sec=max_gap_sec,
        min_span_sec=rmssd_min_span_sec,
    )
    sdnn_clean = _calculate_sdnn_series(
        t_hrv,
        rr_mid_times_sec,
        rr_ms,
        float(window_sec),
        min_span_sec=rmssd_min_span_sec,
        max_gap_sec=max_gap_sec,
    )

    summary: Dict[str, float] = {
        "rr_n_clean": int(rr_ms.size),
        "rmssd_nan_pct_clean": float(100.0 * np.mean(~np.isfinite(rmssd_clean))) if rmssd_clean.size else np.nan,
    }

    if rr_ms.size < 1:
        summary.update(
            {
                "rmssd_mean": np.nan,
                "rmssd_median": np.nan,
                "sdnn_mean": np.nan,
                "sdnn_median": np.nan,
                "sdnn": np.nan,
                "lf": np.nan,
                "hf": np.nan,
                "lf_hf": np.nan,
                "lf_fixed_mean": np.nan,
                "lf_fixed_median": np.nan,
                "hf_fixed_mean": np.nan,
                "hf_fixed_median": np.nan,
                "lf_n_segments_used": 0,
                "lf_hf_fixed_median": np.nan,
                "lf_hf_fixed_mean": np.nan,
                "lf_hf_fixed_n_windows_valid": 0,
                "lf_hf_fixed_n_windows_total": 0,
                "lf_hf_fixed_valid_pct": np.nan,
                "lf_hf_fixed_valid_min": 0.0,
                "lf_hf_fixed_total_min": 0.0,
                "lf_hf_fixed_window_sec": float(fixed_win_sec),
                "lf_hf_fixed_hop_sec": float(fixed_hop_sec),
            }
        )
        return summary

    summary.update(
        _build_summary_from_clean_rr_and_times(
            rr_mid_times_sec,
            rr_ms,
            rmssd_windows_list_clean,
            sdnn_clean,
            fs_resample=fs_resample_global,
            max_gap_sec=max_gap_sec,
            min_freq_span_sec=min_freq_span_sec,
        )
    )

    lfhf_fixed = _calculate_lfhf_fixed_windows(
        rr_mid=rr_mid_times_sec,
        rr_ms=rr_ms,
        duration_sec=duration_sec,
        window_sec=fixed_win_sec,
        hop_sec=fixed_hop_sec,
        fs_resample=fs_resample_global,
        max_gap_sec=max_gap_sec,
        min_rr=fixed_min_rr,
    )
    return _apply_fixed_window_frequency_summary(
        summary,
        lfhf_fixed,
        fixed_win_sec=fixed_win_sec,
        fixed_hop_sec=fixed_hop_sec,
    )


def summarize_hrv_halves_from_clean_rr(
    rr_mid_times_sec: np.ndarray,
    rr_ms: np.ndarray,
    split_sec: float,
    duration_sec: float,
    *,
    target_fs: Optional[float] = None,
    window_sec: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    rr_mid_times_sec = np.asarray(rr_mid_times_sec, dtype=float)
    rr_ms = np.asarray(rr_ms, dtype=float)
    split_sec = float(split_sec)
    duration_sec = float(duration_sec)

    first_mask = rr_mid_times_sec < split_sec
    second_mask = rr_mid_times_sec >= split_sec

    first_summary = summarize_hrv_from_clean_rr(
        rr_mid_times_sec[first_mask],
        rr_ms[first_mask],
        max(0.0, min(split_sec, duration_sec)),
        target_fs=target_fs,
        window_sec=window_sec,
    )
    second_summary = summarize_hrv_from_clean_rr(
        rr_mid_times_sec[second_mask] - split_sec,
        rr_ms[second_mask],
        max(0.0, duration_sec - split_sec),
        target_fs=target_fs,
        window_sec=window_sec,
    )
    return {"first_half": first_summary, "second_half": second_summary}


def compute_hrv_from_pat_signal_with_tv_metrics(
    pat_signal: np.ndarray,
    fs: float,
    aux_df: Optional["pd.DataFrame"] = None,
    target_fs: Optional[float] = None,
    window_sec: Optional[float] = None,
    tv_window_sec: Optional[float] = None,
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

    min_freq_span_sec_tv = float(
        getattr(config, "HRV_TV_MIN_FREQ_DOMAIN_SEC", getattr(config, "HRV_MIN_FREQ_DOMAIN_SEC"))
    )
    max_gap_sec_tv = float(
        getattr(config, "HRV_TV_MAX_TACHO_GAP_SEC", getattr(config, "HRV_MAX_RR_GAP_SEC"))
    )

    rr_sec_physio_clean, rr_mid_physio_clean, duration_sec = hr_metrics.extract_clean_rr_from_pat(
        pat_signal,
        fs,
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

    rmssd_1hz_raw, _rmssd_windows_list_raw = _calculate_rmssd_series(
        t_hrv,
        rr_mid_sleep,
        rr_ms_sleep,
        float(window_sec),
        max_gap_sec=max_gap_sec,
        min_span_sec=rmssd_min_span_sec,
    )

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
    sdnn_clean = _calculate_sdnn_series(
        t_hrv,
        rr_mid_for_calc,
        rr_ms_for_calc,
        float(window_sec),
        min_span_sec=rmssd_min_span_sec,
        max_gap_sec=max_gap_sec,
    )

    summary = _build_summary_from_clean_rr_and_times(
        rr_mid_for_calc,
        rr_ms_for_calc,
        rmssd_windows_list_clean,
        sdnn_clean,
        fs_resample=fs_resample_global,
        max_gap_sec=max_gap_sec,
        min_freq_span_sec=min_freq_span_sec,
    )

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
    summary = _apply_fixed_window_frequency_summary(
        summary,
        lfhf_fixed,
        fixed_win_sec=fixed_win_sec,
        fixed_hop_sec=fixed_hop_sec,
    )

    if rr_mid_for_calc.size == 0:
        tv = {
            "rmssd_ms": np.full_like(t_hrv, np.nan, dtype=float),
            "sdnn_ms_raw": np.full_like(t_hrv, np.nan, dtype=float),
            "sdnn_ms": np.full_like(t_hrv, np.nan, dtype=float),
            "lf_raw": np.full_like(t_hrv, np.nan, dtype=float),
            "lf": np.full_like(t_hrv, np.nan, dtype=float),
            "hf_raw": np.full_like(t_hrv, np.nan, dtype=float),
            "hf": np.full_like(t_hrv, np.nan, dtype=float),
            "lf_hf_raw": np.full_like(t_hrv, np.nan, dtype=float),
            "lf_hf": np.full_like(t_hrv, np.nan, dtype=float),
            "tv_window_sec": np.array([float(tv_window_sec)], dtype=float),
        }
        return t_hrv, rmssd_1hz_raw, rmssd_1hz_clean, summary, tv

    tv_raw = _calculate_hrv_windowed_series(
        t_grid=t_hrv,
        rr_mid=rr_mid_sleep,
        rr_ms=rr_ms_sleep,
        window_sec=float(tv_window_sec),
        fs_resample=float(fs_resample_tv),
        min_rr=int(min_rr_tv),
        min_span_sec=float(min_freq_span_sec_tv),
        max_gap_sec=float(max_gap_sec_tv),
    )

    tv_clean = _calculate_hrv_windowed_series(
        t_grid=t_hrv,
        rr_mid=rr_mid_for_calc,
        rr_ms=rr_ms_for_calc,
        window_sec=float(tv_window_sec),
        fs_resample=float(fs_resample_tv),
        min_rr=int(min_rr_tv),
        min_span_sec=float(min_freq_span_sec_tv),
        max_gap_sec=float(max_gap_sec_tv),
    )
    tv = {
        "rmssd_ms": np.asarray(rmssd_1hz_clean, dtype=float),
        "sdnn_ms_raw": np.asarray(tv_raw.get("sdnn_ms", np.full_like(t_hrv, np.nan, dtype=float)), dtype=float),
        "sdnn_ms": np.asarray(tv_clean.get("sdnn_ms", np.full_like(t_hrv, np.nan, dtype=float)), dtype=float),
        "lf_raw": np.asarray(tv_raw.get("lf", np.full_like(t_hrv, np.nan, dtype=float)), dtype=float),
        "lf": np.asarray(tv_clean.get("lf", np.full_like(t_hrv, np.nan, dtype=float)), dtype=float),
        "hf_raw": np.asarray(tv_raw.get("hf", np.full_like(t_hrv, np.nan, dtype=float)), dtype=float),
        "hf": np.asarray(tv_clean.get("hf", np.full_like(t_hrv, np.nan, dtype=float)), dtype=float),
        "lf_hf_raw": np.asarray(tv_raw.get("lf_hf", np.full_like(t_hrv, np.nan, dtype=float)), dtype=float),
        "lf_hf": np.asarray(tv_clean.get("lf_hf", np.full_like(t_hrv, np.nan, dtype=float)), dtype=float),
    }
    tv["tv_window_sec"] = np.array([float(tv_window_sec)], dtype=float)

    return t_hrv, rmssd_1hz_raw, rmssd_1hz_clean, summary, tv
