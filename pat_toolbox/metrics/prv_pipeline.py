from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np

from .. import config, masking
from . import hr as hr_metrics
from .prv_frequency_domain import (
    _calculate_lfhf_fixed_windows,
    _lf_hf_from_pr,
    _lf_hf_from_pr_segmented,
)
from .prv_time_domain import _calculate_rmssd_series, _rmssd, _sdnn, _time_domain_metrics_from_window

if TYPE_CHECKING:
    import pandas as pd


def _calculate_prv_windowed_series(
    t_grid: np.ndarray,
    pr_mid: np.ndarray,
    pr_ms: np.ndarray,
    window_sec: float,
    fs_resample: float,
    min_pr: int,
    *,
    min_span_sec: float = 120.0,
    max_gap_sec: float = 3.0,
) -> Dict[str, np.ndarray]:
    """
    Compute windowed PRV metrics on a time grid:
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

    if pr_mid.size == 0 or pr_ms.size == 0 or t_grid.size == 0:
        return out

    half = 0.5 * float(window_sec)

    n = pr_mid.size
    left = 0
    right = 0

    n_total = 0
    n_fail_time_domain = 0
    n_fail_psd = 0
    n_ok = 0

    min_intervals_td = int(getattr(config, "PRV_MIN_INTERVALS_PER_WINDOW", min_pr))
    min_cov_td = float(getattr(config, "PRV_MIN_WINDOW_COVERAGE", 0.0))
    min_span_td = float(getattr(config, "PRV_RMSSD_MIN_SPAN_SEC", min_span_sec))
    max_gap_td = float(getattr(config, "PRV_MAX_PR_GAP_SEC", max_gap_sec))

    for i, t in enumerate(t_grid):
        n_total += 1
        start = t - half
        end = t + half

        while left < n and pr_mid[left] < start:
            left += 1
        if right < left:
            right = left
        while right < n and pr_mid[right] < end:
            right += 1

        pr_win_ms = pr_ms[left:right]
        pr_mid_win = pr_mid[left:right]

        rmssd_i, sdnn_i = _time_domain_metrics_from_window(
            pr_mid_win,
            pr_win_ms,
            window_sec=float(window_sec),
            min_intervals=min_intervals_td,
            max_gap_sec=max_gap_td,
            min_span_sec=min_span_td,
            min_cov=min_cov_td,
        )
        if not (np.isfinite(rmssd_i) and np.isfinite(sdnn_i)):
            n_fail_time_domain += 1
            continue

        out["rmssd_ms"][i] = rmssd_i
        out["sdnn_ms"][i] = sdnn_i

        k = right - left
        if k < min_pr:
            continue
        span = float(pr_mid_win[-1] - pr_mid_win[0]) if pr_mid_win.size >= 2 else 0.0
        if span < float(min_span_sec):
            continue

        gaps = np.diff(pr_mid_win) if pr_mid_win.size >= 2 else np.array([])
        if gaps.size > 0 and np.any(gaps > float(max_gap_sec)):
            continue

        lf, hf, lf_hf = _lf_hf_from_pr(pr_win_ms, pr_mid_win, fs_resample=float(fs_resample))
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


def _subset_pr_by_sleep_and_events(
    pr_mid_times_sec: np.ndarray,
    pr_ms: np.ndarray,
    aux_df: Optional["pd.DataFrame"],
    *,
    include_set: Optional[set[int]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pr_mid_times_sec = np.asarray(pr_mid_times_sec, dtype=float)
    pr_ms = np.asarray(pr_ms, dtype=float)

    policy = masking.policy_from_config(
        include_stages=include_set,
        force_sleep=(include_set is not None),
    )
    bundle = masking.build_pr_mask_bundle(pr_mid_times_sec, aux_df, policy=policy)

    pr_mid_sleep = pr_mid_times_sec[bundle.sleep_keep]
    pr_ms_sleep = pr_ms[bundle.sleep_keep]

    pr_mid_clean = pr_mid_times_sec[bundle.combined_keep]
    pr_ms_clean = pr_ms[bundle.combined_keep]
    return pr_mid_sleep, pr_ms_sleep, pr_mid_clean, pr_ms_clean


def _calculate_sdnn_series(
    t_grid: np.ndarray,
    pr_mid: np.ndarray,
    pr_ms: np.ndarray,
    window_sec: float,
    *,
    min_span_sec: float,
    max_gap_sec: float,
) -> np.ndarray:
    out = np.full_like(t_grid, np.nan, dtype=float)
    if pr_mid.size == 0 or pr_ms.size == 0 or t_grid.size == 0:
        return out

    half = 0.5 * float(window_sec)
    left = 0
    right = 0
    n = pr_mid.size
    min_intervals_td = int(getattr(config, "PRV_MIN_INTERVALS_PER_WINDOW", 0))
    min_cov_td = float(getattr(config, "PRV_MIN_WINDOW_COVERAGE", 0.0))
    min_span_td = float(getattr(config, "PRV_RMSSD_MIN_SPAN_SEC", min_span_sec))
    max_gap_td = float(getattr(config, "PRV_MAX_PR_GAP_SEC", max_gap_sec))

    for i, t in enumerate(t_grid):
        start = t - half
        end = t + half
        while left < n and pr_mid[left] < start:
            left += 1
        if right < left:
            right = left
        while right < n and pr_mid[right] < end:
            right += 1

        pr_win_ms = pr_ms[left:right]
        pr_mid_win = pr_mid[left:right]
        rmssd_i, sdnn_i = _time_domain_metrics_from_window(
            pr_mid_win,
            pr_win_ms,
            window_sec=float(window_sec),
            min_intervals=min_intervals_td,
            max_gap_sec=max_gap_td,
            min_span_sec=min_span_td,
            min_cov=min_cov_td,
        )
        if not np.isfinite(sdnn_i):
            continue
        out[i] = sdnn_i
    return out


def _build_summary_from_clean_pr_and_times(
    pr_mid_clean: np.ndarray,
    pr_ms_clean: np.ndarray,
    rmssd_windows_list_clean: list[float],
    sdnn_series_clean: np.ndarray,
    *,
    fs_resample: float,
    max_gap_sec: float,
    min_freq_span_sec: float,
) -> Dict[str, float]:
    sdnn_ms = _sdnn(pr_ms_clean)
    lf, hf, lf_hf, lf_info = _lf_hf_from_pr_segmented(
        pr_ms_clean,
        pr_mid_clean,
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
            "rmssd_mean": float(_rmssd(pr_ms_clean)),
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


def _fixed_window_plot_series(
    lfhf_fixed: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    centers = np.asarray(lfhf_fixed.get("t_win_center_sec", np.array([], dtype=float)), dtype=float)
    return {
        "spectral_t_sec": centers,
        "lf_fixed": np.asarray(lfhf_fixed.get("lf_ms2", np.array([], dtype=float)), dtype=float),
        "hf_fixed": np.asarray(lfhf_fixed.get("hf_ms2", np.array([], dtype=float)), dtype=float),
        "lf_hf_fixed": np.asarray(lfhf_fixed.get("lf_hf", np.array([], dtype=float)), dtype=float),
    }


def compute_prv_from_pat_signal(
    pat_signal: np.ndarray,
    fs: float,
    aux_df: Optional["pd.DataFrame"] = None,
    target_fs: Optional[float] = None,
    window_sec: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Dict[str, float]]]:
    """
    Compute PRV from PAT signal using the same PR extraction/cleaning as HR.

    Outputs:
      - t_prv
      - rmssd_1hz_raw: after sleep masking, before event exclusion
      - rmssd_1hz_clean: after sleep masking + event exclusion
      - summary: global PRV metrics on clean PR
    """
    if target_fs is None:
        target_fs = float(getattr(config, "PRV_TARGET_FS_HZ"))
    if window_sec is None:
        window_sec = float(getattr(config, "PRV_WINDOW_SEC"))

    if fs <= 0:
        raise ValueError("Sampling frequency fs must be positive.")

    n_samples = len(pat_signal)
    if pat_signal.ndim != 1 or n_samples == 0:
        raise ValueError("PAT signal must be a non-empty 1D array.")

    max_gap_sec = float(getattr(config, "PRV_MAX_PR_GAP_SEC"))
    rmssd_min_span_sec = float(getattr(config, "PRV_RMSSD_MIN_SPAN_SEC"))
    fs_resample = float(getattr(config, "PRV_TACHO_RESAMPLE_HZ"))
    min_freq_span_sec = float(getattr(config, "PRV_MIN_FREQ_DOMAIN_SEC"))

    pr_sec_physio_clean, pr_mid_physio_clean, duration_sec = hr_metrics.extract_clean_pr_from_pat(
        pat_signal,
        fs,
    )

    t_prv = np.arange(0, duration_sec, 1.0 / float(target_fs))

    if pr_sec_physio_clean.size < 1:
        nan_array = np.full_like(t_prv, fill_value=np.nan, dtype=float)
        return t_prv, nan_array, nan_array, None

    pr_ms_physio_clean = pr_sec_physio_clean * 1000.0

    bundle = masking.build_pr_mask_bundle(pr_mid_physio_clean, aux_df)
    pr_mid_sleep = pr_mid_physio_clean[bundle.sleep_keep]
    pr_ms_sleep = pr_ms_physio_clean[bundle.sleep_keep]

    rmssd_1hz_raw, _rmssd_windows_list_raw = _calculate_rmssd_series(
        t_prv,
        pr_mid_sleep,
        pr_ms_sleep,
        float(window_sec),
        max_gap_sec=max_gap_sec,
        min_span_sec=rmssd_min_span_sec,
    )

    if pr_ms_sleep.size < 1:
        nan_array = np.full_like(t_prv, fill_value=np.nan, dtype=float)
        return t_prv, rmssd_1hz_raw, nan_array, None

    pr_mid_for_calc = pr_mid_physio_clean[bundle.combined_keep]
    pr_ms_for_calc = pr_ms_physio_clean[bundle.combined_keep]

    if pr_ms_for_calc.size < 1:
        nan_array = np.full_like(t_prv, fill_value=np.nan, dtype=float)
        return t_prv, rmssd_1hz_raw, nan_array, None

    rmssd_1hz_clean, rmssd_windows_list_clean = _calculate_rmssd_series(
        t_prv,
        pr_mid_for_calc,
        pr_ms_for_calc,
        float(window_sec),
        max_gap_sec=max_gap_sec,
        min_span_sec=rmssd_min_span_sec,
    )

    summary = _build_summary_from_clean_pr_and_times(
        pr_mid_for_calc,
        pr_ms_for_calc,
        rmssd_windows_list_clean,
        fs_resample=fs_resample,
        max_gap_sec=max_gap_sec,
        min_freq_span_sec=min_freq_span_sec,
    )

    return t_prv, rmssd_1hz_raw, rmssd_1hz_clean, summary


def summarize_prv_from_pr(
    pr_mid_times_sec: np.ndarray,
    pr_ms: np.ndarray,
    duration_sec: float,
    aux_df: Optional["pd.DataFrame"],
    *,
    include_set: Optional[set[int]] = None,
    target_fs: Optional[float] = None,
    window_sec: Optional[float] = None,
) -> Dict[str, float]:
    if target_fs is None:
        target_fs = float(getattr(config, "PRV_TARGET_FS_HZ"))
    if window_sec is None:
        window_sec = float(getattr(config, "PRV_WINDOW_SEC"))

    max_gap_sec = float(getattr(config, "PRV_MAX_PR_GAP_SEC"))
    rmssd_min_span_sec = float(getattr(config, "PRV_RMSSD_MIN_SPAN_SEC"))
    fs_resample_global = float(getattr(config, "PRV_TACHO_RESAMPLE_HZ"))
    min_freq_span_sec = float(getattr(config, "PRV_MIN_FREQ_DOMAIN_SEC"))
    fixed_win_sec = float(getattr(config, "PRV_LFHF_FIXED_WINDOW_SEC", 300.0))
    fixed_hop_sec = float(getattr(config, "PRV_LFHF_FIXED_HOP_SEC", fixed_win_sec))
    fixed_min_pr = int(getattr(config, "PRV_LFHF_FIXED_MIN_PR", 0))

    pr_mid_sleep, pr_ms_sleep, pr_mid_clean, pr_ms_clean = _subset_pr_by_sleep_and_events(
        pr_mid_times_sec,
        pr_ms,
        aux_df,
        include_set=include_set,
    )

    t_prv = np.arange(0, float(duration_sec), 1.0 / float(target_fs))
    rmssd_raw, _ = _calculate_rmssd_series(
        t_prv,
        pr_mid_sleep,
        pr_ms_sleep,
        float(window_sec),
        max_gap_sec=max_gap_sec,
        min_span_sec=rmssd_min_span_sec,
    )
    rmssd_clean, rmssd_windows_list_clean = _calculate_rmssd_series(
        t_prv,
        pr_mid_clean,
        pr_ms_clean,
        float(window_sec),
        max_gap_sec=max_gap_sec,
        min_span_sec=rmssd_min_span_sec,
    )
    sdnn_clean = _calculate_sdnn_series(
        t_prv,
        pr_mid_clean,
        pr_ms_clean,
        float(window_sec),
        min_span_sec=rmssd_min_span_sec,
        max_gap_sec=max_gap_sec,
    )

    summary: Dict[str, float] = {
        "pr_n_sleep": int(pr_ms_sleep.size),
        "pr_n_clean": int(pr_ms_clean.size),
        "rmssd_nan_pct_raw": float(100.0 * np.mean(~np.isfinite(rmssd_raw))) if rmssd_raw.size else np.nan,
        "rmssd_nan_pct_clean": float(100.0 * np.mean(~np.isfinite(rmssd_clean))) if rmssd_clean.size else np.nan,
    }

    if pr_ms_clean.size < 1:
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
        _build_summary_from_clean_pr_and_times(
            pr_mid_clean,
            pr_ms_clean,
            rmssd_windows_list_clean,
            sdnn_clean,
            fs_resample=fs_resample_global,
            max_gap_sec=max_gap_sec,
            min_freq_span_sec=min_freq_span_sec,
        )
    )

    lfhf_fixed = _calculate_lfhf_fixed_windows(
        pr_mid=pr_mid_clean,
        pr_ms=pr_ms_clean,
        duration_sec=float(duration_sec),
        window_sec=fixed_win_sec,
        hop_sec=fixed_hop_sec,
        fs_resample=fs_resample_global,
        max_gap_sec=max_gap_sec,
        min_pr=fixed_min_pr,
    )
    return _apply_fixed_window_frequency_summary(
        summary,
        lfhf_fixed,
        fixed_win_sec=fixed_win_sec,
        fixed_hop_sec=fixed_hop_sec,
    )


def summarize_prv_from_clean_pr(
    pr_mid_times_sec: np.ndarray,
    pr_ms: np.ndarray,
    duration_sec: float,
    *,
    target_fs: Optional[float] = None,
    window_sec: Optional[float] = None,
) -> Dict[str, float]:
    if target_fs is None:
        target_fs = float(getattr(config, "PRV_TARGET_FS_HZ"))
    if window_sec is None:
        window_sec = float(getattr(config, "PRV_WINDOW_SEC"))

    max_gap_sec = float(getattr(config, "PRV_MAX_PR_GAP_SEC"))
    rmssd_min_span_sec = float(getattr(config, "PRV_RMSSD_MIN_SPAN_SEC"))
    fs_resample_global = float(getattr(config, "PRV_TACHO_RESAMPLE_HZ"))
    min_freq_span_sec = float(getattr(config, "PRV_MIN_FREQ_DOMAIN_SEC"))
    fixed_win_sec = float(getattr(config, "PRV_LFHF_FIXED_WINDOW_SEC", 300.0))
    fixed_hop_sec = float(getattr(config, "PRV_LFHF_FIXED_HOP_SEC", fixed_win_sec))
    fixed_min_pr = int(getattr(config, "PRV_LFHF_FIXED_MIN_PR", 0))

    pr_mid_times_sec = np.asarray(pr_mid_times_sec, dtype=float)
    pr_ms = np.asarray(pr_ms, dtype=float)
    duration_sec = float(duration_sec)

    t_prv = np.arange(0, duration_sec, 1.0 / float(target_fs)) if duration_sec > 0 else np.array([], dtype=float)
    rmssd_clean, rmssd_windows_list_clean = _calculate_rmssd_series(
        t_prv,
        pr_mid_times_sec,
        pr_ms,
        float(window_sec),
        max_gap_sec=max_gap_sec,
        min_span_sec=rmssd_min_span_sec,
    )
    sdnn_clean = _calculate_sdnn_series(
        t_prv,
        pr_mid_times_sec,
        pr_ms,
        float(window_sec),
        min_span_sec=rmssd_min_span_sec,
        max_gap_sec=max_gap_sec,
    )

    summary: Dict[str, float] = {
        "pr_n_clean": int(pr_ms.size),
        "rmssd_nan_pct_clean": float(100.0 * np.mean(~np.isfinite(rmssd_clean))) if rmssd_clean.size else np.nan,
    }

    if pr_ms.size < 1:
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
        _build_summary_from_clean_pr_and_times(
            pr_mid_times_sec,
            pr_ms,
            rmssd_windows_list_clean,
            sdnn_clean,
            fs_resample=fs_resample_global,
            max_gap_sec=max_gap_sec,
            min_freq_span_sec=min_freq_span_sec,
        )
    )

    lfhf_fixed = _calculate_lfhf_fixed_windows(
        pr_mid=pr_mid_times_sec,
        pr_ms=pr_ms,
        duration_sec=duration_sec,
        window_sec=fixed_win_sec,
        hop_sec=fixed_hop_sec,
        fs_resample=fs_resample_global,
        max_gap_sec=max_gap_sec,
        min_pr=fixed_min_pr,
    )
    return _apply_fixed_window_frequency_summary(
        summary,
        lfhf_fixed,
        fixed_win_sec=fixed_win_sec,
        fixed_hop_sec=fixed_hop_sec,
    )


def summarize_prv_halves_from_clean_pr(
    pr_mid_times_sec: np.ndarray,
    pr_ms: np.ndarray,
    split_sec: float,
    duration_sec: float,
    *,
    target_fs: Optional[float] = None,
    window_sec: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    pr_mid_times_sec = np.asarray(pr_mid_times_sec, dtype=float)
    pr_ms = np.asarray(pr_ms, dtype=float)
    split_sec = float(split_sec)
    duration_sec = float(duration_sec)

    first_mask = pr_mid_times_sec < split_sec
    second_mask = pr_mid_times_sec >= split_sec

    first_summary = summarize_prv_from_clean_pr(
        pr_mid_times_sec[first_mask],
        pr_ms[first_mask],
        max(0.0, min(split_sec, duration_sec)),
        target_fs=target_fs,
        window_sec=window_sec,
    )
    second_summary = summarize_prv_from_clean_pr(
        pr_mid_times_sec[second_mask] - split_sec,
        pr_ms[second_mask],
        max(0.0, duration_sec - split_sec),
        target_fs=target_fs,
        window_sec=window_sec,
    )
    return {"first_half": first_summary, "second_half": second_summary}


def compute_prv_from_pat_signal_with_tv_metrics(
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
    Same as compute_prv_from_pat_signal, but also returns time-varying PRV metrics.
    """
    if target_fs is None:
        target_fs = float(getattr(config, "PRV_TARGET_FS_HZ"))
    if window_sec is None:
        window_sec = float(getattr(config, "PRV_WINDOW_SEC"))
    if tv_window_sec is None:
        tv_window_sec = float(getattr(config, "PRV_TV_WINDOW_SEC"))

    if fs <= 0:
        raise ValueError("Sampling frequency fs must be positive.")
    n_samples = len(pat_signal)
    if pat_signal.ndim != 1 or n_samples == 0:
        raise ValueError("PAT signal must be a non-empty 1D array.")

    max_gap_sec = float(getattr(config, "PRV_MAX_PR_GAP_SEC"))
    rmssd_min_span_sec = float(getattr(config, "PRV_RMSSD_MIN_SPAN_SEC"))

    fs_resample_global = float(getattr(config, "PRV_TACHO_RESAMPLE_HZ"))
    min_freq_span_sec = float(getattr(config, "PRV_MIN_FREQ_DOMAIN_SEC"))

    fs_resample_tv = float(getattr(config, "PRV_TV_TACHO_RESAMPLE_HZ"))
    min_pr_tv = int(getattr(config, "PRV_TV_MIN_PR_PER_WINDOW"))

    min_freq_span_sec_tv = float(
        getattr(config, "PRV_TV_MIN_FREQ_DOMAIN_SEC", getattr(config, "PRV_MIN_FREQ_DOMAIN_SEC"))
    )
    max_gap_sec_tv = float(
        getattr(config, "PRV_TV_MAX_TACHO_GAP_SEC", getattr(config, "PRV_MAX_PR_GAP_SEC"))
    )

    pr_sec_physio_clean, pr_mid_physio_clean, duration_sec = hr_metrics.extract_clean_pr_from_pat(
        pat_signal,
        fs,
    )
    t_prv = np.arange(0, duration_sec, 1.0 / float(target_fs))

    if pr_sec_physio_clean.size < 1:
        nan_array = np.full_like(t_prv, fill_value=np.nan, dtype=float)
        return t_prv, nan_array, nan_array, None, None

    pr_ms_physio_clean = pr_sec_physio_clean * 1000.0

    bundle = masking.build_pr_mask_bundle(pr_mid_physio_clean, aux_df)
    pr_mid_sleep = pr_mid_physio_clean[bundle.sleep_keep]
    pr_ms_sleep = pr_ms_physio_clean[bundle.sleep_keep]

    if pr_ms_sleep.size < 1:
        nan_array = np.full_like(t_prv, fill_value=np.nan, dtype=float)
        return t_prv, nan_array, nan_array, None, None

    rmssd_1hz_raw, _rmssd_windows_list_raw = _calculate_rmssd_series(
        t_prv,
        pr_mid_sleep,
        pr_ms_sleep,
        float(window_sec),
        max_gap_sec=max_gap_sec,
        min_span_sec=rmssd_min_span_sec,
    )

    pr_mid_for_calc = pr_mid_physio_clean[bundle.combined_keep]
    pr_ms_for_calc = pr_ms_physio_clean[bundle.combined_keep]

    if pr_ms_for_calc.size < 1:
        nan_array = np.full_like(t_prv, fill_value=np.nan, dtype=float)
        return t_prv, rmssd_1hz_raw, nan_array, None, None

    rmssd_1hz_clean, rmssd_windows_list_clean = _calculate_rmssd_series(
        t_prv,
        pr_mid_for_calc,
        pr_ms_for_calc,
        float(window_sec),
        max_gap_sec=max_gap_sec,
        min_span_sec=rmssd_min_span_sec,
    )
    sdnn_clean = _calculate_sdnn_series(
        t_prv,
        pr_mid_for_calc,
        pr_ms_for_calc,
        float(window_sec),
        min_span_sec=rmssd_min_span_sec,
        max_gap_sec=max_gap_sec,
    )

    # Keep clean time-domain traces visible only where the center timepoint itself
    # survives the final-analysis mask, rather than allowing window support from
    # nearby included data to populate excluded center times.
    center_time_bundle = masking.build_mask_bundle(t_prv, aux_df)
    center_keep = np.asarray(center_time_bundle.combined_keep, dtype=bool)
    if center_keep.size == t_prv.size:
        rmssd_1hz_clean = np.asarray(rmssd_1hz_clean, dtype=float)
        sdnn_clean = np.asarray(sdnn_clean, dtype=float)
        rmssd_1hz_clean[~center_keep] = np.nan
        sdnn_clean[~center_keep] = np.nan

    # Keep displayed time-domain coverage aligned across RMSSD and SDNN. RMSSD has
    # additional robustness rejection on successive PR differences; when that
    # invalidates a window, we hide the corresponding SDNN estimate as well so the
    # two time-domain traces reflect the same accepted windows.
    rmssd_clean_valid = np.isfinite(np.asarray(rmssd_1hz_clean, dtype=float))
    sdnn_clean = np.asarray(sdnn_clean, dtype=float)
    sdnn_clean[~rmssd_clean_valid] = np.nan

    summary = _build_summary_from_clean_pr_and_times(
        pr_mid_for_calc,
        pr_ms_for_calc,
        rmssd_windows_list_clean,
        sdnn_clean,
        fs_resample=fs_resample_global,
        max_gap_sec=max_gap_sec,
        min_freq_span_sec=min_freq_span_sec,
    )

    fixed_win_sec = float(getattr(config, "PRV_LFHF_FIXED_WINDOW_SEC", 300.0))
    fixed_hop_sec = float(getattr(config, "PRV_LFHF_FIXED_HOP_SEC", fixed_win_sec))
    fs_resample_fixed = float(getattr(config, "PRV_TACHO_RESAMPLE_HZ", 4.0))
    fixed_min_pr = int(getattr(config, "PRV_LFHF_FIXED_MIN_PR", 0))

    lfhf_fixed = _calculate_lfhf_fixed_windows(
        pr_mid=pr_mid_for_calc,
        pr_ms=pr_ms_for_calc,
        duration_sec=duration_sec,
        window_sec=fixed_win_sec,
        hop_sec=fixed_hop_sec,
        fs_resample=fs_resample_fixed,
        max_gap_sec=max_gap_sec,
        min_pr=fixed_min_pr,
    )
    summary = _apply_fixed_window_frequency_summary(
        summary,
        lfhf_fixed,
        fixed_win_sec=fixed_win_sec,
        fixed_hop_sec=fixed_hop_sec,
    )
    lfhf_fixed_raw = _calculate_lfhf_fixed_windows(
        pr_mid=pr_mid_sleep,
        pr_ms=pr_ms_sleep,
        duration_sec=duration_sec,
        window_sec=fixed_win_sec,
        hop_sec=fixed_hop_sec,
        fs_resample=fs_resample_fixed,
        max_gap_sec=max_gap_sec,
        min_pr=fixed_min_pr,
    )

    if pr_mid_for_calc.size == 0:
        tv = {
            "rmssd_ms": np.full_like(t_prv, np.nan, dtype=float),
            "sdnn_ms_raw": np.full_like(t_prv, np.nan, dtype=float),
            "sdnn_ms": np.full_like(t_prv, np.nan, dtype=float),
            "lf_raw": np.full_like(t_prv, np.nan, dtype=float),
            "lf": np.full_like(t_prv, np.nan, dtype=float),
            "hf_raw": np.full_like(t_prv, np.nan, dtype=float),
            "hf": np.full_like(t_prv, np.nan, dtype=float),
            "lf_hf_raw": np.full_like(t_prv, np.nan, dtype=float),
            "lf_hf": np.full_like(t_prv, np.nan, dtype=float),
            "spectral_t_sec": np.array([], dtype=float),
            "lf_fixed_raw": np.array([], dtype=float),
            "lf_fixed": np.array([], dtype=float),
            "hf_fixed_raw": np.array([], dtype=float),
            "hf_fixed": np.array([], dtype=float),
            "lf_hf_fixed_raw": np.array([], dtype=float),
            "lf_hf_fixed": np.array([], dtype=float),
            "spectral_window_sec": np.array([float(fixed_win_sec)], dtype=float),
            "spectral_hop_sec": np.array([float(fixed_hop_sec)], dtype=float),
            "tv_window_sec": np.array([float(tv_window_sec)], dtype=float),
        }
        return t_prv, rmssd_1hz_raw, rmssd_1hz_clean, summary, tv

    tv_raw = _calculate_prv_windowed_series(
        t_grid=t_prv,
        pr_mid=pr_mid_sleep,
        pr_ms=pr_ms_sleep,
        window_sec=float(tv_window_sec),
        fs_resample=float(fs_resample_tv),
        min_pr=int(min_pr_tv),
        min_span_sec=float(min_freq_span_sec_tv),
        max_gap_sec=float(max_gap_sec_tv),
    )

    tv_clean = _calculate_prv_windowed_series(
        t_grid=t_prv,
        pr_mid=pr_mid_for_calc,
        pr_ms=pr_ms_for_calc,
        window_sec=float(tv_window_sec),
        fs_resample=float(fs_resample_tv),
        min_pr=int(min_pr_tv),
        min_span_sec=float(min_freq_span_sec_tv),
        max_gap_sec=float(max_gap_sec_tv),
    )
    sdnn_tv_raw = np.asarray(tv_raw.get("sdnn_ms", np.full_like(t_prv, np.nan, dtype=float)), dtype=float)
    rmssd_raw_valid = np.isfinite(np.asarray(rmssd_1hz_raw, dtype=float))
    sdnn_tv_raw[~rmssd_raw_valid] = np.nan

    sdnn_tv_clean = np.asarray(tv_clean.get("sdnn_ms", np.full_like(t_prv, np.nan, dtype=float)), dtype=float)
    if center_keep.size == t_prv.size:
        sdnn_tv_clean[~center_keep] = np.nan
    sdnn_tv_clean[~rmssd_clean_valid] = np.nan

    tv = {
        "rmssd_ms": np.asarray(rmssd_1hz_clean, dtype=float),
        "sdnn_ms_raw": sdnn_tv_raw,
        "sdnn_ms": sdnn_tv_clean,
        "lf_raw": np.asarray(tv_raw.get("lf", np.full_like(t_prv, np.nan, dtype=float)), dtype=float),
        "lf": np.asarray(tv_clean.get("lf", np.full_like(t_prv, np.nan, dtype=float)), dtype=float),
        "hf_raw": np.asarray(tv_raw.get("hf", np.full_like(t_prv, np.nan, dtype=float)), dtype=float),
        "hf": np.asarray(tv_clean.get("hf", np.full_like(t_prv, np.nan, dtype=float)), dtype=float),
        "lf_hf_raw": np.asarray(tv_raw.get("lf_hf", np.full_like(t_prv, np.nan, dtype=float)), dtype=float),
        "lf_hf": np.asarray(tv_clean.get("lf_hf", np.full_like(t_prv, np.nan, dtype=float)), dtype=float),
    }
    tv.update(_fixed_window_plot_series(lfhf_fixed_raw))
    tv["lf_fixed_raw"] = np.asarray(tv.pop("lf_fixed"), dtype=float)
    tv["hf_fixed_raw"] = np.asarray(tv.pop("hf_fixed"), dtype=float)
    tv["lf_hf_fixed_raw"] = np.asarray(tv.pop("lf_hf_fixed"), dtype=float)
    tv.update(_fixed_window_plot_series(lfhf_fixed))
    tv["spectral_window_sec"] = np.array([float(fixed_win_sec)], dtype=float)
    tv["spectral_hop_sec"] = np.array([float(fixed_hop_sec)], dtype=float)
    tv["tv_window_sec"] = np.array([float(tv_window_sec)], dtype=float)

    return t_prv, rmssd_1hz_raw, rmssd_1hz_clean, summary, tv
