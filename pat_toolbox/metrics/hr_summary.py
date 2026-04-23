from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .. import config, features, paths
from ..io.aux_events import compute_sleep_timing_from_aux


def append_hr_hrv_summary(
    edf_path: Path,
    hrv_summary: Optional[Dict[str, float]] = None,
    mayer_peak_freq: Optional[float] = None,
    resp_peak_freq: Optional[float] = None,
    *,
    t_hr: Optional[np.ndarray] = None,
    hr_calc: Optional[np.ndarray] = None,
    t_hrv: Optional[np.ndarray] = None,
    hrv_clean: Optional[np.ndarray] = None,
    hrv_raw: Optional[np.ndarray] = None,
    hrv_tv: Optional[Dict[str, np.ndarray]] = None,
    hrv_mask_info: Optional[Dict[str, object]] = None,
    hrv_midpoint_halves: Optional[Dict[str, Dict[str, float]]] = None,
    aux_df: Optional[Any] = None,
    psd_features: Optional[Dict[str, float]] = None,
    pat_burden: Optional[float] = None,
    pat_burden_diag: Optional[dict] = None,
    sleep_combo_summaries: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Path:
    """
    Append one PAT-only summary row with:
      - HRV summary
      - spectral peaks / PSD features
      - PAT HR / HRV valid coverage
      - aux event counts
      - sleep-stage masking stats
      - PAT burden
    """

    def _sample_dt_sec(t: Optional[np.ndarray], default_fs: float) -> float:
        if t is not None:
            tt = np.asarray(t, dtype=float)
            if tt.size >= 2:
                dt = np.diff(tt)
                dt = dt[np.isfinite(dt) & (dt > 0)]
                if dt.size > 0:
                    return float(np.median(dt))
        fs = float(default_fs)
        return 1.0 / fs if fs > 0 else 1.0

    def _coverage_stats(
        x: Optional[np.ndarray],
        *,
        t: Optional[np.ndarray] = None,
        default_fs: float = 1.0,
    ) -> Dict[str, float]:
        out = {
            "valid_pct": np.nan,
            "valid_min": np.nan,
            "total_min": np.nan,
            "n_valid": np.nan,
            "n_total": np.nan,
        }
        if x is None:
            return out
        arr = np.asarray(x, dtype=float)
        if arr.size == 0:
            out.update({
                "valid_pct": 0.0,
                "valid_min": 0.0,
                "total_min": 0.0,
                "n_valid": 0.0,
                "n_total": 0.0,
            })
            return out
        ok = np.isfinite(arr)
        n_valid = float(np.count_nonzero(ok))
        n_total = float(arr.size)
        dt_sec = _sample_dt_sec(t, default_fs)
        out["valid_pct"] = 100.0 * n_valid / n_total if n_total > 0 else np.nan
        out["valid_min"] = (n_valid * dt_sec) / 60.0
        out["total_min"] = (n_total * dt_sec) / 60.0
        out["n_valid"] = n_valid
        out["n_total"] = n_total
        return out

    def _hrv_tv_csv_prefix(key: str) -> str:
        mapping = {
            "sdnn_ms_raw": "hrv_tv_sdnn_pre_final_exclusion",
            "sdnn_ms": "hrv_tv_sdnn_final_analysis",
            "lf_raw": "hrv_tv_lf_pre_final_exclusion",
            "lf": "hrv_tv_lf_final_analysis",
            "hf_raw": "hrv_tv_hf_pre_final_exclusion",
            "hf": "hrv_tv_hf_final_analysis",
            "lf_hf_raw": "hrv_tv_lf_hf_pre_final_exclusion",
            "lf_hf": "hrv_tv_lf_hf_final_analysis",
        }
        return mapping.get(key, f"hrv_tv_{key}")

    def _mask_breakdown_stats(
        t: Optional[np.ndarray],
        mask_info: Optional[Dict[str, object]],
    ) -> Dict[str, float]:
        out = {
            "selected_policy_min": np.nan,
            "clean_kept_min": np.nan,
            "clean_kept_pct_of_selected": np.nan,
            "excluded_total_min": np.nan,
            "excluded_total_pct_of_selected": np.nan,
            "excluded_apnea_only_min": np.nan,
            "excluded_apnea_only_pct_of_selected": np.nan,
            "excluded_quality_only_min": np.nan,
            "excluded_quality_only_pct_of_selected": np.nan,
            "excluded_desat_only_min": np.nan,
            "excluded_desat_only_pct_of_selected": np.nan,
            "excluded_overlap_min": np.nan,
            "excluded_overlap_pct_of_selected": np.nan,
        }
        if t is None or not mask_info:
            return out

        sleep_keep = mask_info.get("sleep_keep")
        apnea_keep = mask_info.get("apnea_keep")
        quality_keep = mask_info.get("quality_keep")
        desat_keep = mask_info.get("desat_keep")
        combined_keep = mask_info.get("combined_keep")
        required = [sleep_keep, apnea_keep, quality_keep, desat_keep, combined_keep]
        if not all(isinstance(v, np.ndarray) and np.size(v) == np.size(t) for v in required):
            return out

        dt_sec = _sample_dt_sec(t, float(getattr(config, "HRV_TARGET_FS_HZ", 1.0)))
        sleep_keep = np.asarray(sleep_keep, dtype=bool)
        apnea_keep = np.asarray(apnea_keep, dtype=bool)
        quality_keep = np.asarray(quality_keep, dtype=bool)
        desat_keep = np.asarray(desat_keep, dtype=bool)
        combined_keep = np.asarray(combined_keep, dtype=bool)

        selected_n = int(np.count_nonzero(sleep_keep))
        if selected_n <= 0:
            return out

        apnea_excl = sleep_keep & (~apnea_keep)
        quality_excl = sleep_keep & (~quality_keep)
        desat_excl = sleep_keep & (~desat_keep)
        excl_count = apnea_excl.astype(int) + quality_excl.astype(int) + desat_excl.astype(int)
        only_apnea = (excl_count == 1) & apnea_excl
        only_quality = (excl_count == 1) & quality_excl
        only_desat = (excl_count == 1) & desat_excl
        overlap = excl_count > 1
        excluded_total = sleep_keep & (~combined_keep)

        def _min(mask: np.ndarray) -> float:
            return float(np.count_nonzero(mask) * dt_sec / 60.0)

        def _pct(mask: np.ndarray) -> float:
            return float(100.0 * np.count_nonzero(mask) / selected_n)

        out["selected_policy_min"] = _min(sleep_keep)
        out["clean_kept_min"] = _min(combined_keep)
        out["clean_kept_pct_of_selected"] = _pct(combined_keep)
        out["excluded_total_min"] = _min(excluded_total)
        out["excluded_total_pct_of_selected"] = _pct(excluded_total)
        out["excluded_apnea_only_min"] = _min(only_apnea)
        out["excluded_apnea_only_pct_of_selected"] = _pct(only_apnea)
        out["excluded_quality_only_min"] = _min(only_quality)
        out["excluded_quality_only_pct_of_selected"] = _pct(only_quality)
        out["excluded_desat_only_min"] = _min(only_desat)
        out["excluded_desat_only_pct_of_selected"] = _pct(only_desat)
        out["excluded_overlap_min"] = _min(overlap)
        out["excluded_overlap_pct_of_selected"] = _pct(overlap)
        return out

    def _fmt6(x: Optional[float]) -> str:
        if x is None:
            return ""
        try:
            xf = float(x)
            if not np.isfinite(xf):
                return ""
            return f"{xf:.6f}"
        except Exception:
            return ""

    def _fmt1(x: Optional[float]) -> str:
        if x is None:
            return ""
        try:
            xf = float(x)
            if not np.isfinite(xf):
                return ""
            return f"{xf:.1f}"
        except Exception:
            return ""

    def _fmt_sci(x: Optional[float]) -> str:
        if x is None:
            return ""
        try:
            xf = float(x)
            if not np.isfinite(xf):
                return ""
            return f"{xf:.4e}"
        except Exception:
            return ""

    def _count_flags(df: Any, col: str) -> tuple[Optional[int], Optional[float]]:
        if df is None:
            return (None, None)
        try:
            if not hasattr(df, "columns") or col not in df.columns:
                return (None, None)
            total = int(len(df))
            if total <= 0:
                return (0, 0.0)
            c = int(df[col].fillna(0).astype(int).sum())
            pct = 100.0 * c / total
            return (c, pct)
        except Exception:
            return (None, None)

    def _finite_stats(x: Optional[np.ndarray]) -> Dict[str, float]:
        out = {
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "nan_pct": np.nan,
            "n_used": np.nan,
        }
        if x is None:
            return out
        arr = np.asarray(x, dtype=float)
        if arr.size == 0:
            out["nan_pct"] = 100.0
            out["n_used"] = 0.0
            return out
        ok = np.isfinite(arr)
        out["nan_pct"] = float(100.0 * np.mean(~ok))
        out["n_used"] = float(np.count_nonzero(ok))
        if not np.any(ok):
            return out
        arr_ok = arr[ok]
        out["min"] = float(np.min(arr_ok))
        out["max"] = float(np.max(arr_ok))
        out["mean"] = float(np.mean(arr_ok))
        out["median"] = float(np.median(arr_ok))
        out["std"] = float(np.std(arr_ok))
        return out

    def _sleep_stage_stats(df: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if df is None or not hasattr(df, "columns"):
            return out

        stage_col = getattr(config, "AUX_CSV_STAGE_CODE_COLUMN", "stage_code")
        if stage_col not in df.columns:
            return out

        try:
            s = np.asarray(df[stage_col].to_numpy(dtype=float))
            ok = np.isfinite(s)
            if not np.any(ok):
                return out

            stage_i = np.round(s[ok]).astype(int)
            total = int(stage_i.size)
            if total <= 0:
                return out

            enabled = bool(getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False))
            policy = str(getattr(config, "SLEEP_STAGE_POLICY", "all_sleep"))

            try:
                include_set = set(config.sleep_include_numeric())
            except Exception:
                include_set = {1, 2, 3}

            included = np.array([int(x) in include_set for x in stage_i], dtype=bool)
            inc_n = int(np.sum(included))
            exc_n = int(total - inc_n)

            def pct(n: int) -> float:
                return 100.0 * float(n) / float(total)

            counts = {k: int(np.sum(stage_i == k)) for k in [0, 1, 2, 3]}

            out.update(
                {
                    "sleep_mask_enabled": int(enabled),
                    "sleep_policy": policy,
                    "sleep_included_n": inc_n,
                    "sleep_included_pct": pct(inc_n),
                    "sleep_excluded_n": exc_n,
                    "sleep_excluded_pct": pct(exc_n),
                    "sleep_wake_n": counts[0],
                    "sleep_wake_pct": pct(counts[0]),
                    "sleep_light_n": counts[1],
                    "sleep_light_pct": pct(counts[1]),
                    "sleep_deep_n": counts[2],
                    "sleep_deep_pct": pct(counts[2]),
                    "sleep_rem_n": counts[3],
                    "sleep_rem_pct": pct(counts[3]),
                }
            )
            return out
        except Exception:
            return out

    summary_folder = paths.get_output_folder(config.HR_OUTPUT_SUBFOLDER)

    summary_parts = features.enabled_feature_parts(("hr", "hrv", "psd", "delta_hr", "pat_burden", "sleep_combo_summary")) or ["SUMMARY"]

    if isinstance(sleep_combo_summaries, dict) and sleep_combo_summaries:
        sleep_stage_policy = "multi_sleep_summary"
    else:
        sleep_stage_policy = (
            str(getattr(config, "SLEEP_STAGE_POLICY", "unknown"))
            .replace(" ", "_")
            .replace("/", "-")
        )
    run_id = getattr(config, "RUN_ID", "default")
    summary_filename = f"{'_'.join(summary_parts)}_summary__{sleep_stage_policy}__{run_id}.csv"
    summary_path = summary_folder / summary_filename

    row: Dict[str, Any] = {
        "edf_file": edf_path.name,
    }
    has_aux_summary_context = features.any_enabled("hrv", "psd", "delta_hr", "pat_burden", "sleep_combo_summary")

    if features.is_enabled("psd"):
        row["selected_mayer_peak_hz"] = mayer_peak_freq
        row["selected_resp_peak_hz"] = resp_peak_freq

    if features.is_enabled("pat_burden"):
        row["selected_pat_burden"] = pat_burden
        if isinstance(pat_burden_diag, dict):
            row["selected_pat_burden_sleep_hours"] = pat_burden_diag.get("sleep_hours")
            row["selected_pat_burden_total_area_min"] = pat_burden_diag.get("total_area_min")
            row["selected_pat_burden_n_episodes"] = pat_burden_diag.get("n_episodes")
            row["selected_pat_burden_n_episodes_used"] = pat_burden_diag.get("n_episodes_used")
            row["selected_pat_burden_relative"] = int(bool(pat_burden_diag.get("relative", False)))
            row["selected_pat_burden_nan_pct"] = pat_burden_diag.get("nan_pct_inside")
        else:
            row["selected_pat_burden_sleep_hours"] = np.nan
            row["selected_pat_burden_total_area_min"] = np.nan
            row["selected_pat_burden_n_episodes"] = np.nan
            row["selected_pat_burden_n_episodes_used"] = np.nan
            row["selected_pat_burden_relative"] = np.nan
            row["selected_pat_burden_nan_pct"] = np.nan

    if features.is_enabled("hrv") and hrv_summary is not None:
        row.update(
            {
                "selected_rmssd_mean_ms": hrv_summary.get("rmssd_mean", np.nan),
                "selected_rmssd_median_ms": hrv_summary.get("rmssd_median", np.nan),
                "selected_sdnn_mean_ms": hrv_summary.get("sdnn_mean", np.nan),
                "selected_sdnn_median_ms": hrv_summary.get("sdnn_median", np.nan),
                "selected_lf": hrv_summary.get("lf", np.nan),
                "selected_lf_fixed_median": hrv_summary.get("lf_fixed_median", np.nan),
                "selected_hf": hrv_summary.get("hf", np.nan),
                "selected_hf_fixed_median": hrv_summary.get("hf_fixed_median", np.nan),
                "selected_lf_hf": hrv_summary.get("lf_hf", np.nan),
                "selected_lf_hf_fixed_median": hrv_summary.get("lf_hf_fixed_median", np.nan),
                "selected_lf_hf_fixed_mean": hrv_summary.get("lf_hf_fixed_mean", np.nan),
                "selected_lf_hf_fixed_n_windows_valid": hrv_summary.get("lf_hf_fixed_n_windows_valid", np.nan),
                "selected_lf_hf_fixed_n_windows_total": hrv_summary.get("lf_hf_fixed_n_windows_total", np.nan),
                "selected_lf_hf_fixed_valid_pct": hrv_summary.get("lf_hf_fixed_valid_pct", np.nan),
                "selected_lf_hf_fixed_valid_min": hrv_summary.get("lf_hf_fixed_valid_min", np.nan),
                "selected_lf_hf_fixed_total_min": hrv_summary.get("lf_hf_fixed_total_min", np.nan),
                "selected_lf_hf_fixed_window_sec": hrv_summary.get("lf_hf_fixed_window_sec", np.nan),
                "selected_lf_hf_fixed_hop_sec": hrv_summary.get("lf_hf_fixed_hop_sec", np.nan),
            }
        )
    elif features.is_enabled("hrv"):
        row.update(
            {
                "selected_rmssd_mean_ms": np.nan,
                "selected_rmssd_median_ms": np.nan,
                "selected_sdnn_mean_ms": np.nan,
                "selected_sdnn_median_ms": np.nan,
                "selected_lf": np.nan,
                "selected_lf_fixed_median": np.nan,
                "selected_hf": np.nan,
                "selected_hf_fixed_median": np.nan,
                "selected_lf_hf": np.nan,
                "selected_lf_hf_fixed_median": np.nan,
                "selected_lf_hf_fixed_mean": np.nan,
                "selected_lf_hf_fixed_n_windows_valid": np.nan,
                "selected_lf_hf_fixed_n_windows_total": np.nan,
                "selected_lf_hf_fixed_valid_pct": np.nan,
                "selected_lf_hf_fixed_valid_min": np.nan,
                "selected_lf_hf_fixed_total_min": np.nan,
                "selected_lf_hf_fixed_window_sec": np.nan,
                "selected_lf_hf_fixed_hop_sec": np.nan,
            }
        )

    if features.is_enabled("psd") and psd_features:
        row.update(
            {
                "selected_psd_pow_vlf": psd_features.get("pow_vlf"),
                "selected_psd_pow_mayer": psd_features.get("pow_mayer"),
                "selected_psd_pow_resp": psd_features.get("pow_resp"),
                "selected_psd_norm_mayer": psd_features.get("norm_mayer"),
                "selected_psd_norm_resp": psd_features.get("norm_resp"),
                "selected_psd_valid_windows": psd_features.get("n_windows"),
            }
        )

    if features.is_enabled("hr"):
        hr_cov = _coverage_stats(
            hr_calc,
            t=t_hr,
            default_fs=float(getattr(config, "HR_TARGET_FS_HZ", 1.0)),
        )
        hr_stats = _finite_stats(hr_calc)
        row["selected_hr_min_bpm"] = hr_stats["min"]
        row["selected_hr_max_bpm"] = hr_stats["max"]
        row["selected_hr_mean_bpm"] = hr_stats["mean"]
        row["selected_hr_median_bpm"] = hr_stats["median"]
        row["selected_hr_std_bpm"] = hr_stats["std"]
        row["selected_hr_valid_pct"] = hr_cov["valid_pct"]
        row["selected_hr_valid_min"] = hr_cov["valid_min"]
    if features.is_enabled("hrv"):
        hrv_clean_cov = _coverage_stats(
            hrv_clean,
            t=t_hrv,
            default_fs=float(getattr(config, "HRV_TARGET_FS_HZ", 1.0)),
        )
        hrv_raw_cov = _coverage_stats(
            hrv_raw,
            t=t_hrv,
            default_fs=float(getattr(config, "HRV_TARGET_FS_HZ", 1.0)),
        )
        row["selected_hrv_rmssd_final_analysis_valid_pct"] = hrv_clean_cov["valid_pct"]
        row["selected_hrv_rmssd_final_analysis_valid_min"] = hrv_clean_cov["valid_min"]
        row["selected_hrv_rmssd_pre_final_exclusion_valid_pct"] = hrv_raw_cov["valid_pct"]
        row["selected_hrv_rmssd_pre_final_exclusion_valid_min"] = hrv_raw_cov["valid_min"]

    if features.is_enabled("hrv") and isinstance(hrv_tv, dict):
        for k, v in hrv_tv.items():
            if v is None or k in {"tv_window_sec", "spectral_window_sec", "spectral_hop_sec"}:
                continue
            try:
                arr = np.asarray(v)
            except Exception:
                continue
            if arr.ndim == 0 or np.size(arr) != np.size(t_hrv):
                continue
            try:
                cov = _coverage_stats(
                    arr,
                    t=t_hrv,
                    default_fs=float(getattr(config, "HRV_TARGET_FS_HZ", 1.0)),
                )
                prefix = _hrv_tv_csv_prefix(k)
                row[f"{prefix}_valid_pct"] = cov["valid_pct"]
                row[f"{prefix}_valid_min"] = cov["valid_min"]
            except Exception:
                pass

    if features.is_enabled("hrv"):
        mask_breakdown = _mask_breakdown_stats(t_hrv, hrv_mask_info)
        row["selected_hrv_selected_policy_min"] = mask_breakdown["selected_policy_min"]
        row["selected_hrv_clean_kept_min"] = mask_breakdown["clean_kept_min"]
        row["selected_hrv_clean_kept_pct_of_selected"] = mask_breakdown["clean_kept_pct_of_selected"]
        row["selected_hrv_mask_excluded_total_min"] = mask_breakdown["excluded_total_min"]
        row["selected_hrv_mask_excluded_total_pct_of_selected"] = mask_breakdown["excluded_total_pct_of_selected"]
        row["selected_hrv_excluded_apnea_only_min"] = mask_breakdown["excluded_apnea_only_min"]
        row["selected_hrv_excluded_apnea_only_pct_of_selected"] = mask_breakdown["excluded_apnea_only_pct_of_selected"]
        row["selected_hrv_excluded_quality_only_min"] = mask_breakdown["excluded_quality_only_min"]
        row["selected_hrv_excluded_quality_only_pct_of_selected"] = mask_breakdown["excluded_quality_only_pct_of_selected"]
        row["selected_hrv_excluded_desat_only_min"] = mask_breakdown["excluded_desat_only_min"]
        row["selected_hrv_excluded_desat_only_pct_of_selected"] = mask_breakdown["excluded_desat_only_pct_of_selected"]
        row["selected_hrv_excluded_overlap_min"] = mask_breakdown["excluded_overlap_min"]
        row["selected_hrv_excluded_overlap_pct_of_selected"] = mask_breakdown["excluded_overlap_pct_of_selected"]

    if features.is_enabled("hrv") and isinstance(hrv_midpoint_halves, dict):
        for half_key, prefix in [("first_half", "nrem_first_half"), ("second_half", "nrem_second_half")]:
            half_summary = hrv_midpoint_halves.get(half_key)
            if not isinstance(half_summary, dict):
                continue
            for src, dst in [
                ("rmssd_mean", "rmssd_mean_ms"),
                ("rmssd_median", "rmssd_median_ms"),
                ("sdnn_mean", "sdnn_mean_ms"),
                ("sdnn_median", "sdnn_median_ms"),
                ("lf", "lf_ms2"),
                ("lf_fixed_median", "lf_fixed_median_ms2"),
                ("hf", "hf_ms2"),
                ("hf_fixed_median", "hf_fixed_median_ms2"),
                ("lf_hf", "lf_hf_ratio"),
                ("lf_hf_fixed_mean", "lf_hf_fixed_mean"),
                ("lf_hf_fixed_median", "lf_hf_fixed_median"),
                ("lf_hf_fixed_n_windows_valid", "lf_hf_fixed_n_windows_valid"),
                ("lf_hf_fixed_valid_min", "lf_hf_fixed_valid_min"),
            ]:
                row[f"{prefix}_{dst}"] = half_summary.get(src, np.nan)

    if has_aux_summary_context and aux_df is not None and hasattr(aux_df, "__len__"):
        try:
            row["aux_rows"] = int(len(aux_df))
        except Exception:
            row["aux_rows"] = ""

        aux_keys = [
            ("desat", "desat_flag"),
            ("exclude_hr", "exclude_hr_flag"),
            ("exclude_pat", "exclude_pat_flag"),
            ("evt_central_3", "evt_central_3"),
            ("evt_obstructive_3", "evt_obstructive_3"),
            ("evt_unclassified_3", "evt_unclassified_3"),
            ("evt_central_4", "evt_central_4"),
            ("evt_obstructive_4", "evt_obstructive_4"),
            ("evt_unclassified_4", "evt_unclassified_4"),
        ]
        for short, col in aux_keys:
            c, p = _count_flags(aux_df, col)
            if c is not None:
                row[f"{short}_n"] = c
            if p is not None:
                row[f"{short}_pct"] = p

        sleep_timing = compute_sleep_timing_from_aux(aux_df)
        if sleep_timing:
            row["sleep_onset_h_from_start"] = sleep_timing.get("sleep_onset_rel_h", np.nan)
            row["sleep_onset_hhmm_from_start"] = sleep_timing.get("sleep_onset_rel_hhmm", "")
            row["sleep_midpoint_h_from_start"] = sleep_timing.get("sleep_midpoint_rel_h", np.nan)
            row["sleep_midpoint_hhmm_from_start"] = sleep_timing.get("sleep_midpoint_rel_hhmm", "")
            row["sleep_end_h_from_start"] = sleep_timing.get("sleep_end_rel_h", np.nan)
            row["sleep_end_hhmm_from_start"] = sleep_timing.get("sleep_end_rel_hhmm", "")

    if has_aux_summary_context:
        row.update(_sleep_stage_stats(aux_df))

    if features.is_enabled("sleep_combo_summary") and isinstance(sleep_combo_summaries, dict):
        for key in ["pre_sleep_wake", "all_sleep", "wake_sleep", "nrem", "deep", "rem"]:
            item_obj = sleep_combo_summaries.get(key)
            if not isinstance(item_obj, dict):
                continue
            item: Dict[str, Any] = item_obj

            prefix = f"combo_{key}"
            row[f"{prefix}_sleep_hours"] = item.get("sleep_hours", np.nan)
            if features.is_enabled("pat_burden"):
                row[f"{prefix}_pat_burden"] = item.get("pat_burden", np.nan)

            hr_summary_obj = item.get("hr_summary")
            hr_item: Dict[str, Any] = hr_summary_obj if isinstance(hr_summary_obj, dict) else {}
            if features.is_enabled("hr"):
                row[f"{prefix}_hr_mean"] = hr_item.get("mean", np.nan)
                row[f"{prefix}_hr_median"] = hr_item.get("median", np.nan)
                row[f"{prefix}_hr_std"] = hr_item.get("std", np.nan)

            hrv_summary_obj = item.get("hrv_summary")
            hrv_item: Dict[str, Any] = hrv_summary_obj if isinstance(hrv_summary_obj, dict) else {}
            if features.is_enabled("hrv"):
                for src, dst in [
                    ("rmssd_mean", "rmssd_mean_ms"),
                    ("rmssd_median", "rmssd_median_ms"),
                    ("sdnn_mean", "sdnn_mean_ms"),
                    ("sdnn_median", "sdnn_median_ms"),
                    ("lf", "lf"),
                    ("hf", "hf"),
                    ("lf_hf", "lf_hf"),
                    ("lf_hf_fixed_median", "lf_hf_fixed_median"),
                    ("lf_hf_fixed_mean", "lf_hf_fixed_mean"),
                    ("lf_hf_fixed_n_windows_valid", "lf_hf_fixed_n_windows_valid"),
                    ("lf_hf_fixed_n_windows_total", "lf_hf_fixed_n_windows_total"),
                    ("lf_hf_fixed_valid_pct", "lf_hf_fixed_valid_pct"),
                    ("lf_hf_fixed_valid_min", "lf_hf_fixed_valid_min"),
                    ("lf_hf_fixed_total_min", "lf_hf_fixed_total_min"),
                    ("lf_hf_fixed_window_sec", "lf_hf_fixed_window_sec"),
                    ("lf_hf_fixed_hop_sec", "lf_hf_fixed_hop_sec"),
                ]:
                    row[f"{prefix}_{dst}"] = hrv_item.get(src, np.nan)

            psd_item_obj = item.get("psd_features")
            psd_item: Dict[str, Any] = psd_item_obj if isinstance(psd_item_obj, dict) else {}
            if features.is_enabled("psd"):
                row[f"{prefix}_psd_valid_windows"] = psd_item.get("n_windows", np.nan)

            hr_response_obj = item.get("hr_event_response_summary")
            hr_response_item: Dict[str, Any] = hr_response_obj if isinstance(hr_response_obj, dict) else {}
            if features.is_enabled("delta_hr"):
                row[f"{prefix}_trough_to_peak_response_mean"] = hr_response_item.get("trough_to_peak_response_mean", np.nan)
                row[f"{prefix}_mean_to_peak_response_mean"] = hr_response_item.get("mean_to_peak_response_mean", np.nan)
                row[f"{prefix}_event_windows_total"] = hr_response_item.get("n_event_windows", np.nan)
                row[f"{prefix}_event_windows_used"] = hr_response_item.get("n_used_windows", np.nan)

    base_order = [
        "edf_file",
        "selected_hr_min_bpm", "selected_hr_max_bpm", "selected_hr_mean_bpm", "selected_hr_median_bpm", "selected_hr_std_bpm", "selected_hr_valid_pct", "selected_hr_valid_min",
        "selected_rmssd_mean_ms", "selected_rmssd_median_ms", "selected_sdnn_mean_ms", "selected_sdnn_median_ms",
        "selected_hrv_rmssd_final_analysis_valid_pct", "selected_hrv_rmssd_final_analysis_valid_min", "selected_hrv_rmssd_pre_final_exclusion_valid_pct",
        "selected_hrv_rmssd_pre_final_exclusion_valid_min", "hrv_tv_sdnn_final_analysis_valid_pct", "hrv_tv_sdnn_final_analysis_valid_min",
        "hrv_tv_sdnn_pre_final_exclusion_valid_pct", "hrv_tv_sdnn_pre_final_exclusion_valid_min",
        "selected_lf", "selected_lf_fixed_median", "selected_hf", "selected_hf_fixed_median", "selected_lf_hf", "selected_lf_hf_fixed_median", "selected_lf_hf_fixed_mean", "selected_lf_hf_fixed_n_windows_valid",
        "selected_lf_hf_fixed_n_windows_total", "selected_lf_hf_fixed_valid_pct", "selected_lf_hf_fixed_valid_min", "selected_lf_hf_fixed_total_min",
        "selected_lf_hf_fixed_window_sec", "selected_lf_hf_fixed_hop_sec",
        "hrv_tv_lf_final_analysis_valid_pct", "hrv_tv_lf_final_analysis_valid_min", "hrv_tv_lf_pre_final_exclusion_valid_pct", "hrv_tv_lf_pre_final_exclusion_valid_min",
        "hrv_tv_hf_final_analysis_valid_pct", "hrv_tv_hf_final_analysis_valid_min", "hrv_tv_hf_pre_final_exclusion_valid_pct", "hrv_tv_hf_pre_final_exclusion_valid_min",
        "hrv_tv_lf_hf_final_analysis_valid_pct", "hrv_tv_lf_hf_final_analysis_valid_min", "hrv_tv_lf_hf_pre_final_exclusion_valid_pct", "hrv_tv_lf_hf_pre_final_exclusion_valid_min",
        "selected_mayer_peak_hz", "selected_resp_peak_hz", "selected_psd_pow_vlf", "selected_psd_pow_mayer", "selected_psd_pow_resp", "selected_psd_norm_mayer",
        "selected_psd_norm_resp", "selected_psd_valid_windows", "selected_pat_burden", "selected_pat_burden_sleep_hours",
        "selected_pat_burden_total_area_min", "selected_pat_burden_n_episodes", "selected_pat_burden_n_episodes_used",
        "selected_pat_burden_relative", "selected_pat_burden_nan_pct", "selected_hrv_selected_policy_min", "selected_hrv_clean_kept_min",
        "selected_hrv_clean_kept_pct_of_selected", "selected_hrv_mask_excluded_total_min", "selected_hrv_mask_excluded_total_pct_of_selected",
        "selected_hrv_excluded_apnea_only_min", "selected_hrv_excluded_apnea_only_pct_of_selected",
        "selected_hrv_excluded_quality_only_min", "selected_hrv_excluded_quality_only_pct_of_selected",
        "selected_hrv_excluded_desat_only_min", "selected_hrv_excluded_desat_only_pct_of_selected",
        "selected_hrv_excluded_overlap_min", "selected_hrv_excluded_overlap_pct_of_selected", "sleep_onset_h_from_start", "sleep_onset_hhmm_from_start",
        "sleep_midpoint_h_from_start", "sleep_midpoint_hhmm_from_start", "sleep_end_h_from_start", "sleep_end_hhmm_from_start",
        "nrem_first_half_rmssd_mean_ms", "nrem_first_half_rmssd_median_ms", "nrem_first_half_sdnn_mean_ms", "nrem_first_half_sdnn_median_ms", "nrem_first_half_lf_ms2", "nrem_first_half_lf_fixed_median_ms2", "nrem_first_half_hf_ms2", "nrem_first_half_hf_fixed_median_ms2",
        "nrem_first_half_lf_hf_ratio", "nrem_first_half_lf_hf_fixed_mean", "nrem_first_half_lf_hf_fixed_median", "nrem_first_half_lf_hf_fixed_n_windows_valid", "nrem_first_half_lf_hf_fixed_valid_min",
        "nrem_second_half_rmssd_mean_ms", "nrem_second_half_rmssd_median_ms", "nrem_second_half_sdnn_mean_ms", "nrem_second_half_sdnn_median_ms", "nrem_second_half_lf_ms2", "nrem_second_half_lf_fixed_median_ms2", "nrem_second_half_hf_ms2", "nrem_second_half_hf_fixed_median_ms2",
        "nrem_second_half_lf_hf_ratio", "nrem_second_half_lf_hf_fixed_mean", "nrem_second_half_lf_hf_fixed_median", "nrem_second_half_lf_hf_fixed_n_windows_valid", "nrem_second_half_lf_hf_fixed_valid_min",
        "aux_rows", "desat_n", "desat_pct", "exclude_hr_n", "exclude_hr_pct",
        "exclude_pat_n", "exclude_pat_pct", "evt_central_3_n", "evt_central_3_pct", "evt_obstructive_3_n",
        "evt_obstructive_3_pct", "evt_unclassified_3_n", "evt_unclassified_3_pct", "evt_central_4_n",
        "evt_central_4_pct", "evt_obstructive_4_n", "evt_obstructive_4_pct", "evt_unclassified_4_n",
        "evt_unclassified_4_pct", "sleep_mask_enabled", "sleep_policy", "sleep_included_n",
        "sleep_included_pct", "sleep_excluded_n", "sleep_excluded_pct", "sleep_wake_n", "sleep_wake_pct",
        "sleep_light_n", "sleep_light_pct", "sleep_deep_n", "sleep_deep_pct", "sleep_rem_n", "sleep_rem_pct",
    ]

    row_keys = list(row.keys())
    fieldnames = [c for c in base_order if c in row_keys]
    extras = sorted([k for k in row_keys if k not in set(fieldnames)])
    fieldnames.extend(extras)

    def _format_row_for_csv(d: Dict[str, Any], cols: Sequence[str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for k in cols:
            v = d.get(k, "")
            if v is None:
                out[k] = ""
            elif isinstance(v, (int, np.integer)):
                out[k] = str(int(v))
            elif isinstance(v, (float, np.floating)):
                if k.startswith("selected_psd_pow_"):
                    out[k] = _fmt_sci(float(v))
                elif k.endswith("_pct"):
                    out[k] = _fmt1(float(v))
                elif k.endswith("_valid_min") or k.endswith("_total_min"):
                    out[k] = _fmt1(float(v))
                else:
                    out[k] = _fmt6(float(v))
            else:
                out[k] = str(v)
        return out

    if not summary_path.exists():
        summary_folder.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow(_format_row_for_csv(row, fieldnames))
        return summary_path

    with summary_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        existing_fieldnames = r.fieldnames or []
        existing_rows = list(r)

    union_fieldnames = list(existing_fieldnames)
    for k in fieldnames:
        if k not in union_fieldnames:
            union_fieldnames.append(k)

    if union_fieldnames != existing_fieldnames:
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=union_fieldnames)
            w.writeheader()
            for old in existing_rows:
                full_old = {k: old.get(k, "") for k in union_fieldnames}
                w.writerow(full_old)
            w.writerow(_format_row_for_csv(row, union_fieldnames))
        return summary_path

    with summary_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=existing_fieldnames)
        w.writerow(_format_row_for_csv(row, existing_fieldnames))

    return summary_path


def append_hr_correlation_to_summary(
    edf_path: Path,
    pearson_r: Optional[float],
    spear_rho: Optional[float],
    rmse: Optional[float],
    hrv_summary: Optional[Dict[str, float]] = None,
    mayer_peak_freq: Optional[float] = None,
    resp_peak_freq: Optional[float] = None,
    *,
    t_hr: Optional[np.ndarray] = None,
    hr_calc: Optional[np.ndarray] = None,
    hr_edf: Optional[np.ndarray] = None,
    t_hrv: Optional[np.ndarray] = None,
    hrv_clean: Optional[np.ndarray] = None,
    hrv_raw: Optional[np.ndarray] = None,
    hrv_tv: Optional[Dict[str, np.ndarray]] = None,
    hrv_mask_info: Optional[Dict[str, object]] = None,
    hrv_midpoint_halves: Optional[Dict[str, Dict[str, float]]] = None,
    aux_df: Optional[Any] = None,
    psd_features: Optional[Dict[str, float]] = None,
    pat_burden: Optional[float] = None,
    pat_burden_diag: Optional[dict] = None,
) -> Path:
    """
    Backward-compatible wrapper.
    Proprietary/reference HR and correlation inputs are ignored.
    """
    return append_hr_hrv_summary(
        edf_path=edf_path,
        hrv_summary=hrv_summary,
        mayer_peak_freq=mayer_peak_freq,
        resp_peak_freq=resp_peak_freq,
        t_hr=t_hr,
        hr_calc=hr_calc,
        t_hrv=t_hrv,
        hrv_clean=hrv_clean,
        hrv_raw=hrv_raw,
        hrv_tv=hrv_tv,
        hrv_mask_info=hrv_mask_info,
        hrv_midpoint_halves=hrv_midpoint_halves,
        aux_df=aux_df,
        psd_features=psd_features,
        pat_burden=pat_burden,
        pat_burden_diag=pat_burden_diag,
    )
