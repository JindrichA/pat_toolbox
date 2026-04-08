from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .. import config, features, paths


def append_hr_hrv_summary(
    edf_path: Path,
    hrv_summary: Optional[Dict[str, float]] = None,
    mayer_peak_freq: Optional[float] = None,
    resp_peak_freq: Optional[float] = None,
    *,
    hr_calc: Optional[np.ndarray] = None,
    delta_hr_calc: Optional[np.ndarray] = None,
    delta_hr_calc_evt: Optional[np.ndarray] = None,
    hrv_clean: Optional[np.ndarray] = None,
    hrv_raw: Optional[np.ndarray] = None,
    hrv_tv: Optional[Dict[str, np.ndarray]] = None,
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
      - PAT HR / HRV NaN quality
      - aux event counts
      - sleep-stage masking stats
      - PAT burden
    """

    def _nan_pct(x: Optional[np.ndarray]) -> Optional[float]:
        if x is None:
            return None
        x = np.asarray(x)
        if x.size == 0:
            return None
        return float(100.0 * np.mean(~np.isfinite(x)))

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

    summary_parts = ["HR", *features.enabled_feature_parts(("hrv", "psd", "delta_hr", "pat_burden"))]

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

    if features.is_enabled("psd"):
        row["mayer_peak_hz"] = mayer_peak_freq
        row["resp_peak_hz"] = resp_peak_freq

    if features.is_enabled("pat_burden"):
        row["pat_burden"] = pat_burden
        if isinstance(pat_burden_diag, dict):
            row["pat_burden_sleep_hours"] = pat_burden_diag.get("sleep_hours")
            row["pat_burden_total_area_min"] = pat_burden_diag.get("total_area_min")
            row["pat_burden_n_episodes"] = pat_burden_diag.get("n_episodes")
            row["pat_burden_n_episodes_used"] = pat_burden_diag.get("n_episodes_used")
            row["pat_burden_relative"] = int(bool(pat_burden_diag.get("relative", False)))
            row["pat_burden_nan_pct"] = pat_burden_diag.get("nan_pct_inside")
        else:
            row["pat_burden_sleep_hours"] = np.nan
            row["pat_burden_total_area_min"] = np.nan
            row["pat_burden_n_episodes"] = np.nan
            row["pat_burden_n_episodes_used"] = np.nan
            row["pat_burden_relative"] = np.nan
            row["pat_burden_nan_pct"] = np.nan

    if features.is_enabled("hrv") and hrv_summary is not None:
        row.update(
            {
                "rmssd_mean_ms": hrv_summary.get("rmssd_mean", np.nan),
                "rmssd_median_ms": hrv_summary.get("rmssd_median", np.nan),
                "sdnn_ms": hrv_summary.get("sdnn", np.nan),
                "lf": hrv_summary.get("lf", np.nan),
                "hf": hrv_summary.get("hf", np.nan),
                "lf_hf": hrv_summary.get("lf_hf", np.nan),
                "lf_n_segments_used": hrv_summary.get("lf_n_segments_used", np.nan),
                "lf_hf_fixed_median": hrv_summary.get("lf_hf_fixed_median", np.nan),
                "lf_hf_fixed_mean": hrv_summary.get("lf_hf_fixed_mean", np.nan),
                "lf_hf_fixed_n_windows_valid": hrv_summary.get("lf_hf_fixed_n_windows_valid", np.nan),
                "lf_hf_fixed_n_windows_total": hrv_summary.get("lf_hf_fixed_n_windows_total", np.nan),
                "lf_hf_fixed_window_sec": hrv_summary.get("lf_hf_fixed_window_sec", np.nan),
                "lf_hf_fixed_hop_sec": hrv_summary.get("lf_hf_fixed_hop_sec", np.nan),
            }
        )
    elif features.is_enabled("hrv"):
        row.update(
            {
                "rmssd_mean_ms": np.nan,
                "rmssd_median_ms": np.nan,
                "sdnn_ms": np.nan,
                "lf": np.nan,
                "hf": np.nan,
                "lf_hf": np.nan,
                "lf_n_segments_used": np.nan,
                "lf_hf_fixed_median": np.nan,
                "lf_hf_fixed_mean": np.nan,
                "lf_hf_fixed_n_windows_valid": np.nan,
                "lf_hf_fixed_n_windows_total": np.nan,
                "lf_hf_fixed_window_sec": np.nan,
                "lf_hf_fixed_hop_sec": np.nan,
            }
        )

    if features.is_enabled("psd") and psd_features:
        row.update(
            {
                "psd_pow_vlf": psd_features.get("pow_vlf"),
                "psd_pow_mayer": psd_features.get("pow_mayer"),
                "psd_pow_resp": psd_features.get("pow_resp"),
                "psd_norm_mayer": psd_features.get("norm_mayer"),
                "psd_norm_resp": psd_features.get("norm_resp"),
                "psd_valid_windows": psd_features.get("n_windows"),
            }
        )

    row["hr_pat_nan_pct"] = _nan_pct(hr_calc)
    if features.is_enabled("delta_hr"):
        delta_all = _finite_stats(delta_hr_calc)
        delta_evt = _finite_stats(delta_hr_calc_evt)
        row.update(
            {
                "delta_hr_mean": delta_all["mean"],
                "delta_hr_median": delta_all["median"],
                "delta_hr_std": delta_all["std"],
                "delta_hr_nan_pct": delta_all["nan_pct"],
                "delta_hr_n_used": delta_all["n_used"],
                "delta_hr_evt_mean": delta_evt["mean"],
                "delta_hr_evt_median": delta_evt["median"],
                "delta_hr_evt_std": delta_evt["std"],
                "delta_hr_evt_nan_pct": delta_evt["nan_pct"],
                "delta_hr_evt_n_used": delta_evt["n_used"],
            }
        )
    if features.is_enabled("hrv"):
        row["hrv_rmssd_clean_nan_pct"] = _nan_pct(hrv_clean)
        row["hrv_rmssd_raw_nan_pct"] = _nan_pct(hrv_raw)

    if features.is_enabled("hrv") and isinstance(hrv_tv, dict):
        for k, v in hrv_tv.items():
            if v is None:
                continue
            try:
                row[f"hrv_tv_{k}_nan_pct"] = _nan_pct(np.asarray(v))
            except Exception:
                pass

    if aux_df is not None and hasattr(aux_df, "__len__"):
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

    row.update(_sleep_stage_stats(aux_df))

    if features.is_enabled("sleep_combo_summary") and isinstance(sleep_combo_summaries, dict):
        for key in ["all_sleep", "wake_sleep", "nrem", "deep", "rem"]:
            item_obj = sleep_combo_summaries.get(key)
            if not isinstance(item_obj, dict):
                continue
            item: Dict[str, Any] = item_obj

            prefix = f"combo_{key}"
            row[f"{prefix}_sleep_hours"] = item.get("sleep_hours", np.nan)
            row[f"{prefix}_pat_burden"] = item.get("pat_burden", np.nan)

            hrv_summary_obj = item.get("hrv_summary")
            hrv_item: Dict[str, Any] = hrv_summary_obj if isinstance(hrv_summary_obj, dict) else {}
            for src, dst in [
                ("rmssd_mean", "rmssd_mean_ms"),
                ("sdnn", "sdnn_ms"),
                ("lf_hf", "lf_hf"),
                ("lf_n_segments_used", "lf_n_segments_used"),
                ("lf_hf_fixed_n_windows_valid", "lf_hf_fixed_n_windows_valid"),
            ]:
                row[f"{prefix}_{dst}"] = hrv_item.get(src, np.nan)

            psd_item_obj = item.get("psd_features")
            psd_item: Dict[str, Any] = psd_item_obj if isinstance(psd_item_obj, dict) else {}
            row[f"{prefix}_psd_valid_windows"] = psd_item.get("n_windows", np.nan)

            delta_item_obj = item.get("delta_hr_summary")
            delta_item: Dict[str, Any] = delta_item_obj if isinstance(delta_item_obj, dict) else {}
            row[f"{prefix}_delta_hr_mean"] = delta_item.get("full_mean", np.nan)
            row[f"{prefix}_delta_hr_evt_mean"] = delta_item.get("event_mean", np.nan)

            hr_response_obj = item.get("hr_event_response_summary")
            hr_response_item: Dict[str, Any] = hr_response_obj if isinstance(hr_response_obj, dict) else {}
            row[f"{prefix}_peak_minus_baseline_hr"] = hr_response_item.get("peak_minus_baseline", np.nan)
            row[f"{prefix}_peak_to_trough_hr"] = hr_response_item.get("peak_to_trough", np.nan)
            row[f"{prefix}_post_peak_minus_pre_mean_hr"] = hr_response_item.get("post_peak_minus_pre_mean", np.nan)
            row[f"{prefix}_event_windows_total"] = hr_response_item.get("n_event_windows", np.nan)
            row[f"{prefix}_event_windows_used"] = hr_response_item.get("n_used_windows", np.nan)

    base_order = [
        "edf_file", "rmssd_mean_ms", "rmssd_median_ms", "sdnn_ms", "lf", "hf", "lf_hf",
        "lf_n_segments_used", "lf_hf_fixed_median", "lf_hf_fixed_mean", "lf_hf_fixed_n_windows_valid",
        "lf_hf_fixed_n_windows_total", "lf_hf_fixed_window_sec", "lf_hf_fixed_hop_sec", "mayer_peak_hz",
        "resp_peak_hz", "psd_pow_vlf", "psd_pow_mayer", "psd_pow_resp", "psd_norm_mayer",
        "psd_norm_resp", "psd_valid_windows", "pat_burden", "pat_burden_sleep_hours",
        "pat_burden_total_area_min", "pat_burden_n_episodes", "pat_burden_n_episodes_used",
        "pat_burden_relative", "pat_burden_nan_pct", "hr_pat_nan_pct", "delta_hr_mean", "delta_hr_median",
        "delta_hr_std", "delta_hr_nan_pct", "delta_hr_n_used", "delta_hr_evt_mean", "delta_hr_evt_median",
        "delta_hr_evt_std", "delta_hr_evt_nan_pct", "delta_hr_evt_n_used", "hrv_rmssd_clean_nan_pct",
        "hrv_rmssd_raw_nan_pct", "aux_rows", "desat_n", "desat_pct", "exclude_hr_n", "exclude_hr_pct",
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
                if k.startswith("psd_pow_"):
                    out[k] = _fmt_sci(float(v))
                elif k.endswith("_pct") or k.endswith("_nan_pct"):
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
    hr_calc: Optional[np.ndarray] = None,
    hr_edf: Optional[np.ndarray] = None,
    delta_hr_calc: Optional[np.ndarray] = None,
    delta_hr_calc_evt: Optional[np.ndarray] = None,
    hrv_clean: Optional[np.ndarray] = None,
    hrv_raw: Optional[np.ndarray] = None,
    hrv_tv: Optional[Dict[str, np.ndarray]] = None,
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
        hr_calc=hr_calc,
        delta_hr_calc=delta_hr_calc,
        delta_hr_calc_evt=delta_hr_calc_evt,
        hrv_clean=hrv_clean,
        hrv_raw=hrv_raw,
        hrv_tv=hrv_tv,
        aux_df=aux_df,
        psd_features=psd_features,
        pat_burden=pat_burden,
        pat_burden_diag=pat_burden_diag,
    )
