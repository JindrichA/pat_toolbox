from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

from .. import config, features, io_aux_csv, masking, sleep_mask
from ..io.aux_events import compute_sleep_timing_from_aux
from .prv_plot_utils import (
    _add_colored_event_key,
    _add_metric_legend,
    _add_summary_line,
    _bin_series_mean_ci,
    _overlay_events_on_single_axis_whole_night,
    _plot_binned_series_with_support,
)
from .segment_plot_helpers import _overlay_pat_burden_area
from .summary_hypnogram import _plot_sleep_stagegram_on_axis
from .utils import _count_flags, _fmt, _shade_masked_regions

if TYPE_CHECKING:
    import pandas as pd


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
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "valid_pct": None,
        "valid_min": None,
        "total_min": None,
        "n_valid": None,
        "n_total": None,
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
    dt_sec = _sample_dt_sec(t, default_fs)
    n_valid = float(np.count_nonzero(ok))
    n_total = float(arr.size)
    out["valid_pct"] = 100.0 * n_valid / n_total if n_total > 0 else np.nan
    out["valid_min"] = (n_valid * dt_sec) / 60.0
    out["total_min"] = (n_total * dt_sec) / 60.0
    out["n_valid"] = n_valid
    out["n_total"] = n_total
    return out


def _fmt_pct(x: Optional[float], ndigits: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x:.{ndigits}f}%"


def _fmt_sci(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x:.2e}"


def _fmt_num(x: Optional[float], nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x:.{nd}f}"


def _fmt_int(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{int(x)}"


def _active_aux_flag_rows(aux_df, policy: masking.MaskPolicy) -> List[List[str]]:
    labels = {
        "desat_flag": "Desaturation flags",
        "exclude_hr_flag": "Exclude HR flags",
        "exclude_pat_flag": "Exclude PAT flags",
        "evt_central_3": "Central A/H 3%",
        "evt_obstructive_3": "Obstructive A/H 3%",
        "evt_unclassified_3": "Unclassified A/H 3%",
        "evt_central_4": "Central A/H 4%",
        "evt_obstructive_4": "Obstructive A/H 4%",
        "evt_unclassified_4": "Unclassified A/H 4%",
    }
    active_cols = list(policy.exclusion_columns)
    if policy.use_desat_windows:
        desat_col = str(getattr(config, "PRV_EXCLUSION_DESAT_COLUMN_KEY", "desat_flag"))
        if desat_col not in active_cols:
            active_cols.append(desat_col)

    rows: List[List[str]] = []
    for col in active_cols:
        label = labels.get(col)
        if label is None:
            continue
        n, pct = _count_flags(aux_df, col)
        rows.append([f"  {label}", f"{n:d} ({pct})"])
    return rows


def _append_coverage_rows(rows: List[List[str]], label: str, stats: Dict[str, Optional[float]]) -> None:
    rows.append([f"  {label} valid [min]", _fmt_num(stats.get("valid_min"), 1)])
    rows.append([f"  {label} valid [%]", _fmt_pct(stats.get("valid_pct"), 1)])


def _prv_tv_display_label(key: str) -> str:
    mapping = {
        "sdnn_ms_raw": "SDNN pre-final exclusion",
        "sdnn_ms": "SDNN final-analysis",
        "lf_raw": "LF pre-final exclusion",
        "lf": "LF final-analysis",
        "hf_raw": "HF pre-final exclusion",
        "hf": "HF final-analysis",
        "lf_hf_raw": "LF/HF pre-final exclusion",
        "lf_hf": "LF/HF final-analysis",
    }
    return mapping.get(key, key)


def _build_mask_breakdown_rows(
    t_prv: Optional[np.ndarray],
    prv_mask_info: Optional[Dict[str, object]],
) -> List[List[str]]:
    if t_prv is None or not prv_mask_info:
        return []

    sleep_keep = prv_mask_info.get("sleep_keep")
    apnea_keep = prv_mask_info.get("apnea_keep")
    quality_keep = prv_mask_info.get("quality_keep")
    desat_keep = prv_mask_info.get("desat_keep")
    combined_keep = prv_mask_info.get("combined_keep")

    if not all(isinstance(v, np.ndarray) and np.size(v) == np.size(t_prv) for v in [sleep_keep, apnea_keep, quality_keep, desat_keep, combined_keep]):
        return []

    dt_sec = _sample_dt_sec(t_prv, float(getattr(config, "PRV_TARGET_FS_HZ", 1.0)))
    sleep_keep = np.asarray(sleep_keep, dtype=bool)
    apnea_keep = np.asarray(apnea_keep, dtype=bool)
    quality_keep = np.asarray(quality_keep, dtype=bool)
    desat_keep = np.asarray(desat_keep, dtype=bool)
    combined_keep = np.asarray(combined_keep, dtype=bool)

    selected_n = int(np.count_nonzero(sleep_keep))
    if selected_n <= 0:
        return []

    apnea_excl = sleep_keep & (~apnea_keep)
    quality_excl = sleep_keep & (~quality_keep)
    desat_excl = sleep_keep & (~desat_keep)
    excl_count = apnea_excl.astype(int) + quality_excl.astype(int) + desat_excl.astype(int)

    only_apnea = excl_count == 1
    only_quality = excl_count == 1
    only_desat = excl_count == 1
    only_apnea &= apnea_excl
    only_quality &= quality_excl
    only_desat &= desat_excl
    overlap = excl_count > 1
    excluded_total = sleep_keep & (~combined_keep)

    def _min(mask: np.ndarray) -> float:
        return float(np.count_nonzero(mask) * dt_sec / 60.0)

    def _pct(mask: np.ndarray) -> float:
        return float(100.0 * np.count_nonzero(mask) / selected_n)

    rows: List[List[str]] = [["Selected-policy exclusion breakdown", ""]]
    rows += [["  Selected-policy time [min]", _fmt_num(_min(sleep_keep), 1)]]
    rows += [["  Final clean kept [min]", _fmt_num(_min(combined_keep), 1)], ["  Final clean kept [% of selected-policy]", _fmt_pct(_pct(combined_keep), 1)]]
    rows += [["  Mask-excluded total [min]", _fmt_num(_min(excluded_total), 1)], ["  Mask-excluded total [% of selected-policy]", _fmt_pct(_pct(excluded_total), 1)]]
    rows += [["  Apnea-only excluded [min]", _fmt_num(_min(only_apnea), 1)], ["  Apnea-only excluded [% of selected-policy]", _fmt_pct(_pct(only_apnea), 1)]]
    rows += [["  Quality-flag-only excluded [min]", _fmt_num(_min(only_quality), 1)], ["  Quality-flag-only excluded [% of selected-policy]", _fmt_pct(_pct(only_quality), 1)]]
    rows += [["  Desat-only excluded [min]", _fmt_num(_min(only_desat), 1)], ["  Desat-only excluded [% of selected-policy]", _fmt_pct(_pct(only_desat), 1)]]
    rows += [["  Overlap excluded [min]", _fmt_num(_min(overlap), 1)], ["  Overlap excluded [% of selected-policy]", _fmt_pct(_pct(overlap), 1)]]
    rows += [["", ""], ["Note", "Mask-based only; upstream PR cleaning / PAT artifact loss is not included here."]]
    return rows


def _format_sleep_timing_value(timing: Dict[str, Any], key: str) -> str:
    rel_h = timing.get(f"sleep_{key}_rel_h")
    rel_hhmm = timing.get(f"sleep_{key}_rel_hhmm")
    return f"{_fmt_num(rel_h, 2)} h ({rel_hhmm} from start)"


def _build_sleep_timing_rows(aux_df: Optional["pd.DataFrame"]) -> List[List[str]]:
    if aux_df is None:
        return []
    timing = compute_sleep_timing_from_aux(aux_df)
    if not timing:
        return []
    return [
        ["Sleep timing", ""],
        ["  Sleep onset", _format_sleep_timing_value(timing, "onset")],
        ["  Sleep midpoint", _format_sleep_timing_value(timing, "midpoint")],
        ["  Sleep end", _format_sleep_timing_value(timing, "end")],
    ]


def _wrap_csv_columns(cols: list[str], *, per_line: int = 3) -> str:
    if not cols:
        return "none"
    chunks = [cols[i:i + per_line] for i in range(0, len(cols), per_line)]
    return "\n".join(", ".join(chunk) for chunk in chunks)


def _finite_stats(y: Optional[np.ndarray]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_total": None, "n_used": None, "min": None, "max": None, "mean": None, "median": None, "std": None, "nan_pct": None}
    if y is None:
        return out
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        out["n_total"] = 0.0
        out["n_used"] = 0.0
        out["nan_pct"] = 100.0
        return out
    out["n_total"] = float(y.size)
    out["nan_pct"] = float(100.0 * np.mean(~np.isfinite(y)))
    ok = np.isfinite(y)
    out["n_used"] = float(np.count_nonzero(ok))
    if np.count_nonzero(ok) == 0:
        return out
    yy = y[ok]
    out["min"] = float(np.min(yy))
    out["max"] = float(np.max(yy))
    out["mean"] = float(np.mean(yy))
    out["median"] = float(np.median(yy))
    out["std"] = float(np.std(yy))
    return out


def _sleep_stage_rows(aux_df: Optional["pd.DataFrame"]) -> List[List[str]]:
    rows: List[List[str]] = []
    if aux_df is None:
        return rows
    time_col = getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec")
    stage_code_col = getattr(config, "AUX_CSV_STAGE_CODE_COLUMN", "stage_code")
    if time_col not in aux_df.columns or stage_code_col not in aux_df.columns:
        return rows
    stage = aux_df[stage_code_col].to_numpy(dtype=float)
    ok = np.isfinite(stage)
    if not np.any(ok):
        return rows
    stage_i = np.round(stage[ok]).astype(int)
    total = int(stage_i.size)
    if total <= 0:
        return rows
    enabled = bool(getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False))
    policy = str(getattr(config, "SLEEP_STAGE_POLICY", "all_sleep"))
    try:
        include_set = set(config.sleep_include_numeric())
    except Exception:
        include_set = {1, 2, 3}
    included = np.array([s in include_set for s in stage_i], dtype=bool)
    inc_n = int(np.sum(included))
    exc_n = int(total - inc_n)
    counts = {k: int(np.sum(stage_i == k)) for k in [0, 1, 2, 3]}
    names = {0: "Wake", 1: "Light", 2: "Deep", 3: "REM"}
    pct = lambda n: f"{(100.0 * n / total):.1f}%"
    rows.append(["Sleep-stage masking", ""])
    rows.append(["  Enabled", "Yes" if enabled else "No"])
    rows.append(["  Policy", policy])
    rows.append(["  Included (by policy)", f"{inc_n} ({pct(inc_n)})"])
    rows.append(["  Excluded (by policy)", f"{exc_n} ({pct(exc_n)})"])
    rows.append(["  Stage breakdown", ""])
    for k in [0, 1, 2, 3]:
        rows.append([f"    {names[k]}", f"{counts[k]} ({pct(counts[k])})"])
    return rows


def _sleep_combo_row_values(item: Dict[str, Any]) -> tuple[str, list[str], list[str], list[str]]:
    label = str(item.get("label", "subset"))
    sleep_hours = _fmt_num(item.get("sleep_hours"), 2)
    hr_summary_obj = item.get("hr_summary")
    hr_summary: Dict[str, Any] = hr_summary_obj if isinstance(hr_summary_obj, dict) else {}
    prv_summary_obj = item.get("prv_summary")
    prv_summary: Dict[str, Any] = prv_summary_obj if isinstance(prv_summary_obj, dict) else {}
    psd_features_obj = item.get("psd_features")
    psd_features: Dict[str, Any] = psd_features_obj if isinstance(psd_features_obj, dict) else {}
    hr_response_obj = item.get("hr_event_response_summary")
    hr_response: Dict[str, Any] = hr_response_obj if isinstance(hr_response_obj, dict) else {}
    pwa_drop_obj = item.get("pwa_drop_summary")
    pwa_drop: Dict[str, Any] = pwa_drop_obj if isinstance(pwa_drop_obj, dict) else {}
    burden = item.get("pat_burden")

    core_primary = [f"{sleep_hours} h"]
    if features.is_enabled("hr"):
        core_primary.extend([
            f"{_fmt(hr_summary.get('mean'), 1)} bpm",
            f"{_fmt(hr_summary.get('median'), 1)} bpm",
            f"{_fmt(hr_summary.get('std'), 1)} bpm",
        ])

    core_secondary: list[str] = []
    right: list[str] = []
    if features.is_enabled("prv"):
        core_primary.extend([
            f"{_fmt(prv_summary.get('rmssd_mean'), 1)} ms",
            f"{_fmt(prv_summary.get('rmssd_median'), 1)} ms",
        ])
        core_secondary.extend([
            f"{_fmt(prv_summary.get('sdnn_mean'), 1)} ms",
            f"{_fmt(prv_summary.get('sdnn_median'), 1)} ms",
            f"{_fmt(prv_summary.get('rmssd_valid_min'), 1)} min",
            f"{_fmt(prv_summary.get('rmssd_valid_pct'), 1)}%",
            f"{_fmt(prv_summary.get('sdnn_valid_min'), 1)} min",
            f"{_fmt(prv_summary.get('sdnn_valid_pct'), 1)}%",
            f"{_fmt(prv_summary.get('lf'), 2)}",
            f"{_fmt(prv_summary.get('hf'), 2)}",
            f"{_fmt(prv_summary.get('lf_hf'), 2)}",
        ])
        right.extend([
            f"{_fmt(prv_summary.get('rmssd_p90'), 1)} ms",
            f"{_fmt(prv_summary.get('rmssd_iqr'), 1)} ms",
            f"{_fmt(prv_summary.get('sdnn_p90'), 1)} ms",
            f"{_fmt(prv_summary.get('sdnn_iqr'), 1)} ms",
            f"{_fmt(prv_summary.get('ipi_mean_ms'), 1)} ms",
            f"{_fmt(prv_summary.get('ipi_median_ms'), 1)} ms",
            f"{_fmt(prv_summary.get('lf_fixed_p90'), 2)}",
            f"{_fmt(prv_summary.get('hf_fixed_p90'), 2)}",
            f"{_fmt(prv_summary.get('lf_hf_fixed_p90'), 2)}",
        ])

    if features.is_enabled("psd"):
        right.append(_fmt_int(psd_features.get("n_windows")))
    if features.is_enabled("delta_hr"):
        right.extend([
            f"{_fmt(hr_response.get('trough_to_peak_response_mean'), 2)} bpm",
            f"{_fmt(hr_response.get('mean_to_peak_response_mean'), 2)} bpm",
            f"{_fmt_int(hr_response.get('n_used_windows'))}/{_fmt_int(hr_response.get('n_event_windows'))}",
        ])
    if features.is_enabled("pat_burden"):
        right.append(_fmt(burden, 3))
    if features.is_enabled("pwa_drop"):
        right.extend([
            _fmt_int(pwa_drop.get("n_drops")),
            _fmt(pwa_drop.get("drop_rate_per_sleep_hour"), 2),
            _fmt(pwa_drop.get("mean_amplitude_pct"), 1),
        ])
    return label, core_primary, core_secondary, right


def _sleep_combo_tables(sleep_combo_summaries: Optional[Dict[str, Dict[str, object]]]) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    if not sleep_combo_summaries:
        return [], [], []

    primary_headers = ["Subset", "Sleep h"]
    if features.is_enabled("hr"):
        primary_headers.extend(["HR mean", "HR med", "HR std"])
    if features.is_enabled("prv"):
        primary_headers.extend(["RMSSD mean", "RMSSD med"])

    secondary_headers = ["Subset"]
    if features.is_enabled("prv"):
        secondary_headers.extend(["SDNN mean", "SDNN med", "RMSSD valid\n[min]", "RMSSD valid\n[%]", "SDNN valid\n[min]", "SDNN valid\n[%]", "LF mean\n[ms^2]", "HF mean\n[ms^2]", "LF/HF mean\n[-]"])

    right_headers = ["Subset"]
    if features.is_enabled("prv"):
        right_headers.extend(["RMSSD p90", "RMSSD IQR", "SDNN p90", "SDNN IQR", "IPI mean", "IPI med", "LF fixed p90", "HF fixed p90", "LF/HF fixed p90"])
    if features.is_enabled("psd"):
        right_headers.append("PSD win")
    if features.is_enabled("delta_hr"):
        right_headers.extend(["Tr-Pk resp", "Mean-Pk dHR", "win used/tot"])
    if features.is_enabled("pat_burden"):
        right_headers.append("Burden")
    if features.is_enabled("pwa_drop"):
        right_headers.extend(["PWA n", "PWA /h", "PWA amp %"])

    primary_rows: list[list[str]] = [primary_headers]
    secondary_rows: list[list[str]] = [secondary_headers] if len(secondary_headers) > 1 else []
    right_rows: list[list[str]] = [right_headers] if len(right_headers) > 1 else []

    for key in ["pre_sleep_wake", "all_sleep", "wake_sleep", "nrem", "deep", "rem"]:
        item_obj = sleep_combo_summaries.get(key)
        if not isinstance(item_obj, dict):
            continue
        item: Dict[str, Any] = item_obj
        label, primary_values, secondary_values, right_values = _sleep_combo_row_values(item)
        primary_rows.append([label, *primary_values])
        if secondary_rows:
            secondary_rows.append([label, *secondary_values])
        if right_rows:
            right_rows.append([label, *right_values])
    return primary_rows, secondary_rows, right_rows


def _render_sleep_combo_page(
    edf_base: str,
    primary_rows: list[list[str]],
    secondary_rows: list[list[str]],
    right_rows: list[list[str]],
):
    n_axes = 1 + int(bool(secondary_rows)) + int(bool(right_rows))
    fig, axes = plt.subplots(n_axes, 1, figsize=(11.69, 8.27))
    axes_list = [axes] if not isinstance(axes, np.ndarray) else list(axes)
    fig.suptitle(f"{edf_base} – Summary (Sleep-Subset Comparison)", fontsize=16, y=0.985)

    for ax in axes_list:
        ax.axis("off")

    primary_table = axes_list[0].table(
        cellText=primary_rows[1:],
        colLabels=primary_rows[0],
        loc="center",
        cellLoc="left",
    )
    primary_table.auto_set_font_size(False)
    primary_table.set_fontsize(9)
    primary_table.auto_set_column_width(col=list(range(len(primary_rows[0]))))
    primary_table.scale(1.0, 1.4)
    axes_list[0].set_title("Subset core metrics I", fontsize=12, pad=22)
    axes_list[0].text(
        0.5,
        0.98,
        "HR mean/med/std = PAT-derived HR summary within each subset after the selected-policy combined mask\n"
        "(sleep + event/quality/desat exclusion as configured), not a residual signal.",
        transform=axes_list[0].transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )

    next_axis_idx = 1
    if secondary_rows and next_axis_idx < len(axes_list):
        secondary_table = axes_list[next_axis_idx].table(
            cellText=secondary_rows[1:],
            colLabels=secondary_rows[0],
            loc="center",
            cellLoc="left",
        )
        secondary_table.auto_set_font_size(False)
        secondary_table.set_fontsize(9)
        secondary_table.auto_set_column_width(col=list(range(len(secondary_rows[0]))))
        secondary_table.scale(1.0, 1.4)
        axes_list[next_axis_idx].set_title("Subset core metrics II", fontsize=12, pad=14)
        next_axis_idx += 1

    if right_rows and next_axis_idx < len(axes_list):
        axes_list[next_axis_idx].set_title("Subset event-response / burden metrics", fontsize=12, pad=56)
        axes_list[next_axis_idx].text(
            0.5,
            0.98,
            "Tr-Pk resp = mean(recovery max HR - event-window trough) across valid events\n"
            "Mean-Pk dHR = mean(recovery max HR - event-window mean HR) across valid events\n"
            "win used/tot = valid event windows / all detected event windows",
            transform=axes_list[next_axis_idx].transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
        right_table = axes_list[next_axis_idx].table(
            cellText=right_rows[1:],
            colLabels=right_rows[0],
            loc="center",
            cellLoc="left",
        )
        right_table.auto_set_font_size(False)
        right_table.set_fontsize(10)
        right_table.scale(1.1, 1.4)

    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.95))
    return fig


def _build_quality_rows(
    t_hr: Optional[np.ndarray],
    hr_calc: Optional[np.ndarray],
    t_prv: Optional[np.ndarray],
    prv_clean: Optional[np.ndarray],
    prv_raw: Optional[np.ndarray],
    prv_tv: Optional[Dict[str, np.ndarray]],
) -> tuple[List[List[str]], List[List[str]], List[List[str]]]:
    hr_rows: List[List[str]] = []
    ts_rows: List[List[str]] = []
    spectral_rows: List[List[str]] = []
    if features.is_enabled("hr"):
        hr_cov = _coverage_stats(
            hr_calc,
            t=t_hr,
            default_fs=float(getattr(config, "HR_TARGET_FS_HZ", 1.0)),
        )
        hr_pat_stats = _finite_stats(hr_calc)
        hr_rows += [["PAT-derived HR summary", ""], ["  HR min / max [bpm]", f"{_fmt_num(hr_pat_stats['min'], 2)} / {_fmt_num(hr_pat_stats['max'], 2)}"], ["  HR mean [bpm]", _fmt_num(hr_pat_stats["mean"], 2)], ["  HR median [bpm]", _fmt_num(hr_pat_stats["median"], 2)], ["  HR std [bpm]", _fmt_num(hr_pat_stats["std"], 2)], ["", ""], ["Valid signal coverage", ""]]
        _append_coverage_rows(hr_rows, "HR (PAT-derived)", hr_cov)

    if features.is_enabled("prv"):
        prv_clean_cov = _coverage_stats(
            prv_clean,
            t=t_prv,
            default_fs=float(getattr(config, "PRV_TARGET_FS_HZ", 1.0)),
        )
        prv_raw_cov = _coverage_stats(
            prv_raw,
            t=t_prv,
            default_fs=float(getattr(config, "PRV_TARGET_FS_HZ", 1.0)),
        )
        ts_rows += [["PRV time-series valid coverage", ""]]
        _append_coverage_rows(ts_rows, "RMSSD final-analysis", prv_clean_cov)
        _append_coverage_rows(ts_rows, "RMSSD pre-final exclusion", prv_raw_cov)
        if isinstance(prv_tv, dict) and len(prv_tv) > 0:
            time_series_keys = ["sdnn_ms", "sdnn_ms_raw"]
            spectral_keys = ["lf", "lf_raw", "hf", "hf_raw", "lf_hf", "lf_hf_raw"]

            available_time_series = [k for k in time_series_keys if k in prv_tv and prv_tv.get(k) is not None]
            if available_time_series:
                for k in available_time_series:
                    stats = _coverage_stats(
                        prv_tv.get(k),
                        t=t_prv,
                        default_fs=float(getattr(config, "PRV_TARGET_FS_HZ", 1.0)),
                    )
                    _append_coverage_rows(ts_rows, _prv_tv_display_label(k), stats)

            available_spectral = [k for k in spectral_keys if k in prv_tv and prv_tv.get(k) is not None]
            if available_spectral:
                spectral_rows += [["PRV spectral valid coverage", ""]]
                for k in available_spectral:
                    stats = _coverage_stats(
                        prv_tv.get(k),
                        t=t_prv,
                        default_fs=float(getattr(config, "PRV_TARGET_FS_HZ", 1.0)),
                    )
                    _append_coverage_rows(spectral_rows, _prv_tv_display_label(k), stats)
    return hr_rows, ts_rows, spectral_rows


def _build_time_series_feature_rows(
    prv_summary: Optional[Dict[str, float]],
    hr_event_response_summary: Optional[Dict[str, float]] = None,
    pwa_drop_summary: Optional[Dict[str, float]] = None,
) -> List[List[str]]:
    rows: List[List[str]] = []
    if features.is_enabled("prv"):
        rmssd_mean = prv_summary.get("rmssd_mean") if prv_summary else None
        sdnn = prv_summary.get("sdnn") if prv_summary else None
        rmssd_median = prv_summary.get("rmssd_median") if prv_summary else None
        sdnn_mean = prv_summary.get("sdnn_mean") if prv_summary else None
        sdnn_median = prv_summary.get("sdnn_median") if prv_summary else None
        rows += [
            ["Selected-policy time-series features", ""],
            ["  RMSSD mean [ms]", _fmt(rmssd_mean, 2)],
            ["  RMSSD median [ms]", _fmt(rmssd_median, 2)],
            ["  RMSSD p75 / p90 [ms]", f"{_fmt(prv_summary.get('rmssd_p75') if prv_summary else None, 2)} / {_fmt(prv_summary.get('rmssd_p90') if prv_summary else None, 2)}"],
            ["  RMSSD IQR [ms]", _fmt(prv_summary.get("rmssd_iqr") if prv_summary else None, 2)],
            ["  RMSSD p90/median [-]", _fmt(prv_summary.get("rmssd_p90_over_median") if prv_summary else None, 3)],
            ["  SDNN mean [ms]", _fmt(sdnn_mean, 2)],
            ["  SDNN median [ms]", _fmt(sdnn_median, 2)],
            ["  SDNN p75 / p90 [ms]", f"{_fmt(prv_summary.get('sdnn_p75') if prv_summary else None, 2)} / {_fmt(prv_summary.get('sdnn_p90') if prv_summary else None, 2)}"],
            ["  SDNN IQR [ms]", _fmt(prv_summary.get("sdnn_iqr") if prv_summary else None, 2)],
            ["  SDNN p90/median [-]", _fmt(prv_summary.get("sdnn_p90_over_median") if prv_summary else None, 3)],
            ["  IPI mean / median [ms]", f"{_fmt(prv_summary.get('ipi_mean_ms') if prv_summary else None, 2)} / {_fmt(prv_summary.get('ipi_median_ms') if prv_summary else None, 2)}"],
            ["  IPI p75 / p90 [ms]", f"{_fmt(prv_summary.get('ipi_ms_p75') if prv_summary else None, 2)} / {_fmt(prv_summary.get('ipi_ms_p90') if prv_summary else None, 2)}"],
        ]
    if features.is_enabled("delta_hr"):
        if rows:
            rows += [["", ""]]
        item = hr_event_response_summary if isinstance(hr_event_response_summary, dict) else {}
        rows += [
            ["Selected-policy event-response HR", ""],
            ["  Trough-to-peak response mean [bpm]", _fmt(item.get("trough_to_peak_response_mean"), 2)],
            ["  Mean-to-peak response mean [bpm]", _fmt(item.get("mean_to_peak_response_mean"), 2)],
            ["  Event windows used", _fmt_int(item.get("n_used_windows"))],
            ["  Event windows total", _fmt_int(item.get("n_event_windows"))],
        ]
    if features.is_enabled("pwa_drop"):
        if rows:
            rows += [["", ""]]
        item = pwa_drop_summary if isinstance(pwa_drop_summary, dict) else {}
        rows += [
            ["Selected-policy PWA-drop", ""],
            ["  Detected drops [n]", _fmt_int(item.get("n_drops"))],
            ["  Drop rate [/sleep h]", _fmt(item.get("drop_rate_per_sleep_hour"), 2)],
            ["  Mean amplitude [%]", _fmt(item.get("mean_amplitude_pct"), 2)],
            ["  Mean duration [s]", _fmt(item.get("mean_duration_sec"), 2)],
            ["  Mean AUC [%·s]", _fmt(item.get("mean_auc_pct_sec"), 2)],
            ["  Event-overlap drops [n]", _fmt_int(item.get("n_drops_event_overlap"))],
            ["  Event-overlap drops [%]", _fmt(item.get("event_overlap_pct"), 2)],
        ]
    return rows


def _build_spectral_feature_rows(
    prv_summary: Optional[Dict[str, float]],
    mayer_peak_freq: Optional[float],
    resp_peak_freq: Optional[float],
    psd_features: Optional[Dict[str, float]],
) -> List[List[str]]:
    rows: List[List[str]] = []
    if features.is_enabled("prv"):
        lf = prv_summary.get("lf") if prv_summary else None
        hf = prv_summary.get("hf") if prv_summary else None
        lf_hf = prv_summary.get("lf_hf") if prv_summary else None
        rows += [["Selected-policy spectral parameters", ""], ["  LF mean [ms^2]", _fmt(lf, 2)], ["  HF mean [ms^2]", _fmt(hf, 2)], ["  LF/HF mean [-]", _fmt(lf_hf, 2)]]
        if prv_summary:
            rows += [
                ["  LF median [ms^2]", _fmt(prv_summary.get("lf_fixed_median"), 2)],
                ["  LF p75 / p90 [ms^2]", f"{_fmt(prv_summary.get('lf_fixed_p75'), 2)} / {_fmt(prv_summary.get('lf_fixed_p90'), 2)}"],
                ["  LF IQR [ms^2]", _fmt(prv_summary.get("lf_fixed_iqr"), 2)],
                ["  HF median [ms^2]", _fmt(prv_summary.get("hf_fixed_median"), 2)],
                ["  HF p75 / p90 [ms^2]", f"{_fmt(prv_summary.get('hf_fixed_p75'), 2)} / {_fmt(prv_summary.get('hf_fixed_p90'), 2)}"],
                ["  HF IQR [ms^2]", _fmt(prv_summary.get("hf_fixed_iqr"), 2)],
                ["  LF/HF median [-]", _fmt(prv_summary.get("lf_hf_fixed_median"), 2)],
                ["  LF/HF p75 / p90 [-]", f"{_fmt(prv_summary.get('lf_hf_fixed_p75'), 2)} / {_fmt(prv_summary.get('lf_hf_fixed_p90'), 2)}"],
                ["  LF/HF IQR [-]", _fmt(prv_summary.get("lf_hf_fixed_iqr"), 2)],
                ["  Valid LF/HF windows [n]", _fmt_int(prv_summary.get("lf_hf_fixed_n_windows_valid"))],
                ["  Total LF/HF windows [n]", _fmt_int(prv_summary.get("lf_hf_fixed_n_windows_total"))],
                ["  Valid LF/HF [min]", _fmt_num(prv_summary.get("lf_hf_fixed_valid_min"), 1)],
                ["  Valid LF/HF [%]", _fmt_pct(prv_summary.get("lf_hf_fixed_valid_pct"), 1)],
                ["  Total LF/HF [min]", _fmt_num(prv_summary.get("lf_hf_fixed_total_min"), 1)],
                ["  LF/HF window [s]", _fmt(prv_summary.get("lf_hf_fixed_window_sec"), 0)],
                ["  LF/HF hop [s]", _fmt(prv_summary.get("lf_hf_fixed_hop_sec"), 0)],
            ]

    if features.is_enabled("psd"):
        if rows:
            rows += [["", ""]]
        rows += [["Selected-policy spectral analysis", ""], ["  Mayer peak [Hz]", _fmt(mayer_peak_freq, 3)], ["  Resp peak [Hz]", _fmt(resp_peak_freq, 3)]]
        if psd_features:
            rows += [["  PSD mode", str(psd_features.get("psd_mode", "matched"))], ["  VLF power (0.0033–0.04 Hz)", _fmt_sci(psd_features.get("pow_vlf"))], ["  Mayer power (0.04–0.15 Hz)", _fmt_sci(psd_features.get("pow_mayer"))], ["  Resp power (0.15–0.50 Hz)", _fmt_sci(psd_features.get("pow_resp"))], ["  Mayer power (norm)", _fmt_pct(psd_features.get("norm_mayer"), 1)], ["  Resp power (norm)", _fmt_pct(psd_features.get("norm_resp"), 1)], ["  Valid PSD windows", _fmt_int(psd_features.get("n_windows"))], ["  PSD diagnostic", str(psd_features.get("psd_diag_reason", "")) or "ok"]]
    return rows


def _build_event_response_rows(sleep_combo_summaries: Optional[Dict[str, Dict[str, object]]]) -> List[List[str]]:
    if not features.is_enabled("delta_hr") or not isinstance(sleep_combo_summaries, dict):
        return []
    rows: List[List[str]] = [["Event-response HR metrics", ""], ["Subset", "Tr-Pk resp | Mean-Pk dHR | windows used/total"]]
    for key in ["pre_sleep_wake", "all_sleep", "wake_sleep", "nrem", "deep", "rem"]:
        item = sleep_combo_summaries.get(key)
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", key))
        hr_response_obj = item.get("hr_event_response_summary")
        hr_response: Dict[str, Any] = hr_response_obj if isinstance(hr_response_obj, dict) else {}
        rows.append([
            label,
            f"{_fmt(hr_response.get('trough_to_peak_response_mean'), 2)} bpm | "
            f"{_fmt(hr_response.get('mean_to_peak_response_mean'), 2)} bpm | "
            f"{_fmt_int(hr_response.get('n_used_windows'))}/{_fmt_int(hr_response.get('n_event_windows'))}",
        ])
    return rows


def _build_pat_burden_rows(
    pat_burden: Optional[float],
    pat_burden_diag: Optional[Dict[str, float]],
) -> List[List[str]]:
    if not features.is_enabled("pat_burden"):
        return []
    rows: List[List[str]] = [["Selected-policy PAT burden", ""]]
    unit = "rel·min/h" if isinstance(pat_burden_diag, dict) and pat_burden_diag.get("relative") else "amp·min/h"
    burden_val = float(pat_burden) if pat_burden is not None and np.isfinite(pat_burden) else None
    rows += [[f"  Burden [{unit}]", _fmt(burden_val, 3)]]
    if isinstance(pat_burden_diag, dict):
        rows += [
            ["  Sleep hours", _fmt_num(pat_burden_diag.get("sleep_hours"), 2)],
            ["  Episodes total", _fmt_int(pat_burden_diag.get("n_episodes"))],
            ["  Episodes used", _fmt_int(pat_burden_diag.get("n_episodes_used"))],
            ["  Episodes skipped", _fmt_int(pat_burden_diag.get("n_episodes_skipped"))],
            ["  Total area [min]", _fmt_num(pat_burden_diag.get("total_area_min"), 2)],
            ["  PAT AMP finite [min]", _fmt_num(pat_burden_diag.get("pat_amp_finite_min"), 2)],
            ["  Selected sleep [min]", _fmt_num(pat_burden_diag.get("sleep_selected_min"), 2)],
            ["  Inside event/desat [min]", _fmt_num(pat_burden_diag.get("inside_event_desat_min"), 2)],
            ["  Inside event/desat finite [min]", _fmt_num(pat_burden_diag.get("inside_event_desat_finite_min"), 2)],
            ["  Invalid PAT AMP inside [min]", _fmt_num(pat_burden_diag.get("pat_amp_invalid_inside_min"), 2)],
            ["  Baseline lookback [s]", _fmt_num(pat_burden_diag.get("baseline_lookback_sec"), 0)],
            ["  Baseline percentile", _fmt_num(pat_burden_diag.get("baseline_pctl"), 0)],
            ["  Baseline min samples", _fmt_int(pat_burden_diag.get("baseline_min_samples"))],
            ["  Min episode [s]", _fmt_num(pat_burden_diag.get("min_episode_sec"), 0)],
        ]
        skipped = pat_burden_diag.get("skipped_reason_counts")
        if isinstance(skipped, dict) and skipped:
            rows += [["", ""], ["  Skipped episode reasons", ""]]
            for reason, count in sorted(skipped.items()):
                rows.append([f"    {reason}", _fmt_int(count)])
    return rows


def _format_hour_tick(x: float, _pos: float) -> str:
    total_minutes = int(round(float(x) * 60.0))
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours:d}:{minutes:02d}"


def _apply_front_page_mask_layers(
    ax,
    t_sec: np.ndarray,
    y: np.ndarray,
    aux_df: Optional["pd.DataFrame"],
) -> None:
    if aux_df is None or t_sec.size == 0:
        return
    if bool(getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False)):
        m_sleep_keep = sleep_mask.build_sleep_include_mask_for_times(t_sec, aux_df)
        if m_sleep_keep is not None:
            _shade_masked_regions(ax, t_sec=t_sec, masked=~np.asarray(m_sleep_keep, dtype=bool), color="#6c757d", alpha=0.10)
    m_evt_keep = io_aux_csv.build_time_exclusion_mask(t_sec, aux_df)
    if m_evt_keep is not None:
        _shade_masked_regions(ax, t_sec=t_sec, masked=~np.asarray(m_evt_keep, dtype=bool), color="#c1121f", alpha=0.08)
    m_keep = sleep_mask.build_global_include_mask_for_times(t_sec, aux_df, apply_sleep=True, apply_events=True)
    invalid_mask = ~np.isfinite(y)
    if m_keep is not None and np.size(m_keep) == np.size(invalid_mask):
        invalid_mask = invalid_mask & np.asarray(m_keep, dtype=bool)
    if np.any(invalid_mask):
        _shade_masked_regions(ax, t_sec=t_sec, masked=invalid_mask, color="#d4a017", alpha=0.22)


def _add_event_vascular_mask_legend(fig) -> None:
    handles = [
        Line2D([0], [0], color="#6c757d", linewidth=6, alpha=0.10, label="Stage-policy excluded"),
        Line2D([0], [0], color="#c1121f", linewidth=6, alpha=0.08, label="Event-excluded"),
        Line2D([0], [0], color="#d4a017", linewidth=6, alpha=0.22, label="Metric invalid"),
        Line2D([0], [0], color="tab:blue", linewidth=1.4, alpha=0.55, label="Event/desaturation markers"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=4,
        fontsize=6.5,
        frameon=False,
    )


def _panel_badge(ax, text: str) -> None:
    ax.text(
        0.99,
        0.95,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.78, edgecolor="none", pad=0.25),
        zorder=10,
    )


def _front_page_bin_edges(aux_df: Optional["pd.DataFrame"], *t_arrays: Optional[np.ndarray]) -> Optional[np.ndarray]:
    bin_sec = 60.0 * float(getattr(config, "SUMMARY_FRONT_PAGE_BIN_MINUTES", 15.0))
    if bin_sec <= 0:
        bin_sec = 900.0
    t_max = 0.0
    if aux_df is not None:
        time_col = getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec")
        if time_col in aux_df.columns:
            tt = np.asarray(aux_df[time_col].to_numpy(dtype=float), dtype=float)
            if np.any(np.isfinite(tt)):
                t_max = max(t_max, float(np.nanmax(tt)))
    for arr in t_arrays:
        if arr is None:
            continue
        aa = np.asarray(arr, dtype=float)
        if np.any(np.isfinite(aa)):
            t_max = max(t_max, float(np.nanmax(aa)))
    if t_max <= 0:
        return None
    n_bins = max(1, int(np.ceil(t_max / bin_sec)))
    return np.arange(0.0, (n_bins + 1) * bin_sec + 1e-9, bin_sec, dtype=float)


def _bin_centers(edges_sec: np.ndarray) -> np.ndarray:
    return 0.5 * (edges_sec[:-1] + edges_sec[1:])


def _binned_event_metric(
    events: Optional[list[Dict[str, float]]],
    edges_sec: np.ndarray,
    *,
    time_key: str,
    value_key: str,
    reducer: str = "mean",
) -> np.ndarray:
    out = np.full(edges_sec.size - 1, np.nan, dtype=float)
    if not events:
        return out
    for i in range(edges_sec.size - 1):
        a = float(edges_sec[i])
        b = float(edges_sec[i + 1])
        vals = []
        for ev in events:
            t = float(ev.get(time_key, np.nan))
            v = float(ev.get(value_key, np.nan))
            if np.isfinite(t) and np.isfinite(v) and a <= t < b:
                vals.append(v)
        if vals:
            out[i] = float(np.nansum(vals)) if reducer == "sum" else float(np.nanmean(vals))
    return out


def _binned_event_count(
    events: Optional[list[Dict[str, float]]],
    edges_sec: np.ndarray,
    *,
    time_key: str,
) -> np.ndarray:
    out = np.zeros(edges_sec.size - 1, dtype=float)
    if not events:
        return out
    for i in range(edges_sec.size - 1):
        a = float(edges_sec[i])
        b = float(edges_sec[i + 1])
        out[i] = float(sum(1 for ev in events if np.isfinite(float(ev.get(time_key, np.nan))) and a <= float(ev.get(time_key, np.nan)) < b))
    return out


def _binned_series_mean(t_sec: Optional[np.ndarray], y: Optional[np.ndarray], edges_sec: np.ndarray) -> np.ndarray:
    out = np.full(edges_sec.size - 1, np.nan, dtype=float)
    if t_sec is None or y is None:
        return out
    tt = np.asarray(t_sec, dtype=float)
    yy = np.asarray(y, dtype=float)
    if tt.size == 0 or yy.size != tt.size:
        return out
    for i in range(edges_sec.size - 1):
        a = float(edges_sec[i])
        b = float(edges_sec[i + 1])
        m = (tt >= a) & (tt < b) & np.isfinite(yy)
        if np.any(m):
            out[i] = float(np.nanmean(yy[m]))
    return out

def _binned_sleep_hours_for_edges(
    t_sec: Optional[np.ndarray],
    aux_df: Optional["pd.DataFrame"],
    edges_sec: np.ndarray,
) -> np.ndarray:
    out = np.full(edges_sec.size - 1, np.nan, dtype=float)
    if t_sec is None or aux_df is None:
        return out
    tt = np.asarray(t_sec, dtype=float)
    if tt.size < 2:
        return out
    try:
        policy = masking.policy_from_config()
        bundle = masking.build_mask_bundle(tt, aux_df, policy=policy)
    except Exception:
        return out
    sleep_keep = np.asarray(bundle.sleep_keep, dtype=bool)
    if sleep_keep.size != tt.size:
        return out
    dt = np.diff(tt)
    dt = np.clip(dt, 0.0, None)
    interval_keep = sleep_keep[:-1] & sleep_keep[1:]
    t_mid = 0.5 * (tt[:-1] + tt[1:])
    for i in range(edges_sec.size - 1):
        a = float(edges_sec[i])
        b = float(edges_sec[i + 1])
        m = (t_mid >= a) & (t_mid < b) & interval_keep
        out[i] = float(np.sum(dt[m]) / 3600.0)
    return out


def _plot_event_vascular_panel(
    ax,
    t_center_h: np.ndarray,
    values: np.ndarray,
    *,
    color: str,
    ylabel: str,
    label: str,
    badge: str,
    summary_value: Optional[float] = None,
    ci95: Optional[np.ndarray] = None,
) -> None:
    t_center_h = np.asarray(t_center_h, dtype=float)
    values = np.asarray(values, dtype=float)
    _plot_binned_series_with_support(
        ax,
        t_center_h,
        values,
        bin_sec=60.0 * float(getattr(config, "SUMMARY_FRONT_PAGE_BIN_MINUTES", 5.0)),
        color=color,
        linewidth=1.3,
        label=label,
        show_markers=True,
    )
    if ci95 is not None and np.size(ci95) == np.size(values):
        ok = np.isfinite(t_center_h) & np.isfinite(values) & np.isfinite(ci95)
        if np.any(ok):
            ax.errorbar(t_center_h[ok], values[ok], yerr=np.asarray(ci95, dtype=float)[ok], fmt="none", elinewidth=0.8, capsize=2, alpha=0.40, color=color, zorder=2)
    _add_summary_line(ax, summary_value, color=color)
    has_summary_line = False
    if summary_value is not None:
        try:
            has_summary_line = bool(np.isfinite(float(summary_value)))
        except Exception:
            has_summary_line = False
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.75)
    _panel_badge(ax, badge)
    _add_metric_legend(ax, loc="lower right", fontsize=6, include_summary_lines=has_summary_line, summary_color=color, include_median_line=False)


def build_front_page(
    *,
    edf_base: str,
    aux_df: Optional["pd.DataFrame"],
    t_hr_calc: Optional[np.ndarray],
    hr_calc: Optional[np.ndarray],
    hr_calc_raw: Optional[np.ndarray],
    hr_event_windows: Optional[list[Dict[str, float]]],
    prv_summary: Optional[Dict[str, float]],
    t_prv: Optional[np.ndarray],
    prv_clean: Optional[np.ndarray],
    hr_event_response_summary: Optional[Dict[str, float]],
    pat_burden: Optional[float],
    pat_burden_diag: Optional[Dict[str, float]],
    pat_burden_episodes: Optional[list[Dict[str, float]]],
    t_pat_amp: Optional[np.ndarray],
    pat_amp: Optional[np.ndarray],
    pwa_drop_summary: Optional[Dict[str, float]],
    t_pwa: Optional[np.ndarray],
    pwa_series: Optional[np.ndarray],
    pwa_drop_events: Optional[list[Dict[str, float]]],
    t_spo2: Optional[np.ndarray] = None,
    spo2: Optional[np.ndarray] = None,
    event_spec: Optional[list[Any]] = None,
):
    mode = str(getattr(config, "SUMMARY_FRONT_PAGE_MODE", "prv")).lower()
    if mode != "event_vascular":
        return None

    edges_sec = _front_page_bin_edges(aux_df, t_hr_calc, t_prv, t_pat_amp, t_pwa)
    if edges_sec is None or edges_sec.size < 2:
        return None

    bin_sec = 60.0 * float(getattr(config, "SUMMARY_FRONT_PAGE_BIN_MINUTES", 5.0))
    bin_min = bin_sec / 60.0
    centers_sec = _bin_centers(edges_sec)
    centers_h = centers_sec / 3600.0

    panels: list[dict[str, Any]] = []
    if features.is_enabled("hr") and t_hr_calc is not None and hr_calc is not None:
        t_bin_h, y_bin, y_ci = _bin_series_mean_ci(
            np.asarray(t_hr_calc, dtype=float),
            np.asarray(hr_calc, dtype=float),
            bin_sec=bin_sec,
            min_count=int(getattr(config, "PRV_PLOT_BIN_MIN_COUNT", 3)),
        )
        hr_mean = float(np.nanmean(np.asarray(hr_calc, dtype=float))) if np.any(np.isfinite(np.asarray(hr_calc, dtype=float))) else np.nan
        panels.append({
            "key": "hr",
            "t_h": t_bin_h,
            "y": y_bin,
            "ci": y_ci,
            "color": "tab:green",
            "ylabel": "PAT Derived\nHR [bpm]",
            "label": "PAT Derived HR",
            "summary": hr_mean,
            "badge": f"Mean HR {_fmt(hr_mean, 1)} bpm",
        })

    if features.is_enabled("pwa_drop"):
        pwa_vals = _binned_event_count(pwa_drop_events, edges_sec, time_key="t_center")
        pwa_summary = None if not isinstance(pwa_drop_summary, dict) else pwa_drop_summary.get("drop_rate_per_sleep_hour")
        panels.append({
            "key": "pwa_drop",
            "t_h": centers_h,
            "y": pwa_vals,
            "ci": None,
            "color": "tab:purple",
            "ylabel": "PWA Drops\n[n/bin]",
            "label": "PWA Drops",
            "summary": None,
            "badge": (
                f"n {_fmt_int(pwa_drop_summary.get('n_drops'))} | rate {_fmt(pwa_summary, 2)}/h | amp {_fmt(pwa_drop_summary.get('mean_amplitude_pct'), 1)}%"
                if isinstance(pwa_drop_summary, dict)
                else "PWA-drop unavailable"
            ),
        })

    if bool(getattr(config, "ENABLE_SPO2_VALIDATION_PLOTS", False)) and t_spo2 is not None and spo2 is not None:
        t_bin_h, y_bin, y_ci = _bin_series_mean_ci(
            np.asarray(t_spo2, dtype=float),
            np.asarray(spo2, dtype=float),
            bin_sec=bin_sec,
            min_count=int(getattr(config, "PRV_PLOT_BIN_MIN_COUNT", 3)),
        )
        spo2_mean = float(np.nanmean(np.asarray(spo2, dtype=float))) if np.any(np.isfinite(np.asarray(spo2, dtype=float))) else np.nan
        panels.append({
            "key": "spo2",
            "t_h": t_bin_h,
            "y": y_bin,
            "ci": y_ci,
            "color": "tab:red",
            "ylabel": "SpO2\n[%]",
            "label": "SpO2",
            "summary": spo2_mean,
            "badge": f"Mean SpO2 {_fmt(spo2_mean, 1)}%",
        })

    if features.is_enabled("pat_burden"):
        burden_area = _binned_event_metric(pat_burden_episodes, edges_sec, time_key="t_start", value_key="area_min", reducer="sum")
        sleep_h = _binned_sleep_hours_for_edges(t_pat_amp, aux_df, edges_sec)
        burden_vals = np.full_like(burden_area, np.nan, dtype=float)
        ok_sleep = np.isfinite(sleep_h) & (sleep_h > 0)
        burden_area = np.where(np.isfinite(burden_area), burden_area, 0.0)
        burden_vals[ok_sleep] = burden_area[ok_sleep] / sleep_h[ok_sleep]
        unit = "rel·min/h" if isinstance(pat_burden_diag, dict) and pat_burden_diag.get("relative") else "amp·min/h"
        panels.append({
            "key": "pat_burden",
            "t_h": centers_h,
            "y": burden_vals,
            "ci": None,
            "color": "#2a9d8f",
            "ylabel": f"PAT burden\n[{unit}]",
            "label": "PAT burden",
            "summary": pat_burden,
            "badge": (
                f"Selected {_fmt(pat_burden, 3)} {unit} | episodes "
                f"{_fmt_int(pat_burden_diag.get('n_episodes_used')) if isinstance(pat_burden_diag, dict) else 'NA'}/{_fmt_int(pat_burden_diag.get('n_episodes')) if isinstance(pat_burden_diag, dict) else 'NA'}"
                if pat_burden is not None or isinstance(pat_burden_diag, dict)
                else "PAT burden unavailable"
            ),
        })

    if features.is_enabled("delta_hr"):
        delta_vals = _binned_event_metric(hr_event_windows, edges_sec, time_key="event_start_t", value_key="mean_to_peak_response", reducer="mean")
        delta_summary = None if not isinstance(hr_event_response_summary, dict) else hr_event_response_summary.get("mean_to_peak_response_mean")
        panels.append({
            "key": "delta_hr",
            "t_h": centers_h,
            "y": delta_vals,
            "ci": None,
            "color": "tab:blue",
            "ylabel": "dHR\n[bpm]",
            "label": "Delta-HR",
            "summary": delta_summary,
            "badge": (
                f"Tr-Pk {_fmt(hr_event_response_summary.get('trough_to_peak_response_mean'), 2)} bpm | "
                f"Mean-Pk {_fmt(delta_summary, 2)} bpm | "
                f"used/tot {_fmt_int(hr_event_response_summary.get('n_used_windows'))}/{_fmt_int(hr_event_response_summary.get('n_event_windows'))}"
                if isinstance(hr_event_response_summary, dict)
                else "Delta-HR unavailable"
            ),
        })

    if not panels:
        return None

    fig = plt.figure(figsize=(11.69, 8.27))
    gs = fig.add_gridspec(1 + len(panels), 1, height_ratios=[0.7] + [1.0] * len(panels))
    fig._event_key_y = 0.965
    ax_h = fig.add_subplot(gs[0])
    data_axes = [fig.add_subplot(gs[i + 1], sharex=ax_h) for i in range(len(panels))]
    ok = _plot_sleep_stagegram_on_axis(
        ax_h,
        edf_base=edf_base,
        aux_df=aux_df,
        title="Overnight Event-Related Vascular Response Overview",
        show_stats=False,
        title_pad=2.0,
    )
    if not ok:
        plt.close(fig)
        return None

    if event_spec is not None:
        _add_colored_event_key(fig, list(event_spec))

    fig.text(
        0.5,
        0.935,
        f"Selected-policy event-related vascular response summary. HR is displayed as {bin_min:.0f} min binned means with 95% CI. "
        f"PWA-drop is count per {bin_min:.0f} min bin. PAT burden is event/desat drop area normalized by selected sleep hours in each bin. "
        f"Delta-HR is mean event-window HR rise per bin. Dashed lines show selected-policy summary values where applicable.",
        ha="center",
        va="top",
        fontsize=7.5,
    )
    _add_event_vascular_mask_legend(fig)

    if aux_df is not None and event_spec is not None:
        x_end = float(edges_sec[-1]) if edges_sec.size else 0.0
        for ax in data_axes:
            _overlay_events_on_single_axis_whole_night(
                ax=ax,
                aux_df=aux_df,
                start_sec=0.0,
                end_sec=x_end,
                event_spec=list(event_spec),
                show_legend_labels=False,
                event_style="short",
            )
    for ax, panel in zip(data_axes, panels):
        if panel["key"] == "hr":
            _apply_front_page_mask_layers(ax, np.asarray(t_hr_calc, dtype=float), np.asarray(hr_calc, dtype=float), aux_df)
        elif panel["key"] == "pat_burden" and t_pat_amp is not None and pat_amp is not None:
            _apply_front_page_mask_layers(ax, np.asarray(t_pat_amp, dtype=float), np.asarray(pat_amp, dtype=float), aux_df)
        elif panel["key"] == "pwa_drop" and t_pwa is not None and pwa_series is not None:
            _apply_front_page_mask_layers(ax, np.asarray(t_pwa, dtype=float), np.asarray(pwa_series, dtype=float), aux_df)
        elif panel["key"] == "spo2" and t_spo2 is not None and spo2 is not None:
            _apply_front_page_mask_layers(ax, np.asarray(t_spo2, dtype=float), np.asarray(spo2, dtype=float), aux_df)
        _plot_event_vascular_panel(
            ax,
            panel["t_h"],
            panel["y"],
            color=panel["color"],
            ylabel=panel["ylabel"],
            label=panel["label"],
            badge=panel["badge"],
            summary_value=panel["summary"],
            ci95=panel["ci"],
        )

    for ax in data_axes:
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.xaxis.set_major_formatter(FuncFormatter(_format_hour_tick))
        ax.set_xlim(float(edges_sec[0]) / 3600.0, float(edges_sec[-1]) / 3600.0)
    for ax in data_axes[:-1]:
        ax.tick_params(labelbottom=False)
    data_axes[-1].set_xlabel("Time since recording start [hours]")
    fig.tight_layout(rect=(0.03, 0.03, 0.98, 0.85))
    return fig


def _render_table_page(
    title: str,
    rows: List[List[str]],
    *,
    edf_base: str,
    font_size: int = 11,
    scale_y: float = 1.45,
):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")

    wrapped_rows: List[List[str]] = []
    for row in rows:
        metric = str(row[0]) if len(row) > 0 else ""
        value = str(row[1]) if len(row) > 1 else ""
        if metric and "\n" not in metric and len(metric) > 34:
            metric = textwrap.fill(metric, width=34)
        if value and "\n" not in value and len(value) > 42:
            value = textwrap.fill(value, width=42)
        wrapped_rows.append([metric, value])

    table = ax.table(cellText=wrapped_rows, colLabels=["Metric", "Value"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            continue
        row_idx = r - 1
        if 0 <= row_idx < len(wrapped_rows):
            metric = str(wrapped_rows[row_idx][0]) if len(wrapped_rows[row_idx]) > 0 else ""
            value = str(wrapped_rows[row_idx][1]) if len(wrapped_rows[row_idx]) > 1 else ""
            if metric and not value:
                cell.set_text_props(weight="bold")
    n_rows = len(rows) + 1
    if n_rows > 36:
        scale_y_eff = min(scale_y, 1.05)
    elif n_rows > 30:
        scale_y_eff = min(scale_y, 1.20)
    else:
        scale_y_eff = scale_y
    table.scale(1.15, scale_y_eff)
    for row_idx, row in enumerate(wrapped_rows, start=1):
        if len(row) < 2:
            continue
        metric = str(row[0])
        value = str(row[1])
        line_count = max(metric.count("\n") + 1, value.count("\n") + 1, 1)
        if line_count <= 1:
            continue
        base_h = table[(row_idx, 0)].get_height()
        extra_scale = 1.0 + 0.75 * (line_count - 1)
        for col_idx in range(2):
            table[(row_idx, col_idx)].set_height(base_h * extra_scale)
    ax.set_title(f"{edf_base} – {title}", fontsize=16, pad=18)
    fig.tight_layout(rect=(0.0, 0.02, 0.82, 1.0))
    return fig


def _render_comparison_table_page(
    title: str,
    rows: List[List[str]],
    *,
    edf_base: str,
    headers: List[str],
    font_size: int = 11,
    scale_y: float = 1.35,
):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
    table.auto_set_column_width(col=list(range(len(headers))))
    table.scale(1.0, scale_y)
    ax.set_title(f"{edf_base} – {title}", fontsize=16, pad=18)
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.98))
    return fig


def _build_midpoint_half_rows(prv_midpoint_halves: Optional[Dict[str, Dict[str, float]]]) -> List[List[str]]:
    if not prv_midpoint_halves:
        return []
    first = prv_midpoint_halves.get("first_half") or {}
    second = prv_midpoint_halves.get("second_half") or {}
    if not first and not second:
        return []
    return [
        ["RMSSD mean [ms]", _fmt(first.get("rmssd_mean"), 2), _fmt(second.get("rmssd_mean"), 2)],
        ["RMSSD median [ms]", _fmt(first.get("rmssd_median"), 2), _fmt(second.get("rmssd_median"), 2)],
        ["RMSSD p75 [ms]", _fmt(first.get("rmssd_p75"), 2), _fmt(second.get("rmssd_p75"), 2)],
        ["RMSSD p90 [ms]", _fmt(first.get("rmssd_p90"), 2), _fmt(second.get("rmssd_p90"), 2)],
        ["RMSSD IQR [ms]", _fmt(first.get("rmssd_iqr"), 2), _fmt(second.get("rmssd_iqr"), 2)],
        ["SDNN mean [ms]", _fmt(first.get("sdnn_mean"), 2), _fmt(second.get("sdnn_mean"), 2)],
        ["SDNN median [ms]", _fmt(first.get("sdnn_median"), 2), _fmt(second.get("sdnn_median"), 2)],
        ["SDNN p75 [ms]", _fmt(first.get("sdnn_p75"), 2), _fmt(second.get("sdnn_p75"), 2)],
        ["SDNN p90 [ms]", _fmt(first.get("sdnn_p90"), 2), _fmt(second.get("sdnn_p90"), 2)],
        ["SDNN IQR [ms]", _fmt(first.get("sdnn_iqr"), 2), _fmt(second.get("sdnn_iqr"), 2)],
        ["IPI median [ms]", _fmt(first.get("ipi_median_ms"), 2), _fmt(second.get("ipi_median_ms"), 2)],
        ["LF mean [ms^2]", _fmt(first.get("lf"), 2), _fmt(second.get("lf"), 2)],
        ["LF median [ms^2]", _fmt(first.get("lf_fixed_median"), 2), _fmt(second.get("lf_fixed_median"), 2)],
        ["LF p90 [ms^2]", _fmt(first.get("lf_fixed_p90"), 2), _fmt(second.get("lf_fixed_p90"), 2)],
        ["HF mean [ms^2]", _fmt(first.get("hf"), 2), _fmt(second.get("hf"), 2)],
        ["HF median [ms^2]", _fmt(first.get("hf_fixed_median"), 2), _fmt(second.get("hf_fixed_median"), 2)],
        ["HF p90 [ms^2]", _fmt(first.get("hf_fixed_p90"), 2), _fmt(second.get("hf_fixed_p90"), 2)],
        ["LF/HF mean [-]", _fmt(first.get("lf_hf"), 2), _fmt(second.get("lf_hf"), 2)],
        ["LF/HF median [-]", _fmt(first.get("lf_hf_fixed_median"), 2), _fmt(second.get("lf_hf_fixed_median"), 2)],
        ["LF/HF p90 [-]", _fmt(first.get("lf_hf_fixed_p90"), 2), _fmt(second.get("lf_hf_fixed_p90"), 2)],
        ["Valid LF/HF [min]", _fmt_num(first.get("lf_hf_fixed_valid_min"), 1), _fmt_num(second.get("lf_hf_fixed_valid_min"), 1)],
    ]


def build_summary_pages(
    edf_base: str,
    pearson_r: Optional[float],
    spear_rho: Optional[float],
    rmse: Optional[float],
    prv_summary: Optional[Dict[str, float]],
    mayer_peak_freq: Optional[float],
    resp_peak_freq: Optional[float],
    aux_df: Optional["pd.DataFrame"],
    *,
    t_hr_calc: Optional[np.ndarray] = None,
    hr_calc: Optional[np.ndarray] = None,
    t_hr_edf: Optional[np.ndarray] = None,
    hr_edf: Optional[np.ndarray] = None,
    t_prv: Optional[np.ndarray] = None,
    prv_clean: Optional[np.ndarray] = None,
    prv_raw: Optional[np.ndarray] = None,
    prv_tv: Optional[Dict[str, np.ndarray]] = None,
    psd_features: Optional[Dict[str, float]] = None,
    pat_burden: Optional[float] = None,
    pat_burden_diag: Optional[Dict[str, float]] = None,
    pwa_drop_summary: Optional[Dict[str, float]] = None,
    sleep_combo_summaries: Optional[Dict[str, Dict[str, object]]] = None,
    prv_mask_info: Optional[Dict[str, object]] = None,
    prv_midpoint_halves: Optional[Dict[str, Dict[str, float]]] = None,
    hr_event_response_summary: Optional[Dict[str, float]] = None,
):
    pearson_r = None
    spear_rho = None
    rmse = None
    t_hr_edf = None
    hr_edf = None
    figs = []
    has_aux_summary_context = features.any_enabled("prv", "psd", "delta_hr", "pat_burden", "pwa_drop", "sleep_combo_summary")

    rows_hr_quality, rows_ts_coverage, rows_spectral_coverage = _build_quality_rows(t_hr_calc, hr_calc, t_prv, prv_clean, prv_raw, prv_tv)

    if rows_hr_quality:
        figs.append(_render_table_page("Summary (Selected-Policy HR & Coverage)", rows_hr_quality, edf_base=edf_base, font_size=12, scale_y=1.55))

    rows_ts_features = _build_time_series_feature_rows(prv_summary, hr_event_response_summary, pwa_drop_summary)
    if rows_ts_features:
        figs.append(_render_table_page("Summary (Selected-Policy Time-Series Features)", rows_ts_features, edf_base=edf_base, font_size=12, scale_y=1.35))

    rows_pat_burden = _build_pat_burden_rows(pat_burden, pat_burden_diag)
    if rows_pat_burden:
        figs.append(_render_table_page("Summary (Selected-Policy PAT Burden)", rows_pat_burden, edf_base=edf_base, font_size=12, scale_y=1.35))

    if rows_ts_coverage:
        figs.append(_render_table_page("Summary (Selected-Policy Time-Series Coverage)", rows_ts_coverage, edf_base=edf_base, font_size=12, scale_y=1.35))

    rows_spectral = _build_spectral_feature_rows(prv_summary, mayer_peak_freq, resp_peak_freq, psd_features)
    if rows_spectral:
        figs.append(_render_table_page("Summary (Selected-Policy Spectral Parameters)", rows_spectral, edf_base=edf_base, font_size=12, scale_y=1.35))

    if rows_spectral_coverage:
        figs.append(_render_table_page("Summary (Selected-Policy Spectral Coverage)", rows_spectral_coverage, edf_base=edf_base, font_size=12, scale_y=1.35))

    rows_sleep_timing = _build_sleep_timing_rows(aux_df) if has_aux_summary_context else []
    if rows_sleep_timing:
        figs.append(_render_table_page("Summary (Sleep Timing)", rows_sleep_timing, edf_base=edf_base, font_size=12, scale_y=1.35))

    rows_midpoint_halves = _build_midpoint_half_rows(prv_midpoint_halves)
    if rows_midpoint_halves:
        figs.append(
            _render_comparison_table_page(
                "Summary (NREM Sleep Midpoint PRV Halves)",
                rows_midpoint_halves,
                edf_base=edf_base,
                headers=["Metric", "NREM first half", "NREM second half"],
                font_size=12,
                scale_y=1.4,
            )
        )

    rows_p3 = _build_mask_breakdown_rows(t_prv, prv_mask_info)
    if rows_p3:
        figs.append(_render_table_page("Summary (Selected-Policy Exclusion Breakdown)", rows_p3, edf_base=edf_base, font_size=12, scale_y=1.35))

    combo_primary_rows, combo_secondary_rows, combo_right_rows = _sleep_combo_tables(sleep_combo_summaries) if features.is_enabled("sleep_combo_summary") else ([], [], [])
    if combo_primary_rows:
        figs.append(_render_sleep_combo_page(edf_base, combo_primary_rows, combo_secondary_rows, combo_right_rows))

    rows_p4: List[List[str]] = []
    if has_aux_summary_context and aux_df is not None:
        policy = masking.policy_from_config()
        aux_total = len(aux_df)
        rows_p4 += [["Overall event summary (aux CSV)", ""], ["  Samples (rows)", f"{aux_total:d}"], ["  Active exclusion columns", _wrap_csv_columns(list(policy.exclusion_columns))]]
        rows_p4 += _active_aux_flag_rows(aux_df, policy)
        rows_p4 += [["", ""]]
        rows_p4 += _sleep_stage_rows(aux_df)
    elif has_aux_summary_context:
        rows_p4 += [["Event summary", "No aux_df available"]]
    if rows_p4:
        fig4 = _render_table_page("Summary (Events, Sleep Stages & Selected Policy)", rows_p4, edf_base=edf_base, font_size=12, scale_y=1.25)
        figs.append(fig4)
    return figs
