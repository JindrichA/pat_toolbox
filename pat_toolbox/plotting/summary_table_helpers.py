from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from .. import config, features, masking
from ..io.aux_events import compute_sleep_timing_from_aux
from .utils import _count_flags, _fmt

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


def _append_coverage_rows(rows: List[List[str]], label: str, stats: Dict[str, Optional[float]]) -> None:
    rows.append([f"  {label} valid [min]", _fmt_num(stats.get("valid_min"), 1)])
    rows.append([f"  {label} valid [%]", _fmt_pct(stats.get("valid_pct"), 1)])


def _hrv_tv_display_label(key: str) -> str:
    mapping = {
        "sdnn_ms_raw": "SDNN pre-exclusion",
        "sdnn_ms": "SDNN clean",
        "lf_raw": "LF pre-exclusion",
        "lf": "LF clean",
        "hf_raw": "HF pre-exclusion",
        "hf": "HF clean",
        "lf_hf_raw": "LF/HF pre-exclusion",
        "lf_hf": "LF/HF clean",
    }
    return mapping.get(key, key)


def _build_mask_breakdown_rows(
    t_hrv: Optional[np.ndarray],
    hrv_mask_info: Optional[Dict[str, object]],
) -> List[List[str]]:
    if t_hrv is None or not hrv_mask_info:
        return []

    sleep_keep = hrv_mask_info.get("sleep_keep")
    apnea_keep = hrv_mask_info.get("apnea_keep")
    quality_keep = hrv_mask_info.get("quality_keep")
    desat_keep = hrv_mask_info.get("desat_keep")
    combined_keep = hrv_mask_info.get("combined_keep")

    if not all(isinstance(v, np.ndarray) and np.size(v) == np.size(t_hrv) for v in [sleep_keep, apnea_keep, quality_keep, desat_keep, combined_keep]):
        return []

    dt_sec = _sample_dt_sec(t_hrv, float(getattr(config, "HRV_TARGET_FS_HZ", 1.0)))
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
    rows += [["", ""], ["Note", "Mask-based only; upstream RR cleaning / PAT artifact loss is not included here."]]
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


def _sleep_combo_row_values(item: Dict[str, Any]) -> tuple[str, list[str]]:
    label = str(item.get("label", "subset"))
    sleep_hours = _fmt_num(item.get("sleep_hours"), 2)
    hr_summary_obj = item.get("hr_summary")
    hr_summary: Dict[str, Any] = hr_summary_obj if isinstance(hr_summary_obj, dict) else {}
    hrv_summary_obj = item.get("hrv_summary")
    hrv_summary: Dict[str, Any] = hrv_summary_obj if isinstance(hrv_summary_obj, dict) else {}
    psd_features_obj = item.get("psd_features")
    psd_features: Dict[str, Any] = psd_features_obj if isinstance(psd_features_obj, dict) else {}
    hr_response_obj = item.get("hr_event_response_summary")
    hr_response: Dict[str, Any] = hr_response_obj if isinstance(hr_response_obj, dict) else {}
    burden = item.get("pat_burden")

    left = [f"{sleep_hours} h"]
    if features.is_enabled("hr"):
        left.extend([
            f"{_fmt(hr_summary.get('mean'), 1)} bpm",
            f"{_fmt(hr_summary.get('median'), 1)} bpm",
            f"{_fmt(hr_summary.get('std'), 1)} bpm",
        ])
    if features.is_enabled("hrv"):
        left.extend([
            f"{_fmt(hrv_summary.get('rmssd_mean'), 1)} ms",
            f"{_fmt(hrv_summary.get('sdnn'), 1)} ms",
            f"{_fmt(hrv_summary.get('lf'), 2)}",
            f"{_fmt(hrv_summary.get('hf'), 2)}",
            f"{_fmt(hrv_summary.get('lf_hf'), 2)}",
        ])

    right: list[str] = []
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
    return label, left + right


def _sleep_combo_tables(sleep_combo_summaries: Optional[Dict[str, Dict[str, object]]]) -> tuple[list[list[str]], list[list[str]]]:
    if not sleep_combo_summaries:
        return [], []

    left_headers = ["Subset", "Sleep h"]
    if features.is_enabled("hr"):
        left_headers.extend(["HR mean", "HR med", "HR std"])
    if features.is_enabled("hrv"):
        left_headers.extend(["RMSSD", "SDNN", "LF fixed\n[ms^2]", "HF fixed\n[ms^2]", "LF/HF fixed\n[-]"])

    right_headers = ["Subset"]
    if features.is_enabled("psd"):
        right_headers.append("PSD win")
    if features.is_enabled("delta_hr"):
        right_headers.extend(["Tr-Pk resp", "Mean-Pk dHR", "win used/tot"])
    if features.is_enabled("pat_burden"):
        right_headers.append("Burden")

    left_rows: list[list[str]] = [left_headers]
    right_rows: list[list[str]] = [right_headers] if len(right_headers) > 1 else []

    for key in ["all_sleep", "wake_sleep", "nrem", "deep", "rem"]:
        item_obj = sleep_combo_summaries.get(key)
        if not isinstance(item_obj, dict):
            continue
        item: Dict[str, Any] = item_obj
        label, values = _sleep_combo_row_values(item)
        left_width = len(left_headers) - 1
        left_rows.append([label, *values[:left_width]])
        if right_rows:
            right_rows.append([label, *values[left_width:]])
    return left_rows, right_rows


def _render_sleep_combo_page(
    edf_base: str,
    left_rows: list[list[str]],
    right_rows: list[list[str]],
):
    fig, axes = plt.subplots(2 if right_rows else 1, 1, figsize=(11.69, 8.27))
    axes_list = [axes] if not isinstance(axes, np.ndarray) else list(axes)
    fig.suptitle(f"{edf_base} – Summary (Sleep-Subset Comparison)", fontsize=16, y=0.985)

    for ax in axes_list:
        ax.axis("off")

    left_table = axes_list[0].table(
        cellText=left_rows[1:],
        colLabels=left_rows[0],
        loc="center",
        cellLoc="left",
    )
    left_table.auto_set_font_size(False)
    left_table.set_fontsize(9)
    left_table.auto_set_column_width(col=list(range(len(left_rows[0]))))
    left_table.scale(1.0, 1.45)
    axes_list[0].set_title("Subset core metrics", fontsize=12, pad=10)
    axes_list[0].text(
        0.5,
        1.02,
        "HR mean/med/std = PAT-derived HR summary within each subset after the selected-policy combined mask\n"
        "(sleep + event/quality/desat exclusion as configured), not a residual signal.",
        transform=axes_list[0].transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
    )

    if right_rows and len(axes_list) > 1:
        axes_list[1].set_title("Subset event-response / burden metrics", fontsize=12, pad=56)
        axes_list[1].text(
            0.5,
            1.02,
            "Tr-Pk resp = mean(recovery max HR - event-window trough) across valid events\n"
            "Mean-Pk dHR = mean(recovery max HR - event-window mean HR) across valid events\n"
            "win used/tot = valid event windows / all detected event windows",
            transform=axes_list[1].transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
        )
        right_table = axes_list[1].table(
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
    t_hrv: Optional[np.ndarray],
    hrv_clean: Optional[np.ndarray],
    hrv_raw: Optional[np.ndarray],
    hrv_tv: Optional[Dict[str, np.ndarray]],
) -> List[List[str]]:
    rows: List[List[str]] = []
    if features.is_enabled("hr"):
        hr_cov = _coverage_stats(
            hr_calc,
            t=t_hr,
            default_fs=float(getattr(config, "HR_TARGET_FS_HZ", 1.0)),
        )
        hr_pat_stats = _finite_stats(hr_calc)
        rows += [["PAT-derived HR summary", ""], ["  HR n used", _fmt_int(hr_pat_stats["n_used"])], ["  HR min / max [bpm]", f"{_fmt_num(hr_pat_stats['min'], 2)} / {_fmt_num(hr_pat_stats['max'], 2)}"], ["  HR mean / median [bpm]", f"{_fmt_num(hr_pat_stats['mean'], 2)} / {_fmt_num(hr_pat_stats['median'], 2)}"], ["  HR std [bpm]", _fmt_num(hr_pat_stats["std"], 2)], ["", ""], ["Valid signal coverage", ""]]
        _append_coverage_rows(rows, "HR (PAT-derived)", hr_cov)

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
        if rows and rows[-1] != ["", ""]:
            rows += [["", ""]]
        if not rows:
            rows += [["Valid signal coverage", ""]]
        _append_coverage_rows(rows, "HRV RMSSD clean", hrv_clean_cov)
        _append_coverage_rows(rows, "HRV RMSSD raw", hrv_raw_cov)
        if isinstance(hrv_tv, dict) and len(hrv_tv) > 0:
            rows += [["", ""], ["Sliding-window HRV valid coverage", ""]]
            for k, v in sorted(hrv_tv.items()):
                if k == "tv_window_sec":
                    continue
                stats = _coverage_stats(
                    v,
                    t=t_hrv,
                    default_fs=float(getattr(config, "HRV_TARGET_FS_HZ", 1.0)),
                )
                _append_coverage_rows(rows, _hrv_tv_display_label(k), stats)
    return rows


def _build_hrv_psd_rows(
    hrv_summary: Optional[Dict[str, float]],
    mayer_peak_freq: Optional[float],
    resp_peak_freq: Optional[float],
    psd_features: Optional[Dict[str, float]],
) -> List[List[str]]:
    rows: List[List[str]] = []
    if features.is_enabled("hrv"):
        rmssd_mean = hrv_summary.get("rmssd_mean") if hrv_summary else None
        rmssd_median = hrv_summary.get("rmssd_median") if hrv_summary else None
        sdnn = hrv_summary.get("sdnn") if hrv_summary else None
        lf = hrv_summary.get("lf") if hrv_summary else None
        hf = hrv_summary.get("hf") if hrv_summary else None
        lf_hf = hrv_summary.get("lf_hf") if hrv_summary else None
        rows += [["Selected-policy HRV summary (clean RR)", ""], ["  RMSSD mean [ms]", _fmt(rmssd_mean, 2)], ["  RMSSD median [ms]", _fmt(rmssd_median, 2)], ["  SDNN [ms]", _fmt(sdnn, 2)], ["  LF fixed-window mean [ms^2]", _fmt(lf, 2)], ["  HF fixed-window mean [ms^2]", _fmt(hf, 2)], ["  LF/HF fixed-window mean [-]", _fmt(lf_hf, 2)]]
        if hrv_summary:
            rows += [["  LF fixed-window median [ms^2]", _fmt(hrv_summary.get("lf_fixed_median"), 2)], ["  HF fixed-window median [ms^2]", _fmt(hrv_summary.get("hf_fixed_median"), 2)], ["  LF/HF fixed-window median [-]", _fmt(hrv_summary.get("lf_hf_fixed_median"), 2)], ["  Legacy segmented LF segments used", _fmt_int(hrv_summary.get("lf_n_segments_used"))], ["  Fixed-window valid windows [n]", _fmt_int(hrv_summary.get("lf_hf_fixed_n_windows_valid"))], ["  Fixed-window total windows [n]", _fmt_int(hrv_summary.get("lf_hf_fixed_n_windows_total"))], ["  Fixed-window valid [min]", _fmt_num(hrv_summary.get("lf_hf_fixed_valid_min"), 1)], ["  Fixed-window valid [%]", _fmt_pct(hrv_summary.get("lf_hf_fixed_valid_pct"), 1)], ["  Fixed-window total [min]", _fmt_num(hrv_summary.get("lf_hf_fixed_total_min"), 1)], ["  Fixed window [s]", _fmt(hrv_summary.get("lf_hf_fixed_window_sec"), 0)], ["  Fixed hop [s]", _fmt(hrv_summary.get("lf_hf_fixed_hop_sec"), 0)]]

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
    for key in ["all_sleep", "wake_sleep", "nrem", "deep", "rem"]:
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
    table = ax.table(cellText=rows, colLabels=["Metric", "Value"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            continue
        row_idx = r - 1
        if 0 <= row_idx < len(rows):
            metric = str(rows[row_idx][0]) if len(rows[row_idx]) > 0 else ""
            value = str(rows[row_idx][1]) if len(rows[row_idx]) > 1 else ""
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
    for row_idx, row in enumerate(rows, start=1):
        if len(row) < 2:
            continue
        value = str(row[1])
        line_count = max(1, value.count("\n") + 1)
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


def _build_midpoint_half_rows(hrv_midpoint_halves: Optional[Dict[str, Dict[str, float]]]) -> List[List[str]]:
    if not hrv_midpoint_halves:
        return []
    first = hrv_midpoint_halves.get("first_half") or {}
    second = hrv_midpoint_halves.get("second_half") or {}
    if not first and not second:
        return []
    return [
        ["RMSSD mean [ms]", _fmt(first.get("rmssd_mean"), 2), _fmt(second.get("rmssd_mean"), 2)],
        ["SDNN [ms]", _fmt(first.get("sdnn"), 2), _fmt(second.get("sdnn"), 2)],
        ["LF fixed-window mean [ms^2]", _fmt(first.get("lf"), 2), _fmt(second.get("lf"), 2)],
        ["HF fixed-window mean [ms^2]", _fmt(first.get("hf"), 2), _fmt(second.get("hf"), 2)],
        ["LF/HF fixed-window mean [-]", _fmt(first.get("lf_hf"), 2), _fmt(second.get("lf_hf"), 2)],
        ["LF/HF fixed-window median [-]", _fmt(first.get("lf_hf_fixed_median"), 2), _fmt(second.get("lf_hf_fixed_median"), 2)],
        ["Fixed-window valid windows [n]", _fmt_int(first.get("lf_hf_fixed_n_windows_valid")), _fmt_int(second.get("lf_hf_fixed_n_windows_valid"))],
    ]


def build_summary_pages(
    edf_base: str,
    pearson_r: Optional[float],
    spear_rho: Optional[float],
    rmse: Optional[float],
    hrv_summary: Optional[Dict[str, float]],
    mayer_peak_freq: Optional[float],
    resp_peak_freq: Optional[float],
    aux_df: Optional["pd.DataFrame"],
    *,
    t_hr_calc: Optional[np.ndarray] = None,
    hr_calc: Optional[np.ndarray] = None,
    t_hr_edf: Optional[np.ndarray] = None,
    hr_edf: Optional[np.ndarray] = None,
    t_hrv: Optional[np.ndarray] = None,
    hrv_clean: Optional[np.ndarray] = None,
    hrv_raw: Optional[np.ndarray] = None,
    hrv_tv: Optional[Dict[str, np.ndarray]] = None,
    psd_features: Optional[Dict[str, float]] = None,
    pat_burden: Optional[float] = None,
    pat_burden_diag: Optional[Dict[str, float]] = None,
    sleep_combo_summaries: Optional[Dict[str, Dict[str, object]]] = None,
    hrv_mask_info: Optional[Dict[str, object]] = None,
    hrv_midpoint_halves: Optional[Dict[str, Dict[str, float]]] = None,
):
    pearson_r = None
    spear_rho = None
    rmse = None
    t_hr_edf = None
    hr_edf = None
    figs = []

    rows_p1 = _build_quality_rows(t_hr_calc, hr_calc, t_hrv, hrv_clean, hrv_raw, hrv_tv)

    if rows_p1:
        figs.append(_render_table_page("Summary (Selected-Policy HR & Quality)", rows_p1, edf_base=edf_base, font_size=12, scale_y=1.55))

    rows_p2 = _build_hrv_psd_rows(hrv_summary, mayer_peak_freq, resp_peak_freq, psd_features)

    if rows_p2:
        if features.is_enabled("hrv") and features.is_enabled("psd"):
            title = "Summary (Selected-Policy HRV & Spectral)"
        elif features.is_enabled("hrv"):
            title = "Summary (Selected-Policy HRV)"
        else:
            title = "Summary (Selected-Policy Spectral)"
        figs.append(_render_table_page(title, rows_p2, edf_base=edf_base, font_size=12, scale_y=1.35))

    rows_sleep_timing = _build_sleep_timing_rows(aux_df)
    if rows_sleep_timing:
        figs.append(_render_table_page("Summary (Sleep Timing)", rows_sleep_timing, edf_base=edf_base, font_size=12, scale_y=1.35))

    rows_midpoint_halves = _build_midpoint_half_rows(hrv_midpoint_halves)
    if rows_midpoint_halves:
        figs.append(
            _render_comparison_table_page(
                "Summary (Sleep Midpoint HRV Halves)",
                rows_midpoint_halves,
                edf_base=edf_base,
                headers=["Metric", "First half", "Second half"],
                font_size=12,
                scale_y=1.4,
            )
        )

    rows_p3 = _build_mask_breakdown_rows(t_hrv, hrv_mask_info)
    if rows_p3:
        figs.append(_render_table_page("Summary (Selected-Policy Exclusion Breakdown)", rows_p3, edf_base=edf_base, font_size=12, scale_y=1.35))

    combo_left_rows, combo_right_rows = _sleep_combo_tables(sleep_combo_summaries) if features.is_enabled("sleep_combo_summary") else ([], [])
    if combo_left_rows:
        figs.append(_render_sleep_combo_page(edf_base, combo_left_rows, combo_right_rows))

    rows_p4: List[List[str]] = []
    if aux_df is not None:
        policy = masking.policy_from_config()
        aux_total = len(aux_df)
        desat_n, desat_pct = _count_flags(aux_df, "desat_flag")
        excl_hr_n, excl_hr_pct = _count_flags(aux_df, "exclude_hr_flag")
        excl_pat_n, excl_pat_pct = _count_flags(aux_df, "exclude_pat_flag")
        cen3_n, cen3_pct = _count_flags(aux_df, "evt_central_3")
        obs3_n, obs3_pct = _count_flags(aux_df, "evt_obstructive_3")
        unc3_n, unc3_pct = _count_flags(aux_df, "evt_unclassified_3")
        cen4_n, cen4_pct = _count_flags(aux_df, "evt_central_4")
        obs4_n, obs4_pct = _count_flags(aux_df, "evt_obstructive_4")
        unc4_n, unc4_pct = _count_flags(aux_df, "evt_unclassified_4")
        rows_p4 += [["Overall event summary (aux CSV)", ""], ["  Samples (rows)", f"{aux_total:d}"], ["  Active exclusion columns", _wrap_csv_columns(list(policy.exclusion_columns))], ["  Desaturation flags", f"{desat_n:d} ({desat_pct})"], ["  Exclude HR flags", f"{excl_hr_n:d} ({excl_hr_pct})"], ["  Exclude PAT flags", f"{excl_pat_n:d} ({excl_pat_pct})"], ["  Central A/H 3%", f"{cen3_n:d} ({cen3_pct})"], ["  Obstructive A/H 3%", f"{obs3_n:d} ({obs3_pct})"], ["  Unclassified A/H 3%", f"{unc3_n:d} ({unc3_pct})"]]
        if (cen4_n + obs4_n + unc4_n) > 0:
            rows_p4 += [["  Central A/H 4%", f"{cen4_n:d} ({cen4_pct})"], ["  Obstructive A/H 4%", f"{obs4_n:d} ({obs4_pct})"], ["  Unclassified A/H 4%", f"{unc4_n:d} ({unc4_pct})"]]
        rows_p4 += [["", ""]]
        rows_p4 += _sleep_stage_rows(aux_df)
    else:
        rows_p4 += [["Event summary", "No aux_df available"]]
    if features.is_enabled("pat_burden"):
        rows_p4 += [["", ""], ["PAT burden (event+desat within included sleep)", ""]]
        burden_val = float(pat_burden) if pat_burden is not None and np.isfinite(pat_burden) else None
        unit = "rel·min/h" if isinstance(pat_burden_diag, dict) and pat_burden_diag.get("relative") else "amp·min/h"
        rows_p4 += [[f"  Burden [{unit}]", _fmt(burden_val, 3)]]
        if isinstance(pat_burden_diag, dict):
            rows_p4 += [["  Sleep hours", _fmt_num(pat_burden_diag.get("sleep_hours"), 2)], ["  Episodes (total)", _fmt_int(pat_burden_diag.get("n_episodes"))], ["  Episodes used", _fmt_int(pat_burden_diag.get("n_episodes_used"))], ["  Total area [min]", _fmt_num(pat_burden_diag.get("total_area_min"), 2)]]
    fig4 = _render_table_page("Summary (Events, Sleep Stages & Selected Policy)", rows_p4, edf_base=edf_base, font_size=12, scale_y=1.25)
    figs.append(fig4)
    return figs
