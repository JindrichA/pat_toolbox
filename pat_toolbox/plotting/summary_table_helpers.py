from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from .. import config, masking
from .utils import _count_flags, _fmt

if TYPE_CHECKING:
    import pandas as pd


def _nan_pct(x: Optional[np.ndarray]) -> Optional[float]:
    if x is None:
        return None
    x = np.asarray(x)
    if x.size == 0:
        return None
    return float(100.0 * np.mean(~np.isfinite(x)))


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


def _sleep_combo_rows(sleep_combo_summaries: Optional[Dict[str, Dict[str, object]]]) -> List[List[str]]:
    rows: List[List[str]] = []
    if not sleep_combo_summaries:
        return rows
    rows.append(["Sleep subset comparison", ""])
    rows.append(["Subset", "Sleep h | RMSSD | SDNN | LF/HF | PSD win | Burden"])
    for key in ["all_sleep", "wake_sleep", "nrem", "deep", "rem"]:
        item_obj = sleep_combo_summaries.get(key)
        if not isinstance(item_obj, dict):
            continue
        item: Dict[str, Any] = item_obj
        label = str(item.get("label", key))
        sleep_hours = _fmt_num(item.get("sleep_hours"), 2)
        hrv_summary_obj = item.get("hrv_summary")
        hrv_summary: Dict[str, Any] = hrv_summary_obj if isinstance(hrv_summary_obj, dict) else {}
        psd_features_obj = item.get("psd_features")
        psd_features: Dict[str, Any] = psd_features_obj if isinstance(psd_features_obj, dict) else {}
        burden = item.get("pat_burden")
        value = f"{sleep_hours} h | {_fmt(hrv_summary.get('rmssd_mean'), 1)} ms | {_fmt(hrv_summary.get('sdnn'), 1)} ms | {_fmt(hrv_summary.get('lf_hf'), 2)} | {_fmt_int(psd_features.get('n_windows'))} | {_fmt(burden, 3)}"
        rows.append([label, value])
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
    n_rows = len(rows) + 1
    if n_rows > 36:
        scale_y_eff = min(scale_y, 1.05)
    elif n_rows > 30:
        scale_y_eff = min(scale_y, 1.20)
    else:
        scale_y_eff = scale_y
    table.scale(1.15, scale_y_eff)
    ax.set_title(f"{edf_base} – {title}", fontsize=16, pad=18)
    fig.tight_layout(rect=[0, 0.02, 0.82, 1])
    return fig


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
    delta_hr_calc: Optional[np.ndarray] = None,
    delta_hr_edf: Optional[np.ndarray] = None,
    delta_hr_calc_evt: Optional[np.ndarray] = None,
    delta_hr_edf_evt: Optional[np.ndarray] = None,
    pat_burden: Optional[float] = None,
    pat_burden_diag: Optional[Dict[str, float]] = None,
    sleep_combo_summaries: Optional[Dict[str, Dict[str, object]]] = None,
):
    pearson_r = None
    spear_rho = None
    rmse = None
    t_hr_edf = None
    hr_edf = None
    delta_hr_edf = None
    delta_hr_edf_evt = None

    hr_pat_nan = _nan_pct(hr_calc)
    hrv_clean_nan = _nan_pct(hrv_clean)
    hrv_raw_nan = _nan_pct(hrv_raw)
    hr_pat_stats = _finite_stats(hr_calc)
    rows_p1 = [["PAT-derived HR summary", ""], ["  HR n used", _fmt_int(hr_pat_stats["n_used"])], ["  HR min / max [bpm]", f"{_fmt_num(hr_pat_stats['min'], 2)} / {_fmt_num(hr_pat_stats['max'], 2)}"], ["  HR mean / median [bpm]", f"{_fmt_num(hr_pat_stats['mean'], 2)} / {_fmt_num(hr_pat_stats['median'], 2)}"], ["  HR std [bpm]", _fmt_num(hr_pat_stats["std"], 2)], ["", ""], ["Signal quality (NaN %)", ""], ["  HR (PAT-derived) NaN %", _fmt_pct(hr_pat_nan, 1)], ["  HRV RMSSD clean NaN %", _fmt_pct(hrv_clean_nan, 1)], ["  HRV RMSSD raw NaN %", _fmt_pct(hrv_raw_nan, 1)]]
    if isinstance(hrv_tv, dict) and len(hrv_tv) > 0:
        rows_p1 += [["", ""], ["Time-varying HRV quality", ""]]
        for k, v in sorted(hrv_tv.items()):
            rows_p1.append([f"  {k} NaN %", _fmt_pct(_nan_pct(v), 1)])
    fig1 = _render_table_page("Summary (PAT HR & Quality)", rows_p1, edf_base=edf_base, font_size=12, scale_y=1.55)

    rmssd_mean = hrv_summary.get("rmssd_mean") if hrv_summary else None
    rmssd_median = hrv_summary.get("rmssd_median") if hrv_summary else None
    sdnn = hrv_summary.get("sdnn") if hrv_summary else None
    lf = hrv_summary.get("lf") if hrv_summary else None
    hf = hrv_summary.get("hf") if hrv_summary else None
    lf_hf = hrv_summary.get("lf_hf") if hrv_summary else None
    rows_p2 = [["HRV summary (clean RR, after masking/rejection)", ""], ["  RMSSD mean [ms]", _fmt(rmssd_mean, 2)], ["  RMSSD median [ms]", _fmt(rmssd_median, 2)], ["  SDNN [ms]", _fmt(sdnn, 2)], ["  LF", _fmt(lf, 4)], ["  HF", _fmt(hf, 4)], ["  LF/HF", _fmt(lf_hf, 2)]]
    if hrv_summary:
        rows_p2 += [["  LF segments used", _fmt_int(hrv_summary.get("lf_n_segments_used"))], ["  Fixed LF/HF median", _fmt(hrv_summary.get("lf_hf_fixed_median"), 2)], ["  Fixed LF/HF mean", _fmt(hrv_summary.get("lf_hf_fixed_mean"), 2)], ["  Fixed LF/HF valid windows", _fmt_int(hrv_summary.get("lf_hf_fixed_n_windows_valid"))], ["  Fixed LF/HF total windows", _fmt_int(hrv_summary.get("lf_hf_fixed_n_windows_total"))], ["  Fixed window [s]", _fmt(hrv_summary.get("lf_hf_fixed_window_sec"), 0)], ["  Fixed hop [s]", _fmt(hrv_summary.get("lf_hf_fixed_hop_sec"), 0)]]
    rows_p2 += [["", ""], ["Spectral analysis (PAT volume)", ""], ["  Mayer peak [Hz]", _fmt(mayer_peak_freq, 3)], ["  Resp peak [Hz]", _fmt(resp_peak_freq, 3)]]
    if psd_features:
        rows_p2 += [["  PSD mode", str(psd_features.get("psd_mode", "matched"))], ["  VLF power (0.0033–0.04 Hz)", _fmt_sci(psd_features.get("pow_vlf"))], ["  Mayer power (0.04–0.15 Hz)", _fmt_sci(psd_features.get("pow_mayer"))], ["  Resp power (0.15–0.50 Hz)", _fmt_sci(psd_features.get("pow_resp"))], ["  Mayer power (norm)", _fmt_pct(psd_features.get("norm_mayer"), 1)], ["  Resp power (norm)", _fmt_pct(psd_features.get("norm_resp"), 1)], ["  Valid PSD windows", _fmt_int(psd_features.get("n_windows"))], ["  PSD diagnostic", str(psd_features.get("psd_diag_reason", "")) or "ok"]]
    fig2 = _render_table_page("Summary (HRV & Spectral)", rows_p2, edf_base=edf_base, font_size=12, scale_y=1.35)

    d_pat_all = _finite_stats(delta_hr_calc)
    d_pat_evt = _finite_stats(delta_hr_calc_evt)
    rows_p3 = [["ΔHR baseline (all finite samples)", ""], ["  ΔHR n used", _fmt_int(d_pat_all["n_used"])], ["  ΔHR min / max [bpm]", f"{_fmt_num(d_pat_all['min'], 2)} / {_fmt_num(d_pat_all['max'], 2)}"], ["  ΔHR mean / median [bpm]", f"{_fmt_num(d_pat_all['mean'], 2)} / {_fmt_num(d_pat_all['median'], 2)}"], ["  ΔHR std [bpm]", _fmt_num(d_pat_all["std"], 2)], ["  ΔHR NaN %", _fmt_pct(d_pat_all["nan_pct"], 1)], ["", ""], ["ΔHR event-only (inside event/desat windows only)", ""], ["  ΔHR n used", _fmt_int(d_pat_evt["n_used"])], ["  ΔHR min / max [bpm]", f"{_fmt_num(d_pat_evt['min'], 2)} / {_fmt_num(d_pat_evt['max'], 2)}"], ["  ΔHR mean / median [bpm]", f"{_fmt_num(d_pat_evt['mean'], 2)} / {_fmt_num(d_pat_evt['median'], 2)}"], ["  ΔHR std [bpm]", _fmt_num(d_pat_evt["std"], 2)], ["  ΔHR NaN %", _fmt_pct(d_pat_evt["nan_pct"], 1)]]
    fig3 = _render_table_page("Summary (ΔHR: baseline vs event-only)", rows_p3, edf_base=edf_base, font_size=12, scale_y=1.40)

    combo_rows = _sleep_combo_rows(sleep_combo_summaries)
    fig_combo = _render_table_page("Summary (Fixed Sleep Combinations)", combo_rows, edf_base=edf_base, font_size=12, scale_y=1.35) if combo_rows else None

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
        rows_p4 += [["Overall event summary (aux CSV)", ""], ["  Samples (rows)", f"{aux_total:d}"], ["  Active exclusion columns", ", ".join(policy.exclusion_columns) if policy.exclusion_columns else "none"], ["  Desaturation flags", f"{desat_n:d} ({desat_pct})"], ["  Exclude HR flags", f"{excl_hr_n:d} ({excl_hr_pct})"], ["  Exclude PAT flags", f"{excl_pat_n:d} ({excl_pat_pct})"], ["  Central A/H 3%", f"{cen3_n:d} ({cen3_pct})"], ["  Obstructive A/H 3%", f"{obs3_n:d} ({obs3_pct})"], ["  Unclassified A/H 3%", f"{unc3_n:d} ({unc3_pct})"]]
        if (cen4_n + obs4_n + unc4_n) > 0:
            rows_p4 += [["  Central A/H 4%", f"{cen4_n:d} ({cen4_pct})"], ["  Obstructive A/H 4%", f"{obs4_n:d} ({obs4_pct})"], ["  Unclassified A/H 4%", f"{unc4_n:d} ({unc4_pct})"]]
        rows_p4 += [["", ""]]
        rows_p4 += _sleep_stage_rows(aux_df)
    else:
        rows_p4 += [["Event summary", "No aux_df available"]]
    rows_p4 += [["", ""], ["PAT burden (event+desat within included sleep)", ""]]
    burden_val = float(pat_burden) if pat_burden is not None and np.isfinite(pat_burden) else None
    unit = "rel·min/h" if isinstance(pat_burden_diag, dict) and pat_burden_diag.get("relative") else "amp·min/h"
    rows_p4 += [[f"  Burden [{unit}]", _fmt(burden_val, 3)]]
    if isinstance(pat_burden_diag, dict):
        rows_p4 += [["  Sleep hours", _fmt_num(pat_burden_diag.get("sleep_hours"), 2)], ["  Episodes (total)", _fmt_int(pat_burden_diag.get("n_episodes"))], ["  Episodes used", _fmt_int(pat_burden_diag.get("n_episodes_used"))], ["  Total area [min]", _fmt_num(pat_burden_diag.get("total_area_min"), 2)]]
    fig4 = _render_table_page("Summary (Events & Sleep Stages)", rows_p4, edf_base=edf_base, font_size=12, scale_y=1.25)

    figs = [fig1, fig2, fig3]
    if fig_combo is not None:
        figs.append(fig_combo)
    figs.append(fig4)
    return figs
