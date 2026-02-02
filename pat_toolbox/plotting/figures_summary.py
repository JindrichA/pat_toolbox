from __future__ import annotations

from typing import Optional, Dict, TYPE_CHECKING, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt

from .. import config
from .utils import _fmt, _count_flags

if TYPE_CHECKING:
    import pandas as pd


# ============================================================================
# Small formatting / stats helpers (kept local to this module)
# ============================================================================


def _finite_stats(y: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    out = {
        "n_total": None,
        "n_used": None,
        "min": None,
        "max": None,
        "mean": None,
        "median": None,
        "std": None,
        "nan_pct": None,
    }
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



def _nan_pct(x: Optional[np.ndarray]) -> Optional[float]:
    """Percent of non-finite values in x."""
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


def _mask_keep_nonexcluded(
    t_sec: np.ndarray,
    exclusion_zones: Optional[List[Tuple[float, float, str]]],
) -> np.ndarray:
    """
    True where sample is OUTSIDE exclusion zones.

    Convention:
      - exclusion_zones are in HOURS-from-start (like _add_exclusion_spans)
      - t_sec is SECONDS-from-start
    """
    keep = np.ones_like(t_sec, dtype=bool)
    if not exclusion_zones:
        return keep

    for z0_h, z1_h, _lbl in exclusion_zones:
        z0 = float(z0_h) * 3600.0
        z1 = float(z1_h) * 3600.0
        if z1 < z0:
            z0, z1 = z1, z0
        keep &= ~((t_sec >= z0) & (t_sec <= z1))
    return keep


def _masked_stats(
    t_sec: Optional[np.ndarray],
    y: Optional[np.ndarray],
    exclusion_zones: Optional[List[Tuple[float, float, str]]],
) -> Dict[str, Optional[float]]:
    """
    Stats computed ONLY on samples that are:
      - defined by (t_sec, y) aligned arrays,
      - OUTSIDE exclusion_zones,
      - finite.

    Also reports NaN% on the *kept* region (post-exclusion quality).
    """
    out: Dict[str, Optional[float]] = {
        "n_total": None,
        "n_kept": None,
        "n_used": None,
        "min": None,
        "max": None,
        "mean": None,
        "median": None,
        "std": None,
        "nan_pct_used": None,
    }

    if t_sec is None or y is None:
        return out

    t_sec = np.asarray(t_sec)
    y = np.asarray(y)

    if t_sec.size == 0 or y.size == 0 or t_sec.size != y.size:
        return out

    keep = _mask_keep_nonexcluded(t_sec, exclusion_zones)
    y_keep = y[keep].astype(float)

    out["n_total"] = float(y.size)
    out["n_kept"] = float(np.count_nonzero(keep))

    ok = np.isfinite(y_keep)
    out["n_used"] = float(np.count_nonzero(ok))

    # NaN% AFTER exclusion masking (so quality is based on kept region)
    out["nan_pct_used"] = float(100.0 * np.mean(~np.isfinite(y_keep)))

    if np.count_nonzero(ok) == 0:
        return out

    yy = y_keep[ok]
    out["min"] = float(np.min(yy))
    out["max"] = float(np.max(yy))
    out["mean"] = float(np.mean(yy))
    out["median"] = float(np.median(yy))
    out["std"] = float(np.std(yy))

    return out


# ============================================================================
# Sleep-stage masking rows for summary table
# ============================================================================

def _sleep_stage_stats(aux_df: Optional["pd.DataFrame"]) -> Optional[list[list[str]]]:
    """
    Returns rows to append into the summary table with sleep-stage masking stats.
    If stages are not available -> None.
    """
    if aux_df is None:
        return None

    time_col = getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec")
    stage_code_col = getattr(config, "AUX_CSV_STAGE_CODE_COLUMN", "stage_code")

    if time_col not in aux_df.columns or stage_code_col not in aux_df.columns:
        return None

    stage = aux_df[stage_code_col].to_numpy(dtype=float)
    ok = np.isfinite(stage)
    if not np.any(ok):
        return None

    stage_i = np.round(stage[ok]).astype(int)

    total = stage_i.size
    if total <= 0:
        return None

    # Included stages from policy
    try:
        include_set = set(config.sleep_include_numeric())
        policy = str(getattr(config, "SLEEP_STAGE_POLICY", "all_sleep"))
        enabled = bool(getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False))
    except Exception:
        include_set = {1, 2, 3}
        policy = "all_sleep"
        enabled = False

    included = np.array([s in include_set for s in stage_i], dtype=bool)
    inc_n = int(np.sum(included))
    exc_n = int(total - inc_n)

    def pct(n: int) -> str:
        return f"{(100.0 * n / total):.1f}%"

    counts = {k: int(np.sum(stage_i == k)) for k in [0, 1, 2, 3]}
    names = {0: "Wake", 1: "Light", 2: "Deep", 3: "REM"}

    rows: list[list[str]] = []
    rows.append(["", ""])
    rows.append(["Sleep-stage masking", ""])
    rows.append(["  Enabled", "Yes" if enabled else "No"])
    rows.append(["  Policy", policy])
    rows.append(["  Included (by policy)", f"{inc_n} ({pct(inc_n)})"])
    rows.append(["  Excluded (by policy)", f"{exc_n} ({pct(exc_n)})"])

    rows.append(["  Stage breakdown", ""])
    for k in [0, 1, 2, 3]:
        rows.append([f"    {names[k]}", f"{counts[k]} ({pct(counts[k])})"])

    return rows


# ============================================================================
# Summary page figure
# ============================================================================

def _build_summary_figure(
    edf_base: str,
    pearson_r: Optional[float],
    spear_rho: Optional[float],
    rmse: Optional[float],
    hrv_summary: Optional[Dict[str, float]],
    mayer_peak_freq: Optional[float],
    resp_peak_freq: Optional[float],
    aux_df: Optional["pd.DataFrame"],
    *,
    # series for NaN% quality reporting on the summary page
    t_hr_calc: Optional[np.ndarray] = None,
    hr_calc: Optional[np.ndarray] = None,
    t_hr_edf: Optional[np.ndarray] = None,
    hr_edf: Optional[np.ndarray] = None,
    t_hrv: Optional[np.ndarray] = None,
    hrv_clean: Optional[np.ndarray] = None,
    hrv_raw: Optional[np.ndarray] = None,
    hrv_tv: Optional[Dict[str, np.ndarray]] = None,
    psd_features: Optional[Dict[str, float]] = None,
    exclusion_zones: Optional[List[Tuple[float, float, str]]] = None,
    delta_hr_calc: Optional[np.ndarray] = None,
    delta_hr_edf: Optional[np.ndarray] = None,
    delta_hr_calc_evt: Optional[np.ndarray] = None,
    delta_hr_edf_evt: Optional[np.ndarray] = None,

) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")

    # HRV summary values (if present)
    rmssd_mean = hrv_summary.get("rmssd_mean") if hrv_summary else None
    rmssd_median = hrv_summary.get("rmssd_median") if hrv_summary else None
    sdnn = hrv_summary.get("sdnn") if hrv_summary else None
    lf = hrv_summary.get("lf") if hrv_summary else None
    hf = hrv_summary.get("hf") if hrv_summary else None
    lf_hf = hrv_summary.get("lf_hf") if hrv_summary else None

    # Whole-series NaN % quality stats
    hr_calc_nan = _nan_pct(hr_calc)
    hr_edf_nan = _nan_pct(hr_edf)
    hrv_clean_nan = _nan_pct(hrv_clean)
    hrv_raw_nan = _nan_pct(hrv_raw)

    # Build the summary table rows
    cell_text: list[list[str]] = [
        ["HR correlation (Proprietary vs PAT 1Hz)", ""],
        ["  Pearson r", _fmt(pearson_r, 3)],
        ["  Spearman ρ", _fmt(spear_rho, 3)],
        ["  RMSE [bpm]", _fmt(rmse, 2)],
        ["", ""],
        ["HRV Summary (Clean RR)", ""],
        ["  Average overnight RMSSD mean [ms]", _fmt(rmssd_mean, 2)],
        ["  Average overnight RMSSD median [ms]", _fmt(rmssd_median, 2)],
        ["  Average overnight SDNN [ms]", _fmt(sdnn, 2)],
        ["  Average overnight LF", _fmt(lf, 4)],
        ["  Average overnight HF", _fmt(hf, 4)],
        ["  Average overnight LF/HF", _fmt(lf_hf, 2)],
        ["", ""],
    ]

    # Spectral analysis
    cell_text.append(["Spectral Analysis (PAT Volume)", ""])
    cell_text.append(["  Mayer Peak [Hz]", _fmt(mayer_peak_freq, 3)])
    cell_text.append(["  Resp Peak [Hz]", _fmt(resp_peak_freq, 3)])

    if psd_features:
        pow_vlf = psd_features.get("pow_vlf")
        pow_mayer = psd_features.get("pow_mayer")
        pow_resp = psd_features.get("pow_resp")
        norm_mayer = psd_features.get("norm_mayer")
        norm_resp = psd_features.get("norm_resp")
        n_wins = psd_features.get("n_windows")

        cell_text.append(["  VLF Power (0.0033-0.04 Hz)", _fmt_sci(pow_vlf)])
        cell_text.append(["  Mayer Power (0.04-0.15 Hz)", _fmt_sci(pow_mayer)])
        cell_text.append(["  Resp Power (0.15-0.50 Hz)", _fmt_sci(pow_resp)])
        cell_text.append(["  Mayer Power (n.u.)", _fmt_pct(norm_mayer, 1)])
        cell_text.append(["  Resp Power (n.u.)", _fmt_pct(norm_resp, 1)])
        cell_text.append(["  Valid Windows Used", str(int(n_wins)) if n_wins else "NA"])

    # Signal quality
    cell_text.append(["", ""])
    cell_text.append(["Signal Quality (NaN %)", ""])
    cell_text.append(["  HR (PAT-derived) NaN %", _fmt_pct(hr_calc_nan, 1)])
    cell_text.append(["  Proprietary HR (EDF) NaN %", _fmt_pct(hr_edf_nan, 1)])
    cell_text.append(["  HRV RMSSD clean NaN %", _fmt_pct(hrv_clean_nan, 1)])
    cell_text.append(["  HRV RMSSD raw NaN %", _fmt_pct(hrv_raw_nan, 1)])

    # ΔHR summary (INSIDE event/desat windows only)
    # (delta_hr_*_evt are NaN everywhere except inside event/desat windows)
    d_edf_evt = _finite_stats(delta_hr_edf_evt)
    d_pat_evt = _finite_stats(delta_hr_calc_evt)

    cell_text.append(["", ""])
    cell_text.append(["ΔHR Summary (INSIDE event/desat windows only)", ""])

    # EDF ΔHR (event-only)
    cell_text.append(["  ΔHR EDF n used", _fmt_int(d_edf_evt["n_used"])])
    cell_text.append(["  ΔHR EDF min / max [bpm]",
                      f"{_fmt_num(d_edf_evt['min'], 2)} / {_fmt_num(d_edf_evt['max'], 2)}"])
    cell_text.append(["  ΔHR EDF mean / median [bpm]",
                      f"{_fmt_num(d_edf_evt['mean'], 2)} / {_fmt_num(d_edf_evt['median'], 2)}"])
    cell_text.append(["  ΔHR EDF std [bpm]", _fmt_num(d_edf_evt["std"], 2)])
    cell_text.append(["  ΔHR EDF NaN % (overall)", _fmt_pct(d_edf_evt["nan_pct"], 1)])

    # PAT ΔHR (event-only)
    cell_text.append(["  ΔHR PAT n used", _fmt_int(d_pat_evt["n_used"])])
    cell_text.append(["  ΔHR PAT min / max [bpm]",
                      f"{_fmt_num(d_pat_evt['min'], 2)} / {_fmt_num(d_pat_evt['max'], 2)}"])
    cell_text.append(["  ΔHR PAT mean / median [bpm]",
                      f"{_fmt_num(d_pat_evt['mean'], 2)} / {_fmt_num(d_pat_evt['median'], 2)}"])
    cell_text.append(["  ΔHR PAT std [bpm]", _fmt_num(d_pat_evt["std"], 2)])
    cell_text.append(["  ΔHR PAT NaN % (overall)", _fmt_pct(d_pat_evt["nan_pct"], 1)])

    # Optional: add TV metrics NaN %
    if isinstance(hrv_tv, dict) and t_hrv is not None and np.size(t_hrv) > 0:
        for key in ["sdnn_ms", "lf", "hf", "lf_hf", "rmssd_ms"]:
            y = hrv_tv.get(key, None)
            if y is None or np.size(y) != np.size(t_hrv):
                continue
            cell_text.append([f"  HRV 5-min sliding {key} NaN %", _fmt_pct(_nan_pct(np.asarray(y)), 1)])

    # Aux CSV event summary
    if aux_df is not None:
        aux_total = len(aux_df)

        desat_n, desat_pct = _count_flags(aux_df, "desat_flag")
        excl_n, excl_pct = _count_flags(aux_df, "exclude_hr_flag")
        excl_pat_n, excl_pat_pct = _count_flags(aux_df, "exclude_pat_flag")

        cen3_n, cen3_pct = _count_flags(aux_df, "evt_central_3")
        obs3_n, obs3_pct = _count_flags(aux_df, "evt_obstructive_3")
        unc3_n, unc3_pct = _count_flags(aux_df, "evt_unclassified_3")

        cen4_n, cen4_pct = _count_flags(aux_df, "evt_central_4")
        obs4_n, obs4_pct = _count_flags(aux_df, "evt_obstructive_4")
        unc4_n, unc4_pct = _count_flags(aux_df, "evt_unclassified_4")

        cell_text.extend(
            [
                ["", ""],
                ["Event summary (aux CSV)", ""],
                ["  Samples (rows)", f"{aux_total:d}"],
                ["  Desaturation flags", f"{desat_n:d} ({desat_pct})"],
                ["  Exclude HR flags", f"{excl_n:d} ({excl_pct})"],
                ["  Exclude PAT flags", f"{excl_pat_n:d} ({excl_pat_pct})"],
                ["  Central A/H 3%", f"{cen3_n:d} ({cen3_pct})"],
                ["  Obstructive A/H 3%", f"{obs3_n:d} ({obs3_pct})"],
                ["  Unclassified A/H 3%", f"{unc3_n:d} ({unc3_pct})"],
            ]
        )

        if (cen4_n + obs4_n + unc4_n) > 0:
            cell_text.extend(
                [
                    ["  Central A/H 4%", f"{cen4_n:d} ({cen4_pct})"],
                    ["  Obstructive A/H 4%", f"{obs4_n:d} ({obs4_pct})"],
                    ["  Unclassified A/H 4%", f"{unc4_n:d} ({unc4_pct})"],
                ]
            )

    # Sleep-stage masking summary
    sleep_rows = _sleep_stage_stats(aux_df)
    if sleep_rows is not None:
        cell_text.extend(sleep_rows)

    # ------------------------------------------------------------------
    # Table rendering (auto-scale for long tables)
    # ------------------------------------------------------------------
    n_rows = len(cell_text)
    # Heuristic: shrink font/scale when we have lots of lines
    if n_rows > 55:
        font_size = 7
        scale_y = 0.85
    elif n_rows > 45:
        font_size = 8
        scale_y = 0.95
    else:
        font_size = 9
        scale_y = 1.2

    table = ax.table(
        cellText=cell_text,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.2, scale_y)

    ax.set_title(f"{edf_base} - Summary", fontsize=14, pad=20)
    fig.tight_layout()
    return fig


# ============================================================================
# Sleep stagegram (hypnogram) page
# ============================================================================

def _build_sleep_stagegram_figure(
    edf_base: str,
    aux_df: Optional["pd.DataFrame"],
) -> Optional[plt.Figure]:
    """
    Professional-style hypnogram (sleep stagegram) page.
    """
    if aux_df is None or len(aux_df) == 0:
        return None

    time_col = getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec")
    stage_col = getattr(config, "AUX_CSV_STAGE_CODE_COLUMN", "stage_code")

    if time_col not in aux_df.columns or stage_col not in aux_df.columns:
        return None

    t = aux_df[time_col].to_numpy(dtype=float)
    s = aux_df[stage_col].to_numpy(dtype=float)

    ok = np.isfinite(t) & np.isfinite(s)
    if not np.any(ok):
        return None

    t = t[ok]
    s = np.round(s[ok]).astype(int)

    # Ensure time ordering
    order = np.argsort(t)
    t = t[order]
    s = s[order]

    # De-duplicate equal timestamps
    if t.size >= 2:
        keep = np.ones_like(t, dtype=bool)
        keep[:-1] = (t[1:] != t[:-1])
        t = t[keep]
        s = s[keep]

    if t.size == 0:
        return None

    # Detect wraparound (midnight) and unwrap if needed
    dt = np.diff(t)
    wrap_idx = np.where(dt < -12 * 3600)[0]
    if wrap_idx.size > 0:
        t2 = t.copy()
        for i in wrap_idx:
            t2[i + 1:] += 24 * 3600
        t = t2

    # Shift so stagegram starts at 0
    t0 = float(t[0])
    t = t - t0

    # Map stage codes -> y positions (Wake on top)
    y_map = {0: 3, 3: 2, 1: 1, 2: 0}
    y = np.array([y_map.get(int(x), np.nan) for x in s], dtype=float)
    oky = np.isfinite(y)
    if not np.any(oky):
        return None

    t = t[oky]
    s = s[oky]
    y = y[oky]
    if t.size == 0:
        return None

    enabled = bool(getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False))
    policy = str(getattr(config, "SLEEP_STAGE_POLICY", "all_sleep"))
    try:
        include_set = set(config.sleep_include_numeric())
    except Exception:
        include_set = {1, 2, 3}

    included = np.array([int(si) in include_set for si in s], dtype=bool)

    xh = t / 3600.0

    # Build edges for step plot
    if xh.size == 1:
        step_h = 1.0 / 3600.0
        edges = np.array([xh[0], xh[0] + step_h], dtype=float)
    else:
        d = np.diff(xh)
        dpos = d[np.isfinite(d) & (d > 0)]
        step_h = float(np.median(dpos)) if dpos.size else (1.0 / 3600.0)

        edges = np.empty(xh.size + 1, dtype=float)
        edges[:-1] = xh
        edges[-1] = xh[-1] + step_h

    fig, ax = plt.subplots(figsize=(11.69, 8.27))

    # Background bands
    bands = [
        (3, "Wake", 0.08),
        (2, "REM", 0.06),
        (1, "NREM (Light)", 0.05),
        (0, "NREM (Deep)", 0.05),
    ]
    for y0, _name, alpha in bands:
        ax.axhspan(y0 - 0.5, y0 + 0.5, alpha=alpha, zorder=0)

    # Step plot
    x_step = edges
    y_step = np.r_[y, y[-1]]
    ax.step(x_step, y_step, where="post", linewidth=3.0, zorder=3)

    # Shade excluded sleep stages (policy-based)
    if enabled and included.size == xh.size:
        exc = ~included
        if np.any(exc):
            idx = np.where(exc)[0]
            splits = np.where(np.diff(idx) > 1)[0] + 1
            groups = np.split(idx, splits)
            for g in groups:
                if g.size == 0:
                    continue
                i0 = int(g[0])
                i1 = int(g[-1]) + 1
                x0 = float(edges[i0])
                x1 = float(edges[min(i1, edges.size - 1)])
                if x1 > x0:
                    ax.axvspan(x0, x1, color="k", alpha=0.06, zorder=1)

    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(["Wake", "REM", "NREM-Light", "NREM-Deep"])

    ax.set_ylim(-0.7, 3.7)
    ax.set_xlim(0.0, edges[-1])

    def _h_to_clock(h: float) -> str:
        h = h % 24
        hh = int(h)
        mm = int(round((h - hh) * 60))
        return f"{hh:02d}:{mm:02d}"

    ax.set_xticks(np.arange(np.floor(xh.min()), np.ceil(xh.max()) + 1))
    ax.set_xticklabels([_h_to_clock(h) for h in ax.get_xticks()])
    ax.set_xlabel("Time since recording start [hours]")
    ax.set_title(f"{edf_base} - Hypnogram", fontsize=14, pad=12)

    ax.grid(True, which="major", axis="x", alpha=0.35)
    ax.grid(True, which="minor", axis="x", alpha=0.18)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    ax.grid(True, which="major", axis="y", alpha=0.20)

    total = int(len(s))
    if total > 0:
        def pct(n: int) -> str:
            return f"{(100.0 * n / total):.1f}%"

        counts = {k: int(np.sum(s == k)) for k in [0, 1, 2, 3]}
        inc_n = int(np.sum(included)) if included.size == s.size else 0
        exc_n = int(total - inc_n) if included.size == s.size else 0

        stats_lines = [
            f"Masking: {'ON' if enabled else 'OFF'} ({policy})",
            f"Included: {inc_n} ({pct(inc_n)})" if included.size == s.size else "Included: NA",
            f"Excluded: {exc_n} ({pct(exc_n)})" if included.size == s.size else "Excluded: NA",
            "",
            f"Wake: {counts[0]} ({pct(counts[0])})",
            f"REM:  {counts[3]} ({pct(counts[3])})",
            f"NREM-Light: {counts[1]} ({pct(counts[1])})",
            f"NREM-Deep:  {counts[2]} ({pct(counts[2])})",
        ]

        ax.text(
            0.99,
            0.98,
            "\n".join(stats_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", alpha=0.10, pad=0.4),
            zorder=10,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig
