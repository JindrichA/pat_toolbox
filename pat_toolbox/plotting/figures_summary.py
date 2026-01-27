# pat_toolbox/plotting/figures_summary.py
from __future__ import annotations

from typing import Optional, Dict, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

from .. import config
from .utils import _fmt, _count_flags

if TYPE_CHECKING:
    import pandas as pd


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

    # Stage counts
    counts = {k: int(np.sum(stage_i == k)) for k in [0, 1, 2, 3]}

    # Pretty names
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
        psd_features: Optional[Dict[str, float]] = None,  # <--- NEW ARGUMENT (Dictionary of spectral features)
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")

    rmssd_mean = hrv_summary.get("rmssd_mean") if hrv_summary else None
    rmssd_median = hrv_summary.get("rmssd_median") if hrv_summary else None
    sdnn = hrv_summary.get("sdnn") if hrv_summary else None
    lf = hrv_summary.get("lf") if hrv_summary else None
    hf = hrv_summary.get("hf") if hrv_summary else None
    lf_hf = hrv_summary.get("lf_hf") if hrv_summary else None

    def _nan_pct_local(x: Optional[np.ndarray]) -> Optional[float]:
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

    # NaN % quality stats (whole-series)
    hr_calc_nan = _nan_pct_local(hr_calc)
    hr_edf_nan = _nan_pct_local(hr_edf)
    hrv_clean_nan = _nan_pct_local(hrv_clean)
    hrv_raw_nan = _nan_pct_local(hrv_raw)

    cell_text = [
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

    # --- NEW: SPECTRAL ANALYSIS SECTION ---
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

    cell_text.append(["", ""])
    cell_text.append(["Signal Quality (NaN %)", ""])
    cell_text.append(["  HR (PAT-derived) NaN %", _fmt_pct(hr_calc_nan, 1)], )
    cell_text.append(["  Proprietary HR (EDF) NaN %", _fmt_pct(hr_edf_nan, 1)], )
    cell_text.append(["  HRV RMSSD clean NaN %", _fmt_pct(hrv_clean_nan, 1)], )
    cell_text.append(["  HRV RMSSD raw NaN %", _fmt_pct(hrv_raw_nan, 1)], )

    # Optional: add TV metrics NaN %
    if isinstance(hrv_tv, dict) and t_hrv is not None and np.size(t_hrv) > 0:
        for key in ["sdnn_ms", "lf", "hf", "lf_hf", "rmssd_ms"]:
            y = hrv_tv.get(key, None)
            if y is None or np.size(y) != np.size(t_hrv):
                continue
            pct = _nan_pct_local(np.asarray(y))
            cell_text.append([f"  HRV 5-min sliding {key} NaN %", _fmt_pct(pct, 1)])

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

    table = ax.table(
        cellText=cell_text,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)

    ax.set_title(f"{edf_base} - Summary", fontsize=14, pad=20)
    fig.tight_layout()
    return fig


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

    order = np.argsort(t)
    t = t[order]
    s = s[order]

    if t.size >= 2:
        keep = np.ones_like(t, dtype=bool)
        keep[:-1] = (t[1:] != t[:-1])
        t = t[keep]
        s = s[keep]

    if t.size == 0:
        return None

    dt = np.diff(t)
    wrap_idx = np.where(dt < -12 * 3600)[0]  # backward jump = midnight

    if wrap_idx.size > 0:
        t2 = t.copy()
        for i in wrap_idx:
            t2[i + 1:] += 24 * 3600
        t = t2

    t0 = float(t[0])
    t = t - t0

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

    if xh.size == 1:
        dt_h = 1.0 / 3600.0
        edges = np.array([xh[0], xh[0] + dt_h], dtype=float)
    else:
        d = np.diff(xh)
        dpos = d[np.isfinite(d) & (d > 0)]
        step_h = float(np.median(dpos)) if dpos.size else (1.0 / 3600.0)

        edges = np.empty(xh.size + 1, dtype=float)
        edges[:-1] = xh
        edges[-1] = xh[-1] + step_h

    fig, ax = plt.subplots(figsize=(11.69, 8.27))

    bands = [
        (3, "Wake", 0.08),
        (2, "REM", 0.06),
        (1, "NREM (Light)", 0.05),
        (0, "NREM (Deep)", 0.05),
    ]
    for y0, _name, alpha in bands:
        ax.axhspan(y0 - 0.5, y0 + 0.5, alpha=alpha, zorder=0)

    x_step = edges
    y_step = np.r_[y, y[-1]]

    ax.step(
        x_step,
        y_step,
        where="post",
        linewidth=3.0,
        zorder=3,
    )

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

    def _h_to_clock(h):
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