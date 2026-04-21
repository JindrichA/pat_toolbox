from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from .. import config
from ..io.aux_events import compute_sleep_timing_from_aux

if TYPE_CHECKING:
    import pandas as pd


def _build_sleep_stagegram_figure(
    edf_base: str,
    aux_df: Optional["pd.DataFrame"],
):
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
        keep[:-1] = t[1:] != t[:-1]
        t = t[keep]
        s = s[keep]
    if t.size == 0:
        return None
    dt = np.diff(t)
    wrap_idx = np.where(dt < -12 * 3600)[0]
    if wrap_idx.size > 0:
        t2 = t.copy()
        for i in wrap_idx:
            t2[i + 1:] += 24 * 3600
        t = t2
    t = t - float(t[0])
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
    ax.step(edges, np.r_[y, y[-1]], where="post", linewidth=3.0, zorder=3)
    if enabled and included.size == xh.size:
        exc = ~included
        if np.any(exc):
            idx = np.where(exc)[0]
            splits = np.where(np.diff(idx) > 1)[0] + 1
            for g in np.split(idx, splits):
                if g.size == 0:
                    continue
                i0 = int(g[0]); i1 = int(g[-1]) + 1
                x0 = float(edges[i0]); x1 = float(edges[min(i1, edges.size - 1)])
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
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.grid(True, which="major", axis="x", alpha=0.25)
    sleep_timing = compute_sleep_timing_from_aux(aux_df)
    if sleep_timing:
        line_specs = [
            (sleep_timing.get("sleep_onset_rel_h"), "Sleep onset", "tab:green", "--"),
            (sleep_timing.get("sleep_midpoint_rel_h"), "Sleep midpoint", "tab:purple", "-"),
            (sleep_timing.get("sleep_end_rel_h"), "Sleep end", "tab:red", "--"),
        ]
        for x_h, label, color, style in line_specs:
            if x_h is not None and np.isfinite(x_h):
                ax.axvline(float(x_h), color=color, linestyle=style, linewidth=1.8, alpha=0.9, zorder=4)
    total = int(len(s))
    if total > 0:
        pct = lambda n: f"{(100.0 * n / total):.1f}%"
        counts = {k: int(np.sum(s == k)) for k in [0, 1, 2, 3]}
        inc_n = int(np.sum(included)) if included.size == s.size else 0
        exc_n = int(total - inc_n) if included.size == s.size else 0
        stats_lines = [f"Masking: {'ON' if enabled else 'OFF'} ({policy})", f"Included: {inc_n} ({pct(inc_n)})" if included.size == s.size else "Included: NA", f"Excluded: {exc_n} ({pct(exc_n)})" if included.size == s.size else "Excluded: NA", "", f"Wake: {counts[0]} ({pct(counts[0])})", f"REM:  {counts[3]} ({pct(counts[3])})", f"NREM-Light: {counts[1]} ({pct(counts[1])})", f"NREM-Deep:  {counts[2]} ({pct(counts[2])})"]
        ax.text(0.99, 0.98, "\n".join(stats_lines), transform=ax.transAxes, ha="right", va="top", fontsize=9, bbox=dict(boxstyle="round", alpha=0.10, pad=0.4), zorder=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig
