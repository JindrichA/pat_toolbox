from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from .. import sleep_mask
from .. import config
from .. import io_aux_csv
from ..metrics.pat_burden import compute_pat_burden_from_pat_amp
from .specs import EventSpec, DEFAULT_EVENT_PLOT_SPEC
from .utils import _add_exclusion_spans, _shade_masked_regions, _h_to_hhmm

if TYPE_CHECKING:
    import pandas as pd


def _overlay_pat_burden_area(
    ax: plt.Axes,
    *,
    t_sec_all: np.ndarray,
    pat_amp_all: np.ndarray,
    aux_df,
    seg_start_sec: float,
    seg_end_sec: float,
) -> None:
    """
    Kept for future use, but currently not shown in report.
    """
    if aux_df is None:
        return
    if t_sec_all is None or pat_amp_all is None:
        return

    t = np.asarray(t_sec_all, dtype=float)
    y = np.asarray(pat_amp_all, dtype=float)
    if t.size == 0 or y.size == 0 or t.size != y.size:
        return

    m_sleep_keep = sleep_mask.build_sleep_include_mask_for_times(t, aux_df)
    if m_sleep_keep is None:
        m_sleep_keep = np.ones_like(t, dtype=bool)

    m_evt_keep = io_aux_csv.build_time_exclusion_mask(t, aux_df)
    if m_evt_keep is None:
        return

    m_inside = np.asarray(m_sleep_keep, bool) & (~np.asarray(m_evt_keep, bool))

    min_ep_sec = float(getattr(config, "PAT_BURDEN_MIN_EPISODE_SEC", 5.0))
    lookback = float(getattr(config, "PAT_BURDEN_BASELINE_LOOKBACK_SEC", 30.0))
    pctl = float(getattr(config, "PAT_BURDEN_BASELINE_PCTL", 95.0))
    min_base_n = int(getattr(config, "PAT_BURDEN_BASELINE_MIN_SAMPLES", 5))

    m_baseline_ok = np.asarray(m_sleep_keep, bool) & np.asarray(m_evt_keep, bool) & np.isfinite(y)

    def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
        mask = np.asarray(mask, dtype=bool)
        if mask.size == 0 or not np.any(mask):
            return []
        d = np.diff(mask.astype(int))
        starts = np.where(d == 1)[0] + 1
        ends = np.where(d == -1)[0] + 1
        if mask[0]:
            starts = np.r_[0, starts]
        if mask[-1]:
            ends = np.r_[ends, mask.size]
        return [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]

    episodes = _runs(m_inside)

    for s, e in episodes:
        t0 = float(t[s])
        t1 = float(t[e - 1])
        if not (np.isfinite(t0) and np.isfinite(t1)):
            continue
        if (t1 - t0) < min_ep_sec:
            continue

        w0 = t0 - lookback
        w1 = t0
        m_pre = (t >= w0) & (t < w1) & m_baseline_ok
        if np.count_nonzero(m_pre) < min_base_n:
            continue

        baseline = float(np.nanpercentile(y[m_pre], pctl))
        if not np.isfinite(baseline):
            continue

        tt = t[s:e]
        yy = y[s:e]
        good = np.isfinite(tt) & np.isfinite(yy)
        if np.count_nonzero(good) < 2:
            continue
        tt = tt[good]
        yy = yy[good]

        m_seg = (tt >= seg_start_sec) & (tt <= seg_end_sec)
        if np.count_nonzero(m_seg) < 2:
            continue
        tt = tt[m_seg]
        yy = yy[m_seg]

        below = yy < baseline
        if not np.any(below):
            continue

        ax.plot(
            tt / 3600.0,
            np.full_like(tt, baseline, dtype=float),
            linestyle="--",
            linewidth=1.1,
            alpha=0.7,
            color="0.25",
            label="_nolegend_",
            zorder=2,
        )

        ax.fill_between(
            tt / 3600.0,
            yy,
            baseline,
            where=below,
            interpolate=True,
            alpha=0.22,
            color="tab:red",
            label="_nolegend_",
            zorder=1,
        )


def _plot_no_bridge(
    ax: plt.Axes,
    x_sec: np.ndarray,
    y: np.ndarray,
    *,
    label: str,
    linestyle: str = "--",
    linewidth: float = 1.0,
    color: Optional[str] = None,
    alpha: float = 0.6,
    zorder: int = 0,
    gap_factor: float = 2.5,
    min_gap_sec: float = 2.0,
):
    x_sec = np.asarray(x_sec, dtype=float)
    y = np.asarray(y, dtype=float)
    if x_sec.size == 0 or y.size == 0 or x_sec.size != y.size:
        return

    ok = np.isfinite(x_sec) & np.isfinite(y)
    if not np.any(ok):
        return

    idx_ok = np.where(ok)[0]
    x_ok = x_sec[idx_ok]
    y_ok = y[idx_ok]

    order = np.argsort(x_ok)
    x_ok = x_ok[order]
    y_ok = y_ok[order]

    d = np.diff(x_ok)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return
    med_dt = float(np.median(d))

    gap_thr = max(float(min_gap_sec), float(gap_factor) * med_dt)
    cut = np.where(np.diff(x_ok) > gap_thr)[0] + 1
    runs = np.split(np.arange(x_ok.size), cut)

    first = True
    for r in runs:
        if r.size < 2:
            continue
        ax.plot(
            x_ok[r] / 3600.0,
            y_ok[r],
            linestyle=linestyle,
            linewidth=linewidth,
            color=color,
            alpha=alpha,
            label=label if first else "_nolegend_",
            zorder=zorder,
        )
        first = False


def _apply_global_mask_to_series(
    t_sec: np.ndarray,
    y: np.ndarray,
    aux_df: Optional["pd.DataFrame"],
) -> np.ndarray:
    y2 = y.astype(float, copy=True)

    if aux_df is None or t_sec is None or t_sec.size == 0:
        return y2

    m_keep = sleep_mask.build_global_include_mask_for_times(
        t_sec, aux_df, apply_sleep=True, apply_events=True
    )
    if m_keep is None:
        return y2

    y2[~m_keep] = np.nan
    return y2


def _plot_segment_hr(
    ax: plt.Axes,
    t_hr_edf: Optional[np.ndarray],
    hr_edf: Optional[np.ndarray],
    t_hr_calc: Optional[np.ndarray],
    hr_calc: Optional[np.ndarray],
    t_hr_edf_raw: Optional[np.ndarray],
    hr_edf_raw: Optional[np.ndarray],
    t_hr_calc_raw: Optional[np.ndarray],
    hr_calc_raw: Optional[np.ndarray],
    seg_start_sec: float,
    seg_end_sec: float,
    exclusion_zones: List[Tuple[float, float, str]],
    t_seg_h_start: float,
    t_seg_h_end: float,
    aux_df: Optional["pd.DataFrame"],
    t_seg_sec: np.ndarray,
) -> tuple[Optional[float], Optional[float]]:
    t_hr_edf = None
    hr_edf = None
    t_hr_edf_raw = None
    hr_edf_raw = None

    _add_exclusion_spans(ax, exclusion_zones, t_seg_h_start, t_seg_h_end, label_once=True)

    if aux_df is not None and t_seg_sec.size > 0:
        m_keep = sleep_mask.build_global_include_mask_for_times(
            t_seg_sec, aux_df, apply_sleep=True, apply_events=True
        )
        if m_keep is not None:
            _shade_masked_regions(
                ax,
                t_sec=t_seg_sec,
                masked=~m_keep,
                color="0.6",
                alpha=0.22,
            )

    seg_hr_min: Optional[float] = None
    seg_hr_max: Optional[float] = None
    summary_lines: List[str] = []

    def _stats_line(name: str, y_seg: np.ndarray) -> None:
        ok = np.isfinite(y_seg)
        n_total = int(y_seg.size)
        n_used = int(np.count_nonzero(ok))

        if n_used == 0:
            summary_lines.append(f"{name}: no finite samples after masking (used 0/{n_total})")
            return

        yy = y_seg[ok]
        summary_lines.append(
            f"{name}: n={n_used}/{n_total} | "
            f"min={np.min(yy):.2f}, max={np.max(yy):.2f}, "
            f"mean={np.mean(yy):.2f}, median={np.median(yy):.2f}, std={np.std(yy):.2f}"
        )

        nonlocal seg_hr_min, seg_hr_max
        y0 = float(np.min(yy))
        y1 = float(np.max(yy))
        seg_hr_min = y0 if seg_hr_min is None else min(seg_hr_min, y0)
        seg_hr_max = y1 if seg_hr_max is None else max(seg_hr_max, y1)

    if t_hr_calc_raw is None:
        t_hr_calc_raw = t_hr_calc
    if hr_calc_raw is None:
        hr_calc_raw = hr_calc

    if t_hr_calc_raw is not None and hr_calc_raw is not None:
        mask_calc = (t_hr_calc_raw >= seg_start_sec) & (t_hr_calc_raw <= seg_end_sec)
        if np.any(mask_calc):
            t_sec_seg = t_hr_calc_raw[mask_calc]
            y_raw = hr_calc_raw[mask_calc].astype(float)

            _plot_no_bridge(
                ax,
                x_sec=t_sec_seg,
                y=y_raw,
                label="HR raw",
                linestyle="--",
                linewidth=1.0,
                color="tab:blue",
                alpha=0.6,
                zorder=0,
            )

            if t_hr_calc is not None and hr_calc is not None and np.size(hr_calc) == np.size(t_hr_calc):
                mask_clean = (t_hr_calc >= seg_start_sec) & (t_hr_calc <= seg_end_sec)
                if np.any(mask_clean) and np.size(t_hr_calc[mask_clean]) == np.size(t_sec_seg):
                    y_clean = hr_calc[mask_clean].astype(float)
                else:
                    y_clean = _apply_global_mask_to_series(t_sec_seg, y_raw, aux_df)
            else:
                y_clean = _apply_global_mask_to_series(t_sec_seg, y_raw, aux_df)

            ok = np.isfinite(y_clean)
            if np.any(ok):
                _plot_no_bridge(
                    ax,
                    x_sec=t_sec_seg,
                    y=y_clean,
                    label="HR used",
                    linestyle="-",
                    linewidth=1.4,
                    color="tab:blue",
                    alpha=0.95,
                    zorder=3,
                )

            _stats_line("HR", y_clean)

    ax.set_ylabel("HR [bpm]")
    ax.grid(True)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
        fontsize=9,
    )

    if summary_lines:
        ax.text(
            0.01,
            0.92,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            fontsize=9,
            va="top",
        )

    if seg_hr_min is not None and seg_hr_max is not None:
        margin = 0.1 * (seg_hr_max - seg_hr_min + 1e-6)
        ax.set_ylim(seg_hr_min - margin, seg_hr_max + margin)

    return seg_hr_min, seg_hr_max


def _plot_segment_hrv(
    ax: plt.Axes,
    t_hrv: np.ndarray,
    hrv_clean: np.ndarray,
    hrv_raw: Optional[np.ndarray],
    seg_start_sec: float,
    seg_end_sec: float,
    exclusion_zones: List[Tuple[float, float, str]],
    t_seg_h_start: float,
    t_seg_h_end: float,
    aux_df: Optional["pd.DataFrame"],
) -> tuple[Optional[float], Optional[float]]:
    _add_exclusion_spans(ax, exclusion_zones, t_seg_h_start, t_seg_h_end, label_once=True)

    hrv_ymin = hrv_ymax = None
    use_raw = hrv_raw is not None and np.size(hrv_raw) == np.size(hrv_clean)

    mask = (t_hrv >= seg_start_sec) & (t_hrv <= seg_end_sec)
    if np.any(mask):
        t_sec_seg = t_hrv[mask].astype(float)
        legend_patches: List[Patch] = []

        if aux_df is not None and bool(getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False)):
            m_sleep_keep = sleep_mask.build_sleep_include_mask_for_times(t_sec_seg, aux_df)
            if m_sleep_keep is not None:
                _shade_masked_regions(
                    ax,
                    t_sec=t_sec_seg,
                    masked=~m_sleep_keep,
                    color="0.7",
                    alpha=0.18,
                )
                legend_patches.append(Patch(facecolor="0.7", alpha=0.18, label="Sleep masked"))

        if aux_df is not None:
            m_evt_keep = io_aux_csv.build_time_exclusion_mask(t_sec_seg, aux_df)
            if m_evt_keep is not None:
                _shade_masked_regions(
                    ax,
                    t_sec=t_sec_seg,
                    masked=~m_evt_keep,
                    color="tab:red",
                    alpha=0.12,
                )
                legend_patches.append(Patch(facecolor="tab:red", alpha=0.12, label="Events excluded"))

        if use_raw:
            yc_seg = hrv_clean[mask]
            yr_seg = hrv_raw[mask]
            rr_only_excluded = np.isfinite(yr_seg) & ~np.isfinite(yc_seg)

            if np.any(rr_only_excluded):
                _shade_masked_regions(
                    ax,
                    t_sec=t_sec_seg,
                    masked=rr_only_excluded,
                    color="tab:blue",
                    alpha=0.22,
                )
                legend_patches.append(Patch(facecolor="tab:blue", alpha=0.22, label="RR rejected"))

        if use_raw:
            yr_seg = hrv_raw[mask]
            raw_missing = ~np.isfinite(yr_seg)
            if np.any(raw_missing):
                _shade_masked_regions(
                    ax,
                    t_sec=t_sec_seg,
                    masked=raw_missing,
                    color="gold",
                    alpha=0.45,
                )
                legend_patches.append(Patch(facecolor="gold", alpha=0.45, label="Raw HRV missing"))

        yc = hrv_clean[mask].astype(float)

        if use_raw:
            yr = hrv_raw[mask].astype(float)
            if np.any(np.isfinite(yr)):
                _plot_no_bridge(
                    ax,
                    x_sec=t_sec_seg,
                    y=yr,
                    label="HRV RMSSD (raw)",
                    linestyle="--",
                    linewidth=1.1,
                    color="tab:green",
                    alpha=0.45,
                    zorder=1,
                )

        if np.any(np.isfinite(yc)):
            _plot_no_bridge(
                ax,
                x_sec=t_sec_seg,
                y=yc,
                label="HRV RMSSD (used)",
                linestyle="-",
                linewidth=1.8,
                color="tab:green",
                alpha=0.95,
                zorder=3,
            )

            y0 = float(np.nanmin(yc[np.isfinite(yc)]))
            y1 = float(np.nanmax(yc[np.isfinite(yc)]))
            if np.isfinite(y0) and np.isfinite(y1) and y1 > y0:
                m = 0.10 * (y1 - y0)
                ax.set_ylim(y0 - m, y1 + m)

            hrv_ymin, hrv_ymax = ax.get_ylim()
        else:
            ax.text(
                0.01,
                0.92,
                "HRV: no valid RMSSD windows in this segment",
                transform=ax.transAxes,
                fontsize=9,
                va="top",
            )

    ax.set_ylabel("RMSSD [ms]")
    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()

    if "legend_patches" in locals() and legend_patches:
        keep_patch_labels = {"Events excluded", "Sleep masked"}
        for p in legend_patches:
            if p.get_label() in keep_patch_labels:
                handles.append(p)
                labels.append(p.get_label())

    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen and l != "_nolegend_":
            h2.append(h)
            l2.append(l)
            seen.add(l)

    ax.legend(
        h2,
        l2,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
        fontsize=9,
    )

    return hrv_ymin, hrv_ymax


def _plot_segment_delta_hr(
    ax: plt.Axes,
    t_hr_edf: Optional[np.ndarray],
    delta_hr_edf: Optional[np.ndarray],
    t_hr_calc: Optional[np.ndarray],
    delta_hr_calc: Optional[np.ndarray],
    delta_hr_edf_evt: Optional[np.ndarray],
    delta_hr_calc_evt: Optional[np.ndarray],
    seg_start_sec: float,
    seg_end_sec: float,
    exclusion_zones: List[Tuple[float, float, str]],
    t_seg_h_start: float,
    t_seg_h_end: float,
    aux_df: Optional["pd.DataFrame"],
) -> tuple[Optional[float], Optional[float]]:
    t_hr_edf = None
    delta_hr_edf = None
    delta_hr_edf_evt = None

    _add_exclusion_spans(ax, exclusion_zones, t_seg_h_start, t_seg_h_end, label_once=True)

    y_min = y_max = None
    summary_lines: List[str] = []

    def _stats_line(name: str, y_seg: np.ndarray) -> None:
        ok = np.isfinite(y_seg)
        n_total = int(y_seg.size)
        n_used = int(np.count_nonzero(ok))

        if n_used == 0:
            summary_lines.append(f"{name}: no finite samples after masking (used 0/{n_total})")
            return

        yy = y_seg[ok]
        summary_lines.append(
            f"{name}: n={n_used}/{n_total} | "
            f"min={np.min(yy):.1f}, max={np.max(yy):.1f}, "
            f"mean={np.mean(yy):.1f}, median={np.median(yy):.1f}, std={np.std(yy):.1f}"
        )

    if t_hr_calc is not None:
        mask_seg = (t_hr_calc >= seg_start_sec) & (t_hr_calc <= seg_end_sec)
        if np.any(mask_seg):
            t_sec_seg = t_hr_calc[mask_seg]
            th = t_sec_seg / 3600.0

            if delta_hr_calc is not None and np.size(delta_hr_calc) == np.size(t_hr_calc):
                y_full = delta_hr_calc[mask_seg].astype(float)
                ok = np.isfinite(y_full)
                if np.any(ok):
                    ax.plot(
                        th,
                        np.ma.masked_invalid(y_full),
                        label="ΔHR (full)",
                        linewidth=0.9,
                        alpha=0.45,
                        linestyle="--",
                        zorder=2,
                    )
                    _stats_line("ΔHR full", y_full)

            if delta_hr_calc_evt is not None and np.size(delta_hr_calc_evt) == np.size(t_hr_calc):
                y_evt = delta_hr_calc_evt[mask_seg].astype(float)
                ok2 = np.isfinite(y_evt)
                if np.any(ok2):
                    ax.plot(
                        th,
                        np.ma.masked_invalid(y_evt),
                        label="ΔHR (events)",
                        linewidth=1.4,
                        alpha=0.95,
                        zorder=4,
                    )
                    _stats_line("ΔHR events", y_evt)

    ax.set_ylabel("ΔHR [bpm]")
    ax.grid(True)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
        fontsize=9,
    )

    ax.axhline(0.0, linewidth=0.8, alpha=0.5, zorder=0)

    if not summary_lines:
        ax.text(
            0.01,
            0.92,
            "ΔHR:",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
        )
    else:
        ax.text(
            0.01,
            0.92,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            fontsize=9,
            va="top",
        )

    if y_min is not None and y_max is not None:
        margin = 0.15 * (y_max - y_min + 1e-6)
        ax.set_ylim(y_min - margin, y_max + margin)

    return y_min, y_max


def _overlay_events_on_axes(
    aux_df: Optional["pd.DataFrame"],
    seg_start_sec: float,
    seg_end_sec: float,
    ax_hr: plt.Axes,
    ax_hrv: Optional[plt.Axes],
    ax_amp: Optional[plt.Axes],
    ax_delta: Optional[plt.Axes],
    hr_ylim: tuple[float, float],
    hrv_ylim: Optional[tuple[float, float]],
    amp_ylim: Optional[tuple[float, float]],
    delta_ylim: Optional[tuple[float, float]],
    event_spec: List[EventSpec] = DEFAULT_EVENT_PLOT_SPEC,
) -> None:
    if aux_df is None:
        return

    time_col = getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec")
    if time_col not in aux_df.columns:
        return

    mask = (aux_df[time_col] >= seg_start_sec) & (aux_df[time_col] <= seg_end_sec)
    if not mask.any():
        return

    seg = aux_df.loc[mask]

    used_hr = set()
    used_hrv = set()
    used_amp = set()
    used_delta = set()

    hr_ymin, hr_ymax = hr_ylim
    hrv_ymin, hrv_ymax = hrv_ylim if hrv_ylim is not None else (None, None)
    amp_ymin, amp_ymax = amp_ylim if amp_ylim is not None else (None, None)
    delta_ymin, delta_ymax = delta_ylim if delta_ylim is not None else (None, None)

    def _scatter_desat(ax: plt.Axes, t_h: np.ndarray, y0: float, y1: float, color: str, label: str) -> None:
        y_desat = y0 + 0.03 * (y1 - y0)
        ax.scatter(
            t_h,
            np.full_like(t_h, y_desat, dtype=float),
            marker="v",
            s=25,
            color=color,
            alpha=0.85,
            label=label,
            zorder=5,
        )

    for spec in event_spec:
        if spec.col not in seg.columns:
            continue

        m = seg[spec.col] == 1
        if not m.any():
            continue

        t_evt_h = seg.loc[m, time_col].to_numpy(float) / 3600.0

        label_hr = spec.label if spec.label not in used_hr else "_nolegend_"
        used_hr.add(spec.label)

        if spec.col == "desat_flag":
            _scatter_desat(ax_hr, t_evt_h, hr_ymin, hr_ymax, spec.color, label_hr)
        else:
            first = label_hr != "_nolegend_"
            for x in t_evt_h:
                ax_hr.axvline(
                    x,
                    color=spec.color,
                    linestyle="-",
                    linewidth=2.0,
                    alpha=0.9,
                    label=spec.label if first else "_nolegend_",
                    zorder=5,
                )
                first = False

        if ax_hrv is not None and hrv_ymin is not None and hrv_ymax is not None:
            label_hrv = spec.label if spec.label not in used_hrv else "_nolegend_"
            used_hrv.add(spec.label)

            if spec.col == "desat_flag":
                _scatter_desat(ax_hrv, t_evt_h, hrv_ymin, hrv_ymax, spec.color, label_hrv)
            else:
                first = label_hrv != "_nolegend_"
                for x in t_evt_h:
                    ax_hrv.axvline(
                        x,
                        color=spec.color,
                        linestyle="-",
                        linewidth=1.8,
                        alpha=0.85,
                        label=spec.label if first else "_nolegend_",
                        zorder=5,
                    )
                    first = False

        if ax_amp is not None and amp_ymin is not None and amp_ymax is not None:
            label_amp = spec.label if spec.label not in used_amp else "_nolegend_"
            used_amp.add(spec.label)

            if spec.col == "desat_flag":
                _scatter_desat(ax_amp, t_evt_h, amp_ymin, amp_ymax, spec.color, label_amp)
            else:
                first = label_amp != "_nolegend_"
                for x in t_evt_h:
                    ax_amp.axvline(
                        x,
                        color=spec.color,
                        linestyle="-",
                        linewidth=1.8,
                        alpha=0.85,
                        label=spec.label if first else "_nolegend_",
                        zorder=5,
                    )
                    first = False

        if ax_delta is not None and delta_ymin is not None and delta_ymax is not None:
            label_delta = spec.label if spec.label not in used_delta else "_nolegend_"
            used_delta.add(spec.label)

            if spec.col == "desat_flag":
                _scatter_desat(ax_delta, t_evt_h, delta_ymin, delta_ymax, spec.color, label_delta)
            else:
                first = label_delta != "_nolegend_"
                for x in t_evt_h:
                    ax_delta.axvline(
                        x,
                        color=spec.color,
                        linestyle="-",
                        linewidth=1.8,
                        alpha=0.85,
                        label=spec.label if first else "_nolegend_",
                        zorder=5,
                    )
                    first = False


def _add_segment_pages_to_pdf(
    pdf: PdfPages,
    *,
    signal_raw: np.ndarray,
    signal_filt: np.ndarray,
    sfreq: float,
    segment_minutes: float,
    title_prefix: str,
    channel_name: str,
    t_hr_calc: Optional[np.ndarray],
    hr_calc: Optional[np.ndarray],
    t_hr_edf: Optional[np.ndarray],
    hr_edf: Optional[np.ndarray],
    t_hrv: Optional[np.ndarray],
    hrv_clean: Optional[np.ndarray],
    hrv_raw: Optional[np.ndarray],
    aux_df: Optional["pd.DataFrame"],
    exclusion_zones: List[Tuple[float, float, str]],
    t_pat_amp: Optional[np.ndarray],
    pat_amp: Optional[np.ndarray],
    delta_hr_calc: Optional[np.ndarray],
    delta_hr_edf: Optional[np.ndarray],
    delta_hr_calc_evt: Optional[np.ndarray],
    delta_hr_edf_evt: Optional[np.ndarray],
    t_hr_calc_raw: Optional[np.ndarray],
    hr_calc_raw: Optional[np.ndarray],
    t_hr_edf_raw: Optional[np.ndarray],
    hr_edf_raw: Optional[np.ndarray],
) -> None:
    """
    Segment page subplot order NOW:
      1) HR
      2) ΔHR (if enabled + available)
      3) HRV RMSSD (if available)

    PAT raw signal and PAT amp / burden are intentionally hidden.
    """
    n_samples = len(signal_raw)
    samples_per_segment = int(segment_minutes * 60.0 * sfreq)
    segment_index = 0

    use_pat_signal_plot = False
    use_pat_amp = False
    use_hrv = t_hrv is not None and hrv_clean is not None and np.size(hrv_clean) > 0

    for start in range(0, n_samples, samples_per_segment):
        end = min(start + samples_per_segment, n_samples)
        segment_index += 1

        t_seg_sec = np.arange(start, end) / sfreq
        seg_start_sec = float(t_seg_sec[0])
        seg_end_sec = float(t_seg_sec[-1])

        t_seg_h = t_seg_sec / 3600.0
        t_h_start = float(t_seg_h[0])
        t_h_end = float(t_seg_h[-1])

        enable_delta = bool(getattr(config, "ENABLE_DELTA_HR", True))
        delta_mode = str(getattr(config, "DELTA_HR_PLOT_MODE", "subplot")).lower()

        has_any_delta = (
            (delta_hr_calc is not None and np.size(delta_hr_calc) > 0)
            or (delta_hr_calc_evt is not None and np.size(delta_hr_calc_evt) > 0)
        )
        use_delta_subplot = enable_delta and (delta_mode == "subplot") and has_any_delta

        n_rows = 1
        if use_delta_subplot:
            n_rows += 1
        if use_hrv:
            n_rows += 1

        height_ratios: List[float] = [1]
        if use_delta_subplot:
            height_ratios.append(1)
        if use_hrv:
            height_ratios.append(1)

        fig, axes = plt.subplots(
            n_rows,
            1,
            figsize=(11.69, 8.27),
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        if n_rows == 1:
            axes = [axes]

        idx = 0
        ax_hr = axes[idx]
        idx += 1

        ax_delta = None
        if use_delta_subplot:
            ax_delta = axes[idx]
            idx += 1

        ax_hrv = None
        if use_hrv:
            ax_hrv = axes[idx]
            idx += 1

        _plot_segment_hr(
            ax_hr,
            t_hr_edf=t_hr_edf,
            hr_edf=hr_edf,
            t_hr_calc=t_hr_calc,
            hr_calc=hr_calc,
            t_hr_edf_raw=t_hr_edf_raw,
            hr_edf_raw=hr_edf_raw,
            t_hr_calc_raw=t_hr_calc_raw,
            hr_calc_raw=hr_calc_raw,
            seg_start_sec=seg_start_sec,
            seg_end_sec=seg_end_sec,
            exclusion_zones=exclusion_zones,
            t_seg_h_start=t_h_start,
            t_seg_h_end=t_h_end,
            aux_df=aux_df,
            t_seg_sec=t_seg_sec,
        )
        hr_ylim = ax_hr.get_ylim()

        delta_ylim = None
        if ax_delta is not None:
            _plot_segment_delta_hr(
                ax_delta,
                t_hr_edf=t_hr_edf,
                delta_hr_edf=delta_hr_edf,
                t_hr_calc=t_hr_calc,
                delta_hr_calc=delta_hr_calc,
                delta_hr_edf_evt=delta_hr_edf_evt,
                delta_hr_calc_evt=delta_hr_calc_evt,
                seg_start_sec=seg_start_sec,
                seg_end_sec=seg_end_sec,
                exclusion_zones=exclusion_zones,
                t_seg_h_start=t_h_start,
                t_seg_h_end=t_h_end,
                aux_df=aux_df,
            )
            delta_ylim = ax_delta.get_ylim()

        hrv_ylim = None
        if ax_hrv is not None and t_hrv is not None and hrv_clean is not None:
            _plot_segment_hrv(
                ax_hrv,
                t_hrv,
                hrv_clean,
                hrv_raw,
                seg_start_sec,
                seg_end_sec,
                exclusion_zones,
                t_h_start,
                t_h_end,
                aux_df=aux_df,
            )
            hrv_ylim = ax_hrv.get_ylim()

        _overlay_events_on_axes(
            aux_df,
            seg_start_sec,
            seg_end_sec,
            ax_hr=ax_hr,
            ax_hrv=ax_hrv if use_hrv else None,
            ax_amp=None,
            ax_delta=ax_delta if use_delta_subplot else None,
            hr_ylim=hr_ylim,
            hrv_ylim=hrv_ylim,
            amp_ylim=None,
            delta_ylim=delta_ylim,
        )

        if ax_hrv is not None:
            ax_hrv.set_xlabel("Time (hours from recording start)")
        elif ax_delta is not None:
            ax_delta.set_xlabel("Time (hours from recording start)")
        else:
            ax_hr.set_xlabel("Time (hours from recording start)")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)