from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple, cast

import numpy as np
from matplotlib.patches import Patch

from .. import config, io_aux_csv, sleep_mask
from ..metrics.hr_event_response import extract_event_hr_windows
from .specs import DEFAULT_EVENT_PLOT_SPEC, EventSpec
from .utils import _add_exclusion_spans, _shade_masked_regions

if TYPE_CHECKING:
    import pandas as pd


def _plot_no_bridge(
    ax,
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

    x_ok = x_sec[np.where(ok)[0]]
    y_ok = y[np.where(ok)[0]]
    order = np.argsort(x_ok)
    x_ok = x_ok[order]
    y_ok = y_ok[order]
    d = np.diff(x_ok)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return
    gap_thr = max(float(min_gap_sec), float(gap_factor) * float(np.median(d)))
    cut = np.where(np.diff(x_ok) > gap_thr)[0] + 1
    runs = np.split(np.arange(x_ok.size), cut)
    first = True
    for r in runs:
        if r.size < 2:
            continue
        ax.plot(x_ok[r] / 3600.0, y_ok[r], linestyle=linestyle, linewidth=linewidth, color=color, alpha=alpha, label=label if first else "_nolegend_", zorder=zorder)
        first = False


def _apply_global_mask_to_series(
    t_sec: np.ndarray,
    y: np.ndarray,
    aux_df: Optional["pd.DataFrame"],
) -> np.ndarray:
    y2 = y.astype(float, copy=True)
    if aux_df is None or t_sec is None or t_sec.size == 0:
        return y2
    m_keep = sleep_mask.build_global_include_mask_for_times(t_sec, aux_df, apply_sleep=True, apply_events=True)
    if m_keep is None:
        return y2
    y2[~m_keep] = np.nan
    return y2


def _overlay_pat_burden_area(
    ax,
    *,
    t_sec_all: np.ndarray,
    pat_amp_all: np.ndarray,
    aux_df: Optional["pd.DataFrame"],
    seg_start_sec: float,
    seg_end_sec: float,
) -> None:
    if aux_df is None:
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

    for s, e in _runs(m_inside):
        t0 = float(t[s])
        t1 = float(t[e - 1])
        if not (np.isfinite(t0) and np.isfinite(t1)) or (t1 - t0) < min_ep_sec:
            continue
        m_pre = (t >= (t0 - lookback)) & (t < t0) & m_baseline_ok
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

        ax.plot(tt / 3600.0, np.full_like(tt, baseline, dtype=float), linestyle="--", linewidth=1.1, alpha=0.7, color="0.25", label="_nolegend_", zorder=2)
        ax.fill_between(tt / 3600.0, yy, baseline, where=below, interpolate=True, alpha=0.24, color="#2a9d8f", label="_nolegend_", zorder=1)


def _plot_segment_pat_amp(
    ax,
    t_pat_amp: np.ndarray,
    pat_amp: np.ndarray,
    seg_start_sec: float,
    seg_end_sec: float,
    exclusion_zones: List[Tuple[float, float, str]],
    t_seg_h_start: float,
    t_seg_h_end: float,
    aux_df: Optional["pd.DataFrame"],
) -> tuple[Optional[float], Optional[float]]:
    _add_exclusion_spans(ax, exclusion_zones, t_seg_h_start, t_seg_h_end, label_once=False)
    mask = (t_pat_amp >= seg_start_sec) & (t_pat_amp <= seg_end_sec)
    if not np.any(mask):
        ax.set_ylabel("PAT AMP")
        ax.grid(True)
        return None, None

    t_sec_seg = t_pat_amp[mask].astype(float)
    y_seg = pat_amp[mask].astype(float)

    legend_patches: List[Patch] = []
    combined_keep = None
    if aux_df is not None:
        if bool(getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False)):
            m_sleep_keep = sleep_mask.build_sleep_include_mask_for_times(t_sec_seg, aux_df)
            if m_sleep_keep is not None:
                _shade_masked_regions(ax, t_sec=t_sec_seg, masked=~m_sleep_keep, color="0.7", alpha=0.18)
                legend_patches.append(Patch(facecolor="0.7", alpha=0.18, label="Sleep masked"))
        m_evt_keep = io_aux_csv.build_time_exclusion_mask(t_sec_seg, aux_df)
        if m_evt_keep is not None:
            _shade_masked_regions(ax, t_sec=t_sec_seg, masked=~m_evt_keep, color="tab:red", alpha=0.12)
            legend_patches.append(Patch(facecolor="tab:red", alpha=0.12, label="Events excluded"))
        combined_keep = sleep_mask.build_global_include_mask_for_times(t_sec_seg, aux_df, apply_sleep=True, apply_events=True)

    invalid_mask = ~np.isfinite(y_seg)
    if combined_keep is not None and np.size(combined_keep) == np.size(invalid_mask):
        invalid_mask = invalid_mask & np.asarray(combined_keep, dtype=bool)
    if np.any(invalid_mask):
        _shade_masked_regions(ax, t_sec=t_sec_seg, masked=invalid_mask, color="gold", alpha=0.45)
        legend_patches.append(Patch(facecolor="gold", alpha=0.45, label="Metric invalid"))

    if np.any(np.isfinite(y_seg)):
        _plot_no_bridge(ax, x_sec=t_sec_seg, y=y_seg, label="PAT AMP", linestyle="-", linewidth=1.1, color="tab:orange", alpha=0.9, zorder=3)
        _overlay_pat_burden_area(ax, t_sec_all=t_pat_amp, pat_amp_all=pat_amp, aux_df=aux_df, seg_start_sec=seg_start_sec, seg_end_sec=seg_end_sec)
        legend_patches.append(Patch(facecolor="#2a9d8f", alpha=0.24, label="PAT burden area"))
        yy = y_seg[np.isfinite(y_seg)]
        y0 = float(np.min(yy))
        y1 = float(np.max(yy))
        if np.isfinite(y0) and np.isfinite(y1) and y1 > y0:
            margin = 0.10 * (y1 - y0)
            ax.set_ylim(y0 - margin, y1 + margin)
    else:
        ax.text(0.01, 0.92, "PAT AMP unavailable in this segment", transform=ax.transAxes, fontsize=9, va="top")

    ax.set_ylabel("PAT AMP")
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen and l != "_nolegend_":
            h2.append(h)
            l2.append(l)
            seen.add(l)
    for patch in legend_patches:
        if patch.get_label() not in seen:
            h2.append(patch)
            l2.append(patch.get_label())
            seen.add(patch.get_label())
    if h2:
        ax.legend(h2, l2, loc="upper right", fontsize=8, framealpha=0.9)
    return ax.get_ylim()


def _plot_segment_pwa_drop(
    ax,
    t_pwa: np.ndarray,
    pwa_series: np.ndarray,
    pwa_drop_events: Optional[list[dict[str, float]]],
    seg_start_sec: float,
    seg_end_sec: float,
    exclusion_zones: List[Tuple[float, float, str]],
    t_seg_h_start: float,
    t_seg_h_end: float,
    aux_df: Optional["pd.DataFrame"],
) -> tuple[Optional[float], Optional[float]]:
    _add_exclusion_spans(ax, exclusion_zones, t_seg_h_start, t_seg_h_end, label_once=True)
    mask = (t_pwa >= seg_start_sec) & (t_pwa <= seg_end_sec)
    if not np.any(mask):
        ax.set_ylabel("PWA")
        ax.grid(True)
        return None, None

    t_sec_seg = t_pwa[mask].astype(float)
    y_seg = pwa_series[mask].astype(float)
    legend_patches: List[Patch] = []
    combined_keep = None
    if aux_df is not None:
        if bool(getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False)):
            m_sleep_keep = sleep_mask.build_sleep_include_mask_for_times(t_sec_seg, aux_df)
            if m_sleep_keep is not None:
                _shade_masked_regions(ax, t_sec=t_sec_seg, masked=~m_sleep_keep, color="0.7", alpha=0.18)
                legend_patches.append(Patch(facecolor="0.7", alpha=0.18, label="Sleep masked"))
        m_evt_keep = io_aux_csv.build_time_exclusion_mask(t_sec_seg, aux_df)
        if m_evt_keep is not None:
            _shade_masked_regions(ax, t_sec=t_sec_seg, masked=~m_evt_keep, color="tab:red", alpha=0.12)
            legend_patches.append(Patch(facecolor="tab:red", alpha=0.12, label="Events excluded"))
        combined_keep = sleep_mask.build_global_include_mask_for_times(t_sec_seg, aux_df, apply_sleep=True, apply_events=True)

    invalid_mask = ~np.isfinite(y_seg)
    if combined_keep is not None and np.size(combined_keep) == np.size(invalid_mask):
        invalid_mask = invalid_mask & np.asarray(combined_keep, dtype=bool)
    if np.any(invalid_mask):
        _shade_masked_regions(ax, t_sec=t_sec_seg, masked=invalid_mask, color="gold", alpha=0.45)
        legend_patches.append(Patch(facecolor="gold", alpha=0.45, label="Metric invalid"))

    y_min = y_max = None
    if np.any(np.isfinite(y_seg)):
        _plot_no_bridge(ax, x_sec=t_sec_seg, y=y_seg, label="PWA", linestyle="-", linewidth=1.1, color="tab:purple", alpha=0.9, zorder=3)
        yy = y_seg[np.isfinite(y_seg)]
        y_min = float(np.min(yy))
        y_max = float(np.max(yy))

    events = pwa_drop_events or []
    first = True
    for event in events:
        t0 = float(event.get("t_start", np.nan))
        t1 = float(event.get("t_end", np.nan))
        if not (np.isfinite(t0) and np.isfinite(t1) and t1 > t0):
            continue
        if t1 < seg_start_sec or t0 > seg_end_sec:
            continue
        ax.axvspan(t0 / 3600.0, t1 / 3600.0, color="#9467bd", alpha=0.16, label="PWA-drop" if first else "_nolegend_", zorder=1)
        first = False

    ax.set_ylabel("PWA")
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen and l != "_nolegend_":
            h2.append(h)
            l2.append(l)
            seen.add(l)
    for patch in legend_patches:
        if patch.get_label() not in seen:
            h2.append(patch)
            l2.append(patch.get_label())
            seen.add(patch.get_label())
    if h2:
        ax.legend(h2, l2, loc="upper right", fontsize=8, framealpha=0.9)
    if y_min is not None and y_max is not None:
        margin = 0.10 * (y_max - y_min + 1e-6)
        ax.set_ylim(y_min - margin, y_max + margin)
    return y_min, y_max


def _plot_segment_hr(
    ax,
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
    legend_patches: List[Patch] = []
    _add_exclusion_spans(ax, exclusion_zones, t_seg_h_start, t_seg_h_end, label_once=True)

    seg_hr_min: Optional[float] = None
    seg_hr_max: Optional[float] = None
    def _update_limits(y_seg: np.ndarray) -> None:
        ok = np.isfinite(y_seg)
        if not np.any(ok):
            return
        yy = y_seg[ok]
        nonlocal seg_hr_min, seg_hr_max
        y0 = float(np.min(yy))
        y1 = float(np.max(yy))
        seg_hr_min = y0 if seg_hr_min is None else min(seg_hr_min, y0)
        seg_hr_max = y1 if seg_hr_max is None else max(seg_hr_max, y1)

    show_raw_debug = bool(getattr(config, "PLOT_SHOW_RAW_DEBUG_OVERLAYS", False))

    if t_hr_calc_raw is None:
        t_hr_calc_raw = t_hr_calc
    if hr_calc_raw is None:
        hr_calc_raw = hr_calc

    y_clean = None
    t_sec_seg = None

    if show_raw_debug and t_hr_calc_raw is not None and hr_calc_raw is not None:
        mask_calc = (t_hr_calc_raw >= seg_start_sec) & (t_hr_calc_raw <= seg_end_sec)
        if np.any(mask_calc):
            t_sec_seg = t_hr_calc_raw[mask_calc]
            y_raw = hr_calc_raw[mask_calc].astype(float)
            _plot_no_bridge(ax, x_sec=t_sec_seg, y=y_raw, label="HR raw", linestyle="--", linewidth=1.0, color="tab:blue", alpha=0.6, zorder=0)
            if t_hr_calc is not None and hr_calc is not None and np.size(hr_calc) == np.size(t_hr_calc):
                mask_clean = (t_hr_calc >= seg_start_sec) & (t_hr_calc <= seg_end_sec)
                if np.any(mask_clean) and np.size(t_hr_calc[mask_clean]) == np.size(t_sec_seg):
                    y_clean = hr_calc[mask_clean].astype(float)
                else:
                    y_clean = _apply_global_mask_to_series(t_sec_seg, y_raw, aux_df)
            else:
                y_clean = _apply_global_mask_to_series(t_sec_seg, y_raw, aux_df)

    if y_clean is None and t_hr_calc is not None and hr_calc is not None:
        mask_clean = (t_hr_calc >= seg_start_sec) & (t_hr_calc <= seg_end_sec)
        if np.any(mask_clean):
            t_sec_seg = t_hr_calc[mask_clean]
            y_clean = hr_calc[mask_clean].astype(float)

    combined_keep = None
    if aux_df is not None and t_sec_seg is not None and t_sec_seg.size > 0:
        if bool(getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False)):
            m_sleep_keep = sleep_mask.build_sleep_include_mask_for_times(t_sec_seg, aux_df)
            if m_sleep_keep is not None:
                _shade_masked_regions(ax, t_sec=t_sec_seg, masked=~m_sleep_keep, color="0.7", alpha=0.18)
                legend_patches.append(Patch(facecolor="0.7", alpha=0.18, label="Sleep masked"))
        m_evt_keep = io_aux_csv.build_time_exclusion_mask(t_sec_seg, aux_df)
        if m_evt_keep is not None:
            _shade_masked_regions(ax, t_sec=t_sec_seg, masked=~m_evt_keep, color="tab:red", alpha=0.12)
            legend_patches.append(Patch(facecolor="tab:red", alpha=0.12, label="Events excluded"))
        combined_keep = sleep_mask.build_global_include_mask_for_times(t_sec_seg, aux_df, apply_sleep=True, apply_events=True)

    if y_clean is not None and t_sec_seg is not None:
        invalid_mask = ~np.isfinite(y_clean)
        if combined_keep is not None and np.size(combined_keep) == np.size(invalid_mask):
            invalid_mask = invalid_mask & np.asarray(combined_keep, dtype=bool)
        if np.any(invalid_mask):
            _shade_masked_regions(ax, t_sec=t_sec_seg, masked=invalid_mask, color="gold", alpha=0.45)
            legend_patches.append(Patch(facecolor="gold", alpha=0.45, label="Metric invalid"))

    if y_clean is not None and t_sec_seg is not None and np.any(np.isfinite(y_clean)):
        _plot_no_bridge(ax, x_sec=t_sec_seg, y=y_clean, label="HR used", linestyle="-", linewidth=1.4, color="tab:blue", alpha=0.95, zorder=3)
        _update_limits(y_clean)

    ax.set_ylabel("HR [bpm]")
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l and l != "_nolegend_" and l not in seen:
            h2.append(h)
            l2.append(l)
            seen.add(l)
    for patch in legend_patches:
        if patch.get_label() not in seen:
            h2.append(patch)
            l2.append(patch.get_label())
            seen.add(patch.get_label())
    if h2:
        ax.legend(h2, l2, loc="upper right", fontsize=8, framealpha=0.9)
    if seg_hr_min is not None and seg_hr_max is not None:
        margin = 0.1 * (seg_hr_max - seg_hr_min + 1e-6)
        ax.set_ylim(seg_hr_min - margin, seg_hr_max + margin)
    return seg_hr_min, seg_hr_max


def _plot_segment_prv(
    ax,
    t_prv: np.ndarray,
    prv_clean: np.ndarray,
    prv_raw: Optional[np.ndarray],
    seg_start_sec: float,
    seg_end_sec: float,
    exclusion_zones: List[Tuple[float, float, str]],
    t_seg_h_start: float,
    t_seg_h_end: float,
    aux_df: Optional["pd.DataFrame"],
    *,
    ylabel: str,
    empty_text: str,
    clean_label: str,
    raw_label: str,
) -> tuple[Optional[float], Optional[float]]:
    _add_exclusion_spans(ax, exclusion_zones, t_seg_h_start, t_seg_h_end, label_once=True)
    prv_ymin = prv_ymax = None
    legend_patches: List[Patch] = []
    show_raw_debug = bool(getattr(config, "PLOT_SHOW_RAW_DEBUG_OVERLAYS", False))
    use_raw = show_raw_debug and prv_raw is not None and np.size(prv_raw) == np.size(prv_clean)
    mask = (t_prv >= seg_start_sec) & (t_prv <= seg_end_sec)
    if np.any(mask):
        t_sec_seg = t_prv[mask].astype(float)
        combined_keep = None
        if aux_df is not None and bool(getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False)):
            m_sleep_keep = sleep_mask.build_sleep_include_mask_for_times(t_sec_seg, aux_df)
            if m_sleep_keep is not None:
                _shade_masked_regions(ax, t_sec=t_sec_seg, masked=~m_sleep_keep, color="0.7", alpha=0.18)
                legend_patches.append(Patch(facecolor="0.7", alpha=0.18, label="Sleep masked"))
        if aux_df is not None:
            m_evt_keep = io_aux_csv.build_time_exclusion_mask(t_sec_seg, aux_df)
            if m_evt_keep is not None:
                _shade_masked_regions(ax, t_sec=t_sec_seg, masked=~m_evt_keep, color="tab:red", alpha=0.12)
                legend_patches.append(Patch(facecolor="tab:red", alpha=0.12, label="Events excluded"))
            combined_keep = sleep_mask.build_global_include_mask_for_times(t_sec_seg, aux_df, apply_sleep=True, apply_events=True)
        if use_raw:
            prv_raw_arr = cast(np.ndarray, prv_raw)
            yc_seg = prv_clean[mask]
            yr_seg = prv_raw_arr[mask]
            pr_only_excluded = np.isfinite(yr_seg) & ~np.isfinite(yc_seg)
            if np.any(pr_only_excluded):
                _shade_masked_regions(ax, t_sec=t_sec_seg, masked=pr_only_excluded, color="tab:blue", alpha=0.22)
                legend_patches.append(Patch(facecolor="tab:blue", alpha=0.22, label="Additional calc exclusion"))
        yc = prv_clean[mask].astype(float)
        invalid_mask = ~np.isfinite(yc)
        if combined_keep is not None and np.size(combined_keep) == np.size(invalid_mask):
            invalid_mask = invalid_mask & np.asarray(combined_keep, dtype=bool)
        if np.any(invalid_mask):
            _shade_masked_regions(ax, t_sec=t_sec_seg, masked=invalid_mask, color="gold", alpha=0.45)
            legend_patches.append(Patch(facecolor="gold", alpha=0.45, label="Metric invalid"))
        if use_raw:
            prv_raw_arr = cast(np.ndarray, prv_raw)
            yr = prv_raw_arr[mask].astype(float)
            if np.any(np.isfinite(yr)):
                _plot_no_bridge(ax, x_sec=t_sec_seg, y=yr, label=raw_label, linestyle="--", linewidth=1.1, color="tab:green", alpha=0.45, zorder=1)
        if np.any(np.isfinite(yc)):
            _plot_no_bridge(ax, x_sec=t_sec_seg, y=yc, label=clean_label, linestyle="-", linewidth=1.8, color="tab:green", alpha=0.95, zorder=3)
            y0 = float(np.nanmin(yc[np.isfinite(yc)]))
            y1 = float(np.nanmax(yc[np.isfinite(yc)]))
            if np.isfinite(y0) and np.isfinite(y1) and y1 > y0:
                m = 0.10 * (y1 - y0)
                ax.set_ylim(y0 - m, y1 + m)
            prv_ymin, prv_ymax = ax.get_ylim()
        else:
            ax.text(0.01, 0.92, empty_text, transform=ax.transAxes, fontsize=9, va="top")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen and l != "_nolegend_":
            h2.append(h)
            l2.append(l)
            seen.add(l)
    for patch in legend_patches:
        if patch.get_label() not in seen:
            h2.append(patch)
            l2.append(patch.get_label())
            seen.add(patch.get_label())
    if h2:
        ax.legend(h2, l2, loc="upper right", fontsize=8, framealpha=0.9)
    return prv_ymin, prv_ymax


def _plot_segment_delta_hr(
    ax,
    t_hr_edf: Optional[np.ndarray],
    t_hr_calc: Optional[np.ndarray],
    hr_calc_raw: Optional[np.ndarray],
    seg_start_sec: float,
    seg_end_sec: float,
    exclusion_zones: List[Tuple[float, float, str]],
    t_seg_h_start: float,
    t_seg_h_end: float,
    aux_df: Optional["pd.DataFrame"],
) -> tuple[Optional[float], Optional[float]]:
    t_hr_edf = None
    y_min = y_max = None
    if t_hr_calc is not None and hr_calc_raw is not None:
        mask_seg = (t_hr_calc >= seg_start_sec) & (t_hr_calc <= seg_end_sec)
        if np.any(mask_seg):
            t_sec_seg = t_hr_calc[mask_seg]
            th = t_sec_seg / 3600.0
            y_hr = np.asarray(hr_calc_raw)[mask_seg].astype(float)
            if np.any(np.isfinite(y_hr)):
                ax.plot(th, np.ma.masked_invalid(y_hr), label="HR raw", linewidth=1.0, alpha=0.55, linestyle="-", color="tab:blue", zorder=2)
                yy = y_hr[np.isfinite(y_hr)]
                y_min = float(np.min(yy))
                y_max = float(np.max(yy))

            windows = extract_event_hr_windows(t_hr_calc, np.asarray(hr_calc_raw, dtype=float), aux_df, include_set=set(config.sleep_include_numeric()) if getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False) else None) if aux_df is not None else []
            used_windows = 0
            for w in windows:
                if w["event_end_t"] < seg_start_sec or w["event_start_t"] > seg_end_sec:
                    continue
                used_windows += 1
                ax.axvspan(w["event_start_t"] / 3600.0, w["event_end_t"] / 3600.0, color="tab:cyan", alpha=0.12, label="Event window" if used_windows == 1 else "_nolegend_", zorder=0)
                ax.axvspan(w["recovery_start_t"] / 3600.0, w["recovery_end_t"] / 3600.0, color="tab:green", alpha=0.10, label="Recovery window" if used_windows == 1 else "_nolegend_", zorder=0)
                ax.plot(
                    [w["event_start_t"] / 3600.0, w["event_end_t"] / 3600.0],
                    [w["event_mean_hr"], w["event_mean_hr"]],
                    linestyle="--",
                    linewidth=1.0,
                    color="0.35",
                    alpha=0.85,
                    label="Event mean" if used_windows == 1 else "_nolegend_",
                    zorder=1,
                )
                if np.isfinite(w["event_min_t"]) and np.isfinite(w["event_min_hr"]):
                    ax.scatter(w["event_min_t"] / 3600.0, w["event_min_hr"], color="black", s=14, zorder=4, marker="v", label="Event minimum" if used_windows == 1 else "_nolegend_")
                if np.isfinite(w["recovery_max_t"]) and np.isfinite(w["recovery_max_hr"]):
                    ax.scatter(w["recovery_max_t"] / 3600.0, w["recovery_max_hr"], color="tab:red", s=18, zorder=4, label="Recovery maximum" if used_windows == 1 else "_nolegend_")

    ax.set_ylabel("Event HR [bpm]")
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    if any(label and label != "_nolegend_" for label in labels):
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    if y_min is not None and y_max is not None:
        margin = 0.15 * (y_max - y_min + 1e-6)
        ax.set_ylim(y_min - margin, y_max + margin)
    return y_min, y_max


def _overlay_events_on_axes(
    aux_df: Optional["pd.DataFrame"],
    seg_start_sec: float,
    seg_end_sec: float,
    ax_hr,
    ax_prv,
    ax_prv_sdnn,
    ax_amp,
    ax_delta,
    hr_ylim: Optional[tuple[float, float]],
    prv_ylim,
    prv_sdnn_ylim,
    amp_ylim,
    delta_ylim,
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
    used_hr = set(); used_prv = set(); used_prv_sdnn = set(); used_amp = set(); used_delta = set()
    if hr_ylim is None:
        hr_ymin, hr_ymax = (0.0, 1.0)
    else:
        hr_ymin, hr_ymax = hr_ylim
    prv_ymin, prv_ymax = prv_ylim if prv_ylim is not None else (None, None)
    prv_sdnn_ymin, prv_sdnn_ymax = prv_sdnn_ylim if prv_sdnn_ylim is not None else (None, None)
    amp_ymin, amp_ymax = amp_ylim if amp_ylim is not None else (None, None)
    delta_ymin, delta_ymax = delta_ylim if delta_ylim is not None else (None, None)

    def _event_runs(t_evt_sec: np.ndarray) -> list[tuple[float, float]]:
        if t_evt_sec.size == 0:
            return []
        if t_evt_sec.size == 1:
            return [(float(t_evt_sec[0]), float(t_evt_sec[0]))]
        diffs = np.diff(t_evt_sec)
        pos = diffs[np.isfinite(diffs) & (diffs > 0)]
        step = float(np.median(pos)) if pos.size else 1.0
        gap_thr = 1.5 * step
        cuts = np.where(diffs > gap_thr)[0] + 1
        groups = np.split(t_evt_sec, cuts)
        return [(float(g[0]), float(g[-1] + step)) for g in groups if g.size > 0]

    def _draw_desat_runs(ax, runs: list[tuple[float, float]], color: str, label: str, *, alpha: float) -> None:
        first = label != "_nolegend_"
        for x0_sec, x1_sec in runs:
            ax.axvspan(x0_sec / 3600.0, x1_sec / 3600.0, color=color, alpha=alpha, label=label if first else "_nolegend_", zorder=4)
            first = False

    for spec in event_spec:
        if spec.col not in seg.columns:
            continue
        m = seg[spec.col] == 1
        if not m.any():
            continue
        t_evt_sec = seg.loc[m, time_col].to_numpy(float)
        t_evt_h = seg.loc[m, time_col].to_numpy(float) / 3600.0
        desat_runs = _event_runs(t_evt_sec) if spec.col == "desat_flag" else []
        if ax_hr is not None:
            label_hr = spec.label if spec.label not in used_hr else "_nolegend_"; used_hr.add(spec.label)
            if spec.col == "desat_flag":
                _draw_desat_runs(ax_hr, desat_runs, spec.color, label_hr, alpha=0.14)
            else:
                first = label_hr != "_nolegend_"
                for x in t_evt_h:
                    ax_hr.axvline(x, color=spec.color, linestyle="-", linewidth=2.0, alpha=0.9, label=spec.label if first else "_nolegend_", zorder=5); first = False
        if ax_prv is not None and prv_ymin is not None and prv_ymax is not None:
            label_prv = spec.label if spec.label not in used_prv else "_nolegend_"; used_prv.add(spec.label)
            if spec.col == "desat_flag":
                _draw_desat_runs(ax_prv, desat_runs, spec.color, label_prv, alpha=0.10)
            else:
                first = label_prv != "_nolegend_"
                for x in t_evt_h:
                    ax_prv.axvline(x, color=spec.color, linestyle="-", linewidth=1.8, alpha=0.85, label=spec.label if first else "_nolegend_", zorder=5); first = False
        if ax_prv_sdnn is not None and prv_sdnn_ymin is not None and prv_sdnn_ymax is not None:
            label_prv_sdnn = spec.label if spec.label not in used_prv_sdnn else "_nolegend_"; used_prv_sdnn.add(spec.label)
            if spec.col == "desat_flag":
                _draw_desat_runs(ax_prv_sdnn, desat_runs, spec.color, label_prv_sdnn, alpha=0.10)
            else:
                first = label_prv_sdnn != "_nolegend_"
                for x in t_evt_h:
                    ax_prv_sdnn.axvline(x, color=spec.color, linestyle="-", linewidth=1.8, alpha=0.85, label=spec.label if first else "_nolegend_", zorder=5); first = False
        if ax_amp is not None and amp_ymin is not None and amp_ymax is not None:
            label_amp = spec.label if spec.label not in used_amp else "_nolegend_"; used_amp.add(spec.label)
            if spec.col == "desat_flag":
                _draw_desat_runs(ax_amp, desat_runs, spec.color, label_amp, alpha=0.10)
            else:
                first = label_amp != "_nolegend_"
                for x in t_evt_h:
                    ax_amp.axvline(x, color=spec.color, linestyle="-", linewidth=1.8, alpha=0.85, label=spec.label if first else "_nolegend_", zorder=5); first = False
        if ax_delta is not None and delta_ymin is not None and delta_ymax is not None:
            label_delta = spec.label if spec.label not in used_delta else "_nolegend_"; used_delta.add(spec.label)
            if spec.col == "desat_flag":
                _draw_desat_runs(ax_delta, desat_runs, spec.color, label_delta, alpha=0.10)
            else:
                first = label_delta != "_nolegend_"
                for x in t_evt_h:
                    ax_delta.axvline(x, color=spec.color, linestyle="-", linewidth=1.8, alpha=0.85, label=spec.label if first else "_nolegend_", zorder=5); first = False
