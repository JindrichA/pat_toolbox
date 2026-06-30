from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter

from .. import config, features
from .prv_plot_utils import _add_colored_event_key
from .segment_plot_helpers import (
    _overlay_events_on_axes,
    _plot_segment_actigraph,
    _plot_segment_delta_hr,
    _plot_segment_hr,
    _plot_segment_prv,
    _plot_segment_pat_amp,
    _plot_segment_pat_paper_harmonics,
    _plot_segment_pwa_drop,
    _plot_segment_spo2,
)
from .specs import EventSpec

if TYPE_CHECKING:
    import pandas as pd


def _format_hour_tick(x: float, _pos: float) -> str:
    total_minutes = int(round(float(x) * 60.0))
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours:d}:{minutes:02d}"


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
    t_prv: Optional[np.ndarray],
    prv_clean: Optional[np.ndarray],
    prv_raw: Optional[np.ndarray],
    prv_sdnn_clean: Optional[np.ndarray],
    prv_sdnn_raw: Optional[np.ndarray],
    aux_df: Optional["pd.DataFrame"],
    exclusion_zones: List[Tuple[float, float, str]],
    event_spec: List[EventSpec],
    t_pat_amp: Optional[np.ndarray],
    pat_amp: Optional[np.ndarray],
    t_pwa: Optional[np.ndarray],
    pwa_series: Optional[np.ndarray],
    pwa_drop_events_by_variant: Optional[dict[str, list[dict[str, float]]]],
    t_spo2: Optional[np.ndarray] = None,
    spo2: Optional[np.ndarray] = None,
    t_actigraph: Optional[np.ndarray] = None,
    actigraph: Optional[np.ndarray] = None,
    t_hr_calc_raw: Optional[np.ndarray] = None,
    hr_calc_raw: Optional[np.ndarray] = None,
    t_hr_edf_raw: Optional[np.ndarray] = None,
    hr_edf_raw: Optional[np.ndarray] = None,
    pat_paper_harmonics_windows: Optional[list[dict[str, float]]] = None,
    include_event_vascular: Optional[bool] = None,
    include_prv: Optional[bool] = None,
) -> None:
    if include_event_vascular is None:
        include_event_vascular = True
    if include_prv is None:
        include_prv = bool(getattr(config, "ENABLE_PRV_REPORT_PAGES", True))

    n_samples = len(signal_raw)
    samples_per_segment = int(segment_minutes * 60.0 * sfreq)
    segment_index = 0
    use_hr = include_event_vascular and features.segment_plot_requested("hr") and t_hr_calc is not None and hr_calc is not None
    use_prv = include_prv and features.segment_plot_requested("prv") and t_prv is not None and prv_clean is not None and np.size(prv_clean) > 0
    use_prv_sdnn = use_prv and prv_sdnn_clean is not None and np.size(prv_sdnn_clean) == np.size(t_prv)
    use_pat_amp = include_event_vascular and features.segment_plot_requested("pat_burden")
    use_pwa_drop = include_event_vascular and features.segment_plot_requested("pwa_drop")
    use_spo2 = include_event_vascular and bool(getattr(config, "ENABLE_SPO2_VALIDATION_PLOTS", False)) and t_spo2 is not None and spo2 is not None and np.size(spo2) > 0 and np.size(spo2) == np.size(t_spo2)
    use_paper_harmonics = (
        include_event_vascular
        and bool(getattr(config, "ENABLE_PAT_PAPER_HARMONICS_SEGMENT_QC", False))
        and features.segment_plot_requested("pat_paper_harmonics")
        and bool(pat_paper_harmonics_windows)
    )
    use_actigraph = include_event_vascular and t_actigraph is not None and actigraph is not None and np.size(actigraph) > 0 and np.size(actigraph) == np.size(t_actigraph)
    act_ylim = None
    if use_actigraph and actigraph is not None:
        yy = np.abs(np.asarray(actigraph, dtype=float))
        yy = yy[np.isfinite(yy)]
        if yy.size > 0:
            pct = tuple(getattr(config, "ACTIGRAPH_SEGMENT_YLIM_PERCENTILES", (1.0, 99.0)))
            lo, hi = np.nanpercentile(yy, [float(pct[0]), float(pct[1])])
            if np.isfinite(lo) and np.isfinite(hi):
                lo = 0.0
                if hi <= lo:
                    pad = 1.0 if hi == 0 else abs(float(hi)) * 0.1
                    hi += pad
                else:
                    pad = 0.05 * hi
                    hi += pad
                act_ylim = (float(lo), float(hi))

    for start in range(0, n_samples, samples_per_segment):
        end = min(start + samples_per_segment, n_samples)
        segment_index += 1
        t_seg_sec = np.arange(start, end) / sfreq
        seg_start_sec = float(t_seg_sec[0])
        seg_end_sec = float(t_seg_sec[-1])
        t_seg_h = t_seg_sec / 3600.0
        t_h_start = float(t_seg_h[0])
        t_h_end = float(t_seg_h[-1])

        enable_delta = include_event_vascular and features.segment_plot_requested("delta_hr")
        delta_mode = str(getattr(config, "DELTA_HR_PLOT_MODE", "subplot")).lower()
        has_any_delta = hr_calc_raw is not None and t_hr_calc is not None
        use_delta_subplot = enable_delta and (delta_mode == "subplot") and has_any_delta

        n_rows = (1 if use_hr else 0) + (1 if use_delta_subplot else 0) + (1 if use_prv else 0) + (1 if use_prv_sdnn else 0) + (1 if use_pat_amp else 0) + (1 if use_pwa_drop else 0) + (1 if use_spo2 else 0) + (1 if use_paper_harmonics else 0) + (1 if use_actigraph else 0)
        if n_rows == 0:
            continue
        height_ratios: List[float] = []
        if use_hr:
            height_ratios.append(1.0)
        if use_delta_subplot:
            height_ratios.append(1.0)
        if use_prv:
            height_ratios.append(1.0)
        if use_prv_sdnn:
            height_ratios.append(1.0)
        if use_pat_amp:
            height_ratios.append(1.0)
        if use_pwa_drop:
            height_ratios.append(1.0)
        if use_spo2:
            height_ratios.append(1.0)
        if use_paper_harmonics:
            height_ratios.append(0.85)
        if use_actigraph:
            height_ratios.append(0.5)
        fig, axes = plt.subplots(n_rows, 1, figsize=(11.69, 8.27), sharex=True, gridspec_kw={"height_ratios": height_ratios})
        if n_rows == 1:
            axes = [axes]
        fig._event_key_y = 0.975
        _add_colored_event_key(fig, list(event_spec))

        idx = 0
        ax_hr = axes[idx] if use_hr else None
        if use_hr:
            idx += 1
        ax_delta = axes[idx] if use_delta_subplot else None
        if use_delta_subplot:
            idx += 1
        ax_prv = axes[idx] if use_prv else None
        if use_prv:
            idx += 1
        ax_prv_sdnn = axes[idx] if use_prv_sdnn else None
        if use_prv_sdnn:
            idx += 1
        ax_pat_amp = axes[idx] if use_pat_amp else None
        if use_pat_amp:
            idx += 1
        ax_pwa_drop = axes[idx] if use_pwa_drop else None
        if use_pwa_drop:
            idx += 1
        ax_spo2 = axes[idx] if use_spo2 else None
        if use_spo2:
            idx += 1
        ax_paper_harmonics = axes[idx] if use_paper_harmonics else None
        if use_paper_harmonics:
            idx += 1
        ax_actigraph = axes[idx] if use_actigraph else None

        hr_ylim = None
        if ax_hr is not None:
            _plot_segment_hr(ax_hr, t_hr_edf=t_hr_edf, hr_edf=hr_edf, t_hr_calc=t_hr_calc, hr_calc=hr_calc, t_hr_edf_raw=t_hr_edf_raw, hr_edf_raw=hr_edf_raw, t_hr_calc_raw=t_hr_calc_raw, hr_calc_raw=hr_calc_raw, seg_start_sec=seg_start_sec, seg_end_sec=seg_end_sec, exclusion_zones=exclusion_zones, t_seg_h_start=t_h_start, t_seg_h_end=t_h_end, aux_df=aux_df, t_seg_sec=t_seg_sec)
            hr_ylim = ax_hr.get_ylim()

        delta_ylim = None
        if ax_delta is not None:
            _plot_segment_delta_hr(ax_delta, t_hr_edf=t_hr_edf, t_hr_calc=t_hr_calc, hr_calc_raw=hr_calc_raw, seg_start_sec=seg_start_sec, seg_end_sec=seg_end_sec, exclusion_zones=exclusion_zones, t_seg_h_start=t_h_start, t_seg_h_end=t_h_end, aux_df=aux_df)
            delta_ylim = ax_delta.get_ylim()

        prv_ylim = None
        if ax_prv is not None and t_prv is not None and prv_clean is not None:
            _plot_segment_prv(ax_prv, t_prv, prv_clean, prv_raw, seg_start_sec, seg_end_sec, exclusion_zones, t_h_start, t_h_end, aux_df=aux_df, ylabel="RMSSD [ms]", empty_text="PRV: no valid RMSSD windows in this segment", clean_label="PRV RMSSD (final-analysis)", raw_label="PRV RMSSD (pre-final exclusion)")
            prv_ylim = ax_prv.get_ylim()

        prv_sdnn_ylim = None
        if ax_prv_sdnn is not None and t_prv is not None and prv_sdnn_clean is not None:
            _plot_segment_prv(ax_prv_sdnn, t_prv, prv_sdnn_clean, prv_sdnn_raw, seg_start_sec, seg_end_sec, exclusion_zones, t_h_start, t_h_end, aux_df=aux_df, ylabel="SDNN [ms]", empty_text="PRV: no valid SDNN windows in this segment", clean_label="PRV SDNN (final-analysis)", raw_label="PRV SDNN (pre-final exclusion)")
            prv_sdnn_ylim = ax_prv_sdnn.get_ylim()

        amp_ylim = None
        if ax_pat_amp is not None and t_pat_amp is not None and pat_amp is not None and np.size(pat_amp) > 0:
            amp_ylim = _plot_segment_pat_amp(ax_pat_amp, t_pat_amp, pat_amp, seg_start_sec, seg_end_sec, exclusion_zones, t_h_start, t_h_end, aux_df=aux_df)
        elif ax_pat_amp is not None:
            ax_pat_amp.set_ylabel("DERIVED\nPAT_AMP")
            ax_pat_amp.text(0.5, 0.5, "DERIVED_PAT_AMP unavailable", transform=ax_pat_amp.transAxes, ha="center", va="center", fontsize=9)
            ax_pat_amp.grid(True)

        pwa_ylim = None
        if ax_pwa_drop is not None and t_pwa is not None and pwa_series is not None and np.size(pwa_series) > 0 and np.size(pwa_series) == np.size(t_pwa):
            pwa_ylim = _plot_segment_pwa_drop(ax_pwa_drop, t_pwa, pwa_series, pwa_drop_events_by_variant, seg_start_sec, seg_end_sec, exclusion_zones, t_h_start, t_h_end, aux_df=aux_df)
        elif ax_pwa_drop is not None:
            ax_pwa_drop.set_ylabel("PWA")
            ax_pwa_drop.text(0.5, 0.5, "VIEW_PAT-derived PWA unavailable", transform=ax_pwa_drop.transAxes, ha="center", va="center", fontsize=9)
            ax_pwa_drop.grid(True)
        spo2_ylim = None
        if ax_spo2 is not None and t_spo2 is not None and spo2 is not None:
            spo2_ylim = _plot_segment_spo2(ax_spo2, t_spo2, spo2, seg_start_sec, seg_end_sec, exclusion_zones, t_h_start, t_h_end, aux_df=aux_df)

        if ax_paper_harmonics is not None:
            _plot_segment_pat_paper_harmonics(ax_paper_harmonics, pat_paper_harmonics_windows, seg_start_sec, seg_end_sec)

        if ax_actigraph is not None and t_actigraph is not None and actigraph is not None:
            _plot_segment_actigraph(ax_actigraph, t_actigraph, actigraph, seg_start_sec, seg_end_sec, act_ylim=act_ylim)

        _overlay_events_on_axes(aux_df, seg_start_sec, seg_end_sec, ax_hr=ax_hr, ax_prv=ax_prv if use_prv else None, ax_prv_sdnn=ax_prv_sdnn if use_prv_sdnn else None, ax_amp=ax_pat_amp if use_pat_amp else None, ax_delta=ax_delta if use_delta_subplot else None, ax_pwa=ax_pwa_drop if use_pwa_drop else None, ax_spo2=ax_spo2 if use_spo2 else None, hr_ylim=hr_ylim, prv_ylim=prv_ylim, prv_sdnn_ylim=prv_sdnn_ylim, amp_ylim=amp_ylim, delta_ylim=delta_ylim, pwa_ylim=pwa_ylim, spo2_ylim=spo2_ylim, event_spec=event_spec)

        if ax_actigraph is not None:
            ax_actigraph.set_xlabel("Time (hours from recording start)")
        elif ax_paper_harmonics is not None:
            ax_paper_harmonics.set_xlabel("Time (hours from recording start)")
        elif ax_spo2 is not None:
            ax_spo2.set_xlabel("Time (hours from recording start)")
        elif ax_pwa_drop is not None:
            ax_pwa_drop.set_xlabel("Time (hours from recording start)")
        elif ax_pat_amp is not None:
            ax_pat_amp.set_xlabel("Time (hours from recording start)")
        elif ax_prv_sdnn is not None:
            ax_prv_sdnn.set_xlabel("Time (hours from recording start)")
        elif ax_prv is not None:
            ax_prv.set_xlabel("Time (hours from recording start)")
        elif ax_delta is not None:
            ax_delta.set_xlabel("Time (hours from recording start)")
        elif ax_hr is not None:
            ax_hr.set_xlabel("Time (hours from recording start)")

        for ax in axes:
            ax.xaxis.set_major_formatter(FuncFormatter(_format_hour_tick))

        title_parts: list[str] = []
        if include_event_vascular:
            title_parts.append("Event-Related Vascular Response")
        if include_prv:
            title_parts.append("PRV")
        if title_parts:
            fig.suptitle(f"{title_prefix} - {' + '.join(title_parts)} segment {segment_index} ({t_h_start:.2f}-{t_h_end:.2f} h)", fontsize=11, y=0.995)

        fig.tight_layout(rect=(0.04, 0.05, 0.88, 0.84))
        fig.subplots_adjust(hspace=0.22)
        pdf.savefig(fig)
        plt.close(fig)


def _add_event_vascular_segment_pages_to_pdf(pdf: PdfPages, **kwargs) -> None:
    _add_segment_pages_to_pdf(pdf, include_event_vascular=True, include_prv=False, **kwargs)


def _add_prv_segment_pages_to_pdf(pdf: PdfPages, **kwargs) -> None:
    _add_segment_pages_to_pdf(pdf, include_event_vascular=False, include_prv=True, **kwargs)


__all__ = ["_add_segment_pages_to_pdf", "_add_event_vascular_segment_pages_to_pdf", "_add_prv_segment_pages_to_pdf"]
