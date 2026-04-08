from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .. import config
from .segment_plot_helpers import (
    _overlay_events_on_axes,
    _plot_segment_delta_hr,
    _plot_segment_hr,
    _plot_segment_hrv,
)
from .specs import EventSpec

if TYPE_CHECKING:
    import pandas as pd


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
    event_spec: List[EventSpec],
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
    n_samples = len(signal_raw)
    samples_per_segment = int(segment_minutes * 60.0 * sfreq)
    segment_index = 0
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
        has_any_delta = ((delta_hr_calc is not None and np.size(delta_hr_calc) > 0) or (delta_hr_calc_evt is not None and np.size(delta_hr_calc_evt) > 0))
        use_delta_subplot = enable_delta and (delta_mode == "subplot") and has_any_delta

        n_rows = 1 + (1 if use_delta_subplot else 0) + (1 if use_hrv else 0)
        height_ratios: List[float] = [1] + ([1] if use_delta_subplot else []) + ([1] if use_hrv else [])
        fig, axes = plt.subplots(n_rows, 1, figsize=(11.69, 8.27), sharex=True, gridspec_kw={"height_ratios": height_ratios})
        if n_rows == 1:
            axes = [axes]

        idx = 0
        ax_hr = axes[idx]
        idx += 1
        ax_delta = axes[idx] if use_delta_subplot else None
        if use_delta_subplot:
            idx += 1
        ax_hrv = axes[idx] if use_hrv else None

        _plot_segment_hr(ax_hr, t_hr_edf=t_hr_edf, hr_edf=hr_edf, t_hr_calc=t_hr_calc, hr_calc=hr_calc, t_hr_edf_raw=t_hr_edf_raw, hr_edf_raw=hr_edf_raw, t_hr_calc_raw=t_hr_calc_raw, hr_calc_raw=hr_calc_raw, seg_start_sec=seg_start_sec, seg_end_sec=seg_end_sec, exclusion_zones=exclusion_zones, t_seg_h_start=t_h_start, t_seg_h_end=t_h_end, aux_df=aux_df, t_seg_sec=t_seg_sec)
        hr_ylim = ax_hr.get_ylim()

        delta_ylim = None
        if ax_delta is not None:
            _plot_segment_delta_hr(ax_delta, t_hr_edf=t_hr_edf, delta_hr_edf=delta_hr_edf, t_hr_calc=t_hr_calc, delta_hr_calc=delta_hr_calc, delta_hr_edf_evt=delta_hr_edf_evt, delta_hr_calc_evt=delta_hr_calc_evt, seg_start_sec=seg_start_sec, seg_end_sec=seg_end_sec, exclusion_zones=exclusion_zones, t_seg_h_start=t_h_start, t_seg_h_end=t_h_end, aux_df=aux_df)
            delta_ylim = ax_delta.get_ylim()

        hrv_ylim = None
        if ax_hrv is not None and t_hrv is not None and hrv_clean is not None:
            _plot_segment_hrv(ax_hrv, t_hrv, hrv_clean, hrv_raw, seg_start_sec, seg_end_sec, exclusion_zones, t_h_start, t_h_end, aux_df=aux_df)
            hrv_ylim = ax_hrv.get_ylim()

        _overlay_events_on_axes(aux_df, seg_start_sec, seg_end_sec, ax_hr=ax_hr, ax_hrv=ax_hrv if use_hrv else None, ax_amp=None, ax_delta=ax_delta if use_delta_subplot else None, hr_ylim=hr_ylim, hrv_ylim=hrv_ylim, amp_ylim=None, delta_ylim=delta_ylim, event_spec=event_spec)

        if ax_hrv is not None:
            ax_hrv.set_xlabel("Time (hours from recording start)")
        elif ax_delta is not None:
            ax_delta.set_xlabel("Time (hours from recording start)")
        else:
            ax_hr.set_xlabel("Time (hours from recording start)")

        fig.tight_layout(rect=(0.04, 0.05, 0.88, 0.98))
        fig.subplots_adjust(hspace=0.22)
        pdf.savefig(fig)
        plt.close(fig)


__all__ = ["_add_segment_pages_to_pdf"]
