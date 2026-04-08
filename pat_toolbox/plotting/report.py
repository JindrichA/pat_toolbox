from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

import numpy as np

from .. import config
from .report_helpers import _build_report_context, _build_report_figures, _write_report_pdf

if TYPE_CHECKING:
    import pandas as pd


def plot_pat_and_hr_segments_to_pdf(
    signal_raw: np.ndarray,
    signal_filt: np.ndarray,
    sfreq: float,
    pdf_path: Path,
    segment_minutes: Optional[float] = None,
    title_prefix: str = "",
    channel_name: str = "",
    t_hr_calc: Optional[np.ndarray] = None,
    hr_calc: Optional[np.ndarray] = None,
    t_hr_edf: Optional[np.ndarray] = None,
    hr_edf: Optional[np.ndarray] = None,
    t_hrv: Optional[np.ndarray] = None,
    hrv_rmssd: Optional[np.ndarray] = None,
    hrv_rmssd_raw: Optional[np.ndarray] = None,
    hrv_tv: Optional[Dict[str, np.ndarray]] = None,
    pearson_r: Optional[float] = None,
    spear_rho: Optional[float] = None,
    rmse: Optional[float] = None,
    hrv_summary: Optional[Dict[str, float]] = None,
    aux_df: "Optional[pd.DataFrame]" = None,
    t_pat_amp: Optional[np.ndarray] = None,
    pat_amp: Optional[np.ndarray] = None,
    delta_hr_calc: Optional[np.ndarray] = None,
    delta_hr_edf: Optional[np.ndarray] = None,
    delta_hr_calc_evt: Optional[np.ndarray] = None,
    delta_hr_edf_evt: Optional[np.ndarray] = None,
    t_hr_calc_raw: Optional[np.ndarray] = None,
    hr_calc_raw: Optional[np.ndarray] = None,
    t_hr_edf_raw: Optional[np.ndarray] = None,
    hr_edf_raw: Optional[np.ndarray] = None,
    pat_burden: Optional[float] = None,
    pat_burden_diag: Optional[Dict[str, float]] = None,
    sleep_combo_summaries: Optional[Dict[str, Dict[str, object]]] = None,
    hrv_mask_info: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
    if segment_minutes is None:
        segment_minutes = config.SEGMENT_MINUTES

    n_samples = len(signal_raw)
    if n_samples == 0 or sfreq <= 0:
        raise ValueError("Signal is empty or sampling frequency invalid.")
    if len(signal_filt) != n_samples:
        raise ValueError("Raw and filtered signal lengths differ.")

    samples_per_segment = int(segment_minutes * 60.0 * sfreq)
    if samples_per_segment <= 0:
        raise ValueError("Computed non-positive samples_per_segment.")

    context = _build_report_context(signal_raw, sfreq, pdf_path, aux_df)
    duration_sec = n_samples / sfreq

    figures = _build_report_figures(
        edf_base=context["edf_base"],
        duration_sec=duration_sec,
        exclusion_zones=context["exclusion_zones"],
        event_spec=context["event_spec"],
        psd_features=context["psd_features"],
        mayer_peak_freq=context["mayer_peak_freq"],
        resp_peak_freq=context["resp_peak_freq"],
        t_hr_calc=t_hr_calc,
        hr_calc=hr_calc,
        t_hrv=t_hrv,
        hrv_rmssd=hrv_rmssd,
        hrv_rmssd_raw=hrv_rmssd_raw,
        hrv_tv=hrv_tv,
        hrv_summary=hrv_summary,
        aux_df=aux_df,
        delta_hr_calc=delta_hr_calc,
        delta_hr_calc_evt=delta_hr_calc_evt,
        pat_burden=pat_burden,
        pat_burden_diag=pat_burden_diag,
        sleep_combo_summaries=sleep_combo_summaries,
        hrv_mask_info=hrv_mask_info,
    )

    segment_kwargs = dict(
        signal_raw=signal_raw,
        signal_filt=signal_filt,
        sfreq=sfreq,
        segment_minutes=float(segment_minutes),
        title_prefix=title_prefix,
        channel_name=channel_name,
        t_hr_calc=t_hr_calc,
        hr_calc=hr_calc,
        t_hr_edf=None,
        hr_edf=None,
        t_hrv=t_hrv,
        hrv_clean=hrv_rmssd,
        hrv_raw=hrv_rmssd_raw,
        aux_df=aux_df,
        exclusion_zones=context["exclusion_zones"],
        event_spec=context["event_spec"],
        t_pat_amp=t_pat_amp,
        pat_amp=pat_amp,
        delta_hr_calc=delta_hr_calc,
        delta_hr_edf=None,
        delta_hr_calc_evt=delta_hr_calc_evt,
        delta_hr_edf_evt=None,
        t_hr_calc_raw=t_hr_calc_raw,
        hr_calc_raw=hr_calc_raw,
        t_hr_edf_raw=None,
        hr_edf_raw=None,
    )

    _write_report_pdf(
        pdf_path,
        fig_stage_tv=figures["fig_stage_tv"],
        fig_stage=figures["fig_stage"],
        fig_psd_zoom=context["fig_psd_zoom"],
        fig_psd_full=context["fig_psd_full"],
        fig_ov=figures["fig_ov"],
        summary_pages=figures["summary_pages"],
        segment_kwargs=segment_kwargs,
    )

    return context["psd_features"]
