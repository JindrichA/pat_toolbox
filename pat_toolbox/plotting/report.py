from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .. import config
from ..metrics.psd import compute_psd_figures_and_peaks

from .utils import _infer_edf_base, _compute_exclusion_zones
from .figures_summary import build_summary_pages, _build_sleep_stagegram_figure
from .figures_hrv import (
    _build_hrv_overview_figure,
    _build_stagegram_and_hrv_tv_figure,
)
from .segments import _add_segment_pages_to_pdf

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
) -> Dict[str, float]:
    """
    Plotting-only mode:
      - keep upstream API compatible
      - hide proprietary/device HR and its comparison from all plots
      - keep PAT-derived HR / HRV / PAT amplitude / PSD plotting active

    Report order:
      1) full-night pages
      2) summary tables
      3) segment pages
    """

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

    edf_base = _infer_edf_base(pdf_path)

    use_hrv = t_hrv is not None and hrv_rmssd is not None and np.size(hrv_rmssd) > 0
    exclusion_zones = _compute_exclusion_zones(aux_df)

    (
        psd_features,
        fig_psd_zoom,
        fig_psd_full,
        _psd_png_zoom,
        _psd_png_full,
    ) = compute_psd_figures_and_peaks(
        signal_raw,
        sfreq,
        edf_base=edf_base,
        aux_df=aux_df,
    )

    mayer_peak_freq = psd_features.get("mayer_peak_hz")
    resp_peak_freq = psd_features.get("resp_peak_hz")
    duration_sec = n_samples / sfreq

    # Hide proprietary/device HR from plots
    t_hr_edf_plot = None
    hr_edf_plot = None
    delta_hr_edf_plot = None
    delta_hr_edf_evt_plot = None
    t_hr_edf_raw_plot = None
    hr_edf_raw_plot = None

    pearson_r_plot = None
    spear_rho_plot = None
    rmse_plot = None

    summary_pages = build_summary_pages(
        edf_base=edf_base,
        pearson_r=pearson_r_plot,
        spear_rho=spear_rho_plot,
        rmse=rmse_plot,
        hrv_summary=hrv_summary,
        mayer_peak_freq=mayer_peak_freq,
        resp_peak_freq=resp_peak_freq,
        aux_df=aux_df,
        t_hr_calc=t_hr_calc,
        hr_calc=hr_calc,
        t_hr_edf=t_hr_edf_plot,
        hr_edf=hr_edf_plot,
        t_hrv=t_hrv,
        hrv_clean=hrv_rmssd,
        hrv_raw=hrv_rmssd_raw,
        hrv_tv=hrv_tv,
        psd_features=psd_features,
        delta_hr_calc=delta_hr_calc,
        delta_hr_edf=delta_hr_edf_plot,
        delta_hr_calc_evt=delta_hr_calc_evt,
        delta_hr_edf_evt=delta_hr_edf_evt_plot,
        pat_burden=pat_burden,
        pat_burden_diag=pat_burden_diag,
    )

    has_tv = (
        use_hrv
        and t_hrv is not None
        and hrv_tv is not None
        and isinstance(hrv_tv, dict)
        and len(hrv_tv) > 0
    )

    fig_stage = None
    fig_stage_tv = None

    if has_tv:
        fig_stage_tv = _build_stagegram_and_hrv_tv_figure(
            edf_base=edf_base,
            aux_df=aux_df,
            t_hrv=t_hrv,
            hrv_rmssd=hrv_rmssd,
            hrv_tv=hrv_tv,
            exclusion_zones=exclusion_zones,
        )
    else:
        fig_stage = _build_sleep_stagegram_figure(
            edf_base=edf_base,
            aux_df=aux_df,
        )

    fig_ov = None
    if use_hrv and t_hrv is not None and hrv_rmssd is not None:
        fig_ov = _build_hrv_overview_figure(
            edf_base=edf_base,
            t_hrv=t_hrv,
            hrv_clean=hrv_rmssd,
            hrv_raw=hrv_rmssd_raw,
            aux_df=aux_df,
            exclusion_zones=exclusion_zones,
            duration_sec_fallback=duration_sec,
        )

    try:
        with PdfPages(str(pdf_path)) as pdf:
            # ---------------------------------------------------------
            # 1) FULL-NIGHT PAGES FIRST
            # ---------------------------------------------------------
            if fig_stage_tv is not None:
                pdf.savefig(fig_stage_tv)
                plt.close(fig_stage_tv)
            elif fig_stage is not None:
                pdf.savefig(fig_stage)
                plt.close(fig_stage)

            pdf.savefig(fig_psd_zoom)
            plt.close(fig_psd_zoom)

            pdf.savefig(fig_psd_full)
            plt.close(fig_psd_full)

            if fig_ov is not None:
                pdf.savefig(fig_ov)
                plt.close(fig_ov)

            # ---------------------------------------------------------
            # 2) SUMMARY TABLES
            # ---------------------------------------------------------
            for fig in summary_pages:
                pdf.savefig(fig)
                plt.close(fig)

            # ---------------------------------------------------------
            # 3) SEGMENT PAGES LAST
            # ---------------------------------------------------------
            _add_segment_pages_to_pdf(
                pdf,
                signal_raw=signal_raw,
                signal_filt=signal_filt,
                sfreq=sfreq,
                segment_minutes=float(segment_minutes),
                title_prefix=title_prefix,
                channel_name=channel_name,
                t_hr_calc=t_hr_calc,
                hr_calc=hr_calc,
                t_hr_edf=t_hr_edf_plot,
                hr_edf=hr_edf_plot,
                t_hrv=t_hrv,
                hrv_clean=hrv_rmssd,
                hrv_raw=hrv_rmssd_raw,
                aux_df=aux_df,
                exclusion_zones=exclusion_zones,
                t_pat_amp=t_pat_amp,
                pat_amp=pat_amp,
                delta_hr_calc=delta_hr_calc,
                delta_hr_edf=delta_hr_edf_plot,
                delta_hr_calc_evt=delta_hr_calc_evt,
                delta_hr_edf_evt=delta_hr_edf_evt_plot,
                t_hr_calc_raw=t_hr_calc_raw,
                hr_calc_raw=hr_calc_raw,
                t_hr_edf_raw=t_hr_edf_raw_plot,
                hr_edf_raw=hr_edf_raw_plot,
            )

    except Exception:
        if pdf_path.exists():
            try:
                pdf_path.unlink()
            except Exception:
                pass
        raise

    return psd_features