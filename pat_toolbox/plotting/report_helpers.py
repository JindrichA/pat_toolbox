from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .. import features
from ..metrics.psd import compute_psd_figures_and_peaks
from .feature_overview_builders import (
    _build_event_response_overview_figure,
    _build_hr_overview_figure,
    _build_hrv_rmssd_overview_figure,
    _build_multi_series_overview_figure,
    _build_pat_burden_overview_figure,
    _build_single_series_overview_figure,
)
from .figures_hrv import _build_stagegram_and_hrv_tv_figure
from .figures_summary import _build_sleep_stagegram_figure, build_summary_pages
from .segments import _add_segment_pages_to_pdf
from .specs import active_event_plot_spec
from .utils import _compute_exclusion_zones, _infer_edf_base

if TYPE_CHECKING:
    import pandas as pd


def _close_figure(fig) -> None:
    if fig is None:
        return
    try:
        plt.close(fig)
    except Exception:
        pass


def _build_report_context(
    signal_raw: np.ndarray,
    sfreq: float,
    pdf_path: Path,
    aux_df: Optional["pd.DataFrame"],
):
    edf_base = _infer_edf_base(pdf_path)
    exclusion_zones = _compute_exclusion_zones(aux_df)
    event_spec = active_event_plot_spec()
    psd_features = None
    fig_psd_zoom = None
    fig_psd_full = None
    if features.is_enabled("psd"):
        psd_features, fig_psd_zoom, fig_psd_full, _psd_png_zoom, _psd_png_full = compute_psd_figures_and_peaks(
            signal_raw,
            sfreq,
            edf_base=edf_base,
            aux_df=aux_df,
        )
    return {
        "edf_base": edf_base,
        "exclusion_zones": exclusion_zones,
        "event_spec": event_spec,
        "psd_features": psd_features,
        "fig_psd_zoom": fig_psd_zoom,
        "fig_psd_full": fig_psd_full,
        "mayer_peak_freq": None if psd_features is None else psd_features.get("mayer_peak_hz"),
        "resp_peak_freq": None if psd_features is None else psd_features.get("resp_peak_hz"),
    }


def _build_summary_pages_for_enabled_features(
    *,
    edf_base: str,
    mayer_peak_freq,
    resp_peak_freq,
    aux_df,
    t_hr_calc,
    hr_calc,
    t_hrv,
    hrv_rmssd,
    hrv_rmssd_raw,
    hrv_tv,
    hrv_summary,
    psd_features,
    pat_burden,
    pat_burden_diag,
    sleep_combo_summaries,
    hrv_mask_info,
    hrv_midpoint_halves,
):
    return build_summary_pages(
        edf_base=edf_base,
        pearson_r=None,
        spear_rho=None,
        rmse=None,
        hrv_summary=hrv_summary,
        mayer_peak_freq=mayer_peak_freq,
        resp_peak_freq=resp_peak_freq,
        aux_df=aux_df,
        t_hr_calc=t_hr_calc,
        hr_calc=hr_calc,
        t_hr_edf=None,
        hr_edf=None,
        t_hrv=t_hrv,
        hrv_clean=hrv_rmssd,
        hrv_raw=hrv_rmssd_raw,
        hrv_tv=hrv_tv,
        psd_features=psd_features,
        pat_burden=pat_burden,
        pat_burden_diag=pat_burden_diag,
        sleep_combo_summaries=sleep_combo_summaries,
        hrv_mask_info=hrv_mask_info,
        hrv_midpoint_halves=hrv_midpoint_halves,
    )


def _build_hrv_report_figures(
    *,
    edf_base: str,
    duration_sec: float,
    exclusion_zones,
    event_spec,
    t_hrv,
    hrv_rmssd,
    hrv_rmssd_raw,
    hrv_tv,
    aux_df,
    sleep_combo_summaries,
    hrv_mask_info,
) -> Dict[str, Any]:
    use_hrv = features.is_enabled("hrv") and t_hrv is not None and hrv_rmssd is not None and np.size(hrv_rmssd) > 0
    has_tv = use_hrv and t_hrv is not None and hrv_tv is not None and isinstance(hrv_tv, dict) and len(hrv_tv) > 0

    fig_stage = None
    fig_stage_tv = None

    if has_tv:
        fig_stage_tv = _build_stagegram_and_hrv_tv_figure(
            edf_base=edf_base,
            aux_df=aux_df,
            t_hrv=cast(np.ndarray, t_hrv),
            hrv_rmssd=hrv_rmssd,
            hrv_tv=cast(Dict[str, np.ndarray], hrv_tv),
            exclusion_zones=exclusion_zones,
            sleep_combo_summaries=sleep_combo_summaries,
            event_spec=event_spec,
            hrv_mask_info=hrv_mask_info,
        )
    else:
        fig_stage = _build_sleep_stagegram_figure(edf_base=edf_base, aux_df=aux_df)

    return {
        "fig_stage": fig_stage,
        "fig_stage_tv": fig_stage_tv,
        "fig_ov": None,
    }


def _build_non_hrv_report_figures(*, edf_base: str, aux_df) -> Dict[str, Any]:
    return {
        "fig_stage": _build_sleep_stagegram_figure(edf_base=edf_base, aux_df=aux_df),
        "fig_stage_tv": None,
        "fig_ov": None,
    }


def _build_feature_overview_figures(
    *,
    edf_base: str,
    duration_sec: float,
    exclusion_zones,
    event_spec,
    aux_df,
    t_hr_calc,
    hr_calc,
    hr_calc_raw,
    t_hrv,
    hrv_rmssd,
    hrv_rmssd_raw,
    hrv_tv,
    hrv_mask_info,
    t_pat_amp,
    pat_amp,
) -> list[Any]:
    figs: list[Any] = []

    if features.is_enabled("hr"):
        fig = _build_hr_overview_figure(
            edf_base=edf_base,
            t_hr=t_hr_calc,
            hr_clean=hr_calc,
            hr_raw=hr_calc_raw,
            aux_df=aux_df,
            exclusion_zones=exclusion_zones,
            duration_sec_fallback=duration_sec,
            event_spec=event_spec,
        )
        if fig is not None:
            figs.append(fig)

    if features.is_enabled("hrv"):
        fig = _build_hrv_rmssd_overview_figure(
            edf_base=edf_base,
            t_hrv=t_hrv,
            hrv_clean=hrv_rmssd,
            hrv_raw=hrv_rmssd_raw,
            aux_df=aux_df,
            exclusion_zones=exclusion_zones,
            duration_sec_fallback=duration_sec,
            event_spec=event_spec,
            hrv_mask_info=hrv_mask_info,
        )
        if fig is not None:
            figs.append(fig)

        if isinstance(hrv_tv, dict):
            t_spectral = hrv_tv.get("spectral_t_sec")
            fig = _build_single_series_overview_figure(
                edf_base=edf_base,
                title="HRV-SDNN Overview",
                ylabel="SDNN [ms]",
                t_sec=t_hrv,
                y=hrv_tv.get("sdnn_ms"),
                y_raw=hrv_tv.get("sdnn_ms_raw"),
                color="tab:green",
                aux_df=aux_df,
                exclusion_zones=exclusion_zones,
                duration_sec_fallback=duration_sec,
                event_spec=event_spec,
                hrv_mask_info=hrv_mask_info,
            )
            if fig is not None:
                figs.append(fig)

            lf_hf_series = []
            if t_spectral is not None and hrv_tv.get("lf_fixed") is not None:
                lf_hf_series.append({"label": "LF", "y": hrv_tv.get("lf_fixed"), "y_raw": hrv_tv.get("lf_fixed_raw"), "color": "tab:orange"})
            if t_spectral is not None and hrv_tv.get("hf_fixed") is not None:
                lf_hf_series.append({"label": "HF", "y": hrv_tv.get("hf_fixed"), "y_raw": hrv_tv.get("hf_fixed_raw"), "color": "tab:blue"})
            fig = _build_multi_series_overview_figure(
                edf_base=edf_base,
                title="HRV-LF-HF Overview",
                ylabel="LF & HF [ms²]",
                t_sec=t_spectral,
                series=lf_hf_series,
                aux_df=aux_df,
                exclusion_zones=exclusion_zones,
                duration_sec_fallback=duration_sec,
                event_spec=event_spec,
                yscale="log",
                hrv_mask_info=hrv_mask_info,
            )
            if fig is not None:
                figs.append(fig)

            fig = _build_single_series_overview_figure(
                edf_base=edf_base,
                title="HRV-LF-HF Ratio Overview",
                ylabel="LF/HF [-]",
                t_sec=t_spectral,
                y=hrv_tv.get("lf_hf_fixed"),
                y_raw=hrv_tv.get("lf_hf_fixed_raw"),
                color="tab:purple",
                aux_df=aux_df,
                exclusion_zones=exclusion_zones,
                duration_sec_fallback=duration_sec,
                event_spec=event_spec,
                hrv_mask_info=hrv_mask_info,
            )
            if fig is not None:
                figs.append(fig)

    if features.is_enabled("delta_hr"):
        fig = _build_event_response_overview_figure(
            edf_base=edf_base,
            t_hr=t_hr_calc,
            hr_raw=hr_calc_raw,
            aux_df=aux_df,
            exclusion_zones=exclusion_zones,
            duration_sec_fallback=duration_sec,
            event_spec=event_spec,
        )
        if fig is not None:
            figs.append(fig)

    if features.is_enabled("pat_burden"):
        fig = _build_pat_burden_overview_figure(
            edf_base=edf_base,
            t_pat_amp=t_pat_amp,
            pat_amp=pat_amp,
            aux_df=aux_df,
            exclusion_zones=exclusion_zones,
            duration_sec_fallback=duration_sec,
            event_spec=event_spec,
        )
        if fig is not None:
            figs.append(fig)

    return figs


def _build_report_figures(
    *,
    edf_base: str,
    duration_sec: float,
    exclusion_zones,
    event_spec,
    psd_features,
    mayer_peak_freq,
    resp_peak_freq,
    t_hr_calc,
    hr_calc,
    t_hrv,
    hrv_rmssd,
    hrv_rmssd_raw,
    hrv_tv,
    hrv_summary,
    aux_df,
    pat_burden,
    pat_burden_diag,
    sleep_combo_summaries,
    hrv_mask_info,
    hrv_midpoint_halves,
    hr_calc_raw,
    t_pat_amp,
    pat_amp,
):
    summary_pages = _build_summary_pages_for_enabled_features(
        edf_base=edf_base,
        mayer_peak_freq=mayer_peak_freq,
        resp_peak_freq=resp_peak_freq,
        aux_df=aux_df,
        t_hr_calc=t_hr_calc,
        hr_calc=hr_calc,
        t_hrv=t_hrv,
        hrv_rmssd=hrv_rmssd,
        hrv_rmssd_raw=hrv_rmssd_raw,
        hrv_tv=hrv_tv,
        hrv_summary=hrv_summary,
        psd_features=psd_features,
        pat_burden=pat_burden,
        pat_burden_diag=pat_burden_diag,
        sleep_combo_summaries=sleep_combo_summaries,
        hrv_mask_info=hrv_mask_info,
        hrv_midpoint_halves=hrv_midpoint_halves,
    )

    if features.is_enabled("hrv"):
        figure_bundle = _build_hrv_report_figures(
            edf_base=edf_base,
            duration_sec=duration_sec,
            exclusion_zones=exclusion_zones,
            event_spec=event_spec,
            t_hrv=t_hrv,
            hrv_rmssd=hrv_rmssd,
            hrv_rmssd_raw=hrv_rmssd_raw,
            hrv_tv=hrv_tv,
            aux_df=aux_df,
            sleep_combo_summaries=sleep_combo_summaries,
            hrv_mask_info=hrv_mask_info,
        )
    else:
        figure_bundle = _build_non_hrv_report_figures(edf_base=edf_base, aux_df=aux_df)

    overview_figures = _build_feature_overview_figures(
        edf_base=edf_base,
        duration_sec=duration_sec,
        exclusion_zones=exclusion_zones,
        event_spec=event_spec,
        aux_df=aux_df,
        t_hr_calc=t_hr_calc,
        hr_calc=hr_calc,
        hr_calc_raw=hr_calc_raw,
        t_hrv=t_hrv,
        hrv_rmssd=hrv_rmssd,
        hrv_rmssd_raw=hrv_rmssd_raw,
        hrv_tv=hrv_tv,
        hrv_mask_info=hrv_mask_info,
        t_pat_amp=t_pat_amp,
        pat_amp=pat_amp,
    )

    return {
        "summary_pages": summary_pages,
        "overview_figures": overview_figures,
        **figure_bundle,
    }


def _write_report_pdf(
    pdf_path: Path,
    *,
    fig_stage_tv,
    fig_stage,
    fig_psd_zoom,
    fig_psd_full,
    fig_ov,
    overview_figures: List,
    summary_pages: List,
    segment_kwargs: Mapping[str, Any],
) -> None:
    try:
        with PdfPages(str(pdf_path)) as pdf:
            if fig_stage_tv is not None:
                pdf.savefig(fig_stage_tv)
                _close_figure(fig_stage_tv)
                fig_stage_tv = None
            elif fig_stage is not None:
                pdf.savefig(fig_stage)
                _close_figure(fig_stage)
                fig_stage = None

            if fig_psd_zoom is not None:
                pdf.savefig(fig_psd_zoom)
                _close_figure(fig_psd_zoom)
                fig_psd_zoom = None
            if fig_psd_full is not None:
                pdf.savefig(fig_psd_full)
                _close_figure(fig_psd_full)
                fig_psd_full = None

            if fig_ov is not None:
                pdf.savefig(fig_ov)
                _close_figure(fig_ov)
                fig_ov = None

            for fig in overview_figures:
                pdf.savefig(fig)
                _close_figure(fig)
            overview_figures = []

            for fig in summary_pages:
                pdf.savefig(fig)
                _close_figure(fig)
            summary_pages = []

            _add_segment_pages_to_pdf(pdf, **cast(dict[str, Any], segment_kwargs))
    except Exception:
        if pdf_path.exists():
            try:
                pdf_path.unlink()
            except Exception:
                pass
        raise
    finally:
        _close_figure(fig_stage_tv)
        _close_figure(fig_stage)
        _close_figure(fig_ov)
        _close_figure(fig_psd_zoom)
        _close_figure(fig_psd_full)
        for fig in overview_figures:
            _close_figure(fig)
        for fig in summary_pages:
            _close_figure(fig)
