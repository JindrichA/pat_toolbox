from __future__ import annotations

from typing import Any, cast

from . import config, features, paths, plotting
from .context import RecordingContext
from .io_aux_csv import save_sleep_timing_to_csv
from .metrics import hr as hr_metrics
from .metrics import hrv as hrv_metrics
from .metrics.pat_burden_io import save_pat_burden_episodes_to_csv


def build_pdf_step(ctx: RecordingContext) -> None:
    if not features.is_enabled("report_pdf"):
        ctx.pdf_path = None
        return
    assert ctx.view_pat is not None and ctx.view_pat_filt is not None and ctx.sfreq is not None
    out_folder = paths.get_output_folder()
    suffix = "_multi_sleep_summary" if getattr(ctx, "sleep_combo_summaries", None) else (config.sleep_stage_suffix() if getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False) else "")
    feature_parts = ["VIEW_PAT", *features.enabled_feature_parts(("hr", "hrv", "psd", "delta_hr", "pat_burden"))]
    pdf_name = f"{ctx.edf_base}__{'_'.join(feature_parts)}_{config.SEGMENT_MINUTES}min_overlay{suffix}.pdf"
    ctx.pdf_path = out_folder / pdf_name
    psd_results_dict = plotting.plot_pat_and_hr_segments_to_pdf(
        signal_raw=ctx.view_pat,
        signal_filt=ctx.view_pat_filt,
        sfreq=ctx.sfreq,
        pdf_path=ctx.pdf_path,
        segment_minutes=config.SEGMENT_MINUTES,
        title_prefix=ctx.edf_base,
        channel_name=config.VIEW_PAT_CHANNEL_NAME,
        t_hr_calc=ctx.t_hr_calc,
        hr_calc=ctx.hr_calc,
        t_hr_edf=None,
        hr_edf=None,
        t_hr_calc_raw=ctx.t_hr_calc,
        hr_calc_raw=getattr(ctx, "hr_calc_raw", None),
        t_hr_edf_raw=None,
        hr_edf_raw=None,
        t_hrv=ctx.t_hrv,
        hrv_rmssd=ctx.hrv_rmssd_clean,
        hrv_rmssd_raw=ctx.hrv_rmssd_raw,
        hrv_tv=ctx.hrv_tv,
        pearson_r=None,
        spear_rho=None,
        rmse=None,
        hrv_summary=ctx.hrv_summary,
        aux_df=ctx.aux_df,
        t_pat_amp=ctx.t_pat_amp,
        pat_amp=ctx.pat_amp,
        pat_burden=getattr(ctx, "pat_burden", None),
        pat_burden_diag=getattr(ctx, "pat_burden_diag", None),
        sleep_combo_summaries=getattr(ctx, "sleep_combo_summaries", None),
        hrv_mask_info=getattr(ctx, "hrv_mask_info", None),
        hrv_midpoint_halves=getattr(ctx, "hrv_midpoint_halves", None),
    )
    if features.is_enabled("psd"):
        ctx.psd_features = psd_results_dict
        ctx.mayer_peak_freq = psd_results_dict.get("mayer_peak_hz")
        ctx.resp_peak_freq = psd_results_dict.get("resp_peak_hz")
    print(f"  Saved feature report to: {ctx.pdf_path}")


def export_feature_csvs_step(ctx: RecordingContext) -> None:
    if features.is_enabled("hr") and ctx.t_hr_calc is not None and ctx.hr_calc is not None:
        try:
            ctx.hr_csv_path = hr_metrics.save_hr_series_to_csv(ctx.edf_path, ctx.t_hr_calc, ctx.hr_calc)
        except Exception as e:
            print(f"  WARNING: could not save HR CSV for {ctx.edf_path.name}: {e}")

    if features.is_enabled("hrv") and ctx.t_hrv is not None and ctx.hrv_rmssd_clean is not None:
        try:
            ctx.hrv_csv_path = hrv_metrics.save_hrv_bundle_to_csv(ctx.edf_path, ctx.t_hrv, ctx.hrv_rmssd_clean, rmssd_raw=ctx.hrv_rmssd_raw, hrv_tv=ctx.hrv_tv)
        except Exception as e:
            print(f"  WARNING: could not save HRV CSV for {ctx.edf_path.name}: {e}")
        try:
            if ctx.hrv_mask_info is not None:
                ctx.hrv_mask_csv_path = hrv_metrics.save_hrv_mask_to_csv(ctx.edf_path, ctx.t_hrv, ctx.hrv_mask_info)
        except Exception as e:
            print(f"  WARNING: could not save HRV mask CSV for {ctx.edf_path.name}: {e}")

    if features.is_enabled("pat_burden") and getattr(ctx, "pat_burden_episodes", None):
        try:
            episodes = ctx.pat_burden_episodes
            if episodes is not None:
                ctx.pat_burden_csv_path = save_pat_burden_episodes_to_csv(ctx.edf_path, cast(list[dict[str, Any]], episodes))
        except Exception as e:
            print(f"  WARNING: could not save PAT burden CSV for {ctx.edf_path.name}: {e}")

    if ctx.aux_df is not None:
        try:
            ctx.sleep_timing_csv_path = save_sleep_timing_to_csv(ctx.edf_path, ctx.aux_df)
        except Exception as e:
            print(f"  WARNING: could not save sleep timing CSV for {ctx.edf_path.name}: {e}")


def build_peaks_debug_pdf_step(ctx: RecordingContext) -> None:
    if not features.is_enabled("peaks_debug_pdf"):
        return
    try:
        pdf_path = hr_metrics.create_peaks_debug_pdf_for_edf(ctx.edf_path)
        try:
            ctx.peaks_pdf_path = pdf_path
        except Exception:
            pass
    except Exception as e:
        print(f"  WARNING: could not create peaks debug PDF for {ctx.edf_path.name}: {e}")


def append_summary_step(ctx: RecordingContext) -> None:
    if not features.summary_requested():
        return
    hr_metrics.append_hr_hrv_summary(
        ctx.edf_path,
        ctx.hrv_summary,
        ctx.mayer_peak_freq,
        ctx.resp_peak_freq,
        t_hr=ctx.t_hr_calc,
        hr_calc=ctx.hr_calc,
        t_hrv=ctx.t_hrv,
        hrv_clean=ctx.hrv_rmssd_clean,
        hrv_raw=ctx.hrv_rmssd_raw,
        hrv_tv=ctx.hrv_tv,
        hrv_mask_info=ctx.hrv_mask_info,
        hrv_midpoint_halves=getattr(ctx, "hrv_midpoint_halves", None),
        aux_df=ctx.aux_df,
        psd_features=getattr(ctx, "psd_features", None),
        pat_burden=getattr(ctx, "pat_burden", None),
        pat_burden_diag=getattr(ctx, "pat_burden_diag", None),
        sleep_combo_summaries=getattr(ctx, "sleep_combo_summaries", None),
    )
