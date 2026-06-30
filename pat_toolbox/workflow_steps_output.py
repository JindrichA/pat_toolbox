from __future__ import annotations

from typing import Any, cast

from . import config, features, paths, plotting
from .context import RecordingContext
from .io_aux_csv import save_sleep_timing_to_csv
from .metrics import hr as hr_metrics
from .metrics import prv as prv_metrics
from .metrics.hr_event_response import save_event_hr_windows_to_csv
from .metrics.pat_burden_io import save_pat_burden_episodes_to_csv, save_pat_burden_summary_to_csv
from .metrics.pwa_drop_io import save_pwa_drop_events_to_csv, save_pwa_drop_summary_to_csv
from .metrics.pat_harmonics_io import save_pat_harmonics_windows_to_csv, save_pat_harmonics_summary_to_csv
from .metrics.pat_paper_harmonics_io import save_pat_paper_harmonics_windows_to_csv, save_pat_paper_harmonics_summary_to_csv


def build_pdf_step(ctx: RecordingContext) -> None:
    if not features.is_enabled("report_pdf"):
        ctx.pdf_path = None
        return
    assert ctx.view_pat is not None and ctx.view_pat_filt is not None and ctx.sfreq is not None
    out_folder = paths.get_output_folder()
    suffix = "_multi_sleep_summary" if getattr(ctx, "sleep_combo_summaries", None) else (config.sleep_stage_suffix() if getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False) else "")
    feature_parts = ["VIEW_PAT", *features.enabled_feature_parts(("hr", "prv", "psd", "delta_hr", "pat_burden", "pwa_drop", "pat_harmonics", "pat_paper_harmonics"))]
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
        t_prv=ctx.t_prv,
        prv_rmssd=ctx.prv_rmssd_clean,
        prv_rmssd_raw=ctx.prv_rmssd_raw,
        prv_tv=ctx.prv_tv,
        pearson_r=None,
        spear_rho=None,
        rmse=None,
        prv_summary=ctx.prv_summary,
        aux_df=ctx.aux_df,
        hr_event_response_summary=getattr(ctx, "hr_event_response_summary", None),
        hr_event_windows=getattr(ctx, "hr_event_windows", None),
        t_pat_amp=ctx.t_pat_amp,
        pat_amp=ctx.pat_amp,
        pat_burden=getattr(ctx, "pat_burden", None),
        pat_burden_diag=getattr(ctx, "pat_burden_diag", None),
        pat_burden_episodes=getattr(ctx, "pat_burden_episodes", None),
        t_pwa=getattr(ctx, "t_pwa", None),
        pwa_series=getattr(ctx, "pwa_series", None),
        pwa_drop_summaries=getattr(ctx, "pwa_drop_summaries", None),
        pwa_drop_events_by_variant=getattr(ctx, "pwa_drop_events_by_variant", None),
        pat_harmonics_summary=getattr(ctx, "pat_harmonics_summary", None),
        pat_harmonics_windows=getattr(ctx, "pat_harmonics_windows", None),
        pat_paper_harmonics_summary=getattr(ctx, "pat_paper_harmonics_summary", None),
        pat_paper_harmonics_windows=getattr(ctx, "pat_paper_harmonics_windows", None),
        t_spo2=getattr(ctx, "t_spo2", None),
        spo2=getattr(ctx, "spo2", None),
        t_actigraph=getattr(ctx, "t_actigraph", None),
        actigraph=getattr(ctx, "actigraph", None),
        sleep_combo_summaries=getattr(ctx, "sleep_combo_summaries", None),
        prv_mask_info=getattr(ctx, "prv_mask_info", None),
        prv_midpoint_halves=getattr(ctx, "prv_midpoint_halves", None),
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

    if features.is_enabled("delta_hr") and getattr(ctx, "hr_event_windows", None):
        try:
            windows = ctx.hr_event_windows
            if windows is not None:
                ctx.hr_event_csv_path = save_event_hr_windows_to_csv(ctx.edf_path, cast(list[dict[str, Any]], windows))
        except Exception as e:
            print(f"  WARNING: could not save delta HR CSV for {ctx.edf_path.name}: {e}")

    if features.is_enabled("prv") and ctx.t_prv is not None and ctx.prv_rmssd_clean is not None:
        try:
            ctx.prv_csv_path = prv_metrics.save_prv_bundle_to_csv(ctx.edf_path, ctx.t_prv, ctx.prv_rmssd_clean, rmssd_raw=ctx.prv_rmssd_raw, prv_tv=ctx.prv_tv)
        except Exception as e:
            print(f"  WARNING: could not save PRV CSV for {ctx.edf_path.name}: {e}")
        try:
            if ctx.prv_mask_info is not None:
                ctx.prv_mask_csv_path = prv_metrics.save_prv_mask_to_csv(ctx.edf_path, ctx.t_prv, ctx.prv_mask_info)
        except Exception as e:
            print(f"  WARNING: could not save PRV mask CSV for {ctx.edf_path.name}: {e}")

    if features.is_enabled("pat_burden") and getattr(ctx, "pat_burden_episodes", None):
        try:
            episodes = ctx.pat_burden_episodes
            if episodes is not None:
                ctx.pat_burden_csv_path = save_pat_burden_episodes_to_csv(ctx.edf_path, cast(list[dict[str, Any]], episodes))
        except Exception as e:
            print(f"  WARNING: could not save PAT burden CSV for {ctx.edf_path.name}: {e}")
    if features.is_enabled("pat_burden") and (getattr(ctx, "pat_burden_diag", None) is not None or getattr(ctx, "pat_burden", None) is not None):
        try:
            ctx.pat_burden_summary_csv_path = save_pat_burden_summary_to_csv(ctx.edf_path, ctx.pat_burden, getattr(ctx, "pat_burden_diag", None))
        except Exception as e:
            print(f"  WARNING: could not save PAT burden summary CSV for {ctx.edf_path.name}: {e}")

    if features.is_enabled("pwa_drop"):
        ctx.pwa_drop_csv_paths = {}
        ctx.pwa_drop_summary_csv_paths = {}
        events_by_variant = getattr(ctx, "pwa_drop_events_by_variant", None)
        if isinstance(events_by_variant, dict):
            for variant, events in events_by_variant.items():
                if not events:
                    continue
                try:
                    out_path = save_pwa_drop_events_to_csv(ctx.edf_path, cast(list[dict[str, Any]], events), variant=str(variant))
                    if out_path is not None:
                        ctx.pwa_drop_csv_paths[str(variant)] = out_path
                except Exception as e:
                    print(f"  WARNING: could not save PWA drop {variant}% event CSV for {ctx.edf_path.name}: {e}")
        summaries = getattr(ctx, "pwa_drop_summaries", None)
        if isinstance(summaries, dict):
            for variant, summary in summaries.items():
                try:
                    out_path = save_pwa_drop_summary_to_csv(ctx.edf_path, summary, variant=str(variant))
                    if out_path is not None:
                        ctx.pwa_drop_summary_csv_paths[str(variant)] = out_path
                except Exception as e:
                    print(f"  WARNING: could not save PWA drop {variant}% summary CSV for {ctx.edf_path.name}: {e}")

    if features.is_enabled("pat_harmonics") and getattr(ctx, "pat_harmonics_windows", None):
        try:
            windows = ctx.pat_harmonics_windows
            if windows is not None:
                ctx.pat_harmonics_csv_path = save_pat_harmonics_windows_to_csv(ctx.edf_path, cast(list[dict[str, Any]], windows))
        except Exception as e:
            print(f"  WARNING: could not save PAT harmonics windows CSV for {ctx.edf_path.name}: {e}")
    if features.is_enabled("pat_harmonics") and getattr(ctx, "pat_harmonics_summary", None) is not None:
        try:
            ctx.pat_harmonics_summary_csv_path = save_pat_harmonics_summary_to_csv(ctx.edf_path, getattr(ctx, "pat_harmonics_summary", None))
        except Exception as e:
            print(f"  WARNING: could not save PAT harmonics summary CSV for {ctx.edf_path.name}: {e}")

    if features.is_enabled("pat_paper_harmonics") and getattr(ctx, "pat_paper_harmonics_windows", None):
        try:
            windows = ctx.pat_paper_harmonics_windows
            if windows is not None:
                ctx.pat_paper_harmonics_csv_path = save_pat_paper_harmonics_windows_to_csv(ctx.edf_path, cast(list[dict[str, Any]], windows))
        except Exception as e:
            print(f"  WARNING: could not save PAT paper harmonics windows CSV for {ctx.edf_path.name}: {e}")
    if features.is_enabled("pat_paper_harmonics") and getattr(ctx, "pat_paper_harmonics_summary", None) is not None:
        try:
            ctx.pat_paper_harmonics_summary_csv_path = save_pat_paper_harmonics_summary_to_csv(ctx.edf_path, getattr(ctx, "pat_paper_harmonics_summary", None))
        except Exception as e:
            print(f"  WARNING: could not save PAT paper harmonics summary CSV for {ctx.edf_path.name}: {e}")

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


def build_publication_prv_png_step(ctx: RecordingContext) -> None:
    if not bool(getattr(config, "EXPORT_PUBLICATION_PRV_PNG", False)):
        ctx.publication_prv_png_path = None
        return
    if ctx.view_pat is None or ctx.view_pat_filt is None or ctx.sfreq is None:
        ctx.publication_prv_png_path = None
        return
    try:
        ctx.publication_prv_png_path = plotting.save_publication_prv_png(
            edf_base=ctx.edf_base,
            signal_raw=ctx.view_pat,
            signal_filt=ctx.view_pat_filt,
            sfreq=ctx.sfreq,
            t_hr=ctx.t_hr_calc,
            hr=ctx.hr_calc,
            t_prv=ctx.t_prv,
            prv_rmssd=ctx.prv_rmssd_clean,
            prv_tv=ctx.prv_tv,
            aux_df=ctx.aux_df,
            prv_mask_info=ctx.prv_mask_info,
        )
        if ctx.publication_prv_png_path is not None:
            print(f"  Saved publication PRV PNG to: {ctx.publication_prv_png_path}")
        else:
            print(f"  WARNING: no valid 10 min NREM segment found for publication PNG in {ctx.edf_path.name}")
    except Exception as e:
        ctx.publication_prv_png_path = None
        print(f"  WARNING: could not create publication PRV PNG for {ctx.edf_path.name}: {e}")


def append_summary_step(ctx: RecordingContext) -> None:
    if not features.summary_requested():
        return
    hr_metrics.append_hr_prv_summary(
        ctx.edf_path,
        ctx.prv_summary,
        ctx.mayer_peak_freq,
        ctx.resp_peak_freq,
        t_hr=ctx.t_hr_calc,
        hr_calc=ctx.hr_calc,
        t_prv=ctx.t_prv,
        prv_clean=ctx.prv_rmssd_clean,
        prv_raw=ctx.prv_rmssd_raw,
        prv_tv=ctx.prv_tv,
        prv_mask_info=ctx.prv_mask_info,
        prv_midpoint_halves=getattr(ctx, "prv_midpoint_halves", None),
        aux_df=ctx.aux_df,
        hr_event_response_summary=getattr(ctx, "hr_event_response_summary", None),
        pwa_drop_summaries=getattr(ctx, "pwa_drop_summaries", None),
        pat_harmonics_summary=getattr(ctx, "pat_harmonics_summary", None),
        pat_paper_harmonics_summary=getattr(ctx, "pat_paper_harmonics_summary", None),
        psd_features=getattr(ctx, "psd_features", None),
        pat_burden=getattr(ctx, "pat_burden", None),
        pat_burden_diag=getattr(ctx, "pat_burden_diag", None),
        sleep_combo_summaries=getattr(ctx, "sleep_combo_summaries", None),
    )
