# pat_toolbox/workflows.py

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import config, io_edf, filters, paths, plotting, io_aux_csv, masking
from . import sleep_mask
from .context import RecordingContext
from .metrics import hr as hr_metrics
from .metrics import hrv as hrv_metrics
from .metrics import pat_burden as pat_burden_metrics
from .metrics import psd as psd_metrics
from .metrics.hr_delta import compute_delta_hr

if TYPE_CHECKING:
    import pandas as pd


def _compute_delta_hr(ctx: RecordingContext) -> None:
    if not bool(getattr(config, "ENABLE_DELTA_HR", True)):
        ctx.delta_hr_calc = None
        ctx.delta_hr_calc_evt = None
        return

    lag = float(getattr(config, "DELTA_HR_LAG_SEC", 30.0))
    pre = float(getattr(config, "DELTA_HR_PRE_SMOOTH_SEC", 0.0))
    use_abs = bool(getattr(config, "DELTA_HR_ABS", False))

    if ctx.t_hr_calc is not None and getattr(ctx, "hr_calc_raw", None) is not None:
        fs_pat = float(getattr(config, "HR_TARGET_FS_HZ", 1.0))

        ctx.delta_hr_calc = compute_delta_hr(
            ctx.hr_calc_raw,
            lag_sec=lag,
            pre_smooth_sec=pre,
            fs=fs_pat,
            use_abs=use_abs,
        )

        ctx.delta_hr_calc_evt = None
        if ctx.aux_df is not None:
            m_evt_keep = io_aux_csv.build_time_exclusion_mask(ctx.t_hr_calc, ctx.aux_df)
            if m_evt_keep is not None:
                m_inside = ~np.asarray(m_evt_keep, dtype=bool)
                d = ctx.delta_hr_calc.astype(float, copy=True)
                d[~m_inside] = np.nan
                ctx.delta_hr_calc_evt = d
    else:
        ctx.delta_hr_calc = None
        ctx.delta_hr_calc_evt = None


# ----------------------------
# Small helper steps
# ----------------------------

def _load_pat(ctx: RecordingContext) -> None:
    ctx.view_pat, ctx.sfreq = io_edf.read_edf_channel(ctx.edf_path, config.VIEW_PAT_CHANNEL_NAME)

    n_samples = len(ctx.view_pat)
    if n_samples == 0 or (ctx.sfreq is None) or ctx.sfreq <= 0:
        raise ValueError("VIEW_PAT signal is empty or sampling frequency invalid.")


def _filter_pat(ctx: RecordingContext) -> None:
    assert ctx.view_pat is not None and ctx.sfreq is not None
    ctx.view_pat_filt = filters.bandpass_filter(ctx.view_pat, fs=ctx.sfreq)


def _load_pat_amp(ctx: RecordingContext) -> None:
    try:
        pat_amp_signal, pat_amp_fs = io_edf.read_edf_channel(ctx.edf_path, config.PAT_AMP_CHANNEL_NAME)
        if pat_amp_fs <= 0:
            raise ValueError("PAT AMP sampling frequency <= 0")

        n = len(pat_amp_signal)
        if n > 0:
            ctx.t_pat_amp = np.arange(n) / pat_amp_fs
            ctx.pat_amp = pat_amp_signal.astype(float)
        else:
            print("  WARNING: PAT AMP channel exists but is empty.")
            ctx.t_pat_amp, ctx.pat_amp = None, None
    except Exception as e:
        print(f"  WARNING: could not read PAT AMP channel '{config.PAT_AMP_CHANNEL_NAME}': {e}")
        ctx.t_pat_amp, ctx.pat_amp = None, None


def _load_aux_csv(ctx: RecordingContext) -> None:
    try:
        ctx.aux_df = io_aux_csv.read_aux_csv_for_edf(ctx.edf_path)
        if ctx.aux_df is not None:
            ctx.aux_df = sleep_mask.ensure_stage_code_column(ctx.aux_df)
            print(f"  Loaded aux CSV for {ctx.edf_path.name} with {len(ctx.aux_df)} rows.")
        else:
            print(f"  No aux CSV found for {ctx.edf_path.name}.")
    except Exception as e:
        print(f"  WARNING: could not read aux CSV for {ctx.edf_path.name}: {e}")
        ctx.aux_df = None


def _compute_pat_burden(ctx: RecordingContext) -> None:
    if not bool(getattr(config, "ENABLE_PAT_BURDEN", True)):
        ctx.pat_burden = None
        ctx.pat_burden_diag = None
        ctx.pat_burden_episodes = None
        return

    if ctx.aux_df is None or ctx.t_pat_amp is None or ctx.pat_amp is None:
        ctx.pat_burden = None
        ctx.pat_burden_diag = {"reason": "missing_aux_or_pat_amp"}
        ctx.pat_burden_episodes = None
        return

    try:
        val, diag, eps = pat_burden_metrics.compute_pat_burden_from_pat_amp(
            t_sec=ctx.t_pat_amp,
            pat_amp=ctx.pat_amp,
            aux_df=ctx.aux_df,
        )
        ctx.pat_burden = val
        ctx.pat_burden_diag = diag
        ctx.pat_burden_episodes = eps
        if val is not None and np.isfinite(val):
            unit = "rel·min/h" if diag.get("relative") else "amp·min/h"
            print(f"  PAT burden (event+desat): {val:.3f} {unit} (episodes_used={diag.get('n_episodes_used')})")
    except Exception as e:
        ctx.pat_burden = None
        ctx.pat_burden_diag = {"reason": "exception", "error": str(e)}
        ctx.pat_burden_episodes = None


def _compute_sleep_combo_summaries(ctx: RecordingContext) -> None:
    ctx.sleep_combo_summaries = None

    if ctx.view_pat is None or ctx.sfreq is None or ctx.sfreq <= 0:
        return

    try:
        rr_sec, rr_mid, duration_sec = hr_metrics.extract_clean_rr_from_pat(ctx.view_pat, ctx.sfreq)
    except Exception as e:
        print(f"  WARNING: could not extract base RR for sleep-combo summaries: {e}")
        return

    ctx.rr_mid_clean = rr_mid
    ctx.rr_ms_clean = rr_sec * 1000.0
    ctx.rr_duration_sec = float(duration_sec)

    summaries: dict[str, dict[str, object]] = {}
    for key, label, include_set in sleep_mask.fixed_sleep_stage_policies():
        hrv_summary = hrv_metrics.summarize_hrv_from_rr(
            rr_mid,
            ctx.rr_ms_clean,
            duration_sec,
            ctx.aux_df,
            include_set=include_set,
        )
        psd_features = psd_metrics.compute_psd_features_from_rr(
            rr_mid,
            ctx.rr_ms_clean,
            duration_sec,
            ctx.aux_df,
            include_set=include_set,
        )

        burden = np.nan
        burden_diag = None
        if ctx.t_pat_amp is not None and ctx.pat_amp is not None and ctx.aux_df is not None:
            burden, burden_diag, _episodes = pat_burden_metrics.compute_pat_burden_from_pat_amp(
                t_sec=ctx.t_pat_amp,
                pat_amp=ctx.pat_amp,
                aux_df=ctx.aux_df,
                include_set=include_set,
            )

        summaries[key] = {
            "label": label,
            "include_set": set(include_set),
            "sleep_hours": burden_diag.get("sleep_hours") if isinstance(burden_diag, dict) else np.nan,
            "hrv_summary": hrv_summary,
            "psd_features": psd_features,
            "pat_burden": burden,
            "pat_burden_diag": burden_diag,
        }

    ctx.sleep_combo_summaries = summaries


def _compute_hr_from_pat(ctx: RecordingContext) -> None:
    assert ctx.view_pat is not None and ctx.sfreq is not None
    try:
        ctx.t_hr_calc, ctx.hr_calc = hr_metrics.compute_hr_from_pat_signal(ctx.view_pat, fs=ctx.sfreq)

        ctx.hr_calc_raw = None if ctx.hr_calc is None else ctx.hr_calc.copy()

        bundle = masking.build_mask_bundle(ctx.t_hr_calc, ctx.aux_df)
        sleep_mask.apply_sleep_mask_inplace(ctx.hr_calc, bundle.combined_keep)

    except Exception as e:
        print(f"  WARNING: could not compute HR from PAT: {e}")
        ctx.t_hr_calc, ctx.hr_calc = None, None
        ctx.hr_calc_raw = None


def _compute_hrv(ctx: RecordingContext) -> None:
    assert ctx.view_pat is not None and ctx.sfreq is not None

    try:
        (
            ctx.t_hrv,
            ctx.hrv_rmssd_raw,
            ctx.hrv_rmssd_clean,
            ctx.hrv_summary,
            ctx.hrv_tv,
        ) = hrv_metrics.compute_hrv_from_pat_signal_with_tv_metrics(
            ctx.view_pat,
            fs=ctx.sfreq,
            aux_df=ctx.aux_df,
            target_fs=config.HRV_TARGET_FS_HZ,
            window_sec=config.HRV_WINDOW_SEC,
            tv_window_sec=getattr(config, "HRV_TV_WINDOW_SEC"),
        )

        if ctx.hrv_summary is not None:
            s = ctx.hrv_summary
            nseg = s.get("lf_n_segments_used", None)
            nseg_str = f", LF_segments_used={int(nseg)}" if nseg is not None else ""
            print(
                "  HRV summary (Clean): "
                f"RMSSD_mean={s['rmssd_mean']:.2f} ms, "
                f"SDNN={s['sdnn']:.2f} ms"
                f"{nseg_str}"
            )

        if ctx.t_hrv is not None and ctx.hrv_rmssd_clean is not None:
            ctx.hrv_csv_path = hrv_metrics.save_hrv_series_to_csv(
                ctx.edf_path, ctx.t_hrv, ctx.hrv_rmssd_clean
            )

        if ctx.t_hrv is not None:
            bundle = masking.build_mask_bundle(ctx.t_hrv, ctx.aux_df)
            ctx.hrv_mask_info = {
                "sleep_keep": np.asarray(bundle.sleep_keep, dtype=bool),
                "event_keep": np.asarray(bundle.event_keep, dtype=bool),
                "desat_keep": np.asarray(bundle.desat_keep, dtype=bool),
                "combined_keep": np.asarray(bundle.combined_keep, dtype=bool),
                "active_exclusion_columns": tuple(bundle.active_exclusion_columns),
                "gated_desat_windows": tuple(bundle.gated_desat_windows),
            }

    except Exception as e:
        print(f"  WARNING: HRV computation failed: {e}")
        ctx.t_hrv = None
        ctx.hrv_rmssd_raw = None
        ctx.hrv_rmssd_clean = None
        ctx.hrv_summary = None
        ctx.hrv_tv = None
        ctx.hrv_mask_info = None


def _build_pdf(ctx: RecordingContext) -> None:
    assert ctx.view_pat is not None and ctx.view_pat_filt is not None and ctx.sfreq is not None

    out_folder = paths.get_output_folder()
    if getattr(ctx, "sleep_combo_summaries", None):
        suffix = "_multi_sleep_summary"
    else:
        suffix = config.sleep_stage_suffix() if getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False) else ""
    pdf_name = f"{ctx.edf_base}__VIEW_PAT_HR_HRV_{config.SEGMENT_MINUTES}min_overlay{suffix}.pdf"
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
        delta_hr_calc=ctx.delta_hr_calc,
        delta_hr_edf=None,
        delta_hr_calc_evt=getattr(ctx, "delta_hr_calc_evt", None),
        delta_hr_edf_evt=None,
        pat_burden=getattr(ctx, "pat_burden", None),
        pat_burden_diag=getattr(ctx, "pat_burden_diag", None),
        sleep_combo_summaries=getattr(ctx, "sleep_combo_summaries", None),
        hrv_mask_info=getattr(ctx, "hrv_mask_info", None),
    )

    ctx.psd_features = psd_results_dict
    ctx.mayer_peak_freq = psd_results_dict.get("mayer_peak_hz")
    ctx.resp_peak_freq = psd_results_dict.get("resp_peak_hz")
    print(f"  Saved VIEW_PAT + HR + HRV overlay plots to: {ctx.pdf_path}")


def _build_peaks_debug_pdf(ctx: RecordingContext) -> None:
    if not getattr(config, "ENABLE_PAT_PEAK_DEBUG_PLOTS", False):
        return

    try:
        pdf_path = hr_metrics.create_peaks_debug_pdf_for_edf(ctx.edf_path)
        try:
            ctx.peaks_pdf_path = pdf_path
        except Exception:
            pass
    except Exception as e:
        print(f"  WARNING: could not create peaks debug PDF for {ctx.edf_path.name}: {e}")


def _append_summary(ctx: RecordingContext) -> None:
    hr_metrics.append_hr_hrv_summary(
        ctx.edf_path,
        ctx.hrv_summary,
        ctx.mayer_peak_freq,
        ctx.resp_peak_freq,
        hr_calc=ctx.hr_calc,
        hrv_clean=ctx.hrv_rmssd_clean,
        hrv_raw=ctx.hrv_rmssd_raw,
        hrv_tv=ctx.hrv_tv,
        aux_df=ctx.aux_df,
        psd_features=getattr(ctx, "psd_features", None),
        pat_burden=getattr(ctx, "pat_burden", None),
        pat_burden_diag=getattr(ctx, "pat_burden_diag", None),
        sleep_combo_summaries=getattr(ctx, "sleep_combo_summaries", None),
    )


# ----------------------------
# Public API
# ----------------------------

def process_view_pat_overlay_for_file(edf_path: Path) -> Path | None:
    print(f"Processing EDF for VIEW_PAT + HR + HRV plotting: {edf_path}")
    ctx = RecordingContext(edf_path=edf_path)

    try:
        # Explicitly disable removed reference-HR fields on context
        ctx.t_hr_edf = None
        ctx.hr_edf = None
        ctx.hr_edf_raw = None
        ctx.delta_hr_edf = None
        ctx.delta_hr_edf_evt = None
        ctx.pearson_r = None
        ctx.spear_rho = None
        ctx.rmse = None

        _load_pat(ctx)
        _filter_pat(ctx)
        _load_pat_amp(ctx)
        _load_aux_csv(ctx)
        _compute_sleep_combo_summaries(ctx)
        _compute_pat_burden(ctx)
        _compute_hr_from_pat(ctx)
        _compute_delta_hr(ctx)
        _compute_hrv(ctx)
        _build_pdf(ctx)
        _build_peaks_debug_pdf(ctx)
        _append_summary(ctx)

        return ctx.pdf_path

    except Exception as e:
        print(f"  ERROR: failed processing {edf_path.name}: {e}")
        return None
