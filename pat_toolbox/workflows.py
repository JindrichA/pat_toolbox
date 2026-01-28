# pat_toolbox/workflows.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, TYPE_CHECKING

import numpy as np

from . import config, io_edf, filters, paths, plotting, io_aux_csv
from . import sleep_mask
from .context import RecordingContext
from .metrics import hr as hr_metrics
from .metrics import hrv as hrv_metrics
from .metrics.hr import compute_hr_correlation

# Make sure plotting import points to where plot_pat_and_hr_segments_to_pdf is exposed
# If you are using the split file structure, this might need to be:
# from .plotting.report import plot_pat_and_hr_segments_to_pdf
# But usually .plotting.__init__ exposes it.

if TYPE_CHECKING:
    import pandas as pd




def _compute_delta_hr(ctx: RecordingContext) -> None:
    if not bool(getattr(config, "ENABLE_DELTA_HR", True)):
        ctx.delta_hr_calc = None
        ctx.delta_hr_edf = None
        return

    lag = float(getattr(config, "DELTA_HR_LAG_SEC", 30.0))
    pre = float(getattr(config, "DELTA_HR_PRE_SMOOTH_SEC", 0.0))
    use_abs = bool(getattr(config, "DELTA_HR_ABS", False))

    # PAT HR (assumed HR_TARGET_FS_HZ grid)
    if ctx.t_hr_calc is not None and ctx.hr_calc is not None and ctx.hr_calc.size > 0:
        fs_pat = float(getattr(config, "HR_TARGET_FS_HZ", 1.0))
        ctx.delta_hr_calc = hr_metrics.compute_delta_hr(
            ctx.t_hr_calc, ctx.hr_calc,
            lag_sec=lag, pre_smooth_sec=pre, fs=fs_pat, use_abs=use_abs
        )
    else:
        ctx.delta_hr_calc = None

    # EDF HR (estimate fs from time vector)
    if ctx.t_hr_edf is not None and ctx.hr_edf is not None and ctx.hr_edf.size > 0:
        dt = np.diff(ctx.t_hr_edf)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        fs_edf = 1.0 / float(np.median(dt)) if dt.size else 1.0
        ctx.delta_hr_edf = hr_metrics.compute_delta_hr(
            ctx.t_hr_edf, ctx.hr_edf,
            lag_sec=lag, pre_smooth_sec=pre, fs=fs_edf, use_abs=use_abs
        )
    else:
        ctx.delta_hr_edf = None



# ----------------------------
# Small helper steps (readable)
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


def _compute_hr_from_pat(ctx: RecordingContext) -> None:
    assert ctx.view_pat is not None and ctx.sfreq is not None
    try:
        ctx.t_hr_calc, ctx.hr_calc = hr_metrics.compute_hr_from_pat_signal(ctx.view_pat, fs=ctx.sfreq)
        m = sleep_mask.build_sleep_include_mask(ctx.t_hr_calc, ctx.aux_df)
        sleep_mask.apply_sleep_mask_inplace(ctx.hr_calc, m)
    except Exception as e:
        print(f"  WARNING: could not compute HR from PAT: {e}")
        ctx.t_hr_calc, ctx.hr_calc = None, None


def _load_hr_from_edf(ctx: RecordingContext) -> None:
    try:
        hr_signal, hr_fs = io_edf.read_edf_channel(ctx.edf_path, config.HR_CHANNEL_NAME)
        if hr_fs <= 0:
            raise ValueError("HR sampling frequency <= 0")

        n = len(hr_signal)
        if n > 0:
            ctx.t_hr_edf = np.arange(n) / hr_fs
            ctx.hr_edf = hr_signal.astype(float) * config.HR_EDF_SCALE_FACTOR
            m = sleep_mask.build_sleep_include_mask(ctx.t_hr_edf, ctx.aux_df)
            sleep_mask.apply_sleep_mask_inplace(ctx.hr_edf, m)
        else:
            print("  WARNING: HR channel exists but is empty.")
    except Exception as e:
        print(f"  WARNING: could not read HR channel '{config.HR_CHANNEL_NAME}': {e}")
        ctx.t_hr_edf, ctx.hr_edf = None, None


def _compute_hr_correlation(ctx: RecordingContext) -> None:
    if (
            ctx.t_hr_edf is None or ctx.hr_edf is None
            or ctx.t_hr_calc is None or ctx.hr_calc is None
    ):
        print("  HR correlation: HR EDF or PAT HR missing, not computed.")
        ctx.pearson_r = ctx.spear_rho = ctx.rmse = None
        return

    ctx.pearson_r, ctx.spear_rho, ctx.rmse = compute_hr_correlation(
        ctx.t_hr_edf,
        ctx.hr_edf,
        ctx.t_hr_calc,
        ctx.hr_calc,
        common_fs=config.HR_TARGET_FS_HZ,
    )

    if ctx.pearson_r is not None:
        print(
            f"  HR correlation: Pearson={ctx.pearson_r:.3f}, "
            f"Spearman={ctx.spear_rho:.3f}, RMSE={ctx.rmse:.2f} bpm"
        )


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

        m_hrv = sleep_mask.build_sleep_include_mask(ctx.t_hrv, ctx.aux_df)
        m_excl = io_aux_csv.build_time_exclusion_mask(ctx.t_hrv, ctx.aux_df)
        if m_excl is not None:
            sleep_mask.apply_sleep_mask_inplace(ctx.hrv_rmssd_raw, m_excl)
            sleep_mask.apply_sleep_mask_inplace(ctx.hrv_rmssd_clean, m_excl)
            if isinstance(ctx.hrv_tv, dict):
                for k, v in list(ctx.hrv_tv.items()):
                    if v is None: continue
                    vv = np.asarray(v)
                    if vv.size == m_excl.size:
                        ctx.hrv_tv[k] = sleep_mask.apply_sleep_mask_inplace(vv, m_excl)

        if m_hrv is not None:
            sleep_mask.apply_sleep_mask_inplace(ctx.hrv_rmssd_raw, m_hrv)
            sleep_mask.apply_sleep_mask_inplace(ctx.hrv_rmssd_clean, m_hrv)
            if isinstance(ctx.hrv_tv, dict):
                for k, v in list(ctx.hrv_tv.items()):
                    if v is None: continue
                    vv = np.asarray(v)
                    if vv.size == m_hrv.size:
                        ctx.hrv_tv[k] = sleep_mask.apply_sleep_mask_inplace(vv, m_hrv)

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

    except Exception as e:
        print(f"  WARNING: HRV computation failed: {e}")
        ctx.t_hrv = ctx.hrv_rmssd_raw = ctx.hrv_rmssd_clean = None
        ctx.hrv_summary = None
        ctx.hrv_tv = None


def _build_pdf(ctx: RecordingContext) -> None:
    assert ctx.view_pat is not None and ctx.view_pat_filt is not None and ctx.sfreq is not None

    out_folder = paths.get_output_folder()
    suffix = config.sleep_stage_suffix() if getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False) else ""
    pdf_name = f"{ctx.edf_base}__VIEW_PAT_HR_HRV_{config.SEGMENT_MINUTES}min_overlay{suffix}.pdf"
    ctx.pdf_path = out_folder / pdf_name

    # We now expect a DICTIONARY back, not a tuple of 2 floats.
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
        t_hr_edf=ctx.t_hr_edf,
        hr_edf=ctx.hr_edf,
        t_hrv=ctx.t_hrv,
        hrv_rmssd=ctx.hrv_rmssd_clean,
        hrv_rmssd_raw=ctx.hrv_rmssd_raw,
        hrv_tv=ctx.hrv_tv,
        pearson_r=ctx.pearson_r,
        spear_rho=ctx.spear_rho,
        rmse=ctx.rmse,
        hrv_summary=ctx.hrv_summary,
        aux_df=ctx.aux_df,
        t_pat_amp=ctx.t_pat_amp,
        pat_amp=ctx.pat_amp,
        delta_hr_calc=getattr(ctx, "delta_hr_calc", None),
        delta_hr_edf=getattr(ctx, "delta_hr_edf", None),
    )

    # Store the dictionary in context for the summary CSV step
    ctx.psd_features = psd_results_dict

    # Extract the peaks for context (backward compatibility)
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
    # We pass the full psd_features dictionary to the CSV writer now
    hr_metrics.append_hr_correlation_to_summary(
        ctx.edf_path,
        ctx.pearson_r,
        ctx.spear_rho,
        ctx.rmse,
        ctx.hrv_summary,
        ctx.mayer_peak_freq,
        ctx.resp_peak_freq,
        hr_calc=ctx.hr_calc,
        hr_edf=ctx.hr_edf,
        hrv_clean=ctx.hrv_rmssd_clean,
        hrv_raw=ctx.hrv_rmssd_raw,
        hrv_tv=ctx.hrv_tv,
        aux_df=ctx.aux_df,
        psd_features=getattr(ctx, "psd_features", None)  # Pass the new dict
    )


# ----------------------------
# Public API
# ----------------------------

def process_view_pat_overlay_for_file(edf_path: Path) -> Path | None:
    print(f"Processing EDF for VIEW_PAT + HR + HRV plotting: {edf_path}")
    ctx = RecordingContext(edf_path=edf_path)

    try:
        _load_pat(ctx)
        _filter_pat(ctx)
        _load_pat_amp(ctx)
        _load_aux_csv(ctx)
        _compute_hr_from_pat(ctx)
        _load_hr_from_edf(ctx)
        _compute_delta_hr(ctx)
        _compute_hr_correlation(ctx)
        _compute_hrv(ctx)
        _build_pdf(ctx)
        _build_peaks_debug_pdf(ctx)
        _append_summary(ctx)

        return ctx.pdf_path

    except Exception as e:
        print(f"  ERROR: failed processing {edf_path.name}: {e}")
        # import traceback
        # traceback.print_exc()
        return None
