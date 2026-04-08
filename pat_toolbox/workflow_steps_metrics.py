from __future__ import annotations

import numpy as np
from typing import cast

from . import config, io_aux_csv, masking, sleep_mask
from .context import RecordingContext
from .metrics import hr as hr_metrics
from .metrics import hrv as hrv_metrics
from .metrics import pat_burden as pat_burden_metrics
from .metrics import psd as psd_metrics
from .metrics.hr_delta import compute_delta_hr


def compute_delta_hr_step(ctx: RecordingContext) -> None:
    if not bool(getattr(config, "ENABLE_DELTA_HR", True)):
        ctx.delta_hr_calc = None
        ctx.delta_hr_calc_evt = None
        return
    lag = float(getattr(config, "DELTA_HR_LAG_SEC", 30.0))
    pre = float(getattr(config, "DELTA_HR_PRE_SMOOTH_SEC", 0.0))
    use_abs = bool(getattr(config, "DELTA_HR_ABS", False))
    if ctx.t_hr_calc is not None and getattr(ctx, "hr_calc_raw", None) is not None:
        fs_pat = float(getattr(config, "HR_TARGET_FS_HZ", 1.0))
        ctx.delta_hr_calc = compute_delta_hr(cast(np.ndarray, ctx.hr_calc_raw), lag_sec=lag, pre_smooth_sec=pre, fs=fs_pat, use_abs=use_abs)
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


def compute_pat_burden_step(ctx: RecordingContext) -> None:
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
        val, diag, eps = pat_burden_metrics.compute_pat_burden_from_pat_amp(t_sec=ctx.t_pat_amp, pat_amp=ctx.pat_amp, aux_df=ctx.aux_df)
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


def compute_sleep_combo_summaries_step(ctx: RecordingContext) -> None:
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
        hrv_summary = hrv_metrics.summarize_hrv_from_rr(rr_mid, ctx.rr_ms_clean, duration_sec, ctx.aux_df, include_set=include_set)
        psd_features = psd_metrics.compute_psd_features_from_rr(rr_mid, ctx.rr_ms_clean, duration_sec, ctx.aux_df, include_set=include_set)
        burden = np.nan
        burden_diag = None
        if ctx.t_pat_amp is not None and ctx.pat_amp is not None and ctx.aux_df is not None:
            burden, burden_diag, _episodes = pat_burden_metrics.compute_pat_burden_from_pat_amp(t_sec=ctx.t_pat_amp, pat_amp=ctx.pat_amp, aux_df=ctx.aux_df, include_set=include_set)
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


def compute_hr_from_pat_step(ctx: RecordingContext) -> None:
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


def compute_hrv_step(ctx: RecordingContext) -> None:
    assert ctx.view_pat is not None and ctx.sfreq is not None
    try:
        ctx.t_hrv, ctx.hrv_rmssd_raw, ctx.hrv_rmssd_clean, ctx.hrv_summary, ctx.hrv_tv = hrv_metrics.compute_hrv_from_pat_signal_with_tv_metrics(
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
            print("  HRV summary (Clean): " f"RMSSD_mean={s['rmssd_mean']:.2f} ms, " f"SDNN={s['sdnn']:.2f} ms" f"{nseg_str}")
        if ctx.t_hrv is not None and ctx.hrv_rmssd_clean is not None:
            ctx.hrv_csv_path = hrv_metrics.save_hrv_series_to_csv(ctx.edf_path, ctx.t_hrv, ctx.hrv_rmssd_clean)
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
