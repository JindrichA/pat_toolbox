from __future__ import annotations

import numpy as np
from typing import cast

from . import config, features, masking, sleep_mask
from .context import RecordingContext
from .metrics import hr as hr_metrics
from .metrics import hrv as hrv_metrics
from .metrics import pat_burden as pat_burden_metrics
from .metrics import psd as psd_metrics
from .metrics.hr_event_response import summarize_event_hr_response


def _hr_summary_for_subset(
    hr_raw: np.ndarray,
    t_hr: np.ndarray,
    aux_df,
    include_set: set[int],
) -> dict[str, float]:
    policy = masking.policy_from_config(include_stages=include_set, force_sleep=True)
    bundle = masking.build_mask_bundle(t_hr, aux_df, policy=policy)

    hr_used = np.asarray(hr_raw, dtype=float).copy()
    hr_used[~np.asarray(bundle.combined_keep, dtype=bool)] = np.nan

    ok = np.isfinite(hr_used)
    if not np.any(ok):
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "n_used": 0.0,
        }

    vals = hr_used[ok]
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "n_used": float(vals.size),
    }


def compute_pat_burden_step(ctx: RecordingContext) -> None:
    if not features.is_enabled("pat_burden"):
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
    if not features.is_enabled("sleep_combo_summary"):
        return
    if not features.any_enabled("hr", "hrv", "psd", "delta_hr", "pat_burden"):
        return
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
    t_hr_subset = None
    hr_raw_subset = None
    if features.any_enabled("hr", "delta_hr"):
        try:
            t_hr_subset, hr_raw_subset = hr_metrics.compute_hr_from_pat_signal(ctx.view_pat, fs=ctx.sfreq)
        except Exception:
            t_hr_subset, hr_raw_subset = None, None

    for key, label, include_set in sleep_mask.fixed_sleep_stage_policies():
        sleep_hours = sleep_mask.compute_sleep_hours_from_aux(ctx.aux_df, include_set=include_set)

        hrv_summary = None
        if features.is_enabled("hrv"):
            hrv_summary = hrv_metrics.summarize_hrv_from_rr(rr_mid, ctx.rr_ms_clean, duration_sec, ctx.aux_df, include_set=include_set)

        hr_summary = None
        if features.is_enabled("hr") and t_hr_subset is not None and hr_raw_subset is not None and ctx.aux_df is not None:
            hr_summary = _hr_summary_for_subset(hr_raw_subset, t_hr_subset, ctx.aux_df, include_set)

        psd_features = None
        if features.is_enabled("psd"):
            psd_features = psd_metrics.compute_psd_features_from_rr(rr_mid, ctx.rr_ms_clean, duration_sec, ctx.aux_df, include_set=include_set)

        burden = np.nan
        burden_diag = None
        if features.is_enabled("pat_burden") and ctx.t_pat_amp is not None and ctx.pat_amp is not None and ctx.aux_df is not None:
            burden, burden_diag, _episodes = pat_burden_metrics.compute_pat_burden_from_pat_amp(t_sec=ctx.t_pat_amp, pat_amp=ctx.pat_amp, aux_df=ctx.aux_df, include_set=include_set)

        hr_event_response_summary = None
        if features.is_enabled("delta_hr") and t_hr_subset is not None and hr_raw_subset is not None and ctx.aux_df is not None:
            hr_event_response_summary = summarize_event_hr_response(
                t_hr_subset,
                hr_raw_subset,
                ctx.aux_df,
                include_set=include_set,
            )

        summaries[key] = {
            "label": label,
            "include_set": set(include_set),
            "sleep_hours": sleep_hours,
            "hr_summary": hr_summary,
            "hrv_summary": hrv_summary,
            "psd_features": psd_features,
            "hr_event_response_summary": hr_event_response_summary,
            "pat_burden": burden,
            "pat_burden_diag": burden_diag,
        }
    ctx.sleep_combo_summaries = summaries


def compute_hr_from_pat_step(ctx: RecordingContext) -> None:
    if not features.is_enabled("hr"):
        ctx.t_hr_calc = None
        ctx.hr_calc = None
        ctx.hr_calc_raw = None
        return
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
    if not features.is_enabled("hrv"):
        ctx.t_hrv = None
        ctx.hrv_rmssd_raw = None
        ctx.hrv_rmssd_clean = None
        ctx.hrv_summary = None
        ctx.hrv_tv = None
        ctx.hrv_mask_info = None
        return
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
        if ctx.t_hrv is not None:
            bundle = masking.build_mask_bundle(ctx.t_hrv, ctx.aux_df)
            ctx.hrv_mask_info = {
                "sleep_keep": np.asarray(bundle.sleep_keep, dtype=bool),
                "apnea_keep": np.asarray(bundle.apnea_keep, dtype=bool),
                "quality_keep": np.asarray(bundle.quality_keep, dtype=bool),
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


def compute_psd_step(ctx: RecordingContext) -> None:
    if not features.is_enabled("psd"):
        ctx.psd_features = None
        ctx.mayer_peak_freq = None
        ctx.resp_peak_freq = None
        return
    if ctx.view_pat is None or ctx.sfreq is None or ctx.sfreq <= 0:
        ctx.psd_features = None
        ctx.mayer_peak_freq = None
        ctx.resp_peak_freq = None
        return

    try:
        rr_sec, rr_mid, duration_sec = hr_metrics.extract_clean_rr_from_pat(ctx.view_pat, ctx.sfreq)
        rr_ms = rr_sec * 1000.0
        ctx.psd_features = psd_metrics.compute_psd_features_from_rr(
            rr_mid,
            rr_ms,
            float(duration_sec),
            ctx.aux_df,
        )
        if ctx.psd_features is not None:
            ctx.mayer_peak_freq = ctx.psd_features.get("mayer_peak_hz")
            ctx.resp_peak_freq = ctx.psd_features.get("resp_peak_hz")
    except Exception as e:
        print(f"  WARNING: PSD computation failed: {e}")
        ctx.psd_features = None
        ctx.mayer_peak_freq = None
        ctx.resp_peak_freq = None
