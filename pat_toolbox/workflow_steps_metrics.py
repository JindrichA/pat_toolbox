from __future__ import annotations

import numpy as np
from typing import cast

from . import config, features, masking, sleep_mask
from .context import RecordingContext
from .io_aux_csv import compute_sleep_timing_from_aux
from .metrics import hr as hr_metrics
from .metrics import prv as prv_metrics
from .metrics import pat_burden as pat_burden_metrics
from .metrics import pwa_drop as pwa_drop_metrics
from .metrics import pat_harmonics as pat_harmonics_metrics
from .metrics import pat_paper_harmonics as pat_paper_harmonics_metrics
from .metrics import psd as psd_metrics
from .metrics.hr_event_response import extract_event_hr_windows, summarize_event_hr_response, summarize_event_hr_response_from_windows


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


def compute_pat_harmonics_step(ctx: RecordingContext) -> None:
    if not features.is_enabled("pat_harmonics"):
        ctx.pat_harmonics_summary = None
        ctx.pat_harmonics_windows = None
        return
    if ctx.view_pat is None or ctx.sfreq is None or ctx.sfreq <= 0:
        ctx.pat_harmonics_summary = None
        ctx.pat_harmonics_windows = None
        return
    include_set = set(config.sleep_include_numeric()) if bool(getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False)) else None
    try:
        summary, windows = pat_harmonics_metrics.compute_pat_harmonics_from_raw_pat(
            ctx.view_pat,
            ctx.sfreq,
            aux_df=ctx.aux_df,
            include_set=include_set,
        )
        ctx.pat_harmonics_summary = summary
        ctx.pat_harmonics_windows = windows
        if isinstance(summary, dict) and np.isfinite(float(summary.get("f0_hz_median", np.nan))):
            print(
                "  PAT harmonics: "
                f"f0_median={float(summary.get('f0_hz_median', np.nan)):.3f} Hz, "
                f"valid_windows={int(summary.get('n_windows_valid', 0))}/{int(summary.get('n_windows_total', 0))}"
            )
    except Exception as e:
        print(f"  WARNING: PAT harmonics computation failed: {e}")
        ctx.pat_harmonics_summary = None
        ctx.pat_harmonics_windows = None


def compute_pat_paper_harmonics_step(ctx: RecordingContext) -> None:
    if not features.is_enabled("pat_paper_harmonics"):
        ctx.pat_paper_harmonics_summary = None
        ctx.pat_paper_harmonics_windows = None
        return
    if ctx.view_pat is None or ctx.sfreq is None or ctx.sfreq <= 0:
        ctx.pat_paper_harmonics_summary = None
        ctx.pat_paper_harmonics_windows = None
        return
    include_set = set(config.sleep_include_numeric()) if bool(getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False)) else None
    try:
        summary, windows = pat_paper_harmonics_metrics.compute_pat_paper_harmonics_from_raw_pat(
            ctx.view_pat,
            ctx.sfreq,
            aux_df=ctx.aux_df,
            include_set=include_set,
        )
        ctx.pat_paper_harmonics_summary = summary
        ctx.pat_paper_harmonics_windows = windows
        if isinstance(summary, dict):
            print(
                "  PAT paper harmonics: "
                f"C0_median={float(summary.get('c0_median', np.nan)):.3g}, "
                f"valid_windows={int(summary.get('n_windows_valid', 0))}/{int(summary.get('n_windows_total', 0))}"
            )
    except Exception as e:
        print(f"  WARNING: PAT paper harmonics computation failed: {e}")
        ctx.pat_paper_harmonics_summary = None
        ctx.pat_paper_harmonics_windows = None


def _compute_single_sleep_combo_summary(
    *,
    key: str,
    label: str,
    include_set: set[int],
    aux_df,
    pr_mid: np.ndarray,
    pr_ms: np.ndarray,
    duration_sec: float,
    t_hr_subset,
    hr_raw_subset,
    ctx: RecordingContext,
) -> dict[str, object]:
    sleep_hours = sleep_mask.compute_sleep_hours_from_aux(aux_df, include_set=include_set)

    prv_summary = None
    if features.is_enabled("prv"):
        prv_summary = prv_metrics.summarize_prv_from_pr(pr_mid, pr_ms, duration_sec, aux_df, include_set=include_set)

    hr_summary = None
    if features.is_enabled("hr") and t_hr_subset is not None and hr_raw_subset is not None and aux_df is not None:
        hr_summary = _hr_summary_for_subset(hr_raw_subset, t_hr_subset, aux_df, include_set)

    psd_features = None
    if features.is_enabled("psd"):
        psd_features = psd_metrics.compute_psd_features_from_pr(pr_mid, pr_ms, duration_sec, aux_df, include_set=include_set)

    burden = np.nan
    burden_diag = None
    if features.is_enabled("pat_burden") and ctx.t_pat_amp is not None and ctx.pat_amp is not None and aux_df is not None:
        burden, burden_diag, _episodes = pat_burden_metrics.compute_pat_burden_from_pat_amp(
            t_sec=ctx.t_pat_amp,
            pat_amp=ctx.pat_amp,
            aux_df=aux_df,
            include_set=include_set,
        )

    hr_event_response_summary = None
    if features.is_enabled("delta_hr") and t_hr_subset is not None and hr_raw_subset is not None and aux_df is not None:
        hr_event_response_summary = summarize_event_hr_response(
            t_hr_subset,
            hr_raw_subset,
            aux_df,
            include_set=include_set,
        )

    pwa_drop_summaries = None
    pwa_signal = ctx.view_pat_filt if ctx.view_pat_filt is not None else ctx.view_pat
    if features.is_enabled("pwa_drop") and pwa_signal is not None and ctx.sfreq is not None and aux_df is not None:
        try:
            pwa_drop_summaries = {}
            variants = getattr(config, "PWA_DROP_VARIANTS", {"30": {"primary_thr_pct": 30.0, "secondary_thr_pct": 20.0}, "50": {"primary_thr_pct": 50.0, "secondary_thr_pct": 40.0}})
            for variant, params in variants.items():
                _t_pwa, _pwa_series, summary, _events = pwa_drop_metrics.compute_pwa_drop_from_pat_signal(
                    pwa_signal,
                    ctx.sfreq,
                    aux_df=aux_df,
                    include_set=include_set,
                    primary_thr_pct=float(params.get("primary_thr_pct", variant)),
                    secondary_thr_pct=float(params.get("secondary_thr_pct", float(params.get("primary_thr_pct", variant)) - 10.0)),
                )
                pwa_drop_summaries[str(variant)] = summary
        except Exception:
            pwa_drop_summaries = None

    return {
        "label": label,
        "include_set": set(include_set),
        "sleep_hours": sleep_hours,
        "hr_summary": hr_summary,
        "prv_summary": prv_summary,
        "psd_features": psd_features,
        "hr_event_response_summary": hr_event_response_summary,
        "pwa_drop_summaries": pwa_drop_summaries,
        "pat_burden": burden,
        "pat_burden_diag": burden_diag,
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


def compute_pwa_drop_step(ctx: RecordingContext) -> None:
    if not features.is_enabled("pwa_drop"):
        ctx.t_pwa = None
        ctx.pwa_series = None
        ctx.pwa_drop_summaries = None
        ctx.pwa_drop_events_by_variant = None
        return
    if ctx.view_pat_filt is None or ctx.sfreq is None or ctx.sfreq <= 0:
        ctx.t_pwa = None
        ctx.pwa_series = None
        ctx.pwa_drop_summaries = None
        ctx.pwa_drop_events_by_variant = None
        return
    include_set = set(config.sleep_include_numeric()) if bool(getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False)) else None
    try:
        summaries: dict[str, dict[str, float]] = {}
        events_by_variant: dict[str, list[dict[str, float]]] = {}
        variants = getattr(config, "PWA_DROP_VARIANTS", {"30": {"primary_thr_pct": 30.0, "secondary_thr_pct": 20.0}, "50": {"primary_thr_pct": 50.0, "secondary_thr_pct": 40.0}})
        for label, params in variants.items():
            t_pwa, pwa_series, summary, events = pwa_drop_metrics.compute_pwa_drop_from_pat_signal(
                ctx.view_pat_filt,
                ctx.sfreq,
                aux_df=ctx.aux_df,
                include_set=include_set,
                primary_thr_pct=float(params.get("primary_thr_pct", label)),
                secondary_thr_pct=float(params.get("secondary_thr_pct", float(params.get("primary_thr_pct", label)) - 10.0)),
            )
            ctx.t_pwa = t_pwa
            ctx.pwa_series = pwa_series
            summary["variant"] = str(label)
            summaries[str(label)] = summary
            events_by_variant[str(label)] = events
        ctx.pwa_drop_summaries = summaries
        ctx.pwa_drop_events_by_variant = events_by_variant
        msg_parts = []
        for label in sorted(summaries):
            summary = summaries[label]
            if isinstance(summary, dict) and np.isfinite(float(summary.get("drop_rate_per_sleep_hour", np.nan))):
                msg_parts.append(f"{label}%: n={int(summary.get('n_drops', 0))}, rate={float(summary.get('drop_rate_per_sleep_hour', np.nan)):.2f}/h")
        if msg_parts:
            print("  PWA drop summaries: " + "; ".join(msg_parts))
    except Exception as e:
        print(f"  WARNING: PWA drop computation failed: {e}")
        ctx.t_pwa = None
        ctx.pwa_series = None
        ctx.pwa_drop_summaries = None
        ctx.pwa_drop_events_by_variant = None


def compute_sleep_combo_summaries_step(ctx: RecordingContext) -> None:
    ctx.sleep_combo_summaries = None
    if not features.is_enabled("sleep_combo_summary"):
        return
    if not features.any_enabled("hr", "prv", "psd", "delta_hr", "pat_burden", "pwa_drop"):
        return
    if ctx.view_pat is None or ctx.sfreq is None or ctx.sfreq <= 0:
        return
    try:
        pr_sec, pr_mid, duration_sec = hr_metrics.extract_clean_pr_from_pat(ctx.view_pat, ctx.sfreq)
    except Exception as e:
        print(f"  WARNING: could not extract base PR for sleep-combo summaries: {e}")
        return
    ctx.pr_mid_clean = pr_mid
    ctx.pr_ms_clean = pr_sec * 1000.0
    ctx.pr_duration_sec = float(duration_sec)

    summaries: dict[str, dict[str, object]] = {}
    t_hr_subset = None
    hr_raw_subset = None
    if features.any_enabled("hr", "delta_hr"):
        try:
            t_hr_subset, hr_raw_subset = hr_metrics.compute_hr_from_pat_signal(ctx.view_pat, fs=ctx.sfreq)
        except Exception:
            t_hr_subset, hr_raw_subset = None, None

    for key, label, include_set in sleep_mask.fixed_sleep_stage_policies():
        summaries[key] = _compute_single_sleep_combo_summary(
            key=key,
            label=label,
            include_set=include_set,
            aux_df=ctx.aux_df,
            pr_mid=pr_mid,
            pr_ms=ctx.pr_ms_clean,
            duration_sec=duration_sec,
            t_hr_subset=t_hr_subset,
            hr_raw_subset=hr_raw_subset,
            ctx=ctx,
        )

    pre_key, pre_label, pre_include_set = sleep_mask.pre_sleep_wake_policy()
    pre_aux_df = sleep_mask.build_pre_sleep_wake_aux_df(ctx.aux_df)
    if pre_aux_df is not None:
        summaries[pre_key] = _compute_single_sleep_combo_summary(
            key=pre_key,
            label=pre_label,
            include_set=pre_include_set,
            aux_df=pre_aux_df,
            pr_mid=pr_mid,
            pr_ms=ctx.pr_ms_clean,
            duration_sec=duration_sec,
            t_hr_subset=t_hr_subset,
            hr_raw_subset=hr_raw_subset,
            ctx=ctx,
        )
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


def compute_delta_hr_step(ctx: RecordingContext) -> None:
    if not features.is_enabled("delta_hr"):
        ctx.hr_event_response_summary = None
        ctx.hr_event_windows = None
        return
    if ctx.t_hr_calc is None or ctx.hr_calc_raw is None or ctx.aux_df is None:
        ctx.hr_event_response_summary = None
        ctx.hr_event_windows = None
        return

    include_set = set(config.sleep_include_numeric()) if bool(getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False)) else None
    try:
        windows = extract_event_hr_windows(
            ctx.t_hr_calc,
            ctx.hr_calc_raw,
            ctx.aux_df,
            include_set=include_set,
        )
        ctx.hr_event_windows = windows
        ctx.hr_event_response_summary = summarize_event_hr_response_from_windows(windows)
    except Exception as e:
        print(f"  WARNING: delta HR computation failed: {e}")
        ctx.hr_event_response_summary = None
        ctx.hr_event_windows = None


def compute_prv_step(ctx: RecordingContext) -> None:
    if not features.is_enabled("prv"):
        ctx.t_prv = None
        ctx.prv_rmssd_raw = None
        ctx.prv_rmssd_clean = None
        ctx.prv_summary = None
        ctx.prv_tv = None
        ctx.prv_mask_info = None
        ctx.prv_midpoint_halves = None
        return
    assert ctx.view_pat is not None and ctx.sfreq is not None
    try:
        ctx.t_prv, ctx.prv_rmssd_raw, ctx.prv_rmssd_clean, ctx.prv_summary, ctx.prv_tv = prv_metrics.compute_prv_from_pat_signal_with_tv_metrics(
            ctx.view_pat,
            fs=ctx.sfreq,
            aux_df=ctx.aux_df,
            target_fs=config.PRV_TARGET_FS_HZ,
            window_sec=config.PRV_WINDOW_SEC,
            tv_window_sec=getattr(config, "PRV_TV_WINDOW_SEC"),
        )
        if ctx.prv_summary is not None:
            s = ctx.prv_summary
            nseg = s.get("lf_n_segments_used", None)
            nseg_str = f", LF_segments_used={int(nseg)}" if nseg is not None else ""
            print("  PRV summary (Clean): " f"RMSSD_mean={s['rmssd_mean']:.2f} ms, " f"SDNN={s['sdnn']:.2f} ms" f"{nseg_str}")
        if ctx.t_prv is not None:
            bundle = masking.build_mask_bundle(ctx.t_prv, ctx.aux_df)
            ctx.prv_mask_info = {
                "sleep_keep": np.asarray(bundle.sleep_keep, dtype=bool),
                "apnea_keep": np.asarray(bundle.apnea_keep, dtype=bool),
                "quality_keep": np.asarray(bundle.quality_keep, dtype=bool),
                "event_keep": np.asarray(bundle.event_keep, dtype=bool),
                "desat_keep": np.asarray(bundle.desat_keep, dtype=bool),
                "combined_keep": np.asarray(bundle.combined_keep, dtype=bool),
                "active_exclusion_columns": tuple(bundle.active_exclusion_columns),
                "gated_desat_windows": tuple(bundle.gated_desat_windows),
            }
        try:
            pr_sec, pr_mid, duration_sec = hr_metrics.extract_clean_pr_from_pat(ctx.view_pat, ctx.sfreq)
            pr_ms = pr_sec * 1000.0
            ctx.pr_mid_clean = pr_mid
            ctx.pr_ms_clean = pr_ms
            ctx.pr_duration_sec = float(duration_sec)
            ctx.sleep_timing = compute_sleep_timing_from_aux(ctx.aux_df) if ctx.aux_df is not None else None
            ctx.prv_midpoint_halves = None
            if ctx.sleep_timing is not None and ctx.aux_df is not None:
                onset_sec = float(ctx.sleep_timing.get("sleep_onset_rel_sec", np.nan))
                split_sec = float(ctx.sleep_timing.get("sleep_midpoint_rel_sec", np.nan))
                end_sec = float(ctx.sleep_timing.get("sleep_end_rel_sec", np.nan))
                if np.isfinite(onset_sec) and np.isfinite(split_sec) and np.isfinite(end_sec) and onset_sec < split_sec < end_sec:
                    include_set = {1, 2}
                    _pr_mid_sleep, _pr_ms_sleep, pr_mid_clean, pr_ms_clean = prv_metrics._subset_pr_by_sleep_and_events(
                        pr_mid,
                        pr_ms,
                        ctx.aux_df,
                        include_set=include_set,
                    )
                    in_sleep_window = (pr_mid_clean >= onset_sec) & (pr_mid_clean < end_sec)
                    pr_mid_sleep_window = pr_mid_clean[in_sleep_window] - onset_sec
                    pr_ms_sleep_window = pr_ms_clean[in_sleep_window]
                    ctx.prv_midpoint_halves = prv_metrics.summarize_prv_halves_from_clean_pr(
                        pr_mid_sleep_window,
                        pr_ms_sleep_window,
                        split_sec=split_sec - onset_sec,
                        duration_sec=end_sec - onset_sec,
                        target_fs=config.PRV_TARGET_FS_HZ,
                        window_sec=config.PRV_WINDOW_SEC,
                    )
        except Exception as e:
            print(f"  WARNING: could not compute sleep-midpoint PRV halves: {e}")
            ctx.prv_midpoint_halves = None
    except Exception as e:
        print(f"  WARNING: PRV computation failed: {e}")
        ctx.t_prv = None
        ctx.prv_rmssd_raw = None
        ctx.prv_rmssd_clean = None
        ctx.prv_summary = None
        ctx.prv_tv = None
        ctx.prv_mask_info = None
        ctx.prv_midpoint_halves = None


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
        pr_sec, pr_mid, duration_sec = hr_metrics.extract_clean_pr_from_pat(ctx.view_pat, ctx.sfreq)
        pr_ms = pr_sec * 1000.0
        ctx.psd_features = psd_metrics.compute_psd_features_from_pr(
            pr_mid,
            pr_ms,
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
