from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, features, filters, io_aux_csv, io_edf, sleep_mask
from .context import RecordingContext


def load_pat(ctx: RecordingContext) -> None:
    ctx.view_pat, ctx.sfreq = io_edf.read_edf_channel(ctx.edf_path, config.VIEW_PAT_CHANNEL_NAME)
    n_samples = len(ctx.view_pat)
    if n_samples == 0 or (ctx.sfreq is None) or ctx.sfreq <= 0:
        raise ValueError("VIEW_PAT signal is empty or sampling frequency invalid.")


def filter_pat(ctx: RecordingContext) -> None:
    assert ctx.view_pat is not None and ctx.sfreq is not None
    ctx.view_pat_filt = filters.bandpass_filter(ctx.view_pat, fs=ctx.sfreq)


def load_actigraph(ctx: RecordingContext) -> None:
    ctx.t_actigraph = None
    ctx.actigraph = None
    if not features.any_enabled("pat_burden", "pwa_drop", "delta_hr"):
        return
    try:
        act_name = getattr(config, "ACTIGRAPH_CHANNEL_NAME", "ACTIGRAPH")
        act_signal, act_fs = io_edf.read_edf_channel(ctx.edf_path, act_name)
        if act_fs <= 0 or act_signal is None or len(act_signal) == 0:
            return
        hp = getattr(config, "ACT_HP_HZ", 0.5)
        lp = getattr(config, "ACT_LP_HZ", 2.0)
        env = filters.actigraph_motion_envelope(act_signal, act_fs, hp_hz=hp, lp_hz=lp)
        if bool(getattr(config, "ACT_ENV_ZSCORE", True)):
            med = float(np.nanmedian(env))
            sd = float(np.nanstd(env))
            env = (env - med) / (sd + 1e-9)
        ctx.t_actigraph = np.arange(len(env), dtype=float) / float(act_fs)
        ctx.actigraph = env.astype(float)
        print(f"  Loaded ACTIGRAPH motion envelope ({act_fs:.3g} Hz).")
    except Exception as e:
        print(f"  WARNING: could not read/process ACTIGRAPH channel '{getattr(config, 'ACTIGRAPH_CHANNEL_NAME', 'ACTIGRAPH')}': {e}")
        ctx.t_actigraph = None
        ctx.actigraph = None


def load_pat_amp(ctx: RecordingContext) -> None:
    if not features.is_enabled("pat_burden"):
        ctx.t_pat_amp = None
        ctx.pat_amp = None
        return
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


def load_spo2(ctx: RecordingContext) -> None:
    ctx.t_spo2 = None
    ctx.spo2 = None
    ctx.spo2_channel_name = None
    if not bool(getattr(config, "ENABLE_SPO2_VALIDATION_PLOTS", False)):
        return
    try:
        candidates = getattr(config, "SPO2_CHANNEL_CANDIDATES", ("SpO2", "SPO2"))
        spo2_signal, spo2_fs, channel_name = io_edf.read_first_available_edf_channel(ctx.edf_path, candidates)
        if spo2_fs <= 0:
            raise ValueError("SpO2 sampling frequency <= 0")
        n = len(spo2_signal)
        if n <= 0:
            raise ValueError("SpO2 channel is empty")
        ctx.t_spo2 = np.arange(n) / spo2_fs
        ctx.spo2 = spo2_signal.astype(float)
        ctx.spo2_channel_name = channel_name
        print(f"  Loaded SpO2 channel '{channel_name}' ({spo2_fs:.3g} Hz).")
    except Exception as e:
        print(f"  WARNING: could not read optional SpO2 channel: {e}")


def load_aux_csv(ctx: RecordingContext) -> None:
    try:
        ctx.aux_df = io_aux_csv.read_aux_csv_for_edf(ctx.edf_path)
        if ctx.aux_df is not None:
            ctx.aux_df = sleep_mask.ensure_stage_code_column(ctx.aux_df)
            print(f"  Loaded aux CSV for {ctx.edf_path.name} with {len(ctx.aux_df)} rows.")
            if (
                bool(getattr(config, "ENABLE_SPO2_VALIDATION_PLOTS", False))
                and ctx.spo2 is None
                and "spo2" in ctx.aux_df.columns
                and getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec") in ctx.aux_df.columns
            ):
                t = ctx.aux_df[getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec")].to_numpy(dtype=float)
                y = pd.to_numeric(ctx.aux_df["spo2"], errors="coerce").to_numpy(dtype=float)
                if t.size == y.size and np.any(np.isfinite(t)) and np.any(np.isfinite(y)):
                    ok = np.isfinite(t)
                    ctx.t_spo2 = t[ok]
                    ctx.spo2 = y[ok].astype(float)
                    ctx.spo2_channel_name = "aux_csv:spo2"
                    print("  Loaded SpO2 from aux CSV column 'spo2'.")
        else:
            print(f"  No aux CSV found for {ctx.edf_path.name}.")
    except Exception as e:
        print(f"  WARNING: could not read aux CSV for {ctx.edf_path.name}: {e}")
        ctx.aux_df = None
