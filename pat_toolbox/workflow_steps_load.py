from __future__ import annotations

import numpy as np

from . import config, filters, io_aux_csv, io_edf, sleep_mask
from .context import RecordingContext


def load_pat(ctx: RecordingContext) -> None:
    ctx.view_pat, ctx.sfreq = io_edf.read_edf_channel(ctx.edf_path, config.VIEW_PAT_CHANNEL_NAME)
    n_samples = len(ctx.view_pat)
    if n_samples == 0 or (ctx.sfreq is None) or ctx.sfreq <= 0:
        raise ValueError("VIEW_PAT signal is empty or sampling frequency invalid.")


def filter_pat(ctx: RecordingContext) -> None:
    assert ctx.view_pat is not None and ctx.sfreq is not None
    ctx.view_pat_filt = filters.bandpass_filter(ctx.view_pat, fs=ctx.sfreq)


def load_pat_amp(ctx: RecordingContext) -> None:
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


def load_aux_csv(ctx: RecordingContext) -> None:
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
