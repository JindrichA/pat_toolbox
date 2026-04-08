from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import config, filters, io_edf, paths, plotting
from .hr_compute import _detect_pat_peaks


def create_peaks_debug_pdf_for_edf(edf_path: Path) -> Path | None:
    """
    Create a debug PDF showing PAT signal and detected peaks (1 min per page),
    with ACTIGRAPH subplot if available.
    """
    print(f"Creating PAT peaks debug PDF for: {edf_path}")

    try:
        pat_signal, fs = io_edf.read_edf_channel(edf_path, config.VIEW_PAT_CHANNEL_NAME)
    except Exception as e:
        print(f"  WARNING: cannot create peaks debug PDF for {edf_path.name}: {e}")
        return None

    n_samples = len(pat_signal)
    if n_samples == 0 or fs <= 0:
        print("  WARNING: PAT signal empty or invalid fs, skipping debug PDF.")
        return None

    try:
        pat_filt, peaks = _detect_pat_peaks(pat_signal, fs)
    except Exception as e:
        print(f"  WARNING: peak detection failed for {edf_path.name}: {e}")
        return None

    act_to_plot = None
    act_fs = None
    act_label = getattr(config, "ACTIGRAPH_CHANNEL_NAME", "ACTIGRAPH")

    try:
        act_name = getattr(config, "ACTIGRAPH_CHANNEL_NAME", "ACTIGRAPH")
        act_signal, act_fs = io_edf.read_edf_channel(edf_path, act_name)

        if act_fs <= 0 or act_signal is None or len(act_signal) == 0:
            print("  WARNING: ACTIGRAPH empty or invalid fs, ignoring.")
            act_to_plot, act_fs = None, None
        else:
            hp = getattr(config, "ACT_HP_HZ", 0.5)
            lp = getattr(config, "ACT_LP_HZ", 2.0)
            do_z = getattr(config, "ACT_ENV_ZSCORE", True)

            env = filters.actigraph_motion_envelope(act_signal, act_fs, hp_hz=hp, lp_hz=lp)

            if do_z:
                m = float(np.nanmedian(env))
                s = float(np.nanstd(env))
                env = (env - m) / (s + 1e-9)
                act_label = f"{act_name} envelope (z)"
            else:
                act_label = f"{act_name} envelope"

            act_to_plot = env

    except Exception as e:
        print(f"  WARNING: could not read/process ACTIGRAPH: {e}")
        act_to_plot, act_fs = None, None

    out_folder = paths.get_output_folder()
    edf_base = edf_path.stem
    pdf_name = f"{edf_base}__PAT_Peaks_{config.PAT_PEAK_DEBUG_SEGMENT_MINUTES}min.pdf"
    pdf_path = out_folder / pdf_name

    plotting.plot_pat_with_peaks_segments_to_pdf(
        signal_raw=pat_signal,
        signal_filt=pat_filt,
        peak_indices=peaks,
        sfreq=fs,
        pdf_path=pdf_path,
        segment_minutes=config.PAT_PEAK_DEBUG_SEGMENT_MINUTES,
        title_prefix=edf_base,
        channel_name=config.VIEW_PAT_CHANNEL_NAME,
        actigraph=act_to_plot,
        act_sfreq=act_fs,
        act_label=act_label,
    )

    print(f"  Saved PAT peaks debug PDF to: {pdf_path}")
    return pdf_path
