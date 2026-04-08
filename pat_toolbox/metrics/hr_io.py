from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from .. import config, io_edf, paths
from .hr_compute import compute_hr_from_pat_signal


def save_hr_series_to_csv(
    edf_path: Path,
    t_hr: np.ndarray,
    hr_1hz: np.ndarray,
) -> Path | None:
    if t_hr.size == 0 or hr_1hz.size == 0:
        return None

    hr_folder = paths.get_output_folder(config.HR_OUTPUT_SUBFOLDER)
    edf_base = edf_path.stem
    out_csv = hr_folder / f"{edf_base}__HR_1Hz.csv"

    data = np.column_stack([t_hr, hr_1hz])
    header = "time_sec,hr_bpm"
    np.savetxt(out_csv, data, delimiter=",", header=header, comments="")
    return out_csv


def compute_hr_for_edf_file(
    edf_path: Path,
    save_csv: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper: read PAT from EDF, compute HR 1 Hz, optionally save CSV.
    """
    print(f"Computing HR from PAT for: {edf_path}")

    try:
        pat_signal, fs = io_edf.read_edf_channel(edf_path, config.VIEW_PAT_CHANNEL_NAME)
    except Exception as e:
        print(f"  WARNING: skipping HR for {edf_path.name}: {e}")
        return np.array([]), np.array([])

    try:
        t_hr, hr_1hz = compute_hr_from_pat_signal(pat_signal, fs)
    except Exception as e:
        print(f"  WARNING: could not compute HR from PAT for {edf_path.name}: {e}")
        return np.array([]), np.array([])

    if save_csv and t_hr.size > 0:
        out_csv = save_hr_series_to_csv(edf_path, t_hr, hr_1hz)
        print(f"  Saved HR 1 Hz CSV to: {out_csv}")

    return t_hr, hr_1hz
