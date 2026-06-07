# pat_toolbox/io_edf.py
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pyedflib

from . import config


def list_edf_files(edf_folder: Path | None = None) -> List[Path]:
    """
    List all EDF files in the given folder (default: config.EDF_FOLDER).
    """
    if edf_folder is None:
        edf_folder = config.EDF_FOLDER

    if not edf_folder.is_dir():
        raise FileNotFoundError(f"EDF folder does not exist: {edf_folder}")

    return sorted(p for p in edf_folder.iterdir() if p.suffix.lower() == ".edf")


def read_edf_channel(edf_path: Path, channel_name: str) -> Tuple[np.ndarray, float]:
    """
    Read a single channel from an EDF file.

    Returns:
        signal: np.ndarray (float)
        sample_frequency: float (Hz)
    """
    try:
        f = pyedflib.EdfReader(str(edf_path))
    except Exception as e:
        raise RuntimeError(f"Could not open EDF file {edf_path}: {e}") from e

    try:
        labels = f.getSignalLabels()
        if channel_name not in labels:
            raise ValueError(
                f"Channel '{channel_name}' not found in EDF {edf_path.name}. "
                f"Available channels: {labels}"
            )

        idx = labels.index(channel_name)
        signal = f.readSignal(idx).astype(float)
        sfreq = float(f.getSampleFrequency(idx))
    finally:
        f.close()

    return signal, sfreq


def read_first_available_edf_channel(edf_path: Path, channel_names: Iterable[str]) -> Tuple[np.ndarray, float, str]:
    labels = tuple(str(name) for name in channel_names)
    last_error: Exception | None = None
    for channel_name in labels:
        try:
            signal, sfreq = read_edf_channel(edf_path, channel_name)
            return signal, sfreq, channel_name
        except Exception as exc:
            last_error = exc
    raise ValueError(f"None of the EDF channels {labels} were found in {edf_path.name}: {last_error}")
