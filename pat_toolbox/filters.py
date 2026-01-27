# pat_toolbox/filters.py
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt

from . import config


def bandpass_filter(
    data: np.ndarray,
    fs: float,
    lowcut: Optional[float] = None,
    highcut: Optional[float] = None,
    order: Optional[int] = None,
) -> np.ndarray:
    """
    Apply a Butterworth band-pass filter to 1D data.

    Args:
        data: 1D signal
        fs: sampling frequency [Hz]
        lowcut, highcut: cutoff frequencies [Hz]. If None, use config defaults.
        order: filter order. If None, use config default.
    """
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")

    if lowcut is None:
        lowcut = config.PAT_BANDPASS_LOWCUT_HZ
    if highcut is None:
        highcut = config.PAT_BANDPASS_HIGHCUT_HZ
    if order is None:
        order = config.PAT_BANDPASS_ORDER

    nyq = 0.5 * fs

    low = lowcut / nyq
    high = highcut / nyq

    # Clamp to stable range
    if high >= 1.0:
        high = 0.99
    if low <= 0:
        low = 0.001

    if high <= low:
        raise ValueError(
            f"Invalid bandpass range after normalization: low={low}, high={high}"
        )

    b, a = butter(order, [low, high], btype="band")
    filtered = filtfilt(b, a, data)
    return filtered


import numpy as np
from scipy.signal import butter, filtfilt

def _butter_filter(x: np.ndarray, fs: float, low: float | None = None, high: float | None = None, order: int = 3):
    nyq = 0.5 * fs
    if low is not None and high is not None:
        btype, wn = "bandpass", [low / nyq, high / nyq]
    elif low is not None:
        btype, wn = "highpass", low / nyq
    elif high is not None:
        btype, wn = "lowpass", high / nyq
    else:
        return x.astype(float)

    b, a = butter(order, wn, btype=btype)
    return filtfilt(b, a, x.astype(float))

def actigraph_motion_envelope(act: np.ndarray, fs: float, hp_hz: float = 0.5, lp_hz: float = 2.0) -> np.ndarray:
    """HP -> abs -> LP : gives a smooth 'motion energy' curve."""
    xhp = _butter_filter(act, fs, low=hp_hz, high=None, order=3)
    env = _butter_filter(np.abs(xhp), fs, low=None, high=lp_hz, order=3)
    return env