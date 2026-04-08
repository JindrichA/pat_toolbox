from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .. import config, paths


def save_hrv_series_to_csv(
    edf_path: Path,
    t_hrv: np.ndarray,
    rmssd_1hz: np.ndarray,
) -> Optional[Path]:
    """
    Save HRV 1 Hz series (RMSSD sliding) to CSV: time_sec, rmssd_ms
    """
    if t_hrv.size == 0 or rmssd_1hz.size == 0:
        return None

    hrv_sub = getattr(config, "HRV_OUTPUT_SUBFOLDER", getattr(config, "HR_OUTPUT_SUBFOLDER", "HRV"))
    hrv_folder = paths.get_output_folder(hrv_sub)

    edf_base = edf_path.stem
    out_csv = hrv_folder / f"{edf_base}__HRV_RMSSD_1Hz_Clean.csv"

    data = np.column_stack([t_hrv, rmssd_1hz])
    header = "time_sec,rmssd_ms"
    np.savetxt(out_csv, data, delimiter=",", header=header, comments="")
    return out_csv
