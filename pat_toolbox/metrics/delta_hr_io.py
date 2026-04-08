from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .. import config, paths


def save_delta_hr_series_to_csv(
    edf_path: Path,
    t_hr: np.ndarray,
    delta_hr: np.ndarray,
    *,
    delta_hr_evt: Optional[np.ndarray] = None,
) -> Path | None:
    if t_hr.size == 0 or delta_hr.size == 0:
        return None

    out_folder = paths.get_output_folder(getattr(config, "DELTA_HR_OUTPUT_SUBFOLDER", config.HR_OUTPUT_SUBFOLDER))
    edf_base = edf_path.stem
    out_csv = out_folder / f"{edf_base}__DeltaHR_1Hz.csv"

    cols = [t_hr, delta_hr]
    headers = ["time_sec", "delta_hr_bpm"]
    if delta_hr_evt is not None and np.size(delta_hr_evt) == np.size(t_hr):
        cols.append(np.asarray(delta_hr_evt, dtype=float))
        headers.append("delta_hr_event_only_bpm")

    data = np.column_stack(cols)
    np.savetxt(out_csv, data, delimiter=",", header=",".join(headers), comments="")
    return out_csv
