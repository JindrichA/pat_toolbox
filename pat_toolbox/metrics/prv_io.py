from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .. import config, paths


def save_prv_series_to_csv(
    edf_path: Path,
    t_prv: np.ndarray,
    rmssd_1hz: np.ndarray,
) -> Optional[Path]:
    """
    Save PRV 1 Hz series (RMSSD sliding) to CSV: time_sec, rmssd_ms
    """
    if t_prv.size == 0 or rmssd_1hz.size == 0:
        return None

    prv_sub = getattr(config, "PRV_OUTPUT_SUBFOLDER", getattr(config, "HR_OUTPUT_SUBFOLDER", "PRV"))
    prv_folder = paths.get_output_folder(prv_sub)

    edf_base = edf_path.stem
    out_csv = prv_folder / f"{edf_base}__PRV_RMSSD_1Hz_Clean.csv"

    data = np.column_stack([t_prv, rmssd_1hz])
    header = "time_sec,rmssd_ms"
    np.savetxt(out_csv, data, delimiter=",", header=header, comments="")
    return out_csv


def save_prv_bundle_to_csv(
    edf_path: Path,
    t_prv: np.ndarray,
    rmssd_clean: np.ndarray,
    *,
    rmssd_raw: Optional[np.ndarray] = None,
    prv_tv: Optional[dict[str, np.ndarray]] = None,
) -> Optional[Path]:
    if t_prv.size == 0 or rmssd_clean.size == 0:
        return None

    prv_sub = getattr(config, "PRV_OUTPUT_SUBFOLDER", getattr(config, "HR_OUTPUT_SUBFOLDER", "PRV"))
    prv_folder = paths.get_output_folder(prv_sub)
    edf_base = edf_path.stem
    out_csv = prv_folder / f"{edf_base}__PRV_Features_1Hz.csv"

    cols = [t_prv, rmssd_clean]
    headers = ["time_sec", "rmssd_clean_ms"]

    if rmssd_raw is not None and np.size(rmssd_raw) == np.size(t_prv):
        cols.append(np.asarray(rmssd_raw, dtype=float))
        headers.append("rmssd_raw_ms")

    if isinstance(prv_tv, dict):
        for key in ["sdnn_ms_raw", "sdnn_ms", "lf_raw", "lf", "hf_raw", "hf", "lf_hf_raw", "lf_hf"]:
            value = prv_tv.get(key)
            if value is not None and np.size(value) == np.size(t_prv):
                cols.append(np.asarray(value, dtype=float))
                headers.append(key)

    data = np.column_stack(cols)
    np.savetxt(out_csv, data, delimiter=",", header=",".join(headers), comments="")
    return out_csv


def save_prv_mask_to_csv(
    edf_path: Path,
    t_prv: np.ndarray,
    prv_mask_info: dict[str, object],
) -> Optional[Path]:
    if t_prv.size == 0 or not prv_mask_info:
        return None

    prv_sub = getattr(config, "PRV_OUTPUT_SUBFOLDER", getattr(config, "HR_OUTPUT_SUBFOLDER", "PRV"))
    prv_folder = paths.get_output_folder(prv_sub)
    edf_base = edf_path.stem
    out_csv = prv_folder / f"{edf_base}__PRV_Mask_1Hz.csv"

    cols = [np.asarray(t_prv, dtype=float)]
    headers = ["time_sec"]

    for key in ["sleep_keep", "apnea_keep", "quality_keep", "event_keep", "desat_keep", "combined_keep"]:
        value = prv_mask_info.get(key)
        if value is None or np.size(value) != np.size(t_prv):
            continue
        cols.append(np.asarray(value, dtype=bool).astype(int))
        headers.append(key)

    if len(cols) <= 1:
        return None

    data = np.column_stack(cols)
    np.savetxt(out_csv, data, delimiter=",", header=",".join(headers), comments="")
    return out_csv
