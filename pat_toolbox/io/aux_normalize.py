from __future__ import annotations

from typing import Dict, Optional
import warnings

import numpy as np
import pandas as pd

from .. import config


def parse_time_column_to_seconds(time_series: pd.Series) -> Optional[np.ndarray]:
    """
    Convert an aux 'Time' column into seconds-from-start.

    Handles time-of-day strings that cross midnight (e.g., 22:00 -> 07:00)
    by unwrapping across 24h boundaries.
    """
    if time_series is None or len(time_series) == 0:
        return None

    num = pd.to_numeric(time_series, errors="coerce").to_numpy(dtype=float)
    if np.isfinite(num).mean() > 0.9:
        return num - np.nanmin(num)

    dt = None
    time_as_str = time_series.astype("string").str.strip()

    for fmt in (
        "%H:%M:%S.%f",
        "%H:%M:%S",
        "%I:%M:%S.%f %p",
        "%I:%M:%S %p",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        parsed = pd.to_datetime(time_as_str, format=fmt, errors="coerce")
        if parsed.notna().mean() > 0.9:
            dt = parsed
            break

    if dt is None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format, so each element will be parsed individually",
                category=UserWarning,
            )
            dt = pd.to_datetime(time_as_str, errors="coerce")

    if dt.notna().mean() > 0.9:
        dt_valid = dt.dropna()

        if dt_valid.dt.normalize().nunique() == 1:
            sec_of_day = (
                dt.dt.hour.to_numpy(dtype=float) * 3600.0
                + dt.dt.minute.to_numpy(dtype=float) * 60.0
                + dt.dt.second.to_numpy(dtype=float)
                + dt.dt.microsecond.to_numpy(dtype=float) * 1e-6
            )

            out = sec_of_day.copy()
            valid_mask = np.isfinite(out)
            idx = np.where(valid_mask)[0]
            if idx.size == 0:
                return None

            for k in range(1, idx.size):
                i_prev = idx[k - 1]
                i_cur = idx[k]
                if out[i_cur] < out[i_prev] - 1e-6:
                    out[i_cur:] += 86400.0

            out -= out[idx[0]]
            return out

        t0 = dt.iloc[dt.first_valid_index()]
        return (dt - t0).dt.total_seconds().to_numpy(dtype=float)

    td = pd.to_timedelta(time_series, errors="coerce")
    if td.notna().mean() > 0.9:
        sec = td.dt.total_seconds().to_numpy(dtype=float)
        return sec - np.nanmin(sec)

    return None


def normalize_aux_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize aux CSV into canonical columns."""
    if df is None or df.empty:
        return df

    df = df.copy()

    raw_time = getattr(config, "AUX_CSV_TIME_COLUMN_RAW", "Time")
    time_sec = getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec")

    if raw_time in df.columns:
        sec = parse_time_column_to_seconds(df[raw_time])
        if sec is None:
            sec = np.arange(len(df), dtype=float)
        df[time_sec] = sec
    else:
        df[time_sec] = np.arange(len(df), dtype=float)

    rename_map: Dict[str, str] = {}
    for internal, csv_name in getattr(config, "COL_NAMES", {}).items():
        if csv_name in df.columns:
            rename_map[csv_name] = internal
    df = df.rename(columns=rename_map)

    flag_like = [
        "desat_flag",
        "exclude_spo2_flag",
        "exclude_hr_flag",
        "exclude_pat_flag",
        "evt_central_3",
        "evt_central_4",
        "evt_obstructive_3",
        "evt_obstructive_4",
        "evt_unclassified_3",
        "evt_unclassified_4",
    ]
    for col in flag_like:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    stage_code_col = getattr(config, "AUX_CSV_STAGE_CODE_COLUMN", "stage_code")
    if "stage" in df.columns and stage_code_col not in df.columns:
        mapping = getattr(config, "SLEEP_MAPPING", {})
        mapping_upper = {str(k).strip().upper(): int(v) for k, v in mapping.items()}
        stage_norm = df["stage"].astype(str).str.strip().str.upper()
        df[stage_code_col] = stage_norm.map(mapping_upper).astype(float)

    return df
