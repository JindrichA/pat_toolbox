# pat_toolbox/io_aux_csv.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from . import config


# =============================================================================
# File discovery / reading
# =============================================================================

def find_aux_csv_for_edf(edf_path: Path) -> Optional[Path]:
    """
    Given an EDF path, find the corresponding auxiliary CSV.

    Default strategy:
      <edf_stem>.csv in the same folder.
    """
    if not getattr(config, "AUX_CSV_ENABLED", True):
        return None

    aux_path = edf_path.with_suffix(getattr(config, "AUX_CSV_EXTENSION", ".csv"))
    return aux_path if aux_path.exists() else None


def read_raw_aux_csv(aux_path: Path) -> Optional[pd.DataFrame]:
    """
    Low-level CSV reader.
    """
    try:
        usecols = getattr(config, "AUX_CSV_USE_COLUMNS", None)
        df = pd.read_csv(aux_path, sep=getattr(config, "AUX_CSV_SEP", ","), usecols=usecols)
    except Exception as e:
        print(f"  WARNING: could not read aux CSV {aux_path}: {e}")
        return None

    if df is None or df.empty:
        print(f"  WARNING: aux CSV {aux_path} is empty.")
        return None

    return df


# =============================================================================
# Time parsing / normalization
# =============================================================================

def _parse_time_column_to_seconds(time_series: pd.Series) -> Optional[np.ndarray]:
    """
    Convert an aux 'Time' column into seconds-from-start.

    Handles time-of-day strings that cross midnight (e.g., 22:00 -> 07:00)
    by unwrapping across 24h boundaries.
    """
    if time_series is None or len(time_series) == 0:
        return None

    # 1) numeric
    num = pd.to_numeric(time_series, errors="coerce").to_numpy(dtype=float)
    if np.isfinite(num).mean() > 0.9:
        # normalize to start at 0
        return num - np.nanmin(num)

    # 2) datetime parse
    dt = pd.to_datetime(time_series, errors="coerce")
    if dt.notna().mean() > 0.9:
        dt_valid = dt.dropna()

        # If all rows have the same normalized date, it's probably "time-of-day only"
        # (pandas picked a default date). Then unwrap midnight crossings.
        if dt_valid.dt.normalize().nunique() == 1:
            sec_of_day = (
                dt.dt.hour.to_numpy(dtype=float) * 3600.0
                + dt.dt.minute.to_numpy(dtype=float) * 60.0
                + dt.dt.second.to_numpy(dtype=float)
                + dt.dt.microsecond.to_numpy(dtype=float) * 1e-6
            )

            # unwrap: whenever time decreases, add 24h to subsequent samples
            out = sec_of_day.copy()
            valid_mask = np.isfinite(out)
            idx = np.where(valid_mask)[0]
            if idx.size == 0:
                return None

            for k in range(1, idx.size):
                i_prev = idx[k - 1]
                i_cur = idx[k]
                if out[i_cur] < out[i_prev] - 1e-6:  # crossed midnight
                    out[i_cur:] += 86400.0

            # normalize to start at 0 using first valid
            out -= out[idx[0]]
            return out

        # Otherwise, dt includes real dates -> safe to compute elapsed seconds normally
        t0 = dt.iloc[dt.first_valid_index()]
        return (dt - t0).dt.total_seconds().to_numpy(dtype=float)

    # 3) timedelta parse
    td = pd.to_timedelta(time_series, errors="coerce")
    if td.notna().mean() > 0.9:
        sec = td.dt.total_seconds().to_numpy(dtype=float)
        return sec - np.nanmin(sec)

    return None



def normalize_aux_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize aux CSV into canonical columns.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # ---- time axis
    raw_time = getattr(config, "AUX_CSV_TIME_COLUMN_RAW", "Time")
    time_sec = getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec")

    if raw_time in df.columns:
        sec = _parse_time_column_to_seconds(df[raw_time])
        if sec is None:
            sec = np.arange(len(df), dtype=float)
        df[time_sec] = sec
    else:
        df[time_sec] = np.arange(len(df), dtype=float)

    # ---- rename columns via COL_NAMES (internal -> csv)
    rename_map: Dict[str, str] = {}
    for internal, csv_name in getattr(config, "COL_NAMES", {}).items():
        if csv_name in df.columns:
            rename_map[csv_name] = internal
    df = df.rename(columns=rename_map)

    # ---- normalize flag-like columns
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

    # ---- stage_code
    stage_code_col = getattr(config, "AUX_CSV_STAGE_CODE_COLUMN", "stage_code")
    if "stage" in df.columns and stage_code_col not in df.columns:
        mapping = getattr(config, "SLEEP_MAPPING", {})
        mapping_upper = {str(k).strip().upper(): int(v) for k, v in mapping.items()}
        stage_norm = df["stage"].astype(str).str.strip().str.upper()
        df[stage_code_col] = stage_norm.map(mapping_upper).astype(float)

    return df


def read_aux_csv_for_edf(edf_path: Path) -> Optional[pd.DataFrame]:
    aux_csv = find_aux_csv_for_edf(edf_path)
    if aux_csv is None:
        return None

    df = read_raw_aux_csv(aux_csv)
    if df is None:
        return None

    return normalize_aux_df(df)


# =============================================================================
# Simple helpers
# =============================================================================

def get_event_times(
    aux_df: pd.DataFrame,
    event_col: str,
    time_col: Optional[str] = None,
) -> np.ndarray:
    if aux_df is None or len(aux_df) == 0:
        return np.array([], dtype=float)

    if time_col is None:
        time_col = getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec")

    if event_col not in aux_df.columns:
        return np.array([], dtype=float)

    m = aux_df[event_col] == 1
    return aux_df.loc[m, time_col].to_numpy(dtype=float)


# =============================================================================
# Desaturation windows
# =============================================================================

def _estimate_dt_sec(t: np.ndarray) -> float:
    d = np.diff(t[np.isfinite(t)])
    d = d[d > 0]
    return float(np.median(d)) if d.size else 1.0


def _flag_runs_to_windows(
    t_sec: np.ndarray,
    flag01: np.ndarray,
    *,
    min_run_sec: float,
    start_pad_sec: float,
    end_pad_sec: float,
) -> List[Tuple[float, float]]:
    idx = np.where(flag01 == 1)[0]
    if idx.size == 0:
        return []

    dt = _estimate_dt_sec(t_sec)

    cuts = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, cuts)

    windows: List[Tuple[float, float]] = []

    for g in groups:
        s = t_sec[g[0]]
        e = t_sec[g[-1]] + dt
        dur = e - s

        if dur >= min_run_sec:
            windows.append((s - start_pad_sec, e + end_pad_sec))
        else:
            # marker-style fallback
            for i in g:
                ti = t_sec[i]
                windows.append((ti - start_pad_sec, ti + end_pad_sec))

    return windows


def desat_windows_from_aux(aux_df: pd.DataFrame) -> List[Tuple[float, float]]:
    if aux_df is None or len(aux_df) == 0:
        return []

    time_col = getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec")
    key = getattr(config, "HRV_EXCLUSION_DESAT_COLUMN_KEY", "desat_flag")

    desat_col = key if key in aux_df.columns else "desat_flag"
    if desat_col not in aux_df.columns:
        return []

    start_pad = float(getattr(config, "HRV_EXCLUSION_DESAT_START_PAD_SEC", 0.0))
    end_pad = float(getattr(config, "HRV_EXCLUSION_DESAT_END_PAD_SEC", 0.0))
    min_run = float(getattr(config, "HRV_EXCLUSION_DESAT_MIN_RUN_SEC", 0.0))

    t = aux_df[time_col].to_numpy(dtype=float)
    f = pd.to_numeric(aux_df[desat_col], errors="coerce").fillna(0).to_numpy(dtype=int)

    windows = _flag_runs_to_windows(
        t_sec=t,
        flag01=f,
        min_run_sec=min_run,
        start_pad_sec=start_pad,
        end_pad_sec=end_pad,
    )

    if windows:
        lens = np.array([b - a for a, b in windows])
        print(
            f"  DESAT WINDOWS: n={len(windows)}, "
            f"median={np.median(lens):.1f}s, "
            f"p90={np.percentile(lens, 90):.1f}s"
        )
    else:
        print("  DESAT WINDOWS: n=0")

    return windows


# =============================================================================
# RR exclusion mask (FINAL)
# =============================================================================

def get_rr_exclusion_mask(
    rr_mid_times_sec: np.ndarray,
    aux_df: pd.DataFrame,
) -> np.ndarray:
    rr_mid_times_sec = np.asarray(rr_mid_times_sec, dtype=float)
    keep = np.ones_like(rr_mid_times_sec, dtype=bool)

    if aux_df is None or len(aux_df) == 0 or rr_mid_times_sec.size == 0:
        return keep

    time_col = getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec")

    # ----------------------------
    # 1) Event-based exclusion (always applies)
    # ----------------------------
    event_cols = getattr(config, "HRV_EXCLUSION_EVENT_COLUMNS", []) or []
    pre = float(getattr(config, "HRV_EXCLUSION_PRE_SEC", 0.0))
    post = float(getattr(config, "HRV_EXCLUSION_POST_SEC", 0.0))

    event_times_list = []
    for col in event_cols:
        if col in aux_df.columns:
            t = get_event_times(aux_df, col, time_col=time_col)
            if t.size > 0:
                event_times_list.append(t)

    event_times = np.unique(np.concatenate(event_times_list)) if event_times_list else np.array([], dtype=float)

    # Apply event exclusion windows
    for t in event_times:
        keep[(rr_mid_times_sec >= t - pre) & (rr_mid_times_sec <= t + post)] = False

    # ----------------------------
    # 2) Desat windows, but ONLY if there is an event near/inside them
    # ----------------------------
    if bool(getattr(config, "HRV_EXCLUSION_USE_DESAT_WINDOWS", False)):
        windows = desat_windows_from_aux(aux_df)

        if windows and event_times.size > 0:
            lookback = float(getattr(config, "HRV_EXCLUSION_DESAT_LOOKBACK_SEC", 120.0))
            lookahead = float(getattr(config, "HRV_EXCLUSION_DESAT_LOOKAHEAD_SEC", 120.0))

            # Keep only desat windows that have at least one event within [start-lookback, end+lookahead]
            gated = []
            event_times_sorted = np.sort(event_times)

            for a, b in windows:
                a = float(a); b = float(b)
                if not (np.isfinite(a) and np.isfinite(b) and b > a):
                    continue

                A = a - lookback
                B = b + lookahead

                # fast check: any event in [A, B]?
                i0 = np.searchsorted(event_times_sorted, A, side="left")
                i1 = np.searchsorted(event_times_sorted, B, side="right")
                if i1 > i0:
                    gated.append((a, b))

            # Apply ONLY gated desat windows
            if gated:
                gated = sorted(gated)
                starts = np.array([x for x, _ in gated], dtype=float)
                ends = np.array([y for _, y in gated], dtype=float)

                idx = np.searchsorted(starts, rr_mid_times_sec, side="right") - 1
                valid = idx >= 0
                in_win = np.zeros_like(keep)
                in_win[valid] = rr_mid_times_sec[valid] < ends[idx[valid]]
                keep[in_win] = False

                # optional debug
                n_exc = int(np.sum(~keep))
                print(
                    f"  HRV RR exclusion (EVENT+DESAT gated): excluded {n_exc}/{keep.size} RR "
                    f"({100*n_exc/max(1, keep.size):.1f}%) | gated_desat_windows={len(gated)}/{len(windows)}"
                )
            else:
                n_exc = int(np.sum(~keep))
                print(
                    f"  HRV RR exclusion (EVENT+DESAT gated): excluded {n_exc}/{keep.size} RR "
                    f"({100*n_exc/max(1, keep.size):.1f}%) | gated_desat_windows=0/{len(windows)}"
                )

        else:
            # If there are no events at all, desats do NOT exclude anything (your requested behavior)
            n_exc = int(np.sum(~keep))
            print(
                f"  HRV RR exclusion (EVENT+DESAT gated): excluded {n_exc}/{keep.size} RR "
                f"({100*n_exc/max(1, keep.size):.1f}%) | (no events -> desats ignored)"
            )

    return keep




def build_time_exclusion_mask(
    t_grid_sec: np.ndarray,
    aux_df: pd.DataFrame,
) -> Optional[np.ndarray]:
    """
    Mask on a regular time grid (typically t_hrv 1 Hz):
      True  = KEEP
      False = EXCLUDE (inside event or desat windows)

    Use this ONLY for plotting / forcing NaNs on the plotted series.
    """
    if aux_df is None or len(aux_df) == 0 or t_grid_sec is None or np.size(t_grid_sec) == 0:
        return None

    t = np.asarray(t_grid_sec, dtype=float)
    keep = np.ones_like(t, dtype=bool)

    # 1) event-based windows (pre/post)
    pre = float(getattr(config, "HRV_EXCLUSION_PRE_SEC", 0.0))
    post = float(getattr(config, "HRV_EXCLUSION_POST_SEC", 0.0))
    for col in getattr(config, "HRV_EXCLUSION_EVENT_COLUMNS", []):
        if col not in aux_df.columns:
            continue
        times = get_event_times(aux_df, col)
        for te in times:
            keep[(t >= te - pre) & (t <= te + post)] = False

    # 2) desat-run windows
    if bool(getattr(config, "HRV_EXCLUSION_USE_DESAT_WINDOWS", False)):
        windows = desat_windows_from_aux(aux_df)
        for a, b in windows:
            keep[(t >= float(a)) & (t < float(b))] = False

    return keep
