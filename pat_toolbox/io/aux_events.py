from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import config


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
            for i in g:
                ti = t_sec[i]
                windows.append((ti - start_pad_sec, ti + end_pad_sec))

    return windows


def desat_windows_from_aux(aux_df: pd.DataFrame) -> List[Tuple[float, float]]:
    if aux_df is None or len(aux_df) == 0:
        return []

    if hasattr(aux_df, "attrs"):
        cached = aux_df.attrs.get("_desat_windows_cache", None)
        if cached is not None:
            return cached

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

    if hasattr(aux_df, "attrs"):
        aux_df.attrs["_desat_windows_cache"] = windows

    do_print = True
    if hasattr(aux_df, "attrs"):
        if aux_df.attrs.get("_desat_windows_logged", False):
            do_print = False
        else:
            aux_df.attrs["_desat_windows_logged"] = True

    if do_print:
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


def get_rr_exclusion_mask(
    rr_mid_times_sec: np.ndarray,
    aux_df: pd.DataFrame,
) -> np.ndarray:
    from .. import masking

    rr_mid_times_sec = np.asarray(rr_mid_times_sec, dtype=float)
    bundle = masking.build_rr_mask_bundle(rr_mid_times_sec, aux_df)
    keep = np.asarray(bundle.combined_keep, dtype=bool)

    if bool(getattr(config, "HRV_EXCLUSION_USE_DESAT_WINDOWS", False)):
        n_exc = int(np.sum(~keep))
        if bundle.gated_desat_windows and bundle.active_event_times_sec.size > 0:
            print(
                f"  HRV RR exclusion (EVENT+DESAT gated): excluded {n_exc}/{keep.size} RR "
                f"({100*n_exc/max(1, keep.size):.1f}%) | "
                f"gated_desat_windows={len(bundle.gated_desat_windows)}"
            )
        else:
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

    IMPORTANT: This MUST match get_rr_exclusion_mask() logic,
    including EVENT-gated desat windows.
    """
    if aux_df is None or len(aux_df) == 0 or t_grid_sec is None or np.size(t_grid_sec) == 0:
        return None

    from .. import masking

    bundle = masking.build_mask_bundle(np.asarray(t_grid_sec, dtype=float), aux_df)
    return np.asarray(bundle.event_keep & bundle.desat_keep, dtype=bool)


def build_event_exclusion_mask(
    t_grid_sec: np.ndarray,
    aux_df: pd.DataFrame,
) -> Optional[np.ndarray]:
    """
    Mask on a regular time grid:
      True  = KEEP
      False = EXCLUDE (inside EVENT windows only, using HRV_EXCLUSION_EVENT_COLUMNS + PRE/POST)

    This is used for plotting (RED shading).
    """
    if aux_df is None or len(aux_df) == 0 or t_grid_sec is None or np.size(t_grid_sec) == 0:
        return None

    from .. import masking

    bundle = masking.build_mask_bundle(np.asarray(t_grid_sec, dtype=float), aux_df)
    return np.asarray(bundle.event_keep, dtype=bool)
