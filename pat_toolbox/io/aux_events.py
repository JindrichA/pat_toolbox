from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import config, paths


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


def _sorted_stage_timeline(aux_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    time_col = getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec")
    stage_col = getattr(config, "AUX_CSV_STAGE_CODE_COLUMN", "stage_code")
    if aux_df is None or len(aux_df) == 0:
        return np.array([], dtype=float), np.array([], dtype=int)
    if time_col not in aux_df.columns or stage_col not in aux_df.columns:
        return np.array([], dtype=float), np.array([], dtype=int)

    t = aux_df[time_col].to_numpy(dtype=float)
    s = aux_df[stage_col].to_numpy(dtype=float)
    ok = np.isfinite(t) & np.isfinite(s)
    if not np.any(ok):
        return np.array([], dtype=float), np.array([], dtype=int)

    t = t[ok]
    s = np.round(s[ok]).astype(int)
    order = np.argsort(t)
    t = t[order]
    s = s[order]
    if t.size >= 2:
        keep = np.ones_like(t, dtype=bool)
        keep[:-1] = t[1:] != t[:-1]
        t = t[keep]
        s = s[keep]
    if t.size == 0:
        return np.array([], dtype=float), np.array([], dtype=int)

    dt = np.diff(t)
    wrap_idx = np.where(dt < -12 * 3600)[0]
    if wrap_idx.size > 0:
        t2 = t.copy()
        for i in wrap_idx:
            t2[i + 1 :] += 24 * 3600
        t = t2

    return t, s


def _hhmm_from_rel_sec(sec: float) -> str:
    if not np.isfinite(sec):
        return ""
    total_min = int(round(sec / 60.0))
    hh = total_min // 60
    mm = total_min % 60
    return f"{hh:02d}:{mm:02d}"


def compute_sleep_timing_from_aux(aux_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if aux_df is None or len(aux_df) == 0:
        return None

    if hasattr(aux_df, "attrs"):
        cached = aux_df.attrs.get("_sleep_timing_cache", None)
        if cached is not None:
            return cached

    t, s = _sorted_stage_timeline(aux_df)
    if t.size == 0 or s.size == 0:
        return None

    sleep_mask = np.isin(s, [1, 2, 3])
    if not np.any(sleep_mask):
        return None

    dt_sec = _estimate_dt_sec(t)
    start_sec = float(t[0])
    onset_sec = float(t[np.flatnonzero(sleep_mask)[0]])
    end_sec = float(t[np.flatnonzero(sleep_mask)[-1]] + dt_sec)
    midpoint_sec = 0.5 * (onset_sec + end_sec)

    out: Dict[str, Any] = {
        "recording_start_sec": start_sec,
        "sleep_onset_sec": onset_sec,
        "sleep_end_sec": end_sec,
        "sleep_midpoint_sec": midpoint_sec,
        "sleep_onset_rel_sec": float(onset_sec - start_sec),
        "sleep_end_rel_sec": float(end_sec - start_sec),
        "sleep_midpoint_rel_sec": float(midpoint_sec - start_sec),
    }
    out["sleep_onset_rel_h"] = out["sleep_onset_rel_sec"] / 3600.0
    out["sleep_end_rel_h"] = out["sleep_end_rel_sec"] / 3600.0
    out["sleep_midpoint_rel_h"] = out["sleep_midpoint_rel_sec"] / 3600.0
    out["sleep_onset_rel_hhmm"] = _hhmm_from_rel_sec(out["sleep_onset_rel_sec"])
    out["sleep_end_rel_hhmm"] = _hhmm_from_rel_sec(out["sleep_end_rel_sec"])
    out["sleep_midpoint_rel_hhmm"] = _hhmm_from_rel_sec(out["sleep_midpoint_rel_sec"])

    if hasattr(aux_df, "attrs"):
        aux_df.attrs["_sleep_timing_cache"] = out
    return out


def save_sleep_timing_to_csv(edf_path: Path, aux_df: pd.DataFrame) -> Optional[Path]:
    timing = compute_sleep_timing_from_aux(aux_df)
    if not timing:
        return None

    out_folder = paths.get_output_folder("SleepTiming")
    out_csv = out_folder / f"{edf_path.stem}__Sleep_Timing.csv"
    fieldnames = [
        "edf_file",
        "sleep_onset_rel_sec",
        "sleep_onset_rel_h",
        "sleep_onset_rel_hhmm",
        "sleep_end_rel_sec",
        "sleep_end_rel_h",
        "sleep_end_rel_hhmm",
        "sleep_midpoint_rel_sec",
        "sleep_midpoint_rel_h",
        "sleep_midpoint_rel_hhmm",
    ]
    row = {"edf_file": edf_path.name, **{k: timing.get(k, "") for k in fieldnames if k != "edf_file"}}
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    return out_csv


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
