from __future__ import annotations

from typing import Optional, TYPE_CHECKING, Any
import numpy as np

if TYPE_CHECKING:
    import pandas as pd
else:  # pragma: no cover
    pd = Any

from . import config


FIXED_SLEEP_STAGE_SETS = {
    "all_sleep": {1, 2, 3},
    "wake_sleep": {0, 1, 2, 3},
    "nrem": {1, 2},
    "deep": {2},
    "rem": {3},
}

FIXED_SLEEP_STAGE_LABELS = {
    "all_sleep": "All sleep",
    "wake_sleep": "Wake+sleep",
    "nrem": "NREM",
    "deep": "Deep",
    "rem": "REM only",
}


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _to_float_or_nan(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def fixed_sleep_stage_policies() -> list[tuple[str, str, set[int]]]:
    return [
        (key, FIXED_SLEEP_STAGE_LABELS[key], set(FIXED_SLEEP_STAGE_SETS[key]))
        for key in ["all_sleep", "wake_sleep", "nrem", "deep", "rem"]
    ]


def _resolve_include_set(include_set: Optional[set[int]] = None) -> set[int]:
    if include_set is not None:
        return {int(x) for x in include_set}
    return {int(x) for x in config.sleep_include_numeric()}


def _build_mask_from_stage_arrays(
    t_sec: np.ndarray,
    aux_t: np.ndarray,
    aux_s: np.ndarray,
    include_set: set[int],
) -> Optional[np.ndarray]:
    tt = np.asarray(t_sec, dtype=float)
    if tt.size == 0:
        return None

    ok = np.isfinite(aux_t) & np.isfinite(aux_s)
    if not np.any(ok):
        return None

    aux_t = np.asarray(aux_t[ok], dtype=float)
    aux_s = np.round(np.asarray(aux_s[ok], dtype=float)).astype(int)

    order = np.argsort(aux_t)
    aux_t = aux_t[order]
    aux_s = aux_s[order]

    idx = np.searchsorted(aux_t, tt, side="left")
    idx0 = np.clip(idx - 1, 0, len(aux_t) - 1)
    idx1 = np.clip(idx, 0, len(aux_t) - 1)

    d0 = np.abs(tt - aux_t[idx0])
    d1 = np.abs(tt - aux_t[idx1])
    pick = np.where(d1 < d0, idx1, idx0)
    return np.isin(aux_s[pick], list(include_set))


# -----------------------------------------------------------------------------
# stage handling
# -----------------------------------------------------------------------------

def ensure_stage_code_column(aux_df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Ensure aux_df contains numeric sleep stage codes in
    config.AUX_CSV_STAGE_CODE_COLUMN (default: 'stage_code').

    If stage info is missing, aux_df is returned unchanged.
    """
    if aux_df is None:
        return aux_df

    time_col = getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec")
    if time_col not in aux_df.columns:
        return aux_df

    stage_code_col = getattr(config, "AUX_CSV_STAGE_CODE_COLUMN", "stage_code")
    if stage_code_col in aux_df.columns:
        return aux_df

    raw_stage_col = None
    try:
        raw_stage_col = config.COL_NAMES.get("stage", None)
    except Exception:
        raw_stage_col = None

    if raw_stage_col is None or raw_stage_col not in aux_df.columns:
        return aux_df

    mapping = getattr(config, "SLEEP_MAPPING", {}) or {}

    s = aux_df[raw_stage_col].astype(str).str.strip()
    stage_num = np.full(len(aux_df), np.nan, dtype=float)

    # numeric parse first
    num = np.array([_to_float_or_nan(x) for x in s.to_list()], dtype=float)
    numeric_ok = np.isfinite(num)
    stage_num[numeric_ok] = num[numeric_ok]

    # mapping for string labels
    for lab, code in mapping.items():
        m = (~numeric_ok) & (s == str(lab))
        if np.any(m):
            stage_num[m] = float(code)

    aux_df[stage_code_col] = stage_num
    return aux_df


# -----------------------------------------------------------------------------
# include masks (sleep stage)
# -----------------------------------------------------------------------------

def build_sleep_include_mask(
    t_sec: np.ndarray,
    aux_df: Optional["pd.DataFrame"],
    include_set: Optional[set[int]] = None,
) -> Optional[np.ndarray]:
    """
    Build boolean mask on a regular time grid (typically 1 Hz).

    Returns:
      - None if masking disabled or stage info missing
      - boolean array same length as t_sec otherwise
    """
    if aux_df is None:
        return None

    if not bool(getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False)):
        return None

    time_col = getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec")
    stage_col = getattr(config, "AUX_CSV_STAGE_CODE_COLUMN", "stage_code")

    if time_col not in aux_df.columns:
        return None

    aux_df = ensure_stage_code_column(aux_df)
    if stage_col not in aux_df.columns:
        return None

    aux_t = aux_df[time_col].to_numpy(dtype=float)
    aux_s = aux_df[stage_col].to_numpy(dtype=float)

    return _build_mask_from_stage_arrays(
        t_sec,
        aux_t,
        aux_s,
        _resolve_include_set(include_set),
    )


def build_sleep_include_mask_for_times(
    t_sec: np.ndarray,
    aux_df: Optional["pd.DataFrame"],
    include_set: Optional[set[int]] = None,
    *,
    ignore_config: bool = False,
) -> Optional[np.ndarray]:
    """
    Same as build_sleep_include_mask(), but for arbitrary times
    (e.g. RR mid-times).

    Behavior:
      - masking OFF  -> all True
      - missing aux  -> None
    """
    if t_sec is None or np.size(t_sec) == 0:
        return None

    enabled = bool(getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False)) or bool(ignore_config)
    if not enabled:
        return np.ones_like(np.asarray(t_sec), dtype=bool)

    if aux_df is None:
        return None

    time_col = getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec")
    stage_col = getattr(config, "AUX_CSV_STAGE_CODE_COLUMN", "stage_code")

    if time_col not in aux_df.columns:
        return None

    aux_df = ensure_stage_code_column(aux_df)
    if stage_col not in aux_df.columns:
        return None

    t_aux = aux_df[time_col].to_numpy(dtype=float)
    s_aux = aux_df[stage_col].to_numpy(dtype=float)
    return _build_mask_from_stage_arrays(
        t_sec,
        t_aux,
        s_aux,
        _resolve_include_set(include_set),
    )


def compute_sleep_hours_from_aux(
    aux_df: Optional["pd.DataFrame"],
    include_set: Optional[set[int]] = None,
) -> float:
    """
    Estimate included sleep hours directly from the aux stage timeline.

    This is useful for summaries that should report sleep hours even when other
    features such as PAT burden are disabled.
    """
    if aux_df is None:
        return float("nan")

    time_col = getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec")
    stage_col = getattr(config, "AUX_CSV_STAGE_CODE_COLUMN", "stage_code")
    if time_col not in aux_df.columns:
        return float("nan")

    aux_df = ensure_stage_code_column(aux_df)
    if stage_col not in aux_df.columns:
        return float("nan")

    t_aux = aux_df[time_col].to_numpy(dtype=float)
    s_aux = aux_df[stage_col].to_numpy(dtype=float)
    ok = np.isfinite(t_aux) & np.isfinite(s_aux)
    if np.count_nonzero(ok) < 2:
        return float("nan")

    t_aux = t_aux[ok]
    s_aux = np.round(s_aux[ok]).astype(int)
    order = np.argsort(t_aux)
    t_aux = t_aux[order]
    s_aux = s_aux[order]

    include = np.isin(s_aux, list(_resolve_include_set(include_set)))
    dt = np.diff(t_aux)
    dt = np.clip(dt, 0.0, None)
    if dt.size == 0:
        return float("nan")

    keep_interval = include[:-1] & include[1:]
    return float(np.sum(dt[keep_interval]) / 3600.0)


# -----------------------------------------------------------------------------
# apply helper
# -----------------------------------------------------------------------------

def apply_sleep_mask_inplace(
    y: Optional[np.ndarray],
    include_mask: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """
    Set samples to NaN where include_mask is False.
    """
    if y is None or include_mask is None:
        return y

    y = np.asarray(y)
    if y.size != include_mask.size:
        return y

    bad = ~include_mask
    if np.any(bad):
        y[bad] = np.nan

    return y


def build_global_include_mask_for_times(
    t_sec: np.ndarray,
    aux_df: Optional["pd.DataFrame"],
    *,
    apply_sleep: bool = True,
    apply_events: bool = True,
) -> Optional[np.ndarray]:
    """
    Returns boolean include mask (True = keep) that combines:
      - sleep-stage include mask
      - event exclusion include mask
    """
    if t_sec is None or np.size(t_sec) == 0:
        return None

    keep = np.ones_like(np.asarray(t_sec, dtype=float), dtype=bool)

    if aux_df is None:
        return keep

    # sleep
    if apply_sleep and bool(getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False)):
        m_sleep = build_sleep_include_mask_for_times(t_sec, aux_df)
        if m_sleep is None:
            return None
        keep &= m_sleep

    # events
    if apply_events:
        try:
            from . import io_aux_csv
            m_evt = io_aux_csv.build_time_exclusion_mask(t_sec, aux_df)
        except Exception:
            m_evt = None
        if m_evt is None:
            return None
        keep &= np.asarray(m_evt, dtype=bool)

    return keep



# -----------------------------------------------------------------------------
# public API
# -----------------------------------------------------------------------------

__all__ = [
    "ensure_stage_code_column",
    "build_sleep_include_mask",
    "build_sleep_include_mask_for_times",
    "apply_sleep_mask_inplace",
]
