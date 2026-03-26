from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import config, sleep_mask


@dataclass(frozen=True)
class MaskPolicy:
    sleep_enabled: bool
    include_stages: frozenset[int]
    exclusion_columns: tuple[str, ...]
    use_desat_windows: bool
    event_pre_sec: float
    event_post_sec: float
    desat_column: str
    desat_start_pad_sec: float
    desat_end_pad_sec: float
    desat_min_run_sec: float
    desat_lookback_sec: float
    desat_lookahead_sec: float


@dataclass
class MaskBundle:
    t_sec: np.ndarray
    sleep_keep: np.ndarray
    event_keep: np.ndarray
    desat_keep: np.ndarray
    combined_keep: np.ndarray
    active_exclusion_columns: tuple[str, ...]
    active_event_times_sec: np.ndarray
    gated_desat_windows: tuple[tuple[float, float], ...]


def policy_from_config(
    *,
    include_stages: Optional[set[int]] = None,
    force_sleep: Optional[bool] = None,
) -> MaskPolicy:
    sleep_enabled = bool(getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False))
    if force_sleep is not None:
        sleep_enabled = bool(force_sleep)

    stages = include_stages
    if stages is None:
        try:
            stages = set(config.sleep_include_numeric())
        except Exception:
            stages = {1, 2, 3}

    exclusion_columns = tuple(
        str(col)
        for col in (getattr(config, "HRV_EXCLUSION_EVENT_COLUMNS", []) or [])
        if str(col)
    )
    return MaskPolicy(
        sleep_enabled=sleep_enabled,
        include_stages=frozenset(int(x) for x in stages),
        exclusion_columns=exclusion_columns,
        use_desat_windows=bool(getattr(config, "HRV_EXCLUSION_USE_DESAT_WINDOWS", False)),
        event_pre_sec=float(getattr(config, "HRV_EXCLUSION_PRE_SEC", 0.0)),
        event_post_sec=float(getattr(config, "HRV_EXCLUSION_POST_SEC", 0.0)),
        desat_column=str(getattr(config, "HRV_EXCLUSION_DESAT_COLUMN_KEY", "desat_flag")),
        desat_start_pad_sec=float(getattr(config, "HRV_EXCLUSION_DESAT_START_PAD_SEC", 0.0)),
        desat_end_pad_sec=float(getattr(config, "HRV_EXCLUSION_DESAT_END_PAD_SEC", 0.0)),
        desat_min_run_sec=float(getattr(config, "HRV_EXCLUSION_DESAT_MIN_RUN_SEC", 0.0)),
        desat_lookback_sec=float(getattr(config, "HRV_EXCLUSION_DESAT_LOOKBACK_SEC", 120.0)),
        desat_lookahead_sec=float(getattr(config, "HRV_EXCLUSION_DESAT_LOOKAHEAD_SEC", 120.0)),
    )


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
) -> list[tuple[float, float]]:
    idx = np.where(flag01 == 1)[0]
    if idx.size == 0:
        return []

    dt = _estimate_dt_sec(t_sec)
    cuts = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, cuts)
    windows: list[tuple[float, float]] = []
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


def _desat_windows_from_aux(aux_df, policy: MaskPolicy) -> list[tuple[float, float]]:
    if aux_df is None or len(aux_df) == 0:
        return []

    cache_key = (
        "_desat_windows_cache",
        policy.desat_column,
        policy.desat_start_pad_sec,
        policy.desat_end_pad_sec,
        policy.desat_min_run_sec,
    )
    if hasattr(aux_df, "attrs"):
        cached = aux_df.attrs.get(cache_key, None)
        if cached is not None:
            return cached

    time_col = getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec")
    desat_col = policy.desat_column if policy.desat_column in aux_df.columns else "desat_flag"
    if desat_col not in aux_df.columns or time_col not in aux_df.columns:
        return []

    t = aux_df[time_col].to_numpy(dtype=float)
    f = np.nan_to_num(aux_df[desat_col].to_numpy(dtype=float), nan=0.0).astype(int)
    windows = _flag_runs_to_windows(
        t_sec=t,
        flag01=f,
        min_run_sec=policy.desat_min_run_sec,
        start_pad_sec=policy.desat_start_pad_sec,
        end_pad_sec=policy.desat_end_pad_sec,
    )
    if hasattr(aux_df, "attrs"):
        aux_df.attrs[cache_key] = windows
    return windows


def _event_times_from_aux(aux_df, columns: tuple[str, ...]) -> np.ndarray:
    if aux_df is None or len(aux_df) == 0:
        return np.array([], dtype=float)
    time_col = getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec")
    if time_col not in aux_df.columns:
        return np.array([], dtype=float)

    times: list[np.ndarray] = []
    for col in columns:
        if col not in aux_df.columns:
            continue
        m = np.nan_to_num(aux_df[col].to_numpy(dtype=float), nan=0.0).astype(int) == 1
        if np.any(m):
            times.append(aux_df.loc[m, time_col].to_numpy(dtype=float))
    if not times:
        return np.array([], dtype=float)
    return np.unique(np.concatenate(times))


def _gated_desat_windows(aux_df, policy: MaskPolicy, active_event_times: np.ndarray) -> tuple[tuple[float, float], ...]:
    if not policy.use_desat_windows:
        return ()
    windows = _desat_windows_from_aux(aux_df, policy)
    if not windows or active_event_times.size == 0:
        return ()

    gated: list[tuple[float, float]] = []
    event_times_sorted = np.sort(active_event_times)
    for a, b in windows:
        if not (np.isfinite(a) and np.isfinite(b) and b > a):
            continue
        A = a - policy.desat_lookback_sec
        B = b + policy.desat_lookahead_sec
        i0 = np.searchsorted(event_times_sorted, A, side="left")
        i1 = np.searchsorted(event_times_sorted, B, side="right")
        if i1 > i0:
            gated.append((float(a), float(b)))
    return tuple(gated)


def build_mask_bundle(
    t_sec: np.ndarray,
    aux_df,
    *,
    policy: Optional[MaskPolicy] = None,
) -> MaskBundle:
    tt = np.asarray(t_sec, dtype=float)
    sleep_keep = np.ones_like(tt, dtype=bool)
    event_keep = np.ones_like(tt, dtype=bool)
    desat_keep = np.ones_like(tt, dtype=bool)

    if policy is None:
        policy = policy_from_config()

    if tt.size == 0:
        return MaskBundle(
            t_sec=tt,
            sleep_keep=sleep_keep,
            event_keep=event_keep,
            desat_keep=desat_keep,
            combined_keep=sleep_keep,
            active_exclusion_columns=policy.exclusion_columns,
            active_event_times_sec=np.array([], dtype=float),
            gated_desat_windows=(),
        )

    if aux_df is not None and policy.sleep_enabled:
        m_sleep = sleep_mask.build_sleep_include_mask_for_times(
            tt,
            aux_df,
            include_set=set(policy.include_stages),
            ignore_config=True,
        )
        if m_sleep is not None:
            sleep_keep = np.asarray(m_sleep, dtype=bool)

    active_event_times = _event_times_from_aux(aux_df, policy.exclusion_columns)
    for te in active_event_times:
        event_keep[(tt >= te - policy.event_pre_sec) & (tt <= te + policy.event_post_sec)] = False

    gated_desats = _gated_desat_windows(aux_df, policy, active_event_times)
    for a, b in gated_desats:
        desat_keep[(tt >= a) & (tt < b)] = False

    combined_keep = sleep_keep & event_keep & desat_keep
    return MaskBundle(
        t_sec=tt,
        sleep_keep=sleep_keep,
        event_keep=event_keep,
        desat_keep=desat_keep,
        combined_keep=combined_keep,
        active_exclusion_columns=policy.exclusion_columns,
        active_event_times_sec=active_event_times,
        gated_desat_windows=gated_desats,
    )


def build_rr_mask_bundle(
    rr_mid_times_sec: np.ndarray,
    aux_df,
    *,
    policy: Optional[MaskPolicy] = None,
) -> MaskBundle:
    return build_mask_bundle(rr_mid_times_sec, aux_df, policy=policy)


def false_runs_from_mask(t_sec: np.ndarray, keep_mask: np.ndarray) -> list[tuple[float, float]]:
    t_sec = np.asarray(t_sec, dtype=float)
    keep_mask = np.asarray(keep_mask, dtype=bool)
    if t_sec.size == 0 or keep_mask.size != t_sec.size:
        return []

    bad = ~keep_mask
    if not np.any(bad):
        return []

    idx = np.where(bad)[0]
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    dt = _estimate_dt_sec(t_sec)
    runs: list[tuple[float, float]] = []
    for g in groups:
        if g.size == 0:
            continue
        x0 = float(t_sec[g[0]])
        x1 = float(t_sec[g[-1]] + dt)
        if x1 > x0:
            runs.append((x0, x1))
    return runs


__all__ = [
    "MaskPolicy",
    "MaskBundle",
    "policy_from_config",
    "build_mask_bundle",
    "build_rr_mask_bundle",
    "false_runs_from_mask",
]
