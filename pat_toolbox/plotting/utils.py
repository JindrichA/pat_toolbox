from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING, List

import numpy as np
import matplotlib.pyplot as plt

from .. import config, io_aux_csv

if TYPE_CHECKING:
    import pandas as pd


def _nan_pct(x: Optional[np.ndarray]) -> Optional[float]:
    """Return percent NaN/Inf in array. None -> None."""
    if x is None:
        return None
    x = np.asarray(x)
    if x.size == 0:
        return None
    return float(100.0 * np.mean(~np.isfinite(x)))


def _infer_edf_base(pdf_path: Path) -> str:
    try:
        return pdf_path.stem.split("__")[0]
    except Exception:
        return pdf_path.stem


def _fmt(x: Optional[float], ndigits: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x:.{ndigits}f}"


def _shade_masked_regions(
    ax: plt.Axes,
    t_sec: np.ndarray,
    masked: np.ndarray,
    *,
    color: str = "0.5",
    alpha: float = 0.18,
    zorder: float = 0.05,  # keep this BELOW exclusion spans
):
    """
    Shade contiguous masked (True) regions along time axis.
    t_sec is in seconds.
    """
    if t_sec is None or masked is None:
        return
    if len(t_sec) == 0 or not np.any(masked):
        return

    idx = np.where(masked)[0]
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)

    for g in groups:
        if g.size == 0:
            continue
        x0 = t_sec[g[0]] / 3600.0
        x1 = (t_sec[g[-1]] + 1.0) / 3600.0  # +1s for visibility
        ax.axvspan(x0, x1, color=color, alpha=alpha, zorder=zorder)


def _maybe_add_legend(ax: plt.Axes, *args, **kwargs) -> None:
    """
    Call legend() only if there is at least one artist with a usable label.
    Prevents: 'No artists with labels found to put in legend.'
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    usable = [
        (h, lab) for h, lab in zip(handles, labels)
        if lab and (not str(lab).startswith("_"))
    ]
    if not usable:
        return

    h2, l2 = zip(*usable)
    ax.legend(h2, l2, *args, **kwargs)


def _count_flags(aux_df: Optional["pd.DataFrame"], col: str) -> tuple[int, str]:
    """Return (count, percentage_string) for a given aux column."""
    if aux_df is None or col not in aux_df.columns:
        return 0, "0.0%"
    total = len(aux_df)
    if total <= 0:
        return 0, "0.0%"
    c = int(aux_df[col].fillna(0).astype(int).sum())
    pct = 100.0 * c / total
    return c, f"{pct:.1f}%"


def _compute_exclusion_zones(aux_df: Optional["pd.DataFrame"]) -> List[Tuple[float, float, str]]:
    zones: List[Tuple[float, float, str]] = []
    if aux_df is None:
        return zones

    # Event zones (pre/post)
    hrv_event_cols = getattr(config, "HRV_EXCLUSION_EVENT_COLUMNS", None) or []
    pre_sec = float(getattr(config, "HRV_EXCLUSION_PRE_SEC", 0.0))
    post_sec = float(getattr(config, "HRV_EXCLUSION_POST_SEC", 0.0))
    time_col = getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec")

    event_times_set = set()
    for col in hrv_event_cols:
        if col in aux_df.columns:
            times = io_aux_csv.get_event_times(aux_df, col, time_col=time_col)
            event_times_set.update(times)

    for t_event in sorted(event_times_set):
        a = float(t_event - pre_sec)
        b = float(t_event + post_sec)
        if b > a:
            zones.append((a, b, "HRV Exclusion (events)"))

    # Desat-run zones — ONLY if associated with an event
    if bool(getattr(config, "HRV_EXCLUSION_USE_DESAT_WINDOWS", False)):
        windows = io_aux_csv.desat_windows_from_aux(aux_df)

        # need events to gate desats
        if windows and event_times_set:
            event_times = np.array(sorted(event_times_set), dtype=float)

            lookback = float(getattr(config, "HRV_EXCLUSION_DESAT_LOOKBACK_SEC", 120.0))
            lookahead = float(getattr(config, "HRV_EXCLUSION_DESAT_LOOKAHEAD_SEC", 120.0))

            for a, b in windows:
                if not (np.isfinite(a) and np.isfinite(b) and b > a):
                    continue

                # expanded window for association test
                A = a - lookback
                B = b + lookahead

                # any event time inside [A, B]?
                i0 = np.searchsorted(event_times, A, side="left")
                i1 = np.searchsorted(event_times, B, side="right")

                if i1 > i0:
                    zones.append((float(a), float(b), "HRV Exclusion (event+desat)"))

    zones.sort(key=lambda x: x[0])
    print(f"  Calculated {len(zones)} HRV exclusion zone(s).")
    return zones


def _add_exclusion_spans(
    ax: plt.Axes,
    exclusion_zones: List[Tuple[float, float, str]],
    start_h: float,
    end_h: float,
    label_once: bool = True,
) -> None:
    first = True
    for t_start_sec, t_end_sec, label in exclusion_zones:
        t_start_h = t_start_sec / 3600.0
        t_end_h = t_end_sec / 3600.0
        if max(t_start_h, start_h) < min(t_end_h, end_h):
            ax.axvspan(
                t_start_h,
                t_end_h,
                facecolor="red",
                alpha=0.15,
                zorder=0.20,  # draw ABOVE the gray NaN shading
                label=label if (label_once and first) else "_nolegend_",
            )
            first = False


def _h_to_hhmm(h: float) -> str:
    if not np.isfinite(h):
        return "NA"
    total_min = int(round(h * 60.0))
    hh = total_min // 60
    mm = total_min % 60
    return f"{hh:02d}:{mm:02d}"
