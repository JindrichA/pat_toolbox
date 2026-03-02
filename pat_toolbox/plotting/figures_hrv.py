from __future__ import annotations

from typing import Optional, Tuple, Dict, TYPE_CHECKING, List
import numpy as np
import matplotlib.pyplot as plt

from .. import config
from .specs import EventSpec, DEFAULT_EVENT_PLOT_SPEC
from .utils import _add_exclusion_spans, _shade_masked_regions, _maybe_add_legend

if TYPE_CHECKING:
    import pandas as pd


def _overlay_events_on_single_axis_whole_night(
    ax: plt.Axes,
    aux_df: Optional["pd.DataFrame"],
    start_sec: float,
    end_sec: float,
    event_spec: List[EventSpec] = DEFAULT_EVENT_PLOT_SPEC,
) -> None:
    if aux_df is None:
        return

    time_col = getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec")
    if time_col not in aux_df.columns:
        return

    mask = (aux_df[time_col] >= start_sec) & (aux_df[time_col] <= end_sec)
    if not mask.any():
        return

    seg = aux_df.loc[mask]
    used = set()

    for spec in event_spec:
        if spec.col not in seg.columns:
            continue

        m = seg[spec.col] == 1
        if not m.any():
            continue

        t_evt_h = seg.loc[m, time_col].to_numpy(float) / 3600.0
        show_label = spec.label if spec.label not in used else "_nolegend_"
        used.add(spec.label)

        if spec.col == "desat_flag":
            y_min, y_max = ax.get_ylim()
            y_desat = y_min + 0.03 * (y_max - y_min)
            ax.scatter(
                t_evt_h,
                np.full_like(t_evt_h, y_desat, dtype=float),
                marker="v",
                s=25,
                color=spec.color,
                alpha=0.9,
                label=show_label,
                zorder=4,
            )
        else:
            first_line = (show_label != "_nolegend_")
            for x in t_evt_h:
                ax.axvline(
                    x,
                    color=spec.color,
                    linestyle="-",
                    linewidth=1.6,
                    alpha=0.9,
                    label=spec.label if first_line else "_nolegend_",
                    zorder=4,
                )
                first_line = False


def _build_hrv_overview_figure(
    edf_base: str,
    t_hrv: np.ndarray,
    hrv_clean: np.ndarray,
    hrv_raw: Optional[np.ndarray],
    aux_df: Optional["pd.DataFrame"],
    exclusion_zones: List[Tuple[float, float, str]],
    duration_sec_fallback: float,
    event_spec: List[EventSpec] = DEFAULT_EVENT_PLOT_SPEC,
) -> Optional[plt.Figure]:
    if t_hrv is None or hrv_clean is None or t_hrv.size == 0:
        return None

    overview_hours = getattr(config, "OVERVIEW_PANEL_HOURS", 2.0)
    panel_sec = overview_hours * 3600.0

    duration_hrv_sec = float(t_hrv[-1]) if float(t_hrv[-1]) > 0 else duration_sec_fallback
    n_panels = max(1, int(np.ceil(duration_hrv_sec / panel_sec)))

    fig, axes = plt.subplots(n_panels, 1, figsize=(11.69, 8.27), sharex=False)
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(f"{edf_base} - HRV overview ({overview_hours:.1f} h panels)", fontsize=12)

    time_col = getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec") if aux_df is not None else None
    use_raw = hrv_raw is not None and np.size(hrv_raw) == np.size(hrv_clean)

    for p, ax in enumerate(axes):
        start_sec = p * panel_sec
        end_sec = min((p + 1) * panel_sec, duration_hrv_sec)

        start_h = start_sec / 3600.0
        end_h = end_sec / 3600.0

        ax.set_title(f"{start_h:.2f}–{end_h:.2f} h", fontsize=9, loc="left")
        ax.set_xlim(start_h, end_h)

        _add_exclusion_spans(ax, exclusion_zones, start_h, end_h, label_once=True)

        mask = (t_hrv >= start_sec) & (t_hrv <= end_sec)
        if np.any(mask):
            th = t_hrv[mask] / 3600.0

            yr = None
            okr = None

            if use_raw:
                yr = hrv_raw[mask]
                okr = np.isfinite(yr)
                if np.any(okr):
                    ax.plot(
                        th[okr],
                        yr[okr],
                        label="Raw HRV (Unmasked)",
                        linewidth=0.5,
                        color="tab:gray",
                        alpha=0.6,
                        zorder=1,
                    )

            yc = hrv_clean[mask]

            masked = ~np.isfinite(yc)
            _shade_masked_regions(
                ax,
                t_sec=t_hrv[mask],
                masked=masked,
                color="0.6",
                alpha=0.20,
            )

            okc = np.isfinite(yc)
            if np.any(okc):
                ax.plot(
                    th[okc],
                    yc[okc],
                    label="Clean HRV (Masked)",
                    linewidth=0.9,
                    zorder=2,
                )
            else:
                ax.text(
                    0.01,
                    0.92,
                    "HRV clean: no finite samples (all NaN after masking)",
                    transform=ax.transAxes,
                    fontsize=9,
                    va="top",
                )
                if yr is not None and okr is not None and np.any(okr):
                    y0 = float(np.nanmin(yr[okr]))
                    y1 = float(np.nanmax(yr[okr]))
                    if np.isfinite(y0) and np.isfinite(y1) and y1 > y0:
                        m = 0.10 * (y1 - y0)
                        ax.set_ylim(y0 - m, y1 + m)

        if aux_df is not None and time_col and time_col in aux_df.columns:
            aux_mask = (aux_df[time_col] >= start_sec) & (aux_df[time_col] <= end_sec)
            if aux_mask.any():
                seg = aux_df.loc[aux_mask]
                used = set()

                for spec in event_spec:
                    if spec.col not in seg.columns:
                        continue
                    m = seg[spec.col] == 1
                    if not m.any():
                        continue

                    t_evt_h = seg.loc[m, time_col].to_numpy(float) / 3600.0
                    show_label = spec.label if spec.label not in used else "_nolegend_"
                    used.add(spec.label)

                    if spec.col == "desat_flag":
                        y_min, y_max = ax.get_ylim()
                        y_desat = y_min + 0.03 * (y_max - y_min)
                        ax.scatter(
                            t_evt_h,
                            np.full_like(t_evt_h, y_desat, dtype=float),
                            marker="v",
                            s=25,
                            color=spec.color,
                            alpha=0.9,
                            label=show_label,
                            zorder=3,
                        )
                    else:
                        first_line = (show_label != "_nolegend_")
                        for x in t_evt_h:
                            ax.axvline(
                                x,
                                color=spec.color,
                                linestyle="-",
                                linewidth=1.6,
                                alpha=0.9,
                                label=spec.label if first_line else "_nolegend_",
                                zorder=3,
                            )
                            first_line = False

        ax.grid(True)
        ax.set_ylabel("RMSSD [ms]")
        if p == n_panels - 1:
            ax.set_xlabel("Time (hours from recording start)")
        if p == 0:
            _maybe_add_legend(ax, loc="upper right", fontsize=8)

    fig.tight_layout(rect=[0.04, 0.05, 0.98, 0.95])
    return fig


def _build_hrv_tv_metrics_figure(
    edf_base: str,
    t_hrv: np.ndarray,
    hrv_tv: Dict[str, np.ndarray],
    exclusion_zones: List[Tuple[float, float, str]],
    aux_df: Optional["pd.DataFrame"],
) -> Optional[plt.Figure]:
    if t_hrv is None or t_hrv.size == 0 or not hrv_tv:
        return None

    metrics = [
        ("sdnn_ms", "SDNN [ms]"),
        ("lf", "LF [ms²]"),
        ("hf", "HF [ms²]"),
        ("lf_hf", "LF/HF [-]"),
    ]

    series: List[tuple[str, str, np.ndarray]] = []
    for key, ylabel in metrics:
        y = hrv_tv.get(key, None)
        if y is None:
            continue
        if np.size(y) != np.size(t_hrv):
            continue
        series.append((key, ylabel, y))

    if not series:
        return None

    n_rows = len(series)
    fig, axes = plt.subplots(n_rows, 1, figsize=(11.69, 8.27), sharex=True)
    if n_rows == 1:
        axes = [axes]

    tv_win = getattr(config, "HRV_TV_WINDOW_SEC", None)
    if tv_win is not None and tv_win > 0:
        title = f"{edf_base} - HRV TV metrics (sliding {tv_win/60.0:.1f} min window)"
    else:
        title = f"{edf_base} - HRV TV metrics (sliding window)"

    fig.suptitle(title, fontsize=12)

    t_h = t_hrv / 3600.0
    start_h = float(t_h[0])
    end_h = float(t_h[-1])

    start_sec = float(t_hrv[0])
    end_sec = float(t_hrv[-1])

    for ax, (key, ylabel, y) in zip(axes, series):
        _add_exclusion_spans(ax, exclusion_zones, start_h, end_h, label_once=False)

        ok = np.isfinite(y)
        if np.any(ok):
            ax.plot(t_h[ok], y[ok], linewidth=0.9, label=key)

        ax.relim()
        ax.autoscale_view()

        _overlay_events_on_single_axis_whole_night(
            ax=ax,
            aux_df=aux_df,
            start_sec=start_sec,
            end_sec=end_sec,
            event_spec=DEFAULT_EVENT_PLOT_SPEC,
        )

        ax.set_ylabel(ylabel)
        ax.grid(True)

        if ax is axes[0]:
            _maybe_add_legend(ax, loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (hours from recording start)")
    fig.tight_layout(rect=[0.04, 0.05, 0.98, 0.95])
    return fig
