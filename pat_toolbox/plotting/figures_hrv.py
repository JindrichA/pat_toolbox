from __future__ import annotations

from typing import Optional, Tuple, Dict, TYPE_CHECKING, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from .. import config
from .specs import EventSpec, DEFAULT_EVENT_PLOT_SPEC
from .utils import _add_exclusion_spans, _shade_masked_regions, _maybe_add_legend

if TYPE_CHECKING:
    import pandas as pd


def git(
    ax: plt.Axes,
    y: np.ndarray,
    *,
    mean_label: str = "Mean",
    median_label: str = "Median",
) -> None:
    """
    Add dashed horizontal lines for mean and median of finite values in y.
    """
    if y is None:
        return

    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return

    y_mean = float(np.nanmean(y))
    y_median = float(np.nanmedian(y))

    ax.axhline(
        y_mean,
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        color="black",
        label=f"{mean_label}: {y_mean:.2f}",
        zorder=2,
    )
    ax.axhline(
        y_median,
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        color="0.35",
        label=f"{median_label}: {y_median:.2f}",
        zorder=2,
    )


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


def _plot_sleep_stagegram_on_ax(
    ax: plt.Axes,
    edf_base: str,
    aux_df: Optional["pd.DataFrame"],
    show_title: bool = True,
    show_xlabel: bool = True,
    show_stats_box: bool = True,
) -> bool:
    """
    Plot the sleep stagegram directly onto a provided axis.
    Returns True if plotted successfully, otherwise False.
    """
    if aux_df is None or len(aux_df) == 0:
        return False

    time_col = getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec")
    stage_col = getattr(config, "AUX_CSV_STAGE_CODE_COLUMN", "stage_code")

    if time_col not in aux_df.columns or stage_col not in aux_df.columns:
        return False

    t = aux_df[time_col].to_numpy(dtype=float)
    s = aux_df[stage_col].to_numpy(dtype=float)

    ok = np.isfinite(t) & np.isfinite(s)
    if not np.any(ok):
        return False

    t = t[ok]
    s = np.round(s[ok]).astype(int)

    order = np.argsort(t)
    t = t[order]
    s = s[order]

    if t.size >= 2:
        keep = np.ones_like(t, dtype=bool)
        keep[:-1] = (t[1:] != t[:-1])
        t = t[keep]
        s = s[keep]

    if t.size == 0:
        return False

    dt = np.diff(t)
    wrap_idx = np.where(dt < -12 * 3600)[0]
    if wrap_idx.size > 0:
        t2 = t.copy()
        for i in wrap_idx:
            t2[i + 1:] += 24 * 3600
        t = t2

    t0 = float(t[0])
    t = t - t0

    y_map = {0: 3, 3: 2, 1: 1, 2: 0}
    y = np.array([y_map.get(int(x), np.nan) for x in s], dtype=float)
    oky = np.isfinite(y)
    if not np.any(oky):
        return False

    t = t[oky]
    s = s[oky]
    y = y[oky]
    if t.size == 0:
        return False

    enabled = bool(getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False))
    policy = str(getattr(config, "SLEEP_STAGE_POLICY", "all_sleep"))
    try:
        include_set = set(config.sleep_include_numeric())
    except Exception:
        include_set = {1, 2, 3}

    included = np.array([int(si) in include_set for si in s], dtype=bool)
    xh = t / 3600.0

    if xh.size == 1:
        step_h = 1.0 / 3600.0
        edges = np.array([xh[0], xh[0] + step_h], dtype=float)
    else:
        d = np.diff(xh)
        dpos = d[np.isfinite(d) & (d > 0)]
        step_h = float(np.median(dpos)) if dpos.size else (1.0 / 3600.0)

        edges = np.empty(xh.size + 1, dtype=float)
        edges[:-1] = xh
        edges[-1] = xh[-1] + step_h

    bands = [
        (3, "Wake", 0.08),
        (2, "REM", 0.06),
        (1, "NREM (Light)", 0.05),
        (0, "NREM (Deep)", 0.05),
    ]
    for y0, _name, alpha in bands:
        ax.axhspan(y0 - 0.5, y0 + 0.5, alpha=alpha, zorder=0)

    x_step = edges
    y_step = np.r_[y, y[-1]]
    ax.step(x_step, y_step, where="post", linewidth=3.0, zorder=3)

    if enabled and included.size == xh.size:
        exc = ~included
        if np.any(exc):
            idx = np.where(exc)[0]
            splits = np.where(np.diff(idx) > 1)[0] + 1
            groups = np.split(idx, splits)
            for g in groups:
                if g.size == 0:
                    continue
                i0 = int(g[0])
                i1 = int(g[-1]) + 1
                x0 = float(edges[i0])
                x1 = float(edges[min(i1, edges.size - 1)])
                if x1 > x0:
                    ax.axvspan(x0, x1, color="k", alpha=0.06, zorder=1)

    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(["Wake", "REM", "NREM-Light", "NREM-Deep"])

    ax.set_ylim(-0.7, 3.7)
    ax.set_xlim(0.0, edges[-1])

    if show_xlabel:
        ax.set_xlabel("Time (hours from recording start)")

    if show_title:
        ax.set_title(f"{edf_base} - Hypnogram", fontsize=14, pad=12)

    ax.grid(True, which="major", axis="x", alpha=0.35)
    ax.grid(True, which="minor", axis="x", alpha=0.18)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.grid(True, which="major", axis="y", alpha=0.20)

    if show_stats_box:
        total = int(len(s))
        if total > 0:
            def pct(n: int) -> str:
                return f"{(100.0 * n / total):.1f}%"

            counts = {k: int(np.sum(s == k)) for k in [0, 1, 2, 3]}
            inc_n = int(np.sum(included)) if included.size == s.size else 0
            exc_n = int(total - inc_n) if included.size == s.size else 0

            stats_lines = [
                f"Masking: {'ON' if enabled else 'OFF'} ({policy})",
                f"Included: {inc_n} ({pct(inc_n)})" if included.size == s.size else "Included: NA",
                f"Excluded: {exc_n} ({pct(exc_n)})" if included.size == s.size else "Excluded: NA",
                "",
                f"Wake: {counts[0]} ({pct(counts[0])})",
                f"REM:  {counts[3]} ({pct(counts[3])})",
                f"NREM-Light: {counts[1]} ({pct(counts[1])})",
                f"NREM-Deep:  {counts[2]} ({pct(counts[2])})",
            ]

            ax.text(
                0.99,
                0.98,
                "\n".join(stats_lines),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", alpha=0.10, pad=0.4),
                zorder=10,
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return True


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
            _add_mean_median_lines(ax, y)

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


def _build_stagegram_and_hrv_tv_figure(
    edf_base: str,
    aux_df: Optional["pd.DataFrame"],
    t_hrv: np.ndarray,
    hrv_rmssd: Optional[np.ndarray],
    hrv_tv: Dict[str, np.ndarray],
    exclusion_zones: List[Tuple[float, float, str]],
) -> Optional[plt.Figure]:
    """
    Combined page with:
      - sleep stagegram on top
      - overnight RMSSD below
      - HRV TV metrics below that

    All subplots share the same x-axis in hours-from-start.
    """
    if t_hrv is None or t_hrv.size == 0:
        return None

    series: List[tuple[str, str, np.ndarray]] = []

    if hrv_rmssd is not None and np.size(hrv_rmssd) == np.size(t_hrv):
        series.append(("rmssd", "RMSSD [ms]", np.asarray(hrv_rmssd, dtype=float)))

    metrics = [
        ("sdnn_ms", "SDNN [ms]"),
        ("lf", "LF [ms²]"),
        ("hf", "HF [ms²]"),
        ("lf_hf", "LF/HF [-]"),
    ]

    for key, ylabel in metrics:
        y = hrv_tv.get(key, None) if hrv_tv is not None else None
        if y is None:
            continue
        if np.size(y) != np.size(t_hrv):
            continue
        series.append((key, ylabel, np.asarray(y, dtype=float)))

    if not series:
        return None

    n_rows = 1 + len(series)
    height_ratios = [1.2] + [1.0] * len(series)

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(11.69, 8.27),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )

    axes = list(np.atleast_1d(axes))
    ax_stage = axes[0]
    data_axes = axes[1:]

    plotted_stage = _plot_sleep_stagegram_on_ax(
        ax=ax_stage,
        edf_base=edf_base,
        aux_df=aux_df,
        show_title=False,
        show_xlabel=False,
        show_stats_box=False,
    )

    if not plotted_stage:
        ax_stage.text(
            0.5,
            0.5,
            "Sleep stagegram unavailable",
            ha="center",
            va="center",
            transform=ax_stage.transAxes,
            fontsize=11,
        )
        ax_stage.set_ylabel("Stage")
        ax_stage.grid(True, axis="x", alpha=0.35)

    tv_win = getattr(config, "HRV_TV_WINDOW_SEC", None)
    if tv_win is not None and tv_win > 0:
        fig.suptitle(
            f"{edf_base} - Sleep stagegram + overnight RMSSD + HRV TV metrics "
            f"(sliding {tv_win/60.0:.1f} min window)",
            fontsize=12,
        )
    else:
        fig.suptitle(
            f"{edf_base} - Sleep stagegram + overnight RMSSD + HRV TV metrics",
            fontsize=12,
        )

    t_h = t_hrv / 3600.0
    start_h = float(t_h[0])
    end_h = float(t_h[-1])

    start_sec = float(t_hrv[0])
    end_sec = float(t_hrv[-1])

    ax_stage.set_xlim(start_h, end_h)

    for ax, (key, ylabel, y) in zip(data_axes, series):
        _add_exclusion_spans(ax, exclusion_zones, start_h, end_h, label_once=False)

        ok = np.isfinite(y)
        if np.any(ok):
            label = "RMSSD" if key == "rmssd" else key
            ax.plot(t_h[ok], y[ok], linewidth=0.9, label=label)
            _add_mean_median_lines(ax, y)

        ax.relim()
        ax.autoscale_view()

        _overlay_events_on_single_axis_whole_night(
            ax=ax,
            aux_df=aux_df,
            start_sec=start_sec,
            end_sec=end_sec,
            event_spec=DEFAULT_EVENT_PLOT_SPEC,
        )

        ax.set_xlim(start_h, end_h)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    if data_axes:
        _maybe_add_legend(data_axes[0], loc="upper right", fontsize=8)
        data_axes[-1].set_xlabel("Time (hours from recording start)")

    fig.tight_layout(rect=[0.04, 0.05, 0.98, 0.95])
    return fig