from __future__ import annotations

from typing import Optional, Tuple, Dict, TYPE_CHECKING, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import blended_transform_factory

from .. import config
from .specs import EventSpec, DEFAULT_EVENT_PLOT_SPEC
from .utils import _add_exclusion_spans, _shade_masked_regions

if TYPE_CHECKING:
    import pandas as pd


def _shade_hrv_mask_layers(
    ax: plt.Axes,
    t_sec: np.ndarray,
    hrv_mask_info: Optional[Dict[str, object]],
) -> None:
    if not hrv_mask_info:
        return

    sleep_keep = hrv_mask_info.get("sleep_keep")
    if sleep_keep is not None:
        _shade_masked_regions(
            ax,
            t_sec=t_sec,
            masked=~np.asarray(sleep_keep, dtype=bool),
            color="#6c757d",
            alpha=0.10,
            zorder=0.03,
        )

    calc_excluded = hrv_mask_info.get("combined_keep")
    if calc_excluded is not None:
        _shade_masked_regions(
            ax,
            t_sec=t_sec,
            masked=~np.asarray(calc_excluded, dtype=bool),
            color="#c1121f",
            alpha=0.08,
            zorder=0.04,
        )


def _format_nrem_legend_label(
    label: str,
    metric_key: str,
    sleep_combo_summaries: Optional[Dict[str, Dict[str, object]]],
) -> str:
    if not isinstance(sleep_combo_summaries, dict):
        return label

    nrem_item = sleep_combo_summaries.get("nrem")
    if not isinstance(nrem_item, dict):
        return label

    hrv_summary = nrem_item.get("hrv_summary")
    if not isinstance(hrv_summary, dict):
        return label

    value = hrv_summary.get(metric_key)
    if value is None or not np.isfinite(value):
        return label

    if metric_key in {"rmssd_mean", "sdnn"}:
        value_str = f"{float(value):.1f} ms"
    elif metric_key in {"lf", "hf"}:
        value_str = f"{float(value):.4f}"
    else:
        value_str = f"{float(value):.2f}"

    return f"{label} (NREM mean: {value_str})"


def _bin_series_mean_ci(
    t_sec: np.ndarray,
    y: np.ndarray,
    *,
    bin_sec: Optional[float] = None,
    min_count: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin a time series into fixed-width bins and return:
      - bin centers in hours
      - mean per bin
      - 95% CI half-width per bin

    NaNs are ignored. Bins with fewer than min_count values are returned as NaN.
    """

    if bin_sec is None:
        bin_sec = float(getattr(config, "HRV_PLOT_BIN_SEC", 5.0 * 60.0))
    if min_count is None:
        min_count = int(getattr(config, "HRV_PLOT_BIN_MIN_COUNT", 3))

    t_sec = np.asarray(t_sec, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(t_sec) & np.isfinite(y)
    if not np.any(ok):
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    t_sec = t_sec[ok]
    y = y[ok]

    order = np.argsort(t_sec)
    t_sec = t_sec[order]
    y = y[order]

    start_sec = 0.0
    end_sec = float(t_sec[-1])

    edges = np.arange(start_sec, end_sec + bin_sec, bin_sec, dtype=float)
    if edges.size < 2:
        edges = np.array([start_sec, start_sec + bin_sec], dtype=float)

    centers_h = 0.5 * (edges[:-1] + edges[1:]) / 3600.0
    means = np.full(edges.size - 1, np.nan, dtype=float)
    ci95 = np.full(edges.size - 1, np.nan, dtype=float)

    for i in range(edges.size - 1):
        if i == edges.size - 2:
            m = (t_sec >= edges[i]) & (t_sec <= edges[i + 1])
        else:
            m = (t_sec >= edges[i]) & (t_sec < edges[i + 1])

        vals = y[m]
        vals = vals[np.isfinite(vals)]

        n = vals.size
        if n < min_count:
            continue

        mu = float(np.mean(vals))
        if n > 1:
            se = float(np.std(vals, ddof=1) / np.sqrt(n))
            ci = 1.96 * se
        else:
            ci = 0.0

        means[i] = mu
        ci95[i] = ci

    return centers_h, means, ci95




def _add_mean_median_lines(
    ax: plt.Axes,
    y: np.ndarray,
    *,
    color: str = "black",
    alpha: float = 0.5,
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
        color=color,
        alpha=alpha,
        label="_nolegend_",
        zorder=1,
    )

    ax.axhline(
        y_median,
        linestyle=":",
        linewidth=1.0,
        color=color,
        alpha=alpha,
        label="_nolegend_",
        zorder=1,
    )


def _add_metric_legend(
    ax: plt.Axes,
    *,
    loc: str = "lower right",
    fontsize: int = 6,
    include_summary_lines: bool = False,
    summary_color: str = "black",
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    usable = [
        (h, lab)
        for h, lab in zip(handles, labels)
        if lab and (not str(lab).startswith("_"))
    ]
    if include_summary_lines:
        usable.extend(
            [
                (
                    Line2D([0], [0], color=summary_color, linestyle="--", linewidth=1.6),
                    "Dashed line = displayed-series mean",
                ),
                (
                    Line2D([0], [0], color=summary_color, linestyle=":", linewidth=1.6),
                    "Dotted line = displayed-series median",
                ),
            ]
        )

    if not usable:
        return

    legend_handles, legend_labels = zip(*usable)
    ax.legend(legend_handles, legend_labels, loc=loc, fontsize=fontsize)


def _add_colored_event_key(fig: plt.Figure, event_spec: List[EventSpec]) -> None:
    if not event_spec:
        return
    textprops = {"fontsize": 7.5, "color": "black"}

    style_desc = {
        "desat_flag": "blue triangle",
        "evt_central_3": "red",
        "evt_obstructive_3": "brown dashed",
        "evt_unclassified_3": "green dotted",
        "exclude_hr_flag": "purple",
        "exclude_pat_flag": "olive",
    }
    lines: List[HPacker] = []
    chunk: List[TextArea] = [TextArea("Event markers: ", textprops=textprops)]
    for idx, spec in enumerate(event_spec):
        desc = style_desc.get(spec.col, spec.color)
        chunk.extend(
            [
                TextArea(spec.label, textprops={**textprops, "color": spec.color}),
                TextArea(f" = {desc}", textprops=textprops),
            ]
        )
        if idx != len(event_spec) - 1:
            chunk.append(TextArea(" | ", textprops=textprops))
        if (idx + 1) % 3 == 0 and idx != len(event_spec) - 1:
            lines.append(HPacker(children=chunk, align="center", pad=0, sep=0))
            chunk = []
    if chunk:
        lines.append(HPacker(children=chunk, align="center", pad=0, sep=0))

    packed = VPacker(children=lines, align="center", pad=0, sep=2)
    anchored = AnchoredOffsetbox(
        loc="upper center",
        child=packed,
        frameon=False,
        bbox_to_anchor=(0.5, 0.885),
        bbox_transform=fig.transFigure,
        borderpad=0.0,
        pad=0.0,
    )
    fig.add_artist(anchored)


def _overlay_events_on_single_axis_whole_night(
    ax: plt.Axes,
    aux_df: Optional["pd.DataFrame"],
    start_sec: float,
    end_sec: float,
    event_spec: List[EventSpec] = DEFAULT_EVENT_PLOT_SPEC,
    show_legend_labels: bool = True,
    event_style: str = "full",
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
    event_row = 0

    for spec in event_spec:
        if spec.col not in seg.columns:
            continue

        m = seg[spec.col] == 1
        if not m.any():
            continue

        t_evt_h = seg.loc[m, time_col].to_numpy(float) / 3600.0
        if show_legend_labels:
            show_label = spec.label if spec.label not in used else "_nolegend_"
            used.add(spec.label)
        else:
            show_label = "_nolegend_"

        if spec.col == "desat_flag":
            y_min, y_max = ax.get_ylim()
            if ax.get_yscale() == "log" and y_min > 0 and y_max > y_min:
                log_y_min = np.log10(y_min)
                log_y_max = np.log10(y_max)
                y_desat = 10 ** (log_y_min + 0.08 * (log_y_max - log_y_min))
            else:
                y_desat = y_min + 0.03 * (y_max - y_min)
            ax.scatter(
                t_evt_h,
                np.full_like(t_evt_h, y_desat, dtype=float),
                marker="v",
                s=32,
                color=spec.color,
                alpha=0.95,
                label=show_label,
                zorder=4,
            )
        else:
            line_color = spec.color
            line_style = "-"
            line_width = 1.0
            line_alpha = 0.35

            if spec.col == "evt_obstructive_3":
                line_color = "tab:brown"
                line_style = "--"
            elif spec.col == "evt_unclassified_3":
                line_style = ":"

            first_line = (show_label != "_nolegend_")
            if event_style == "short":
                y1 = 0.98 - 0.06 * event_row
                y0 = max(0.72, y1 - 0.12)
                trans = blended_transform_factory(ax.transData, ax.transAxes)
                for x in t_evt_h:
                    ax.plot(
                        [x, x],
                        [y0, y1],
                        color=line_color,
                        linestyle=line_style,
                        linewidth=1.7,
                        alpha=0.85,
                        transform=trans,
                        label=spec.label if first_line else "_nolegend_",
                        zorder=4,
                        solid_capstyle="round",
                    )
                    first_line = False
                event_row += 1
            else:
                for x in t_evt_h:
                    ax.axvline(
                        x,
                        color=line_color,
                        linestyle=line_style,
                        linewidth=line_width,
                        alpha=line_alpha,
                        label=spec.label if first_line else "_nolegend_",
                        zorder=0,
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

    y_map = {
        3: 3,  # REM (top)
        0: 2,  # Wake
        1: 1,  # NREM-Light
        2: 0,  # NREM-Deep (bottom)
    }
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
        (3, "REM", 0.08),
        (2, "Wake", 0.06),
        (1, "NREM-Light", 0.05),
        (0, "NREM-Deep", 0.05),
    ]
    for y0, _name, alpha in bands:
        ax.axhspan(y0 - 0.5, y0 + 0.5, alpha=alpha, zorder=0)

    x_step = edges
    y_step = np.r_[y, y[-1]]

    stage_colors = {
        3: "#d62728",  # REM (red)
        0: "#ff7f0e",  # Wake (orange)
        1: "#1f77b4",  # Light (blue)
        2: "#e377c2",  # Deep (pink)
    }

    for i in range(len(y)):
        x0 = edges[i]
        x1 = edges[i + 1]
        yi = y[i]
        si = s[i]

        color = stage_colors.get(si, "black")

        # horizontal segment
        ax.hlines(
            yi,
            x0,
            x1,
            colors=color,
            linewidth=4.0,
            zorder=3,
        )

        # vertical transition (except last)
        if i < len(y) - 1:
            ax.vlines(
                x1,
                yi,
                y[i + 1],
                colors="black",
                linewidth=2.0,
                zorder=3,
            )

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
    ax.set_yticklabels(["REM", "Wake", "Light Sleep", "Deep Sleep"])

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
    hrv_mask_info: Optional[Dict[str, object]] = None,
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

    fig.suptitle(
        f"{edf_base} - HRV Overview",
        fontsize=11,
        y=0.985,
    )

    use_raw = hrv_raw is not None and np.size(hrv_raw) == np.size(hrv_clean)

    for p, ax in enumerate(axes):
        start_sec = p * panel_sec
        end_sec = min((p + 1) * panel_sec, duration_hrv_sec)

        start_h = start_sec / 3600.0
        end_h = end_sec / 3600.0

        ax.text(
            0.01,
            0.98,
            f"{start_h:.2f}-{end_h:.2f} h",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="none", pad=0.2),
            zorder=5,
        )
        ax.set_xlim(start_h, end_h)

        _add_exclusion_spans(ax, exclusion_zones, start_h, end_h, label_once=True)

        mask = (t_hrv >= start_sec) & (t_hrv <= end_sec)
        if np.any(mask):
            if hrv_mask_info is not None:
                mask_slice = {
                    key: np.asarray(value)[mask]
                    for key, value in hrv_mask_info.items()
                    if isinstance(value, np.ndarray) and np.size(value) == np.size(t_hrv)
                }
                _shade_hrv_mask_layers(ax, t_hrv[mask], mask_slice)

            th = t_hrv[mask] / 3600.0

            yr = None
            okr = None

            if use_raw:
                yr = hrv_raw[mask]
                okr = np.isfinite(yr)
                if np.any(okr):
                    ax.plot(
                        th,
                        np.ma.masked_invalid(yr),
                        label="Sleep-masked HRV",
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
                    th,
                    np.ma.masked_invalid(yc),
                    label="Clean HRV (sleep+event masked)",
                    linewidth=1.2,
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

        _overlay_events_on_single_axis_whole_night(
            ax=ax,
            aux_df=aux_df,
            start_sec=start_sec,
            end_sec=end_sec,
            event_spec=event_spec,
            show_legend_labels=False,
            event_style="short",
        )

        ax.grid(True)
        ax.set_ylabel("RMSSD [ms]")
        if p == n_panels - 1:
            ax.set_xlabel("Time (hours from recording start)")
        if p == 0:
            _add_metric_legend(ax, loc="lower right", fontsize=5)

    fig.tight_layout(rect=(0.04, 0.05, 0.98, 0.94))
    fig.subplots_adjust(hspace=0.22)
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

    panels: List[dict] = []

    y_sdnn = hrv_tv.get("sdnn_ms", None)
    if y_sdnn is not None and np.size(y_sdnn) == np.size(t_hrv):
        panels.append(
            {
                "kind": "single",
                "key": "sdnn_ms",
                "ylabel": "SDNN [ms]",
                "series": [("SDNN", np.asarray(y_sdnn, dtype=float), "tab:green")],
            }
        )

    y_lf = hrv_tv.get("lf", None)
    y_hf = hrv_tv.get("hf", None)
    lf_hf_series = []
    if y_lf is not None and np.size(y_lf) == np.size(t_hrv):
        lf_hf_series.append(("LF", np.asarray(y_lf, dtype=float), "tab:orange"))
    if y_hf is not None and np.size(y_hf) == np.size(t_hrv):
        lf_hf_series.append(("HF", np.asarray(y_hf, dtype=float), "tab:blue"))
    if lf_hf_series:
        panels.append(
            {
                "kind": "multi",
                "key": "lf_hf_power",
                "ylabel": "LF / HF [ms²] (log10)",
                "yscale": "log",
                "series": lf_hf_series,
            }
        )

    y_ratio = hrv_tv.get("lf_hf", None)
    if y_ratio is not None and np.size(y_ratio) == np.size(t_hrv):
        panels.append(
            {
                "kind": "single",
                "key": "lf_hf",
                "ylabel": "LF/HF [-]",
                "series": [("LF/HF", np.asarray(y_ratio, dtype=float), "tab:purple")],
            }
        )

    if not panels:
        return None

    n_rows = len(panels)
    fig, axes = plt.subplots(n_rows, 1, figsize=(11.69, 8.27), sharex=True)
    if n_rows == 1:
        axes = [axes]

    tv_win = getattr(config, "HRV_TV_WINDOW_SEC", None)
    if tv_win is not None and tv_win > 0:
        title = f"{edf_base} - HRV TV metrics (sliding {tv_win/60.0:.1f} min window)"
    else:
        title = f"{edf_base} - HRV TV metrics (sliding window)"

    fig.suptitle(title.replace("HRV TV metrics", "HRV TV"), fontsize=11, y=0.985)

    t_h = t_hrv / 3600.0
    start_h = float(t_h[0])
    end_h = float(t_h[-1])

    start_sec = float(t_hrv[0])
    end_sec = float(t_hrv[-1])

    legend_ax = None

    for ax, panel in zip(axes, panels):
        _add_exclusion_spans(ax, exclusion_zones, start_h, end_h, label_once=False)

        plotted_any = False
        for label, y, color in panel["series"]:
            ok = np.isfinite(y)
            if not np.any(ok):
                continue

            p = np.nanpercentile(y, 99)
            y_plot = np.clip(y, None, p)
            t_bin_h, y_bin, y_ci = _bin_series_mean_ci(
                t_hrv,
                y_plot,
                bin_sec=float(getattr(config, "HRV_PLOT_BIN_SEC", 15.0 * 60.0)),
            )

            okb = np.isfinite(y_bin)
            if not np.any(okb):
                continue

            plotted_any = True
            ax.plot(
                t_bin_h[okb],
                y_bin[okb],
                linewidth=1.3,
                label=label,
                color=color,
                zorder=2,
            )
            ax.errorbar(
                t_bin_h[okb],
                y_bin[okb],
                yerr=y_ci[okb],
                fmt="none",
                elinewidth=0.9,
                capsize=2,
                alpha=0.45,
                color=color,
                zorder=2,
            )

            if panel["kind"] == "single":
                _add_mean_median_lines(ax, y_bin[okb], color=color)
            elif panel["key"] == "lf_hf_power":
                _add_mean_median_lines(ax, y_bin[okb], color=color)

        ax.relim()
        ax.autoscale_view()

        _overlay_events_on_single_axis_whole_night(
            ax=ax,
            aux_df=aux_df,
            start_sec=start_sec,
            end_sec=end_sec,
            event_spec=DEFAULT_EVENT_PLOT_SPEC,
            show_legend_labels=False,
        )

        ax.set_ylabel(panel["ylabel"])
        if panel.get("yscale") == "log":
            ax.set_yscale("log")

        ax.grid(True)

        if legend_ax is None and plotted_any:
            legend_ax = ax

    if legend_ax is not None:
        _add_metric_legend(
            legend_ax,
            loc="lower right",
            fontsize=5,
            include_summary_lines=True,
            summary_color=panels[0]["series"][0][2],
        )

    axes[-1].set_xlabel("Time (hours from recording start)")
    fig.tight_layout(rect=(0.04, 0.05, 0.98, 0.94))
    fig.subplots_adjust(hspace=0.22)
    return fig


def _build_stagegram_and_hrv_tv_figure(
    edf_base: str,
    aux_df: Optional["pd.DataFrame"],
    t_hrv: np.ndarray,
    hrv_rmssd: Optional[np.ndarray],
    hrv_tv: Dict[str, np.ndarray],
    exclusion_zones: List[Tuple[float, float, str]],
    sleep_combo_summaries: Optional[Dict[str, Dict[str, object]]] = None,
    event_spec: Optional[List[EventSpec]] = None,
    hrv_mask_info: Optional[Dict[str, object]] = None,
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
    if event_spec is None:
        event_spec = DEFAULT_EVENT_PLOT_SPEC

    panels: List[dict] = []

    if hrv_rmssd is not None and np.size(hrv_rmssd) == np.size(t_hrv):
        panels.append(
            {
                "kind": "single",
                "key": "rmssd",
                "ylabel": "RMSSD [ms]",
                "series": [(
                    _format_nrem_legend_label("RMSSD", "rmssd_mean", sleep_combo_summaries),
                    np.asarray(hrv_rmssd, dtype=float),
                    "tab:green",
                )],
            }
        )

    y_sdnn = hrv_tv.get("sdnn_ms", None) if hrv_tv is not None else None
    if y_sdnn is not None and np.size(y_sdnn) == np.size(t_hrv):
        panels.append(
            {
                "kind": "single",
                "key": "sdnn_ms",
                "ylabel": "SDNN [ms]",
                "series": [(
                    _format_nrem_legend_label("SDNN", "sdnn", sleep_combo_summaries),
                    np.asarray(y_sdnn, dtype=float),
                    "tab:green",
                )],
            }
        )

    y_lf = hrv_tv.get("lf", None) if hrv_tv is not None else None
    y_hf = hrv_tv.get("hf", None) if hrv_tv is not None else None
    lf_hf_series = []
    if y_lf is not None and np.size(y_lf) == np.size(t_hrv):
        lf_hf_series.append((
            _format_nrem_legend_label("LF", "lf", sleep_combo_summaries),
            np.asarray(y_lf, dtype=float),
            "tab:orange",
        ))
    if y_hf is not None and np.size(y_hf) == np.size(t_hrv):
        lf_hf_series.append((
            _format_nrem_legend_label("HF", "hf", sleep_combo_summaries),
            np.asarray(y_hf, dtype=float),
            "tab:blue",
        ))
    if lf_hf_series:
        panels.append(
            {
                "kind": "multi",
                "key": "lf_hf_power",
                "ylabel": "LF / HF [ms²] (log10)",
                "yscale": "log",
                "series": lf_hf_series,
            }
        )

    y_ratio = hrv_tv.get("lf_hf", None) if hrv_tv is not None else None
    if y_ratio is not None and np.size(y_ratio) == np.size(t_hrv):
        panels.append(
            {
                "kind": "single",
                "key": "lf_hf",
                "ylabel": "LF/HF [-]",
                "series": [(
                    _format_nrem_legend_label("LF/HF", "lf_hf", sleep_combo_summaries),
                    np.asarray(y_ratio, dtype=float),
                    "tab:purple",
                )],
            }
        )

    if not panels:
        return None

    n_rows = 1 + len(panels)
    height_ratios = [0.7] + [1.0] * len(panels)

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
            "Sleep hypnogram unavailable",
            ha="center",
            va="center",
            transform=ax_stage.transAxes,
            fontsize=11,
        )
        ax_stage.set_ylabel("Stage")
        ax_stage.grid(True, axis="x", alpha=0.35)

    tv_win = getattr(config, "HRV_TV_WINDOW_SEC", None)
    hrv_window_sec = float(getattr(config, "HRV_WINDOW_SEC", 300.0))
    hrv_step_hz = float(getattr(config, "HRV_TARGET_FS_HZ", 1.0))
    plot_bin_sec = float(getattr(config, "HRV_PLOT_BIN_SEC", 10.0 * 60.0))
    if tv_win is not None and tv_win > 0:
        fig.suptitle(
            f"{edf_base} - Hypnogram + HRV ({tv_win/60.0:.1f} min window)",
            fontsize=11,
            y=0.988,
        )
    else:
        fig.suptitle(
            f"{edf_base} - Hypnogram + HRV",
            fontsize=11,
            y=0.988,
        )

    step_sec = 1.0 / hrv_step_hz if hrv_step_hz > 0 else np.nan
    fig.text(
        0.5,
        0.948,
        (
            f"PAT-derived HRV. Sliding {hrv_window_sec/60.0:.1f} min window, "
            f"evaluated every {step_sec:.0f} s; displayed as {plot_bin_sec/60.0:.0f} min "
            f"binned mean +/- 95% CI.\n"
            f"Dashed/dotted lines show displayed-series mean/median. "
            f"Legend NREM mean is a post-hoc NREM-only reference.\n"
            f"Top markers indicate active exclusion events."
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    _add_colored_event_key(fig, event_spec)

    t_h = t_hrv / 3600.0
    start_h = float(t_h[0])
    end_h = float(t_h[-1])

    start_sec = float(t_hrv[0])
    end_sec = float(t_hrv[-1])

    ax_stage.set_xlim(start_h, end_h)

    for ax, panel in zip(data_axes, panels):
        plotted_any = False
        for label, y, color in panel["series"]:
            ok = np.isfinite(y)
            if not np.any(ok):
                continue

            p = np.nanpercentile(y, 99)
            y_plot = np.clip(y, None, p)
            t_bin_h, y_bin, y_ci = _bin_series_mean_ci(
                t_hrv,
                y_plot,
                bin_sec=float(getattr(config, "HRV_PLOT_BIN_SEC", 15.0 * 60.0)),
            )

            okb = np.isfinite(y_bin)
            if not np.any(okb):
                continue

            plotted_any = True
            show_label = label

            ax.plot(
                t_bin_h[okb],
                y_bin[okb],
                linewidth=1.3,
                label=show_label,
                color=color,
                zorder=2,
            )
            ax.errorbar(
                t_bin_h[okb],
                y_bin[okb],
                yerr=y_ci[okb],
                fmt="none",
                elinewidth=0.9,
                capsize=2,
                alpha=0.45,
                color=color,
                zorder=2,
            )

            if panel["kind"] == "single":
                _add_mean_median_lines(ax, y_bin[okb], color=color)
            elif panel["key"] == "lf_hf_power":
                _add_mean_median_lines(ax, y_bin[okb], color=color)

        ax.relim()
        ax.autoscale_view()

        _overlay_events_on_single_axis_whole_night(
            ax=ax,
            aux_df=aux_df,
            start_sec=start_sec,
            end_sec=end_sec,
            event_spec=event_spec,
            show_legend_labels=False,
            event_style="short",
        )

        ax.set_xlim(start_h, end_h)
        ax.set_ylabel(panel["ylabel"])
        if panel.get("yscale") == "log":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.75)

        if plotted_any:
            _add_metric_legend(
                ax,
                loc="lower right",
                fontsize=6,
                include_summary_lines=plotted_any,
                summary_color=panel["series"][0][2],
            )

    if data_axes:
        data_axes[-1].set_xlabel("Time (hours from recording start)")

    fig.tight_layout(rect=(0.04, 0.05, 0.98, 0.86))
    fig.subplots_adjust(hspace=0.22)
    return fig
