from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np

from .. import config
from .hrv_plot_utils import (
    _add_colored_event_key,
    _add_mean_median_lines,
    _add_metric_legend,
    _bin_series_mean_ci,
    _format_nrem_legend_label,
    _overlay_events_on_single_axis_whole_night,
    _plot_sleep_stagegram_on_ax,
    _shade_hrv_mask_layers,
)
from .specs import DEFAULT_EVENT_PLOT_SPEC, EventSpec
from .utils import _add_exclusion_spans, _shade_masked_regions

if TYPE_CHECKING:
    import pandas as pd


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
) -> Optional[Any]:
    if t_hrv is None or hrv_clean is None or t_hrv.size == 0:
        return None
    overview_hours = getattr(config, "OVERVIEW_PANEL_HOURS", 2.0)
    panel_sec = overview_hours * 3600.0
    duration_hrv_sec = float(t_hrv[-1]) if float(t_hrv[-1]) > 0 else duration_sec_fallback
    n_panels = max(1, int(np.ceil(duration_hrv_sec / panel_sec)))

    fig, axes = plt.subplots(n_panels, 1, figsize=(11.69, 8.27), sharex=False)
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(f"{edf_base} - HRV Overview", fontsize=11, y=0.985)
    use_raw = hrv_raw is not None and np.size(hrv_raw) == np.size(hrv_clean)

    for p, ax in enumerate(axes):
        start_sec = p * panel_sec
        end_sec = min((p + 1) * panel_sec, duration_hrv_sec)
        start_h = start_sec / 3600.0
        end_h = end_sec / 3600.0
        ax.text(0.01, 0.98, f"{start_h:.2f}-{end_h:.2f} h", transform=ax.transAxes, ha="left", va="top", fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="none", pad=0.2), zorder=5)
        ax.set_xlim(start_h, end_h)
        _add_exclusion_spans(ax, exclusion_zones, start_h, end_h, label_once=True)

        mask = (t_hrv >= start_sec) & (t_hrv <= end_sec)
        if np.any(mask):
            if hrv_mask_info is not None:
                mask_slice: Dict[str, object] = {key: np.asarray(value)[mask] for key, value in hrv_mask_info.items() if isinstance(value, np.ndarray) and np.size(value) == np.size(t_hrv)}
                _shade_hrv_mask_layers(ax, t_hrv[mask], mask_slice)

            th = t_hrv[mask] / 3600.0
            yr = None
            okr = None
            if use_raw:
                yr = np.asarray(hrv_raw)[mask]
                okr = np.isfinite(yr)
                if np.any(okr):
                    ax.plot(th, np.ma.masked_invalid(yr), label="Sleep-masked HRV", linewidth=0.5, color="tab:gray", alpha=0.6, zorder=1)

            yc = hrv_clean[mask]
            masked = ~np.isfinite(yc)
            _shade_masked_regions(ax, t_sec=t_hrv[mask], masked=masked, color="0.6", alpha=0.20)
            okc = np.isfinite(yc)
            if np.any(okc):
                ax.plot(th, np.ma.masked_invalid(yc), label="Clean HRV (sleep+event masked)", linewidth=1.2, zorder=2)
            else:
                ax.text(0.01, 0.92, "HRV clean: no finite samples (all NaN after masking)", transform=ax.transAxes, fontsize=9, va="top")
                if yr is not None and okr is not None and np.any(okr):
                    y0 = float(np.nanmin(yr[okr]))
                    y1 = float(np.nanmax(yr[okr]))
                    if np.isfinite(y0) and np.isfinite(y1) and y1 > y0:
                        m = 0.10 * (y1 - y0)
                        ax.set_ylim(y0 - m, y1 + m)

        _overlay_events_on_single_axis_whole_night(ax=ax, aux_df=aux_df, start_sec=start_sec, end_sec=end_sec, event_spec=event_spec, show_legend_labels=False, event_style="short")
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
) -> Optional[Any]:
    if t_hrv is None or t_hrv.size == 0 or not hrv_tv:
        return None
    panels: List[dict] = []
    y_sdnn = hrv_tv.get("sdnn_ms", None)
    if y_sdnn is not None and np.size(y_sdnn) == np.size(t_hrv):
        panels.append({"kind": "single", "key": "sdnn_ms", "ylabel": "SDNN [ms]", "series": [("SDNN", np.asarray(y_sdnn, dtype=float), "tab:green")]})
    y_lf = hrv_tv.get("lf", None)
    y_hf = hrv_tv.get("hf", None)
    lf_hf_series = []
    if y_lf is not None and np.size(y_lf) == np.size(t_hrv):
        lf_hf_series.append(("LF", np.asarray(y_lf, dtype=float), "tab:orange"))
    if y_hf is not None and np.size(y_hf) == np.size(t_hrv):
        lf_hf_series.append(("HF", np.asarray(y_hf, dtype=float), "tab:blue"))
    if lf_hf_series:
        panels.append({"kind": "multi", "key": "lf_hf_power", "ylabel": "LF / HF [ms²] (log10)", "yscale": "log", "series": lf_hf_series})
    y_ratio = hrv_tv.get("lf_hf", None)
    if y_ratio is not None and np.size(y_ratio) == np.size(t_hrv):
        panels.append({"kind": "single", "key": "lf_hf", "ylabel": "LF/HF [-]", "series": [("LF/HF", np.asarray(y_ratio, dtype=float), "tab:purple")]})
    if not panels:
        return None

    fig, axes = plt.subplots(len(panels), 1, figsize=(11.69, 8.27), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    tv_win = getattr(config, "HRV_TV_WINDOW_SEC", None)
    title = f"{edf_base} - HRV TV ({tv_win/60.0:.1f} min window)" if tv_win is not None and tv_win > 0 else f"{edf_base} - HRV TV (sliding window)"
    fig.suptitle(title, fontsize=11, y=0.985)

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
            t_bin_h, y_bin, y_ci = _bin_series_mean_ci(t_hrv, y_plot, bin_sec=float(getattr(config, "HRV_PLOT_BIN_SEC", 15.0 * 60.0)))
            okb = np.isfinite(y_bin)
            if not np.any(okb):
                continue
            plotted_any = True
            ax.plot(t_bin_h[okb], y_bin[okb], linewidth=1.3, label=label, color=color, zorder=2)
            ax.errorbar(t_bin_h[okb], y_bin[okb], yerr=y_ci[okb], fmt="none", elinewidth=0.9, capsize=2, alpha=0.45, color=color, zorder=2)
            if panel["kind"] == "single" or panel["key"] == "lf_hf_power":
                _add_mean_median_lines(ax, y_bin[okb], color=color)
        ax.relim()
        ax.autoscale_view()
        _overlay_events_on_single_axis_whole_night(ax=ax, aux_df=aux_df, start_sec=start_sec, end_sec=end_sec, event_spec=DEFAULT_EVENT_PLOT_SPEC, show_legend_labels=False)
        ax.set_ylabel(panel["ylabel"])
        if panel.get("yscale") == "log":
            ax.set_yscale("log")
        ax.grid(True)
        if legend_ax is None and plotted_any:
            legend_ax = ax
    if legend_ax is not None:
        _add_metric_legend(legend_ax, loc="lower right", fontsize=5, include_summary_lines=True, summary_color=panels[0]["series"][0][2])
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
) -> Optional[Any]:
    if t_hrv is None or t_hrv.size == 0:
        return None
    if event_spec is None:
        event_spec = DEFAULT_EVENT_PLOT_SPEC

    panels: List[dict] = []
    if hrv_rmssd is not None and np.size(hrv_rmssd) == np.size(t_hrv):
        panels.append({"kind": "single", "key": "rmssd", "ylabel": "RMSSD [ms]", "series": [(_format_nrem_legend_label("RMSSD", "rmssd_mean", sleep_combo_summaries), np.asarray(hrv_rmssd, dtype=float), "tab:green")]})
    y_sdnn = hrv_tv.get("sdnn_ms", None) if hrv_tv is not None else None
    if y_sdnn is not None and np.size(y_sdnn) == np.size(t_hrv):
        panels.append({"kind": "single", "key": "sdnn_ms", "ylabel": "SDNN [ms]", "series": [(_format_nrem_legend_label("SDNN", "sdnn", sleep_combo_summaries), np.asarray(y_sdnn, dtype=float), "tab:green")]})
    y_lf = hrv_tv.get("lf", None) if hrv_tv is not None else None
    y_hf = hrv_tv.get("hf", None) if hrv_tv is not None else None
    lf_hf_series = []
    if y_lf is not None and np.size(y_lf) == np.size(t_hrv):
        lf_hf_series.append((_format_nrem_legend_label("LF", "lf", sleep_combo_summaries), np.asarray(y_lf, dtype=float), "tab:orange"))
    if y_hf is not None and np.size(y_hf) == np.size(t_hrv):
        lf_hf_series.append((_format_nrem_legend_label("HF", "hf", sleep_combo_summaries), np.asarray(y_hf, dtype=float), "tab:blue"))
    if lf_hf_series:
        panels.append({"kind": "multi", "key": "lf_hf_power", "ylabel": "LF / HF [ms²] (log10)", "yscale": "log", "series": lf_hf_series})
    y_ratio = hrv_tv.get("lf_hf", None) if hrv_tv is not None else None
    if y_ratio is not None and np.size(y_ratio) == np.size(t_hrv):
        panels.append({"kind": "single", "key": "lf_hf", "ylabel": "LF/HF [-]", "series": [(_format_nrem_legend_label("LF/HF", "lf_hf", sleep_combo_summaries), np.asarray(y_ratio, dtype=float), "tab:purple")]})
    if not panels:
        return None

    fig, axes = plt.subplots(1 + len(panels), 1, figsize=(11.69, 8.27), sharex=True, gridspec_kw={"height_ratios": [0.7] + [1.0] * len(panels)})
    axes = list(np.atleast_1d(axes))
    ax_stage = axes[0]
    data_axes = axes[1:]

    plotted_stage = _plot_sleep_stagegram_on_ax(ax=ax_stage, edf_base=edf_base, aux_df=aux_df, show_title=False, show_xlabel=False, show_stats_box=False)
    if not plotted_stage:
        ax_stage.text(0.5, 0.5, "Sleep hypnogram unavailable", ha="center", va="center", transform=ax_stage.transAxes, fontsize=11)
        ax_stage.set_ylabel("Stage")
        ax_stage.grid(True, axis="x", alpha=0.35)

    tv_win = getattr(config, "HRV_TV_WINDOW_SEC", None)
    hrv_window_sec = float(getattr(config, "HRV_WINDOW_SEC", 300.0))
    hrv_step_hz = float(getattr(config, "HRV_TARGET_FS_HZ", 1.0))
    plot_bin_sec = float(getattr(config, "HRV_PLOT_BIN_SEC", 10.0 * 60.0))
    fig.suptitle(f"{edf_base} - Hypnogram + HRV ({tv_win/60.0:.1f} min window)" if tv_win is not None and tv_win > 0 else f"{edf_base} - Hypnogram + HRV", fontsize=11, y=0.988)
    step_sec = 1.0 / hrv_step_hz if hrv_step_hz > 0 else np.nan
    fig.text(0.5, 0.948, f"PAT-derived HRV. Sliding {hrv_window_sec/60.0:.1f} min window, evaluated every {step_sec:.0f} s; displayed as {plot_bin_sec/60.0:.0f} min binned mean +/- 95% CI.\nDashed/dotted lines show displayed-series mean/median. Legend NREM mean is a post-hoc NREM-only reference.\nTop markers indicate active exclusion events.", ha="center", va="top", fontsize=8)
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
            t_bin_h, y_bin, y_ci = _bin_series_mean_ci(t_hrv, y_plot, bin_sec=float(getattr(config, "HRV_PLOT_BIN_SEC", 15.0 * 60.0)))
            okb = np.isfinite(y_bin)
            if not np.any(okb):
                continue
            plotted_any = True
            ax.plot(t_bin_h[okb], y_bin[okb], linewidth=1.3, label=label, color=color, zorder=2)
            ax.errorbar(t_bin_h[okb], y_bin[okb], yerr=y_ci[okb], fmt="none", elinewidth=0.9, capsize=2, alpha=0.45, color=color, zorder=2)
            if panel["kind"] == "single" or panel["key"] == "lf_hf_power":
                _add_mean_median_lines(ax, y_bin[okb], color=color)
        ax.relim()
        ax.autoscale_view()
        if hrv_mask_info is not None and panel["key"] == "rmssd":
            _shade_hrv_mask_layers(ax, t_hrv, hrv_mask_info)
        _overlay_events_on_single_axis_whole_night(ax=ax, aux_df=aux_df, start_sec=start_sec, end_sec=end_sec, event_spec=event_spec, show_legend_labels=False, event_style="short")
        ax.set_xlim(start_h, end_h)
        ax.set_ylabel(panel["ylabel"])
        if panel.get("yscale") == "log":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.75)
        if plotted_any:
            _add_metric_legend(ax, loc="lower right", fontsize=6, include_summary_lines=plotted_any, summary_color=panel["series"][0][2])

    if data_axes:
        data_axes[-1].set_xlabel("Time (hours from recording start)")
    fig.tight_layout(rect=(0.04, 0.05, 0.98, 0.86))
    fig.subplots_adjust(hspace=0.22)
    return fig
