from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from .. import config, sleep_mask
from ..metrics.hr_event_response import extract_event_hr_windows
from .hrv_plot_utils import _add_colored_event_key, _overlay_events_on_single_axis_whole_night, _shade_hrv_mask_layers
from .segment_plot_helpers import _overlay_pat_burden_area
from .specs import DEFAULT_EVENT_PLOT_SPEC, EventSpec
from .utils import _add_exclusion_spans, _shade_masked_regions

if TYPE_CHECKING:
    import pandas as pd


def _panel_bounds(t_sec: np.ndarray, duration_sec_fallback: float) -> list[tuple[float, float]]:
    overview_hours = float(getattr(config, "OVERVIEW_PANEL_HOURS", 2.0))
    panel_sec = overview_hours * 3600.0
    duration_sec = float(t_sec[-1]) if t_sec.size > 0 and float(t_sec[-1]) > 0 else float(duration_sec_fallback)
    n_panels = max(1, int(np.ceil(duration_sec / panel_sec)))
    return [
        (p * panel_sec, min((p + 1) * panel_sec, duration_sec))
        for p in range(n_panels)
    ]


def _init_overview_figure(edf_base: str, title: str, n_panels: int) -> tuple[Any, list[Any]]:
    fig, axes = plt.subplots(n_panels, 1, figsize=(11.69, 8.27), sharex=False)
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(f"{edf_base} - {title}", fontsize=11, y=0.992)
    return fig, list(axes)


def _overview_header_text(title: str) -> str:
    overview_hours = float(getattr(config, "OVERVIEW_PANEL_HOURS", 2.0))
    parts = [f"Overview page: {title}.", f"Panels are split into {overview_hours:.0f} h windows."]
    if "HRV" in title:
        parts.append(
            f"HRV uses current stage filtering first, then event exclusion for the clean signal (window={float(getattr(config, 'HRV_WINDOW_SEC', 300.0)) / 60.0:.1f} min)."
        )
    elif title == "HR Overview":
        parts.append(
            f"HR uses PAT-derived RR extraction and current HR settings (target fs={float(getattr(config, 'HR_TARGET_FS_HZ', 1.0)):.1f} Hz)."
        )
    elif title == "Event-Response HR Overview":
        parts.append(
            f"Event-response HR view using smooth={float(getattr(config, 'HR_EVENT_SMOOTH_SEC', 5.0)):.0f}s, event={float(getattr(config, 'HR_EVENT_WINDOW_SEC', 15.0)):.0f}s, recovery end={float(getattr(config, 'HR_EVENT_RECOVERY_END_SEC', 45.0)):.0f}s, desat extension={'ON' if bool(getattr(config, 'HR_EVENT_USE_DESAT_EXTENSION', False)) else 'OFF'}."
        )
    elif title == "PAT-Burden Overview":
        parts.append(
            "PAT burden shading marks burden area inside excluded event/desaturation regions; calculations still follow burden-specific baseline rules."
        )
    else:
        parts.append("Event markers use the current plotting and masking configuration.")
    return " ".join(parts)


def _event_marker_text() -> str:
    return (
        "Event markers: blue=desaturation, red=central A/H 3%, "
        "brown dashed=obstructive A/H 3%, green dotted=unclassified A/H 3%, "
        "Signal quality: PAT excluded = olive."
    )


def _overview_header_legend(title: str) -> list[Line2D]:
    show_raw_debug = bool(getattr(config, "PLOT_SHOW_RAW_DEBUG_OVERLAYS", False))
    if title == "HR Overview":
        handles = [
            Line2D([0], [0], color="tab:blue", linewidth=1.2, label="HR used"),
            Line2D([0], [0], color="tab:blue", linewidth=1.4, alpha=0.55, label="Event/desaturation markers"),
        ]
        if show_raw_debug:
            handles.insert(1, Line2D([0], [0], color="tab:gray", linewidth=0.7, alpha=0.6, label="HR raw"))
        return handles
    if title == "HRV-RMSSD Overview":
        handles = [
            Line2D([0], [0], color="tab:green", linewidth=1.2, label="Final-analysis RMSSD"),
            Line2D([0], [0], color="#6c757d", linewidth=6, alpha=0.10, label="Stage-policy excluded"),
            Line2D([0], [0], color="#c1121f", linewidth=6, alpha=0.08, label="Event-excluded"),
            Line2D([0], [0], color="#d4a017", linewidth=6, alpha=0.22, label="Metric invalid"),
        ]
        if show_raw_debug:
            handles.insert(1, Line2D([0], [0], color="tab:gray", linewidth=0.6, alpha=0.6, label="Pre-final-exclusion RMSSD"))
        return handles
    if title == "HRV-SDNN Overview":
        handles = [
            Line2D([0], [0], color="tab:green", linewidth=1.2, label="Final-analysis SDNN"),
            Line2D([0], [0], color="#6c757d", linewidth=6, alpha=0.10, label="Stage-policy excluded"),
            Line2D([0], [0], color="#c1121f", linewidth=6, alpha=0.08, label="Event-excluded"),
            Line2D([0], [0], color="#d4a017", linewidth=6, alpha=0.22, label="Metric invalid"),
        ]
        if show_raw_debug:
            handles.insert(0, Line2D([0], [0], color="tab:gray", linewidth=0.7, alpha=0.6, label="Pre-final-exclusion SDNN"))
        return handles
    if title == "HRV-LF-HF Overview":
        handles = [
            Line2D([0], [0], color="tab:orange", linewidth=1.2, label="LF"),
            Line2D([0], [0], color="tab:blue", linewidth=1.2, label="HF"),
            Line2D([0], [0], color="#6c757d", linewidth=6, alpha=0.10, label="Stage-policy excluded"),
            Line2D([0], [0], color="#c1121f", linewidth=6, alpha=0.08, label="Event-excluded"),
            Line2D([0], [0], color="#d4a017", linewidth=6, alpha=0.22, label="Metric invalid"),
        ]
        if show_raw_debug:
            handles.insert(0, Line2D([0], [0], color="tab:gray", linewidth=0.7, alpha=0.35, label="Pre-final-exclusion LF and HF traces"))
        return handles
    if title == "HRV-LF-HF Ratio Overview":
        handles = [
            Line2D([0], [0], color="tab:purple", linewidth=1.2, label="Final-analysis LF/HF"),
            Line2D([0], [0], color="#6c757d", linewidth=6, alpha=0.10, label="Stage-policy excluded"),
            Line2D([0], [0], color="#c1121f", linewidth=6, alpha=0.08, label="Event-excluded"),
            Line2D([0], [0], color="#d4a017", linewidth=6, alpha=0.22, label="Metric invalid"),
        ]
        if show_raw_debug:
            handles.insert(0, Line2D([0], [0], color="tab:gray", linewidth=0.7, alpha=0.6, label="Pre-final-exclusion LF/HF"))
        return handles
    if title == "Event-Response HR Overview":
        return [
            Line2D([0], [0], color="tab:blue", linewidth=1.0, alpha=0.7, label="HR raw"),
            Line2D([0], [0], color="tab:cyan", linewidth=6, alpha=0.12, label="Event window"),
            Line2D([0], [0], color="tab:green", linewidth=6, alpha=0.10, label="Recovery window"),
            Line2D([0], [0], color="0.35", linestyle="--", linewidth=1.0, label="Event mean"),
            Line2D([0], [0], color="black", marker="v", linestyle="None", label="Event minimum"),
            Line2D([0], [0], color="tab:red", marker="o", linestyle="None", label="Recovery maximum"),
        ]
    if title == "PAT-Burden Overview":
        return [
            Line2D([0], [0], color="tab:orange", linewidth=1.1, label="PAT AMP"),
            Line2D([0], [0], color="0.25", linestyle="--", linewidth=1.1, label="Local burden baseline"),
            Line2D([0], [0], color="tab:red", linewidth=6, alpha=0.22, label="Burden area shading"),
        ]
    return []


def _decorate_overview_figure(fig: Any, title: str) -> None:
    fig.text(0.04, 0.958, _overview_header_text(title), ha="left", va="top", fontsize=7.5)
    handles = _overview_header_legend(title)
    if handles:
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.972), fontsize=6.8, frameon=False, ncol=1)


def _prepare_panel(
    ax: Any,
    start_sec: float,
    end_sec: float,
    exclusion_zones,
    aux_df: Optional["pd.DataFrame"],
    event_spec: Sequence[EventSpec],
    *,
    show_exclusion_spans: bool = True,
) -> None:
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
    if show_exclusion_spans:
        _add_exclusion_spans(ax, exclusion_zones, start_h, end_h, label_once=True)
    _overlay_events_on_single_axis_whole_night(
        ax=ax,
        aux_df=aux_df,
        start_sec=start_sec,
        end_sec=end_sec,
        event_spec=list(event_spec),
        show_legend_labels=False,
        event_style="short",
    )
    ax.grid(True, alpha=0.75)


def _format_hour_tick(x: float, _pos: float) -> str:
    total_minutes = int(round(float(x) * 60.0))
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours:d}:{minutes:02d}"


def _shade_metric_invalid_regions(
    ax: Any,
    t_sec: np.ndarray,
    invalid_mask: np.ndarray,
    *,
    hrv_mask_info: Optional[Dict[str, object]] = None,
) -> None:
    invalid_mask = np.asarray(invalid_mask, dtype=bool)
    if hrv_mask_info:
        combined_keep = hrv_mask_info.get("combined_keep")
        if isinstance(combined_keep, np.ndarray) and np.size(combined_keep) == np.size(t_sec):
            invalid_mask = invalid_mask & np.asarray(combined_keep, dtype=bool)
    _shade_masked_regions(
        ax,
        t_sec=t_sec,
        masked=invalid_mask,
        color="#d4a017",
        alpha=0.22,
    )


def _finalize_overview_figure(fig: Any, axes: list[Any], ylabel: str) -> Any:
    for i, ax in enumerate(axes):
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_formatter(FuncFormatter(_format_hour_tick))
        if i == len(axes) - 1:
            ax.set_xlabel("Time (hours from recording start)")
    fig.tight_layout(rect=(0.04, 0.05, 0.98, 0.87))
    fig.subplots_adjust(hspace=0.22)
    return fig


def _build_hr_overview_figure(
    edf_base: str,
    t_hr: Optional[np.ndarray],
    hr_clean: Optional[np.ndarray],
    hr_raw: Optional[np.ndarray],
    aux_df: Optional["pd.DataFrame"],
    exclusion_zones,
    duration_sec_fallback: float,
    event_spec: Sequence[EventSpec] = DEFAULT_EVENT_PLOT_SPEC,
) -> Optional[Any]:
    if t_hr is None or hr_clean is None or t_hr.size == 0:
        return None

    bounds = _panel_bounds(t_hr, duration_sec_fallback)
    title = "HR Overview"
    fig, axes = _init_overview_figure(edf_base, title, len(bounds))
    _decorate_overview_figure(fig, title)
    _add_colored_event_key(fig, list(event_spec))
    show_raw_debug = bool(getattr(config, "PLOT_SHOW_RAW_DEBUG_OVERLAYS", False))
    use_raw = show_raw_debug and hr_raw is not None and np.size(hr_raw) == np.size(hr_clean)

    for idx, (ax, (start_sec, end_sec)) in enumerate(zip(axes, bounds)):
        _prepare_panel(ax, start_sec, end_sec, exclusion_zones, aux_df, event_spec, show_exclusion_spans=False)
        mask = (t_hr >= start_sec) & (t_hr <= end_sec)
        if not np.any(mask):
            continue
        t_panel = t_hr[mask]
        th = t_panel / 3600.0
        if use_raw:
            yr = np.asarray(hr_raw)[mask].astype(float)
            if np.any(np.isfinite(yr)):
                ax.plot(th, np.ma.masked_invalid(yr), label="HR raw", linewidth=0.7, color="tab:gray", alpha=0.6, zorder=1)
        yc = np.asarray(hr_clean)[mask].astype(float)
        masked = ~np.isfinite(yc)
        _shade_masked_regions(ax, t_sec=t_panel, masked=masked, color="0.6", alpha=0.20)
        if np.any(np.isfinite(yc)):
            ax.plot(th, np.ma.masked_invalid(yc), label="HR used", linewidth=1.2, color="tab:blue", zorder=2)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=5)

    return _finalize_overview_figure(fig, axes, "HR [bpm]")


def _build_hrv_rmssd_overview_figure(
    edf_base: str,
    t_hrv: np.ndarray,
    hrv_clean: np.ndarray,
    hrv_raw: Optional[np.ndarray],
    aux_df: Optional["pd.DataFrame"],
    exclusion_zones,
    duration_sec_fallback: float,
    event_spec: Sequence[EventSpec] = DEFAULT_EVENT_PLOT_SPEC,
    hrv_mask_info: Optional[Dict[str, object]] = None,
) -> Optional[Any]:
    if t_hrv is None or hrv_clean is None or t_hrv.size == 0:
        return None

    bounds = _panel_bounds(t_hrv, duration_sec_fallback)
    title = "HRV-RMSSD Overview"
    fig, axes = _init_overview_figure(edf_base, title, len(bounds))
    _decorate_overview_figure(fig, title)
    _add_colored_event_key(fig, list(event_spec))
    show_raw_debug = bool(getattr(config, "PLOT_SHOW_RAW_DEBUG_OVERLAYS", False))
    use_raw = show_raw_debug and hrv_raw is not None and np.size(hrv_raw) == np.size(hrv_clean)

    for idx, (ax, (start_sec, end_sec)) in enumerate(zip(axes, bounds)):
        _prepare_panel(ax, start_sec, end_sec, exclusion_zones, aux_df, event_spec)
        mask = (t_hrv >= start_sec) & (t_hrv <= end_sec)
        if not np.any(mask):
            continue
        panel_hrv_mask_info: Optional[Dict[str, object]] = None
        if hrv_mask_info is not None:
            mask_slice: Dict[str, object] = {
                key: np.asarray(value)[mask]
                for key, value in hrv_mask_info.items()
                if isinstance(value, np.ndarray) and np.size(value) == np.size(t_hrv)
            }
            panel_hrv_mask_info = mask_slice
            _shade_hrv_mask_layers(ax, t_hrv[mask], mask_slice)
        th = t_hrv[mask] / 3600.0
        if use_raw:
            yr = np.asarray(hrv_raw)[mask].astype(float)
            if np.any(np.isfinite(yr)):
                ax.plot(th, np.ma.masked_invalid(yr), label="Pre-final-exclusion RMSSD", linewidth=0.6, color="tab:gray", alpha=0.6, zorder=1)
        yc = hrv_clean[mask].astype(float)
        _shade_metric_invalid_regions(
            ax,
            t_hrv[mask],
            ~np.isfinite(yc),
            hrv_mask_info=panel_hrv_mask_info,
        )
        if np.any(np.isfinite(yc)):
            ax.plot(th, np.ma.masked_invalid(yc), label="Final-analysis RMSSD", linewidth=1.2, color="tab:green", zorder=2)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=5)

    return _finalize_overview_figure(fig, axes, "RMSSD [ms]")


def _build_single_series_overview_figure(
    edf_base: str,
    title: str,
    ylabel: str,
    t_sec: Optional[np.ndarray],
    y: Optional[np.ndarray],
    *,
    y_raw: Optional[np.ndarray] = None,
    color: str,
    aux_df: Optional["pd.DataFrame"],
    exclusion_zones,
    duration_sec_fallback: float,
    event_spec: Sequence[EventSpec] = DEFAULT_EVENT_PLOT_SPEC,
    yscale: Optional[str] = None,
    hrv_mask_info: Optional[Dict[str, object]] = None,
) -> Optional[Any]:
    if t_sec is None or y is None or np.size(t_sec) == 0 or np.size(y) != np.size(t_sec):
        return None

    bounds = _panel_bounds(np.asarray(t_sec), duration_sec_fallback)
    fig, axes = _init_overview_figure(edf_base, title, len(bounds))
    _decorate_overview_figure(fig, title)
    _add_colored_event_key(fig, list(event_spec))

    for ax, (start_sec, end_sec) in zip(axes, bounds):
        _prepare_panel(ax, start_sec, end_sec, exclusion_zones, aux_df, event_spec)
        mask = (t_sec >= start_sec) & (t_sec <= end_sec)
        if not np.any(mask):
            continue
        panel_hrv_mask_info: Optional[Dict[str, object]] = None
        if hrv_mask_info is not None:
            mask_slice: Dict[str, object] = {
                key: np.asarray(value)[mask]
                for key, value in hrv_mask_info.items()
                if isinstance(value, np.ndarray) and np.size(value) == np.size(t_sec)
            }
            panel_hrv_mask_info = mask_slice
            _shade_hrv_mask_layers(ax, np.asarray(t_sec)[mask], mask_slice)
        th = np.asarray(t_sec)[mask] / 3600.0
        yy_raw = None
        show_raw_debug = bool(getattr(config, "PLOT_SHOW_RAW_DEBUG_OVERLAYS", False))
        if show_raw_debug and y_raw is not None and np.size(y_raw) == np.size(t_sec):
            yy_raw = np.asarray(y_raw)[mask].astype(float)
            if np.any(np.isfinite(yy_raw)):
                ax.plot(th, np.ma.masked_invalid(yy_raw), linewidth=0.7, color="tab:gray", alpha=0.6, zorder=1)
        yy = np.asarray(y)[mask].astype(float)
        _shade_metric_invalid_regions(
            ax,
            np.asarray(t_sec)[mask],
            ~np.isfinite(yy),
            hrv_mask_info=panel_hrv_mask_info,
        )
        if np.any(np.isfinite(yy)):
            ax.plot(th, np.ma.masked_invalid(yy), linewidth=1.2, color=color, zorder=2)
        if yscale is not None:
            ax.set_yscale(yscale)

    return _finalize_overview_figure(fig, axes, ylabel)


def _build_multi_series_overview_figure(
    edf_base: str,
    title: str,
    ylabel: str,
    t_sec: Optional[np.ndarray],
    series: Sequence[dict[str, Any]],
    *,
    aux_df: Optional["pd.DataFrame"],
    exclusion_zones,
    duration_sec_fallback: float,
    event_spec: Sequence[EventSpec] = DEFAULT_EVENT_PLOT_SPEC,
    yscale: Optional[str] = None,
    hrv_mask_info: Optional[Dict[str, object]] = None,
) -> Optional[Any]:
    if t_sec is None or np.size(t_sec) == 0 or not series:
        return None

    bounds = _panel_bounds(np.asarray(t_sec), duration_sec_fallback)
    fig, axes = _init_overview_figure(edf_base, title, len(bounds))
    _decorate_overview_figure(fig, title)
    _add_colored_event_key(fig, list(event_spec))

    for idx, (ax, (start_sec, end_sec)) in enumerate(zip(axes, bounds)):
        _prepare_panel(ax, start_sec, end_sec, exclusion_zones, aux_df, event_spec)
        panel_mask = (t_sec >= start_sec) & (t_sec <= end_sec)
        if np.any(panel_mask):
            panel_hrv_mask_info: Optional[Dict[str, object]] = None
            if hrv_mask_info is not None:
                mask_slice: Dict[str, object] = {
                    key: np.asarray(value)[panel_mask]
                    for key, value in hrv_mask_info.items()
                    if isinstance(value, np.ndarray) and np.size(value) == np.size(t_sec)
                }
                panel_hrv_mask_info = mask_slice
                _shade_hrv_mask_layers(ax, np.asarray(t_sec)[panel_mask], mask_slice)
            panel_nonfinite_masks: list[np.ndarray] = []
            for spec in series:
                y = spec.get("y")
                if y is None or np.size(y) != np.size(t_sec):
                    continue
                yy_panel = np.asarray(y)[panel_mask].astype(float)
                panel_nonfinite_masks.append(~np.isfinite(yy_panel))
            if panel_nonfinite_masks:
                masked = np.logical_and.reduce(panel_nonfinite_masks)
                _shade_metric_invalid_regions(
                    ax,
                    np.asarray(t_sec)[panel_mask],
                    masked,
                    hrv_mask_info=panel_hrv_mask_info,
                )
        any_plotted = False
        for spec in series:
            y = spec.get("y")
            if y is None or np.size(y) != np.size(t_sec):
                continue
            y_raw = spec.get("y_raw")
            mask = (t_sec >= start_sec) & (t_sec <= end_sec)
            if not np.any(mask):
                continue
            show_raw_debug = bool(getattr(config, "PLOT_SHOW_RAW_DEBUG_OVERLAYS", False))
            if show_raw_debug and y_raw is not None and np.size(y_raw) == np.size(t_sec):
                yy_raw = np.asarray(y_raw)[mask].astype(float)
                if np.any(np.isfinite(yy_raw)):
                    th = np.asarray(t_sec)[mask] / 3600.0
                    ax.plot(
                        th,
                        np.ma.masked_invalid(yy_raw),
                        linewidth=0.7,
                        color="tab:gray",
                        alpha=0.35,
                        label="_nolegend_",
                        zorder=1,
                    )
            yy = np.asarray(y)[mask].astype(float)
            if not np.any(np.isfinite(yy)):
                continue
            th = np.asarray(t_sec)[mask] / 3600.0
            ax.plot(
                th,
                np.ma.masked_invalid(yy),
                linewidth=float(spec.get("linewidth", 1.2)),
                color=str(spec.get("color", "tab:blue")),
                alpha=float(spec.get("alpha", 0.95)),
                label=str(spec.get("label", "series")),
                zorder=2,
            )
            any_plotted = True
        if yscale is not None:
            ax.set_yscale(yscale)
        if idx == 0 and any_plotted:
            ax.legend(loc="lower right", fontsize=5)

    return _finalize_overview_figure(fig, axes, ylabel)


def _build_event_response_overview_figure(
    edf_base: str,
    t_hr: Optional[np.ndarray],
    hr_raw: Optional[np.ndarray],
    *,
    aux_df: Optional["pd.DataFrame"],
    exclusion_zones,
    duration_sec_fallback: float,
    event_spec: Sequence[EventSpec] = DEFAULT_EVENT_PLOT_SPEC,
) -> Optional[Any]:
    show_raw_debug = bool(getattr(config, "PLOT_SHOW_RAW_DEBUG_OVERLAYS", False))
    if (not show_raw_debug) or t_hr is None or hr_raw is None or np.size(t_hr) == 0 or np.size(hr_raw) != np.size(t_hr):
        return None

    bounds = _panel_bounds(np.asarray(t_hr), duration_sec_fallback)
    title = "Event-Response HR Overview"
    fig, axes = _init_overview_figure(edf_base, title, len(bounds))
    _decorate_overview_figure(fig, title)
    _add_colored_event_key(fig, list(event_spec))

    include_set = set(config.sleep_include_numeric()) if getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False) else None
    windows = extract_event_hr_windows(np.asarray(t_hr), np.asarray(hr_raw, dtype=float), aux_df, include_set=include_set) if aux_df is not None else []

    for idx, (ax, (start_sec, end_sec)) in enumerate(zip(axes, bounds)):
        _prepare_panel(ax, start_sec, end_sec, exclusion_zones, aux_df, event_spec)
        mask = (t_hr >= start_sec) & (t_hr <= end_sec)
        if not np.any(mask):
            continue
        th = np.asarray(t_hr)[mask] / 3600.0
        yy = np.asarray(hr_raw)[mask].astype(float)
        if np.any(np.isfinite(yy)):
            ax.plot(th, np.ma.masked_invalid(yy), linewidth=1.0, color="tab:blue", alpha=0.7, label="HR raw", zorder=2)

        used_windows = 0
        for w in windows:
            if w["event_end_t"] < start_sec or w["event_start_t"] > end_sec:
                continue
            used_windows += 1
            ax.axvspan(w["event_start_t"] / 3600.0, w["event_end_t"] / 3600.0, color="tab:cyan", alpha=0.12, label="Event window" if used_windows == 1 and idx == 0 else "_nolegend_", zorder=0)
            ax.axvspan(w["recovery_start_t"] / 3600.0, w["recovery_end_t"] / 3600.0, color="tab:green", alpha=0.10, label="Recovery window" if used_windows == 1 and idx == 0 else "_nolegend_", zorder=0)
            ax.plot(
                [w["event_start_t"] / 3600.0, w["event_end_t"] / 3600.0],
                [w["event_mean_hr"], w["event_mean_hr"]],
                linestyle="--",
                linewidth=0.9,
                color="0.35",
                alpha=0.8,
                label="Event mean" if used_windows == 1 and idx == 0 else "_nolegend_",
                zorder=1,
            )
            if np.isfinite(w["event_min_t"]) and np.isfinite(w["event_min_hr"]):
                ax.scatter(w["event_min_t"] / 3600.0, w["event_min_hr"], color="black", s=12, marker="v", zorder=4, label="Event minimum" if used_windows == 1 and idx == 0 else "_nolegend_")
            if np.isfinite(w["recovery_max_t"]) and np.isfinite(w["recovery_max_hr"]):
                ax.scatter(w["recovery_max_t"] / 3600.0, w["recovery_max_hr"], color="tab:red", s=16, zorder=4, label="Recovery maximum" if used_windows == 1 and idx == 0 else "_nolegend_")

        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="lower right", fontsize=5)

    return _finalize_overview_figure(fig, axes, "Event HR [bpm]")


def _build_pat_burden_overview_figure(
    edf_base: str,
    t_pat_amp: Optional[np.ndarray],
    pat_amp: Optional[np.ndarray],
    aux_df: Optional["pd.DataFrame"],
    exclusion_zones,
    duration_sec_fallback: float,
    event_spec: Sequence[EventSpec] = DEFAULT_EVENT_PLOT_SPEC,
) -> Optional[Any]:
    if t_pat_amp is None or pat_amp is None or np.size(t_pat_amp) == 0 or np.size(t_pat_amp) != np.size(pat_amp):
        return None

    bounds = _panel_bounds(np.asarray(t_pat_amp), duration_sec_fallback)
    title = "PAT-Burden Overview"
    fig, axes = _init_overview_figure(edf_base, title, len(bounds))
    _decorate_overview_figure(fig, title)
    _add_colored_event_key(fig, list(event_spec))

    for idx, (ax, (start_sec, end_sec)) in enumerate(zip(axes, bounds)):
        _prepare_panel(ax, start_sec, end_sec, exclusion_zones, aux_df, event_spec)
        mask = (t_pat_amp >= start_sec) & (t_pat_amp <= end_sec)
        if not np.any(mask):
            continue
        t_panel = np.asarray(t_pat_amp)[mask]
        y_panel = np.asarray(pat_amp)[mask].astype(float)
        if aux_df is not None:
            m_keep = sleep_mask.build_global_include_mask_for_times(t_panel, aux_df, apply_sleep=True, apply_events=True)
            if m_keep is not None:
                _shade_masked_regions(ax, t_sec=t_panel, masked=~m_keep, color="0.6", alpha=0.18)
        if np.any(np.isfinite(y_panel)):
            ax.plot(t_panel / 3600.0, np.ma.masked_invalid(y_panel), linewidth=1.1, color="tab:orange", alpha=0.9, label="PAT AMP", zorder=3)
            _overlay_pat_burden_area(ax, t_sec_all=np.asarray(t_pat_amp), pat_amp_all=np.asarray(pat_amp), aux_df=aux_df, seg_start_sec=start_sec, seg_end_sec=end_sec)
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="lower right", fontsize=5)

    return _finalize_overview_figure(fig, axes, "PAT AMP")
