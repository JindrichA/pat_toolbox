from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .. import sleep_mask
from .. import config
from .specs import EventSpec, DEFAULT_EVENT_PLOT_SPEC
from .utils import _add_exclusion_spans, _shade_masked_regions, _maybe_add_legend, _h_to_hhmm
from .. import io_aux_csv
from matplotlib.patches import Patch

if TYPE_CHECKING:
    import pandas as pd


def _plot_segment_pat(ax: plt.Axes, t_h: np.ndarray, seg_raw: np.ndarray, seg_filt: np.ndarray, title: str) -> None:
    ax.set_title(title, fontsize=12)
    ax.plot(t_h, seg_raw, label="PAT raw", linewidth=0.8)
    ax.plot(t_h, seg_filt, label="PAT filtered", linewidth=1.0)
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
        fontsize=9,
    )


def _apply_global_mask_to_series(
    t_sec: np.ndarray,
    y: np.ndarray,
    aux_df: Optional["pd.DataFrame"],
) -> np.ndarray:
    """
    Returns a copy of y where excluded samples (sleep + events, per global mask)
    are set to NaN. If mask can't be built, returns y unchanged.
    """
    y2 = y.astype(float, copy=True)

    if aux_df is None or t_sec is None or t_sec.size == 0:
        return y2

    m_keep = sleep_mask.build_global_include_mask_for_times(
        t_sec, aux_df, apply_sleep=True, apply_events=True
    )
    if m_keep is None:
        return y2

    y2[~m_keep] = np.nan
    return y2

def _plot_raw_dashed(
    ax: plt.Axes,
    t_sec: np.ndarray,
    y: np.ndarray,
    *,
    label: str,
    color: str,
    zorder: int = 0,
):
    ok = np.isfinite(y)
    if np.any(ok):
        ax.plot(
            t_sec[ok] / 3600.0,
            y[ok],
            linestyle="--",
            linewidth=1.0,
            color=color,
            alpha=0.6,
            label=label,
            zorder=zorder,
        )


def _plot_segment_hr(
    ax: plt.Axes,
    t_hr_edf: Optional[np.ndarray],
    hr_edf: Optional[np.ndarray],
    t_hr_calc: Optional[np.ndarray],
    hr_calc: Optional[np.ndarray],
    seg_start_sec: float,
    seg_end_sec: float,
    exclusion_zones: List[Tuple[float, float, str]],
    t_seg_h_start: float,
    t_seg_h_end: float,
    aux_df: Optional["pd.DataFrame"],
    t_seg_sec: np.ndarray,
) -> tuple[Optional[float], Optional[float]]:
    _add_exclusion_spans(ax, exclusion_zones, t_seg_h_start, t_seg_h_end, label_once=True)




    # -------------------------------------------------
    # Sleep-stage masking (gray shading, HRV-style)
    # -------------------------------------------------
    # -------------------------------------------------
    # Sleep-stage masking (gray shading, HRV-style)
    # Use continuous segment timebase so shading covers
    # the full segment (not only where HR samples exist).
    # -------------------------------------------------
    # -------------------------------------------------
    # Mask shading (HRV-style): sleep + events combined
    # -------------------------------------------------
    if aux_df is not None and t_seg_sec.size > 0:
        m_keep = sleep_mask.build_global_include_mask_for_times(t_seg_sec, aux_df, apply_sleep=True, apply_events=True)
        if m_keep is not None:
            _shade_masked_regions(
                ax,
                t_sec=t_seg_sec,  # seconds
                masked=~m_keep,  # shade excluded
                color="0.6",
                alpha=0.22,
            )

    seg_hr_min: Optional[float] = None
    seg_hr_max: Optional[float] = None
    summary_lines: List[str] = []

    def _stats_line(name: str, y_seg: np.ndarray) -> None:
        ok = np.isfinite(y_seg)
        n_total = int(y_seg.size)
        n_used = int(np.count_nonzero(ok))

        if n_used == 0:
            summary_lines.append(f"{name}: no finite samples after masking (used 0/{n_total})")
            return

        yy = y_seg[ok]
        summary_lines.append(
            f"{name}: n={n_used}/{n_total} | "
            f"min={np.min(yy):.2f}, max={np.max(yy):.2f}, "
            f"mean={np.mean(yy):.2f}, median={np.median(yy):.2f}, std={np.std(yy):.2f}"
        )

        # Update ylim stats (based on NON-excluded samples only)
        nonlocal seg_hr_min, seg_hr_max
        y0 = float(np.min(yy))
        y1 = float(np.max(yy))
        seg_hr_min = y0 if seg_hr_min is None else min(seg_hr_min, y0)
        seg_hr_max = y1 if seg_hr_max is None else max(seg_hr_max, y1)

    # -------------------------
    # EDF HR
    # -------------------------
    if t_hr_edf is not None and hr_edf is not None:
        mask_edf = (t_hr_edf >= seg_start_sec) & (t_hr_edf <= seg_end_sec)
        if np.any(mask_edf):
            t_sec_seg = t_hr_edf[mask_edf]
            t_h = t_sec_seg / 3600.0
            y_raw = hr_edf[mask_edf].astype(float)

            # 1) RAW (dashed, everywhere)
            _plot_raw_dashed(
                ax,
                t_sec_seg,
                y_raw,
                label="EDF HR (raw)",
                color="0.5",
                zorder=0,
            )

            # 2) CLEAN (solid, masked)
            y_seg = _apply_global_mask_to_series(t_sec_seg, y_raw, aux_df)

            ok_plot = np.isfinite(y_seg)
            if np.any(ok_plot):
                ax.plot(
                    t_h[ok_plot],
                    y_seg[ok_plot],
                    label="original HR from the EDF",
                    linewidth=1.0,
                    alpha=0.7,
                    zorder=1,
                )

            # Summary + ylim from NON-excluded only
            _stats_line("EDF", y_seg)

    # -------------------------
    # PAT HR
    # -------------------------
    if t_hr_calc is not None and hr_calc is not None:
        mask_calc = (t_hr_calc >= seg_start_sec) & (t_hr_calc <= seg_end_sec)
        if np.any(mask_calc):
            t_sec_seg = t_hr_calc[mask_calc]
            t_h = t_sec_seg / 3600.0
            y_raw = hr_calc[mask_calc].astype(float)

            # 1) RAW (dashed, everywhere)
            _plot_raw_dashed(
                ax,
                t_sec_seg,
                y_raw,
                label="PAT HR (raw)",
                color="tab:blue",
                zorder=0,
            )

            # 2) CLEAN (solid, masked)
            y_seg = _apply_global_mask_to_series(t_sec_seg, y_raw, aux_df)

            ok_plot = np.isfinite(y_seg)
            if np.any(ok_plot):
                ax.plot(
                    t_h[ok_plot],
                    y_seg[ok_plot],
                    label="Jindrich HR from RAW PAT",
                    linewidth=1.0,
                    zorder=3,
                )

            _stats_line("PAT", y_seg)

    ax.set_ylabel("HR [bpm]")
    ax.grid(True)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
        fontsize=9,
    )
    # Show summary like your ΔHR plot (top-left text)
    if summary_lines:
        ax.text(
            0.01,
            0.92,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            fontsize=9,
            va="top",
        )

    # Use NON-excluded min/max for ylim
    if seg_hr_min is not None and seg_hr_max is not None:
        margin = 0.1 * (seg_hr_max - seg_hr_min + 1e-6)
        ax.set_ylim(seg_hr_min - margin, seg_hr_max + margin)

    return seg_hr_min, seg_hr_max







def _plot_segment_hrv(
    ax: plt.Axes,
    t_hrv: np.ndarray,
    hrv_clean: np.ndarray,
    hrv_raw: Optional[np.ndarray],
    seg_start_sec: float,
    seg_end_sec: float,
    exclusion_zones: List[Tuple[float, float, str]],
    t_seg_h_start: float,
    t_seg_h_end: float,
    aux_df: Optional["pd.DataFrame"],
) -> tuple[Optional[float], Optional[float]]:
    _add_exclusion_spans(ax, exclusion_zones, t_seg_h_start, t_seg_h_end, label_once=True)

    hrv_ymin = hrv_ymax = None
    use_raw = hrv_raw is not None and np.size(hrv_raw) == np.size(hrv_clean)

    mask = (t_hrv >= seg_start_sec) & (t_hrv <= seg_end_sec)
    if np.any(mask):
        th = t_hrv[mask] / 3600.0

        t_sec_seg = t_hrv[mask].astype(float)

        legend_patches: List[Patch] = []

        # -------------------------------------------------
        # GRAY = sleep-stage masking (policy)
        # -------------------------------------------------
        if aux_df is not None and bool(getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False)):
            m_sleep_keep = sleep_mask.build_sleep_include_mask_for_times(t_sec_seg, aux_df)
            if m_sleep_keep is not None:
                _shade_masked_regions(
                    ax,
                    t_sec=t_sec_seg,
                    masked=~m_sleep_keep,
                    color="0.7",
                    alpha=0.18,
                )
                legend_patches.append(Patch(facecolor="0.7", alpha=0.18, label="Sleep masked"))

        # -------------------------------------------------
        # RED = event windows (HRV exclusion logic - events only, gated consistent)
        # IMPORTANT: this MUST match your RR exclusion event windows (pre/post)
        # -------------------------------------------------
        if aux_df is not None:
            m_evt_keep = io_aux_csv.build_time_exclusion_mask(t_sec_seg, aux_df)

            if m_evt_keep is not None:
                _shade_masked_regions(
                    ax,
                    t_sec=t_sec_seg,
                    masked=~m_evt_keep,
                    color="tab:red",
                    alpha=0.12,
                )
                legend_patches.append(Patch(facecolor="tab:red", alpha=0.12, label="Events excluded"))

        # -------------------------------------------------
        # BLUE = RR-only exclusions (raw finite, clean NaN)
        # -------------------------------------------------
        if use_raw:
            yc_seg = hrv_clean[mask]
            yr_seg = hrv_raw[mask]
            rr_only_excluded = np.isfinite(yr_seg) & ~np.isfinite(yc_seg)

            if np.any(rr_only_excluded):
                _shade_masked_regions(
                    ax,
                    t_sec=t_sec_seg,
                    masked=rr_only_excluded,
                    color="tab:blue",
                    alpha=0.22,
                )
                legend_patches.append(Patch(facecolor="tab:blue", alpha=0.22, label="RR rejected"))

        # 4) Raw HRV missing (no RMSSD computed at all) -> very light gray
        if use_raw:
            yc_seg = hrv_clean[mask]
            yr_seg = hrv_raw[mask]

            raw_missing = ~np.isfinite(yr_seg)
            if np.any(raw_missing):
                _shade_masked_regions(
                    ax,
                    t_sec=t_sec_seg,
                    masked=raw_missing,
                    color="gold",  # or "#FFD700"
                    alpha=0.45,  # more visible
                )
                legend_patches.append(
                    Patch(facecolor="gold", alpha=0.45, label="Raw HRV missing")
                )

        if use_raw:
            yr = hrv_raw[mask]
            okr = np.isfinite(yr)
            if np.any(okr):
                ax.plot(
                    th[okr],
                    yr[okr],
                    label="Raw HRV (Unmasked)",
                    linewidth=1.0,
                    linestyle="--",
                    color="tab:gray",
                    zorder=1,
                )

        yc = hrv_clean[mask]
        okc = np.isfinite(yc)
        if np.any(okc):
            ax.plot(th[okc], yc[okc], label="Clean HRV (Masked)", linewidth=1.5, zorder=2)
            y_min, y_max = ax.get_ylim()
            hrv_ymin, hrv_ymax = float(y_min), float(y_max)
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

                y_min, y_max = ax.get_ylim()
                hrv_ymin, hrv_ymax = float(y_min), float(y_max)

    ax.set_ylabel("RMSSD [ms]")
    ax.grid(True)


    # Ensure shading appears in legend
    # Build a single legend outside-right (lines + selected patches)
    handles, labels = ax.get_legend_handles_labels()

    # OPTIONAL: if you still want patches, keep only a couple (or none)
    # e.g. keep events + sleep only:
    if "legend_patches" in locals() and legend_patches:
        keep_patch_labels = {"Events excluded", "Sleep masked"}  # adjust or empty set()
        for p in legend_patches:
            if p.get_label() in keep_patch_labels:
                handles.append(p)
                labels.append(p.get_label())

    # De-duplicate
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen and l != "_nolegend_":
            h2.append(h)
            l2.append(l)
            seen.add(l)

    ax.legend(
        h2, l2,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
        fontsize=9,
    )

    return hrv_ymin, hrv_ymax

def _mask_keep_nonexcluded(
    t_sec: np.ndarray,
    exclusion_zones: List[Tuple[float, float, str]],
) -> np.ndarray:
    """
    Returns a boolean mask (same length as t_sec) that is True for samples that are NOT inside exclusion_zones.

    Assumes exclusion_zones times are in HOURS-from-start (same convention used by _add_exclusion_spans),
    while t_sec is in SECONDS-from-start.
    """
    keep = np.ones_like(t_sec, dtype=bool)
    if not exclusion_zones:
        return keep

    for z0_h, z1_h, _label in exclusion_zones:
        z0 = float(z0_h) * 3600.0
        z1 = float(z1_h) * 3600.0
        if z1 < z0:
            z0, z1 = z1, z0
        keep &= ~((t_sec >= z0) & (t_sec <= z1))

    return keep



def _plot_segment_delta_hr(
    ax: plt.Axes,
    t_hr_edf: Optional[np.ndarray],
    delta_hr_edf: Optional[np.ndarray],
    t_hr_calc: Optional[np.ndarray],
    delta_hr_calc: Optional[np.ndarray],
    delta_hr_edf_evt: Optional[np.ndarray],
    delta_hr_calc_evt: Optional[np.ndarray],
    seg_start_sec: float,
    seg_end_sec: float,
    exclusion_zones: List[Tuple[float, float, str]],
    t_seg_h_start: float,
    t_seg_h_end: float,
    aux_df: Optional["pd.DataFrame"],
) -> tuple[Optional[float], Optional[float]]:
    _add_exclusion_spans(ax, exclusion_zones, t_seg_h_start, t_seg_h_end, label_once=True)

    y_min = y_max = None
    summary_lines: List[str] = []

    def _stats_line(name: str, y_seg: np.ndarray) -> None:
        ok = np.isfinite(y_seg)
        n_total = int(y_seg.size)
        n_used = int(np.count_nonzero(ok))

        if n_used == 0:
            summary_lines.append(f"{name}: no finite samples after masking (used 0/{n_total})")
            return

        yy = y_seg[ok]
        summary_lines.append(
            f"{name}: n={n_used}/{n_total} | "
            f"min={np.min(yy):.1f}, max={np.max(yy):.1f}, "
            f"mean={np.mean(yy):.1f}, median={np.median(yy):.1f}, std={np.std(yy):.1f}"
        )

    # ---- EDF ΔHR ----
    if t_hr_edf is not None:
        mask_seg = (t_hr_edf >= seg_start_sec) & (t_hr_edf <= seg_end_sec)
        if np.any(mask_seg):

            t_sec_seg = t_hr_edf[mask_seg]
            th = t_sec_seg / 3600.0

            # 1) FULL thin
            if delta_hr_edf is not None and np.size(delta_hr_edf) == np.size(t_hr_edf):
                y_full = delta_hr_edf[mask_seg].astype(float)
                if np.any(np.isfinite(y_full)):
                    ax.plot(
                        th,
                        np.ma.masked_invalid(y_full),
                        label="ΔHR EDF (full)",
                        linewidth=0.8,
                        alpha=0.45,
                        linestyle="--",
                        zorder=1,
                    )

            # 2) EVENT-ONLY thick
            if delta_hr_edf_evt is not None and np.size(delta_hr_edf_evt) == np.size(t_hr_edf):
                y_evt = delta_hr_edf_evt[mask_seg].astype(float)
                if np.any(np.isfinite(y_evt)):
                    ax.plot(
                        th,
                        np.ma.masked_invalid(y_evt),
                        label="ΔHR EDF (events)",
                        linewidth=2.2,
                        alpha=0.95,
                        zorder=3,
                    )

    # ---- PAT ΔHR ----
    if t_hr_calc is not None:
        mask_seg = (t_hr_calc >= seg_start_sec) & (t_hr_calc <= seg_end_sec)
        if np.any(mask_seg):

            t_sec_seg = t_hr_calc[mask_seg]
            th = t_sec_seg / 3600.0

            # 1) FULL thin
            if delta_hr_calc is not None and np.size(delta_hr_calc) == np.size(t_hr_calc):
                y_full = delta_hr_calc[mask_seg].astype(float)
                ok = np.isfinite(y_full)
                if np.any(ok):
                    ax.plot(th, np.ma.masked_invalid(y_full),
                            label="ΔHR PAT (full)",
                            linewidth=0.9, alpha=0.45, linestyle="--", zorder=2)

            # 2) EVENT-ONLY thick
            if delta_hr_calc_evt is not None and np.size(delta_hr_calc_evt) == np.size(t_hr_calc):
                y_evt = delta_hr_calc_evt[mask_seg].astype(float)
                ok2 = np.isfinite(y_evt)
                if np.any(ok2):
                    ax.plot(th, np.ma.masked_invalid(y_evt),
                            label="ΔHR PAT (events)",
                            linewidth=2.6, alpha=0.95, zorder=4)

    ax.set_ylabel("ΔHR [bpm]")
    ax.grid(True)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
        fontsize=9,
    )

    # Zero line (put it here so it’s always visible)
    ax.axhline(0.0, linewidth=0.8, alpha=0.5, zorder=0)

    if not summary_lines:
        ax.text(
            0.01, 0.92,
            "ΔHR: no data in this segment",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
        )
    else:
        ax.text(
            0.01, 0.92,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            fontsize=9,
            va="top",
        )

    if y_min is not None and y_max is not None:
        margin = 0.15 * (y_max - y_min + 1e-6)
        ax.set_ylim(y_min - margin, y_max + margin)

    return y_min, y_max




def _plot_segment_pat_amp(
    ax: plt.Axes,
    t_pat_amp: np.ndarray,
    pat_amp: np.ndarray,
    seg_start_sec: float,
    seg_end_sec: float,
    exclusion_zones: List[Tuple[float, float, str]],
    t_seg_h_start: float,
    t_seg_h_end: float,
) -> tuple[Optional[float], Optional[float]]:
    _add_exclusion_spans(ax, exclusion_zones, t_seg_h_start, t_seg_h_end, label_once=True)

    y_min = y_max = None

    mask = (t_pat_amp >= seg_start_sec) & (t_pat_amp <= seg_end_sec)
    if np.any(mask):
        th = t_pat_amp[mask] / 3600.0
        y = pat_amp[mask].astype(float)
        ok = np.isfinite(y)
        if np.any(ok):
            ax.plot(th[ok], y[ok], label="DERIVED_PAT_AMP", linewidth=1.0, zorder=1)
            y_min = float(np.nanmin(y[ok]))
            y_max = float(np.nanmax(y[ok]))
            if np.isfinite(y_min) and np.isfinite(y_max):
                margin = 0.1 * max(1e-6, y_max - y_min)
                ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_ylabel("PAT amp")
    ax.grid(True)
    ax.legend(loc="upper right")
    return y_min, y_max


def _overlay_events_on_axes(
    aux_df: Optional["pd.DataFrame"],
    seg_start_sec: float,
    seg_end_sec: float,
    ax_hr: plt.Axes,
    ax_hrv: Optional[plt.Axes],
    ax_amp: Optional[plt.Axes],
    ax_delta: Optional[plt.Axes],
    hr_ylim: tuple[float, float],
    hrv_ylim: Optional[tuple[float, float]],
    amp_ylim: Optional[tuple[float, float]],
    delta_ylim: Optional[tuple[float, float]],
    event_spec: List[EventSpec] = DEFAULT_EVENT_PLOT_SPEC,
) -> None:
    if aux_df is None:
        return

    time_col = getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec")
    if time_col not in aux_df.columns:
        return

    mask = (aux_df[time_col] >= seg_start_sec) & (aux_df[time_col] <= seg_end_sec)
    if not mask.any():
        return

    seg = aux_df.loc[mask]

    used_hr = set()
    used_hrv = set()
    used_amp = set()
    used_delta = set()

    hr_ymin, hr_ymax = hr_ylim
    hrv_ymin, hrv_ymax = hrv_ylim if hrv_ylim is not None else (None, None)
    amp_ymin, amp_ymax = amp_ylim if amp_ylim is not None else (None, None)
    delta_ymin, delta_ymax = delta_ylim if delta_ylim is not None else (None, None)


    def _scatter_desat(ax: plt.Axes, t_h: np.ndarray, y0: float, y1: float, color: str, label: str) -> None:
        y_desat = y0 + 0.03 * (y1 - y0)
        ax.scatter(
            t_h,
            np.full_like(t_h, y_desat, dtype=float),
            marker="v",
            s=25,
            color=color,
            alpha=0.85,
            label=label,
            zorder=5,
        )

    for spec in event_spec:
        if spec.col not in seg.columns:
            continue

        m = seg[spec.col] == 1
        if not m.any():
            continue

        t_evt_h = seg.loc[m, time_col].to_numpy(float) / 3600.0

        label_hr = spec.label if spec.label not in used_hr else "_nolegend_"
        used_hr.add(spec.label)

        if spec.col == "desat_flag":
            _scatter_desat(ax_hr, t_evt_h, hr_ymin, hr_ymax, spec.color, label_hr)
        else:
            first = (label_hr != "_nolegend_")
            for x in t_evt_h:
                ax_hr.axvline(
                    x,
                    color=spec.color,
                    linestyle="-",
                    linewidth=2.0,
                    alpha=0.9,
                    label=spec.label if first else "_nolegend_",
                    zorder=5,
                )
                first = False

        if ax_hrv is not None and hrv_ymin is not None and hrv_ymax is not None:
            label_hrv = spec.label if spec.label not in used_hrv else "_nolegend_"
            used_hrv.add(spec.label)

            if spec.col == "desat_flag":
                _scatter_desat(ax_hrv, t_evt_h, hrv_ymin, hrv_ymax, spec.color, label_hrv)
            else:
                first = (label_hrv != "_nolegend_")
                for x in t_evt_h:
                    ax_hrv.axvline(
                        x,
                        color=spec.color,
                        linestyle="-",
                        linewidth=1.8,
                        alpha=0.85,
                        label=spec.label if first else "_nolegend_",
                        zorder=5,
                    )
                    first = False

        if ax_amp is not None and amp_ymin is not None and amp_ymax is not None:
            label_amp = spec.label if spec.label not in used_amp else "_nolegend_"
            used_amp.add(spec.label)

            if spec.col == "desat_flag":
                _scatter_desat(ax_amp, t_evt_h, amp_ymin, amp_ymax, spec.color, label_amp)
            else:
                first = (label_amp != "_nolegend_")
                for x in t_evt_h:
                    ax_amp.axvline(
                        x,
                        color=spec.color,
                        linestyle="-",
                        linewidth=1.8,
                        alpha=0.85,
                        label=spec.label if first else "_nolegend_",
                        zorder=5,
                    )
                    first = False

        if ax_delta is not None and delta_ymin is not None and delta_ymax is not None:
            label_delta = spec.label if spec.label not in used_delta else "_nolegend_"
            used_delta.add(spec.label)

            if spec.col == "desat_flag":
                _scatter_desat(ax_delta, t_evt_h, delta_ymin, delta_ymax, spec.color, label_delta)
            else:
                first = (label_delta != "_nolegend_")
                for x in t_evt_h:
                    ax_delta.axvline(
                        x,
                        color=spec.color,
                        linestyle="-",
                        linewidth=1.8,
                        alpha=0.85,
                        label=spec.label if first else "_nolegend_",
                        zorder=5,
                    )
                    first = False




def _add_segment_pages_to_pdf(
    pdf: PdfPages,
    *,
    signal_raw: np.ndarray,
    signal_filt: np.ndarray,
    sfreq: float,
    segment_minutes: float,
    title_prefix: str,
    channel_name: str,
    t_hr_calc: Optional[np.ndarray],
    hr_calc: Optional[np.ndarray],
    t_hr_edf: Optional[np.ndarray],
    hr_edf: Optional[np.ndarray],
    t_hrv: Optional[np.ndarray],
    hrv_clean: Optional[np.ndarray],
    hrv_raw: Optional[np.ndarray],
    aux_df: Optional["pd.DataFrame"],
    exclusion_zones: List[Tuple[float, float, str]],
    t_pat_amp: Optional[np.ndarray],
    pat_amp: Optional[np.ndarray],
    delta_hr_calc: Optional[np.ndarray],
    delta_hr_edf: Optional[np.ndarray],
    # NEW: event/desat-only ΔHR arrays (NaN outside event windows)
    delta_hr_calc_evt: Optional[np.ndarray],
    delta_hr_edf_evt: Optional[np.ndarray],
) -> None:
    n_samples = len(signal_raw)
    samples_per_segment = int(segment_minutes * 60.0 * sfreq)
    segment_index = 0

    use_hrv = t_hrv is not None and hrv_clean is not None and np.size(hrv_clean) > 0
    use_pat_amp = (
        t_pat_amp is not None
        and pat_amp is not None
        and np.size(t_pat_amp) > 0
        and np.size(pat_amp) > 0
    )

    for start in range(0, n_samples, samples_per_segment):
        end = min(start + samples_per_segment, n_samples)
        segment_index += 1

        seg_raw = signal_raw[start:end]
        seg_filt = signal_filt[start:end]

        t_seg_sec = np.arange(start, end) / sfreq
        t_seg_h = t_seg_sec / 3600.0

        seg_start_sec = float(t_seg_sec[0])
        seg_end_sec = float(t_seg_sec[-1])
        t_h_start = float(t_seg_h[0])
        t_h_end = float(t_seg_h[-1])

        # ----------------------------
        # ΔHR subplot layout decision
        # ----------------------------
        enable_delta = bool(getattr(config, "ENABLE_DELTA_HR", True))
        delta_mode = str(getattr(config, "DELTA_HR_PLOT_MODE", "subplot")).lower()

        # show ΔHR subplot if ANY ΔHR series exists (full or event-only)
        has_any_delta = (
            (delta_hr_calc is not None and np.size(delta_hr_calc) > 0)
            or (delta_hr_edf is not None and np.size(delta_hr_edf) > 0)
            or (delta_hr_calc_evt is not None and np.size(delta_hr_calc_evt) > 0)
            or (delta_hr_edf_evt is not None and np.size(delta_hr_edf_evt) > 0)
        )

        use_delta_subplot = enable_delta and (delta_mode == "subplot") and has_any_delta

        n_rows = 2 + (1 if use_hrv else 0) + (1 if use_pat_amp else 0) + (1 if use_delta_subplot else 0)
        height_ratios = [2, 1] + ([1] if use_hrv else []) + ([1] if use_pat_amp else []) + ([1] if use_delta_subplot else [])

        fig, axes = plt.subplots(
            n_rows,
            1,
            figsize=(11.69, 8.27),
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        if n_rows == 1:
            axes = [axes]

        ax_pat = axes[0]
        ax_hr = axes[1]

        idx = 2
        ax_hrv = None
        if use_hrv:
            ax_hrv = axes[idx]
            idx += 1

        ax_amp = None
        if use_pat_amp:
            ax_amp = axes[idx]
            idx += 1

        ax_delta = None
        if use_delta_subplot:
            ax_delta = axes[idx]
            idx += 1

        title_lines = []
        if title_prefix:
            title_lines.append(title_prefix)
        if channel_name:
            title_lines.append(channel_name)
        title_lines.append(
            f"Segment {segment_index}: {_h_to_hhmm(t_h_start)}–{_h_to_hhmm(t_h_end)} (hh:mm from start)"
        )
        _plot_segment_pat(ax_pat, t_seg_h, seg_raw, seg_filt, " - ".join(title_lines))

        _plot_segment_hr(
            ax_hr,
            t_hr_edf, hr_edf,
            t_hr_calc, hr_calc,
            seg_start_sec, seg_end_sec,
            exclusion_zones,
            t_h_start, t_h_end,
            aux_df=aux_df,
            t_seg_sec=t_seg_sec,
        )
        hr_ylim = ax_hr.get_ylim()

        hrv_ylim = None
        if use_hrv and ax_hrv is not None and t_hrv is not None and hrv_clean is not None:
            _plot_segment_hrv(
                ax_hrv,
                t_hrv, hrv_clean, hrv_raw,
                seg_start_sec, seg_end_sec,
                exclusion_zones,
                t_h_start, t_h_end,
                aux_df=aux_df,
            )
            hrv_ylim = ax_hrv.get_ylim()

        amp_ylim = None
        if use_pat_amp and ax_amp is not None and t_pat_amp is not None and pat_amp is not None:
            _plot_segment_pat_amp(
                ax_amp,
                t_pat_amp, pat_amp,
                seg_start_sec, seg_end_sec,
                exclusion_zones,
                t_h_start, t_h_end,
            )
            amp_ylim = ax_amp.get_ylim()

        delta_ylim = None
        if use_delta_subplot and ax_delta is not None:
            _plot_segment_delta_hr(
                ax_delta,
                t_hr_edf=t_hr_edf,
                delta_hr_edf=delta_hr_edf,
                t_hr_calc=t_hr_calc,
                delta_hr_calc=delta_hr_calc,
                # NEW:
                delta_hr_edf_evt=delta_hr_edf_evt,
                delta_hr_calc_evt=delta_hr_calc_evt,
                seg_start_sec=seg_start_sec,
                seg_end_sec=seg_end_sec,
                exclusion_zones=exclusion_zones,
                t_seg_h_start=t_h_start,
                t_seg_h_end=t_h_end,
                aux_df=aux_df,
            )
            delta_ylim = ax_delta.get_ylim()

        _overlay_events_on_axes(
            aux_df,
            seg_start_sec,
            seg_end_sec,
            ax_hr=ax_hr,
            ax_hrv=ax_hrv if use_hrv else None,
            ax_amp=ax_amp if use_pat_amp else None,
            ax_delta=ax_delta if use_delta_subplot else None,
            hr_ylim=hr_ylim,
            hrv_ylim=hrv_ylim,
            amp_ylim=amp_ylim,
            delta_ylim=delta_ylim,
        )

        # bottom x-label
        if ax_delta is not None:
            ax_delta.set_xlabel("Time (hours from recording start)")
        elif ax_amp is not None:
            ax_amp.set_xlabel("Time (hours from recording start)")
        elif ax_hrv is not None:
            ax_hrv.set_xlabel("Time (hours from recording start)")
        else:
            ax_hr.set_xlabel("Time (hours from recording start)")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

