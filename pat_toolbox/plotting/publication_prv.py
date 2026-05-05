from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .. import config, masking, paths, sleep_mask
from ..core.pr_cleaning import detect_pat_peaks
from ..io_aux_csv import compute_sleep_timing_from_aux

if TYPE_CHECKING:
    import pandas as pd


def _window_valid_fraction(t_sec: Optional[np.ndarray], y: Optional[np.ndarray], start_sec: float, end_sec: float) -> float:
    if t_sec is None or y is None:
        return 0.0
    t_sec = np.asarray(t_sec, dtype=float)
    y = np.asarray(y, dtype=float)
    if t_sec.size == 0 or y.size != t_sec.size:
        return 0.0
    mask = (t_sec >= start_sec) & (t_sec <= end_sec)
    if not np.any(mask):
        return 0.0
    vals = y[mask]
    return float(np.mean(np.isfinite(vals))) if vals.size else 0.0


def _score_candidate_window(
    start_sec: float,
    end_sec: float,
    *,
    t_hr: Optional[np.ndarray],
    hr: Optional[np.ndarray],
    t_prv: Optional[np.ndarray],
    prv_rmssd: Optional[np.ndarray],
    prv_tv: Optional[Dict[str, np.ndarray]],
    prv_mask_info: Optional[Dict[str, object]],
) -> float:
    hr_frac = _window_valid_fraction(t_hr, hr, start_sec, end_sec)
    rmssd_frac = _window_valid_fraction(t_prv, prv_rmssd, start_sec, end_sec)

    sdnn_frac = 0.0
    lf_frac = 0.0
    hf_frac = 0.0
    ratio_frac = 0.0
    if isinstance(prv_tv, dict):
        sdnn_frac = _window_valid_fraction(t_prv, prv_tv.get("sdnn_ms"), start_sec, end_sec)
        t_spectral = prv_tv.get("spectral_t_sec")
        lf_frac = _window_valid_fraction(t_spectral, prv_tv.get("lf_fixed"), start_sec, end_sec)
        hf_frac = _window_valid_fraction(t_spectral, prv_tv.get("hf_fixed"), start_sec, end_sec)
        ratio_frac = _window_valid_fraction(t_spectral, prv_tv.get("lf_hf_fixed"), start_sec, end_sec)

    keep_frac = 0.0
    if isinstance(prv_mask_info, dict):
        combined_keep = prv_mask_info.get("combined_keep")
        if isinstance(combined_keep, np.ndarray) and t_prv is not None and np.size(combined_keep) == np.size(t_prv):
            keep_frac = _window_valid_fraction(t_prv, np.asarray(combined_keep, dtype=float), start_sec, end_sec)

    # Favor windows with broad metric coverage, slightly weighting the methods
    # panels (PAT->HR->RMSSD/SDNN) more than the spectral summaries.
    return (
        1.0 * hr_frac
        + 1.5 * rmssd_frac
        + 1.5 * sdnn_frac
        + 1.0 * lf_frac
        + 1.0 * hf_frac
        + 1.0 * ratio_frac
        + 1.5 * keep_frac
    )


def _select_best_nrem_window(
    duration_sec: float,
    aux_df: Optional["pd.DataFrame"],
    *,
    t_hr: Optional[np.ndarray],
    hr: Optional[np.ndarray],
    t_prv: Optional[np.ndarray],
    prv_rmssd: Optional[np.ndarray],
    prv_tv: Optional[Dict[str, np.ndarray]],
    prv_mask_info: Optional[Dict[str, object]],
) -> Optional[tuple[float, float]]:
    if aux_df is None or duration_sec <= 0:
        return None

    window_sec = float(getattr(config, "PUBLICATION_PRV_SEGMENT_MIN_SEC", 600.0))
    step_sec = float(getattr(config, "PUBLICATION_PRV_SELECTION_STEP_SEC", 30.0))
    if duration_sec < window_sec:
        return None

    candidate_starts = np.arange(0.0, duration_sec - window_sec + 1e-9, max(1.0, step_sec), dtype=float)
    if candidate_starts.size == 0:
        candidate_starts = np.array([0.0], dtype=float)

    best_window: Optional[tuple[float, float]] = None
    best_score = -np.inf
    include_set = {1, 2}
    policy = masking.policy_from_config(include_stages=include_set, force_sleep=True)

    for start_sec in candidate_starts:
        end_sec = start_sec + window_sec
        t_eval = np.arange(start_sec, end_sec + 1e-9, 1.0, dtype=float)
        sleep_keep = sleep_mask.build_sleep_include_mask_for_times(
            t_eval,
            aux_df,
            include_set=include_set,
            ignore_config=True,
        )
        if sleep_keep is None or not np.all(np.asarray(sleep_keep, dtype=bool)):
            continue

        score = _score_candidate_window(
            start_sec,
            end_sec,
            t_hr=t_hr,
            hr=hr,
            t_prv=t_prv,
            prv_rmssd=prv_rmssd,
            prv_tv=prv_tv,
            prv_mask_info=prv_mask_info,
        )

        # Small tie-break bonus for windows with fewer exclusion events.
        event_times = np.asarray(masking._event_times_from_aux(aux_df, policy.exclusion_columns), dtype=float)
        if event_times.size > 0:
            score -= 0.02 * float(np.sum((event_times >= start_sec) & (event_times <= end_sec)))

        if score > best_score:
            best_score = score
            best_window = (float(start_sec), float(end_sec))

    return best_window


def _plot_masked_line(ax, t_sec: np.ndarray, y: np.ndarray, *, color: str, linewidth: float = 1.2, label: Optional[str] = None) -> None:
    t_sec = np.asarray(t_sec, dtype=float)
    y = np.asarray(y, dtype=float)
    if t_sec.size == 0 or y.size != t_sec.size:
        return
    ax.plot(t_sec / 60.0, np.ma.masked_invalid(y), color=color, linewidth=linewidth, label=label)


def _plot_window_support_series(
    ax,
    centers_sec: np.ndarray,
    y: np.ndarray,
    *,
    window_sec: float,
    start_sec: float,
    color: str,
    linewidth: float = 1.2,
    label: Optional[str] = None,
) -> None:
    centers_sec = np.asarray(centers_sec, dtype=float)
    y = np.asarray(y, dtype=float)
    if centers_sec.size == 0 or y.size != centers_sec.size:
        return
    ok = np.isfinite(centers_sec) & np.isfinite(y)
    if not np.any(ok):
        return
    centers_sec = centers_sec[ok]
    y = y[ok]
    ax.plot((centers_sec - start_sec) / 60.0, y, color=color, linewidth=linewidth, marker="o", markersize=3.0, label=label)


def _select_real_neighbor_mask(t_sec: np.ndarray, y: np.ndarray, start_sec: float, end_sec: float) -> np.ndarray:
    t_sec = np.asarray(t_sec, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(t_sec) & np.isfinite(y)
    idx_all = np.flatnonzero(mask)
    if idx_all.size == 0:
        return np.zeros_like(mask, dtype=bool)

    inside = idx_all[(t_sec[idx_all] >= start_sec) & (t_sec[idx_all] <= end_sec)]
    if inside.size == 0:
        return np.zeros_like(mask, dtype=bool)

    keep = np.zeros_like(mask, dtype=bool)
    keep[inside] = True

    before = idx_all[t_sec[idx_all] < start_sec]
    after = idx_all[t_sec[idx_all] > end_sec]
    if before.size > 0:
        keep[before[-1]] = True
    if after.size > 0:
        keep[after[0]] = True
    return keep


def _select_pat_zoom_window(
    start_sec: float,
    end_sec: float,
    peak_times_sec: np.ndarray,
    *,
    zoom_sec: float = 20.0,
) -> tuple[float, float]:
    if end_sec <= start_sec:
        return start_sec, end_sec
    if (end_sec - start_sec) <= zoom_sec:
        return start_sec, end_sec

    candidate_starts = np.arange(start_sec, end_sec - zoom_sec + 1e-9, 5.0, dtype=float)
    if candidate_starts.size == 0:
        mid = 0.5 * (start_sec + end_sec)
        return mid - 0.5 * zoom_sec, mid + 0.5 * zoom_sec

    best_start = float(candidate_starts[0])
    best_score = -np.inf
    for cand_start in candidate_starts:
        cand_end = cand_start + zoom_sec
        n_peaks = float(np.sum((peak_times_sec >= cand_start) & (peak_times_sec <= cand_end)))
        # Prefer windows with enough peaks and centered away from the edges.
        center_penalty = abs((cand_start + 0.5 * zoom_sec) - (start_sec + end_sec) * 0.5) / max(1.0, end_sec - start_sec)
        score = n_peaks - 0.25 * center_penalty
        if score > best_score:
            best_score = score
            best_start = float(cand_start)
    return best_start, best_start + zoom_sec


def _add_pat_zoom_inset(
    ax,
    *,
    signal_raw: np.ndarray,
    signal_filt: np.ndarray,
    sfreq: float,
    peaks: np.ndarray,
    start_sec: float,
    end_sec: float,
) -> None:
    if not bool(getattr(config, "PUBLICATION_PRV_SHOW_PEAK_ZOOM", True)):
        return

    peak_times_sec = peaks.astype(float) / sfreq
    zoom_sec = float(getattr(config, "PUBLICATION_PRV_PEAK_ZOOM_SEC", 20.0))
    if zoom_sec <= 0:
        return
    if (end_sec - start_sec) <= zoom_sec:
        return

    zoom_start_sec, zoom_end_sec = _select_pat_zoom_window(start_sec, end_sec, peak_times_sec, zoom_sec=zoom_sec)
    zoom_start_idx = max(0, int(np.floor(zoom_start_sec * sfreq)))
    zoom_end_idx = min(signal_raw.size, int(np.ceil(zoom_end_sec * sfreq)))
    if zoom_end_idx <= zoom_start_idx:
        return

    t_zoom_sec = np.arange(zoom_start_idx, zoom_end_idx, dtype=float) / sfreq
    t_zoom_local = t_zoom_sec - zoom_start_sec
    raw_zoom = signal_raw[zoom_start_idx:zoom_end_idx]
    filt_zoom = signal_filt[zoom_start_idx:zoom_end_idx]
    peak_mask = (peak_times_sec >= zoom_start_sec) & (peak_times_sec <= zoom_end_sec)
    peak_times_local = peak_times_sec[peak_mask] - zoom_start_sec
    peak_vals = signal_filt[np.clip(peaks[peak_mask], 0, signal_filt.size - 1)] if np.any(peak_mask) else np.array([], dtype=float)

    axins = inset_axes(ax, width="36%", height="58%", loc="lower left", borderpad=1.2)
    axins.plot(t_zoom_local, raw_zoom, color="0.8", linewidth=0.8)
    axins.plot(t_zoom_local, filt_zoom, color="black", linewidth=1.0)
    if peak_times_local.size:
        axins.scatter(peak_times_local, peak_vals, s=18, color="tab:red", zorder=4)
    axins.set_xlim(0.0, max(0.0, zoom_end_sec - zoom_start_sec))
    axins.set_title(f"Peak-detection zoom ({zoom_end_sec - zoom_start_sec:.0f} s)", fontsize=8, pad=3)
    axins.set_xlabel("s", fontsize=7, labelpad=1)
    axins.tick_params(axis="both", labelsize=7, length=2)
    axins.grid(True, alpha=0.25)

    y_all = np.concatenate([raw_zoom[np.isfinite(raw_zoom)], filt_zoom[np.isfinite(filt_zoom)]])
    if y_all.size:
        y0 = float(np.min(y_all))
        y1 = float(np.max(y_all))
        if y1 > y0:
            m = 0.08 * (y1 - y0)
            axins.set_ylim(y0 - m, y1 + m)

    x0 = (zoom_start_sec - start_sec) / 60.0
    x1 = (zoom_end_sec - start_sec) / 60.0
    if x1 > x0:
        ax.axvspan(x0, x1, color="tab:orange", alpha=0.10, zorder=0)


def _plot_local_hypnogram(ax, aux_df: Optional["pd.DataFrame"], start_sec: float, end_sec: float) -> None:
    if aux_df is None:
        ax.text(0.5, 0.5, "Hypnogram unavailable", ha="center", va="center", transform=ax.transAxes)
        return

    aux_df = sleep_mask.ensure_stage_code_column(aux_df)
    time_col = getattr(config, "AUX_CSV_TIME_SEC_COLUMN", "time_sec")
    stage_col = getattr(config, "AUX_CSV_STAGE_CODE_COLUMN", "stage_code")
    if time_col not in aux_df.columns or stage_col not in aux_df.columns:
        ax.text(0.5, 0.5, "Hypnogram unavailable", ha="center", va="center", transform=ax.transAxes)
        return

    t = aux_df[time_col].to_numpy(dtype=float)
    s = aux_df[stage_col].to_numpy(dtype=float)
    mask = np.isfinite(t) & np.isfinite(s) & (t >= start_sec) & (t <= end_sec)
    if not np.any(mask):
        ax.text(0.5, 0.5, "Hypnogram unavailable", ha="center", va="center", transform=ax.transAxes)
        return

    t = t[mask]
    s = np.round(s[mask]).astype(int)
    order = np.argsort(t)
    t = t[order]
    s = s[order]

    local_min = (t - start_sec) / 60.0
    if local_min.size == 1:
        edges = np.array([local_min[0], local_min[0] + 1.0 / 60.0], dtype=float)
    else:
        d = np.diff(local_min)
        dpos = d[np.isfinite(d) & (d > 0)]
        step_min = float(np.median(dpos)) if dpos.size else (1.0 / 60.0)
        edges = np.empty(local_min.size + 1, dtype=float)
        edges[:-1] = local_min
        edges[-1] = local_min[-1] + step_min

    y_map = {3: 3, 0: 2, 1: 1, 2: 0}
    stage_colors = {3: "#d62728", 0: "#ff7f0e", 1: "#1f77b4", 2: "#e377c2"}
    y = np.array([y_map.get(int(x), np.nan) for x in s], dtype=float)
    ok = np.isfinite(y)
    local_min = local_min[ok]
    s = s[ok]
    y = y[ok]
    if local_min.size == 0:
        ax.text(0.5, 0.5, "Hypnogram unavailable", ha="center", va="center", transform=ax.transAxes)
        return

    for i in range(y.size):
        x0 = edges[i]
        x1 = edges[i + 1]
        yi = y[i]
        si = s[i]
        color = stage_colors.get(int(si), "black")
        ax.hlines(yi, x0, x1, colors=color, linewidth=4.0, zorder=3)
        if i < y.size - 1:
            next_color = stage_colors.get(int(s[i + 1]), color)
            ax.vlines(x1, yi, y[i + 1], colors=next_color, linewidth=1.0, alpha=0.9, zorder=3)

    sleep_timing = compute_sleep_timing_from_aux(aux_df)
    if sleep_timing:
        line_specs = [
            (sleep_timing.get("sleep_onset_rel_sec"), "tab:green", "--"),
            (sleep_timing.get("sleep_midpoint_rel_sec"), "tab:purple", "-"),
            (sleep_timing.get("sleep_end_rel_sec"), "tab:red", "--"),
        ]
        for x_sec, color, style in line_specs:
            if x_sec is None or not np.isfinite(x_sec):
                continue
            if start_sec <= float(x_sec) <= end_sec:
                ax.axvline((float(x_sec) - start_sec) / 60.0, color=color, linestyle=style, linewidth=1.5, alpha=0.9, zorder=4)

    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(["REM", "Wake", "Light", "Deep"])
    ax.set_ylim(-0.7, 3.7)
    ax.set_xlim(0.0, max(0.0, (end_sec - start_sec) / 60.0))
    ax.grid(True, axis="x", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_publication_prv_png(
    *,
    edf_base: str,
    signal_raw: np.ndarray,
    signal_filt: np.ndarray,
    sfreq: float,
    t_hr: Optional[np.ndarray],
    hr: Optional[np.ndarray],
    t_prv: Optional[np.ndarray],
    prv_rmssd: Optional[np.ndarray],
    prv_tv: Optional[Dict[str, np.ndarray]],
    aux_df: Optional["pd.DataFrame"],
    prv_mask_info: Optional[Dict[str, object]],
) -> Optional[Path]:
    if not bool(getattr(config, "EXPORT_PUBLICATION_PRV_PNG", False)):
        return None
    if signal_raw.size == 0 or signal_filt.size != signal_raw.size or sfreq <= 0:
        return None
    if t_prv is None or prv_rmssd is None or not isinstance(prv_tv, dict):
        return None

    duration_sec = float(signal_raw.size / sfreq)
    best_window = _select_best_nrem_window(
        duration_sec,
        aux_df,
        t_hr=t_hr,
        hr=hr,
        t_prv=t_prv,
        prv_rmssd=prv_rmssd,
        prv_tv=prv_tv,
        prv_mask_info=prv_mask_info,
    )
    if best_window is None:
        return None

    start_sec, end_sec = best_window
    start_idx = max(0, int(np.floor(start_sec * sfreq)))
    end_idx = min(signal_raw.size, int(np.ceil(end_sec * sfreq)))
    t_pat_seg_sec = np.arange(start_idx, end_idx, dtype=float) / sfreq
    raw_seg = signal_raw[start_idx:end_idx]
    filt_seg = signal_filt[start_idx:end_idx]

    _unused_filt, peaks = detect_pat_peaks(signal_raw, sfreq)
    peak_times_sec = peaks.astype(float) / sfreq
    peak_mask = (peak_times_sec >= start_sec) & (peak_times_sec <= end_sec)
    peak_times_seg_min = (peak_times_sec[peak_mask] - start_sec) / 60.0
    peak_vals = signal_filt[np.clip(peaks[peak_mask], 0, signal_filt.size - 1)] if np.any(peak_mask) else np.array([], dtype=float)

    t_local_min = (t_pat_seg_sec - start_sec) / 60.0

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(10.5, 13.5),
        sharex=True,
        gridspec_kw={"height_ratios": [0.7, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0]},
    )
    axes = list(np.atleast_1d(axes))

    ax_hyp, ax_pat, ax_hr, ax_rmssd, ax_sdnn, ax_lfhf, ax_ratio = axes

    _plot_local_hypnogram(ax_hyp, aux_df, start_sec, end_sec)
    ax_hyp.set_ylabel("Stage")

    ax_pat.plot(t_local_min, raw_seg, color="0.75", linewidth=0.8, label="Raw PAT")
    ax_pat.plot(t_local_min, filt_seg, color="black", linewidth=1.0, label="Filtered PAT")
    if peak_times_seg_min.size:
        ax_pat.scatter(peak_times_seg_min, peak_vals, s=14, color="tab:red", zorder=4, label="Detected peaks")
    _add_pat_zoom_inset(
        ax_pat,
        signal_raw=signal_raw,
        signal_filt=signal_filt,
        sfreq=sfreq,
        peaks=peaks,
        start_sec=start_sec,
        end_sec=end_sec,
    )
    ax_pat.set_ylabel("PAT [a.u.]")
    ax_pat.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_pat.grid(True, alpha=0.35)

    if t_hr is not None and hr is not None:
        mask_hr = (t_hr >= start_sec) & (t_hr <= end_sec)
        _plot_masked_line(ax_hr, t_hr[mask_hr] - start_sec, np.asarray(hr)[mask_hr], color="tab:blue", linewidth=1.2)
    ax_hr.set_ylabel("HR [bpm]")
    ax_hr.grid(True, alpha=0.35)

    mask_prv = (t_prv >= start_sec) & (t_prv <= end_sec)
    _plot_masked_line(ax_rmssd, t_prv[mask_prv] - start_sec, np.asarray(prv_rmssd)[mask_prv], color="tab:green", linewidth=1.2)
    ax_rmssd.set_ylabel("RMSSD [ms]")
    ax_rmssd.grid(True, alpha=0.35)

    sdnn = prv_tv.get("sdnn_ms")
    if sdnn is not None:
        _plot_masked_line(ax_sdnn, t_prv[mask_prv] - start_sec, np.asarray(sdnn)[mask_prv], color="tab:green", linewidth=1.2)
    ax_sdnn.set_ylabel("SDNN [ms]")
    ax_sdnn.grid(True, alpha=0.35)

    t_spectral = np.asarray(prv_tv.get("spectral_t_sec", np.array([], dtype=float)), dtype=float)
    spectral_window_sec = float(np.asarray(prv_tv.get("spectral_window_sec", np.array([getattr(config, "PRV_LFHF_FIXED_WINDOW_SEC", 120.0)])), dtype=float)[0])
    if t_spectral.size > 0:
        lf = prv_tv.get("lf_fixed")
        hf = prv_tv.get("hf_fixed")
        ratio = prv_tv.get("lf_hf_fixed")
        if lf is not None:
            mask_spec_lf = _select_real_neighbor_mask(t_spectral, np.asarray(lf), start_sec, end_sec)
            _plot_window_support_series(ax_lfhf, t_spectral[mask_spec_lf], np.asarray(lf)[mask_spec_lf], window_sec=spectral_window_sec, start_sec=start_sec, color="tab:orange", linewidth=1.2, label="LF")
        if hf is not None:
            mask_spec_hf = _select_real_neighbor_mask(t_spectral, np.asarray(hf), start_sec, end_sec)
            _plot_window_support_series(ax_lfhf, t_spectral[mask_spec_hf], np.asarray(hf)[mask_spec_hf], window_sec=spectral_window_sec, start_sec=start_sec, color="tab:blue", linewidth=1.2, label="HF")
        ax_lfhf.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax_lfhf.set_yscale("log")
        if ratio is not None:
            mask_spec_ratio = _select_real_neighbor_mask(t_spectral, np.asarray(ratio), start_sec, end_sec)
            _plot_window_support_series(ax_ratio, t_spectral[mask_spec_ratio], np.asarray(ratio)[mask_spec_ratio], window_sec=spectral_window_sec, start_sec=start_sec, color="tab:purple", linewidth=1.2)

    ax_lfhf.set_ylabel("LF, HF [ms$^2$]")
    ax_lfhf.grid(True, alpha=0.35)
    ax_ratio.set_ylabel("LF/HF [-]")
    ax_ratio.set_xlabel("Time within selected NREM segment [min]")
    ax_ratio.grid(True, alpha=0.35)
    ax_ratio.set_xlim(0.0, max(0.0, (end_sec - start_sec) / 60.0))

    fig.suptitle(f"{edf_base} - Representative 10 min NREM PAT/PRV Segment", fontsize=14, y=0.995)
    fig.text(
        0.5,
        0.975,
        f"From the original 40 Hz PAT waveform. Segment selected automatically as the highest-coverage contiguous {((end_sec - start_sec) / 60.0):.0f} min NREM interval. {'The PAT panel includes an inset peak-detection zoom. ' if bool(getattr(config, 'PUBLICATION_PRV_SHOW_PEAK_ZOOM', True)) and (end_sec - start_sec) > float(getattr(config, 'PUBLICATION_PRV_PEAK_ZOOM_SEC', 20.0)) else ''}RMSSD/SDNN are shown as native overlapping sliding-window estimates; LF, HF, and LF/HF are shown as native fixed-window estimates.",
        ha="center",
        va="top",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.05, 0.04, 0.98, 0.95))

    out_folder = paths.get_output_folder(getattr(config, "PUBLICATION_PRV_OUTPUT_SUBFOLDER"))
    png_path = out_folder / f"{edf_base}__publication_prv_nrem_10min.png"
    dpi = int(getattr(config, "PUBLICATION_PRV_DPI", 600))
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return png_path


__all__ = ["save_publication_prv_png"]
