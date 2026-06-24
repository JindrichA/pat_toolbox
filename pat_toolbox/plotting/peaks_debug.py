from __future__ import annotations

from pathlib import Path
from typing import Optional, Mapping

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .. import config


def plot_pat_with_peaks_segments_to_pdf(
    signal_raw: np.ndarray,
    signal_filt: np.ndarray,
    peak_indices: np.ndarray,
    sfreq: float,
    pdf_path: Path,
    segment_minutes: Optional[float] = None,
    title_prefix: str = "",
    channel_name: str = "",
    actigraph: Optional[np.ndarray] = None,
    act_sfreq: Optional[float] = None,
    act_label: str = "ACTIGRAPH raw",
    pat_ylim: Optional[tuple[float, float]] = None,
    act_ylim: Optional[tuple[float, float]] = None,
    pwa_debug: Optional[Mapping[str, np.ndarray]] = None,
):
    if segment_minutes is None:
        segment_minutes = getattr(
            config,
            "PAT_PEAK_DEBUG_SEGMENT_MINUTES",
            config.SEGMENT_MINUTES,
        )

    n_samples = len(signal_raw)
    if n_samples == 0 or sfreq <= 0:
        raise ValueError("Signal is empty or sampling frequency invalid.")

    if len(signal_filt) != n_samples:
        raise ValueError("Raw and filtered signal lengths differ.")

    samples_per_segment = int(segment_minutes * 60.0 * sfreq)
    if samples_per_segment <= 0:
        raise ValueError("Computed non-positive samples_per_segment.")

    use_pwa_debug = bool(pwa_debug) and pwa_debug is not None and np.size(pwa_debug.get("signal_smooth", [])) == n_samples

    use_act = (
        actigraph is not None
        and act_sfreq is not None
        and act_sfreq > 0
        and len(actigraph) > 0
    )

    with PdfPages(str(pdf_path)) as pdf:
        segment_index = 0
        for start in range(0, n_samples, samples_per_segment):
            end = min(start + samples_per_segment, n_samples)
            segment_index += 1

            seg_filt = signal_filt[start:end]
            t_seg = np.arange(start, end) / sfreq / 60.0  # minutes

            n_rows = 1 + (1 if use_pwa_debug else 0) + (1 if use_act else 0)
            if n_rows > 1:
                ratios = [2.0]
                if use_pwa_debug:
                    ratios.append(2.0)
                if use_act:
                    ratios.append(1.0)
                fig, axes = plt.subplots(
                    n_rows, 1, figsize=(11.69, 8.27),
                    sharex=True,
                    gridspec_kw={"height_ratios": ratios},
                )
                axes = np.atleast_1d(axes)
                ax = axes[0]
                ax_pwa = axes[1] if use_pwa_debug else None
                ax_act = axes[-1] if use_act else None
            else:
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                ax_pwa = None
                ax_act = None

            title_lines = []
            if title_prefix:
                title_lines.append(title_prefix)
            if channel_name:
                title_lines.append(channel_name)
            title_lines.append(f"Segment {segment_index}: {t_seg[0]:.2f}–{t_seg[-1]:.2f} min")
            ax.set_title(" - ".join(title_lines), fontsize=12)

            ax.plot(t_seg, seg_filt, label="PAT filtered", linewidth=0.8)

            if peak_indices is not None and peak_indices.size > 0:
                mask_peaks = (peak_indices >= start) & (peak_indices < end)
                if np.any(mask_peaks):
                    seg_peak_indices = peak_indices[mask_peaks]
                    t_peaks = seg_peak_indices / sfreq / 60.0
                    y_peaks = signal_filt[seg_peak_indices]
                    ax.scatter(t_peaks, y_peaks, marker="o", s=10, label="Detected peaks", zorder=3)

            ax.set_ylabel("PAT amplitude")
            ax.grid(True)
            ax.legend(loc="upper right")

            if pat_ylim is not None:
                ax.set_ylim(pat_ylim)

            if use_pwa_debug and ax_pwa is not None and pwa_debug is not None:
                sig_pwa = np.asarray(pwa_debug.get("signal_smooth", []), dtype=float)
                seg_pwa = sig_pwa[start:end]
                ax_pwa.plot(t_seg, seg_pwa, label="PWA detector smoothed PAT", linewidth=0.8, color="tab:purple")

                max_idx = np.asarray(pwa_debug.get("max_indices", []), dtype=int)
                min_idx = np.asarray(pwa_debug.get("min_indices", []), dtype=int)
                pair_max = np.asarray(pwa_debug.get("pair_max_indices", []), dtype=int)
                pair_min = np.asarray(pwa_debug.get("pair_min_indices", []), dtype=int)

                for idxs, marker, color, label in [
                    (max_idx, "^", "tab:blue", "all local maxima"),
                    (min_idx, "v", "tab:red", "all local minima"),
                    (pair_max, "o", "black", "accepted PWA max"),
                    (pair_min, "o", "tab:orange", "accepted PWA min"),
                ]:
                    mask = (idxs >= start) & (idxs < end)
                    if np.any(mask):
                        ii = idxs[mask]
                        ax_pwa.scatter(ii / sfreq / 60.0, sig_pwa[ii], marker=marker, s=12, label=label, zorder=3)

                if pair_max.size and pair_min.size:
                    m = (pair_max >= start) & (pair_max < end) & (pair_min >= start) & (pair_min < end)
                    for imax, imin in zip(pair_max[m], pair_min[m]):
                        ax_pwa.vlines(imax / sfreq / 60.0, sig_pwa[imin], sig_pwa[imax], color="0.35", linewidth=0.6, alpha=0.45, zorder=2)

                ax_pwa.set_ylabel("PWA max/min")
                ax_pwa.grid(True)
                ax_pwa.legend(loc="upper right", fontsize=8)

            if use_act and ax_act is not None:
                seg_start_sec = start / sfreq
                seg_end_sec = end / sfreq

                a0 = int(np.floor(seg_start_sec * act_sfreq))
                a1 = int(np.ceil(seg_end_sec * act_sfreq))
                a0 = max(0, a0)
                a1 = min(len(actigraph), a1)

                if a1 > a0:
                    t_act = np.arange(a0, a1) / act_sfreq / 60.0
                    y_act = actigraph[a0:a1].astype(float)
                    ax_act.plot(t_act, y_act, linewidth=0.8, label=act_label)
                    ax_act.legend(loc="upper right")
                else:
                    ax_act.text(0.02, 0.5, "No ACTIGRAPH samples in this segment",
                                transform=ax_act.transAxes)

                ax_act.set_ylabel("Motion")
                ax_act.grid(True)
                ax_act.set_xlabel("Time (minutes from recording start)")

                if act_ylim is not None:
                    ax_act.set_ylim(act_ylim)
            elif ax_pwa is not None:
                ax_pwa.set_xlabel("Time (minutes from recording start)")
            else:
                ax.set_xlabel("Time (minutes from recording start)")

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
