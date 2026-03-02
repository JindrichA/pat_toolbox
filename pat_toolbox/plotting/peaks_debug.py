from __future__ import annotations

from pathlib import Path
from typing import Optional

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

            if use_act:
                fig, (ax, ax_act) = plt.subplots(
                    2, 1, figsize=(11.69, 8.27),
                    sharex=True,
                    gridspec_kw={"height_ratios": [2, 1]},
                )
            else:
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
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
            else:
                ax.set_xlabel("Time (minutes from recording start)")

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
