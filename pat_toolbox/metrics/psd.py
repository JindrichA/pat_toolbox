# pat_toolbox/metrics/psd.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, get_window

from .. import config, paths, sleep_mask, io_aux_csv

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------
# Masked Welch Implementation
# ---------------------------------------------------------------------

def _compute_masked_welch(
        signal: np.ndarray,
        fs: float,
        aux_df: Optional[pd.DataFrame],
        nperseg: int,
        noverlap: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute Welch's PSD using ONLY windows that are 'clean' and (optionally) 'sleep'.

    Logic:
      1. Define sliding windows across the full signal.
      2. Check the midpoint of each window against sleep/artifact masks.
      3. Compute Periodogram (FFT) for valid windows only.
      4. Average the valid periodograms.

    Returns:
        f: Array of sample frequencies.
        Pxx: Power spectral density.
        n_windows: Number of valid windows contributed to the average.
    """
    n_samples = len(signal)

    # 1. Define window steps
    step = nperseg - noverlap
    if step < 1:
        step = 1

    # Starting indices of every possible window
    indices = np.arange(0, n_samples - nperseg + 1, step)

    if len(indices) == 0:
        # Signal shorter than one window? Fallback to standard welch on whatever we have
        return welch(signal, fs=fs, nperseg=nperseg)

    # 2. Determine Validity of each window
    # We check the status at the CENTER of the window
    window_mid_indices = indices + (nperseg // 2)
    window_mid_times = window_mid_indices / fs

    # Default: All windows are valid
    keep_mask = np.ones(len(window_mid_times), dtype=bool)

    if aux_df is not None:
        # A. Sleep Stage Masking
        # If enabled in config, only keep windows in allowed sleep stages
        if getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False):
            m_sleep = sleep_mask.build_sleep_include_mask_for_times(window_mid_times, aux_df)
            if m_sleep is not None:
                keep_mask &= m_sleep

        # B. Event/Artifact Exclusion
        # Re-use the RR exclusion logic (checks desat, exclude_pat, etc.)
        m_excl = io_aux_csv.get_rr_exclusion_mask(window_mid_times, aux_df)
        if m_excl is not None:
            keep_mask &= m_excl

    valid_indices = indices[keep_mask]
    n_windows = len(valid_indices)

    # If everything was masked out, warn and fallback to full signal (or return NaNs)
    if n_windows == 0:
        print("    [PSD] WARNING: No valid 'clean/sleep' windows found. Computing PSD on full signal as fallback.")
        return welch(signal, fs=fs, nperseg=nperseg)

    # 3. Compute and Accumulate FFTs
    # Prepare window function (Hanning) and scale factor
    win = get_window('hann', nperseg)
    # standard Welch normalization factor: 1 / (fs * sum(win^2))
    scale = 1.0 / (fs * (win * win).sum())

    psd_accum = np.zeros(nperseg // 2 + 1, dtype=float)

    # Standard Welch Loop
    for start_idx in valid_indices:
        segment = signal[start_idx: start_idx + nperseg]

        # Detrend: Remove mean (DC offset) of the specific window
        segment = segment - np.mean(segment)

        # Apply Window
        seg_win = segment * win

        # FFT (Real input -> Complex output)
        fft_vals = np.fft.rfft(seg_win)

        # Power calculation: |FFT|^2 * scale
        Pxx_seg = (np.abs(fft_vals) ** 2) * scale

        psd_accum += Pxx_seg

    # 4. Average
    Pxx = psd_accum / n_windows

    # 5. Handle One-Sided Spectrum Scaling
    # (Multiply non-DC/Nyquist components by 2 to conserve total energy)
    if nperseg % 2:
        Pxx[1:] *= 2
    else:
        Pxx[1:-1] *= 2

    # Frequency axis
    f = np.fft.rfftfreq(nperseg, 1.0 / fs)

    return f, Pxx, n_windows


# ---------------------------------------------------------------------
# Main Feature Extraction & Plotting
# ---------------------------------------------------------------------

def compute_psd_figures_and_peaks(
        signal_raw: np.ndarray,
        sfreq: float,
        *,
        edf_base: str,
        aux_df: Optional[pd.DataFrame] = None,
) -> Tuple[
    Dict[str, float],
    "plt.Figure",
    "plt.Figure",
    Path,
    Path,
]:
    """
    Compute PSD on RAW PAT, extract spectral features, and save plots.
    Uses Masked Welch to exclude artifacts and wake/bad sleep stages.

    Returns:
      (features_dict, fig_psd_zoom, fig_psd_full, png_zoom_path, png_full_path)
    """
    n_samples = int(len(signal_raw))
    if n_samples == 0 or sfreq <= 0:
        raise ValueError("Signal is empty or sampling frequency invalid.")

    psd_folder = paths.get_output_folder(getattr(config, "PSD_OUTPUT_SUBFOLDER", "PSD"))
    psd_folder.mkdir(parents=True, exist_ok=True)

    psd_png_zoom = psd_folder / f"{edf_base}__PSD_0-0.5Hz.png"
    psd_png_full = psd_folder / f"{edf_base}__PSD_0-5Hz.png"

    # Welch Parameters
    # 4096 samples @ 250Hz ~ 16 seconds.
    # For robust VLF/Mayer, we want nice long windows if possible.
    # If config not set, default to ~60s window if data permits, or 4096 samples
    nperseg = int(getattr(config, "PSD_NPERSEG", 4096))
    if nperseg > n_samples:
        nperseg = n_samples

    noverlap = nperseg // 2

    # --- 1. Compute Spectrum (Masked) ---
    f, Pxx, n_windows = _compute_masked_welch(
        signal_raw, sfreq, aux_df, nperseg, noverlap
    )

    # Convert to dB for peak finding / plotting
    Pxx_dB = 10.0 * np.log10(Pxx + 1e-20)

    # --- 2. Feature Extraction ---
    max_freq = float(getattr(config, "PSD_MAX_FREQ_HZ", 5.0))
    mayer_band = getattr(config, "PSD_MAYER_BAND", (0.04, 0.15))
    resp_band = getattr(config, "PSD_RESP_BAND", (0.15, 0.50))
    vlf_band = (0.0033, 0.04)

    # Helpers
    def _find_peak_in_band(f_arr, p_arr_db, band, f_max_limit):
        low, high = band
        high = min(high, f_max_limit)
        mask = (f_arr >= low) & (f_arr <= high)
        if not np.any(mask):
            return None, None

        f_sub = f_arr[mask]
        p_sub = p_arr_db[mask]
        idx = np.argmax(p_sub)
        return float(f_sub[idx]), float(p_sub[idx])

    def _integrate_power(f_arr, p_arr_lin, band):
        # Trapezoidal integration
        low, high = band
        mask = (f_arr >= low) & (f_arr <= high)
        if not np.any(mask):
            return 0.0
        return float(np.trapz(p_arr_lin[mask], f_arr[mask]))

    # Peaks (Frequency and dB)
    mayer_peak_hz, mayer_peak_db = _find_peak_in_band(f, Pxx_dB, mayer_band, 0.5)
    resp_peak_hz, resp_peak_db = _find_peak_in_band(f, Pxx_dB, resp_band, 0.5)

    # Absolute Powers (Linear units: voltage^2 / Hz -> voltage^2)
    pow_vlf = _integrate_power(f, Pxx, vlf_band)
    pow_mayer = _integrate_power(f, Pxx, mayer_band)
    pow_resp = _integrate_power(f, Pxx, resp_band)

    # "Total" power for normalization (approximate physiological range 0.0033 - 0.5 Hz)
    pow_total = _integrate_power(f, Pxx, (0.0033, 0.5))

    # Normalized Powers (percentage)
    norm_mayer = (pow_mayer / pow_total * 100.0) if pow_total > 0 else np.nan
    norm_resp = (pow_resp / pow_total * 100.0) if pow_total > 0 else np.nan

    features = {
        "mayer_peak_hz": mayer_peak_hz,
        "resp_peak_hz": resp_peak_hz,
        "pow_vlf": pow_vlf,
        "pow_mayer": pow_mayer,
        "pow_resp": pow_resp,
        "norm_mayer": norm_mayer,
        "norm_resp": norm_resp,
        "n_windows": n_windows
    }

    # --- 3. Figure 1: Zoom (0 - 0.5 Hz) ---
    mask_zoom = (f >= 0.0) & (f <= 0.5)

    fig_psd_zoom, ax_psd_zoom = plt.subplots(figsize=(11.69, 4.0))
    ax_psd_zoom.plot(f[mask_zoom], Pxx_dB[mask_zoom], linewidth=1.2, color='#2c3e50', label="PSD")

    # Shade bands
    ax_psd_zoom.fill_between(f, Pxx_dB, np.min(Pxx_dB[mask_zoom]),
                             where=((f >= mayer_band[0]) & (f <= mayer_band[1])),
                             color='orange', alpha=0.2, label="Mayer Band")
    ax_psd_zoom.fill_between(f, Pxx_dB, np.min(Pxx_dB[mask_zoom]),
                             where=((f >= resp_band[0]) & (f <= resp_band[1])),
                             color='green', alpha=0.2, label="Resp Band")

    # Mark Peaks
    if mayer_peak_hz is not None:
        ax_psd_zoom.axvline(mayer_peak_hz, linestyle="--", color='orange', alpha=0.9, linewidth=1.0)
        ax_psd_zoom.text(mayer_peak_hz, mayer_peak_db, f" {mayer_peak_hz:.3f}Hz",
                         color='orange', fontsize=9, fontweight='bold', va="bottom")

    if resp_peak_hz is not None:
        ax_psd_zoom.axvline(resp_peak_hz, linestyle="--", color='green', alpha=0.9, linewidth=1.0)
        ax_psd_zoom.text(resp_peak_hz, resp_peak_db, f" {resp_peak_hz:.3f}Hz",
                         color='green', fontsize=9, fontweight='bold', va="bottom")

    # Info title
    mask_status = "Clean/Sleep" if aux_df is not None else "Full Signal"
    ax_psd_zoom.set_title(
        f"{edf_base} - PSD ({mask_status}, N={n_windows} windows)\n"
        f"Mayer Pwr: {pow_mayer:.2e} ({norm_mayer:.1f}%) | Resp Pwr: {pow_resp:.2e} ({norm_resp:.1f}%)",
        fontsize=11,
    )
    ax_psd_zoom.set_xlabel("Frequency [Hz]")
    ax_psd_zoom.set_ylabel("Power Spectral Density [dB]")
    ax_psd_zoom.legend(loc="upper right", fontsize=8)
    ax_psd_zoom.grid(True, alpha=0.3)

    fig_psd_zoom.tight_layout()
    fig_psd_zoom.savefig(psd_png_zoom, dpi=150)

    # --- 4. Figure 2: Full (0 - Max Hz) ---
    mask_full = f <= max_freq

    fig_psd_full, ax_psd_full = plt.subplots(figsize=(11.69, 4.0))
    ax_psd_full.plot(f[mask_full], Pxx_dB[mask_full], linewidth=1.0, color='#34495e')

    ax_psd_full.set_title(
        f"{edf_base} - PSD Full ({mask_status}, 0–{max_freq:g} Hz, nperseg={nperseg})",
        fontsize=12,
    )
    ax_psd_full.set_xlabel("Frequency [Hz]")
    ax_psd_full.set_ylabel("Power [dB]")
    ax_psd_full.grid(True, alpha=0.3)

    fig_psd_full.tight_layout()
    fig_psd_full.savefig(psd_png_full, dpi=150)

    return (
        features,
        fig_psd_zoom,
        fig_psd_full,
        psd_png_zoom,
        psd_png_full,
    )