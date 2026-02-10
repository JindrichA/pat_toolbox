from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from .. import config, paths, sleep_mask, io_aux_csv
from . import hr as hr_metrics  # ✅ use the same RR extraction as HRV

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------
# HRV-matched (fixed-window) tachogram PSD
# ---------------------------------------------------------------------

def _tachogram_psd_from_rr(
    rr_ms: np.ndarray,
    rr_mid_times_sec: np.ndarray,
    *,
    fs_resample: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match hrv._lf_hf_from_rr behavior:
      - RR in seconds
      - detrend by mean
      - interpolate onto uniform grid
      - Welch PSD with nperseg=min(len(rr_interp), 256)
    Returns f, Pxx in (sec^2/Hz).
    """
    if rr_ms.size < 4 or rr_mid_times_sec.size < 4:
        return np.array([]), np.array([])

    t = rr_mid_times_sec
    if t.size < 2 or t[-1] <= t[0]:
        return np.array([]), np.array([])

    rr_sec = rr_ms / 1000.0
    rr_detr = rr_sec - np.mean(rr_sec)

    t_grid = np.arange(t[0], t[-1], 1.0 / float(fs_resample))
    if t_grid.size < 8:
        return np.array([]), np.array([])

    rr_interp = np.interp(t_grid, t, rr_detr)

    nperseg = min(len(rr_interp), 256)
    if nperseg < 8:
        return np.array([]), np.array([])

    nfft = int(getattr(config, "PSD_TACHO_NFFT", 2048))  # e.g. 1024 or 2048
    nfft = max(nfft, int(nperseg))  # must be >= nperseg

    f, pxx = welch(
        rr_interp,
        fs=float(fs_resample),
        nperseg=int(nperseg),
        nfft=int(nfft),
    )
    return f, pxx


def _compute_hrv_matched_psd(
    pat_signal: np.ndarray,
    fs_pat: float,
    aux_df: Optional["pd.DataFrame"],
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, float]]:
    """
    Compute PSD on the SAME signal and SAME windows used for HRV LF/HF fixed windows.

    Returns:
      f, Pxx_avg, n_windows_valid, diag
    """
    rr_sec, rr_mid, duration_sec = hr_metrics.extract_clean_rr_from_pat(pat_signal, fs_pat)
    if rr_sec.size < 1:
        return np.array([]), np.array([]), 0, {"reason": "no_rr"}

    rr_ms = rr_sec * 1000.0

    # ---- Sleep masking on RR mid-times (same as HRV) ----
    if aux_df is not None and bool(getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False)):
        m_sleep = sleep_mask.build_sleep_include_mask_for_times(rr_mid, aux_df)
        if m_sleep is not None:
            rr_mid = rr_mid[m_sleep]
            rr_ms = rr_ms[m_sleep]

    if rr_ms.size < 4:
        return np.array([]), np.array([]), 0, {"reason": "rr_removed_by_sleep_mask"}

    # ---- Event exclusion on RR mid-times (same as HRV) ----
    if aux_df is not None and rr_mid.size > 0:
        m_keep = io_aux_csv.get_rr_exclusion_mask(rr_mid, aux_df)
        if m_keep is not None:
            rr_mid = rr_mid[m_keep]
            rr_ms = rr_ms[m_keep]

    if rr_ms.size < 4:
        return np.array([]), np.array([]), 0, {"reason": "rr_removed_by_event_mask"}

    # ---- Fixed window definition (same config as HRV Option A) ----
    window_sec = float(getattr(config, "HRV_LFHF_FIXED_WINDOW_SEC", 300.0))
    hop_sec = float(getattr(config, "HRV_LFHF_FIXED_HOP_SEC", window_sec))
    max_gap_sec = float(getattr(config, "HRV_MAX_RR_GAP_SEC", 4.0))
    min_rr = int(getattr(config, "HRV_LFHF_FIXED_MIN_RR", 0))
    fs_resample = float(getattr(config, "HRV_TACHO_RESAMPLE_HZ", 4.0))

    half = 0.5 * window_sec
    centers = np.arange(half, max(half, float(duration_sec) - half) + 1e-9, hop_sec)
    if centers.size == 0:
        return np.array([]), np.array([]), 0, {"reason": "no_windows_defined"}

    # Two-pointer RR windowing (same pattern as your HRV implementation)
    n = rr_mid.size
    left = 0
    right = 0

    f_ref = None
    acc = None
    n_valid = 0
    n_total = int(centers.size)

    for c in centers:
        start = c - half
        end = c + half

        while left < n and rr_mid[left] < start:
            left += 1
        if right < left:
            right = left
        while right < n and rr_mid[right] < end:
            right += 1

        k = right - left
        if k < 4:
            continue
        if min_rr and k < int(min_rr):
            continue

        rr_win_ms = rr_ms[left:right]
        rr_mid_win = rr_mid[left:right]

        # HRV fixed-window validity rules:
        # - no gaps > max_gap_sec
        # - coverage span >= 0.8*window_sec
        if rr_mid_win.size >= 2:
            if np.any(np.diff(rr_mid_win) > float(max_gap_sec)):
                continue
            span = float(rr_mid_win[-1] - rr_mid_win[0])
            if span < 0.8 * float(window_sec):
                continue
        else:
            continue

        f, pxx = _tachogram_psd_from_rr(rr_win_ms, rr_mid_win, fs_resample=fs_resample)
        if f.size == 0:
            continue

        if f_ref is None:
            f_ref = f
            acc = np.zeros_like(pxx)
        if f.shape != f_ref.shape:
            # Shouldn't happen with consistent fs_resample & nperseg cap, but be safe
            continue

        acc += pxx
        n_valid += 1

    if n_valid == 0:
        return np.array([]), np.array([]), 0, {"reason": "no_valid_windows", "n_total": float(n_total)}

    return f_ref, acc / float(n_valid), n_valid, {
        "n_total": float(n_total),
        "window_sec": float(window_sec),
        "hop_sec": float(hop_sec),
        "fs_resample": float(fs_resample),
    }


# ---------------------------------------------------------------------
# Main Feature Extraction & Plotting (keeps same output logic)
# ---------------------------------------------------------------------

def compute_psd_figures_and_peaks(
    signal_raw: np.ndarray,
    sfreq: float,
    *,
    edf_base: str,
    aux_df: Optional["pd.DataFrame"] = None,
) -> Tuple[
    Dict[str, float],
    "plt.Figure",
    "plt.Figure",
    Path,
    Path,
]:
    """
    PSD now matches HRV LF/HF fixed-window segments & tachogram signal,
    but keeps the same output logic (paths, plots, features dict).
    """
    n_samples = int(len(signal_raw))
    if n_samples == 0 or sfreq <= 0:
        raise ValueError("Signal is empty or sampling frequency invalid.")

    psd_folder = paths.get_output_folder(getattr(config, "PSD_OUTPUT_SUBFOLDER", "PSD"))
    psd_folder.mkdir(parents=True, exist_ok=True)

    psd_png_zoom = psd_folder / f"{edf_base}__PSD_0-0.5Hz.png"
    psd_png_full = psd_folder / f"{edf_base}__PSD_0-5Hz.png"

    # --- 1. Compute Spectrum using HRV-matched signal/windows ---
    f, Pxx, n_windows, diag = _compute_hrv_matched_psd(signal_raw, sfreq, aux_df)

    if f.size == 0 or Pxx.size == 0 or n_windows == 0:
        # Fallback: compute PSD on whole tachogram (no fixed-window filtering)
        # so output files still exist
        rr_sec, rr_mid, _dur = hr_metrics.extract_clean_rr_from_pat(signal_raw, sfreq)
        rr_ms = rr_sec * 1000.0
        fs_resample = float(getattr(config, "HRV_TACHO_RESAMPLE_HZ", 4.0))
        f, Pxx = _tachogram_psd_from_rr(rr_ms, rr_mid, fs_resample=fs_resample)
        n_windows = 0  # indicates fallback
        diag = {"reason": "fallback_whole_tachogram"}

    # Convert to dB for peak finding / plotting
    Pxx_dB = 10.0 * np.log10(Pxx + 1e-20)

    # --- 2. Feature Extraction (kept the same) ---
    max_freq = float(getattr(config, "PSD_MAX_FREQ_HZ", 5.0))
    mayer_band = getattr(config, "PSD_MAYER_BAND", (0.04, 0.15))
    resp_band = getattr(config, "PSD_RESP_BAND", (0.15, 0.50))
    vlf_band = (0.0033, 0.04)

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
        low, high = band
        mask = (f_arr >= low) & (f_arr <= high)
        if not np.any(mask):
            return 0.0
        return float(np.trapz(p_arr_lin[mask], f_arr[mask]))

    mayer_peak_hz, mayer_peak_db = _find_peak_in_band(f, Pxx_dB, mayer_band, 0.5)
    resp_peak_hz, resp_peak_db = _find_peak_in_band(f, Pxx_dB, resp_band, 0.5)

    pow_vlf = _integrate_power(f, Pxx, vlf_band)
    pow_mayer = _integrate_power(f, Pxx, mayer_band)
    pow_resp = _integrate_power(f, Pxx, resp_band)
    pow_total = _integrate_power(f, Pxx, (0.0033, 0.5))

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
        "n_windows": int(n_windows),
        # optional diagnostics (won't break CSV if you ignore unknown keys)
        "psd_diag_reason": str(diag.get("reason", "")),
    }

    # --- 3. Figure 1: Zoom (0 - 0.5 Hz) ---
    mask_zoom = (f >= 0.0) & (f <= 0.5)

    fig_psd_zoom, ax_psd_zoom = plt.subplots(figsize=(11.69, 4.0))
    ax_psd_zoom.plot(f[mask_zoom], Pxx_dB[mask_zoom], linewidth=1.2, label="PSD")

    ax_psd_zoom.fill_between(
        f, Pxx_dB, np.min(Pxx_dB[mask_zoom]),
        where=((f >= mayer_band[0]) & (f <= mayer_band[1])),
        alpha=0.2, label="Mayer Band"
    )
    ax_psd_zoom.fill_between(
        f, Pxx_dB, np.min(Pxx_dB[mask_zoom]),
        where=((f >= resp_band[0]) & (f <= resp_band[1])),
        alpha=0.2, label="Resp Band"
    )

    if mayer_peak_hz is not None:
        ax_psd_zoom.axvline(mayer_peak_hz, linestyle="--", alpha=0.9, linewidth=1.0)
        ax_psd_zoom.text(mayer_peak_hz, mayer_peak_db, f" {mayer_peak_hz:.3f}Hz",
                         fontsize=9, fontweight="bold", va="bottom")

    if resp_peak_hz is not None:
        ax_psd_zoom.axvline(resp_peak_hz, linestyle="--", alpha=0.9, linewidth=1.0)
        ax_psd_zoom.text(resp_peak_hz, resp_peak_db, f" {resp_peak_hz:.3f}Hz",
                         fontsize=9, fontweight="bold", va="bottom")

    mask_status = "HRV-matched RR PSD" if aux_df is not None else "RR PSD"
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
    ax_psd_full.plot(f[mask_full], Pxx_dB[mask_full], linewidth=1.0)

    ax_psd_full.set_title(
        f"{edf_base} - PSD Full ({mask_status}, 0–{max_freq:g} Hz)",
        fontsize=12,
    )
    ax_psd_full.set_xlabel("Frequency [Hz]")
    ax_psd_full.set_ylabel("Power [dB]")
    ax_psd_full.grid(True, alpha=0.3)

    fig_psd_full.tight_layout()
    fig_psd_full.savefig(psd_png_full, dpi=150)

    return features, fig_psd_zoom, fig_psd_full, psd_png_zoom, psd_png_full
