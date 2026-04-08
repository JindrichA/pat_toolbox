from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .. import config, masking, paths
from . import hr as hr_metrics
from .spectral_utils import find_peak_in_band, integrate_band_power, tachogram_psd_from_rr

if TYPE_CHECKING:
    import pandas as pd


def _empty_psd_features(reason: str) -> Dict[str, float]:
    return {
        "mayer_peak_hz": np.nan,
        "resp_peak_hz": np.nan,
        "pow_vlf": np.nan,
        "pow_mayer": np.nan,
        "pow_resp": np.nan,
        "norm_mayer": np.nan,
        "norm_resp": np.nan,
        "n_windows": 0,
        "psd_diag_reason": reason,
    }


def _iter_valid_fixed_rr_windows(
    rr_mid: np.ndarray,
    rr_ms: np.ndarray,
    duration_sec: float,
    *,
    window_sec: float,
    hop_sec: float,
    max_gap_sec: float,
    min_rr: int,
):
    half = 0.5 * float(window_sec)
    centers = np.arange(half, max(half, float(duration_sec) - half) + 1e-9, float(hop_sec))
    if centers.size == 0:
        return

    n = rr_mid.size
    left = 0
    right = 0

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
        if rr_mid_win.size < 2:
            continue
        if np.any(np.diff(rr_mid_win) > float(max_gap_sec)):
            continue
        span = float(rr_mid_win[-1] - rr_mid_win[0])
        if span < 0.8 * float(window_sec):
            continue

        yield rr_mid_win, rr_win_ms


def _compute_hrv_matched_psd(
    pat_signal: np.ndarray,
    fs_pat: float,
    aux_df: Optional["pd.DataFrame"],
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    rr_sec, rr_mid, duration_sec = hr_metrics.extract_clean_rr_from_pat(pat_signal, fs_pat)
    if rr_sec.size < 1:
        return np.array([]), np.array([]), 0, {"reason": "no_rr"}

    rr_ms = rr_sec * 1000.0
    bundle = masking.build_rr_mask_bundle(rr_mid, aux_df)

    rr_mid_sleep = rr_mid[bundle.sleep_keep]
    rr_ms_sleep = rr_ms[bundle.sleep_keep]
    if rr_ms_sleep.size < 4:
        return np.array([]), np.array([]), 0, {"reason": "rr_removed_by_sleep_mask"}

    rr_mid_clean = rr_mid[bundle.combined_keep]
    rr_ms_clean = rr_ms[bundle.combined_keep]
    if rr_ms_clean.size < 4:
        return np.array([]), np.array([]), 0, {"reason": "rr_removed_by_event_mask"}

    window_sec = float(getattr(config, "HRV_LFHF_FIXED_WINDOW_SEC", 300.0))
    hop_sec = float(getattr(config, "HRV_LFHF_FIXED_HOP_SEC", window_sec))
    max_gap_sec = float(getattr(config, "HRV_MAX_RR_GAP_SEC", 4.0))
    min_rr = int(getattr(config, "HRV_LFHF_FIXED_MIN_RR", 0))
    fs_resample = float(getattr(config, "HRV_TACHO_RESAMPLE_HZ", 4.0))

    centers = np.arange(0.5 * window_sec, max(0.5 * window_sec, float(duration_sec) - 0.5 * window_sec) + 1e-9, hop_sec)
    if centers.size == 0:
        return np.array([]), np.array([]), 0, {"reason": "no_windows_defined"}

    f_ref: Optional[np.ndarray] = None
    acc: Optional[np.ndarray] = None
    n_valid = 0

    for rr_mid_win, rr_win_ms in _iter_valid_fixed_rr_windows(
        rr_mid_clean,
        rr_ms_clean,
        duration_sec,
        window_sec=window_sec,
        hop_sec=hop_sec,
        max_gap_sec=max_gap_sec,
        min_rr=min_rr,
    ):
        f, pxx = tachogram_psd_from_rr(rr_win_ms, rr_mid_win, fs_resample=fs_resample)
        if f.size == 0:
            continue
        if f_ref is None:
            f_ref = f
            acc = np.zeros_like(pxx)
        if f.shape != f_ref.shape:
            continue
        acc += pxx
        n_valid += 1

    if n_valid == 0 or f_ref is None or acc is None:
        return np.array([]), np.array([]), 0, {"reason": "no_valid_windows", "n_total": float(centers.size)}

    return f_ref, acc / float(n_valid), n_valid, {
        "n_total": float(centers.size),
        "window_sec": float(window_sec),
        "hop_sec": float(hop_sec),
        "fs_resample": float(fs_resample),
    }


def compute_psd_features_from_rr(
    rr_mid: np.ndarray,
    rr_ms: np.ndarray,
    duration_sec: float,
    aux_df: Optional["pd.DataFrame"],
    *,
    include_set: Optional[set[int]] = None,
) -> Dict[str, float]:
    rr_mid = np.asarray(rr_mid, dtype=float)
    rr_ms = np.asarray(rr_ms, dtype=float)

    if rr_mid.size != rr_ms.size or rr_mid.size < 1:
        return _empty_psd_features("no_rr")

    bundle = masking.build_rr_mask_bundle(
        rr_mid,
        aux_df,
        policy=masking.policy_from_config(include_stages=include_set, force_sleep=(include_set is not None)),
    )
    rr_mid_sleep = rr_mid[bundle.sleep_keep]
    rr_ms_sleep = rr_ms[bundle.sleep_keep]
    if rr_ms_sleep.size < 4:
        return _empty_psd_features("rr_removed_by_sleep_mask")

    rr_mid_clean = rr_mid[bundle.combined_keep]
    rr_ms_clean = rr_ms[bundle.combined_keep]
    if rr_ms_clean.size < 4:
        return _empty_psd_features("rr_removed_by_event_mask")

    window_sec = float(getattr(config, "HRV_LFHF_FIXED_WINDOW_SEC", 300.0))
    hop_sec = float(getattr(config, "HRV_LFHF_FIXED_HOP_SEC", window_sec))
    max_gap_sec = float(getattr(config, "HRV_MAX_RR_GAP_SEC", 4.0))
    min_rr = int(getattr(config, "HRV_LFHF_FIXED_MIN_RR", 0))
    fs_resample = float(getattr(config, "HRV_TACHO_RESAMPLE_HZ", 4.0))

    centers = np.arange(0.5 * window_sec, max(0.5 * window_sec, float(duration_sec) - 0.5 * window_sec) + 1e-9, hop_sec)
    if centers.size == 0:
        return _empty_psd_features("no_windows_defined")

    f_ref: Optional[np.ndarray] = None
    acc: Optional[np.ndarray] = None
    n_valid = 0
    for rr_mid_win, rr_win_ms in _iter_valid_fixed_rr_windows(
        rr_mid_clean,
        rr_ms_clean,
        duration_sec,
        window_sec=window_sec,
        hop_sec=hop_sec,
        max_gap_sec=max_gap_sec,
        min_rr=min_rr,
    ):
        f, pxx = tachogram_psd_from_rr(rr_win_ms, rr_mid_win, fs_resample=fs_resample)
        if f.size == 0:
            continue
        if f_ref is None:
            f_ref = f
            acc = np.zeros_like(pxx)
        if f.shape != f_ref.shape:
            continue
        acc += pxx
        n_valid += 1

    if n_valid == 0 or f_ref is None or acc is None:
        return _empty_psd_features("no_valid_windows")

    pxx = acc / float(n_valid)
    pxx_db = 10.0 * np.log10(pxx + 1e-20)
    mayer_band = getattr(config, "PSD_MAYER_BAND", (0.04, 0.15))
    resp_band = getattr(config, "PSD_RESP_BAND", (0.15, 0.50))
    vlf_band = (0.0033, 0.04)

    mayer_peak_hz, _ = find_peak_in_band(f_ref, pxx_db, mayer_band, 0.5)
    resp_peak_hz, _ = find_peak_in_band(f_ref, pxx_db, resp_band, 0.5)
    pow_vlf = integrate_band_power(f_ref, pxx, vlf_band)
    pow_mayer = integrate_band_power(f_ref, pxx, mayer_band)
    pow_resp = integrate_band_power(f_ref, pxx, resp_band)
    pow_total = integrate_band_power(f_ref, pxx, (0.0033, 0.5))

    return {
        "mayer_peak_hz": np.nan if mayer_peak_hz is None else mayer_peak_hz,
        "resp_peak_hz": np.nan if resp_peak_hz is None else resp_peak_hz,
        "pow_vlf": pow_vlf,
        "pow_mayer": pow_mayer,
        "pow_resp": pow_resp,
        "norm_mayer": (pow_mayer / pow_total * 100.0) if pow_total > 0 else np.nan,
        "norm_resp": (pow_resp / pow_total * 100.0) if pow_total > 0 else np.nan,
        "n_windows": int(n_valid),
        "psd_diag_reason": "",
    }


def compute_psd_figures_and_peaks(
    signal_raw: np.ndarray,
    sfreq: float,
    *,
    edf_base: str,
    aux_df: Optional["pd.DataFrame"] = None,
) -> Tuple[
    Dict[str, float],
    Any,
    Any,
    Path,
    Path,
]:
    if int(len(signal_raw)) == 0 or sfreq <= 0:
        raise ValueError("Signal is empty or sampling frequency invalid.")

    psd_folder = paths.get_output_folder(getattr(config, "PSD_OUTPUT_SUBFOLDER", "PSD"))
    psd_folder.mkdir(parents=True, exist_ok=True)

    psd_png_zoom = psd_folder / f"{edf_base}__PSD_0-0.5Hz.png"
    psd_png_full = psd_folder / f"{edf_base}__PSD_0-5Hz.png"

    f, pxx, n_windows, diag = _compute_hrv_matched_psd(signal_raw, sfreq, aux_df)

    psd_mode = "matched"
    if f.size == 0 or pxx.size == 0 or n_windows == 0:
        rr_sec, rr_mid, _dur = hr_metrics.extract_clean_rr_from_pat(signal_raw, sfreq)
        rr_ms = rr_sec * 1000.0
        fs_resample = float(getattr(config, "HRV_TACHO_RESAMPLE_HZ", 4.0))
        f, pxx = tachogram_psd_from_rr(rr_ms, rr_mid, fs_resample=fs_resample)
        n_windows = 0
        diag = {"reason": "fallback_whole_tachogram"}
        psd_mode = "fallback"

    pxx_db = 10.0 * np.log10(pxx + 1e-20)
    max_freq = float(getattr(config, "PSD_MAX_FREQ_HZ", 5.0))
    mayer_band = getattr(config, "PSD_MAYER_BAND", (0.04, 0.15))
    resp_band = getattr(config, "PSD_RESP_BAND", (0.15, 0.50))
    vlf_band = (0.0033, 0.04)

    mayer_peak_hz, mayer_peak_db = find_peak_in_band(f, pxx_db, mayer_band, 0.5)
    resp_peak_hz, resp_peak_db = find_peak_in_band(f, pxx_db, resp_band, 0.5)

    pow_vlf = integrate_band_power(f, pxx, vlf_band)
    pow_mayer = integrate_band_power(f, pxx, mayer_band)
    pow_resp = integrate_band_power(f, pxx, resp_band)
    pow_total = integrate_band_power(f, pxx, (0.0033, 0.5))

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
        "psd_mode": psd_mode,
        "psd_diag_reason": str(diag.get("reason", "")),
    }

    mask_zoom = (f >= 0.0) & (f <= 0.5)
    fig_psd_zoom, ax_psd_zoom = plt.subplots(figsize=(11.69, 4.0))
    ax_psd_zoom.plot(f[mask_zoom], pxx_db[mask_zoom], linewidth=1.2, label="PSD")

    ax_psd_zoom.fill_between(
        f,
        pxx_db,
        np.min(pxx_db[mask_zoom]),
        where=((f >= mayer_band[0]) & (f <= mayer_band[1])).tolist(),
        alpha=0.2,
        label="Mayer Band",
    )
    ax_psd_zoom.fill_between(
        f,
        pxx_db,
        np.min(pxx_db[mask_zoom]),
        where=((f >= resp_band[0]) & (f <= resp_band[1])).tolist(),
        alpha=0.2,
        label="Resp Band",
    )

    if mayer_peak_hz is not None and mayer_peak_db is not None:
        ax_psd_zoom.axvline(mayer_peak_hz, linestyle="--", alpha=0.9, linewidth=1.0)
        ax_psd_zoom.text(mayer_peak_hz, mayer_peak_db, f" {mayer_peak_hz:.3f}Hz", fontsize=9, fontweight="bold", va="bottom")

    if resp_peak_hz is not None and resp_peak_db is not None:
        ax_psd_zoom.axvline(resp_peak_hz, linestyle="--", alpha=0.9, linewidth=1.0)
        ax_psd_zoom.text(resp_peak_hz, resp_peak_db, f" {resp_peak_hz:.3f}Hz", fontsize=9, fontweight="bold", va="bottom")

    mask_status = "HRV-matched RR PSD" if (psd_mode == "matched" and aux_df is not None) else ("RR PSD" if psd_mode == "matched" else "Fallback whole-tachogram PSD")
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

    mask_full = f <= max_freq
    fig_psd_full, ax_psd_full = plt.subplots(figsize=(11.69, 4.0))
    ax_psd_full.plot(f[mask_full], pxx_db[mask_full], linewidth=1.0)
    ax_psd_full.set_title(f"{edf_base} - PSD Full ({mask_status}, 0–{max_freq:g} Hz)", fontsize=12)
    ax_psd_full.set_xlabel("Frequency [Hz]")
    ax_psd_full.set_ylabel("Power [dB]")
    ax_psd_full.grid(True, alpha=0.3)
    fig_psd_full.tight_layout()
    fig_psd_full.savefig(psd_png_full, dpi=150)

    return features, fig_psd_zoom, fig_psd_full, psd_png_zoom, psd_png_full
