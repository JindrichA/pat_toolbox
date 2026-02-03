# pat_toolbox/metrics/hr.py

from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
from scipy.signal import find_peaks

from .. import config, filters, io_edf, paths, plotting


# ---------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------

import csv
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

from .. import config, paths


def append_hr_correlation_to_summary(
        edf_path: Path,
        pearson_r: Optional[float],
        spear_rho: Optional[float],
        rmse: Optional[float],
        hrv_summary: Optional[Dict[str, float]] = None,
        mayer_peak_freq: Optional[float] = None,
        resp_peak_freq: Optional[float] = None,
        *,
        # --- NEW optional inputs to capture "page 1" + more ---
        hr_calc: Optional[np.ndarray] = None,  # PAT-derived HR (masked)
        hr_edf: Optional[np.ndarray] = None,  # EDF HR (masked)
        hrv_clean: Optional[np.ndarray] = None,  # RMSSD clean (masked)
        hrv_raw: Optional[np.ndarray] = None,  # RMSSD raw (unmasked)
        hrv_tv: Optional[Dict[str, np.ndarray]] = None,  # time-varying HRV dict
        aux_df: Optional[Any] = None,  # pandas df (kept Any to avoid hard dependency)
        psd_features: Optional[Dict[str, float]] = None,  # <--- NEW ARGUMENT ADDED HERE
        pat_burden: Optional[float] = None,
        pat_burden_diag: Optional[dict] = None,
) -> Path:
    """
    Append one row with HR correlation + HRV summary + spectral peaks + extra "summary page" features.

    Extra columns (when available):
      - NaN% for HR/HRV series + TV metrics
      - aux flag counts + percentages
      - sleep-stage masking stats + stage breakdown
      - NEW: PSD Spectral Power features (VLF, Mayer, Resp)

    Returns summary CSV path.
    """

    # ----------------------------
    # helpers
    # ----------------------------
    def _isfinite_scalar(x: Any) -> bool:
        try:
            return bool(np.isfinite(float(x)))
        except Exception:
            return False

    def _nan_pct(x: Optional[np.ndarray]) -> Optional[float]:
        if x is None:
            return None
        x = np.asarray(x)
        if x.size == 0:
            return None
        return float(100.0 * np.mean(~np.isfinite(x)))

    def _fmt6(x: Optional[float]) -> str:
        if x is None:
            return ""
        try:
            xf = float(x)
            if not np.isfinite(xf):
                return ""
            return f"{xf:.6f}"
        except Exception:
            return ""

    def _fmt1(x: Optional[float]) -> str:
        if x is None:
            return ""
        try:
            xf = float(x)
            if not np.isfinite(xf):
                return ""
            return f"{xf:.1f}"
        except Exception:
            return ""

    def _fmt_sci(x: Optional[float]) -> str:
        if x is None:
            return ""
        try:
            xf = float(x)
            if not np.isfinite(xf):
                return ""
            return f"{xf:.4e}"
        except Exception:
            return ""

    def _count_flags(df: Any, col: str) -> tuple[Optional[int], Optional[float]]:
        """
        Returns (count, pct_of_rows) where count is sum of (col==1) after fillna(0).
        """
        if df is None:
            return (None, None)
        try:
            if not hasattr(df, "columns") or col not in df.columns:
                return (None, None)
            total = int(len(df))
            if total <= 0:
                return (0, 0.0)
            c = int(df[col].fillna(0).astype(int).sum())
            pct = 100.0 * c / total
            return (c, pct)
        except Exception:
            return (None, None)

    def _sleep_stage_stats(df: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if df is None or not hasattr(df, "columns"):
            return out

        stage_col = getattr(config, "AUX_CSV_STAGE_CODE_COLUMN", "stage_code")
        if stage_col not in df.columns:
            return out

        try:
            s = np.asarray(df[stage_col].to_numpy(dtype=float))
            ok = np.isfinite(s)
            if not np.any(ok):
                return out

            stage_i = np.round(s[ok]).astype(int)
            total = int(stage_i.size)
            if total <= 0:
                return out

            enabled = bool(getattr(config, "ENABLE_SLEEP_STAGE_MASKING", False))
            policy = str(getattr(config, "SLEEP_STAGE_POLICY", "all_sleep"))

            try:
                include_set = set(config.sleep_include_numeric())
            except Exception:
                include_set = {1, 2, 3}

            included = np.array([int(x) in include_set for x in stage_i], dtype=bool)
            inc_n = int(np.sum(included))
            exc_n = int(total - inc_n)

            def pct(n: int) -> float:
                return 100.0 * float(n) / float(total)

            # stage counts (0,1,2,3)
            counts = {k: int(np.sum(stage_i == k)) for k in [0, 1, 2, 3]}

            out.update(
                {
                    "sleep_mask_enabled": int(enabled),
                    "sleep_policy": policy,
                    "sleep_included_n": inc_n,
                    "sleep_included_pct": pct(inc_n),
                    "sleep_excluded_n": exc_n,
                    "sleep_excluded_pct": pct(exc_n),
                    "sleep_wake_n": counts[0],
                    "sleep_wake_pct": pct(counts[0]),
                    "sleep_light_n": counts[1],
                    "sleep_light_pct": pct(counts[1]),
                    "sleep_deep_n": counts[2],
                    "sleep_deep_pct": pct(counts[2]),
                    "sleep_rem_n": counts[3],
                    "sleep_rem_pct": pct(counts[3]),
                }
            )
            return out
        except Exception:
            return out

    # ----------------------------
    # paths / filename
    # ----------------------------
    summary_folder = paths.get_output_folder(config.HR_OUTPUT_SUBFOLDER)

    sleep_stage_policy = (
        str(getattr(config, "SLEEP_STAGE_POLICY", "unknown"))
        .replace(" ", "_")
        .replace("/", "-")
    )
    run_id = getattr(config, "RUN_ID", "default")
    summary_filename = f"HR_HRV_summary__{sleep_stage_policy}__{run_id}.csv"
    summary_path = summary_folder / summary_filename

    # ----------------------------
    # core row (existing columns)
    # ----------------------------
    row: Dict[str, Any] = {
        "edf_file": edf_path.name,
        "pearson": pearson_r,
        "spearman": spear_rho,
        "rmse_bpm": rmse,
        "mayer_peak_hz": mayer_peak_freq,
        "resp_peak_hz": resp_peak_freq,
    }


    # ----------------------------
    # PAT burden (event+desat)
    # ----------------------------
    row["pat_burden"] = pat_burden

    if isinstance(pat_burden_diag, dict):
        row["pat_burden_sleep_hours"] = pat_burden_diag.get("sleep_hours")
        row["pat_burden_total_area_min"] = pat_burden_diag.get("total_area_min")
        row["pat_burden_n_episodes"] = pat_burden_diag.get("n_episodes")
        row["pat_burden_n_episodes_used"] = pat_burden_diag.get("n_episodes_used")
        row["pat_burden_relative"] = int(bool(pat_burden_diag.get("relative", False)))
        row["pat_burden_nan_pct"] = pat_burden_diag.get("nan_pct_inside")
    else:
        row["pat_burden_sleep_hours"] = np.nan
        row["pat_burden_total_area_min"] = np.nan
        row["pat_burden_n_episodes"] = np.nan
        row["pat_burden_n_episodes_used"] = np.nan
        row["pat_burden_relative"] = np.nan
        row["pat_burden_nan_pct_inside"] = np.nan

    # HRV summary fields
    if hrv_summary is not None:
        row.update(
            {
                "rmssd_mean_ms": hrv_summary.get("rmssd_mean", np.nan),
                "rmssd_median_ms": hrv_summary.get("rmssd_median", np.nan),
                "sdnn_ms": hrv_summary.get("sdnn", np.nan),
                "lf": hrv_summary.get("lf", np.nan),
                "hf": hrv_summary.get("hf", np.nan),
                "lf_hf": hrv_summary.get("lf_hf", np.nan),

                # --- NEW: LF/HF segmentation diagnostics ---
                "lf_n_segments_used": hrv_summary.get("lf_n_segments_used", np.nan),
                # --- NEW: Fixed-window LF/HF (Option A) ---
                "lf_hf_fixed_median": hrv_summary.get("lf_hf_fixed_median", np.nan),
                "lf_hf_fixed_mean": hrv_summary.get("lf_hf_fixed_mean", np.nan),
                "lf_hf_fixed_n_windows_valid": hrv_summary.get("lf_hf_fixed_n_windows_valid", np.nan),
                "lf_hf_fixed_n_windows_total": hrv_summary.get("lf_hf_fixed_n_windows_total", np.nan),
                "lf_hf_fixed_window_sec": hrv_summary.get("lf_hf_fixed_window_sec", np.nan),
                "lf_hf_fixed_hop_sec": hrv_summary.get("lf_hf_fixed_hop_sec", np.nan),

            }
        )
    else:
        row.update(
            {
                "rmssd_mean_ms": np.nan,
                "rmssd_median_ms": np.nan,
                "sdnn_ms": np.nan,
                "lf": np.nan,
                "hf": np.nan,
                "lf_hf": np.nan,

                # --- NEW: LF/HF segmentation diagnostics ---
                "lf_n_segments_used": np.nan,
                # --- NEW: Fixed-window LF/HF (Option A) ---
                "lf_hf_fixed_median": np.nan,
                "lf_hf_fixed_mean": np.nan,
                "lf_hf_fixed_n_windows_valid": np.nan,
                "lf_hf_fixed_n_windows_total": np.nan,
                "lf_hf_fixed_window_sec": np.nan,
                "lf_hf_fixed_hop_sec": np.nan,
                # optional:
                # "lf_n_segments_total": np.nan,
                # "lf_dur_used_sec": np.nan,
            }
        )

    # --- NEW: Spectral Power Features ---
    if psd_features:
        row.update({
            "psd_pow_vlf": psd_features.get("pow_vlf"),
            "psd_pow_mayer": psd_features.get("pow_mayer"),
            "psd_pow_resp": psd_features.get("pow_resp"),
            "psd_norm_mayer": psd_features.get("norm_mayer"),
            "psd_norm_resp": psd_features.get("norm_resp"),
            "psd_valid_windows": psd_features.get("n_windows"),
        })

    # ----------------------------
    # "Page 1" additions: NaN% quality
    # ----------------------------
    row["hr_pat_nan_pct"] = _nan_pct(hr_calc)
    row["hr_edf_nan_pct"] = _nan_pct(hr_edf)
    row["hrv_rmssd_clean_nan_pct"] = _nan_pct(hrv_clean)
    row["hrv_rmssd_raw_nan_pct"] = _nan_pct(hrv_raw)

    # TV metric NaN% (if present)
    if isinstance(hrv_tv, dict):
        for k, v in hrv_tv.items():
            if v is None:
                continue
            try:
                row[f"hrv_tv_{k}_nan_pct"] = _nan_pct(np.asarray(v))
            except Exception:
                # ignore weird entries
                pass

    # ----------------------------
    # "Page 1" additions: aux event summary
    # ----------------------------
    if aux_df is not None and hasattr(aux_df, "__len__"):
        try:
            row["aux_rows"] = int(len(aux_df))
        except Exception:
            row["aux_rows"] = ""

        # keep these aligned with your plotting page and config canonical keys
        aux_keys = [
            ("desat", "desat_flag"),
            ("exclude_hr", "exclude_hr_flag"),
            ("exclude_pat", "exclude_pat_flag"),
            ("evt_central_3", "evt_central_3"),
            ("evt_obstructive_3", "evt_obstructive_3"),
            ("evt_unclassified_3", "evt_unclassified_3"),
            ("evt_central_4", "evt_central_4"),
            ("evt_obstructive_4", "evt_obstructive_4"),
            ("evt_unclassified_4", "evt_unclassified_4"),
        ]
        for short, col in aux_keys:
            c, p = _count_flags(aux_df, col)
            if c is not None:
                row[f"{short}_n"] = c
            if p is not None:
                row[f"{short}_pct"] = p

    # ----------------------------
    # "Page 1" additions: sleep-stage masking stats
    # ----------------------------
    row.update(_sleep_stage_stats(aux_df))

    # ----------------------------
    # Write CSV (upgrade schema if needed)
    # ----------------------------
    # Preferred column order (stable + readable). Any new keys get appended.
    base_order = [
        "edf_file",
        "pearson",
        "spearman",
        "rmse_bpm",
        "rmssd_mean_ms",
        "rmssd_median_ms",
        "sdnn_ms",
        "lf",
        "hf",
        "lf_hf",
        "lf_n_segments_used",
        # --- NEW: Fixed-window LF/HF (Option A) ---
        "lf_hf_fixed_median",
        "lf_hf_fixed_mean",
        "lf_hf_fixed_n_windows_valid",
        "lf_hf_fixed_n_windows_total",
        "lf_hf_fixed_window_sec",
        "lf_hf_fixed_hop_sec",
        "mayer_peak_hz",
        "resp_peak_hz",
        # --- NEW Spectral Cols ---
        "psd_pow_vlf", "psd_pow_mayer", "psd_pow_resp",
        "psd_norm_mayer", "psd_norm_resp", "psd_valid_windows",
        # -------------------------
        "hr_pat_nan_pct",
        "hr_edf_nan_pct",
        "hrv_rmssd_clean_nan_pct",
        "hrv_rmssd_raw_nan_pct",
        "aux_rows",
        "desat_n",
        "desat_pct",
        "exclude_hr_n",
        "exclude_hr_pct",
        "exclude_pat_n",
        "exclude_pat_pct",
        "evt_central_3_n",
        "evt_central_3_pct",
        "evt_obstructive_3_n",
        "evt_obstructive_3_pct",
        "evt_unclassified_3_n",
        "evt_unclassified_3_pct",
        "evt_central_4_n",
        "evt_central_4_pct",
        "evt_obstructive_4_n",
        "evt_obstructive_4_pct",
        "evt_unclassified_4_n",
        "evt_unclassified_4_pct",
        "sleep_mask_enabled",
        "sleep_policy",
        "sleep_included_n",
        "sleep_included_pct",
        "sleep_excluded_n",
        "sleep_excluded_pct",
        "sleep_wake_n",
        "sleep_wake_pct",
        "sleep_light_n",
        "sleep_light_pct",
        "sleep_deep_n",
        "sleep_deep_pct",
        "sleep_rem_n",
        "sleep_rem_pct",
    ]

    # Normalize the aux column names used above (desat_n etc.)
    # We wrote row keys as: desat_n etc. only if present; ensure consistent naming here:
    # The loop wrote short keys as "desat_n/pct" etc; base_order expects that format.
    # But we used short names "evt_central_3" etc in the loop.
    # Fix: rename keys from loop's short names to the exact base_order names.
    # (This keeps your CSV column names explicit and stable.)
    renames = {}
    if "desat_n" not in row and "desat_n" in row:
        pass  # noop

    # Actually, our loop used `short` like "desat", "evt_central_3" etc:
    # so keys are "desat_n", "evt_central_3_n", ... already. Good.

    # Determine final fieldnames
    row_keys = list(row.keys())
    fieldnames = [c for c in base_order if c in row_keys]
    extras = sorted([k for k in row_keys if k not in set(fieldnames)])
    fieldnames.extend(extras)

    def _format_row_for_csv(d: Dict[str, Any], cols: list[str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for k in cols:
            v = d.get(k, "")
            if v is None:
                out[k] = ""
            elif isinstance(v, (int, np.integer)):
                out[k] = str(int(v))
            elif isinstance(v, (float, np.floating)):
                # Use 6 decimals for most numeric fields, 1 decimal for pct fields
                if k.startswith("psd_pow_"):
                    out[k] = _fmt_sci(float(v))
                elif k.endswith("_pct") or k.endswith("_nan_pct"):
                    out[k] = _fmt1(float(v))
                else:
                    out[k] = _fmt6(float(v))
            else:
                # strings / others
                out[k] = str(v)
        return out

    # If file doesn't exist: straightforward write
    if not summary_path.exists():
        summary_folder.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow(_format_row_for_csv(row, fieldnames))
        return summary_path

    # If exists: check header; if mismatch, upgrade (rewrite union header)
    with summary_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        existing_fieldnames = r.fieldnames or []
        existing_rows = list(r)

    # Union header preserving existing order, then adding new columns
    union_fieldnames = list(existing_fieldnames)
    for k in fieldnames:
        if k not in union_fieldnames:
            union_fieldnames.append(k)

    # If headers differ, rewrite full file with union header
    if union_fieldnames != existing_fieldnames:
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=union_fieldnames)
            w.writeheader()
            for old in existing_rows:
                # DictReader returns strings already; just ensure all columns exist
                full_old = {k: old.get(k, "") for k in union_fieldnames}
                w.writerow(full_old)
            w.writerow(_format_row_for_csv(row, union_fieldnames))
        return summary_path

    # Headers match: append one row
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=existing_fieldnames)
        w.writerow(_format_row_for_csv(row, existing_fieldnames))

    return summary_path


# ---------------------------------------------------------------------
# Peak detection + shared RR extraction
# ---------------------------------------------------------------------

def _detect_pat_peaks(
    pat_signal: np.ndarray,
    fs: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter PAT and detect peaks.

    Returns:
        pat_filt: filtered PAT signal (same length as input)
        peak_indices: np.ndarray of integer indices of detected peaks
    """
    if fs <= 0:
        raise ValueError("Sampling frequency fs must be positive.")

    n_samples = len(pat_signal)
    if pat_signal.ndim != 1 or n_samples == 0:
        raise ValueError("PAT signal must be a non-empty 1D array.")

    # 1) Filter PAT
    pat_filt = filters.bandpass_filter(pat_signal, fs=fs)

    # 2) Peak detection parameters
    min_rr_samples = int(config.HR_MIN_RR_SEC * fs)
    if min_rr_samples < 1:
        min_rr_samples = 1

    std_sig = np.std(pat_filt)
    prom = None
    if std_sig > 0:
        prom = config.HR_PEAK_PROMINENCE_FACTOR * std_sig

    if prom is not None and prom > 0:
        peaks, _props = find_peaks(
            pat_filt,
            distance=min_rr_samples,
            prominence=prom,
        )
    else:
        peaks, _props = find_peaks(
            pat_filt,
            distance=min_rr_samples,
        )

    # Optional basic RR sanity filter (very light)
    if len(peaks) >= 2:
        peak_times_sec = peaks / fs
        rr_intervals_sec = np.diff(peak_times_sec)
        valid_rr_mask = (
            (rr_intervals_sec >= config.HR_MIN_RR_SEC)
            & (rr_intervals_sec <= config.HR_MAX_RR_SEC)
        )
        if np.any(valid_rr_mask):
            valid_peak_mask = np.zeros_like(peak_times_sec, dtype=bool)
            valid_peak_mask[:-1] |= valid_rr_mask
            valid_peak_mask[1:] |= valid_rr_mask
            peaks = peaks[valid_peak_mask]

    return pat_filt, peaks


def extract_clean_rr_from_pat(
    pat_signal: np.ndarray,
    fs: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Shared helper for HR + HRV.

    Steps:
      1) Detect peaks on filtered PAT (_detect_pat_peaks)
      2) Compute RR intervals and mid-times
      3) Apply physiologic RR limits
      4) Apply median-based RR outlier rejection
      5) Reject gaps, abrupt jumps, and short/long alternans pairs (double-peak + missed-peak)
      6) Keep only contiguous "good" runs (drop isolated points)

    Returns:
        rr_sec_clean: np.ndarray of RR intervals [s]
        rr_mid_clean: np.ndarray of RR mid-times [s]
        duration_sec: float, total signal duration [s]

    If cleaning fails, rr_sec_clean and rr_mid_clean are empty arrays.
    """
    if fs <= 0:
        raise ValueError("Sampling frequency fs must be positive.")

    n_samples = len(pat_signal)
    if pat_signal.ndim != 1 or n_samples == 0:
        raise ValueError("PAT signal must be a non-empty 1D array.")

    duration_sec = n_samples / fs

    _pat_filt, peaks = _detect_pat_peaks(pat_signal, fs)
    if len(peaks) < 2:
        return np.array([]), np.array([]), duration_sec

    peak_times_sec = peaks / fs

    rr_sec = np.diff(peak_times_sec)
    rr_mid_times = 0.5 * (peak_times_sec[1:] + peak_times_sec[:-1])

    # Physiologic limits
    valid_rr = (rr_sec >= config.HR_MIN_RR_SEC) & (rr_sec <= config.HR_MAX_RR_SEC)
    rr_sec_valid = rr_sec[valid_rr]
    rr_mid_valid = rr_mid_times[valid_rr]

    if rr_sec_valid.size < 1:
        return np.array([]), np.array([]), duration_sec

    # Median-based RR outlier rejection (local median)
    kernel_len = int(getattr(config, "HR_RR_MEDFILT_KERNEL", 5))
    if kernel_len < 1:
        kernel_len = 1
    if kernel_len % 2 == 0:
        kernel_len += 1

    pad = kernel_len // 2
    rr_padded = np.pad(rr_sec_valid, pad_width=pad, mode="edge")
    rr_med = np.empty_like(rr_sec_valid)
    for i in range(rr_sec_valid.size):
        rr_med[i] = np.median(rr_padded[i : i + kernel_len])

    rel_thr = float(getattr(config, "HR_RR_OUTLIER_REL_THR", 0.25))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_dev = np.abs(rr_sec_valid - rr_med) / rr_med
    rel_dev[~np.isfinite(rel_dev)] = 0.0

    good = rel_dev <= rel_thr

    # -----------------------------------------------------------------
    # Gap masking (reject very long RR relative to local median)
    # -----------------------------------------------------------------
    gap_factor = float(getattr(config, "HR_RR_GAP_FACTOR", 2.2))  # typical 1.8–2.5
    gap_ok = rr_sec_valid <= (gap_factor * rr_med)
    good &= gap_ok

    # -----------------------------------------------------------------
    # Abrupt jump masking (helps when detector briefly misfires)
    # -----------------------------------------------------------------
    jump_thr = float(getattr(config, "HR_RR_JUMP_REL_THR", 0.5))  # 0.3–0.8
    if rr_sec_valid.size >= 2:
        rr_prev = rr_sec_valid[:-1]
        rr_next = rr_sec_valid[1:]
        denom = np.maximum(0.30, 0.5 * (rr_prev + rr_next))
        rel_jump = np.abs(rr_next - rr_prev) / denom
        jump_bad = np.zeros_like(rr_sec_valid, dtype=bool)
        jump_bad[1:] = rel_jump > jump_thr
        good &= ~jump_bad

    # -----------------------------------------------------------------
    # Alternans pair rejection: short+long (or long+short) adjacent pairs
    # This is the classic "double peak then missed peak" signature.
    # -----------------------------------------------------------------
    alt_short_rel = float(getattr(config, "HR_RR_ALT_SHORT_REL", 0.25))  # 25% below med
    alt_long_rel = float(getattr(config, "HR_RR_ALT_LONG_REL", 0.35))    # 35% above med

    if rr_sec_valid.size >= 2:
        short = rr_sec_valid < (1.0 - alt_short_rel) * rr_med
        long = rr_sec_valid > (1.0 + alt_long_rel) * rr_med

        pair1 = short[:-1] & long[1:]
        pair2 = long[:-1] & short[1:]

        alt_bad = np.zeros_like(rr_sec_valid, dtype=bool)
        alt_bad[:-1] |= (pair1 | pair2)
        alt_bad[1:]  |= (pair1 | pair2)
        good &= ~alt_bad

    # Apply mask
    rr_sec_good = rr_sec_valid[good]
    rr_mid_good = rr_mid_valid[good]

    if rr_sec_good.size < 1:
        return np.array([]), np.array([]), duration_sec

    # -----------------------------------------------------------------
    # Keep only contiguous "good" runs (drop isolated points)
    # Contiguity is in the ORIGINAL rr index space (before masking), which is what we want.
    # -----------------------------------------------------------------
    min_keep_run = int(getattr(config, "HR_RR_MIN_GOOD_RUN", 3))
    if min_keep_run > 1:
        idx = np.flatnonzero(good)
        if idx.size == 0:
            return np.array([]), np.array([]), duration_sec

        splits = np.where(np.diff(idx) > 1)[0] + 1
        runs = np.split(idx, splits)

        keep = np.zeros_like(good, dtype=bool)
        for r in runs:
            if r.size >= min_keep_run:
                keep[r] = True

        rr_sec_good = rr_sec_valid[keep]
        rr_mid_good = rr_mid_valid[keep]

    if rr_sec_good.size < 1:
        return np.array([]), np.array([]), duration_sec

    return rr_sec_good, rr_mid_good, duration_sec


# ---------------------------------------------------------------------
# Hampel filter helper
# ---------------------------------------------------------------------

def _hampel_filter_1d(
    x: np.ndarray,
    window_size: int,
    n_sigmas: float = 3.0,
) -> np.ndarray:
    """
    Simple Hampel filter: replace points that deviate from the local median
    by more than n_sigmas * local MAD with the local median.

    NOTE: If x contains NaNs, this implementation will not behave well.
    We avoid running Hampel on NaN-heavy arrays by only applying it to HR after
    interpolation and smoothing (or you can add a nan-aware version).
    """
    if x.ndim != 1 or x.size == 0:
        return x

    if window_size < 1:
        return x
    if window_size % 2 == 0:
        window_size += 1

    k = window_size // 2
    n = x.size
    y = x.copy()

    for i in range(n):
        i0 = max(0, i - k)
        i1 = min(n, i + k + 1)
        window = x[i0:i1]
        med = np.median(window)
        mad = np.median(np.abs(window - med))
        if mad == 0:
            continue
        sigma = 1.4826 * mad
        if np.abs(x[i] - med) > n_sigmas * sigma:
            y[i] = med

    return y


# ---------------------------------------------------------------------
# HR computation from PAT
# ---------------------------------------------------------------------

def _interp_with_gaps(
    t_grid: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    max_gap_sec: float,
) -> np.ndarray:
    """
    Interpolate y(t) onto t_grid, but do NOT bridge across gaps in t bigger than max_gap_sec.
    Returns NaN in gap regions.
    """
    out = np.full_like(t_grid, np.nan, dtype=float)
    if t.size < 2:
        return out

    cuts = np.where(np.diff(t) > float(max_gap_sec))[0] + 1
    for idx in np.split(np.arange(t.size), cuts):
        if idx.size < 2:
            continue
        t0, t1 = t[idx[0]], t[idx[-1]]
        mask = (t_grid >= t0) & (t_grid <= t1)
        if np.any(mask):
            out[mask] = np.interp(t_grid[mask], t[idx], y[idx])
    return out


def compute_hr_from_pat_signal(
    pat_signal: np.ndarray,
    fs: float,
    target_fs: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute HR (1 Hz) from PAT using shared RR extraction + smoothing + despiking.
    """
    if target_fs is None:
        target_fs = float(getattr(config, "HR_TARGET_FS_HZ", 1.0))

    rr_sec_clean, rr_mid_clean, duration_sec = extract_clean_rr_from_pat(pat_signal, fs)

    t_hr = np.arange(0, duration_sec, 1.0 / float(target_fs))
    if rr_sec_clean.size < 1:
        hr_1hz = np.full_like(t_hr, fill_value=np.nan, dtype=float)
        return t_hr, hr_1hz

    inst_hr = 60.0 / rr_sec_clean  # bpm

    # Interpolate to regular grid WITHOUT bridging bad segments
    max_gap_sec = float(getattr(config, "HR_MAX_RR_GAP_SEC", 2.5))
    hr_grid = _interp_with_gaps(t_hr, rr_mid_clean, inst_hr, max_gap_sec=max_gap_sec)

    # Light smoothing (moving average) but preserve NaNs (avoid smearing across gaps)
    smooth_sec = float(getattr(config, "HR_SMOOTHING_WINDOW_SEC", 0.0))
    smooth_samples = int(round(smooth_sec * target_fs))

    if smooth_samples > 1:
        # NaN-aware moving average
        kernel = np.ones(smooth_samples, dtype=float)
        x = hr_grid.astype(float)

        valid = np.isfinite(x).astype(float)
        x0 = np.where(np.isfinite(x), x, 0.0)

        num = np.convolve(x0, kernel, mode="same")
        den = np.convolve(valid, kernel, mode="same")
        with np.errstate(divide="ignore", invalid="ignore"):
            hr_smooth = num / den
        hr_smooth[den <= 0] = np.nan
    else:
        hr_smooth = hr_grid

    # Clip only finite values
    hr_smooth = np.where(
        np.isfinite(hr_smooth),
        np.clip(hr_smooth, config.HR_MIN_BPM, config.HR_MAX_BPM),
        np.nan,
    )

    # HR-domain Hampel despiking (only if few NaNs)
    hampel_win_sec = float(getattr(config, "HR_HAMPEL_WINDOW_SEC", 10.0))
    hampel_sigmas = float(getattr(config, "HR_HAMPEL_SIGMA", 3.0))
    hampel_win_samples = int(round(hampel_win_sec * target_fs))

    hr_despiked = hr_smooth
    nan_frac = float(np.mean(~np.isfinite(hr_smooth))) if hr_smooth.size > 0 else 1.0
    if hampel_win_samples > 1 and nan_frac < 0.05:
        hr_despiked = _hampel_filter_1d(
            hr_smooth,
            window_size=hampel_win_samples,
            n_sigmas=hampel_sigmas,
        )

    hr_despiked = np.where(
        np.isfinite(hr_despiked),
        np.clip(hr_despiked, config.HR_MIN_BPM, config.HR_MAX_BPM),
        np.nan,
    )

    # Optional slope limiting (finite-only; preserve NaNs)
    max_delta_per_sec = float(getattr(config, "HR_MAX_DELTA_BPM_PER_SEC", 0.0))
    if max_delta_per_sec > 0:
        hr_limited = hr_despiked.copy()
        max_step = max_delta_per_sec / float(target_fs)
        for i in range(1, len(hr_limited)):
            if not np.isfinite(hr_limited[i]) or not np.isfinite(hr_limited[i - 1]):
                continue
            delta = hr_limited[i] - hr_limited[i - 1]
            if np.abs(delta) > max_step:
                hr_limited[i] = hr_limited[i - 1] + np.sign(delta) * max_step
    else:
        hr_limited = hr_despiked

    hr_limited = np.where(
        np.isfinite(hr_limited),
        np.clip(hr_limited, config.HR_MIN_BPM, config.HR_MAX_BPM),
        np.nan,
    )

    return t_hr, hr_limited


# ---------------------------------------------------------------------
# File-level HR wrapper
# ---------------------------------------------------------------------

def compute_hr_for_edf_file(
    edf_path: Path,
    save_csv: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper: read PAT from EDF, compute HR 1 Hz, optionally save CSV.
    """
    print(f"Computing HR from PAT for: {edf_path}")

    try:
        pat_signal, fs = io_edf.read_edf_channel(edf_path, config.VIEW_PAT_CHANNEL_NAME)
    except Exception as e:
        print(f"  WARNING: skipping HR for {edf_path.name}: {e}")
        return np.array([]), np.array([])

    try:
        t_hr, hr_1hz = compute_hr_from_pat_signal(pat_signal, fs)
    except Exception as e:
        print(f"  WARNING: could not compute HR from PAT for {edf_path.name}: {e}")
        return np.array([]), np.array([])

    if save_csv and t_hr.size > 0:
        hr_folder = paths.get_output_folder(config.HR_OUTPUT_SUBFOLDER)
        edf_base = edf_path.stem
        out_csv = hr_folder / f"{edf_base}__HR_1Hz.csv"

        data = np.column_stack([t_hr, hr_1hz])
        header = "time_sec,hr_bpm"
        np.savetxt(out_csv, data, delimiter=",", header=header, comments="")
        print(f"  Saved HR 1 Hz CSV to: {out_csv}")

    return t_hr, hr_1hz


# ---------------------------------------------------------------------
# Debug plotting of peaks
# ---------------------------------------------------------------------

def create_peaks_debug_pdf_for_edf(edf_path: Path) -> Path | None:
    """
    Create a debug PDF showing PAT signal and detected peaks (1 min per page),
    with ACTIGRAPH subplot if available.

    ACTIGRAPH is preprocessed into a smooth "motion envelope" for easier visual correlation:
      high-pass -> abs -> low-pass -> (optional) z-score
    """
    print(f"Creating PAT peaks debug PDF for: {edf_path}")

    # --- Read PAT
    try:
        pat_signal, fs = io_edf.read_edf_channel(edf_path, config.VIEW_PAT_CHANNEL_NAME)
    except Exception as e:
        print(f"  WARNING: cannot create peaks debug PDF for {edf_path.name}: {e}")
        return None

    n_samples = len(pat_signal)
    if n_samples == 0 or fs <= 0:
        print("  WARNING: PAT signal empty or invalid fs, skipping debug PDF.")
        return None

    # --- Detect peaks
    try:
        pat_filt, peaks = _detect_pat_peaks(pat_signal, fs)
    except Exception as e:
        print(f"  WARNING: peak detection failed for {edf_path.name}: {e}")
        return None

    # --- Read + preprocess ACTIGRAPH (optional)
    act_to_plot = None
    act_fs = None
    act_label = getattr(config, "ACTIGRAPH_CHANNEL_NAME", "ACTIGRAPH")

    try:
        act_name = getattr(config, "ACTIGRAPH_CHANNEL_NAME", "ACTIGRAPH")
        act_signal, act_fs = io_edf.read_edf_channel(edf_path, act_name)

        if act_fs <= 0 or act_signal is None or len(act_signal) == 0:
            print("  WARNING: ACTIGRAPH empty or invalid fs, ignoring.")
            act_to_plot, act_fs = None, None
        else:
            hp = getattr(config, "ACT_HP_HZ", 0.5)
            lp = getattr(config, "ACT_LP_HZ", 2.0)
            do_z = getattr(config, "ACT_ENV_ZSCORE", True)

            env = filters.actigraph_motion_envelope(act_signal, act_fs, hp_hz=hp, lp_hz=lp)

            if do_z:
                m = float(np.nanmedian(env))
                s = float(np.nanstd(env))
                env = (env - m) / (s + 1e-9)
                act_label = f"{act_name} envelope (z)"
            else:
                act_label = f"{act_name} envelope"

            act_to_plot = env

    except Exception as e:
        print(f"  WARNING: could not read/process ACTIGRAPH: {e}")
        act_to_plot, act_fs = None, None

    # --- Output
    out_folder = paths.get_output_folder()
    edf_base = edf_path.stem
    pdf_name = f"{edf_base}__PAT_Peaks_{config.PAT_PEAK_DEBUG_SEGMENT_MINUTES}min.pdf"
    pdf_path = out_folder / pdf_name

    plotting.plot_pat_with_peaks_segments_to_pdf(
        signal_raw=pat_signal,
        signal_filt=pat_filt,
        peak_indices=peaks,
        sfreq=fs,
        pdf_path=pdf_path,
        segment_minutes=config.PAT_PEAK_DEBUG_SEGMENT_MINUTES,
        title_prefix=edf_base,
        channel_name=config.VIEW_PAT_CHANNEL_NAME,
        actigraph=act_to_plot,
        act_sfreq=act_fs,
        act_label=act_label,
    )

    print(f"  Saved PAT peaks debug PDF to: {pdf_path}")
    return pdf_path


# ---------------------------------------------------------------------
# HR correlation vs EDF
# ---------------------------------------------------------------------

from scipy.signal import butter, filtfilt


def _butter_highpass(
    x: np.ndarray,
    fs: float,
    cutoff_hz: float = 0.25,
    order: int = 4,
) -> np.ndarray:
    """
    High-pass to remove slow drift (and quasi-gravity component if present).
    """
    if fs <= 0 or x.size == 0:
        return x.astype(float)

    nyq = 0.5 * fs
    wn = cutoff_hz / nyq
    wn = min(max(wn, 1e-6), 0.999999)

    b, a = butter(order, wn, btype="highpass")
    return filtfilt(b, a, x.astype(float)).astype(float)


def _moving_average(x: np.ndarray, win_samples: int) -> np.ndarray:
    if win_samples <= 1:
        return x
    win_samples = int(max(1, win_samples))
    kernel = np.ones(win_samples, dtype=float) / float(win_samples)
    return np.convolve(x, kernel, mode="same")


def _motion_amplitude_envelope(act: np.ndarray, fs: float) -> np.ndarray:
    """
    "Amplitude of filtered motion":
      1) high-pass (remove drift)
      2) rectify (abs)
      3) smooth -> envelope (easy to compare visually)
    """
    act_hp = _butter_highpass(act, fs, cutoff_hz=0.25, order=4)
    amp = np.abs(act_hp)

    smooth_sec = 0.5
    win = int(round(smooth_sec * fs))
    return _moving_average(amp, win_samples=win)


def _robust_ylim(
    y: np.ndarray,
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
) -> tuple[float, float]:
    """
    Robust y-limits from percentiles (stable across pages within one file).
    """
    y = y[np.isfinite(y)]
    if y.size == 0:
        return (0.0, 1.0)

    lo = float(np.percentile(y, lo_pct))
    hi = float(np.percentile(y, hi_pct))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(y))
        hi = float(np.nanmax(y))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return (0.0, 1.0)

    margin = 0.10 * (hi - lo + 1e-12)
    lo2 = max(0.0, lo - margin)
    hi2 = hi + margin
    return (lo2, hi2)






def compute_hr_correlation(
    t_hr_edf: np.ndarray,
    hr_edf: np.ndarray,
    t_hr_calc: np.ndarray,
    hr_calc: np.ndarray,
    common_fs: float = 1.0,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute correlation between EDF HR and PAT-derived HR.
    """
    if (
        t_hr_edf is None
        or hr_edf is None
        or t_hr_calc is None
        or hr_calc is None
        or len(hr_edf) == 0
        or len(hr_calc) == 0
    ):
        return None, None, None

    t_start = max(t_hr_edf[0], t_hr_calc[0])
    t_end = min(t_hr_edf[-1], t_hr_calc[-1])
    if t_end <= t_start:
        return None, None, None

    t_grid = np.arange(t_start, t_end, 1.0 / common_fs)

    edf_interp = np.interp(t_grid, t_hr_edf, hr_edf, left=np.nan, right=np.nan)
    calc_interp = np.interp(t_grid, t_hr_calc, hr_calc, left=np.nan, right=np.nan)

    mask = ~np.isnan(edf_interp) & ~np.isnan(calc_interp)
    if not np.any(mask):
        return None, None, None

    x = edf_interp[mask]
    y = calc_interp[mask]
    if len(x) < 2:
        return None, None, None

    pearson_r = np.corrcoef(x, y)[0, 1]
    spear_rho = np.corrcoef(
        np.argsort(np.argsort(x)),
        np.argsort(np.argsort(y)),
    )[0, 1]
    rmse = np.sqrt(np.mean((x - y) ** 2))

    return float(pearson_r), float(spear_rho), float(rmse)
