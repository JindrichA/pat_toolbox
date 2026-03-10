# pat_toolbox/config.py
from __future__ import annotations
from typing import Dict
from datetime import datetime
import re
from pathlib import Path

# =============================================================================
# Output base directory (ALL results go here)
# =============================================================================
BASE_OUTPUT_DIR = Path("/Users/jindrich/Projects/PAT_022026_output_data/")
# Debug: limit number of EDF files processed (None = all)
MAX_FILES = None
# =============================================================================
# Paths
# =============================================================================
EDF_FOLDER = Path(
    "/Users/jindrich/Projects/mayo_sleep_pat/SmallDataset21Oct25/Data/Dataset_21Oct25/Data_Only"
)
# Unique run identifier (set once at import time)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
# Optional short message/tag to make folders self-describing (edit this, not dates)
RUN_TAG = "desat_2min_FD_fixed"

def _slug(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s

# =============================================================================
# Sleep stage handling / masking  (MOVED UP so OUTPUT_SUBFOLDER sees it)
# =============================================================================
# Show the PAT raw/filtered subplot at top of each segment page
ENABLE_PAT_SIGNAL_PLOT = True
ENABLE_SLEEP_STAGE_MASKING = True

# Sleep stage filtering policy
# Options:
#   "all_sleep"       - Include all sleep stages (1,2,3)
#   "rem_only"        - Include only REM (3)
#   "nrem_only"       - Include only non-REM (1,2)
#   "deep_only"       - Include only deep sleep (2)
#   "nrem_light_only" - Include only NREM light (1)
#   "custom"          - Use SLEEP_INCLUDE_LABELS / SLEEP_INCLUDE_NUMERIC
SLEEP_STAGE_POLICY = "all_sleep"


# Used only when SLEEP_STAGE_POLICY == "custom"
SLEEP_INCLUDE_LABELS = {"L. Sleep", "D. Sleep", "REM"}
SLEEP_INCLUDE_NUMERIC = {1}  # 0 (wake) is always excluded


def sleep_stage_suffix() -> str:
    try:
        s = (SLEEP_STAGE_POLICY or "all_sleep").lower()
    except NameError:
        s = "all_sleep"
    return f"_{s}"

def run_suffix() -> str:
    """
    Suffix used in output folder names.
    Example: 20260114_093012__nrem_only__desat
    """
    parts = [RUN_ID]
    # include sleep policy so each run is self-describing
    try:
        parts.append(sleep_stage_suffix().lstrip("_"))
    except Exception:
        pass
    tag = _slug(RUN_TAG)
    if tag:
        parts.append(tag)
    return "__".join(parts)

# Automatically unique per run (no manual renaming)
OUTPUT_SUBFOLDER = f"ViewPatPlotsOverlay__{run_suffix()}"
HR_OUTPUT_SUBFOLDER = f"HR__{run_suffix()}"
PSD_OUTPUT_SUBFOLDER = f"PSD__{run_suffix()}"

# =============================================================================
# Channels
# =============================================================================

VIEW_PAT_CHANNEL_NAME = "VIEW_PAT"
HR_CHANNEL_NAME = "DERIVED_HR"
PAT_AMP_CHANNEL_NAME = "DERIVED_PAT_AMP"
ACTIGRAPH_CHANNEL_NAME = "ACTIGRAPH"

# =============================================================================
# Plotting / Segments
# =============================================================================

SEGMENT_MINUTES = 15  # length of each segment for plotting

ENABLE_VIEW_PAT_OVERLAY_PLOTS = True
ENABLE_PAT_PEAK_DEBUG_PLOTS = False
PAT_PEAK_DEBUG_SEGMENT_MINUTES = 1.0  # 1 minute per page

OVERVIEW_PANEL_HOURS = 2.0

# =============================================================================
# PAT filtering
# =============================================================================

PAT_BANDPASS_LOWCUT_HZ = 0.5
PAT_BANDPASS_HIGHCUT_HZ = 8.0
PAT_BANDPASS_ORDER = 4

# =============================================================================
# HR feature
# =============================================================================
ENABLE_HR = True
HR_TARGET_FS_HZ = 1.0  # 1 Hz HR output (time grid)

# Physiologic RR limits (seconds)
HR_MIN_RR_SEC = 0.30
HR_MAX_RR_SEC = 2.50

# HR clamps (bpm)
HR_MIN_BPM = 30.0
HR_MAX_BPM = 220.0

# EDF scale (DERIVED_HR channel)
HR_EDF_SCALE_FACTOR = 0.01

# Peak detection tuning
HR_PEAK_PROMINENCE_FACTOR = 0.30

# HR smoothing / robustness
HR_SMOOTHING_WINDOW_SEC = 4.0
HR_HAMPEL_WINDOW_SEC = 8.0
HR_HAMPEL_SIGMA = 2.0

# Optional HR slope limiter (0 disables)
HR_MAX_DELTA_BPM_PER_SEC = 10.0

# HR interpolation: do NOT bridge gaps larger than this (seconds)
HR_MAX_RR_GAP_SEC = 2.5

# =============================================================================
# RR cleaning (shared by HR + HRV)
# =============================================================================

HR_RR_MEDFILT_KERNEL = 5              # local median window (odd)
HR_RR_OUTLIER_REL_THR = 0.30          # |rr - med| / med <= thr

# Reject "very long" RR relative to local median (missed beats / broken segments)
HR_RR_GAP_FACTOR = 2.4                # rr <= gap_factor * local_median

# Reject abrupt RR jumps (helps with transient mis-detections)
HR_RR_JUMP_REL_THR = 0.6              # relative jump threshold

# Reject adjacent short+long (or long+short) pairs (double- + missed-peak signature)
HR_RR_ALT_SHORT_REL = 0.25            # short if < (1 - x) * local_median
HR_RR_ALT_LONG_REL = 0.35             # long  if > (1 + y) * local_median

# Keep only contiguous runs of >= N RR after masking
HR_RR_MIN_GOOD_RUN = 3
# =============================================================================
# HRV settings
# =============================================================================

HRV_TARGET_FS_HZ = 1.0
HRV_WINDOW_SEC = 300.0

HRV_MIN_INTERVALS_PER_WINDOW = 10      # strongly recommended for PAT
# RMSSD(t) smoothing (only applied when no NaNs in series; see implementation)
HRV_SMOOTHING_WINDOW_SEC = 5.0

# Optional HRV Hampel (if/when you implement it on RMSSD series)
HRV_HAMPEL_WINDOW_SEC = 30.0
HRV_HAMPEL_SIGMA = 3.0

# Gap handling for RMSSD windows
HRV_MAX_RR_GAP_SEC = 4.0              # reject windows spanning gaps > this
HRV_RMSSD_MIN_SPAN_SEC = 10.0         # reject windows with tiny coverage clusters

# Frequency-domain (LF/HF)
HRV_TACHO_RESAMPLE_HZ = 4.0
HRV_MIN_FREQ_DOMAIN_SEC = 120.0       # contiguous span requirement for LF/HF
HRV_MAX_TACHO_GAP_SEC = 3.0           # used by TV metrics windowing



HRV_LFHF_FIXED_WINDOW_SEC = 120.0
HRV_LFHF_FIXED_HOP_SEC = 120.0   # non-overlapping
HRV_LFHF_FIXED_MIN_RR = 0        # optional; set e.g. 200 for stricter filtering


HRV_RMSSD_VETO_BIGDIFF = False
# or, if you want to keep it:
HRV_RMSSD_BIGDIFF_THR_MS = 300.0
HRV_RMSSD_BIGDIFF_MAX_FRAC = 0.35


# =============================================================================
# Time-varying (TV) HRV metrics (for plotting SDNN/LF/HF/LFHF series)
# =============================================================================

HRV_TV_WINDOW_SEC = 120.0             # 5 min window
HRV_TV_STEP_HZ = 1.0                  # evaluate on 1 Hz grid

HRV_TV_TACHO_RESAMPLE_HZ = 4.0
HRV_TV_MIN_RR_PER_WINDOW = 10
#HRV_TV_MIN_FREQ_DOMAIN_SEC = 300.0    # optional separate knob; fallback uses HRV_MIN_FREQ_DOMAIN_SEC
HRV_TV_MAX_TACHO_GAP_SEC = 3.0        # optional separate knob; fallback uses HRV_MAX_TACHO_GAP_SEC
# Separate acceptance criteria for TV spectral metrics
HRV_TV_MIN_FREQ_DOMAIN_SEC = 60.0
HRV_TV_MAX_TACHO_GAP_SEC = 6.0


# =============================================================================
# Auxiliary CSV synchronized to EDF
# =============================================================================

AUX_CSV_ENABLED = True
AUX_CSV_EXTENSION = ".csv"
AUX_CSV_SEP = ","

AUX_CSV_TIME_COLUMN_RAW = "Time"
AUX_CSV_TIME_SEC_COLUMN = "time_sec"
AUX_CSV_USE_COLUMNS = None

# Mapping from internal names -> actual CSV column names
COL_NAMES: Dict[str, str] = {
    "time": "Time",
    "stage": "WP Stages",
    "spo2": "SpO2",
    "desat_flag": "Desaturation",
    "exclude_pat_flag": "Exclude PAT",
    "evt_central_3": "A/H central-3% (Last second)",
    "evt_obstructive_3": "A/H obstructive-3% (Last second)",
    "evt_unclassified_3": "A/H unclassified-3% (Last second)",
}

# HRV event exclusion (based on canonical keys above)
HRV_EXCLUSION_EVENT_COLUMNS = [
    "evt_central_3",
    "evt_obstructive_3",
    "evt_unclassified_3",
    # "desat_flag",
]

#TODO: uncoment, just for debuging and ploting the signals.

#HRV_EXCLUSION_PRE_SEC = 15.0
#HRV_EXCLUSION_POST_SEC = 30.0
HRV_EXCLUSION_PRE_SEC = 0.0
HRV_EXCLUSION_POST_SEC = 0.0

# Turn on desat-dependent exclusion windows (instead of fixed pre/post around events)
#HRV_EXCLUSION_USE_DESAT_WINDOWS = True
HRV_EXCLUSION_USE_DESAT_WINDOWS = False
# Which aux column indicates desaturation (uses COL_NAMES key by default)
HRV_EXCLUSION_DESAT_COLUMN_KEY = "desat_flag"   # maps via COL_NAMES

# How far before desat-start to begin exclusion
HRV_EXCLUSION_DESAT_START_PAD_SEC = 15

# How far after desat-end to extend exclusion
#HRV_EXCLUSION_DESAT_END_PAD_SEC = 30
HRV_EXCLUSION_DESAT_END_PAD_SEC = 0

# Optional: require desat run length
HRV_EXCLUSION_DESAT_MIN_RUN_SEC = 5.0

HRV_EXCLUSION_DESAT_LOOKBACK_SEC = 0.0   # how far before desat window we search for an event
HRV_EXCLUSION_DESAT_LOOKAHEAD_SEC = 0.0  # how far after desat window we search for an event

# =============================================================================
# PSD / spectral analysis
# =============================================================================

PSD_MAX_FREQ_HZ = 5.0
PSD_NPERSEG = 4096

PSD_MAYER_BAND = (0.04, 0.15)
PSD_RESP_BAND = (0.15, 0.23)

# =============================================================================
# RMSSD cleaning / robustness
# =============================================================================

HRV_RMSSD_DIFF_HARD_CAP_MS = 250.0
HRV_RMSSD_DIFF_MAD_SIGMAS = 3.0
HRV_RMSSD_MIN_DIFFS = 3

HRV_RMSSD_FLOOR_MS = 2.0
HRV_MIN_WINDOW_COVERAGE = 0.4

# =============================================================================
# Sleep stage handling / masking
# =============================================================================

ENABLE_SLEEP_STAGE_MASKING = True

# Canonical numeric codes used internally:
#   0 = Wake (masked)
#   1 = Light sleep
#   2 = Deep sleep
#   3 = REM
SLEEP_MAPPING = {
    "WK": 0,          # Wake
    "L. Sleep": 1,    # NREM light
    "D. Sleep": 2,    # NREM deep
    "REM": 3,         # REM
}



# =============================================================================
# Delta HR (ΔHR) feature
# =============================================================================

ENABLE_DELTA_HR = True

# ΔHR(t) = HR(t) - HR(t - lag)
DELTA_HR_LAG_SEC = 30.0

# Optional pre-smoothing on HR before delta (seconds; 0 disables)
DELTA_HR_PRE_SMOOTH_SEC = 0.0

# Use absolute delta (|ΔHR|) instead of signed
DELTA_HR_ABS = False

# Plot mode for segment pages:
#   "subplot" -> extra row showing ΔHR
#   "twinx"   -> overlay ΔHR on HR axis using a 2nd y-axis
DELTA_HR_PLOT_MODE = "subplot"


def sleep_include_numeric() -> set[int]:
    """
    Resolve the include set of numeric sleep codes from SLEEP_STAGE_POLICY.
    Wake (0) is never included.
    """
    s = (SLEEP_STAGE_POLICY or "all_sleep").lower()
    if s == "n2n3_only":
        s = "deep_only"

    if s == "all_sleep":
        return {1, 2, 3}
    if s == "rem_only":
        return {3}
    if s == "nrem_only":
        return {1, 2}
    if s == "deep_only":
        return {2}
    if s == "nrem_light_only":
        return {1}
    if s == "custom":
        inc = set(int(x) for x in (SLEEP_INCLUDE_NUMERIC or set()))
        if not inc and SLEEP_INCLUDE_LABELS:
            for lab in SLEEP_INCLUDE_LABELS:
                if lab in SLEEP_MAPPING:
                    inc.add(int(SLEEP_MAPPING[lab]))
        return {x for x in inc if x != 0}

    return {1, 2, 3}


# =============================================================================
# PAT burden (event+desat region)
# =============================================================================
ENABLE_PAT_BURDEN = True
PAT_BURDEN_BASELINE_LOOKBACK_SEC = 30.0
PAT_BURDEN_BASELINE_MIN_SAMPLES = 5
PAT_BURDEN_BASELINE_PCTL = 95.0
PAT_BURDEN_MIN_EPISODE_SEC = 5.0
PAT_BURDEN_RELATIVE = False
