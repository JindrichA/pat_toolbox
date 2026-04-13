# pat_toolbox/config.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Dict


# =============================================================================
# Run Identity And Primary Paths
# =============================================================================
# These are the first knobs to edit when pointing the pipeline at a new dataset
# or a new output location. They change where files are read from and where all
# generated artifacts are written.

BASE_OUTPUT_DIR = Path("/Users/jindrich/Projects/PAT_022026_output_data/")
EDF_FOLDER = Path(
    "/Users/jindrich/Projects/mayo_sleep_pat/SmallDataset21Oct25/Data/Dataset_21Oct25/Data_Only"
)

# Set to an integer for short debug runs, or keep None to process everything.
MAX_FILES = None

# RUN_ID is generated automatically at import time.
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# RUN_TAG is the human-readable label to help distinguish parameter sweeps.
# Changing it affects output folder names only; it does not change calculations.
RUN_TAG = "deltaHRinclud"

# Top-level feature selection. These switches are meant to answer the question
# "what should this run produce?". If a feature is disabled here, the goal is to
# keep it out of computation, tables, plots, and file outputs.
# Recommended workflow:
#   1. decide which features should be part of the run here
#   2. only then tune the detailed HR / HRV / PSD / burden knobs below
#
# In practice:
#   - hr                  -> PAT-derived heart-rate series and HR summary outputs
#   - hrv                 -> RMSSD/SDNN/LF/HF/LF-HF calculations, HRV plots, HRV CSV
#   - psd                 -> spectral features and PSD report pages
#   - delta_hr            -> event-response HR metrics and event-response HR plots
#   - pat_burden          -> PAT amplitude loading, burden metric, burden subplot/rows
#   - sleep_combo_summary -> extra fixed sleep-subset comparison summaries
#   - report_pdf          -> main multi-page PDF report
#   - peaks_debug_pdf     -> PAT peak-debug PDF
FEATURES = {
    "hr": True,
    "hrv": True,
    "psd": False,
    "delta_hr": True,
    "pat_burden": False,
    "sleep_combo_summary": True,
    "report_pdf": True,
    "peaks_debug_pdf": False,
}


def _slug(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s


# =============================================================================
# Sleep Stage Mapping And Inclusion Policy
# =============================================================================
# These settings decide which sleep stages are considered "included" for HRV,
# PSD, PAT burden, and some plotting masks. This is one of the highest-impact
# groups in the config because it changes which parts of the night contribute.

# Canonical numeric codes used internally:
#   0 = Wake
#   1 = Light sleep
#   2 = Deep sleep
#   3 = REM
SLEEP_MAPPING = {
    "WK": 0,
    "L. Sleep": 1,
    "D. Sleep": 2,
    "REM": 3,
}

# Master switch for sleep-stage masking. Turning this off makes the workflow use
# available time points regardless of stage labels.
ENABLE_SLEEP_STAGE_MASKING = True

# Sleep stage filtering policy.
# Options:
#   "all_sleep"               -> include stages 1, 2, 3
#   "all_sleep_incluidng_wake" -> include 0, 1, 2, 3
#   "rem_only"                -> include only REM
#   "nrem_only"               -> include light + deep
#   "deep_only"               -> include only deep sleep
#   "nrem_light_only"         -> include only light sleep
#   "custom"                  -> use SLEEP_INCLUDE_LABELS / SLEEP_INCLUDE_NUMERIC
SLEEP_STAGE_POLICY = "all_sleep_incluidng_wake"

# Used only when SLEEP_STAGE_POLICY == "custom".
# Numeric codes take priority; labels are a fallback convenience.
SLEEP_INCLUDE_LABELS = {"L. Sleep", "D. Sleep", "REM"}
SLEEP_INCLUDE_NUMERIC = {1}


def sleep_include_numeric() -> set[int]:
    """
    Resolve the include set of numeric sleep codes from SLEEP_STAGE_POLICY.
    """
    s = (SLEEP_STAGE_POLICY or "all_sleep").lower()
    if s == "n2n3_only":
        s = "deep_only"

    if s == "all_sleep":
        return {1, 2, 3}
    if s == "all_sleep_incluidng_wake":
        return {0, 1, 2, 3}
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
    try:
        parts.append(sleep_stage_suffix().lstrip("_"))
    except Exception:
        pass
    tag = _slug(RUN_TAG)
    if tag:
        parts.append(tag)
    return "__".join(parts)


# Output subfolders inherit the current run id / sleep policy / tag.
OUTPUT_SUBFOLDER = f"ViewPatPlotsOverlay__{run_suffix()}"
HR_OUTPUT_SUBFOLDER = f"HR__{run_suffix()}"
HRV_OUTPUT_SUBFOLDER = f"HRV__{run_suffix()}"
PAT_BURDEN_OUTPUT_SUBFOLDER = f"PATBurden__{run_suffix()}"
PSD_OUTPUT_SUBFOLDER = f"PSD__{run_suffix()}"


# =============================================================================
# EDF Channels And Aux CSV Column Mapping
# =============================================================================
# These settings align the code with the actual file/channel names in your EDFs
# and synchronized CSVs. Edit these when a new site or export format uses a
# different naming convention.

VIEW_PAT_CHANNEL_NAME = "VIEW_PAT"
HR_CHANNEL_NAME = "DERIVED_HR"
PAT_AMP_CHANNEL_NAME = "DERIVED_PAT_AMP"
ACTIGRAPH_CHANNEL_NAME = "ACTIGRAPH"

AUX_CSV_ENABLED = True
AUX_CSV_EXTENSION = ".csv"
AUX_CSV_SEP = ","
AUX_CSV_TIME_COLUMN_RAW = "Time"
AUX_CSV_TIME_SEC_COLUMN = "time_sec"
AUX_CSV_USE_COLUMNS = None

# Mapping from internal names to actual CSV column names.
COL_NAMES: Dict[str, str] = {
    "time": "Time",
    "stage": "WP Stages",
    "spo2": "SpO2",
    "desat_flag": "Desaturation",
    "exclude_hr_flag": "Exclude HR",
    "exclude_pat_flag": "Exclude PAT",
    "evt_central_3": "A/H central-3% (Last second)",
    "evt_obstructive_3": "A/H obstructive-3% (Last second)",
    "evt_unclassified_3": "A/H unclassified-3% (Last second)",
}


# =============================================================================
# Shared Exclusion And Event-Gating Policy
# =============================================================================
# These knobs determine which aux events become exclusion masks for downstream
# metrics. If a user wants more or less aggressive masking, this is the first
# place to inspect.

# Shared exclusion inputs used by HR/HRV/PSD/PAT burden plotting and calculations.
HRV_EXCLUSION_EVENT_COLUMNS = [
    "evt_central_3",
    "evt_obstructive_3",
    "evt_unclassified_3",
    "exclude_hr_flag",
    "exclude_pat_flag",
    # "desat_flag",
]

# Fixed exclusion windows around events. Increasing these will remove more data
# around flagged moments; setting them to zero keeps only the event instant logic.
HRV_EXCLUSION_PRE_SEC = 15.0
HRV_EXCLUSION_POST_SEC = 30.0

# Desaturation-dependent exclusion windows. When enabled, exclusion windows can
# be driven by desaturation runs rather than fixed event padding.
HRV_EXCLUSION_USE_DESAT_WINDOWS = False
HRV_EXCLUSION_DESAT_COLUMN_KEY = "desat_flag"
HRV_EXCLUSION_DESAT_START_PAD_SEC = 15
HRV_EXCLUSION_DESAT_END_PAD_SEC = 0
HRV_EXCLUSION_DESAT_MIN_RUN_SEC = 5.0
HRV_EXCLUSION_DESAT_LOOKBACK_SEC = 0.0
HRV_EXCLUSION_DESAT_LOOKAHEAD_SEC = 0.0


# =============================================================================
# Plotting And Report Layout
# =============================================================================
# These settings affect report appearance and pagination, not the underlying
# physiological calculations. They are safe knobs for users who want different
# report density or debugging views.

ENABLE_VIEW_PAT_OVERLAY_PLOTS = FEATURES["report_pdf"]
ENABLE_PAT_SIGNAL_PLOT = True
ENABLE_PAT_PEAK_DEBUG_PLOTS = FEATURES["peaks_debug_pdf"]

SEGMENT_MINUTES = 15
PAT_PEAK_DEBUG_SEGMENT_MINUTES = 1.0
OVERVIEW_PANEL_HOURS = 2.0


# =============================================================================
# PAT Filtering
# =============================================================================
# These bandpass settings affect how the PAT waveform is preprocessed before peak
# detection and downstream metrics. They are important if the signal is noisy or
# if acquisition hardware changes its spectral characteristics.

PAT_BANDPASS_LOWCUT_HZ = 0.5
PAT_BANDPASS_HIGHCUT_HZ = 8.0
PAT_BANDPASS_ORDER = 4


# =============================================================================
# HR Feature
# =============================================================================
# These knobs control PAT-derived heart-rate estimation. Most users should only
# change smoothing, bpm clamps, or gap settings unless they are validating a new
# detector on a different signal quality profile.

ENABLE_HR = FEATURES["hr"]
HR_TARGET_FS_HZ = 1.0

# Physiologic RR limits (seconds).
HR_MIN_RR_SEC = 0.30
HR_MAX_RR_SEC = 2.50

# HR clamps (bpm).
HR_MIN_BPM = 30.0
HR_MAX_BPM = 220.0

# EDF scale for legacy/reference HR channel handling.
HR_EDF_SCALE_FACTOR = 0.01

# Peak detection tuning. Higher prominence is stricter; lower prominence is more
# permissive and may catch more false peaks.
HR_PEAK_PROMINENCE_FACTOR = 0.30

# HR smoothing / robustness.
HR_SMOOTHING_WINDOW_SEC = 4.0
HR_HAMPEL_WINDOW_SEC = 8.0
HR_HAMPEL_SIGMA = 2.0

# Optional HR slope limiter (0 disables).
HR_MAX_DELTA_BPM_PER_SEC = 10.0

# HR interpolation will not bridge gaps larger than this.
HR_MAX_RR_GAP_SEC = 2.5


# =============================================================================
# RR Cleaning Shared By HR And HRV
# =============================================================================
# These thresholds define how aggressively RR intervals are filtered after PAT
# peak detection. Tightening them gives cleaner but shorter RR series; loosening
# them preserves more data but may admit more artifacts.

HR_RR_MEDFILT_KERNEL = 5
HR_RR_OUTLIER_REL_THR = 0.30
HR_RR_GAP_FACTOR = 2.4
HR_RR_JUMP_REL_THR = 0.6
HR_RR_ALT_SHORT_REL = 0.25
HR_RR_ALT_LONG_REL = 0.35
HR_RR_MIN_GOOD_RUN = 3


# =============================================================================
# HRV Core Settings
# =============================================================================
# These control the core RMSSD / SDNN / LF-HF calculations. Window sizes and gap
# criteria are especially important because they decide how much stable data is
# required before a value is reported.

HRV_TARGET_FS_HZ = 1.0
HRV_WINDOW_SEC = 300.0
HRV_MIN_INTERVALS_PER_WINDOW = 6
HRV_SMOOTHING_WINDOW_SEC = 5.0

# Optional HRV Hampel filtering on the RMSSD series.
HRV_HAMPEL_WINDOW_SEC = 30.0
HRV_HAMPEL_SIGMA = 3.0

# Gap handling for RMSSD windows.
HRV_MAX_RR_GAP_SEC = 8.0
HRV_RMSSD_MIN_SPAN_SEC = 5.0

# Frequency-domain settings.
HRV_TACHO_RESAMPLE_HZ = 4.0
HRV_MIN_FREQ_DOMAIN_SEC = 120.0
HRV_MAX_TACHO_GAP_SEC = 3.0

# Publication-style fixed LF/HF windows.
HRV_LFHF_FIXED_WINDOW_SEC = 120.0
HRV_LFHF_FIXED_HOP_SEC = 120.0
HRV_LFHF_FIXED_MIN_RR = 0


# =============================================================================
# RMSSD Robustness Controls
# =============================================================================
# These tune how aggressively successive RR differences are cleaned before RMSSD
# is computed. They are useful when PAT detections contain occasional spikes.

HRV_RMSSD_DIFF_HARD_CAP_MS = 400.0
HRV_RMSSD_DIFF_MAD_SIGMAS = 4.0
HRV_RMSSD_MIN_DIFFS = 3
HRV_RMSSD_FLOOR_MS = 2.0
HRV_MIN_WINDOW_COVERAGE = 0.2

# Optional veto for windows dominated by large beat-to-beat jumps.
HRV_RMSSD_VETO_BIGDIFF = False
HRV_RMSSD_BIGDIFF_THR_MS = 300.0
HRV_RMSSD_BIGDIFF_MAX_FRAC = 0.35


# =============================================================================
# Time-Varying HRV Series For Plotting
# =============================================================================
# These control the sliding-window HRV curves used in reports. They mostly
# affect temporal smoothness and how many windows survive for LF/HF-like traces.

HRV_TV_WINDOW_SEC = 300.0
HRV_TV_STEP_HZ = 1.0
HRV_TV_TACHO_RESAMPLE_HZ = 4.0
HRV_TV_MIN_RR_PER_WINDOW = 10
HRV_TV_MIN_FREQ_DOMAIN_SEC = 60.0
HRV_TV_MAX_TACHO_GAP_SEC = 6.0

# Fixed y-lims are useful when comparing many nights visually.
HRV_PLOT_USE_FIXED_YLIMS = False
HRV_PLOT_RMSSD_YLIM = (0.0, 80.0)
HRV_PLOT_SDNN_YLIM = (0.0, 150.0)
HRV_PLOT_LFHF_POWER_YLIM = (1.0, 10000.0)
HRV_PLOT_LFHF_RATIO_YLIM = (0.0, 10.0)

# Binning used in cross-night HRV summary plots.
HRV_PLOT_BIN_SEC = 10.0 * 60.0
HRV_PLOT_BIN_MIN_COUNT = 3


# =============================================================================
# PSD / Spectral Analysis
# =============================================================================
# These bands are used for the PAT/HRV-related PSD summaries. Change them only
# if you intentionally want different Mayer / respiratory band definitions.

PSD_MAX_FREQ_HZ = 5.0
PSD_NPERSEG = 4096
PSD_MAYER_BAND = (0.04, 0.15)
PSD_RESP_BAND = (0.15, 0.23)


# =============================================================================
# Event-Response HR Feature
# =============================================================================
# This feature now focuses on event-centered HR response summaries and visual
# overlays on the original HR signal rather than on a separate lag-difference
# signal plot.

ENABLE_DELTA_HR = FEATURES["delta_hr"]
# Plot mode for segment pages:
#   "subplot" -> extra row showing event-response HR windows on the HR signal
#   "twinx"   -> reserved for future overlay mode
DELTA_HR_PLOT_MODE = "subplot"

# Event-response HR definition.
# The HR signal is smoothed first, then for each event start time ts:
#   event window    = [ts, ts + HR_EVENT_WINDOW_SEC]
#   recovery window = [ts + HR_EVENT_WINDOW_SEC, ts + HR_EVENT_RECOVERY_END_SEC]
# Two derived metrics are reported from the same windows:
#   - trough-to-peak response = max(HR in recovery window) - min(HR in event window)
#   - mean-to-peak delta HR   = max(HR in recovery window) - mean(HR in event window)
# An event is skipped if either window has insufficient valid HR samples or if a
# new event begins before the recovery window ends.
HR_EVENT_SMOOTH_SEC = 5.0
HR_EVENT_WINDOW_SEC = 15.0
HR_EVENT_RECOVERY_END_SEC = 45.0
HR_EVENT_MIN_SAMPLES = 3


# =============================================================================
# PAT Burden
# =============================================================================
# These settings tune how PAT burden episodes are normalized and baseline-corrected.
# Users interested in the burden metric should mainly experiment here.

ENABLE_PAT_BURDEN = FEATURES["pat_burden"]
PAT_BURDEN_BASELINE_LOOKBACK_SEC = 30.0
PAT_BURDEN_BASELINE_MIN_SAMPLES = 5
PAT_BURDEN_BASELINE_PCTL = 95.0
PAT_BURDEN_MIN_EPISODE_SEC = 5.0
PAT_BURDEN_RELATIVE = False

ENABLE_SLEEP_COMBO_SUMMARY = FEATURES["sleep_combo_summary"]
