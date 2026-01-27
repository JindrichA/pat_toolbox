# pat_toolbox/context.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class RecordingContext:
    # Identity / paths
    edf_path: Path
    edf_base: str = ""

    # Signals
    sfreq: Optional[float] = None
    view_pat: Optional[np.ndarray] = None
    view_pat_filt: Optional[np.ndarray] = None

    # PAT AMP
    t_pat_amp: Optional[np.ndarray] = None
    pat_amp: Optional[np.ndarray] = None

    # HR (PAT derived)
    t_hr_calc: Optional[np.ndarray] = None
    hr_calc: Optional[np.ndarray] = None

    # HR (EDF channel)
    t_hr_edf: Optional[np.ndarray] = None
    hr_edf: Optional[np.ndarray] = None

    # HR correlation
    pearson_r: Optional[float] = None
    spear_rho: Optional[float] = None
    rmse: Optional[float] = None

    # Aux CSV
    aux_df: Optional["pd.DataFrame"] = None

    # HRV
    t_hrv: Optional[np.ndarray] = None
    hrv_rmssd_raw: Optional[np.ndarray] = None
    hrv_rmssd_clean: Optional[np.ndarray] = None
    hrv_summary: Optional[Dict[str, float]] = None
    hrv_tv: Optional[Dict[str, np.ndarray]] = None

    # ACTIGRAPH (motion)
    t_actigraph: Optional[np.ndarray] = None
    actigraph: Optional[np.ndarray] = None


    # PSD peaks
    mayer_peak_freq: Optional[float] = None
    resp_peak_freq: Optional[float] = None

    # Outputs
    pdf_path: Optional[Path] = None
    hrv_csv_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if not self.edf_base:
            self.edf_base = self.edf_path.stem
