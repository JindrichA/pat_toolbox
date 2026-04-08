from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .. import config
from .aux_normalize import normalize_aux_df


def find_aux_csv_for_edf(edf_path: Path) -> Optional[Path]:
    """
    Given an EDF path, find the corresponding auxiliary CSV.

    Default strategy:
      <edf_stem>.csv in the same folder.
    """
    if not getattr(config, "AUX_CSV_ENABLED", True):
        return None

    aux_path = edf_path.with_suffix(getattr(config, "AUX_CSV_EXTENSION", ".csv"))
    return aux_path if aux_path.exists() else None


def read_raw_aux_csv(aux_path: Path) -> Optional[pd.DataFrame]:
    """Low-level CSV reader."""
    try:
        usecols = getattr(config, "AUX_CSV_USE_COLUMNS", None)
        df = pd.read_csv(aux_path, sep=getattr(config, "AUX_CSV_SEP", ","), usecols=usecols)
    except Exception as e:
        print(f"  WARNING: could not read aux CSV {aux_path}: {e}")
        return None

    if df is None or df.empty:
        print(f"  WARNING: aux CSV {aux_path} is empty.")
        return None

    return df


def read_aux_csv_for_edf(edf_path: Path) -> Optional[pd.DataFrame]:
    aux_csv = find_aux_csv_for_edf(edf_path)
    if aux_csv is None:
        return None

    df = read_raw_aux_csv(aux_csv)
    if df is None:
        return None

    return normalize_aux_df(df)
