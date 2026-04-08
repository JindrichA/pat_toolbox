# pat_toolbox/io_aux_csv.py

from __future__ import annotations

from .io.aux_events import (
    build_event_exclusion_mask,
    build_time_exclusion_mask,
    desat_windows_from_aux,
    get_event_times,
    get_rr_exclusion_mask,
)
from .io.aux_normalize import normalize_aux_df, parse_time_column_to_seconds
from .io.aux_reader import find_aux_csv_for_edf, read_aux_csv_for_edf, read_raw_aux_csv

__all__ = [
    "build_event_exclusion_mask",
    "build_time_exclusion_mask",
    "desat_windows_from_aux",
    "find_aux_csv_for_edf",
    "get_event_times",
    "get_rr_exclusion_mask",
    "normalize_aux_df",
    "parse_time_column_to_seconds",
    "read_aux_csv_for_edf",
    "read_raw_aux_csv",
]
