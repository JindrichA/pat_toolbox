from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

from .. import config, paths


def save_pat_harmonics_windows_to_csv(
    edf_path: Path,
    windows: Iterable[dict[str, Any]],
) -> Path | None:
    rows = list(windows)
    if not rows:
        return None
    out_folder = paths.get_output_folder(getattr(config, "PAT_HARMONICS_OUTPUT_SUBFOLDER", config.OUTPUT_SUBFOLDER))
    out_csv = out_folder / f"{edf_path.stem}__PAT_Harmonics_Windows.csv"
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_csv


def save_pat_harmonics_summary_to_csv(
    edf_path: Path,
    summary: Mapping[str, Any] | None,
) -> Path | None:
    if not summary:
        return None
    out_folder = paths.get_output_folder(getattr(config, "PAT_HARMONICS_OUTPUT_SUBFOLDER", config.OUTPUT_SUBFOLDER))
    out_csv = out_folder / f"{edf_path.stem}__PAT_Harmonics_Summary.csv"
    row = {"edf_file": edf_path.name, **dict(summary)}
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return out_csv
