from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

from .. import config, paths


def save_pwa_drop_events_to_csv(
    edf_path: Path,
    events: Iterable[dict[str, Any]],
) -> Path | None:
    rows = list(events)
    if not rows:
        return None

    out_folder = paths.get_output_folder(getattr(config, "PWA_DROP_OUTPUT_SUBFOLDER", config.OUTPUT_SUBFOLDER))
    out_csv = out_folder / f"{edf_path.stem}__PWA_Drop_Events.csv"
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


def save_pwa_drop_summary_to_csv(
    edf_path: Path,
    summary: Mapping[str, Any] | None,
) -> Path | None:
    if not summary:
        return None

    out_folder = paths.get_output_folder(getattr(config, "PWA_DROP_OUTPUT_SUBFOLDER", config.OUTPUT_SUBFOLDER))
    out_csv = out_folder / f"{edf_path.stem}__PWA_Drop_Summary.csv"
    row = {"edf_file": edf_path.name, **dict(summary)}
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return out_csv
