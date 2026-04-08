from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from .. import config, paths


def save_pat_burden_episodes_to_csv(
    edf_path: Path,
    episodes: Iterable[dict[str, Any]],
) -> Path | None:
    rows = list(episodes)
    if not rows:
        return None

    out_folder = paths.get_output_folder(getattr(config, "PAT_BURDEN_OUTPUT_SUBFOLDER", config.OUTPUT_SUBFOLDER))
    edf_base = edf_path.stem
    out_csv = out_folder / f"{edf_base}__PAT_Burden_Episodes.csv"

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
