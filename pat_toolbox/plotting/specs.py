from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class EventSpec:
    col: str
    label: str
    color: str


DEFAULT_EVENT_PLOT_SPEC: List[EventSpec] = [
    EventSpec("desat_flag", "Desat", "tab:blue"),
    EventSpec("evt_central_3", "Central A/H 3%", "tab:red"),
    EventSpec("evt_obstructive_3", "Obstr A/H 3%", "tab:orange"),
    EventSpec("evt_unclassified_3", "Unclass A/H 3%", "tab:green"),
    EventSpec("exclude_hr_flag", "HR excluded", "tab:purple"),
    EventSpec("exclude_pat_flag", "PAT excluded", "tab:olive"),
]
