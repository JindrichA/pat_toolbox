from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .. import config, masking


@dataclass(frozen=True)
class EventSpec:
    col: str
    label: str
    color: str


DEFAULT_EVENT_PLOT_SPEC: List[EventSpec] = [
    EventSpec("desat_flag", "Desaturation", "tab:blue"),
    EventSpec("evt_central_3", "Central A/H 3%", "tab:red"),
    EventSpec("evt_obstructive_3", "Obstr A/H 3%", "tab:brown"),
    EventSpec("evt_unclassified_3", "Unclass A/H 3%", "tab:green"),
    EventSpec("exclude_hr_flag", "HR excluded", "tab:purple"),
    EventSpec("exclude_pat_flag", "PAT excluded", "tab:olive"),
]


def active_event_plot_spec() -> List[EventSpec]:
    policy = masking.policy_from_config()
    active_cols = set(policy.exclusion_columns)
    if policy.use_desat_windows:
        active_cols.add(str(getattr(config, "HRV_EXCLUSION_DESAT_COLUMN_KEY", "desat_flag")))
    return [spec for spec in DEFAULT_EVENT_PLOT_SPEC if spec.col in active_cols]
