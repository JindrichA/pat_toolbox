from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

from . import config


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dependencies: tuple[str, ...] = ()


FEATURE_SPECS: Dict[str, FeatureSpec] = {
    "hr": FeatureSpec("hr"),
    "hrv": FeatureSpec("hrv"),
    "psd": FeatureSpec("psd"),
    "delta_hr": FeatureSpec("delta_hr", dependencies=("hr",)),
    "pat_burden": FeatureSpec("pat_burden"),
    "sleep_combo_summary": FeatureSpec("sleep_combo_summary"),
    "report_pdf": FeatureSpec("report_pdf"),
    "peaks_debug_pdf": FeatureSpec("peaks_debug_pdf"),
}


def configured_features() -> Dict[str, bool]:
    configured = getattr(config, "FEATURES", {})
    merged = {name: False for name in FEATURE_SPECS}
    for name in merged:
        merged[name] = bool(configured.get(name, False))
    return merged


def is_enabled(name: str) -> bool:
    spec = FEATURE_SPECS[name]
    merged = configured_features()
    if not merged.get(name, False):
        return False
    return all(is_enabled(dep) for dep in spec.dependencies)


def any_enabled(*names: str) -> bool:
    return any(is_enabled(name) for name in names)


def workflow_requested() -> bool:
    return any_enabled(
        "hrv",
        "psd",
        "delta_hr",
        "pat_burden",
        "sleep_combo_summary",
        "report_pdf",
    )


def summary_requested() -> bool:
    return any_enabled("hr", "hrv", "psd", "delta_hr", "pat_burden", "sleep_combo_summary")


def segment_plot_requested(name: str) -> bool:
    if not is_enabled("report_pdf"):
        return False
    return is_enabled(name)


def enabled_feature_parts(candidates: Sequence[str]) -> list[str]:
    labels = {
        "hr": "HR",
        "hrv": "HRV",
        "psd": "PSD",
        "delta_hr": "DELTA",
        "pat_burden": "BURDEN",
        "sleep_combo_summary": "SLEEP_COMBO",
    }
    return [labels[name] for name in candidates if is_enabled(name)]
