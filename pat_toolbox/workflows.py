# pat_toolbox/workflows.py

from __future__ import annotations

from pathlib import Path

from .context import RecordingContext
from .workflow_steps_load import filter_pat, load_aux_csv, load_pat, load_pat_amp
from .workflow_steps_metrics import (
    compute_delta_hr_step,
    compute_hr_from_pat_step,
    compute_hrv_step,
    compute_pat_burden_step,
    compute_sleep_combo_summaries_step,
)
from .workflow_steps_output import append_summary_step, build_pdf_step, build_peaks_debug_pdf_step


def process_view_pat_overlay_for_file(edf_path: Path) -> Path | None:
    print(f"Processing EDF for VIEW_PAT + HR + HRV plotting: {edf_path}")
    ctx = RecordingContext(edf_path=edf_path)

    try:
        ctx.t_hr_edf = None
        ctx.hr_edf = None
        ctx.hr_edf_raw = None
        ctx.delta_hr_edf = None
        ctx.delta_hr_edf_evt = None
        ctx.pearson_r = None
        ctx.spear_rho = None
        ctx.rmse = None

        load_pat(ctx)
        filter_pat(ctx)
        load_pat_amp(ctx)
        load_aux_csv(ctx)
        compute_sleep_combo_summaries_step(ctx)
        compute_pat_burden_step(ctx)
        compute_hr_from_pat_step(ctx)
        compute_delta_hr_step(ctx)
        compute_hrv_step(ctx)
        build_pdf_step(ctx)
        build_peaks_debug_pdf_step(ctx)
        append_summary_step(ctx)

        return ctx.pdf_path
    except Exception as e:
        print(f"  ERROR: failed processing {edf_path.name}: {e}")
        return None
