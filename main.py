# main.py
import os
import sys
import subprocess
import atexit
from pathlib import Path

from tqdm import tqdm

from pat_toolbox import config, io_edf, paths, workflows
from pat_toolbox.metrics import hr


def _maybe_start_caffeinate():
    """
    Prevent macOS from sleeping while this Python process is running.
    """
    if sys.platform != "darwin":
        return None
    if not getattr(config, "ENABLE_CAFFEINATE", True):
        return None

    args = getattr(config, "CAFFEINATE_ARGS", ["-dimsu"])
    cmd = ["caffeinate", *args, "-w", str(os.getpid())]

    try:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        def _cleanup():
            try:
                if p.poll() is None:
                    p.terminate()
            except Exception:
                pass

        atexit.register(_cleanup)
        print(f"[caffeinate] Enabled ({' '.join(cmd)})")
        return p
    except Exception as e:
        print(f"[caffeinate] WARNING: could not start caffeinate: {e}")
        return None


def main():
    # ------------------------------------------------------------------
    # Configuration (PyCharm / config-driven)
    # ------------------------------------------------------------------
    edf_folder: Path = config.EDF_FOLDER

    enable_plots = config.ENABLE_VIEW_PAT_OVERLAY_PLOTS
    enable_hr = config.ENABLE_HR
    enable_peak_debug = getattr(config, "ENABLE_PAT_PEAK_DEBUG_PLOTS")

    # ------------------------------------------------------------------
    # Prevent sleep (macOS)
    # ------------------------------------------------------------------
    _maybe_start_caffeinate()

    # ------------------------------------------------------------------
    # Startup info
    # ------------------------------------------------------------------
    print(f"EDF folder           : {edf_folder}")
    print(f"Output PDF folder    : {paths.get_output_folder()}")
    print(f"Segment length       : {config.SEGMENT_MINUTES} minutes")
    print(
        f"Band-pass filter     : "
        f"{config.PAT_BANDPASS_LOWCUT_HZ}–{config.PAT_BANDPASS_HIGHCUT_HZ} Hz "
        f"(order {config.PAT_BANDPASS_ORDER})"
    )
    print(f"VIEW_PAT plotting    : {enable_plots}")
    print(f"HR from PAT          : {enable_hr} (target fs = {config.HR_TARGET_FS_HZ} Hz)")
    print(f"PAT peaks debug PDF  : {enable_peak_debug}")
    print(f"RUN_ID               : {getattr(config, 'RUN_ID', 'default')}")
    print()

    # ------------------------------------------------------------------
    # List EDF files
    # ------------------------------------------------------------------
    try:
        edf_files = io_edf.list_edf_files(edf_folder)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    if not edf_files:
        print("No .edf files found.")
        return

    max_files = getattr(config, "MAX_FILES", None)
    if isinstance(max_files, int) and max_files > 0:
        edf_files = edf_files[:max_files]
        print(f"DEBUG: limiting to first {max_files} EDF file(s).")
        print()


    print(f"Found {len(edf_files)} EDF file(s).")
    print()

    if not enable_plots and not enable_hr and not enable_peak_debug:
        print("Nothing to do (all features disabled).")
        return

    # ------------------------------------------------------------------
    # Main processing loop with tqdm
    # ------------------------------------------------------------------
    for edf_path in tqdm(edf_files, desc="Processing EDF files", unit="file"):
        try:
            did_workflow = False

            # If we run the workflow, it already computes HR/HRV and (optionally)
            # makes the peaks debug PDF based on config.ENABLE_PAT_PEAK_DEBUG_PLOTS.
            if enable_plots:
                workflows.process_view_pat_overlay_for_file(edf_path)
                did_workflow = True

            # Avoid redundant HR computation if workflow already ran.
            # (Note: workflow does not save HR CSV, so keep this if you need CSVs.)
            if enable_hr and not did_workflow:
                hr.compute_hr_for_edf_file(edf_path, save_csv=True)

            # Avoid redundant peaks debug PDF if workflow already ran.
            if enable_peak_debug and not did_workflow:
                hr.create_peaks_debug_pdf_for_edf(edf_path)

        except Exception as e:
            # Never kill the whole batch
            tqdm.write(f"ERROR processing {edf_path.name}: {e}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print()
    print("Done.")
    print("  VIEW_PAT PDFs in      :", paths.get_output_folder())
    print("  HR CSVs (if any) in   :", paths.get_output_folder(config.HR_OUTPUT_SUBFOLDER))
    if enable_peak_debug:
        print("  PAT peaks debug PDFs in:", paths.get_output_folder())


if __name__ == "__main__":
    main()
