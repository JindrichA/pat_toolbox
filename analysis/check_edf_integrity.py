from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pyedflib

from pat_toolbox import config, io_edf


def _check_single_edf(edf_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "file": edf_path,
        "ok": False,
        "n_signals": 0,
        "error": "",
    }

    try:
        reader = pyedflib.EdfReader(str(edf_path))
    except Exception as e:
        out["error"] = f"open_failed: {e}"
        return out

    try:
        labels = list(reader.getSignalLabels())
        n_signals = int(reader.signals_in_file)
        nsamples = reader.getNSamples()

        out["n_signals"] = n_signals

        if n_signals <= 0:
            out["error"] = "no_signals"
            return out

        if len(labels) != n_signals:
            out["error"] = f"label_count_mismatch: labels={len(labels)} signals={n_signals}"
            return out

        if len(nsamples) != n_signals:
            out["error"] = f"sample_count_mismatch: nsamples={len(nsamples)} signals={n_signals}"
            return out

        for idx, label in enumerate(labels):
            sfreq = float(reader.getSampleFrequency(idx))
            n_samples = int(nsamples[idx])

            if sfreq <= 0:
                out["error"] = f"invalid_sample_frequency: channel={label!r} sfreq={sfreq}"
                return out

            if n_samples <= 0:
                out["error"] = f"empty_channel: channel={label!r} n_samples={n_samples}"
                return out

            signal = reader.readSignal(idx)
            if signal is None:
                out["error"] = f"read_failed: channel={label!r} returned None"
                return out

            if len(signal) != n_samples:
                out["error"] = (
                    f"length_mismatch: channel={label!r} expected={n_samples} got={len(signal)}"
                )
                return out

        out["ok"] = True
        return out
    except Exception as e:
        out["error"] = f"read_failed: {e}"
        return out
    finally:
        reader.close()


def main() -> int:
    edf_folder = Path(sys.argv[1]) if len(sys.argv) > 1 else config.EDF_FOLDER

    try:
        edf_files = io_edf.list_edf_files(edf_folder)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 2

    if not edf_files:
        print(f"No .edf files found in {edf_folder}")
        return 0

    print(f"EDF folder: {edf_folder}")
    print(f"Found {len(edf_files)} EDF file(s).")
    print()

    failed: list[dict[str, Any]] = []
    for i, edf_path in enumerate(edf_files, start=1):
        result = _check_single_edf(edf_path)
        if result["ok"]:
            print(f"[{i:>4}/{len(edf_files)}] OK   {edf_path.name} ({result['n_signals']} channels)")
        else:
            failed.append(result)
            print(f"[{i:>4}/{len(edf_files)}] FAIL {edf_path.name} :: {result['error']}")

    print()
    print(f"Checked {len(edf_files)} EDF file(s).")
    print(f"Passed: {len(edf_files) - len(failed)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print()
        print("Failed files:")
        for item in failed:
            print(f"  {item['file'].name}: {item['error']}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
