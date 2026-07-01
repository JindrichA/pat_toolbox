from __future__ import annotations

import argparse
from pathlib import Path

from PyPDF2 import PdfReader, PdfWriter


# This page map matches the current Harmonics report configuration:
# - report_pdf=True
# - pat_harmonics=True
# - pat_paper_harmonics=True
# - hr=True
# - delta_hr=True
# - pat_burden=True
# - pwa_drop=True
# - ENABLE_PRV_REPORT_PAGES=False
# - ENABLE_SPO2_VALIDATION_PLOTS=False
#
# Page indexes are zero-based. With the current report writer, overview pages are
# emitted immediately after the front page.
OUTPUTS = {
    0: "00_front_pages.pdf",
    1: "01_pat_harmonics_overview.pdf",
    2: "02_paper_pat_harmonics_overview.pdf",
    3: "03_paper_pat_harmonics_normalized_coefficients.pdf",
    4: "04_hr_overview.pdf",
    5: "05_dhr_overview.pdf",
    6: "06_pat_burden_overview.pdf",
    7: "07_pwa_drop_overview.pdf",
}

DEFAULT_BASE_OUTPUT_DIR = Path("/Users/jindrich/Projects/mayo_data/output")
DEFAULT_REPORT_PREFIX = "ViewPatPlotsOverlay__"
DEFAULT_RUN_TAG = "harmonics"


def _find_latest_report_folder(base_output_dir: Path) -> Path:
    candidates = [
        path
        for path in base_output_dir.iterdir()
        if path.is_dir()
        and path.name.startswith(DEFAULT_REPORT_PREFIX)
        and path.name.lower().endswith(f"__{DEFAULT_RUN_TAG}")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No {DEFAULT_REPORT_PREFIX}*__{DEFAULT_RUN_TAG} folders found in {base_output_dir}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _source_pdf_files(pdf_folder: Path, output_names: set[str]) -> list[Path]:
    return sorted(
        path
        for path in pdf_folder.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".pdf"
        and path.name not in output_names
    )


def merge_pages(pdf_folder: Path, output_dir: Path | None = None) -> None:
    output_dir = output_dir or pdf_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    output_names = set(OUTPUTS.values())
    pdf_files = _source_pdf_files(pdf_folder, output_names)
    if not pdf_files:
        raise FileNotFoundError(f"No source PDFs found in {pdf_folder}")

    print(f"Source folder: {pdf_folder}")
    print(f"Output folder: {output_dir}")
    print(f"Source PDFs: {len(pdf_files)}")

    for page_index, output_name in OUTPUTS.items():
        writer = PdfWriter()
        added = 0

        for pdf_path in pdf_files:
            try:
                reader = PdfReader(str(pdf_path))
                if len(reader.pages) > page_index:
                    writer.add_page(reader.pages[page_index])
                    added += 1
                else:
                    print(
                        f"Skipping page {page_index + 1} in {pdf_path.name}: "
                        f"file has only {len(reader.pages)} page(s)"
                    )
            except Exception as exc:
                print(f"Skipping {pdf_path.name}: {exc}")

        output_path = output_dir / output_name
        with output_path.open("wb") as f:
            writer.write(f)
        print(f"Created: {output_path} ({added} page(s))")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge the important current Harmonics report pages across all report PDFs."
    )
    parser.add_argument(
        "pdf_folder",
        nargs="?",
        type=Path,
        help="Folder containing per-recording report PDFs. Defaults to latest ViewPatPlotsOverlay__*__harmonics folder.",
    )
    parser.add_argument(
        "--base-output-dir",
        type=Path,
        default=DEFAULT_BASE_OUTPUT_DIR,
        help="Base output directory used when pdf_folder is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder for merged PDFs. Defaults to pdf_folder.",
    )
    args = parser.parse_args()

    pdf_folder = args.pdf_folder or _find_latest_report_folder(args.base_output_dir)
    merge_pages(pdf_folder=pdf_folder, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
