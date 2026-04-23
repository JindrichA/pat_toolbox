import os
from PyPDF2 import PdfReader, PdfWriter

# Folder containing the PDFs
PDF_FOLDER = "/Users/jindrich/Projects/PAT_022026_output_data/ViewPatPlotsOverlay__20260422_182418__nrem_only__ready/"

# Output names by page number (0-based index)
OUTPUTS = {
    0: "HRV_overview.pdf",
    1: "HR_overview.pdf",
    2: "RMSSD_overview.pdf",
    3: "SDNN_overview.pdf",
    4: "LF_and_HF_overview.pdf",
    5: "LF_HF_ratio.pdf",
}

# Collect PDF files
pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
pdf_files.sort()

# Avoid including outputs themselves if script is re-run
output_filenames = set(OUTPUTS.values())
pdf_files = [f for f in pdf_files if f not in output_filenames]

for page_index, output_name in OUTPUTS.items():
    writer = PdfWriter()

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)

        try:
            reader = PdfReader(pdf_path)

            if len(reader.pages) > page_index:
                writer.add_page(reader.pages[page_index])
            else:
                print(f"Skipping page {page_index + 1} in {pdf_file}: file has only {len(reader.pages)} page(s)")

        except Exception as e:
            print(f"Skipping {pdf_file}: {e}")

    output_path = os.path.join(PDF_FOLDER, output_name)

    with open(output_path, "wb") as f:
        writer.write(f)

    print(f"Created: {output_path}")