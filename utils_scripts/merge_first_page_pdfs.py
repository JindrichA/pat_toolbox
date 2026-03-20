import os
from PyPDF2 import PdfReader, PdfWriter

# Folder containing the PDFs
PDF_FOLDER = "/Users/jindrich/Projects/PAT_022026_output_data/ViewPatPlotsOverlay__20260310_153001__all_sleep__desat_2min_fd_fixed/"

# Output PDF (saved in the same folder)
OUTPUT_FILE = os.path.join(PDF_FOLDER, "merged_first_pages.pdf")

writer = PdfWriter()

# Collect PDF files
pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]

# Sort for predictable order
pdf_files.sort()

for pdf_file in pdf_files:

    # avoid including the output file itself
    if pdf_file == "merged_first_pages.pdf":
        continue

    pdf_path = os.path.join(PDF_FOLDER, pdf_file)

    try:
        reader = PdfReader(pdf_path)

        if len(reader.pages) > 0:
            writer.add_page(reader.pages[0])

    except Exception as e:
        print(f"Skipping {pdf_file}: {e}")

# Save merged file
with open(OUTPUT_FILE, "wb") as f:
    writer.write(f)

print(f"Created: {OUTPUT_FILE}")