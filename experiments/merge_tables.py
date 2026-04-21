import os
import pandas as pd

# ----------------------------
# File paths
# ----------------------------
csv_path = "/Users/jindrich/Projects/PAT_022026_output_data/HR__20260413_115908__all_sleep_incluidng_wake__deltahrinclud/HR_HRV_EVENT_HR_summary__multi_sleep_summary__20260413_115908.csv"
xlsx_path = "/Users/jindrich/Projects/mayo_sleep_pat/SmallDataset21Oct25/Data/20251020_parsed_last_deindentified.xlsx"

output_dir = "/Users/jindrich/Projects/pat_toolbox/experiments/merged_tables_output"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "merged_table.csv")

# ----------------------------
# Read files
# ----------------------------
hr_df = pd.read_csv(csv_path)
excel_df = pd.read_excel(xlsx_path)

# Clean column names
hr_df.columns = hr_df.columns.str.strip()
excel_df.columns = excel_df.columns.str.strip()

# ----------------------------
# Parse patient ID from edf_file
# Example: 073304_20250825_221736.edf -> 73304
# ----------------------------
if "edf_file" not in hr_df.columns:
    raise ValueError("Column 'edf_file' not found in HR CSV.")

hr_df["parsed_patient_id"] = (
    hr_df["edf_file"]
    .astype(str)
    .str.extract(r"^0*([0-9]+)_")[0]
)

hr_df["parsed_patient_id"] = pd.to_numeric(
    hr_df["parsed_patient_id"],
    errors="coerce"
).astype("Int64")

if "patient.ID" not in excel_df.columns:
    raise ValueError("Column 'patient.ID' not found in Excel file.")

excel_df["patient.ID"] = pd.to_numeric(
    excel_df["patient.ID"],
    errors="coerce"
).astype("Int64")

# ----------------------------
# Merge tables
# ----------------------------
merged_df = hr_df.merge(
    excel_df,
    left_on="parsed_patient_id",
    right_on="patient.ID",
    how="left",
    suffixes=("_hr", "_excel")
)

# ----------------------------
# Save merged table
# ----------------------------
merged_df.to_csv(output_path, index=False)

# ----------------------------
# Summary
# ----------------------------
print("HR CSV shape:", hr_df.shape)
print("Excel shape:", excel_df.shape)
print("Merged shape:", merged_df.shape)
print("Matched rows:", merged_df["patient.ID"].notna().sum())
print("Unmatched rows:", merged_df["patient.ID"].isna().sum())
print("Saved merged table to:", output_path)