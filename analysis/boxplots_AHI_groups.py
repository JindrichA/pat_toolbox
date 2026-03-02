import pandas as pd
import re

# =========================
# HARD-CODED FILE PATHS
# =========================
EXCEL_PATH = r"/Users/jindrich/Projects/mayo_sleep_pat/SmallDataset21Oct25/Data/20251020_parsed_last_deindentified.xlsx"
CSV_PATH   = r"/Users/jindrich/Projects/mayo_sleep_pat/PAT_outputs012026/HR__20260204_101906__nrem_only__desat_2min_fd_fixed/HR_HRV_summary__nrem_only__20260204_101906.csv"
OUTPUT_PATH = r"merged_output_22012026.csv"

# =========================
# LOAD FILES
# =========================
print("Loading Excel file...")
df_excel = pd.read_excel(EXCEL_PATH)

print("Loading CSV file...")
df_csv = pd.read_csv(CSV_PATH, sep=",", engine="python")

# =========================
# EXTRACT patient.ID FROM CSV (edf_file)
# =========================
# edf_file format: 073304_20250825_221736
def extract_patient_id(edf_file):
    if pd.isna(edf_file):
        return None
    # take first numeric block, remove leading zeros
    return int(re.match(r"(\d+)", str(edf_file)).group(1))

df_csv["patient.ID"] = df_csv["edf_file"].apply(extract_patient_id)

# =========================
# CLEAN EXCEL patient.ID
# =========================
df_excel["patient.ID"] = (
    df_excel["patient.ID"]
    .astype(str)
    .str.strip()
    .astype(int)
)

# =========================
# MERGE
# =========================
print("Merging datasets...")
df_merged = df_csv.merge(
    df_excel,
    on="patient.ID",
    how="left",
    suffixes=("_csv", "_excel")
)

# =========================
# SAVE OUTPUT
# =========================
df_merged.to_csv(OUTPUT_PATH, index=False)

print("Done!")
print(f"Merged rows: {len(df_merged)}")
print(f"Output saved to: {OUTPUT_PATH}")
