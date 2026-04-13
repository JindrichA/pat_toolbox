import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# ----------------------------
# File paths
# ----------------------------
csv_path = "/Users/jindrich/Projects/pat_toolbox/experiments/data/HR_HRV_EVENT_HR_summary__multi_sleep_summary__20260409_152637.csv"
xlsx_path = "/Users/jindrich/Projects/mayo_sleep_pat/SmallDataset21Oct25/Data/20251020_parsed_last_deindentified.xlsx"

output_dir = "/Users/jindrich/Projects/pat_toolbox/experiments/ahi_feature_analysis"
os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# Read files
# ----------------------------
hr_df = pd.read_csv(csv_path)
excel_df = pd.read_excel(xlsx_path)

hr_df.columns = hr_df.columns.str.strip()
excel_df.columns = excel_df.columns.str.strip()

# ----------------------------
# Parse patient ID from edf_file
# Example: 073304_20250825_221736.edf -> 73304
# ----------------------------
hr_df["parsed_patient_id"] = (
    hr_df["edf_file"]
    .astype(str)
    .str.extract(r"^0*([0-9]+)_")[0]
)
hr_df["parsed_patient_id"] = pd.to_numeric(hr_df["parsed_patient_id"], errors="coerce").astype("Int64")

# Excel ID to numeric
excel_df["patient.ID"] = pd.to_numeric(excel_df["patient.ID"], errors="coerce").astype("Int64")

# ----------------------------
# Merge
# ----------------------------
merged = hr_df.merge(
    excel_df,
    left_on="parsed_patient_id",
    right_on="patient.ID",
    how="left",
    suffixes=("_hr", "_excel")
)

print("HR CSV shape:", hr_df.shape)
print("Excel shape:", excel_df.shape)
print("Merged shape:", merged.shape)
print("Matched rows:", merged["patient.ID"].notna().sum())
print("Unmatched rows:", merged["patient.ID"].isna().sum())

# ----------------------------
# AHI groups
# ----------------------------
merged["AHI"] = pd.to_numeric(merged["AHI"], errors="coerce")

bins = [0, 5, 15, 30, np.inf]
group_labels = ["0-5", "5-15", "15-30", ">30"]

merged["AHI_group"] = pd.cut(
    merged["AHI"],
    bins=bins,
    labels=group_labels,
    right=False
)

# ----------------------------
# Feature list you asked for
# ----------------------------
# feature_cols = """
# rmssd_mean_ms rmssd_median_ms sdnn_ms lf hf lf_hf lf_n_segments_used
# lf_hf_fixed_median lf_hf_fixed_mean lf_hf_fixed_n_windows_valid
# lf_hf_fixed_n_windows_total lf_hf_fixed_window_sec lf_hf_fixed_hop_sec
# hr_pat_nan_pct delta_hr_mean delta_hr_median delta_hr_std delta_hr_nan_pct
# delta_hr_n_used delta_hr_evt_mean delta_hr_evt_median delta_hr_evt_std
# delta_hr_evt_nan_pct delta_hr_evt_n_used hrv_rmssd_clean_nan_pct
# hrv_rmssd_raw_nan_pct aux_rows desat_n desat_pct exclude_pat_n
# exclude_pat_pct evt_central_3_n evt_central_3_pct evt_obstructive_3_n
# evt_obstructive_3_pct evt_unclassified_3_n evt_unclassified_3_pct
# sleep_mask_enabled sleep_included_n sleep_included_pct sleep_excluded_n
# sleep_excluded_pct sleep_wake_n sleep_wake_pct sleep_light_n sleep_light_pct
# sleep_deep_n sleep_deep_pct sleep_rem_n sleep_rem_pct
# combo_all_sleep_delta_hr_evt_mean combo_all_sleep_delta_hr_mean
# combo_all_sleep_event_windows_total combo_all_sleep_event_windows_used
# combo_all_sleep_lf_hf combo_all_sleep_lf_hf_fixed_n_windows_valid
# combo_all_sleep_lf_n_segments_used combo_all_sleep_pat_burden
# combo_all_sleep_peak_minus_baseline_hr combo_all_sleep_peak_to_trough_hr
# combo_all_sleep_post_peak_minus_pre_mean_hr combo_all_sleep_psd_valid_windows
# combo_all_sleep_rmssd_mean_ms combo_all_sleep_sdnn_ms
# combo_all_sleep_sleep_hours combo_deep_delta_hr_evt_mean
# combo_deep_delta_hr_mean combo_deep_event_windows_total
# combo_deep_event_windows_used combo_deep_lf_hf
# combo_deep_lf_hf_fixed_n_windows_valid combo_deep_lf_n_segments_used
# combo_deep_pat_burden combo_deep_peak_minus_baseline_hr
# combo_deep_peak_to_trough_hr combo_deep_post_peak_minus_pre_mean_hr
# combo_deep_psd_valid_windows combo_deep_rmssd_mean_ms combo_deep_sdnn_ms
# combo_deep_sleep_hours combo_nrem_delta_hr_evt_mean combo_nrem_delta_hr_mean
# combo_nrem_event_windows_total combo_nrem_event_windows_used combo_nrem_lf_hf
# combo_nrem_lf_hf_fixed_n_windows_valid combo_nrem_lf_n_segments_used
# combo_nrem_pat_burden combo_nrem_peak_minus_baseline_hr
# combo_nrem_peak_to_trough_hr combo_nrem_post_peak_minus_pre_mean_hr
# combo_nrem_psd_valid_windows combo_nrem_rmssd_mean_ms combo_nrem_sdnn_ms
# combo_nrem_sleep_hours combo_rem_delta_hr_evt_mean combo_rem_delta_hr_mean
# combo_rem_event_windows_total combo_rem_event_windows_used combo_rem_lf_hf
# combo_rem_lf_hf_fixed_n_windows_valid combo_rem_lf_n_segments_used
# combo_rem_pat_burden combo_rem_peak_minus_baseline_hr
# combo_rem_peak_to_trough_hr combo_rem_post_peak_minus_pre_mean_hr
# combo_rem_psd_valid_windows combo_rem_rmssd_mean_ms combo_rem_sdnn_ms
# combo_rem_sleep_hours combo_wake_sleep_delta_hr_evt_mean
# combo_wake_sleep_delta_hr_mean combo_wake_sleep_event_windows_total
# combo_wake_sleep_event_windows_used combo_wake_sleep_lf_hf
# combo_wake_sleep_lf_hf_fixed_n_windows_valid combo_wake_sleep_lf_n_segments_used
# combo_wake_sleep_pat_burden combo_wake_sleep_peak_minus_baseline_hr
# combo_wake_sleep_peak_to_trough_hr combo_wake_sleep_post_peak_minus_pre_mean_hr
# combo_wake_sleep_psd_valid_windows combo_wake_sleep_rmssd_mean_ms
# combo_wake_sleep_sdnn_ms combo_wake_sleep_sleep_hours
# hrv_tv_hf_nan_pct hrv_tv_lf_hf_nan_pct hrv_tv_lf_nan_pct
# hrv_tv_rmssd_ms_nan_pct hrv_tv_sdnn_ms_nan_pct hrv_tv_tv_window_sec_nan_pct
# """.split()



feature_cols = """
rmssd_mean_ms rmssd_median_ms sdnn_ms lf hf lf_hf lf_n_segments_used
lf_hf_fixed_median lf_hf_fixed_mean lf_hf_fixed_n_windows_valid
lf_hf_fixed_n_windows_total lf_hf_fixed_window_sec lf_hf_fixed_hop_sec
hr_pat_nan_pct hrv_rmssd_clean_nan_pct hrv_rmssd_raw_nan_pct aux_rows
desat_n desat_pct exclude_pat_n exclude_pat_pct evt_central_3_n
evt_central_3_pct evt_obstructive_3_n evt_obstructive_3_pct
evt_unclassified_3_n evt_unclassified_3_pct sleep_mask_enabled
sleep_policy sleep_included_n sleep_included_pct sleep_excluded_n
sleep_excluded_pct sleep_wake_n sleep_wake_pct sleep_light_n
sleep_light_pct sleep_deep_n sleep_deep_pct sleep_rem_n sleep_rem_pct
combo_all_sleep_event_windows_total combo_all_sleep_event_windows_used
combo_all_sleep_hr_mean combo_all_sleep_hr_median combo_all_sleep_hr_n_used
combo_all_sleep_hr_std combo_all_sleep_lf_hf
combo_all_sleep_lf_hf_fixed_n_windows_valid
combo_all_sleep_lf_n_segments_used
combo_all_sleep_mean_to_peak_response_mean combo_all_sleep_pat_burden
combo_all_sleep_psd_valid_windows combo_all_sleep_rmssd_mean_ms
combo_all_sleep_sdnn_ms combo_all_sleep_sleep_hours
combo_all_sleep_trough_to_peak_response_mean combo_deep_event_windows_total
combo_deep_event_windows_used combo_deep_hr_mean combo_deep_hr_median
combo_deep_hr_n_used combo_deep_hr_std combo_deep_lf_hf
combo_deep_lf_hf_fixed_n_windows_valid combo_deep_lf_n_segments_used
combo_deep_mean_to_peak_response_mean combo_deep_pat_burden
combo_deep_psd_valid_windows combo_deep_rmssd_mean_ms combo_deep_sdnn_ms
combo_deep_sleep_hours combo_deep_trough_to_peak_response_mean
combo_nrem_event_windows_total combo_nrem_event_windows_used
combo_nrem_hr_mean combo_nrem_hr_median combo_nrem_hr_n_used
combo_nrem_hr_std combo_nrem_lf_hf combo_nrem_lf_hf_fixed_n_windows_valid
combo_nrem_lf_n_segments_used combo_nrem_mean_to_peak_response_mean
combo_nrem_pat_burden combo_nrem_psd_valid_windows combo_nrem_rmssd_mean_ms
combo_nrem_sdnn_ms combo_nrem_sleep_hours
combo_nrem_trough_to_peak_response_mean combo_rem_event_windows_total
combo_rem_event_windows_used combo_rem_hr_mean combo_rem_hr_median
combo_rem_hr_n_used combo_rem_hr_std combo_rem_lf_hf
combo_rem_lf_hf_fixed_n_windows_valid combo_rem_lf_n_segments_used
combo_rem_mean_to_peak_response_mean combo_rem_pat_burden
combo_rem_psd_valid_windows combo_rem_rmssd_mean_ms combo_rem_sdnn_ms
combo_rem_sleep_hours combo_rem_trough_to_peak_response_mean
combo_wake_sleep_event_windows_total combo_wake_sleep_event_windows_used
combo_wake_sleep_hr_mean combo_wake_sleep_hr_median
combo_wake_sleep_hr_n_used combo_wake_sleep_hr_std combo_wake_sleep_lf_hf
combo_wake_sleep_lf_hf_fixed_n_windows_valid
combo_wake_sleep_lf_n_segments_used
combo_wake_sleep_mean_to_peak_response_mean combo_wake_sleep_pat_burden
combo_wake_sleep_psd_valid_windows combo_wake_sleep_rmssd_mean_ms
combo_wake_sleep_sdnn_ms combo_wake_sleep_sleep_hours
combo_wake_sleep_trough_to_peak_response_mean hrv_tv_hf_nan_pct
hrv_tv_lf_hf_nan_pct hrv_tv_lf_nan_pct hrv_tv_rmssd_ms_nan_pct
hrv_tv_sdnn_ms_nan_pct hrv_tv_tv_window_sec_nan_pct
""".split()

feature_cols = [
    col for col in feature_cols
    if not (
        col.endswith("_used") or
        col.endswith("_total") or
        col.endswith("_pct")
    )
]


# ----------------------------
# Remove unwanted suffixes
# ----------------------------
feature_cols = [
    col for col in feature_cols
    if not (
        col.endswith("_used") or
        col.endswith("_total") or
        col.endswith("_pct")
or
        col.endswith("_n")
or
        col.endswith("_valid")
    )
]


# Keep only columns that actually exist
feature_cols = [c for c in feature_cols if c in merged.columns]

# Convert candidate features to numeric where possible
for col in feature_cols:
    merged[col] = pd.to_numeric(merged[col], errors="coerce")

# Optional: remove columns that are entirely NaN
feature_cols = [c for c in feature_cols if merged[c].notna().any()]

# ----------------------------
# Helpers
# ----------------------------
def p_to_stars(p):
    if p < 1e-4:
        return "****"
    elif p < 1e-3:
        return "***"
    elif p < 1e-2:
        return "**"
    elif p < 5e-2:
        return "*"
    return "ns"

def add_significance_bar(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="black")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=9)

# ----------------------------
# Pairwise comparisons
# ----------------------------
pairwise_results = []

comparisons = list(itertools.combinations(range(len(group_labels)), 2))
# comparisons is [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

for feature in feature_cols:
    df = merged[["AHI_group", feature]].dropna().copy()
    df = df[df["AHI_group"].isin(group_labels)]

    if df.empty:
        continue

    group_data = []
    available_positions = []
    available_labels = []

    for i, g in enumerate(group_labels):
        vals = df.loc[df["AHI_group"] == g, feature].dropna().values
        if len(vals) > 0:
            group_data.append(vals)
            available_positions.append(i + 1)  # matplotlib boxplot positions start at 1
            available_labels.append(g)

    if len(group_data) < 2:
        continue

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(
        group_data,
        positions=available_positions,
        widths=0.6,
        patch_artist=False,
        showfliers=True
    )

    ax.set_xticks(range(1, len(group_labels) + 1))
    ax.set_xticklabels(group_labels)
    ax.set_xlabel("AHI Group")
    ax.set_ylabel(feature)
    ax.set_title(feature)

    # Pairwise tests on original four group positions
    valid_group_map = {
        g: df.loc[df["AHI_group"] == g, feature].dropna().values
        for g in group_labels
    }

    # Determine y-range for annotation
    y_values = df[feature].values
    y_min = np.nanmin(y_values)
    y_max = np.nanmax(y_values)
    y_range = y_max - y_min

    if not np.isfinite(y_range) or y_range == 0:
        y_range = max(abs(y_max), 1.0)

    step = y_range * 0.08
    bar_h = y_range * 0.02
    current_y = y_max + step

    plot_has_annotations = False

    for i, j in comparisons:
        g1 = group_labels[i]
        g2 = group_labels[j]
        x1 = i + 1
        x2 = j + 1

        vals1 = valid_group_map[g1]
        vals2 = valid_group_map[g2]

        # Need at least 2 groups with data
        if len(vals1) == 0 or len(vals2) == 0:
            continue

        try:
            stat, p = mannwhitneyu(vals1, vals2, alternative="two-sided")
        except ValueError:
            continue

        stars = p_to_stars(p)

        pairwise_results.append({
            "feature": feature,
            "group1": g1,
            "group2": g2,
            "n1": len(vals1),
            "n2": len(vals2),
            "p_value": p,
            "stars": stars
        })

        # Draw all comparisons, including ns
        add_significance_bar(ax, x1, x2, current_y, bar_h, stars)
        current_y += step
        plot_has_annotations = True

    if plot_has_annotations:
        ax.set_ylim(top=current_y + step)

    plt.tight_layout()
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in feature)
    fig.savefig(os.path.join(output_dir, f"{safe_name}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

# ----------------------------
# Save pairwise stats
# ----------------------------
pairwise_df = pd.DataFrame(pairwise_results)
pairwise_csv = os.path.join(output_dir, "pairwise_significance_results.csv")
pairwise_df.to_csv(pairwise_csv, index=False)

print(f"\nSaved plots to: {output_dir}")
print(f"Saved pairwise stats to: {pairwise_csv}")

if not pairwise_df.empty:
    print("\nTop 20 smallest p-values:")
    print(pairwise_df.sort_values("p_value").head(20).to_string(index=False))
else:
    print("\nNo pairwise results were generated.")