import os
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu, spearmanr

# Optional but strongly recommended for residual modeling + LOWESS
try:
    import statsmodels.api as sm
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False
    warnings.warn(
        "statsmodels is not available. Residual modeling and LOWESS plots will be skipped."
    )

# ----------------------------
# File paths
# ----------------------------
csv_path = "/Users/jindrich/Projects/PAT_022026_output_data/HR__20260413_115908__all_sleep_incluidng_wake__deltahrinclud/HR_HRV_EVENT_HR_summary__multi_sleep_summary__20260413_115908.csv"
xlsx_path = "/Users/jindrich/Projects/mayo_sleep_pat/SmallDataset21Oct25/Data/20251020_parsed_last_deindentified.xlsx"

output_dir = "/Users/jindrich/Projects/pat_toolbox/experiments/ahi_feature_analysis_delta_hr_boosted"
os.makedirs(output_dir, exist_ok=True)

plots_dir = os.path.join(output_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# ----------------------------
# Settings
# ----------------------------
GROUP_LABELS = ["0-5", "5-15", "15-30", ">30"]
AHI_BINS = [0, 5, 15, 30, np.inf]

# Stages to analyze
STAGES = ["all_sleep", "wake_sleep", "nrem", "deep", "rem"]

# Main event-response metric
PRIMARY_RESPONSE_COL = "combo_all_sleep_trough_to_peak_response_mean"

# Core confounders for residual view
CORE_CONFOUNDERS = [
    "combo_all_sleep_event_windows_used",
    "desat_pct",
    "hr_pat_nan_pct",
]

# Additional useful severity / quality variables
SEVERITY_COLS = [
    "AHI",
    "desat_n",
    "desat_pct",
    "evt_obstructive_3_n",
    "evt_obstructive_3_pct",
    "combo_all_sleep_event_windows_used",
    "combo_all_sleep_event_windows_total",
]

QUALITY_COLS = [
    "hr_pat_nan_pct",
    "hrv_rmssd_clean_nan_pct",
    "hrv_rmssd_raw_nan_pct",
]

# ----------------------------
# Helpers
# ----------------------------
def safe_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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

def sanitize_filename(name):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(name))

def add_significance_bar(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="black")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=9)

def nonempty_numeric(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s.values

def ensure_cols_exist(df, cols):
    return [c for c in cols if c in df.columns]

def summarize_series(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "n": int(len(s)),
        "mean": float(np.mean(s)),
        "median": float(np.median(s)),
        "std": float(np.std(s, ddof=1)) if len(s) > 1 else np.nan,
        "min": float(np.min(s)),
        "max": float(np.max(s)),
    }

def robust_log1p_divide(numerator, denominator):
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    out = num / np.log1p(den)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out

def plain_divide(numerator, denominator):
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    out = num / den.replace(0, np.nan)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out

def make_scatter_lowess_plot(
    df,
    x_col,
    y_col,
    title,
    out_path,
    color_col=None,
    frac=0.35
):
    plot_df = df[[x_col, y_col] + ([color_col] if color_col else [])].dropna().copy()
    if len(plot_df) < 8:
        return False

    plt.figure(figsize=(8, 6))

    if color_col:
        sc = plt.scatter(
            plot_df[x_col],
            plot_df[y_col],
            c=plot_df[color_col],
            alpha=0.65,
            s=28
        )
        plt.colorbar(sc, label=color_col)
    else:
        plt.scatter(
            plot_df[x_col],
            plot_df[y_col],
            alpha=0.6,
            s=28
        )

    if HAS_STATSMODELS:
        try:
            smoothed = lowess(
                endog=plot_df[y_col].values,
                exog=plot_df[x_col].values,
                frac=frac,
                return_sorted=True
            )
            plt.plot(smoothed[:, 0], smoothed[:, 1], linewidth=3)
        except Exception as e:
            print(f"LOWESS failed for {x_col} vs {y_col}: {e}")

    if "residual" in y_col.lower():
        plt.axhline(0, linestyle="--", linewidth=1)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    return True

def boxplot_with_pairwise_tests(df, feature, group_col, group_labels, out_path):
    sub = df[[group_col, feature]].dropna().copy()
    sub = sub[sub[group_col].isin(group_labels)]

    if sub.empty:
        return []

    group_data = []
    available_positions = []

    for i, g in enumerate(group_labels):
        vals = sub.loc[sub[group_col] == g, feature].dropna().values
        if len(vals) > 0:
            group_data.append(vals)
            available_positions.append(i + 1)

    if len(group_data) < 2:
        return []

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(
        group_data,
        positions=available_positions,
        widths=0.6,
        patch_artist=False,
        showfliers=True
    )

    ax.set_xticks(range(1, len(group_labels) + 1))
    ax.set_xticklabels(group_labels)
    ax.set_xlabel(group_col)
    ax.set_ylabel(feature)
    ax.set_title(feature)

    valid_group_map = {
        g: sub.loc[sub[group_col] == g, feature].dropna().values
        for g in group_labels
    }

    comparisons = list(itertools.combinations(range(len(group_labels)), 2))
    results = []

    y_values = sub[feature].values
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

        if len(vals1) == 0 or len(vals2) == 0:
            continue

        try:
            _, p = mannwhitneyu(vals1, vals2, alternative="two-sided")
        except ValueError:
            continue

        stars = p_to_stars(p)
        results.append({
            "feature": feature,
            "group1": g1,
            "group2": g2,
            "n1": len(vals1),
            "n2": len(vals2),
            "p_value": p,
            "stars": stars
        })

        add_significance_bar(ax, x1, x2, current_y, bar_h, stars)
        current_y += step
        plot_has_annotations = True

    if plot_has_annotations:
        ax.set_ylim(top=current_y + step)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    return results

def fit_residual_model(df, response_col, confounders):
    if not HAS_STATSMODELS:
        return None, None

    use_cols = [response_col] + confounders
    use_cols = [c for c in use_cols if c in df.columns]

    if response_col not in use_cols:
        return None, None

    model_df = df[use_cols].dropna().copy()
    if len(model_df) < 15:
        return None, None

    X = model_df[confounders].copy()
    X = sm.add_constant(X, has_constant="add")
    y = model_df[response_col].copy()

    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        print(f"OLS failed for {response_col}: {e}")
        return None, None

    model_df[f"{response_col}_predicted"] = model.predict(X)
    model_df[f"{response_col}_residual"] = y - model_df[f"{response_col}_predicted"]
    model_df["row_index"] = model_df.index

    return model, model_df

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
if "AHI" not in merged.columns:
    raise ValueError("Column 'AHI' not found after merge.")

merged["AHI"] = pd.to_numeric(merged["AHI"], errors="coerce")
merged["AHI_group"] = pd.cut(
    merged["AHI"],
    bins=AHI_BINS,
    labels=GROUP_LABELS,
    right=False
)

# ----------------------------
# Numeric conversion for key columns
# ----------------------------
candidate_numeric_cols = list(set(
    SEVERITY_COLS +
    QUALITY_COLS +
    [PRIMARY_RESPONSE_COL] +
    CORE_CONFOUNDERS +
    [f"combo_{stage}_trough_to_peak_response_mean" for stage in STAGES] +
    [f"combo_{stage}_mean_to_peak_response_mean" for stage in STAGES] +
    [f"combo_{stage}_event_windows_used" for stage in STAGES] +
    [f"combo_{stage}_event_windows_total" for stage in STAGES] +
    [f"combo_{stage}_sleep_hours" for stage in STAGES] +
    [f"combo_{stage}_hr_mean" for stage in STAGES] +
    [f"combo_{stage}_rmssd_mean_ms" for stage in STAGES] +
    [f"combo_{stage}_sdnn_ms" for stage in STAGES] +
    [f"combo_{stage}_lf_hf" for stage in STAGES]
))
safe_numeric(merged, ensure_cols_exist(merged, candidate_numeric_cols))

# ----------------------------
# Derived delta-HR features
# ----------------------------
for stage in STAGES:
    trough_col = f"combo_{stage}_trough_to_peak_response_mean"
    mean_col = f"combo_{stage}_mean_to_peak_response_mean"
    used_col = f"combo_{stage}_event_windows_used"

    if trough_col in merged.columns and mean_col in merged.columns:
        merged[f"combo_{stage}_dip_depth_mean"] = (
            pd.to_numeric(merged[trough_col], errors="coerce") -
            pd.to_numeric(merged[mean_col], errors="coerce")
        )

    if trough_col in merged.columns and used_col in merged.columns:
        merged[f"combo_{stage}_response_per_event"] = plain_divide(
            merged[trough_col],
            merged[used_col]
        )
        merged[f"combo_{stage}_response_per_log_event"] = robust_log1p_divide(
            merged[trough_col],
            merged[used_col]
        )

# ----------------------------
# Focused delta-HR feature list
# ----------------------------
delta_hr_features = [
    "combo_all_sleep_trough_to_peak_response_mean",
    "combo_all_sleep_mean_to_peak_response_mean",
    "combo_all_sleep_dip_depth_mean",
    "combo_all_sleep_response_per_event",
    "combo_all_sleep_response_per_log_event",

    "combo_wake_sleep_trough_to_peak_response_mean",
    "combo_wake_sleep_mean_to_peak_response_mean",
    "combo_wake_sleep_dip_depth_mean",
    "combo_wake_sleep_response_per_event",
    "combo_wake_sleep_response_per_log_event",

    "combo_nrem_trough_to_peak_response_mean",
    "combo_nrem_mean_to_peak_response_mean",
    "combo_nrem_dip_depth_mean",
    "combo_nrem_response_per_event",
    "combo_nrem_response_per_log_event",

    "combo_deep_trough_to_peak_response_mean",
    "combo_deep_mean_to_peak_response_mean",
    "combo_deep_dip_depth_mean",
    "combo_deep_response_per_event",
    "combo_deep_response_per_log_event",

    "combo_rem_trough_to_peak_response_mean",
    "combo_rem_mean_to_peak_response_mean",
    "combo_rem_dip_depth_mean",
    "combo_rem_response_per_event",
    "combo_rem_response_per_log_event",
]

delta_hr_features = [c for c in delta_hr_features if c in merged.columns]

# ----------------------------
# Summary tables
# ----------------------------
summary_rows = []
for col in delta_hr_features + ensure_cols_exist(merged, SEVERITY_COLS + QUALITY_COLS):
    stats = summarize_series(merged[col])
    stats["feature"] = col
    summary_rows.append(stats)

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df[
    ["feature", "n", "mean", "median", "std", "min", "max"]
].sort_values("feature")
summary_df.to_csv(os.path.join(output_dir, "feature_summary.csv"), index=False)

# ----------------------------
# Stage-level medians table
# ----------------------------
stage_table_rows = []
for stage in STAGES:
    row = {"stage": stage}
    for metric in [
        f"combo_{stage}_trough_to_peak_response_mean",
        f"combo_{stage}_mean_to_peak_response_mean",
        f"combo_{stage}_dip_depth_mean",
        f"combo_{stage}_response_per_log_event",
        f"combo_{stage}_event_windows_used",
        f"combo_{stage}_sleep_hours",
        f"combo_{stage}_hr_mean",
        f"combo_{stage}_rmssd_mean_ms",
        f"combo_{stage}_sdnn_ms",
        f"combo_{stage}_lf_hf",
    ]:
        if metric in merged.columns:
            row[f"{metric}_median"] = pd.to_numeric(
                merged[metric], errors="coerce"
            ).median()
            row[f"{metric}_n_nonnull"] = pd.to_numeric(
                merged[metric], errors="coerce"
            ).notna().sum()
    stage_table_rows.append(row)

stage_summary_df = pd.DataFrame(stage_table_rows)
stage_summary_df.to_csv(os.path.join(output_dir, "stage_summary_medians.csv"), index=False)

# ----------------------------
# Correlations: delta-HR vs severity / quality
# ----------------------------
corr_rows = []
x_cols = [c for c in (SEVERITY_COLS + QUALITY_COLS) if c in merged.columns]

for y_col in delta_hr_features:
    for x_col in x_cols:
        sub = merged[[x_col, y_col]].dropna()
        if len(sub) < 8:
            continue
        try:
            rho, p = spearmanr(sub[x_col], sub[y_col], nan_policy="omit")
        except Exception:
            continue
        corr_rows.append({
            "y_feature": y_col,
            "x_feature": x_col,
            "n": len(sub),
            "spearman_rho": rho,
            "p_value": p
        })

corr_df = pd.DataFrame(corr_rows)
if not corr_df.empty:
    corr_df = corr_df.sort_values(["y_feature", "p_value", "spearman_rho"], ascending=[True, True, False])
corr_df.to_csv(os.path.join(output_dir, "delta_hr_correlations.csv"), index=False)

# ----------------------------
# Pairwise AHI-group boxplots
# ----------------------------
pairwise_results = []
for feature in delta_hr_features:
    out_path = os.path.join(plots_dir, f"boxplot_{sanitize_filename(feature)}.png")
    results = boxplot_with_pairwise_tests(
        df=merged,
        feature=feature,
        group_col="AHI_group",
        group_labels=GROUP_LABELS,
        out_path=out_path
    )
    pairwise_results.extend(results)

pairwise_df = pd.DataFrame(pairwise_results)
pairwise_df.to_csv(
    os.path.join(output_dir, "pairwise_significance_results_delta_hr.csv"),
    index=False
)

# ----------------------------
# Quartile analysis for desaturation burden
# ----------------------------
quartile_col = "desat_pct"
quartile_features = [
    "combo_all_sleep_trough_to_peak_response_mean",
    "combo_all_sleep_mean_to_peak_response_mean",
    "combo_all_sleep_dip_depth_mean",
    "combo_all_sleep_response_per_log_event",
    "hr_pat_nan_pct",
    "hrv_rmssd_clean_nan_pct",
    "combo_all_sleep_event_windows_used"
]
quartile_features = [c for c in quartile_features if c in merged.columns]

quartile_df = merged[[quartile_col] + quartile_features].dropna(subset=[quartile_col]).copy()
if quartile_col in quartile_df.columns and quartile_df[quartile_col].notna().sum() >= 8:
    quartile_df["desat_pct_quartile"] = pd.qcut(
        quartile_df[quartile_col],
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop"
    )
    quartile_summary = quartile_df.groupby("desat_pct_quartile")[quartile_features].median(numeric_only=True)
    quartile_summary.to_csv(os.path.join(output_dir, "desat_pct_quartile_medians.csv"))

# ----------------------------
# Residual models
# ----------------------------
model_summaries = []
residual_join_cols = ["AHI", "AHI_group", "desat_pct", "hr_pat_nan_pct"]

for response_col in [
    "combo_all_sleep_trough_to_peak_response_mean",
    "combo_wake_sleep_trough_to_peak_response_mean",
    "combo_nrem_trough_to_peak_response_mean",
    "combo_deep_trough_to_peak_response_mean",
    "combo_rem_trough_to_peak_response_mean",
]:
    if response_col not in merged.columns:
        continue

    confounders = [
        c for c in [
            response_col.replace("trough_to_peak_response_mean", "event_windows_used"),
            "desat_pct",
            "hr_pat_nan_pct",
        ]
        if c in merged.columns
    ]

    if len(confounders) < 2:
        continue

    model, model_df = fit_residual_model(merged, response_col, confounders)
    if model is None or model_df is None:
        continue

    # Save model coefficients
    coef_df = pd.DataFrame({
        "term": model.params.index,
        "coef": model.params.values,
        "p_value": model.pvalues.values
    })
    coef_df.to_csv(
        os.path.join(output_dir, f"model_coefficients_{sanitize_filename(response_col)}.csv"),
        index=False
    )

    # Save per-row residuals
    residual_export = model_df.copy()
    residual_export = residual_export.rename(columns={
        f"{response_col}_predicted": "predicted",
        f"{response_col}_residual": "residual"
    })
    residual_export.to_csv(
        os.path.join(output_dir, f"residuals_{sanitize_filename(response_col)}.csv"),
        index=False
    )

    model_summaries.append({
        "response_col": response_col,
        "n": int(model.nobs),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "confounders": ", ".join(confounders)
    })

    # Merge residuals back by index for plotting against AHI
    merged.loc[model_df["row_index"], f"{response_col}_predicted"] = model_df[f"{response_col}_predicted"].values
    merged.loc[model_df["row_index"], f"{response_col}_residual"] = model_df[f"{response_col}_residual"].values

model_summary_df = pd.DataFrame(model_summaries)
model_summary_df.to_csv(os.path.join(output_dir, "residual_model_summary.csv"), index=False)

# ----------------------------
# Residual boxplots by AHI group
# ----------------------------
residual_pairwise_results = []
residual_cols = [c for c in merged.columns if c.endswith("_residual")]

for feature in residual_cols:
    out_path = os.path.join(plots_dir, f"boxplot_{sanitize_filename(feature)}.png")
    results = boxplot_with_pairwise_tests(
        df=merged,
        feature=feature,
        group_col="AHI_group",
        group_labels=GROUP_LABELS,
        out_path=out_path
    )
    residual_pairwise_results.extend(results)

residual_pairwise_df = pd.DataFrame(residual_pairwise_results)
residual_pairwise_df.to_csv(
    os.path.join(output_dir, "pairwise_significance_results_residuals.csv"),
    index=False
)

# ----------------------------
# Continuous relationship plots
# ----------------------------
relationship_specs = [
    ("AHI", "combo_all_sleep_trough_to_peak_response_mean", None),
    ("desat_pct", "combo_all_sleep_trough_to_peak_response_mean", None),
    ("combo_all_sleep_event_windows_used", "combo_all_sleep_trough_to_peak_response_mean", "hr_pat_nan_pct"),
    ("hr_pat_nan_pct", "combo_all_sleep_trough_to_peak_response_mean", None),
    ("AHI", "combo_all_sleep_response_per_log_event", None),
    ("desat_pct", "combo_all_sleep_response_per_log_event", None),
]

for x_col, y_col, color_col in relationship_specs:
    if x_col in merged.columns and y_col in merged.columns:
        out_path = os.path.join(
            plots_dir,
            f"scatter_lowess_{sanitize_filename(y_col)}__vs__{sanitize_filename(x_col)}.png"
        )
        make_scatter_lowess_plot(
            df=merged,
            x_col=x_col,
            y_col=y_col,
            title=f"{y_col} vs {x_col}",
            out_path=out_path,
            color_col=color_col
        )

# Residual relationship plots
for response_col in [
    "combo_all_sleep_trough_to_peak_response_mean",
    "combo_wake_sleep_trough_to_peak_response_mean",
    "combo_nrem_trough_to_peak_response_mean",
    "combo_deep_trough_to_peak_response_mean",
    "combo_rem_trough_to_peak_response_mean",
]:
    residual_col = f"{response_col}_residual"
    if residual_col not in merged.columns:
        continue

    for x_col in ["AHI", "desat_pct", "combo_all_sleep_event_windows_used", "hr_pat_nan_pct"]:
        if x_col not in merged.columns:
            continue

        out_path = os.path.join(
            plots_dir,
            f"scatter_lowess_{sanitize_filename(residual_col)}__vs__{sanitize_filename(x_col)}.png"
        )
        make_scatter_lowess_plot(
            df=merged,
            x_col=x_col,
            y_col=residual_col,
            title=f"{residual_col} vs {x_col}",
            out_path=out_path,
            color_col="hr_pat_nan_pct" if "hr_pat_nan_pct" in merged.columns and x_col != "hr_pat_nan_pct" else None
        )

# ----------------------------
# AHI-group medians for selected delta-HR features
# ----------------------------
group_median_features = [
    "combo_all_sleep_trough_to_peak_response_mean",
    "combo_all_sleep_mean_to_peak_response_mean",
    "combo_all_sleep_dip_depth_mean",
    "combo_all_sleep_response_per_log_event",
    "combo_wake_sleep_trough_to_peak_response_mean",
    "combo_nrem_trough_to_peak_response_mean",
    "combo_deep_trough_to_peak_response_mean",
    "combo_rem_trough_to_peak_response_mean",
    "hr_pat_nan_pct",
    "hrv_rmssd_clean_nan_pct",
    "combo_all_sleep_event_windows_used",
]
group_median_features = [c for c in group_median_features if c in merged.columns]

ahi_group_summary = merged.groupby("AHI_group")[group_median_features].median(numeric_only=True)
ahi_group_summary.to_csv(os.path.join(output_dir, "ahi_group_medians_delta_hr.csv"))

# ----------------------------
# Top responders / blunted responders from residual view
# ----------------------------
top_tables = []
residual_target = f"{PRIMARY_RESPONSE_COL}_residual"
id_cols = [c for c in ["parsed_patient_id", "patient.ID", "edf_file", "AHI", "AHI_group"] if c in merged.columns]

if residual_target in merged.columns:
    tmp = merged[id_cols + [residual_target, PRIMARY_RESPONSE_COL] + ensure_cols_exist(merged, CORE_CONFOUNDERS)].copy()
    tmp = tmp.dropna(subset=[residual_target]).sort_values(residual_target, ascending=False)

    top_high = tmp.head(20).copy()
    top_low = tmp.tail(20).copy()

    top_high.to_csv(os.path.join(output_dir, "top_20_high_responders_by_residual.csv"), index=False)
    top_low.to_csv(os.path.join(output_dir, "top_20_blunted_responders_by_residual.csv"), index=False)

# ----------------------------
# Console summary
# ----------------------------
print("\nSaved outputs to:", output_dir)
print("Saved plots to:", plots_dir)

if not pairwise_df.empty:
    print("\nTop 15 smallest p-values for raw delta-HR features:")
    print(pairwise_df.sort_values("p_value").head(15).to_string(index=False))
else:
    print("\nNo raw pairwise delta-HR results generated.")

if not residual_pairwise_df.empty:
    print("\nTop 15 smallest p-values for residual delta-HR features:")
    print(residual_pairwise_df.sort_values("p_value").head(15).to_string(index=False))
else:
    print("\nNo residual pairwise results generated.")

if not corr_df.empty:
    print("\nTop 20 strongest delta-HR correlations by smallest p-value:")
    print(corr_df.sort_values("p_value").head(20).to_string(index=False))
else:
    print("\nNo delta-HR correlation results generated.")

if not model_summary_df.empty:
    print("\nResidual models:")
    print(model_summary_df.to_string(index=False))
else:
    print("\nNo residual models were generated. Check statsmodels availability and data completeness.")