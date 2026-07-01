from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


INPUT_CSV = Path("/Users/jindrich/Projects/pat_toolbox/experiments/merged_tables_output/merged_table_23042026.csv")
OUTPUT_DIR = Path("/Users/jindrich/Projects/pat_toolbox/experiments/merged_tables_output/exploration_23042026")
PLOTS_DIR = OUTPUT_DIR / "plots"

MIN_N = 8
TOP_N = 80
TOP_PLOTS = 16

TARGET_COLS = [
    "AHI",
    "AHI_3_Percent",
    "AHI_4_Percent",
    "RDI",
    "ODI",
    "ODI_3_Percent",
    "ODI_4_Percent",
    "NREM_AHI",
    "REM_AHI",
    "NREM_ODI",
    "REM_ODI",
    "TotalSleepTime",
    "TotalREMTime",
    "TotalDeepSleepTime",
    "TotalLightSleepTime",
    "TotalWakeTime",
    "MeanPRSleep",
    "MinPRSleep",
    "MaxPRSleep",
    "MeanSatValue",
    "MinSatValue",
    "MeanNadirDesaturations",
    "SatBelow90",
    "SatBelow88",
    "AreaUnder90Overall",
    "AreaUnder90PerHour",
    "DesatBurdenOverall",
    "DesatBurdenPerHour",
    "Weight",
    "Height",
    "NeckSize",
    "EpworthScore",
]

FEATURE_PREFIXES = (
    "selected_",
    "combo_",
    "nrem_first_half_",
    "nrem_second_half_",
    "prv_tv_",
    "desat_",
    "evt_",
    "sleep_",
    "aux_",
)

EXCLUDE_FEATURE_PATTERNS = (
    "@units",
    "hhmm",
    "sourcefile",
    "Warnings",
    "patient.",
    "parsed_patient_id",
    "edf_file",
)


def _sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")[:180]


def _to_numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    data = {col: pd.to_numeric(df[col], errors="coerce") for col in cols}
    return pd.DataFrame(data, index=df.index).replace([np.inf, -np.inf], np.nan)


def _is_obvious_pair(feature: str, target: str) -> bool:
    feature_l = feature.lower()
    target_l = target.lower()
    if target in {"MeanPRSleep", "MinPRSleep", "MaxPRSleep"} and "hr_" in feature_l:
        return True
    if target in {"TotalSleepTime", "TotalWakeTime", "TotalREMTime", "TotalDeepSleepTime", "TotalLightSleepTime"}:
        if "sleep_hours" in feature_l or feature_l.startswith("sleep_") or "sleep_time" in feature_l:
            return True
    if target_l.startswith("odi") and (feature_l.startswith("desat_") or "desat" in feature_l):
        return True
    if target_l.startswith("ahi") and feature_l.startswith("evt_"):
        return True
    return False


def _interesting_subset(correlations: pd.DataFrame) -> pd.DataFrame:
    if correlations.empty:
        return correlations.copy()
    keep = [
        not _is_obvious_pair(str(row.feature), str(row.target))
        for row in correlations.itertuples(index=False)
    ]
    out = correlations.loc[keep].copy()
    return out.sort_values(["abs_spearman_rho", "n"], ascending=[False, False]).reset_index(drop=True)


def _physiology_subset(correlations: pd.DataFrame) -> pd.DataFrame:
    if correlations.empty:
        return correlations.copy()
    bad_tokens = (
        "excluded",
        "valid_",
        "valid_pct",
        "valid_min",
        "clean_kept",
        "selected_policy_min",
        "total_min",
        "n_windows",
        "windows_total",
        "windows_valid",
        "sleep_hours",
        "sleep_wake",
        "sleep_light",
        "sleep_deep",
        "sleep_rem",
        "desat_",
        "evt_",
        "aux_rows",
        "mask_",
        "nan_pct",
    )
    keep = []
    for row in correlations.itertuples(index=False):
        feature = str(row.feature)
        feature_l = feature.lower()
        if _is_obvious_pair(feature, str(row.target)):
            keep.append(False)
        elif any(token in feature_l for token in bad_tokens):
            keep.append(False)
        else:
            keep.append(True)
    out = correlations.loc[keep].copy()
    return out.sort_values(["abs_spearman_rho", "n"], ascending=[False, False]).reset_index(drop=True)


def _severity_focused_subset(correlations: pd.DataFrame) -> pd.DataFrame:
    physiology = _physiology_subset(correlations)
    if physiology.empty:
        return physiology
    excluded_targets = {"MeanPRSleep", "MinPRSleep", "MaxPRSleep", "TotalSleepTime", "TotalWakeTime", "TotalREMTime", "TotalDeepSleepTime", "TotalLightSleepTime"}
    out = physiology[~physiology["target"].isin(excluded_targets)].copy()
    return out.sort_values(["abs_spearman_rho", "n"], ascending=[False, False]).reset_index(drop=True)


def _bh_fdr(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    ok = np.isfinite(p)
    if not np.any(ok):
        return pd.Series(q, index=p_values.index)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = float(ranked.size)
    adjusted = ranked * m / np.arange(1, ranked.size + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.clip(adjusted, 0.0, 1.0)
    return pd.Series(q, index=p_values.index)


def _pick_feature_columns(df: pd.DataFrame, target_cols: set[str]) -> list[str]:
    cols = []
    for col in df.columns:
        if col in target_cols:
            continue
        if any(pattern in col for pattern in EXCLUDE_FEATURE_PATTERNS):
            continue
        if col.startswith(FEATURE_PREFIXES):
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().sum() >= MIN_N and vals.nunique(dropna=True) > 2:
                cols.append(col)
    return cols


def _compute_correlations(df_num: pd.DataFrame, feature_cols: list[str], target_cols: list[str]) -> pd.DataFrame:
    rows = []
    for feature in feature_cols:
        x = df_num[feature]
        for target in target_cols:
            y = df_num[target]
            m = x.notna() & y.notna()
            n = int(m.sum())
            if n < MIN_N:
                continue
            if x[m].nunique() < 3 or y[m].nunique() < 3:
                continue
            rho, p_value = spearmanr(x[m], y[m], nan_policy="omit")
            if not np.isfinite(rho):
                continue
            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "n": n,
                    "spearman_rho": float(rho),
                    "abs_spearman_rho": float(abs(rho)),
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                    "feature_median": float(np.nanmedian(x[m])),
                    "target_median": float(np.nanmedian(y[m])),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value_bh"] = _bh_fdr(out["p_value"])
    return out.sort_values(["abs_spearman_rho", "n"], ascending=[False, False]).reset_index(drop=True)


def _plot_pair(df_num: pd.DataFrame, row: pd.Series, out_path: Path) -> None:
    feature = str(row["feature"])
    target = str(row["target"])
    plot_df = df_num[[feature, target]].dropna()
    if len(plot_df) < MIN_N:
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.scatter(plot_df[feature], plot_df[target], alpha=0.7, s=32, color="#2a9d8f", edgecolor="none")
    if plot_df[feature].nunique() > 2:
        z = np.polyfit(plot_df[feature], plot_df[target], deg=1)
        xx = np.linspace(float(plot_df[feature].min()), float(plot_df[feature].max()), 100)
        ax.plot(xx, z[0] * xx + z[1], color="#264653", linewidth=2.0, alpha=0.85)
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    ax.set_title(
        f"{target} vs {feature}\n"
        f"Spearman rho={row['spearman_rho']:.2f}, p={row['p_value']:.3g}, q={row['q_value_bh']:.3g}, n={int(row['n'])}",
        fontsize=10,
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_summary(df: pd.DataFrame, target_cols: list[str], feature_cols: list[str], correlations: pd.DataFrame) -> None:
    lines = [
        "# Merged Table Exploration 23042026",
        "",
        f"Input: `{INPUT_CSV}`",
        f"Rows: {df.shape[0]}",
        f"Columns: {df.shape[1]}",
        f"Numeric target columns found: {len(target_cols)}",
        f"Candidate feature columns found: {len(feature_cols)}",
        f"Minimum pairwise N: {MIN_N}",
        "",
    ]
    if correlations.empty:
        lines.append("No correlations passed the minimum data requirements.")
    else:
        sig = correlations[correlations["q_value_bh"] < 0.10]
        lines.extend(
            [
                f"Correlation pairs tested: {len(correlations)}",
                f"Pairs with BH q < 0.10: {len(sig)}",
                "",
                "## Top Correlations",
                "",
            ]
        )
        for _, row in correlations.head(20).iterrows():
            lines.append(
                f"- `{row['feature']}` vs `{row['target']}`: "
                f"rho={row['spearman_rho']:.3f}, p={row['p_value']:.3g}, "
                f"q={row['q_value_bh']:.3g}, n={int(row['n'])}"
            )
    (OUTPUT_DIR / "README_exploration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for old_plot in PLOTS_DIR.glob("*.png"):
        old_plot.unlink()

    df = pd.read_csv(INPUT_CSV)
    df.columns = df.columns.str.strip()

    target_cols = [col for col in TARGET_COLS if col in df.columns]
    feature_cols = _pick_feature_columns(df, set(target_cols))
    numeric_cols = sorted(set(target_cols + feature_cols))
    df_num = _to_numeric_frame(df, numeric_cols)

    correlations = _compute_correlations(df_num, feature_cols, target_cols)
    interesting = _interesting_subset(correlations)
    physiology = _physiology_subset(correlations)
    severity = _severity_focused_subset(correlations)
    correlations.to_csv(OUTPUT_DIR / "feature_target_spearman_correlations.csv", index=False)
    correlations.head(TOP_N).to_csv(OUTPUT_DIR / "top_feature_target_correlations.csv", index=False)
    interesting.head(TOP_N).to_csv(OUTPUT_DIR / "top_interesting_feature_target_correlations.csv", index=False)
    physiology.head(TOP_N).to_csv(OUTPUT_DIR / "top_physiology_feature_target_correlations.csv", index=False)
    severity.head(TOP_N).to_csv(OUTPUT_DIR / "top_severity_focused_feature_target_correlations.csv", index=False)

    target_summary = df_num[target_cols].describe().T if target_cols else pd.DataFrame()
    feature_summary = df_num[feature_cols].describe().T if feature_cols else pd.DataFrame()
    target_summary.to_csv(OUTPUT_DIR / "target_numeric_summary.csv")
    feature_summary.to_csv(OUTPUT_DIR / "feature_numeric_summary.csv")

    plot_source = severity if not severity.empty else interesting
    if not plot_source.empty:
        plotted_pairs: set[tuple[str, str]] = set()
        for _, row in plot_source.head(TOP_PLOTS * 3).iterrows():
            key = (str(row["feature"]), str(row["target"]))
            if key in plotted_pairs:
                continue
            plotted_pairs.add(key)
            out_name = _sanitize_filename(f"{len(plotted_pairs):02d}_{row['target']}__vs__{row['feature']}.png")
            _plot_pair(df_num, row, PLOTS_DIR / out_name)
            if len(plotted_pairs) >= TOP_PLOTS:
                break

    _write_summary(df, target_cols, feature_cols, interesting)
    print(f"Input rows/cols: {df.shape}")
    print(f"Targets found: {len(target_cols)}")
    print(f"Candidate features found: {len(feature_cols)}")
    print(f"Correlation pairs: {len(correlations)}")
    print(f"Interesting pairs after obvious-pair filtering: {len(interesting)}")
    print(f"Physiology-focused pairs after mask/count filtering: {len(physiology)}")
    print(f"Severity-focused physiology pairs: {len(severity)}")
    print(f"Outputs written to: {OUTPUT_DIR}")
    if not severity.empty:
        print("Top severity-focused physiology correlations:")
        cols = ["feature", "target", "n", "spearman_rho", "p_value", "q_value_bh"]
        print(severity[cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
