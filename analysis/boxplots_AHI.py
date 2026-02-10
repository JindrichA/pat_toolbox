#!/usr/bin/env python3
"""
Boxplots by AHI category with statistical testing and sample size (N).
One PDF, FIRST PAGE is a summary of all visualized data, then one page per metric.

Summary page includes:
- Rows in file, usable rows (AHI present)
- AHI distribution (min/median/mean/max)
- N per AHI category (based on AHI present)
- For each metric: overall non-missing N, N per category, and Kruskal–Wallis p-value
- Notes on tests used (KW + Dunn Bonferroni; stars on plots)
"""

from pathlib import Path
import itertools
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import kruskal
import scikit_posthocs as sp


# =========================
# Paths & configuration
# =========================

CSV_PATH = Path("merged_output_22012026.csv")
OUT_PDF = CSV_PATH.parent / "boxplots_AHI_with_stats_and_N.pdf"

FEATURES = [
    "rmssd_mean_ms",
    "rmssd_median_ms",
    "sdnn_ms",
    "lf",
    "hf",
    "lf_hf",
]

AHI_ORDER = ["AHI < 5", "AHI 5–<15", "AHI 15–<30", "AHI ≥ 30"]


# =========================
# Helper functions
# =========================

def ahi_category(ahi):
    if pd.isna(ahi):
        return pd.NA
    if ahi < 5:
        return "AHI < 5"
    elif ahi < 15:
        return "AHI 5–<15"
    elif ahi < 30:
        return "AHI 15–<30"
    else:
        return "AHI ≥ 30"


def read_table_safely(path: Path) -> pd.DataFrame:
    for kwargs in ({"sep": "\t"}, {"sep": ","}, {"sep": None, "engine": "python"}):
        try:
            df = pd.read_csv(path, **kwargs)
            if "AHI" in df.columns:
                return df
        except Exception:
            pass
    raise RuntimeError("Could not read CSV with any separator.")


def significance_star(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def add_pairwise_annotations(ax, data, pvals, y_offset_frac=0.05):
    # Guard against degenerate data
    nonempty = [x for x in data if len(x) > 0]
    if len(nonempty) == 0:
        return

    ymax = max(x.max() for x in nonempty)
    ymin = min(x.min() for x in nonempty)
    height = ymax - ymin
    if not np.isfinite(height) or height == 0:
        height = max(1.0, abs(ymax) if np.isfinite(ymax) else 1.0)

    line_y = ymax + height * 0.05
    step = height * y_offset_frac

    for (i, j) in itertools.combinations(range(len(AHI_ORDER)), 2):
        p = float(pvals.iloc[i, j])
        if p < 0.05:
            star = significance_star(p)
            x1, x2 = i + 1, j + 1

            ax.plot(
                [x1, x1, x2, x2],
                [line_y, line_y + step, line_y + step, line_y],
                linewidth=1
            )
            ax.text(
                (x1 + x2) / 2,
                line_y + step,
                star,
                ha="center",
                va="bottom",
                fontsize=10
            )
            line_y += step * 1.5


def add_group_counts(ax, counts):
    labels = [f"{cat}\nN={counts.get(cat, 0)}" for cat in AHI_ORDER]
    ax.set_xticks(range(1, len(AHI_ORDER) + 1))
    ax.set_xticklabels(labels)


def safe_kruskal(groups):
    """
    Kruskal fails if fewer than 2 groups have data, or if all values identical.
    Returns (stat, p) or (nan, nan).
    """
    nonempty = [g for g in groups if len(g) > 0]
    if len(nonempty) < 2:
        return np.nan, np.nan

    try:
        stat, p = kruskal(*nonempty)
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


def safe_dunn(groups):
    """
    Dunn test needs at least 2 non-empty groups.
    Returns a square DataFrame aligned to AHI_ORDER (nan where unavailable).
    """
    out = pd.DataFrame(np.nan, index=range(len(AHI_ORDER)), columns=range(len(AHI_ORDER)))
    nonempty_idx = [i for i, g in enumerate(groups) if len(g) > 0]
    if len(nonempty_idx) < 2:
        return out

    try:
        # scikit_posthocs expects a list of arrays; it will ignore empties poorly, so subset
        subgroups = [groups[i] for i in nonempty_idx]
        sub = sp.posthoc_dunn(subgroups, p_adjust="bonferroni")

        # Map back into full matrix
        for a_pos, a_i in enumerate(nonempty_idx):
            for b_pos, b_i in enumerate(nonempty_idx):
                out.iloc[a_i, b_i] = float(sub.iloc[a_pos, b_pos])
        return out
    except Exception:
        return out


def format_p(p):
    if p is None or (isinstance(p, float) and (np.isnan(p) or not np.isfinite(p))):
        return "NA"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def make_summary_page(pdf, df, df_ahi):
    """
    Create the first PDF page: overall summary + per-metric summary table.
    df: full dataframe (after numeric conversion)
    df_ahi: dataframe filtered to rows with AHI_category (AHI present)
    """
    # Overall counts
    n_rows_total = len(df)
    n_rows_ahi = len(df_ahi)

    # AHI stats
    ahi_vals = df_ahi["AHI"].dropna().values
    ahi_min = np.nanmin(ahi_vals) if ahi_vals.size else np.nan
    ahi_med = np.nanmedian(ahi_vals) if ahi_vals.size else np.nan
    ahi_mean = np.nanmean(ahi_vals) if ahi_vals.size else np.nan
    ahi_max = np.nanmax(ahi_vals) if ahi_vals.size else np.nan

    # N per AHI category (based on AHI present)
    n_by_cat_ahi = df_ahi["AHI_category"].value_counts().reindex(AHI_ORDER).fillna(0).astype(int).to_dict()

    # Per-metric summary: total non-missing N, per category N, KW p
    rows = []
    for metric in FEATURES:
        counts = {}
        groups = []
        for cat in AHI_ORDER:
            vals = df_ahi.loc[df_ahi["AHI_category"] == cat, metric].dropna().values
            counts[cat] = int(len(vals))
            groups.append(vals)

        stat, p = safe_kruskal(groups)
        rows.append({
            "Metric": metric,
            "N_total": int(sum(counts.values())),
            "N_<5": counts["AHI < 5"],
            "N_5-<15": counts["AHI 5–<15"],
            "N_15-<30": counts["AHI 15–<30"],
            "N_>=30": counts["AHI ≥ 30"],
            "KW_p": format_p(p),
        })

    summary_tbl = pd.DataFrame(rows)

    # ----- Plot summary page (text + table) -----
    fig = plt.figure(figsize=(11, 8.5))  # landscape-ish page
    fig.suptitle("AHI Category Boxplots — Summary", fontsize=16, y=0.98)

    # Top text block
    text_lines = [
        f"Source file: {CSV_PATH}",
        f"Total rows in file: {n_rows_total}",
        f"Rows with usable AHI (categorized): {n_rows_ahi}",
        "",
        f"AHI summary (usable rows):",
        f"  min={ahi_min:.2f}   median={ahi_med:.2f}   mean={ahi_mean:.2f}   max={ahi_max:.2f}" if np.isfinite(ahi_min) else "  NA",
        "",
        "N by AHI category (based on AHI present):",
        f"  {AHI_ORDER[0]}: {n_by_cat_ahi.get(AHI_ORDER[0], 0)}",
        f"  {AHI_ORDER[1]}: {n_by_cat_ahi.get(AHI_ORDER[1], 0)}",
        f"  {AHI_ORDER[2]}: {n_by_cat_ahi.get(AHI_ORDER[2], 0)}",
        f"  {AHI_ORDER[3]}: {n_by_cat_ahi.get(AHI_ORDER[3], 0)}",
        "",
        "Stats per metric:",
        "  Global: Kruskal–Wallis across available groups",
        "  Post-hoc: Dunn test with Bonferroni correction (stars on plots for p<0.05)",
        "  Note: N per metric can differ by group due to missing values.",
    ]
    fig.text(0.03, 0.86, "\n".join(text_lines), va="top", fontsize=10, family="monospace")

    # Table block
    ax_tbl = fig.add_axes([0.03, 0.06, 0.94, 0.50])  # [left, bottom, width, height]
    ax_tbl.axis("off")

    col_labels = ["Metric", "N_total", "N_<5", "N_5-<15", "N_15-<30", "N_>=30", "KW_p"]
    cell_text = summary_tbl[col_labels].values.tolist()

    table = ax_tbl.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    pdf.savefig(fig)
    plt.close(fig)


# =========================
# Main
# =========================

def main():
    df = read_table_safely(CSV_PATH)

    # Numeric conversion
    for c in ["AHI"] + FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            raise ValueError(f"Missing required column: {c}")

    # Categorize AHI
    df["AHI_category"] = df["AHI"].apply(ahi_category)
    df_ahi = df.dropna(subset=["AHI_category"]).copy()
    df_ahi["AHI_category"] = pd.Categorical(
        df_ahi["AHI_category"], categories=AHI_ORDER, ordered=True
    )

    with PdfPages(OUT_PDF) as pdf:
        # ---- First page summary ----
        make_summary_page(pdf, df, df_ahi)

        # ---- Metric pages ----
        for metric in FEATURES:
            data = []
            counts = {}

            for cat in AHI_ORDER:
                values = df_ahi.loc[df_ahi["AHI_category"] == cat, metric].dropna().values
                data.append(values)
                counts[cat] = int(len(values))

            if sum(counts.values()) == 0:
                continue

            # Stats
            kw_stat, kw_p = safe_kruskal(data)
            dunn = safe_dunn(data)

            # Plot
            fig, ax = plt.subplots(figsize=(9, 5))

            ax.boxplot(
                data,
                showfliers=False,
            )

            ax.set_title(
                f"{metric} by AHI category\n"
                f"Kruskal–Wallis p = {format_p(kw_p)}"
            )
            ax.set_xlabel("AHI Category")
            ax.set_ylabel(metric)

            add_group_counts(ax, counts)

            # Pairwise significance
            add_pairwise_annotations(ax, data, dunn)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[DONE] Saved PDF: {OUT_PDF}")


if __name__ == "__main__":
    main()
