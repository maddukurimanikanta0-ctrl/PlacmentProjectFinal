"""
ml/eda.py
---------
Exploratory Data Analysis helpers: summary statistics and saved chart images.
Dynamically writes chart PNGs to static/charts/ anchored to the project root.
"""

from pathlib import Path
import os
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHARTS_DIR = PROJECT_ROOT / "static" / "charts"

import matplotlib
matplotlib.use("Agg")  # headless backend for server execution
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ml.data_loader import NUMERIC_COLS, TARGET_COL

sns.set_theme(style="whitegrid")


def ensure_charts_dir():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def get_summary_stats(df: pd.DataFrame) -> dict:
    """
    Numeric describe() table + categorical value counts, returned as
    plain Python / JSON-friendly structures for rendering in templates.
    """
    numeric_cols = [c for c in NUMERIC_COLS if c in df.columns]
    describe_df = df[numeric_cols].describe().T.round(2)

    categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    cat_summary = {
        col: df[col].value_counts().head(10).to_dict() for col in categorical_cols
    }

    placement_rate = None
    if TARGET_COL in df.columns:
        placement_rate = round(float(df[TARGET_COL].mean()) * 100, 2)

    return {
        "numeric_describe": describe_df.to_dict(orient="index"),
        "categorical_summary": cat_summary,
        "placement_rate": placement_rate,
        "correlation_with_target": _correlation_with_target(df, numeric_cols),
    }


def _correlation_with_target(df: pd.DataFrame, numeric_cols):
    if TARGET_COL not in df.columns:
        return {}
    corr = df[numeric_cols + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL, errors="ignore")
    corr = corr.sort_values(ascending=False).round(3)
    return corr.to_dict()


def generate_charts(df: pd.DataFrame) -> list:
    """
    Generates a standard set of academic EDA charts and saves them to static/charts/.
    Returns list of saved filenames.
    """
    ensure_charts_dir()
    filenames = []

    # 1. Target class distribution
    if TARGET_COL in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        counts = df[TARGET_COL].value_counts()
        bars = ax.bar(["Not Placed (0)", "Placed (1)"], [counts.get(0, 0), counts.get(1, 0)], color=["#94a3b8", "#3b82f6"], width=0.5)
        ax.set_title("Student Placement Distribution", fontsize=11, fontweight="bold", pad=12)
        ax.set_ylabel("Number of Students")
        y_offset = float(counts.max()) * 0.015
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval + y_offset, f"{int(yval):,}", ha="center", va="bottom", fontsize=10)
        fname = "placement_distribution.png"
        fig.tight_layout()
        fig.savefig(CHARTS_DIR / fname, dpi=120)
        plt.close(fig)
        filenames.append(fname)

    # 2. CGPA distribution by placement status
    if "CGPA" in df.columns and TARGET_COL in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.histplot(
            data=df, x="CGPA", hue=TARGET_COL, kde=True, ax=ax, palette={0: "#94a3b8", 1: "#3b82f6"},
            element="step",
        )
        ax.set_title("CGPA Distribution by Placement Outcome", fontsize=11, fontweight="bold", pad=12)
        fname = "cgpa_distribution.png"
        fig.tight_layout()
        fig.savefig(CHARTS_DIR / fname, dpi=120)
        plt.close(fig)
        filenames.append(fname)

    # 3. Correlation heatmap (numeric features)
    corr_cols = [
        "CGPA", "AttendancePercent", "Internships", "Projects", "Workshops",
        "AptitudeTestScore", "CodingTestScore", "SoftSkillsRating",
        "MockInterviewScore", TARGET_COL
    ]
    numeric_cols = [c for c in corr_cols if c in df.columns]
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(8, 6.5))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, cmap="Blues", annot=True, fmt=".2f", ax=ax, square=True, linewidths=0.5, annot_kws={"size": 9})
        ax.set_title("Feature Correlation Matrix", fontsize=11, fontweight="bold", pad=12)
        plt.xticks(rotation=45, ha="right")
        fname = "correlation_heatmap.png"
        fig.tight_layout()
        fig.savefig(CHARTS_DIR / fname, dpi=120)
        plt.close(fig)
        filenames.append(fname)

    # 4. Placement rate by CollegeTier
    if "CollegeTier" in df.columns and TARGET_COL in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        rate = df.groupby("CollegeTier")[TARGET_COL].mean().sort_values(ascending=False) * 100
        bars = ax.bar(rate.index, rate.values, color=["#60a5fa", "#3b82f6", "#2563eb"], width=0.45)
        ax.set_title("Placement Rate (%) by College Tier", fontsize=11, fontweight="bold", pad=12)
        ax.set_ylabel("Placement Rate (%)")
        ax.set_ylim(0, 100)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=10)
        fname = "placement_by_college_tier.png"
        fig.tight_layout()
        fig.savefig(CHARTS_DIR / fname, dpi=120)
        plt.close(fig)
        filenames.append(fname)

    # 5. Internships vs Placement
    if "Internships" in df.columns and TARGET_COL in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        rate = df.groupby("Internships")[TARGET_COL].mean() * 100
        bars = ax.bar(rate.index.astype(str), rate.values, color="#3b82f6", width=0.5)
        ax.set_title("Placement Rate (%) by Number of Internships", fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel("Number of Internships")
        ax.set_ylabel("Placement Rate (%)")
        fname = "placement_by_internships.png"
        fig.tight_layout()
        fig.savefig(CHARTS_DIR / fname, dpi=120)
        plt.close(fig)
        filenames.append(fname)

    return filenames


def generate_eda_charts(df: pd.DataFrame):
    """Alias for generate_charts for full compatibility."""
    return generate_charts(df)


if __name__ == "__main__":
    from ml.data_loader import load_data, clean_data
    df = clean_data(load_data())
    stats = get_summary_stats(df)
    print("Placement Rate:", stats["placement_rate"])
    charts = generate_charts(df)
    print("Generated charts:", charts)
