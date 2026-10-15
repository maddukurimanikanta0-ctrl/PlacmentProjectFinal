"""
ml/data_loader.py
------------------
Data loading, cleaning, and dataset insights module.
Anchor paths dynamically relative to project root.
Loads either data/placement_data.csv or fallback datasets.
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd

# Anchor path dynamically to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIMARY_DATA_PATH = PROJECT_ROOT / "data" / "placement_data.csv"
FALLBACK_DATA_PATH = PROJECT_ROOT / "data" / "placement_predict_50k_Dataset1.csv"
CHARTS_DIR = PROJECT_ROOT / "static" / "charts"

TARGET_COL = "PlacementStatus"
TARGET = "PlacementStatus"
ID_COLS = ["StudentID"]
ID_COL = "StudentID"
LEAKAGE_COLS = ["Salary Package", "SalaryPackage", "IsAnomaly"]

NUMERIC_WITH_NULLS = [
    "Workshops",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "CodingTestScore",
    "MockInterviewScore",
]

CATEGORICAL_COLS = [
    "Gender",
    "City",
    "CollegeTier",
    "Stream",
    "Specialisation",
    "Hostel",
    "HistoryOfBacklogs",
    "CGPA_Tier",
]

SGPA_COLS = [f"SGPA_Sem{i}" for i in range(1, 9)]

NUMERIC_COLS = SGPA_COLS + [
    "CGPA",
    "AttendancePercent",
    "Internships",
    "Projects",
    "Workshops",
    "Certifications",
    "Publications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "CodingTestScore",
    "MockInterviewScore",
    "ExtraCurricular",
]


def resolve_dataset_path(path=None) -> Path:
    """Finds the dataset file using dynamic relative paths."""
    if path is not None:
        p = Path(path)
        if p.exists():
            return p
    if PRIMARY_DATA_PATH.exists():
        return PRIMARY_DATA_PATH
    if FALLBACK_DATA_PATH.exists():
        return FALLBACK_DATA_PATH

    # If neither exists, attempt to generate fallback data
    try:
        from generate_data import generate_synthetic_data
        print(f"Dataset not found. Generating synthetic dataset at {PRIMARY_DATA_PATH}...")
        generate_synthetic_data(num_samples=5000, output_path=PRIMARY_DATA_PATH)
        return PRIMARY_DATA_PATH
    except Exception as e:
        raise FileNotFoundError(
            f"Dataset not found at {PRIMARY_DATA_PATH} or {FALLBACK_DATA_PATH} and fallback generator failed: {e}"
        )


def load_data(path=None) -> pd.DataFrame:
    """
    Load the raw placement dataset from CSV.
    Supports placement_data.csv and placement_predict_50k_Dataset1.csv.
    """
    csv_path = resolve_dataset_path(path)
    df = pd.read_csv(csv_path)

    # Standardize Salary Package column name while preserving accessibility
    if "Salary Package" in df.columns and "SalaryPackage" not in df.columns:
        df["SalaryPackage"] = df["Salary Package"]
    elif "SalaryPackage" in df.columns and "Salary Package" not in df.columns:
        df["Salary Package"] = df["SalaryPackage"]

    return df


def load_dataset(path=None) -> pd.DataFrame:
    """Alias for load_data for backward compatibility."""
    return load_data(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataframe:
      - impute missing numeric columns with the column median
      - drop exact duplicate rows
      - strip whitespace from string/categorical columns
    """
    clean = df.copy()
    clean.drop_duplicates(inplace=True)

    # Trim whitespace on categorical/text columns
    obj_cols = clean.select_dtypes(include=["object", "string"]).columns
    for col in obj_cols:
        clean[col] = clean[col].astype(str).str.strip()

    # Median-impute numeric columns that had nulls in raw data
    for col in NUMERIC_WITH_NULLS:
        if col in clean.columns and clean[col].isnull().any():
            median_val = clean[col].median()
            clean[col] = clean[col].fillna(median_val)

    return clean


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Alias for clean_data for backward compatibility."""
    return clean_data(df)


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Quick metadata about the dataframe, used by the data_loading page.
    """
    missing = df.isnull().sum()
    missing_dict = {col: int(cnt) for col, cnt in missing.items() if cnt > 0}

    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_counts": missing_dict,
        "missing_total": int(missing.sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": df[TARGET_COL].value_counts().to_dict()
        if TARGET_COL in df.columns
        else {},
    }


def get_feature_target_split(df: pd.DataFrame):
    """
    Split a cleaned dataframe into (X, y) ready for modelling.
    Drops identifier and target leakage columns.
    """
    drop_cols = [c for c in ID_COLS + LEAKAGE_COLS + [TARGET_COL] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    return X, y


def get_dataset_insights() -> dict:
    """
    Computes dataset statistics, correlation matrix, distributions,
    and sample rows for dashboard display and student report.
    """
    from ml.eda import generate_charts, generate_eda_charts

    raw_df = load_data()
    cleaned_df = clean_data(raw_df)

    total_rows = int(cleaned_df.shape[0])
    total_cols = int(cleaned_df.shape[1])

    # Missing counts before cleaning
    null_counts = raw_df.isnull().sum()
    missing_summary = {col: int(cnt) for col, cnt in null_counts.items() if cnt > 0}

    # Target counts & rate
    target_counts = cleaned_df[TARGET_COL].value_counts().to_dict() if TARGET_COL in cleaned_df.columns else {}
    placed_count = target_counts.get(1, 0)
    not_placed_count = target_counts.get(0, 0)
    placement_rate = round((placed_count / total_rows) * 100, 2) if total_rows > 0 else 0.0

    # Placement rate breakdowns
    tier_breakdown = {}
    if "CollegeTier" in cleaned_df.columns and TARGET_COL in cleaned_df.columns:
        tier_breakdown = (
            (cleaned_df.groupby("CollegeTier")[TARGET_COL].mean() * 100)
            .round(1)
            .to_dict()
        )

    stream_breakdown = {}
    if "Stream" in cleaned_df.columns and TARGET_COL in cleaned_df.columns:
        stream_breakdown = (
            (cleaned_df.groupby("Stream")[TARGET_COL].mean() * 100)
            .round(1)
            .to_dict()
        )

    # Correlation matrix for key features
    corr_features = [
        "CGPA",
        "AttendancePercent",
        "Internships",
        "Projects",
        "Workshops",
        "Certifications",
        "AptitudeTestScore",
        "CodingTestScore",
        "SoftSkillsRating",
        "MockInterviewScore",
        TARGET_COL,
    ]
    valid_corr_features = [c for c in corr_features if c in cleaned_df.columns]
    corr_df = cleaned_df[valid_corr_features].corr().round(2)
    corr_matrix = corr_df.to_dict()

    # Sample rows for dataset table preview
    sample_cols = [
        "StudentID",
        "Gender",
        "CollegeTier",
        "Stream",
        "CGPA",
        "AttendancePercent",
        "Internships",
        "CodingTestScore",
        "AptitudeTestScore",
        "PlacementStatus",
    ]
    valid_sample_cols = [c for c in sample_cols if c in cleaned_df.columns]
    sample_rows = cleaned_df[valid_sample_cols].head(12).to_dict(orient="records")

    # Ensure charts exist
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    if not (CHARTS_DIR / "correlation_heatmap.png").exists():
        try:
            generate_charts(cleaned_df)
        except Exception as e:
            print(f"Chart generation warning: {e}")

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "placed_count": placed_count,
        "not_placed_count": not_placed_count,
        "placement_rate": placement_rate,
        "missing_summary": missing_summary,
        "tier_breakdown": tier_breakdown,
        "stream_breakdown": stream_breakdown,
        "corr_cols": valid_corr_features,
        "corr_matrix": corr_matrix,
        "sample_cols": valid_sample_cols,
        "sample_rows": sample_rows,
    }


if __name__ == "__main__":
    raw = load_data()
    print("Dataset loaded successfully.")
    print("Raw shape:", raw.shape)
    summary = get_data_summary(raw)
    print("Summary rows:", summary["n_rows"], "cols:", summary["n_cols"])
    cleaned = clean_data(raw)
    print("Cleaned shape:", cleaned.shape)