"""
ml/preprocessing.py
-------------------
Data cleaning, normalization, categorical encoding, and feature transformation.
Provides scikit-learn ColumnTransformer pipeline and single student schema preparation.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DROP_COLUMNS = [
    "StudentID",
    "IsAnomaly",
    "Salary Package",
    "SalaryPackage",
]

TARGET = "PlacementStatus"
TARGET_COL = "PlacementStatus"

NUMERIC_FIELDS = [
    "SGPA_Sem1",
    "SGPA_Sem2",
    "SGPA_Sem3",
    "SGPA_Sem4",
    "SGPA_Sem5",
    "SGPA_Sem6",
    "SGPA_Sem7",
    "SGPA_Sem8",
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

CATEGORICAL_FIELDS = [
    "Gender",
    "City",
    "CollegeTier",
    "Stream",
    "Specialisation",
    "Hostel",
    "HistoryOfBacklogs",
    "CGPA_Tier",
]

DEFAULT_VALUES = {
    "Gender": "Male",
    "City": "Hyderabad",
    "CollegeTier": "Tier2",
    "Stream": "CS",
    "Specialisation": "DataScience",
    "Hostel": "No",
    "HistoryOfBacklogs": "No",
    "CGPA_Tier": "Mid",
    "CGPA": 7.5,
    "AttendancePercent": 80.0,
    "Internships": 1,
    "Projects": 2,
    "Workshops": 2,
    "Certifications": 2,
    "Publications": 0,
    "AptitudeTestScore": 75.0,
    "SoftSkillsRating": 4.0,
    "CodingTestScore": 75.0,
    "MockInterviewScore": 70.0,
    "ExtraCurricular": 1,
}
for i in range(1, 9):
    DEFAULT_VALUES[f"SGPA_Sem{i}"] = 7.5


def normalization(df: pd.DataFrame, num_cols: list = None):
    """
    MinMaxScaler normalization on numerical features.
    Splits dataset into 80/20 train/test and scales features.
    """
    if num_cols is None:
        num_cols = ["CGPA", "AttendancePercent", "AptitudeTestScore", "CodingTestScore", "Internships"]
    
    valid_cols = [c for c in num_cols if c in df.columns]
    stratify_col = df[TARGET] if TARGET in df.columns else None
    
    train_df, test_df = train_test_split(
        df[valid_cols],
        test_size=0.2,
        random_state=42,
        stratify=stratify_col
    )
    
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    
    train_scaled_df = pd.DataFrame(train_scaled, columns=valid_cols, index=train_df.index)
    test_scaled_df = pd.DataFrame(test_scaled, columns=valid_cols, index=test_df.index)
    return train_scaled_df, test_scaled_df


def build_preprocessor(X: pd.DataFrame, scaler_type: str = "standard") -> ColumnTransformer:
    """
    Constructs an end-to-end ColumnTransformer:
      - Numeric columns: Median SimpleImputer + StandardScaler (or MinMaxScaler)
      - Categorical columns: Most-frequent SimpleImputer + OneHotEncoder(handle_unknown='ignore')
    """
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop"
    )

    return preprocessor


def prepare_features(df: pd.DataFrame):
    """
    Separates feature matrix X and target vector y from dataset.
    Excludes identifiers and target leakage columns.
    """
    clean_df = df.copy()
    drop_list = [c for c in DROP_COLUMNS + [TARGET] if c in clean_df.columns]
    X = clean_df.drop(columns=drop_list, errors="ignore")
    y = clean_df[TARGET] if TARGET in clean_df.columns else None
    return X, y


def prepare_single_student(data: dict) -> pd.DataFrame:
    """
    Parses request payload from web form or REST API.
    Ensures correct data types and applies fallback defaults.
    Returns 1-row DataFrame strictly matching the training feature schema.
    """
    record = {}

    # Numeric academic & skill features
    for field in NUMERIC_FIELDS:
        val = data.get(field, None)
        if val is not None and str(val).strip() != "":
            try:
                record[field] = float(val)
            except ValueError:
                record[field] = float(DEFAULT_VALUES.get(field, 0.0))
        else:
            if field.startswith("SGPA_Sem") and data.get("CGPA"):
                try:
                    record[field] = float(data.get("CGPA"))
                except ValueError:
                    record[field] = float(DEFAULT_VALUES[field])
            else:
                record[field] = float(DEFAULT_VALUES.get(field, 0.0))

    # Categorical features
    for field in CATEGORICAL_FIELDS:
        val = data.get(field, None)
        if val is not None and str(val).strip() != "":
            record[field] = str(val).strip()
        else:
            if field == "CGPA_Tier":
                cgpa = record.get("CGPA", 7.5)
                record[field] = "High" if cgpa >= 8.0 else ("Mid" if cgpa >= 6.5 else "Low")
            else:
                record[field] = DEFAULT_VALUES.get(field, "Unknown")

    return pd.DataFrame([record])
