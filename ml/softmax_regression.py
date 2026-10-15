"""
ml/softmax_regression.py
------------------------
M2: Multinomial Logistic Regression (Softmax Regression).
Multi-class placement tier classification:
  Tier 1: Dream / High Package (> 8.0 LPA)
  Tier 2: Core Engineering (5.0 - 8.0 LPA)
  Tier 3: Entry / Foundational (< 5.0 LPA or Unplaced)
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

from ml.data_loader import load_data, clean_data

TIER_NAMES = [
    "Entry / Foundational (< 5 LPA)",
    "Core Engineering (5 - 8 LPA)",
    "Dream / Premium (> 8 LPA)"
]

NUMERIC_FEATURES = [
    "CGPA", "AttendancePercent", "Internships", "Projects",
    "AptitudeTestScore", "CodingTestScore", "MockInterviewScore"
]


def _get_salary_col(df: pd.DataFrame) -> str:
    for c in ["Salary Package", "SalaryPackage", "Salary"]:
        if c in df.columns:
            return c
    return "Salary"


def _assign_tier(row, sal_col: str) -> int:
    """Assigns 3-class placement tier based on status and salary."""
    if row.get("PlacementStatus", 0) == 0:
        return 0
    salary = float(row.get(sal_col, 4.0))
    if salary >= 8.0:
        return 2
    elif salary >= 5.0:
        return 1
    return 0


def train_softmax_model() -> Dict[str, Any]:
    """Trains multinomial logistic regression model with softmax activation."""
    df = clean_data(load_data())
    sal_col = _get_salary_col(df)
    df_sample = df.sample(n=min(len(df), 6000), random_state=42).copy()
    df_sample["PlacementTier"] = df_sample.apply(lambda r: _assign_tier(r, sal_col), axis=1)

    valid_cols = [c for c in NUMERIC_FEATURES if c in df_sample.columns]
    X = df_sample[valid_cols].values
    y = df_sample["PlacementTier"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    split_idx = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Multinomial logistic regression
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = round(float(accuracy_score(y_test, y_pred)), 4)
    report = classification_report(y_test, y_pred, target_names=TIER_NAMES, output_dict=True)

    tier_coefficients = {}
    for i, tname in enumerate(TIER_NAMES):
        coef_map = {feat: round(float(c), 4) for feat, c in zip(valid_cols, model.coef_[i])}
        sorted_top = sorted(coef_map.items(), key=lambda x: x[1], reverse=True)
        tier_coefficients[tname] = {
            "intercept": round(float(model.intercept_[i]), 4),
            "top_positive": sorted_top[:3],
            "all_coefficients": coef_map
        }

    return {
        "accuracy": acc,
        "classes": TIER_NAMES,
        "features": valid_cols,
        "classification_report": report,
        "tier_coefficients": tier_coefficients,
        "mathematical_formula": "P(y = k | x) = exp(w_k^T x) / sum_j exp(w_j^T x)",
        "test_samples": len(y_test)
    }


def predict_softmax_tier(data: Dict[str, Any]) -> Dict[str, Any]:
    """Infers class probabilities using softmax regression for a student input."""
    df = clean_data(load_data())
    sal_col = _get_salary_col(df)
    df_sample = df.sample(n=min(len(df), 4000), random_state=42).copy()
    df_sample["PlacementTier"] = df_sample.apply(lambda r: _assign_tier(r, sal_col), axis=1)

    valid_cols = [c for c in NUMERIC_FEATURES if c in df_sample.columns]
    X = df_sample[valid_cols].values
    y = df_sample["PlacementTier"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(solver="lbfgs", max_iter=500, random_state=42)
    model.fit(X_scaled, y)

    input_vals = [
        float(data.get("CGPA", 7.5)),
        float(data.get("AttendancePercent", 80.0)),
        float(data.get("Internships", 1.0)),
        float(data.get("Projects", 2.0)),
        float(data.get("AptitudeTestScore", 70.0)),
        float(data.get("CodingTestScore", 75.0)),
        float(data.get("MockInterviewScore", 70.0)),
    ][:len(valid_cols)]

    inp_scaled = scaler.transform([input_vals])

    probs = model.predict_proba(inp_scaled)[0]
    pred_idx = int(model.predict(inp_scaled)[0])

    breakdown = [
        {
            "tier": TIER_NAMES[i],
            "probability": round(float(probs[i]) * 100, 2),
            "is_predicted": (i == pred_idx)
        }
        for i in range(len(TIER_NAMES))
    ]

    return {
        "predicted_tier": TIER_NAMES[pred_idx],
        "tier_id": pred_idx,
        "probabilities": breakdown,
        "features_submitted": dict(zip(valid_cols, input_vals))
    }
