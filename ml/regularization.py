"""
ml/regularization.py
--------------------
M2: Regularization in Linear Models (Ridge L2, Lasso L1, Elastic Net).
Analyzes geometric shrinkage, sparsity induction, and non-zero feature selection.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.data_loader import load_data, clean_data

NUMERIC_FEATURES = [
    "CGPA", "AttendancePercent", "Internships", "Projects", "Workshops",
    "Certifications", "AptitudeTestScore", "CodingTestScore",
    "SoftSkillsRating", "MockInterviewScore"
]


def _get_salary_col(df: pd.DataFrame) -> str:
    for c in ["Salary Package", "SalaryPackage", "Salary"]:
        if c in df.columns:
            return c
    return "Salary"


def compare_regularization(alpha: float = 1.0, l1_ratio: float = 0.5) -> Dict[str, Any]:
    """
    Fits Ridge, Lasso, and ElasticNet models on placement salary regression
    to compare coefficient shrinkage and feature sparsity.
    """
    df = clean_data(load_data())
    sal_col = _get_salary_col(df)
    valid_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    
    valid_df = df[df.get("PlacementStatus", 1) == 1].dropna(subset=valid_cols + [sal_col]).copy()
    valid_df = valid_df[valid_df[sal_col] > 0]
    if len(valid_df) > 5000:
        valid_df = valid_df.sample(n=5000, random_state=42)

    X = valid_df[valid_cols].values
    y = valid_df[sal_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42
    )

    models = {
        "Ridge (L2)": Ridge(alpha=alpha, random_state=42),
        "Lasso (L1)": Lasso(alpha=alpha, random_state=42, max_iter=2000),
        "Elastic Net": ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=2000)
    }

    results = {}
    for name, reg in models.items():
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        mse = float(np.mean((y_test - y_pred) ** 2))
        r2 = float(reg.score(X_test, y_test))
        coefs = [round(float(c), 4) for c in reg.coef_]
        zero_count = int(np.sum(np.abs(reg.coef_) < 1e-4))

        results[name] = {
            "mse": round(mse, 4),
            "r2_score": round(r2, 4),
            "intercept": round(float(reg.intercept_), 4),
            "coefficients": dict(zip(valid_cols, coefs)),
            "zero_features_count": zero_count,
            "active_features_count": len(valid_cols) - zero_count
        }

    return {
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "features": valid_cols,
        "comparison": results,
        "geometric_notes": {
            "L1_Lasso": "Diamond constraint boundary with sharp corners at axes inducing exact zeros (sparsity).",
            "L2_Ridge": "Spherical constraint boundary shrinking all coefficients proportionally without setting them to zero.",
            "Elastic_Net": "Convex combination combining grouping effect of L2 with sparse selection of L1."
        }
    }


def get_coefficient_paths() -> Dict[str, Any]:
    """Generates coefficient path summary across varying regularization strengths."""
    alphas = [0.01, 0.1, 1.0, 10.0, 50.0]
    df = clean_data(load_data())
    sal_col = _get_salary_col(df)
    valid_cols = [c for c in NUMERIC_FEATURES if c in df.columns]

    valid_df = df[df.get("PlacementStatus", 1) == 1].dropna(subset=valid_cols + [sal_col]).copy()
    valid_df = valid_df[valid_df[sal_col] > 0]
    if len(valid_df) > 3000:
        valid_df = valid_df.sample(n=3000, random_state=42)

    X_scaled = StandardScaler().fit_transform(valid_df[valid_cols].values)
    y = valid_df[sal_col].values

    lasso_paths = []
    ridge_paths = []

    for a in alphas:
        l_mod = Lasso(alpha=a, random_state=42, max_iter=2000).fit(X_scaled, y)
        r_mod = Ridge(alpha=a, random_state=42).fit(X_scaled, y)
        lasso_paths.append({
            "alpha": a,
            "zeros": int(np.sum(np.abs(l_mod.coef_) < 1e-4)),
            "r2": round(float(l_mod.score(X_scaled, y)), 4)
        })
        ridge_paths.append({
            "alpha": a,
            "zeros": int(np.sum(np.abs(r_mod.coef_) < 1e-4)),
            "r2": round(float(r_mod.score(X_scaled, y)), 4)
        })

    return {"alphas": alphas, "lasso_paths": lasso_paths, "ridge_paths": ridge_paths}
