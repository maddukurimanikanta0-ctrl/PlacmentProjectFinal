"""
ml/tree_models.py
-----------------
M3: Tree-Based Models, Ensembles & Industrial Workhorses.
Covers CART Decision Trees, Cost-Complexity Pruning (CCP), Random Forests (Bagging),
and Gradient Boosting (GBDT / HistGradientBoosting LightGBM-style).
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

from ml.data_loader import load_data, clean_data

NUMERIC_FEATURES = [
    "CGPA", "AttendancePercent", "Internships", "Projects", "Workshops",
    "Certifications", "AptitudeTestScore", "CodingTestScore",
    "SoftSkillsRating", "MockInterviewScore"
]


def train_decision_tree_analysis(max_depth: int = 5, ccp_alpha: float = 0.0) -> Dict[str, Any]:
    """Trains a Decision Tree with optional Cost-Complexity Pruning (CCP)."""
    df = clean_data(load_data())
    sample_df = df.sample(n=min(len(df), 6000), random_state=42).copy()

    X = sample_df[NUMERIC_FEATURES].values
    y = sample_df["PlacementStatus"].values

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    dt = DecisionTreeClassifier(
        max_depth=max_depth if max_depth > 0 else None,
        ccp_alpha=ccp_alpha,
        criterion="gini",
        random_state=42
    )
    dt.fit(X_train, y_train)

    train_acc = round(float(dt.score(X_train, y_train)), 4)
    test_acc = round(float(dt.score(X_test, y_test)), 4)
    test_auc = round(float(roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1])), 4)

    # Cost-complexity pruning path summary
    path = dt.cost_complexity_pruning_path(X_train, y_train)
    ccp_sample = [
        {"alpha": round(float(a), 5), "impurity": round(float(imp), 4)}
        for a, imp in zip(path.ccp_alphas[::max(1, len(path.ccp_alphas)//6)], path.impurities[::max(1, len(path.impurities)//6)])
    ]

    importances = {feat: round(float(imp), 4) for feat, imp in zip(NUMERIC_FEATURES, dt.feature_importances_)}
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    return {
        "max_depth": max_depth,
        "ccp_alpha": ccp_alpha,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "test_roc_auc": test_auc,
        "tree_depth": dt.get_depth(),
        "n_leaves": dt.get_n_leaves(),
        "pruning_path": ccp_sample[:6],
        "feature_importances": sorted_imp,
        "stopping_rules": {
            "criterion": "Gini Impurity (1 - sum p_i^2)",
            "min_samples_split": 2,
            "min_samples_leaf": 1
        }
    }


def train_random_forest_analysis(n_estimators: int = 100, max_features: str = "sqrt") -> Dict[str, Any]:
    """Evaluates Random Forest with Bagging, Out-of-Bag (OOB) error, and feature subsampling."""
    df = clean_data(load_data())
    sample_df = df.sample(n=min(len(df), 5000), random_state=42).copy()

    X = sample_df[NUMERIC_FEATURES].values
    y = sample_df["PlacementStatus"].values

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    oob = round(float(rf.oob_score_), 4)
    test_acc = round(float(rf.score(X_test, y_test)), 4)
    test_auc = round(float(roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])), 4)

    importances = {feat: round(float(imp), 4) for feat, imp in zip(NUMERIC_FEATURES, rf.feature_importances_)}
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    return {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "oob_score": oob,
        "oob_error": round(1.0 - oob, 4),
        "test_accuracy": test_acc,
        "test_roc_auc": test_auc,
        "variance_reduction_notes": "Averaging B decorrelated trees reduces variance from sigma^2 to rho*sigma^2 + (1-rho)/B*sigma^2.",
        "feature_importances": sorted_imp
    }


def compare_boosting_algorithms() -> Dict[str, Any]:
    """Compares standard GBDT vs Industrial Histogram-based LightGBM-style Gradient Boosting."""
    df = clean_data(load_data())
    sample_df = df.sample(n=min(len(df), 4000), random_state=42).copy()

    X = sample_df[NUMERIC_FEATURES].values
    y = sample_df["PlacementStatus"].values

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    gbdt = GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, max_depth=3, random_state=42)
    hgb = HistGradientBoostingClassifier(max_iter=80, learning_rate=0.1, max_leaf_nodes=31, random_state=42)

    gbdt.fit(X_train, y_train)
    hgb.fit(X_train, y_train)

    gbdt_pred = gbdt.predict(X_test)
    hgb_pred = hgb.predict(X_test)

    return {
        "gradient_boosting": {
            "name": "Standard Gradient Boosting (GBDT)",
            "accuracy": round(float(accuracy_score(y_test, gbdt_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, gbdt.predict_proba(X_test)[:, 1])), 4),
            "approach": "Exact greedy split search on sorted feature values"
        },
        "hist_gradient_boosting": {
            "name": "Histogram Gradient Boosting (LightGBM-Style)",
            "accuracy": round(float(accuracy_score(y_test, hgb_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, hgb.predict_proba(X_test)[:, 1])), 4),
            "approach": "Binned 256-bin histogram splitting with leaf-wise growth"
        },
        "tabular_superiority_rationale": "Tree ensembles handle mixed continuous/categorical features without scale sensitivity and automatically capture non-linear step-function boundaries."
    }


def get_permutation_importance_summary() -> Dict[str, Any]:
    """Calculates Permutation Importance (MDI vs Permutation) on test holdout."""
    df = clean_data(load_data())
    sample_df = df.sample(n=min(len(df), 3000), random_state=42).copy()
    X = sample_df[NUMERIC_FEATURES].values
    y = sample_df["PlacementStatus"].values

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
    r = permutation_importance(rf, X_test, y_test, n_repeats=5, random_state=42)

    perm_list = []
    for i, feat in enumerate(NUMERIC_FEATURES):
        perm_list.append({
            "feature": feat,
            "gini_importance": round(float(rf.feature_importances_[i]), 4),
            "perm_mean": round(float(r.importances_mean[i]), 4),
            "perm_std": round(float(r.importances_std[i]), 4)
        })
    perm_list.sort(key=lambda x: x["perm_mean"], reverse=True)

    return {
        "rankings": perm_list,
        "shap_preview": {
            "top_positive_driver": perm_list[0]["feature"],
            "top_secondary_driver": perm_list[1]["feature"],
            "explanation": "SHAP satisfies local accuracy and consistency: sum phi_i = f(x) - E[f(x)]."
        }
    }
