"""
src/Validation/K_flod.py
------------------------
K-Fold Cross-Validation Module for Machine Learning Placement Prediction.

Evaluates model performance using K-Fold validation on 100 samples from the dataset
with K = 5 (splits into 5 equal folds of 20 samples each, iteratively training on 4 folds
and evaluating on 1 held-out fold).
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Fix Windows console encoding if needed
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Anchor paths dynamically to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from ml.data_loader import load_data, clean_data
from ml.preprocessing import prepare_features, build_preprocessor


def run_kfold_validation(
    n_samples: int = 50,
    k: int = 5,
    random_state: int = 42,
    stratified: bool = True,
    model_type: str = "logistic_regression",
) -> dict:
    """
    Performs K-Fold Cross Validation on n_samples (default: 100) with K = 5 folds.

    Parameters:
        n_samples (int): Number of samples extracted from the dataset (default: 100).
        k (int): Number of cross-validation folds (default: 5).
        random_state (int): Random seed for reproducibility.
        stratified (bool): Whether to use StratifiedKFold preserving class ratio (default: True).
        model_type (str): 'logistic_regression' or 'random_forest'.

    Returns:
        dict: Detailed per-fold results, fold averages, standard deviations, and summary.
    """
    # 1. Load and clean the dataset
    raw_df = load_data()
    df = clean_data(raw_df)

    # 2. Extract exactly n_samples (100 samples)
    if len(df) > n_samples:
        sampled_df = df.sample(n=n_samples, random_state=random_state).copy()
    else:
        sampled_df = df.copy()

    # 3. Separate features X and target y
    X, y = prepare_features(sampled_df)

    # 4. Initialize K-Fold splitter
    if stratified and y.nunique() > 1:
        cv_splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        splitter_name = f"Stratified K-Fold (K={k})"
    else:
        cv_splitter = KFold(n_splits=k, shuffle=True, random_state=random_state)
        splitter_name = f"Standard K-Fold (K={k})"

    fold_results = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    roc_aucs = []

    fold_idx = 1
    for train_idx, val_idx in cv_splitter.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # Pipeline with preprocessor built on training fold to avoid data leakage
        preprocessor = build_preprocessor(X_train_fold)

        if model_type.lower() == "random_forest":
            clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state)
        else:
            clf = LogisticRegression(max_iter=1000, random_state=random_state)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("clf", clf),
        ])

        pipeline.fit(X_train_fold, y_train_fold)

        y_train_pred = pipeline.predict(X_train_fold)
        y_val_pred = pipeline.predict(X_val_fold)

        train_acc = float(accuracy_score(y_train_fold, y_train_pred))
        val_acc = float(accuracy_score(y_val_fold, y_val_pred))
        val_prec = float(precision_score(y_val_fold, y_val_pred, zero_division=0))
        val_rec = float(recall_score(y_val_fold, y_val_pred, zero_division=0))
        val_f1 = float(f1_score(y_val_fold, y_val_pred, zero_division=0))

        val_auc = None
        if hasattr(pipeline, "predict_proba") and y_val_fold.nunique() > 1:
            try:
                proba = pipeline.predict_proba(X_val_fold)[:, 1]
                val_auc = float(roc_auc_score(y_val_fold, proba))
            except Exception:
                val_auc = None

        cm = confusion_matrix(y_val_fold, y_val_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = map(int, cm.ravel())
        else:
            tn = fp = fn = tp = 0

        accuracies.append(val_acc)
        precisions.append(val_prec)
        recalls.append(val_rec)
        f1s.append(val_f1)
        if val_auc is not None:
            roc_aucs.append(val_auc)

        fold_results.append({
            "fold": fold_idx,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "train_accuracy": round(train_acc, 4),
            "val_accuracy": round(val_acc, 4),
            "precision": round(val_prec, 4),
            "recall": round(val_rec, 4),
            "f1_score": round(val_f1, 4),
            "roc_auc": round(val_auc, 4) if val_auc is not None else "N/A",
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "val_placed": int((y_val_fold == 1).sum()),
            "val_not_placed": int((y_val_fold == 0).sum()),
        })

        fold_idx += 1

    # Aggregate statistics
    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))
    mean_prec = float(np.mean(precisions))
    std_prec = float(np.std(precisions))
    mean_rec = float(np.mean(recalls))
    std_rec = float(np.std(recalls))
    mean_f1 = float(np.mean(f1s))
    std_f1 = float(np.std(f1s))
    mean_auc = float(np.mean(roc_aucs)) if len(roc_aucs) > 0 else None
    std_auc = float(np.std(roc_aucs)) if len(roc_aucs) > 0 else None

    # Prepare preview of samples
    preview_cols = [c for c in ["StudentID", "CGPA", "AptitudeTestScore", "CodingTestScore", "PlacementStatus"] if c in sampled_df.columns]
    sample_preview = sampled_df[preview_cols].head(10).to_dict(orient="records")

    results = {
        "validation_method": "K-Fold Cross Validation",
        "k_value": k,
        "splitter_name": splitter_name,
        "model_type": model_type,
        "total_samples": len(sampled_df),
        "samples_per_fold": len(sampled_df) // k,
        "folds": fold_results,
        "mean_accuracy": round(mean_acc, 4),
        "std_accuracy": round(std_acc, 4),
        "mean_precision": round(mean_prec, 4),
        "std_precision": round(std_prec, 4),
        "mean_recall": round(mean_rec, 4),
        "std_recall": round(std_rec, 4),
        "mean_f1_score": round(mean_f1, 4),
        "std_f1_score": round(std_f1, 4),
        "mean_roc_auc": round(mean_auc, 4) if mean_auc is not None else "N/A",
        "std_roc_auc": round(std_auc, 4) if std_auc is not None else "N/A",
        "all_accuracies": [round(a, 4) for a in accuracies],
        "sample_preview": sample_preview,
    }
    return results


def main():
    print("=" * 70)
    print(f"[*] Running K-Fold Cross Validation (100 Samples, K = 5)")
    print("=" * 70)
    res = run_kfold_validation(n_samples=100, k=5, random_state=42)

    print(f"Splitter:               {res['splitter_name']}")
    print(f"Total Samples Drawn:    {res['total_samples']}")
    print(f"Number of Folds (K):    {res['k_value']}")
    print(f"Samples per Val Fold:   {res['samples_per_fold']}")
    print(f"Model:                  {res['model_type']}")
    print("-" * 70)
    print("Per-Fold Performance Breakdown:")
    print(f"{'Fold':<6} | {'Val Acc':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<8}")
    print("-" * 60)
    for f in res["folds"]:
        print(f"Fold {f['fold']:<2} | {f['val_accuracy']*100:>6.2f}%    | {f['precision']:>9.4f}  | {f['recall']:>8.4f}  | {f['f1_score']:>8.4f}  | {str(f['roc_auc']):>7}")
    print("-" * 60)
    print(f"Mean Accuracy:          {res['mean_accuracy']*100:.2f}% (+/- {res['std_accuracy']*100:.2f}%)")
    print(f"Mean Precision:         {res['mean_precision']:.4f} (+/- {res['std_precision']:.4f})")
    print(f"Mean Recall:            {res['mean_recall']:.4f} (+/- {res['std_recall']:.4f})")
    print(f"Mean F1-Score:          {res['mean_f1_score']:.4f} (+/- {res['std_f1_score']:.4f})")
    print(f"Mean ROC-AUC:           {res['mean_roc_auc']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
