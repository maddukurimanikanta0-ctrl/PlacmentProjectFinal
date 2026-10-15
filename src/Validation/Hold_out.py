
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

from sklearn.model_selection import train_test_split
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
    classification_report,
)

from ml.data_loader import load_data, clean_data
from ml.preprocessing import prepare_features, build_preprocessor


def run_holdout_validation(
    n_samples: int = 50,
    test_size: float = 0.20,
    random_state: int = 42,
    model_type: str = "logistic_regression",
) -> dict:

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

    # 4. Partition into Train and Test using Hold-Out split
    stratify_target = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    # 5. Build preprocessor and pipeline
    preprocessor = build_preprocessor(X_train)

    if model_type.lower() == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state)
    else:
        clf = LogisticRegression(max_iter=1000, random_state=random_state)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf),
    ])

    # 6. Fit model on Training Split
    pipeline.fit(X_train, y_train)

    # 7. Evaluate on both Train (check overfitting) and Test (Held-out validation)
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Probabilities for ROC-AUC
    y_test_proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X_test)
            if proba.shape[1] > 1:
                y_test_proba = proba[:, 1]
        except Exception:
            y_test_proba = None

    # Compute metrics
    train_acc = float(accuracy_score(y_train, y_train_pred))
    test_acc = float(accuracy_score(y_test, y_test_pred))
    precision = float(precision_score(y_test, y_test_pred, zero_division=0))
    recall = float(recall_score(y_test, y_test_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_test_pred, zero_division=0))

    roc_auc = None
    if y_test_proba is not None and y_test.nunique() > 1:
        try:
            roc_auc = float(roc_auc_score(y_test, y_test_proba))
        except Exception:
            roc_auc = None

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = map(int, cm.ravel())
    else:
        tn = fp = fn = tp = 0

    class_rep = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)

    # Prepare preview of samples
    preview_cols = [c for c in ["StudentID", "CGPA", "AptitudeTestScore", "CodingTestScore", "PlacementStatus"] if c in sampled_df.columns]
    sample_preview = sampled_df[preview_cols].head(10).to_dict(orient="records")

    results = {
        "validation_method": "Hold-Out Validation",
        "model_type": model_type,
        "total_samples": len(sampled_df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_split_pct": round((1 - test_size) * 100, 1),
        "test_split_pct": round(test_size * 100, 1),
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else "N/A",
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "target_distribution": {
            "total_placed": int((sampled_df["PlacementStatus"] == 1).sum()),
            "total_not_placed": int((sampled_df["PlacementStatus"] == 0).sum()),
            "train_placed": int((y_train == 1).sum()),
            "train_not_placed": int((y_train == 0).sum()),
            "test_placed": int((y_test == 1).sum()),
            "test_not_placed": int((y_test == 0).sum()),
        },
        "sample_preview": sample_preview,
        "classification_report": class_rep,
    }
    return results


def main():
    print("=" * 65)
    print("[*] Running Hold-Out Validation (100 Samples)")
    print("=" * 65)
    res = run_holdout_validation(n_samples=100, test_size=0.20, random_state=42)

    print(f"Total Samples Drawn:     {res['total_samples']}")
    print(f"Training Samples:        {res['train_samples']} ({res['train_split_pct']}%)")
    print(f"Testing/Validation:      {res['test_samples']} ({res['test_split_pct']}%)")
    print(f"Model:                   {res['model_type']}")
    print("-" * 65)
    print("Evaluation Results:")
    print(f"  Validation Accuracy:   {res['test_accuracy'] * 100:.2f}%")
    print(f"  Precision:             {res['precision']:.4f}")
    print(f"  Recall:                {res['recall']:.4f}")
    print(f"  F1-Score:              {res['f1_score']:.4f}")
    print(f"  ROC-AUC:               {res['roc_auc']}")
    print("-" * 65)
    print("Confusion Matrix:")
    cm = res['confusion_matrix']
    print(f"  True Negatives (TN):   {cm['tn']}")
    print(f"  False Positives (FP):  {cm['fp']}")
    print(f"  False Negatives (FN):  {cm['fn']}")
    print(f"  True Positives (TP):   {cm['tp']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
