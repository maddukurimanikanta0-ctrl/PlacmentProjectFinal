"""
ml/logistic_regression.py
-------------------------
Logistic Regression classification module for student placement prediction.
Handles training with Ridge (L2) regularization, coefficient extraction,
odds ratio calculation, evaluation metrics, and inference.
"""

from pathlib import Path
import json
import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from ml.data_loader import load_data, clean_data
from ml.preprocessing import build_preprocessor, prepare_features, prepare_single_student

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH_JOBLIB = MODELS_DIR / "logistic_regression.joblib"
MODEL_PATH_PKL = MODELS_DIR / "logistic_regression.pkl"


def train_logistic_regression(df: pd.DataFrame = None, c_param: float = 1.0) -> dict:
    """
    Trains an end-to-end Logistic Regression Pipeline (ColumnTransformer + LogisticRegression).
    Saves model artifacts (.joblib and .pkl) to models/ directory.
    """
    if df is None:
        df = clean_data(load_data())

    X, y = prepare_features(df)
    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, C=c_param)),
    ])

    pipeline.fit(X_train, y_train)

    # Predictions & probabilities
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    # Extract feature names & coefficients for interpretation
    fitted_prep = pipeline.named_steps["preprocessor"]
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    cat_feature_names = fitted_prep.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols).tolist()
    all_features = num_cols + cat_feature_names

    clf = pipeline.named_steps["clf"]
    coefs = clf.coef_[0]
    coef_df = pd.DataFrame({
        "feature": all_features,
        "coefficient": [round(float(c), 4) for c in coefs],
        "odds_ratio": [round(float(np.exp(c)), 4) for c in coefs],
    }).sort_values("coefficient", ascending=False)

    metrics["top_positive"] = coef_df.head(6).to_dict(orient="records")
    metrics["top_negative"] = coef_df.tail(6).to_dict(orient="records")
    metrics["intercept"] = round(float(clf.intercept_[0]), 4)

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH_JOBLIB)
    with open(MODEL_PATH_PKL, "wb") as f:
        pickle.dump(pipeline, f)

    return metrics


def load_logistic_model():
    """Loads trained logistic regression pipeline from disk."""
    if MODEL_PATH_JOBLIB.exists():
        return joblib.load(MODEL_PATH_JOBLIB)
    if MODEL_PATH_PKL.exists():
        with open(MODEL_PATH_PKL, "rb") as f:
            return pickle.load(f)
    return None


def predict_logistic(student_data: dict) -> dict:
    """Predicts student placement status using Logistic Regression."""
    model = load_logistic_model()
    if model is None:
        raise FileNotFoundError("Logistic regression model not trained yet. Run train_logistic_regression() first.")

    df_student = prepare_single_student(student_data)
    pred_class = int(model.predict(df_student)[0])
    probabilities = model.predict_proba(df_student)[0]
    prob_placed = round(float(probabilities[1]) * 100, 2)
    prob_not_placed = round(float(probabilities[0]) * 100, 2)

    return {
        "prediction": pred_class,
        "status": "Placed" if pred_class == 1 else "Not Placed",
        "probability_placed": prob_placed,
        "probability_not_placed": prob_not_placed,
        "confidence": prob_placed if pred_class == 1 else prob_not_placed,
    }


if __name__ == "__main__":
    print("Training Logistic Regression...")
    res = train_logistic_regression()
    print("Accuracy:", res["accuracy"])
    print("ROC-AUC:", res["roc_auc"])
    print("Top positive features:", res["top_positive"])
