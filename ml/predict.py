"""
ml/predict.py
-------------
Inference and serving pipeline for student placement prediction.
Loads packaged model artifacts dynamically anchored to project root.
Traces end-to-end inference from HTTP request to response with zero training-serving skew.
"""

from pathlib import Path
import joblib
import pickle
import pandas as pd
import numpy as np

from ml.preprocessing import prepare_single_student

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH_JOBLIB = MODELS_DIR / "model.joblib"
MODEL_PATH_PKL = MODELS_DIR / "model.pkl"

SALARY_MODEL_PATH_JOBLIB = MODELS_DIR / "salary_linear_model.joblib"
SALARY_MODEL_PATH_PKL = MODELS_DIR / "salary_linear_model.pkl"

KMEANS_MODEL_PATH_JOBLIB = MODELS_DIR / "kmeans_model.joblib"
KMEANS_MODEL_PATH_PKL = MODELS_DIR / "kmeans_model.pkl"

UNSUP_SCALER_PATH_JOBLIB = MODELS_DIR / "unsupervised_scaler.joblib"
UNSUP_SCALER_PATH_PKL = MODELS_DIR / "unsupervised_scaler.pkl"

UNSUP_IMPUTER_PATH_JOBLIB = MODELS_DIR / "unsupervised_imputer.joblib"
UNSUP_IMPUTER_PATH_PKL = MODELS_DIR / "unsupervised_imputer.pkl"

_MODEL = None
_SALARY_MODEL = None
_KMEANS = None
_UNSUP_SCALER = None
_UNSUP_IMPUTER = None


def _load_artifact(joblib_path: Path, pkl_path: Path):
    """Loads an artifact from joblib or pickle."""
    if joblib_path.exists():
        return joblib.load(joblib_path)
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    return None


def get_model():
    """Lazy loads primary production classification pipeline."""
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_artifact(MODEL_PATH_JOBLIB, MODEL_PATH_PKL)
        if _MODEL is None:
            # Fallback to logistic regression artifact if production model isn't built yet
            _MODEL = _load_artifact(MODELS_DIR / "logistic_regression.joblib", MODELS_DIR / "logistic_regression.pkl")
            if _MODEL is None:
                raise FileNotFoundError(
                    f"Model artifact not found at {MODEL_PATH_JOBLIB} or {MODEL_PATH_PKL}."
                )
    return _MODEL


def get_salary_model():
    """Lazy loads auxiliary salary linear regressor."""
    global _SALARY_MODEL
    if _SALARY_MODEL is None:
        _SALARY_MODEL = _load_artifact(SALARY_MODEL_PATH_JOBLIB, SALARY_MODEL_PATH_PKL)
    return _SALARY_MODEL


def get_unsupervised_models():
    """Lazy loads unsupervised clustering models for student segmentation."""
    global _KMEANS, _UNSUP_SCALER, _UNSUP_IMPUTER
    if _KMEANS is None:
        _KMEANS = _load_artifact(KMEANS_MODEL_PATH_JOBLIB, KMEANS_MODEL_PATH_PKL)
        _UNSUP_SCALER = _load_artifact(UNSUP_SCALER_PATH_JOBLIB, UNSUP_SCALER_PATH_PKL)
        _UNSUP_IMPUTER = _load_artifact(UNSUP_IMPUTER_PATH_JOBLIB, UNSUP_IMPUTER_PATH_PKL)
    return _KMEANS, _UNSUP_SCALER, _UNSUP_IMPUTER


CLUSTER_NAMES = {
    0: "Balanced Regular Performer",
    1: "High Technical Specialist",
    2: "Foundational / Skill-Building Cohort",
    3: "Elite Placement Candidate",
}


def predict_student(data: dict) -> dict:
    """
    End-to-End Prediction Trace:
      1. Parses student inputs into aligned schema DataFrame
      2. Runs pipeline inference (preprocessor + classifier)
      3. Calculates placement probability
      4. Connects salary regression estimate and student cluster archetype
    """
    model = get_model()
    df_student = prepare_single_student(data)

    # Classification inference
    pred_class = int(model.predict(df_student)[0])
    probabilities = model.predict_proba(df_student)[0]
    prob_placed = round(float(probabilities[1]) * 100, 2)
    prob_not_placed = round(float(probabilities[0]) * 100, 2)

    status_label = "Placed" if pred_class == 1 else "Not Placed"
    confidence = prob_placed if pred_class == 1 else prob_not_placed

    # Projected Salary estimate (Linear Regression)
    projected_salary = None
    salary_model = get_salary_model()
    if salary_model is not None and pred_class == 1:
        try:
            cgpa = float(df_student["CGPA"].iloc[0])
            coding = float(df_student["CodingTestScore"].iloc[0])
            aptitude = float(df_student["AptitudeTestScore"].iloc[0])
            features = ["CGPA", "CodingTestScore", "AptitudeTestScore"]
            reg_df = pd.DataFrame([[cgpa, coding, aptitude]], columns=features)
            salary_val = float(salary_model.predict(reg_df)[0])
            projected_salary = f"Rs. {max(3.2, round(salary_val, 2)):.2f} LPA"
        except Exception:
            projected_salary = "Rs. 6.50 LPA (Benchmark Average)"
    elif pred_class == 1:
        projected_salary = "Rs. 6.50 LPA (Benchmark Average)"

    # Cluster Archetype (K-Means)
    cluster_id = None
    cluster_archetype = None
    try:
        kmeans, scaler, imputer = get_unsupervised_models()
        if kmeans is not None and scaler is not None and imputer is not None:
            unsup_cols = [
                "CGPA", "AttendancePercent", "Internships", "Projects", "Workshops",
                "Certifications", "AptitudeTestScore", "CodingTestScore",
                "SoftSkillsRating", "MockInterviewScore",
            ]
            valid_unsup = df_student[unsup_cols].copy()
            X_imp = imputer.transform(valid_unsup)
            X_sc = scaler.transform(X_imp)
            cluster_id = int(kmeans.predict(X_sc)[0])
            cluster_archetype = CLUSTER_NAMES.get(cluster_id, "Standard Student Cohort")
    except Exception:
        cluster_id = 0
        cluster_archetype = CLUSTER_NAMES.get(0)

    # Academic Recommendations
    recommendations = []
    cgpa_val = float(df_student["CGPA"].iloc[0])
    coding_val = float(df_student["CodingTestScore"].iloc[0])
    internships_val = int(df_student["Internships"].iloc[0])
    aptitude_val = float(df_student["AptitudeTestScore"].iloc[0])

    if cgpa_val < 7.0:
        recommendations.append("Prioritize core coursework to raise cumulative CGPA above 7.0 (major campus screening filter).")
    if coding_val < 65:
        recommendations.append("Strengthen Data Structures & Algorithms (aim for 70+ in coding assessment benchmarks).")
    if internships_val == 0:
        recommendations.append("Complete at least 1 industry internship or capstone deployment to boost placement probability.")
    if aptitude_val < 65:
        recommendations.append("Practice quantitative and analytical aptitude tests for initial technical screening rounds.")

    if not recommendations:
        recommendations.append("Strong profile across all indicators. Maintain momentum in technical mock interviews!")

    return {
        "prediction": pred_class,
        "status": status_label,
        "probability_placed": prob_placed,
        "probability_not_placed": prob_not_placed,
        "confidence": confidence,
        "projected_salary": projected_salary,
        "cluster_id": cluster_id,
        "cluster_archetype": cluster_archetype,
        "recommendations": recommendations,
    }
