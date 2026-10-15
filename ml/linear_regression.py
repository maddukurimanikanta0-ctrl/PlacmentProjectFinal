"""
ml/linear_regression.py
-----------------------
Linear Regression module for salary package prediction and regression analysis.
Trains and evaluates Ordinary Least Squares Linear Regression.
Anchored dynamically to project root for saving/loading artifacts.
"""

from pathlib import Path
import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH_JOBLIB = MODELS_DIR / "salary_linear_model.joblib"
MODEL_PATH_PKL = MODELS_DIR / "salary_linear_model.pkl"


def simple_linear_regression(df: pd.DataFrame) -> dict:
    """
    Fits simple linear regression (CGPA -> Salary Package)
    matching academic exercise requirements.
    """
    salary_col = "Salary Package" if "Salary Package" in df.columns else "SalaryPackage"
    clean = df.dropna(subset=["CGPA", salary_col])
    # Filter for placed students with salary > 0
    clean = clean[clean[salary_col] > 0]
    
    X = clean[["CGPA"]]
    y = clean[salary_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    mse = float(mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    return {
        "features": ["CGPA"],
        "slope": float(lr.coef_[0]),
        "intercept": float(lr.intercept_),
        "mse": round(mse, 4),
        "r2_score": round(r2, 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }


def train_salary_regressor(df: pd.DataFrame = None):
    """
    Trains multiple linear regression model predicting salary package
    from CGPA, CodingTestScore, and AptitudeTestScore.
    Saves model to both .joblib and .pkl in models/.
    """
    if df is None:
        from ml.data_loader import load_data, clean_data
        df = clean_data(load_data())

    salary_col = "Salary Package" if "Salary Package" in df.columns else "SalaryPackage"
    placed_df = df[(df["PlacementStatus"] == 1) & (df[salary_col] > 0)].copy() if "PlacementStatus" in df.columns else df[df[salary_col] > 0].copy()

    features = ["CGPA", "CodingTestScore", "AptitudeTestScore"]
    clean_sub = placed_df.dropna(subset=features + [salary_col])
    X = clean_sub[features]
    y = clean_sub[salary_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Save joblib
    joblib.dump(model, MODEL_PATH_JOBLIB)
    # Save pkl
    with open(MODEL_PATH_PKL, "wb") as f:
        pickle.dump(model, f)

    return {
        "features": features,
        "coefficients": {feat: round(float(coef), 4) for feat, coef in zip(features, model.coef_)},
        "intercept": round(float(model.intercept_), 4),
        "mse": round(mse, 4),
        "mae": round(mae, 4),
        "r2_score": round(r2, 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }


def load_salary_model():
    """Dynamically loads trained linear regression model from models/."""
    if MODEL_PATH_JOBLIB.exists():
        return joblib.load(MODEL_PATH_JOBLIB)
    if MODEL_PATH_PKL.exists():
        with open(MODEL_PATH_PKL, "rb") as f:
            return pickle.load(f)
    return None


def predict_salary(cgpa: float, coding_score: float = 70.0, aptitude_score: float = 70.0) -> float:
    """Predicts expected salary package in LPA for a placed candidate."""
    model = load_salary_model()
    if model is None:
        # Fallback estimation heuristic
        return round(3.5 + 0.6 * max(0.0, cgpa - 5.0) + 0.03 * coding_score, 2)
    
    features = ["CGPA", "CodingTestScore", "AptitudeTestScore"]
    reg_df = pd.DataFrame([[float(cgpa), float(coding_score), float(aptitude_score)]], columns=features)
    pred = float(model.predict(reg_df)[0])
    return max(3.0, round(pred, 2))


if __name__ == "__main__":
    from ml.data_loader import load_data, clean_data
    df = clean_data(load_data())
    print("Fitting Simple Linear Regression...")
    simple_res = simple_linear_regression(df)
    print("Simple LR Results:", simple_res)

    print("\nFitting Salary Multiple Regressor...")
    multi_res = train_salary_regressor(df)
    print("Multi LR Results:", multi_res)
