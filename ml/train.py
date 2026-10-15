"""
ml/train.py
-----------
Full training and artifact generation pipeline:
  - Linear Regression (Salary Prediction demo)
  - Logistic Regression (Ridge L2)
  - Decision Tree, Random Forest, and Gradient Boosting
  - Feature importances and coefficient comparison
  - Packaged model artifacts saved as both .joblib and .pkl in models/ directory
"""

from pathlib import Path
import json
import joblib
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CHARTS_DIR = PROJECT_ROOT / "static" / "charts"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

from ml.data_loader import load_data, clean_data
from ml.preprocessing import prepare_features, build_preprocessor
from ml.evaluate import evaluate_model
from ml.linear_regression import train_salary_regressor


def save_artifact(obj, name: str):
    """Saves model in both .joblib and .pkl formats."""
    joblib.dump(obj, MODELS_DIR / f"{name}.joblib")
    with open(MODELS_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


def plot_feature_importance(rf_model, feature_names):
    """Generates feature importance bar chart for Random Forest."""
    importances = rf_model.named_steps["clf"].feature_importances_
    feat_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=feat_df,
        x="importance",
        y="feature",
        palette="Blues_r",
        ax=ax
    )
    ax.set_title("Top Feature Importances (Gini/MDI)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Relative Importance")
    ax.set_ylabel("Student Feature")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "feature_importance.png", dpi=130)
    plt.close(fig)


def main():
    print("Loading and cleaning dataset...")
    raw_df = load_data()
    df = clean_data(raw_df)

    # Train Linear Regression salary model
    print("Training Linear Regression salary model...")
    train_salary_regressor(df)

    X, y = prepare_features(df)
    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
        ]),
        "decision_tree": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", DecisionTreeClassifier(max_depth=7, random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)),
        ]),
        "gradient_boosting": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", HistGradientBoostingClassifier(max_iter=100, random_state=42)),
        ]),
    }

    metrics_results = {}
    fitted_pipelines = {}

    print("\nTraining classifiers...")
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        metrics_results[name] = metrics
        fitted_pipelines[name] = pipeline
        save_artifact(pipeline, name)
        print(f"  -> {name}: Accuracy={metrics['accuracy']} | Precision={metrics['precision']} | Recall={metrics['recall']} | ROC-AUC={metrics['roc_auc']}")

    # Feature names for importance
    fitted_prep = fitted_pipelines["random_forest"].named_steps["preprocessor"]
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    cat_feature_names = fitted_prep.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + cat_feature_names

    try:
        plot_feature_importance(fitted_pipelines["random_forest"], all_feature_names)
    except Exception as e:
        print(f"Warning plotting feature importance: {e}")

    # Top Logistic Regression coefficients
    lr_clf = fitted_pipelines["logistic_regression"].named_steps["clf"]
    coefs = lr_clf.coef_[0]
    coef_df = pd.DataFrame({"feature": all_feature_names, "coefficient": coefs}).sort_values("coefficient", ascending=False)
    top_pos = coef_df.head(5).to_dict(orient="records")
    top_neg = coef_df.tail(5).to_dict(orient="records")

    metrics_results["linear_coefficients"] = {
        "top_positive": top_pos,
        "top_negative": top_neg,
        "intercept": float(lr_clf.intercept_[0]),
    }

    # Save metrics JSON
    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_results, f, indent=2)

    # Primary Production Serving Model
    print("\nRefitting production serving model on full dataset...")
    full_production_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("clf", HistGradientBoostingClassifier(max_iter=120, random_state=42)),
    ])
    full_production_pipeline.fit(X, y)
    save_artifact(full_production_pipeline, "model")
    print("Production model saved to models/model.joblib and models/model.pkl")

    print("\nTraining workflow completed successfully.")


if __name__ == "__main__":
    main()
