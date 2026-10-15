"""
routes/m1_routes.py
-------------------
Module 1: The ML Lifecycle, Ingestion, Skew Prevention, EDA, and Live Serving.
"""

from pathlib import Path
import json
from flask import Blueprint, render_template, request, redirect, url_for
from ml.data_loader import load_data, clean_data, get_data_summary, get_dataset_insights
from ml.eda import get_summary_stats, generate_charts
from ml.preprocessing import normalization, CATEGORICAL_FIELDS
from ml.predict import predict_student

m1_bp = Blueprint("m1", __name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"
UNSUPERVISED_PATH = MODELS_DIR / "unsupervised_summary.json"


def _get_insights():
    try:
        return get_dataset_insights()
    except Exception:
        return {}


def _get_metrics():
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _get_unsupervised():
    if UNSUPERVISED_PATH.exists():
        try:
            with open(UNSUPERVISED_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


@m1_bp.route("/")
def index():
    """Renders comprehensive master dashboard covering M1 to M6."""
    return render_template(
        "index.html",
        insights=_get_insights(),
        metrics=_get_metrics(),
        unsupervised=_get_unsupervised(),
        active_module="m1"
    )


@m1_bp.route("/m1/prediction-service")
def prediction_service():
    """Renders dedicated M1 ML Lifecycle & Live Request Tracing Workbench."""
    return render_template("m1/lifecycle.html", insights=_get_insights(), active_module="m1")


@m1_bp.route("/predict", methods=["POST"])
def predict():
    """Handles web form submission, runs inference, and displays result."""
    try:
        form_data = request.form.to_dict()
        result = predict_student(form_data)
        return render_template("result.html", result=result, inputs=form_data, active_module="m1")
    except Exception as e:
        return render_template("result.html", result={"error": str(e)}, inputs={}, active_module="m1")


@m1_bp.route("/m1/data-pipeline")
@m1_bp.route("/data_loading")
@m1_bp.route("/data-loading")
def data_loading():
    """Renders data loading, schema inspection, and cleaning summary."""
    try:
        raw_df = load_data()
        cleaned_df = clean_data(raw_df)
        raw_summary = get_data_summary(raw_df)
        cleaned_summary = get_data_summary(cleaned_df)

        preview_cols = [
            "StudentID", "Gender", "CollegeTier", "Stream", "CGPA",
            "AttendancePercent", "Internships", "CodingTestScore",
            "AptitudeTestScore", "PlacementStatus"
        ]
        valid_cols = [c for c in preview_cols if c in cleaned_df.columns]
        preview_rows = cleaned_df[valid_cols].head(10).to_dict(orient="records")

        return render_template(
            "data_loading.html",
            raw_summary=raw_summary,
            cleaned_summary=cleaned_summary,
            preview_cols=valid_cols,
            preview_rows=preview_rows,
            active_module="m1",
            error=None
        )
    except Exception as e:
        return render_template("data_loading.html", error=str(e), active_module="m1")


@m1_bp.route("/m1/feature-pipeline")
@m1_bp.route("/preprocessing")
def preprocessing():
    """Renders cleaning, encoding, normalization, and feature transformations."""
    try:
        raw_df = load_data()
        cleaned_df = clean_data(raw_df)

        train_scaled, _ = normalization(cleaned_df)
        columns = list(train_scaled.columns)
        train_rows = train_scaled.head(10).values.tolist()

        cleaning_info = {
            "duplicates_removed": int(raw_df.duplicated().sum()),
            "imputed_cols": ["Workshops", "AptitudeTestScore", "SoftSkillsRating", "CodingTestScore", "MockInterviewScore"]
        }

        return render_template(
            "preprocessing.html",
            cleaning_info=cleaning_info,
            cat_features=CATEGORICAL_FIELDS,
            columns=columns,
            train_rows=train_rows,
            active_module="m1",
            error=None
        )
    except Exception as e:
        return render_template("preprocessing.html", error=str(e), active_module="m1")


@m1_bp.route("/m1/eda")
@m1_bp.route("/eda")
def eda():
    """Renders Exploratory Data Analysis statistics and saved charts."""
    try:
        cleaned_df = clean_data(load_data())
        stats = get_summary_stats(cleaned_df)
        chart_files = generate_charts(cleaned_df)
        return render_template("eda.html", stats=stats, chart_files=chart_files, active_module="m1", error=None)
    except Exception as e:
        return render_template("eda.html", error=str(e), stats={}, chart_files=[], active_module="m1")


@m1_bp.route("/load_page")
def load_page():
    return redirect(url_for("m1.data_loading"))


@m1_bp.route("/eda_page")
def eda_page():
    return redirect(url_for("m1.eda"))


@m1_bp.route("/feature_engg_page")
def feature_engg_page():
    return redirect(url_for("m1.preprocessing"))
