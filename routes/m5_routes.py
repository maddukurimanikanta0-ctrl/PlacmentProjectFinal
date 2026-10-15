"""
routes/m5_routes.py
-------------------
Module 5: Model Evaluation, Selection, and Calibration.
"""

from flask import Blueprint, render_template, request
from src.Validation.Hold_out import run_holdout_validation
from src.Validation.K_flod import run_kfold_validation
from src.Validation.calibration import run_50_sample_calibration
from ml.evaluation_advanced import (
    run_threeway_split_audit,
    compute_comprehensive_metrics,
    compute_regression_metrics_comparison,
    compute_learning_curve_diagnosis
)

m5_bp = Blueprint("m5", __name__)


@m5_bp.route("/model-validation", methods=["GET", "POST"])
@m5_bp.route("/m5/calibration", endpoint="calibration", methods=["GET", "POST"])
@m5_bp.route("/validation", methods=["GET", "POST"])
def validation():
    """Renders Hold-out validation, K-Fold cross validation, and 50-sample calibration lab."""
    n_samples = 50
    k_value = 5
    model_type = "logistic_regression"

    if request.method == "POST":
        try:
            n_samples = int(request.form.get("n_samples", 50))
        except (ValueError, TypeError):
            n_samples = 50
        try:
            k_value = int(request.form.get("k_value", 5))
        except (ValueError, TypeError):
            k_value = 5
        model_type = request.form.get("model_type", "logistic_regression")

    try:
        holdout_res = run_holdout_validation(
            n_samples=n_samples,
            test_size=0.20,
            random_state=42,
            model_type=model_type,
        )
        kfold_res = run_kfold_validation(
            n_samples=n_samples,
            k=k_value,
            random_state=42,
            stratified=True,
            model_type=model_type,
        )
        calib_50 = run_50_sample_calibration(random_state=42, model_type=model_type)

        return render_template(
            "validation.html",
            holdout=holdout_res,
            kfold=kfold_res,
            calib_50=calib_50,
            n_samples=n_samples,
            k_value=k_value,
            model_type=model_type,
            active_module="m5",
            error=None,
        )
    except Exception as e:
        return render_template(
            "validation.html",
            holdout={},
            kfold={},
            calib_50={},
            n_samples=n_samples,
            k_value=k_value,
            model_type=model_type,
            active_module="m5",
            error=str(e),
        )


@m5_bp.route("/m5/holdout", methods=["GET"])
def holdout():
    """Renders M5 3-Way Train/Val/Test Split Discipline & Leak Prevention Audit."""
    try:
        split_data = run_threeway_split_audit()
        return render_template("m5/holdout.html", split_data=split_data, active_module="m5")
    except Exception as e:
        return render_template("m5/holdout.html", split_data={"splits": {}}, active_module="m5", error=str(e))


@m5_bp.route("/m5/kfold", methods=["GET"])
def kfold():
    """Renders M5 Stratified K-Fold and Nested Cross Validation."""
    k_value = request.args.get("k", default=5, type=int)
    try:
        kfold_res = run_kfold_validation(n_samples=100, k=k_value, random_state=42, stratified=True)
        return render_template("m5/kfold.html", kfold=kfold_res, active_module="m5")
    except Exception as e:
        return render_template("m5/kfold.html", kfold={"folds": []}, active_module="m5", error=str(e))


@m5_bp.route("/m5/metrics", methods=["GET"])
def metrics():
    """Renders M5 Comprehensive Classification & Regression Metrics Suite."""
    threshold = request.args.get("threshold", default=0.5, type=float)
    try:
        metrics_data = compute_comprehensive_metrics(threshold=threshold)
        reg_metrics = compute_regression_metrics_comparison()
        return render_template("m5/metrics.html", metrics_data=metrics_data, reg_metrics=reg_metrics, threshold=threshold, active_module="m5")
    except Exception as e:
        return render_template("m5/metrics.html", metrics_data={"confusion_matrix": {}}, reg_metrics={}, threshold=threshold, active_module="m5", error=str(e))


@m5_bp.route("/m5/tuning", methods=["GET"])
def tuning():
    """Renders M5 Hyperparameter Tuning & Diagnostic Learning Curves."""
    try:
        curve_data = compute_learning_curve_diagnosis()
        return render_template("m5/tuning.html", curve_data=curve_data, active_module="m5")
    except Exception as e:
        return render_template("m5/tuning.html", curve_data={"curve_points": []}, active_module="m5", error=str(e))
