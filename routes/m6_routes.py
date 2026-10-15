"""
routes/m6_routes.py
-------------------
Module 6: ML Engineering — From Notebook to Production (Feature Store, ONNX, Serving, Drift, Retraining).
"""

from flask import Blueprint, render_template, request
from ml.ml_engineering import (
    simulate_feature_store,
    detect_training_serving_skew,
    generate_model_card_and_packaging,
    compute_psi_drift,
    get_production_telemetry
)

m6_bp = Blueprint("m6", __name__)


@m6_bp.route("/m6/feature-store", methods=["GET"])
def feature_store():
    """Renders M6 Feast Feature Store architecture and online-offline simulation."""
    student_id = request.args.get("student_id", default=1001, type=int)
    try:
        fs_data = simulate_feature_store(student_id=student_id)
        return render_template("m6/feature_store.html", fs_data=fs_data, active_module="m6")
    except Exception as e:
        return render_template("m6/feature_store.html", fs_data={"features": {}}, active_module="m6", error=str(e))


@m6_bp.route("/m6/packaging", methods=["GET"])
def packaging():
    """Renders M6 Model Packaging, Serialization (ONNX/Joblib), and Model Card."""
    try:
        card_data = generate_model_card_and_packaging()
        return render_template("m6/packaging.html", card_data=card_data, active_module="m6")
    except Exception as e:
        return render_template("m6/packaging.html", card_data={"serialization_formats": [], "training_metadata": {}}, active_module="m6", error=str(e))


@m6_bp.route("/m6/serving", methods=["GET"])
def serving():
    """Renders M6 Production Serving Frameworks, FastAPI vs Flask, and Latency Telemetry."""
    try:
        telemetry = get_production_telemetry()
        return render_template("m6/serving.html", telemetry=telemetry, active_module="m6")
    except Exception as e:
        return render_template("m6/serving.html", telemetry={"latency_percentiles_ms": {}}, active_module="m6", error=str(e))


@m6_bp.route("/m6/monitoring", methods=["GET"])
def monitoring():
    """Renders M6 Production Drift Monitoring & Population Stability Index (PSI)."""
    shift = request.args.get("shift", default=0.0, type=float)
    try:
        psi_data = compute_psi_drift(shift_factor=shift)
        return render_template("m6/monitoring.html", psi_data=psi_data, shift=shift, active_module="m6")
    except Exception as e:
        return render_template("m6/monitoring.html", psi_data={"quantiles_breakdown": []}, shift=shift, active_module="m6", error=str(e))


@m6_bp.route("/m6/retraining", methods=["GET"])
def retraining():
    """Renders M6 Automated Retraining Strategies & MLOps Lifecycle."""
    return render_template("m6/retraining.html", active_module="m6")
