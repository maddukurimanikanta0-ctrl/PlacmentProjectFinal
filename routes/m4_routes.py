"""
routes/m4_routes.py
-------------------
Module 4: Unsupervised Learning (K-Means, Hierarchical, DBSCAN, PCA, Anomaly Detection).
"""

from pathlib import Path
import json
from flask import Blueprint, render_template, request
from ml.data_loader import get_dataset_insights
from ml.unsupervised_advanced import (
    run_hierarchical_clustering,
    run_dbscan_clustering,
    run_pca_detailed,
    run_anomaly_detection
)

m4_bp = Blueprint("m4", __name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
UNSUPERVISED_PATH = MODELS_DIR / "unsupervised_summary.json"


def _get_unsupervised_metrics():
    if UNSUPERVISED_PATH.exists():
        try:
            with open(UNSUPERVISED_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


@m4_bp.route("/clustering")
@m4_bp.route("/m4/kmeans", endpoint="kmeans")
@m4_bp.route("/unsupervised")
def unsupervised():
    """Renders M4 Unsupervised Learning - K-Means, Student Archetypes & PCA Analysis."""
    try:
        insights = get_dataset_insights()
    except Exception:
        insights = {}
    return render_template(
        "unsupervised.html",
        unsupervised=_get_unsupervised_metrics(),
        insights=insights,
        active_module="m4"
    )


@m4_bp.route("/m4/hierarchical", methods=["GET"])
def hierarchical():
    """Renders M4 Hierarchical Agglomerative Clustering and Linkages."""
    n_clusters = request.args.get("n_clusters", default=4, type=int)
    linkage = request.args.get("linkage", default="ward", type=str)
    try:
        hier_data = run_hierarchical_clustering(n_clusters=n_clusters, linkage_method=linkage)
        return render_template("m4/hierarchical.html", hier_data=hier_data, n_clusters=n_clusters, linkage=linkage, active_module="m4")
    except Exception as e:
        return render_template("m4/hierarchical.html", hier_data={"cluster_profiles": []}, n_clusters=n_clusters, linkage=linkage, active_module="m4", error=str(e))


@m4_bp.route("/m4/dbscan", methods=["GET"])
def dbscan():
    """Renders M4 DBSCAN Density Clustering and Outlier/Noise Detection."""
    eps = request.args.get("eps", default=1.8, type=float)
    min_samples = request.args.get("min_samples", default=5, type=int)
    try:
        db_data = run_dbscan_clustering(eps=eps, min_samples=min_samples)
        return render_template("m4/dbscan.html", db_data=db_data, eps=eps, min_samples=min_samples, active_module="m4")
    except Exception as e:
        return render_template("m4/dbscan.html", db_data={"clusters_summary": []}, eps=eps, min_samples=min_samples, active_module="m4", error=str(e))


@m4_bp.route("/m4/pca", methods=["GET"])
def pca():
    """Renders M4 PCA Dimensionality Reduction and Explained Variance Breakdown."""
    try:
        pca_data = run_pca_detailed()
        return render_template("m4/pca.html", pca_data=pca_data, active_module="m4")
    except Exception as e:
        return render_template("m4/pca.html", pca_data={"variance_per_component": []}, active_module="m4", error=str(e))


@m4_bp.route("/m4/anomaly-detection", methods=["GET"])
def anomaly_detection():
    """Renders M4 Isolation Forest Anomaly Detection and Student Outlier Profiling."""
    try:
        ano_data = run_anomaly_detection()
        return render_template("m4/anomaly.html", ano_data=ano_data, active_module="m4")
    except Exception as e:
        return render_template("m4/anomaly.html", ano_data={"sample_anomalies": []}, active_module="m4", error=str(e))
