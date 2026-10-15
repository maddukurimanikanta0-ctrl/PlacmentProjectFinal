"""
routes/m3_routes.py
-------------------
Module 3: Supervised Tree-Based Models & Ensembles (Decision Trees, RF, Boosting, SHAP).
"""

from pathlib import Path
import json
from flask import Blueprint, render_template, request
from ml.data_loader import get_dataset_insights
from ml.tree_models import (
    train_decision_tree_analysis,
    train_random_forest_analysis,
    compare_boosting_algorithms,
    get_permutation_importance_summary
)

m3_bp = Blueprint("m3", __name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"


def _get_model_metrics():
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


@m3_bp.route("/m3/tree-models")
@m3_bp.route("/tree_models")
@m3_bp.route("/tree-models")
def tree_models():
    """Renders M3 Supervised Learning - Tree-Based Models, Ensembles, and Benchmarks."""
    try:
        insights = get_dataset_insights()
    except Exception:
        insights = {}
    return render_template(
        "tree_models.html",
        metrics=_get_model_metrics(),
        insights=insights,
        active_module="m3"
    )


@m3_bp.route("/m3/decision-tree", methods=["GET"])
def decision_tree():
    """Renders M3 Decision Tree with Cost-Complexity Pruning analysis."""
    max_depth = request.args.get("max_depth", default=5, type=int)
    ccp_alpha = request.args.get("ccp_alpha", default=0.0, type=float)
    try:
        dt_res = train_decision_tree_analysis(max_depth=max_depth, ccp_alpha=ccp_alpha)
        return render_template("m3/decision_tree.html", dt_res=dt_res, max_depth=max_depth, ccp_alpha=ccp_alpha, active_module="m3")
    except Exception as e:
        return render_template("m3/decision_tree.html", dt_res={"pruning_path": []}, max_depth=max_depth, ccp_alpha=ccp_alpha, active_module="m3", error=str(e))


@m3_bp.route("/m3/random-forest", methods=["GET"])
def random_forest():
    """Renders M3 Random Forest with Bagging, OOB Error, and Variance Reduction."""
    n_trees = request.args.get("n_trees", default=100, type=int)
    max_features = request.args.get("max_features", default="sqrt", type=str)
    try:
        rf_res = train_random_forest_analysis(n_estimators=n_trees, max_features=max_features)
        return render_template("m3/random_forest.html", rf_res=rf_res, n_trees=n_trees, max_features=max_features, active_module="m3")
    except Exception as e:
        return render_template("m3/random_forest.html", rf_res={}, n_trees=n_trees, max_features=max_features, active_module="m3", error=str(e))


@m3_bp.route("/m3/gradient-boosting", methods=["GET"])
def gradient_boosting():
    """Renders M3 Gradient Boosting vs LightGBM-style HistGradientBoosting comparison."""
    try:
        boost_data = compare_boosting_algorithms()
        return render_template("m3/gradient_boosting.html", boost_data=boost_data, active_module="m3")
    except Exception as e:
        return render_template("m3/gradient_boosting.html", boost_data={}, active_module="m3", error=str(e))


@m3_bp.route("/m3/feature-importance", methods=["GET"])
def feature_importance():
    """Renders M3 Feature Importance: Gini MDI vs Permutation Importance & SHAP."""
    try:
        perm_data = get_permutation_importance_summary()
        return render_template("m3/feature_importance.html", perm_data=perm_data, active_module="m3")
    except Exception as e:
        return render_template("m3/feature_importance.html", perm_data={"rankings": []}, active_module="m3", error=str(e))
