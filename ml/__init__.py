"""
ml package
----------
Unified Classical Machine Learning & ML Engineering Modules (M1–M6):
  - M1: data_loader, eda, preprocessing, predict
  - M2: linear_regression, logistic_regression, regularization, softmax_regression
  - M3: tree_models
  - M4: cluster_pca, unsupervised_advanced
  - M5: evaluation_advanced
  - M6: ml_engineering
"""

from ml.data_loader import load_data, clean_data, get_data_summary, get_dataset_insights
from ml.eda import get_summary_stats, generate_charts
from ml.preprocessing import build_preprocessor, normalization, prepare_features, prepare_single_student
from ml.linear_regression import simple_linear_regression, train_salary_regressor, predict_salary
from ml.logistic_regression import train_logistic_regression, predict_logistic
from ml.predict import predict_student
from ml.regularization import compare_regularization, get_coefficient_paths
from ml.softmax_regression import train_softmax_model, predict_softmax_tier
from ml.tree_models import train_decision_tree_analysis, train_random_forest_analysis, compare_boosting_algorithms
from ml.unsupervised_advanced import run_hierarchical_clustering, run_dbscan_clustering, run_pca_detailed, run_anomaly_detection
from ml.evaluation_advanced import run_threeway_split_audit, compute_comprehensive_metrics
from ml.ml_engineering import simulate_feature_store, detect_training_serving_skew, generate_model_card_and_packaging, compute_psi_drift

__all__ = [
    "load_data", "clean_data", "get_data_summary", "get_dataset_insights",
    "get_summary_stats", "generate_charts", "build_preprocessor", "normalization",
    "prepare_features", "prepare_single_student", "simple_linear_regression",
    "train_salary_regressor", "predict_salary", "train_logistic_regression",
    "predict_logistic", "predict_student", "compare_regularization",
    "train_softmax_model", "predict_softmax_tier", "train_decision_tree_analysis",
    "train_random_forest_analysis", "compare_boosting_algorithms",
    "run_hierarchical_clustering", "run_dbscan_clustering", "run_pca_detailed",
    "run_anomaly_detection", "run_threeway_split_audit", "compute_comprehensive_metrics",
    "simulate_feature_store", "detect_training_serving_skew", "generate_model_card_and_packaging",
    "compute_psi_drift"
]
