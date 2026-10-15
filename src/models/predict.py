"""
src/models/predict.py
---------------------
Compatibility proxy forwarding to ml.predict.
"""

from ml.predict import (
    get_model,
    get_salary_model,
    get_unsupervised_models,
    predict_student,
    CLUSTER_NAMES,
    MODEL_PATH_JOBLIB as MODEL_PATH,
)

__all__ = [
    "get_model",
    "get_salary_model",
    "get_unsupervised_models",
    "predict_student",
    "CLUSTER_NAMES",
    "MODEL_PATH",
]
