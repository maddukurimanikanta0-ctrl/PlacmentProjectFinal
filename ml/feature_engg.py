"""
ml/feature_engg.py
------------------
Maintained for backwards compatibility.
Forwards to ml.preprocessing and ml.linear_regression.
"""

from ml.preprocessing import normalization
from ml.linear_regression import simple_linear_regression

__all__ = ["normalization", "simple_linear_regression"]