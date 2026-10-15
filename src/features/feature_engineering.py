"""
src/features/feature_engineering.py
-----------------------------------
Compatibility proxy forwarding to ml.preprocessing.
"""

from ml.preprocessing import (
    prepare_features,
    prepare_single_student,
    DROP_COLUMNS,
    TARGET,
    NUMERIC_FIELDS,
    CATEGORICAL_FIELDS,
    DEFAULT_VALUES,
)

__all__ = [
    "prepare_features",
    "prepare_single_student",
    "DROP_COLUMNS",
    "TARGET",
    "NUMERIC_FIELDS",
    "CATEGORICAL_FIELDS",
    "DEFAULT_VALUES",
]
