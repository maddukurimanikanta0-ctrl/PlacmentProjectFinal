"""
src/data/load_data.py
---------------------
Compatibility proxy forwarding to ml.data_loader and ml.eda.
"""

from ml.data_loader import (
    load_data,
    load_dataset,
    clean_data,
    clean_dataset,
    get_dataset_insights,
    get_data_summary,
    get_feature_target_split,
    TARGET_COL,
    TARGET,
    ID_COLS,
    ID_COL,
    NUMERIC_COLS,
    NUMERIC_WITH_NULLS,
    CATEGORICAL_COLS,
    SGPA_COLS,
    PROJECT_ROOT,
    PRIMARY_DATA_PATH as DATASET_PATH,
    CHARTS_DIR,
)
from ml.eda import generate_charts, generate_eda_charts

__all__ = [
    "load_data",
    "load_dataset",
    "clean_data",
    "clean_dataset",
    "get_dataset_insights",
    "get_data_summary",
    "get_feature_target_split",
    "generate_charts",
    "generate_eda_charts",
    "TARGET",
    "TARGET_COL",
    "ID_COL",
    "ID_COLS",
    "NUMERIC_COLS",
    "NUMERIC_WITH_NULLS",
    "CATEGORICAL_COLS",
    "SGPA_COLS",
    "PROJECT_ROOT",
    "DATASET_PATH",
    "CHARTS_DIR",
]
