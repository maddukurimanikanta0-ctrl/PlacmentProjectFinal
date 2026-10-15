"""
src/unsupervised/cluster_pca.py
-------------------------------
Compatibility proxy forwarding to ml.cluster_pca.
"""

from ml.cluster_pca import (
    run_unsupervised_pipeline,
    UNSUPERVISED_FEATURES,
    CLUSTER_NAMES,
    CLUSTER_COLORS,
)

if __name__ == "__main__":
    run_unsupervised_pipeline()


