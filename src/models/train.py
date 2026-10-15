"""
src/models/train.py
-------------------
Compatibility proxy forwarding to ml.train.
"""

from ml.train import (
    main,
    train_salary_regressor,
    plot_feature_importance,
    save_artifact,
)

if __name__ == "__main__":
    main()
