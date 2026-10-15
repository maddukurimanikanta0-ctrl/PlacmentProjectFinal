# PlacementPredict — Machine Learning System (M1–M6)

**Course**: 25SC2107E — Machine Learning  
**Project Scope**: Modules M1 to M6 (Complete ML System Lifecycle)  
**Domain**: Student Placement, Career Package & Archetype Prediction  
**Dataset**: `data/placement_predict_50k_Dataset1.csv` (50,000 student records, 32 attributes)  
**Code Quality**: 100% of source files strictly contain **< 300 lines of code** (94 files audited).

---

## 📚 Complete Syllabus Modules (M1–M6)

### M1. PlacementPredict as a System — The ML Lifecycle
* **End-to-End System**: Raw Data $\rightarrow$ Cleaned Dataset $\rightarrow$ Feature Pipeline $\rightarrow$ Model Registry $\rightarrow$ Serving Framework (Flask) $\rightarrow$ Health Check (`/health`) $\rightarrow$ Retraining Loop.
* **Elimination of Training-Serving Skew**: Unified `ColumnTransformer` handles missing value median imputation, standard scaling, and one-hot encoding.
* **Live Serving & Tracing**: Interactive Web UI (`/predict`), JSON REST API (`/api/predict`), and dedicated lifecycle workbench (`/m1/prediction-service`).

### M2. Supervised Learning — Linear Models at Depth
* **Linear Regression (Salary)**: Normal equations vs Gradient Descent estimating projected starting salary package (LPA) from CGPA and assessment scores ($R^2 = 0.613$).
* **Logistic Regression with Ridge (L2)**: Hyperplane decision boundary with sigmoid activation, odds ratios, and log-loss optimization for binary classification (`/logistic-regression`).
* **Regularization Lab**: Ridge ($L_2$ sphere), Lasso ($L_1$ diamond sparsity), and Elastic Net coefficient shrinkage comparison (`/m2/regularization`).
* **Softmax / Multinomial Regression**: Multi-class classification predicting placement package tiers (*Dream*, *Core*, *Foundational*) (`/m2/softmax-regression`).

### M3. Supervised Learning — Tree-Based Models
* **Decision Trees & Pruning**: CART Gini impurity and cost-complexity post-pruning ($\alpha$) (`/m3/decision-tree`).
* **Random Forest (Bagging)**: Bootstrap aggregation, subspace feature sampling, and Out-of-Bag (OOB) error estimation (`/m3/random-forest`).
* **Gradient Boosting**: HistGradientBoosting/GBDT production tabular workhorse achieving **97.96% ROC-AUC** (`/tree-models`).
* **Feature Importance & SHAP**: Gini Mean Decrease Impurity (MDI) vs. Permutation importance ranking (`/m3/feature-importance`).

### M4. Unsupervised Learning — Clustering & Dimensionality Reduction
* **K-Means Clustering**: Lloyd's algorithm with $K=4$ clusters identified via elbow inertia and silhouette analysis ($s = 0.34$) (`/unsupervised`).
* **Hierarchical Agglomerative Clustering**: Bottom-up merging with Single, Complete, Average, and Ward linkages plus dendrogram visualizer (`/m4/hierarchical`).
* **DBSCAN Density-Based Clustering**: Non-convex cluster discovery with $\varepsilon$ and `min_samples`, isolating noise points (`/m4/dbscan`).
* **Principal Component Analysis (PCA)**: Standardized 2D projection capturing 71.38% total feature variance (`/m4/pca`).
* **Anomaly Detection**: Isolation Forest unsupervised outlier detection isolating atypical candidate profiles (`/m4/anomaly-detection`).

### M5. Model Evaluation, Selection, and Calibration
* **3-Way Split Discipline**: Strict 70/15/15 Train / Validation / Test separation eliminating data leakage (`/m5/holdout`).
* **Stratified K-Fold CV**: $K=5$ class-balanced cross-validation verifying model variance (`/m5/kfold`).
* **Complete Metrics Suite**: Confusion matrix, ROC curve, Precision-Recall curve, and business cost-utility threshold tuner (`/m5/metrics`).
* **50-Sample Calibration Lab**: Platt scaling (sigmoid) and Isotonic calibration evaluated on 50 held-out samples with Brier score reduction (`/validation`).
* **Hyperparameter Tuning & Diagnosis**: Grid / Random search with learning curves diagnosing bias vs. variance (`/m5/tuning`).

### M6. ML Engineering, Serving, and Production Operations
* **Feature Store (Feast Simulation)**: Online Redis low-latency cache and offline Parquet historical store eliminating skew (`/m6/feature-store`).
* **Model Packaging & Model Cards**: Versioned serialization with Joblib, ONNX runtime export, and production model cards (`/m6/packaging`).
* **Production Serving & Telemetry**: FastAPI/Flask endpoint simulation, p50/p95/p99 latency tracking, and throughput benchmarking (`/m6/serving`).
* **Data & Concept Drift Monitoring**: Population Stability Index (PSI) and Kolmogorov-Smirnov test detecting feature distribution shifts (`/m6/monitoring`, `/api/drift`).
* **Retraining Loop Trigger**: Automated triggers (Scheduled, Drift-detected, Performance degradation) with shadow deployment gates (`/m6/retraining`).

---

## 🧭 Expandable Header Navigation

The application features a modern, responsive navigation header across desktop, tablet, and mobile:
* **Dropdown Modules (M1–M6)**: Clicking any module button (`M1: Lifecycle`, `M2: Linear`, `M3: Trees`, `M4: Unsupervised`, `M5: Evaluation`, `M6: MLOps`) expands its dedicated models menu.
* **Dedicated Model Routing**: Every model links directly to its own specialized route and view.
* **Active Indicators**: Active modules and models are visually highlighted.
* **Click-Outside & Keyboard Escape**: Clean UI ergonomics with smooth CSS transitions.

---

## 🚀 Quick Start Guide

### 1. Activate Environment
In Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Run Comprehensive Test Suite
```powershell
# Run end-to-end route, header navigation, and calibration verifications
.\.venv\Scripts\python.exe tests/verify_suite.py

# Run unit tests
.\.venv\Scripts\python.exe -m unittest discover tests

# Check strict line length discipline (< 300 lines on all 94 files)
.\.venv\Scripts\python.exe tests/check_line_lengths.py
```

### 3. Start Flask Web Application
```powershell
.\.venv\Scripts\python.exe app.py
```
Open **http://127.0.0.1:5000** in your web browser.

---

## 🖨️ PDF Report Generation (Ctrl + P)
Pressing `Ctrl + P` on the web dashboard activates the print stylesheet (`static/css/print.css`):
* Unrolls tabs and reveals all model benchmarks and visualizations.
* Hides interactive controls and navigation buttons.
* Exports a complete academic PDF report ready for submission.

---

## 📁 Modular Project Structure

```
placement_prediction/
├── data/
│   └── placement_predict_50k_Dataset1.csv # 50k student records (32 attributes)
├── models/
│   ├── model.joblib                       # Production serving artifact
│   ├── logistic_regression.joblib        # M2 Ridge Logistic Regression
│   ├── decision_tree.joblib              # M3 Pruned Decision Tree
│   ├── random_forest.joblib              # M3 Random Forest
│   ├── gradient_boosting.joblib          # M3 Gradient Boosting
│   ├── salary_linear_model.joblib        # M2 Salary Linear Regressor
│   ├── kmeans_model.joblib               # M4 K-Means Clustering model
│   ├── pca_model.joblib                  # M4 PCA Dimensionality model
│   ├── metrics.json                      # Supervised benchmark metrics
│   └── unsupervised_summary.json         # M4 Clustering & PCA summary
├── ml/                                    # Modular Machine Learning Core
│   ├── data_loader.py                    # Dataset ingestion & summary
│   ├── eda.py                            # Exploratory data statistics & charts
│   ├── preprocessing.py                  # Imputation & standard scaling
│   ├── linear_regression.py              # Salary normal eq & gradient descent
│   ├── logistic_regression.py            # Logistic sigmoid & odds ratios
│   ├── regularization.py                 # M2 Ridge / Lasso / Elastic Net
│   ├── softmax_regression.py             # M2 Multinomial placement tiers
│   ├── tree_models.py                    # M3 CART, Random Forest, Boosting
│   ├── unsupervised_advanced.py          # M4 Hierarchical, DBSCAN, PCA, Anomaly
│   ├── evaluation_advanced.py            # M5 3-way split, ROC, Calibration, Tuning
│   ├── ml_engineering.py                 # M6 Feature store, Packaging, Drift, MLOps
│   └── predict.py                        # Real-time multi-model prediction pipeline
├── routes/                                # Modular Flask Blueprints
│   ├── m1_routes.py                      # M1 Lifecycle, Ingestion, EDA, Serving
│   ├── m2_routes.py                      # M2 Linear, Logistic, Regularization, Softmax
│   ├── m3_routes.py                      # M3 Decision Tree, RF, Boosting, Importance
│   ├── m4_routes.py                      # M4 K-Means, Hierarchical, DBSCAN, PCA
│   ├── m5_routes.py                      # M5 Holdout, KFold, Metrics, Calibration, Tuning
│   ├── m6_routes.py                      # M6 Feature Store, Packaging, Serving, Drift
│   └── api_routes.py                     # Health, Predict, Validation, Drift APIs
├── static/
│   ├── css/                              # Modular CSS stylesheets (< 300 lines each)
│   │   ├── variables.css                 # Colors, fonts, and shared tokens
│   │   ├── base.css                      # Global layout, resets, typography
│   │   ├── header_nav.css                # Expandable M1–M6 navigation menu
│   │   ├── components.css                # Badges, cards, alerts, forms, buttons
│   │   ├── dashboard.css                 # Tab panels and metric cards
│   │   ├── results_archetypes.css        # Prediction results & student archetypes
│   │   ├── print.css                     # Print media and PDF report formatting
│   │   └── style.css                     # Master stylesheet aggregating CSS modules
│   └── js/
│       └── script.js                     # Navigation toggles, tabs, and presets
├── templates/                             # Modular Jinja2 Templates (< 300 lines each)
│   ├── base.html                         # Expandable M1–M6 header & container
│   ├── index.html                        # Dashboard tab aggregator
│   ├── partials/                         # Component partials (tabs, KPIs, engine)
│   ├── m1/                               # M1 dedicated pages
│   ├── m2/                               # M2 dedicated pages
│   ├── m3/                               # M3 dedicated pages
│   ├── m4/                               # M4 dedicated pages
│   ├── m5/                               # M5 dedicated pages
│   └── m6/                               # M6 dedicated pages
├── tests/
│   ├── test_prediction.py                # 10 automated unit tests (M1 to M6)
│   ├── verify_suite.py                   # 31-route E2E verification test suite
│   └── check_line_lengths.py             # Strict < 300 lines per file auditor
├── app.py                                # Clean application factory
└── requirements.txt                      # Project dependencies
```
