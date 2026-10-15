"""
tests/test_prediction.py
------------------------
Automated test suite verifying M1 to M6 machine learning pipelines and serving.
Strictly under 250 lines per file.
"""

import unittest
from pathlib import Path
import json

from ml.data_loader import load_data, clean_data, get_data_summary
from ml.preprocessing import prepare_single_student
from ml.predict import predict_student
from ml.linear_regression import simple_linear_regression, predict_salary
from ml.logistic_regression import predict_logistic
from ml.regularization import compare_regularization
from ml.softmax_regression import train_softmax_model, predict_softmax_tier
from ml.tree_models import train_decision_tree_analysis, train_random_forest_analysis
from ml.unsupervised_advanced import run_hierarchical_clustering, run_dbscan_clustering, run_pca_detailed
from ml.ml_engineering import simulate_feature_store, compute_psi_drift


class TestPlacementPredictSystem(unittest.TestCase):

    def test_01_data_loading(self):
        """Test dataset ingestion and cleaning via ml.data_loader."""
        df = load_data()
        self.assertGreater(len(df), 1000)
        self.assertIn("PlacementStatus", df.columns)
        cleaned = clean_data(df)
        self.assertEqual(int(cleaned.duplicated().sum()), 0)

    def test_02_feature_alignment(self):
        """Verify single student payload transforms without schema mismatch."""
        sample_input = {"CGPA": "8.2", "CodingTestScore": "80", "Stream": "CS"}
        df_single = prepare_single_student(sample_input)
        self.assertEqual(len(df_single), 1)
        self.assertEqual(float(df_single["CGPA"].iloc[0]), 8.2)

    def test_03_inference_pipeline(self):
        """Test prediction on strong and weak profiles."""
        strong = {
            "CGPA": "8.8", "AttendancePercent": "90", "Internships": "3",
            "Projects": "3", "Workshops": "2", "Certifications": "3",
            "AptitudeTestScore": "88", "CodingTestScore": "90",
            "SoftSkillsRating": "4.5", "MockInterviewScore": "85",
            "CollegeTier": "Tier1", "Stream": "CS", "HistoryOfBacklogs": "No",
        }
        res = predict_student(strong)
        self.assertIn(res["status"], ["Placed", "Not Placed"])
        self.assertGreaterEqual(res["probability_placed"], 50.0)

    def test_04_m2_linear_and_regularization(self):
        """Test M2 salary regressor and L1/L2 regularization."""
        df = clean_data(load_data())
        res = simple_linear_regression(df)
        self.assertIn("r2_score", res)
        reg_comp = compare_regularization(alpha=1.0)
        self.assertIn("Lasso (L1)", reg_comp["comparison"])
        self.assertGreaterEqual(reg_comp["comparison"]["Lasso (L1)"]["zero_features_count"], 0)

    def test_05_m2_softmax_regression(self):
        """Test M2 Softmax multinomial regression for multi-class tiers."""
        sm_model = train_softmax_model()
        self.assertEqual(len(sm_model["classes"]), 3)
        tier_pred = predict_softmax_tier({"CGPA": 8.5, "CodingTestScore": 85})
        self.assertIn("predicted_tier", tier_pred)
        self.assertEqual(len(tier_pred["probabilities"]), 3)

    def test_06_m3_tree_models(self):
        """Test M3 Decision Tree pruning and Random Forest bagging."""
        dt = train_decision_tree_analysis(max_depth=4)
        self.assertIn("test_accuracy", dt)
        self.assertGreater(len(dt["pruning_path"]), 0)

        rf = train_random_forest_analysis(n_estimators=30)
        self.assertIn("oob_score", rf)
        self.assertGreater(rf["oob_score"], 0.8)

    def test_07_m4_unsupervised_advanced(self):
        """Test M4 Hierarchical clustering, DBSCAN and PCA decomposition."""
        hier = run_hierarchical_clustering(n_clusters=3)
        self.assertEqual(len(hier["cluster_profiles"]), 3)

        db = run_dbscan_clustering(eps=2.0, min_samples=5)
        self.assertIn("noise_points_count", db)

        pca = run_pca_detailed()
        self.assertEqual(len(pca["variance_per_component"]), 5)

    def test_08_m5_validation_and_calibration(self):
        """Test M5 holdout, stratified k-fold, and 50-sample calibration."""
        from src.Validation.Hold_out import run_holdout_validation
        from src.Validation.K_flod import run_kfold_validation
        from src.Validation.calibration import run_50_sample_calibration

        ho = run_holdout_validation(n_samples=50, test_size=0.20, random_state=42)
        self.assertEqual(ho["total_samples"], 50)

        kf = run_kfold_validation(n_samples=50, k=5, random_state=42)
        self.assertEqual(len(kf["folds"]), 5)

        cal = run_50_sample_calibration(random_state=42)
        self.assertEqual(cal["sample_size"], 50)
        self.assertIn("brier_improvement_pct", cal)

    def test_09_m6_ml_engineering(self):
        """Test M6 Feast Feature Store retrieval and PSI drift monitoring."""
        fs = simulate_feature_store(student_id=1001)
        self.assertIn("online_store", fs)
        self.assertGreater(len(fs["features"]), 0)

        drift = compute_psi_drift(shift_factor=0.0)
        self.assertIn("psi_score", drift)
        self.assertIn("alert_level", drift)

    def test_10_post_endpoints(self):
        """Test all interactive POST endpoints on Flask client."""
        from app import app
        client = app.test_client()

        form_payload = {
            "CGPA": "8.5", "AttendancePercent": "90", "Internships": "2",
            "Projects": "2", "Workshops": "1", "Certifications": "2",
            "AptitudeTestScore": "80", "CodingTestScore": "85",
            "SoftSkillsRating": "4.0", "MockInterviewScore": "80",
            "CollegeTier": "Tier1", "Stream": "CS", "HistoryOfBacklogs": "No"
        }
        res = client.post("/predict", data=form_payload)
        self.assertEqual(res.status_code, 200)

        res_api = client.post("/api/predict", json={"CGPA": 8.5, "CodingTestScore": 85})
        self.assertEqual(res_api.status_code, 200)
        self.assertEqual(res_api.get_json()["status"], "success")


if __name__ == "__main__":
    unittest.main()
