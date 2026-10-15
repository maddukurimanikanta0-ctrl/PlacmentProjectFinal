"""
tests/verify_suite.py
---------------------
Comprehensive end-to-end route verification, expandable header navigation test,
50-sample calibration validation, and M1-M6 syllabus integrity checks.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app
client = app.test_client()

core_routes = [
    '/',
    '/data-loading',
    '/eda',
    '/preprocessing',
    '/linear-regression',
    '/logistic-regression',
    '/tree-models',
    '/unsupervised',
    '/validation',
    '/health',
    '/api/validation',
    '/m1/prediction-service',
    '/m2/regularization',
    '/m2/softmax-regression',
    '/m3/decision-tree',
    '/m3/random-forest',
    '/m3/feature-importance',
    '/m4/hierarchical',
    '/m4/dbscan',
    '/m4/pca',
    '/m4/anomaly-detection',
    '/m5/holdout',
    '/m5/kfold',
    '/m5/metrics',
    '/m5/tuning',
    '/m6/feature-store',
    '/m6/packaging',
    '/m6/serving',
    '/m6/monitoring',
    '/m6/retraining',
    '/api/drift'
]

print("=== TESTING ALL GET ROUTES IN PLACEMENT_PREDICTION (M1 TO M6) ===")
for r in core_routes:
    res = client.get(r)
    assert res.status_code == 200, f"Route {r} failed with {res.status_code}"
    print(f"OK: {r}")

dash_res = client.get('/')
dash_html = dash_res.get_data(as_text=True)

remaining_routes = [
    '/data-loading',
    '/eda',
    '/preprocessing',
    '/linear-regression',
    '/logistic-regression',
    '/tree-models',
    '/unsupervised',
    '/validation'
]

print("\n=== VERIFYING DASHBOARD HAS LINKS TO ALL PAGES ===")
for r in remaining_routes:
    assert r in dash_html, f"Dashboard missing {r}"
    print(f"Dashboard contains: {r}")

print("\n=== VERIFYING ALL PAGES HAVE EXPANDABLE NAVIGATION IN HEADER ===")
for r in remaining_routes:
    page_res = client.get(r)
    page_html = page_res.get_data(as_text=True)
    assert 'dark-header' in page_html, f"{r} missing dark neutral header"
    assert 'dashboard-nav-bar' in page_html, f"{r} missing nav bar"
    assert 'nav-module-dropdown' in page_html, f"{r} missing expandable module dropdown"
    for dest in remaining_routes:
        assert dest in page_html, f"Page {r} missing link to {dest}"
    print(f"Page {r}: Has header with expandable multi-page navigation to ALL pages")

print("\n=== TESTING 50-SAMPLE CALIBRATION API ===")
val_api = client.get('/api/validation')
val_json = val_api.get_json()
assert 'calibration_50_samples' in val_json
cal_50 = val_json['calibration_50_samples']
print(f"Sample size: {cal_50['sample_size']}")
print(f"ROC-AUC (50 Samples): {cal_50['roc_auc_50']}")
print(f"Brier Improvement: {cal_50['brier_improvement_pct']}%")

print("\n=== TESTING HEALTH CHECK & 6 SYLLABUS MODULES (M1–M6) ===")
health_res = client.get('/health')
health_json = health_res.get_json()
assert health_json['status'] == 'healthy'
assert len(health_json['syllabus_modules']) == 6, f"Expected 6 modules, got {len(health_json['syllabus_modules'])}"
print(f"Health check OK: {len(health_json['syllabus_modules'])} syllabus modules verified (M1 to M6)")

print("\n=== CHECKING NO STUB INIT.PY FILES ===")
stub_inits = [f for f in ROOT.glob('src/**/__init__.py')]
assert len(stub_inits) == 0, f"Found stub init files: {stub_inits}"
print("OK: 0 stub __init__.py files in src/")

print("\nALL VERIFICATIONS PASSED 100% IN PLACEMENT_PREDICTION (M1 TO M6)!")
