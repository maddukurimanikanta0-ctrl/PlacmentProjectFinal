"""
ml/cluster_pca.py
-----------------
Unsupervised Learning Module:
  - K-Means clustering (Lloyd's algorithm, K=4 student archetypes)
  - PCA (Principal Component Analysis) 2D projection with Explained Variance Ratio
  - Generates cluster visualization chart and unsupervised summary metrics.
  - Saves artifacts in both .joblib and .pkl formats.
"""

from pathlib import Path
import json
import joblib
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CHARTS_DIR = PROJECT_ROOT / "static" / "charts"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

from ml.data_loader import load_data, clean_data

UNSUPERVISED_FEATURES = [
    "CGPA",
    "AttendancePercent",
    "Internships",
    "Projects",
    "Workshops",
    "Certifications",
    "AptitudeTestScore",
    "CodingTestScore",
    "SoftSkillsRating",
    "MockInterviewScore",
]

CLUSTER_NAMES = {
    0: "Balanced Regular Performers",
    1: "High Technical Specialists",
    2: "Foundational / Skill-Building Cohort",
    3: "Elite Placements / All-Rounders",
}

CLUSTER_COLORS = {
    0: "#3b82f6",  # Blue
    1: "#10b981",  # Green
    2: "#f59e0b",  # Amber
    3: "#8b5cf6",  # Purple
}


def save_artifact(obj, name: str):
    """Saves object to both .joblib and .pkl."""
    joblib.dump(obj, MODELS_DIR / f"{name}.joblib")
    with open(MODELS_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


def run_unsupervised_pipeline():
    """Trains K-Means and PCA pipelines and generates visual artifacts."""
    raw_df = load_data()
    df = clean_data(raw_df)

    X_unsup = df[UNSUPERVISED_FEATURES].copy()

    # Preprocessing
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_unsup)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # 1. PCA (2 Components)
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    var_ratio = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    total_var = round(sum(var_ratio) * 100, 2)

    # 2. K-Means (K=4)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    df["Cluster"] = cluster_labels
    df["PC1"] = pca_coords[:, 0]
    df["PC2"] = pca_coords[:, 1]

    # Cluster stats & placement rates
    cluster_summary = {}
    for cid in range(4):
        subset = df[df["Cluster"] == cid]
        size = int(len(subset))
        placement_rate = round(float(subset["PlacementStatus"].mean()) * 100, 2) if "PlacementStatus" in subset.columns else 0.0
        avg_cgpa = round(float(subset["CGPA"].mean()), 2)
        avg_coding = round(float(subset["CodingTestScore"].mean()), 2)
        avg_aptitude = round(float(subset["AptitudeTestScore"].mean()), 2)

        cluster_summary[str(cid)] = {
            "name": CLUSTER_NAMES[cid],
            "size": size,
            "pct": round((size / len(df)) * 100, 1),
            "placement_rate": placement_rate,
            "avg_cgpa": avg_cgpa,
            "avg_coding": avg_coding,
            "avg_aptitude": avg_aptitude,
        }

    # 3. PCA Scatter Plot
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for cid in range(4):
        mask = df["Cluster"] == cid
        ax.scatter(
            df.loc[mask, "PC1"],
            df.loc[mask, "PC2"],
            c=CLUSTER_COLORS[cid],
            label=f"Cluster {cid}: {CLUSTER_NAMES[cid]} ({cluster_summary[str(cid)]['placement_rate']}% Placed)",
            alpha=0.45,
            s=18,
            edgecolors="none"
        )

    ax.set_title(
        f"Unsupervised Student Archetypes (K-Means K=4 & PCA 2D)\nExplained Variance: {total_var}% (PC1: {var_ratio[0]*100:.1f}%, PC2: {var_ratio[1]*100:.1f}%)",
        fontsize=11,
        fontweight="bold",
        pad=12
    )
    ax.set_xlabel(f"Principal Component 1 ({var_ratio[0]*100:.1f}% Variance)")
    ax.set_ylabel(f"Principal Component 2 ({var_ratio[1]*100:.1f}% Variance)")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "pca_clusters.png", dpi=130)
    plt.close(fig)

    # 4. Save artifacts
    save_artifact(kmeans, "kmeans_model")
    save_artifact(pca, "pca_model")
    save_artifact(scaler, "unsupervised_scaler")
    save_artifact(imputer, "unsupervised_imputer")

    summary_payload = {
        "k": 4,
        "features": UNSUPERVISED_FEATURES,
        "pca_variance_ratio": var_ratio,
        "pca_total_variance_pct": total_var,
        "clusters": cluster_summary,
    }

    with open(MODELS_DIR / "unsupervised_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    return summary_payload


if __name__ == "__main__":
    print("Running Unsupervised Learning Pipeline...")
    res = run_unsupervised_pipeline()
    print("PCA Total Variance:", res["pca_total_variance_pct"], "%")
    print("Clusters:", res["clusters"])
