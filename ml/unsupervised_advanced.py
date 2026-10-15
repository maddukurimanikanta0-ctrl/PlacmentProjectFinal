"""
ml/unsupervised_advanced.py
--------------------------
M4: Advanced Unsupervised Learning:
Hierarchical Agglomerative Clustering, DBSCAN Density Clustering,
PCA & Manifold Embeddings (t-SNE/UMAP preview), and Isolation Forest Anomaly Detection.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage

from ml.data_loader import load_data, clean_data

NUMERIC_FEATURES = [
    "CGPA", "AttendancePercent", "Internships", "Projects", "Workshops",
    "Certifications", "AptitudeTestScore", "CodingTestScore",
    "SoftSkillsRating", "MockInterviewScore"
]


def run_hierarchical_clustering(n_clusters: int = 4, linkage_method: str = "ward") -> Dict[str, Any]:
    """Runs Agglomerative Hierarchical clustering and extracts cluster sizes and linkage info."""
    df = clean_data(load_data())
    sample = df.sample(n=min(len(df), 1500), random_state=42).copy()
    X_scaled = StandardScaler().fit_transform(sample[NUMERIC_FEATURES].values)

    model = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage=linkage_method)
    labels = model.fit_predict(X_scaled)

    sil = round(float(silhouette_score(X_scaled, labels)), 4) if n_clusters > 1 else 0.0

    # Cluster sizes and placement rates
    sample["Cluster"] = labels
    cluster_profiles = []
    for c in range(n_clusters):
        c_sub = sample[sample["Cluster"] == c]
        cluster_profiles.append({
            "cluster_id": c,
            "size": len(c_sub),
            "percentage": round((len(c_sub) / len(sample)) * 100, 1),
            "placement_rate": round(float(c_sub["PlacementStatus"].mean() * 100), 1),
            "avg_cgpa": round(float(c_sub["CGPA"].mean()), 2),
            "avg_coding": round(float(c_sub["CodingTestScore"].mean()), 1)
        })

    return {
        "n_clusters": n_clusters,
        "linkage": linkage_method,
        "silhouette_score": sil,
        "cluster_profiles": cluster_profiles,
        "dendrogram_notes": "Ward minimizes within-cluster variance; Complete uses maximum pairwise distance; Single uses minimum pairwise distance."
    }


def run_dbscan_clustering(eps: float = 1.8, min_samples: int = 5) -> Dict[str, Any]:
    """Runs DBSCAN density clustering to identify core points, border points, and outliers (noise)."""
    df = clean_data(load_data())
    sample = df.sample(n=min(len(df), 2000), random_state=42).copy()
    X_scaled = StandardScaler().fit_transform(sample[NUMERIC_FEATURES].values)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)

    n_noise = int(np.sum(labels == -1))
    unique_clusters = [int(lbl) for lbl in set(labels) if lbl != -1]
    n_clusters = len(unique_clusters)

    sample["DBSCAN_Cluster"] = labels
    clusters_info = []
    for cid in unique_clusters[:5]:
        c_sub = sample[sample["DBSCAN_Cluster"] == cid]
        clusters_info.append({
            "cluster_id": cid,
            "size": len(c_sub),
            "placement_rate": round(float(c_sub["PlacementStatus"].mean() * 100), 1)
        })

    return {
        "eps": eps,
        "min_samples": min_samples,
        "num_clusters_found": n_clusters,
        "noise_points_count": n_noise,
        "noise_percentage": round((n_noise / len(sample)) * 100, 2),
        "total_analyzed": len(sample),
        "clusters_summary": clusters_info,
        "density_rationale": "DBSCAN identifies non-convex clusters naturally without pre-specifying K and tags sparse outliers as -1."
    }


def run_pca_detailed() -> Dict[str, Any]:
    """Computes full PCA decomposition with component loadings and explained variance."""
    df = clean_data(load_data())
    sample = df.sample(n=min(len(df), 3000), random_state=42).copy()
    X_scaled = StandardScaler().fit_transform(sample[NUMERIC_FEATURES].values)

    pca = PCA(n_components=5, random_state=42)
    pca.fit(X_scaled)

    explained = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]
    cumulative = [round(float(c) * 100, 2) for c in np.cumsum(pca.explained_variance_ratio_)]

    pc1_loadings = {feat: round(float(w), 4) for feat, w in zip(NUMERIC_FEATURES, pca.components_[0])}
    pc2_loadings = {feat: round(float(w), 4) for feat, w in zip(NUMERIC_FEATURES, pca.components_[1])}

    return {
        "n_components": 5,
        "variance_per_component": explained,
        "cumulative_variance": cumulative,
        "pc1_top_features": sorted(pc1_loadings.items(), key=lambda x: abs(x[1]), reverse=True)[:4],
        "pc2_top_features": sorted(pc2_loadings.items(), key=lambda x: abs(x[1]), reverse=True)[:4],
        "manifold_notes": {
            "t_SNE": "Preserves local neighborhood probabilities with student-t kernel; distances between clusters are non-interpretable.",
            "UMAP": "Fuzzy simplicial sets preserving both local and broader topological structure with faster runtime."
        }
    }


def run_anomaly_detection() -> Dict[str, Any]:
    """Fits Isolation Forest to isolate abnormal student profiles based on tree path length."""
    df = clean_data(load_data())
    sample = df.sample(n=min(len(df), 3000), random_state=42).copy()
    X = sample[NUMERIC_FEATURES].values

    iso = IsolationForest(n_estimators=100, contamination=0.03, random_state=42)
    preds = iso.fit_predict(X)
    scores = iso.decision_function(X)

    sample["AnomalyFlag"] = (preds == -1).astype(int)
    sample["AnomalyScore"] = scores

    anomalies = sample[sample["AnomalyFlag"] == 1].head(5)
    anomaly_samples = []
    for _, row in anomalies.iterrows():
        anomaly_samples.append({
            "student_id": int(row.get("StudentID", 0)),
            "cgpa": float(row["CGPA"]),
            "coding": float(row["CodingTestScore"]),
            "attendance": float(row["AttendancePercent"]),
            "internships": int(row["Internships"]),
            "anomaly_score": round(float(row["AnomalyScore"]), 4),
            "status": "Placed" if row["PlacementStatus"] == 1 else "Not Placed"
        })

    return {
        "contamination_rate": 0.03,
        "total_anomalies_detected": int(np.sum(preds == -1)),
        "detection_method": "Isolation Forest (shorter tree path length = higher anomaly)",
        "sample_anomalies": anomaly_samples
    }
