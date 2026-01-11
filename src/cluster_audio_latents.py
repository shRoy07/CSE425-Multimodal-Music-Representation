import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import json

LATENTS = "results/audio_vae/latents.npy"
TRACK_IDS = "results/audio_vae/track_ids.npy"
MANIFEST = "data/manifest/fma_small_manifest.csv"
OUT_DIR = "results/audio_vae"

# Load data
Z = np.load(LATENTS)
track_ids = np.load(TRACK_IDS)

# Choose number of clusters
K = 5

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = kmeans.fit_predict(Z)

# Metrics
sil = silhouette_score(Z, labels)
ch = calinski_harabasz_score(Z, labels)

metrics = {
    "k": K,
    "silhouette": float(sil),
    "calinski_harabasz": float(ch)
}

# Save metrics
with open(os.path.join(OUT_DIR, "clustering_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# Save assignments
df_clusters = pd.DataFrame({
    "track_id": track_ids,
    "cluster": labels
})

df_clusters.to_csv(os.path.join(OUT_DIR, "cluster_assignments.csv"), index=False)

print("Clustering completed.")
print(metrics)
