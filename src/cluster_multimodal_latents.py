import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X = np.load("results/multimodal/latents.npy")

k = 5
labels = KMeans(n_clusters=k, random_state=0).fit_predict(X)

metrics = {
    "k": k,
    "silhouette": float(silhouette_score(X, labels))
}

with open("results/multimodal/clustering_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

np.savetxt("results/multimodal/cluster_assignments.csv", labels, delimiter=",")

print(metrics)
