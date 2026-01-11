import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

LATENTS = "results/audio_vae/latents.npy"
CLUSTERS = "results/audio_vae/cluster_assignments.csv"
OUT = "results/audio_vae/umap.png"

Z = np.load(LATENTS)
df = pd.read_csv(CLUSTERS)

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)

Z_2d = reducer.fit_transform(Z)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    Z_2d[:, 0],
    Z_2d[:, 1],
    c=df["cluster"],
    s=6,
    cmap="tab10"
)

plt.colorbar(scatter, label="Cluster ID")
plt.title("UMAP Projection of Audio-Only VAE Latent Space")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.savefig(OUT, dpi=300)
plt.close()

print("UMAP plot saved to", OUT)
