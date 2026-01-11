import numpy as np
import umap
import matplotlib.pyplot as plt

X = np.load("results/multimodal/latents.npy")

reducer = umap.UMAP(random_state=42)
X_2d = reducer.fit_transform(X)

plt.figure(figsize=(6,6))
plt.scatter(X_2d[:,0], X_2d[:,1], s=10)
plt.title("Multimodal Latent Space (Audio + Lyrics)")
plt.tight_layout()
plt.savefig("results/multimodal/umap.png")
plt.show()
