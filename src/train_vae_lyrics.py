import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

FEATURE_DIR = "features/lyrics_ft"
OUT_DIR = "results/lyrics_vae"
LATENT_DIM = 8
EPOCHS = 100
BATCH_SIZE = 16
LR = 1e-3

os.makedirs(OUT_DIR, exist_ok=True)

# -------- Load lyric embeddings --------
Xs = []
track_ids = []

for fname in os.listdir(FEATURE_DIR):
    if not fname.endswith(".npy"):
        continue
    tid = int(fname.replace(".npy", ""))
    Xs.append(np.load(os.path.join(FEATURE_DIR, fname)))
    track_ids.append(tid)

X = torch.tensor(np.stack(Xs), dtype=torch.float32)

dataset = TensorDataset(X)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

INPUT_DIM = X.shape[1]

# -------- VAE model --------
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 128)
        self.fc3 = nn.Linear(128, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc2(z))
        return self.fc3(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE(INPUT_DIM, LATENT_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------- Training --------
def loss_fn(x_hat, x, mu, logvar):
    recon = nn.functional.mse_loss(x_hat, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl

for epoch in range(1, EPOCHS + 1):
    total_loss = 0
    for (x,) in loader:
        optimizer.zero_grad()
        x_hat, mu, logvar = model(x)
        loss = loss_fn(x_hat, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{EPOCHS}  Loss: {total_loss:.2f}")

# -------- Save latents --------
model.eval()
with torch.no_grad():
    mu, _ = model.encode(X)

np.save(os.path.join(OUT_DIR, "latents.npy"), mu.numpy())
np.save(os.path.join(OUT_DIR, "track_ids.npy"), np.array(track_ids))

print("Lyrics VAE training complete.")
