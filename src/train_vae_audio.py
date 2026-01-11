import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# -----------------------
# Config
# -----------------------
MANIFEST = "data/manifest/fma_small_manifest.csv"
FEATURE_DIR = "features/audio_ft"
OUT_DIR = "results/audio_vae"
LATENT_DIM = 8
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Load MFCC features
# -----------------------
df = pd.read_csv(MANIFEST)

X = []
track_ids = []

for _, row in df.iterrows():
    tid = row["track_id"]
    feat_path = os.path.join(FEATURE_DIR, f"{tid}.npy")
    if os.path.exists(feat_path):
        X.append(np.load(feat_path))
        track_ids.append(tid)

X = np.array(X, dtype=np.float32)

X_tensor = torch.from_numpy(X)
dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

INPUT_DIM = X.shape[1]

# -----------------------
# VAE Model
# -----------------------
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.fc_dec1 = nn.Linear(latent_dim, 64)
        self.fc_dec2 = nn.Linear(64, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc_dec1(z))
        return self.fc_dec2(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# -----------------------
# Loss
# -----------------------
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl

# -----------------------
# Train
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(INPUT_DIM, LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for (x_batch,) in loader:
        x_batch = x_batch.to(device)

        optimizer.zero_grad()
        recon, mu, logvar = model(x_batch)
        loss = vae_loss(recon, x_batch, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {avg_loss:.4f}")

# -----------------------
# Extract latent vectors
# -----------------------
model.eval()
latents = []

with torch.no_grad():
    for x in X_tensor.to(device):
        mu, _ = model.encode(x.unsqueeze(0))
        latents.append(mu.cpu().numpy()[0])

latents = np.array(latents)

np.save(os.path.join(OUT_DIR, "latents.npy"), latents)
np.save(os.path.join(OUT_DIR, "track_ids.npy"), np.array(track_ids))

print("Latent vectors saved.")
