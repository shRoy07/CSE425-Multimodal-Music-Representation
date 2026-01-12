import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from train_cvae_audio_conditioned_on_lyrics import CVAE

PAIRED_CSV = "data/manifest/paired_manifest.csv"
AUDIO_DIR = "features/audio_ft"
LYRICS_DIR = "features/lyrics_ft"
MODEL_PATH = "results/conditional_cvae/cvae_audio_cond_lyrics.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load paired IDs
df = pd.read_csv(PAIRED_CSV)
track_ids = df["track_id"].tolist()

# Load data
X_audio, X_lyrics = [], []
for tid in track_ids:
    a = np.load(os.path.join(AUDIO_DIR, f"{tid}.npy")).reshape(-1)
    l = np.load(os.path.join(LYRICS_DIR, f"{tid}.npy"))
    X_audio.append(a)
    X_lyrics.append(l)

X_audio = np.stack(X_audio)
X_lyrics = np.stack(X_lyrics)

# Baseline: mean audio vector
audio_mean = X_audio.mean(axis=0)
baseline_mse = ((X_audio - audio_mean) ** 2).mean()

# Load CVAE
audio_dim = X_audio.shape[1]
lyr_dim = X_lyrics.shape[1]

model = CVAE(audio_dim, lyr_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

with torch.no_grad():
    x_a = torch.tensor(X_audio, dtype=torch.float32).to(DEVICE)
    x_l = torch.tensor(X_lyrics, dtype=torch.float32).to(DEVICE)
    recon, _, _ = model(x_a, x_l)
    cvae_mse = nn.MSELoss()(recon, x_a).item()

print("=== Reconstruction Error ===")
print(f"Mean-audio baseline MSE : {baseline_mse:.4f}")
print(f"Conditional CVAE MSE    : {cvae_mse:.4f}")
