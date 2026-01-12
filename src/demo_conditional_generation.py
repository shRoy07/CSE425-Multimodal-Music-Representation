import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from train_cvae_audio_conditioned_on_lyrics import CVAE

AUDIO_DIR = "features/audio_ft"
LYRICS_DIR = "features/lyrics_ft"
MODEL_PATH = "results/conditional_cvae/cvae_audio_cond_lyrics.pt"
PAIRED_IDS = np.load("results/conditional_cvae/track_ids.npy")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pick two different tracks
id_a, id_b = PAIRED_IDS[0], PAIRED_IDS[1]

audio = np.load(os.path.join(AUDIO_DIR, f"{id_a}.npy")).reshape(-1)
lyrics_a = np.load(os.path.join(LYRICS_DIR, f"{id_a}.npy"))
lyrics_b = np.load(os.path.join(LYRICS_DIR, f"{id_b}.npy"))

audio_dim = audio.shape[0]
lyr_dim = lyrics_a.shape[0]

model = CVAE(audio_dim, lyr_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

with torch.no_grad():
    x_a = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    l1 = torch.tensor(lyrics_a, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    l2 = torch.tensor(lyrics_b, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Encode once
    h = model.encoder(torch.cat([x_a, l1], dim=1))
    z = model.mu(h)

    recon_a = model.decoder(torch.cat([z, l1], dim=1)).cpu().numpy().flatten()
    recon_b = model.decoder(torch.cat([z, l2], dim=1)).cpu().numpy().flatten()

plt.figure(figsize=(10,4))
plt.plot(recon_a[:300], label="Condition: lyrics A")
plt.plot(recon_b[:300], label="Condition: lyrics B", alpha=0.7)
plt.legend()
plt.title("Conditional Audio Reconstruction (same z, different lyrics)")
plt.tight_layout()
plt.show()
