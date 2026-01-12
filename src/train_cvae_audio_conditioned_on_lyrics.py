import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PAIRED_CSV = "data/manifest/paired_manifest.csv"
AUDIO_DIR = "features/audio_ft"
LYRICS_DIR = "features/lyrics_ft"
OUT_DIR = "results/conditional_cvae"

LATENT_DIM = 8
BATCH_SIZE = 16
EPOCHS = 200
LR = 1e-3
BETA = 0.1  # KL weight (small helps with tiny paired data)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PairedDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tid = int(self.df.iloc[idx]["track_id"])
        x_audio = np.load(os.path.join(AUDIO_DIR, f"{tid}.npy")).astype(np.float32)
        x_lyr = np.load(os.path.join(LYRICS_DIR, f"{tid}.npy")).astype(np.float32)

        # flatten audio features (MFCC arrays are usually 2D)
        x_audio = x_audio.reshape(-1)

        return tid, x_audio, x_lyr


class CVAE(nn.Module):
    def __init__(self, audio_dim, lyr_dim, latent_dim=8, hidden=256):
        super().__init__()
        enc_in = audio_dim + lyr_dim
        dec_in = latent_dim + lyr_dim

        # Encoder: (audio, lyrics) -> mu, logvar
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

        # Decoder: (z, lyrics) -> reconstructed audio
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, audio_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_audio, x_lyr):
        h = self.encoder(torch.cat([x_audio, x_lyr], dim=1))
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(torch.cat([z, x_lyr], dim=1))
        return recon, mu, logvar


def loss_fn(recon, x, mu, logvar, beta=0.1):
    recon_loss = nn.MSELoss()(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss.item(), kl.item()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ds = PairedDataset(PAIRED_CSV)
    # Inspect dims from one example
    _, x_audio0, x_lyr0 = ds[0]
    audio_dim = x_audio0.shape[0]
    lyr_dim = x_lyr0.shape[0]

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model = CVAE(audio_dim, lyr_dim, latent_dim=LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total = 0.0
        rtot = 0.0
        ktot = 0.0
        for _, x_audio, x_lyr in loader:
            x_audio = x_audio.to(DEVICE)
            x_lyr = x_lyr.to(DEVICE)

            recon, mu, logvar = model(x_audio, x_lyr)
            loss, rloss, kl = loss_fn(recon, x_audio, mu, logvar, beta=BETA)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()
            rtot += rloss
            ktot += kl

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | loss {total/len(loader):.4f} | recon {rtot/len(loader):.4f} | kl {ktot/len(loader):.4f}")

    # Save model
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "cvae_audio_cond_lyrics.pt"))
    print("Saved:", os.path.join(OUT_DIR, "cvae_audio_cond_lyrics.pt"))

    # Save latent means for analysis
    model.eval()
    all_ids, all_mu = [], []
    with torch.no_grad():
        for tid, x_audio, x_lyr in DataLoader(ds, batch_size=64, shuffle=False):
            x_audio = x_audio.to(DEVICE)
            x_lyr = x_lyr.to(DEVICE)
            h = model.encoder(torch.cat([x_audio, x_lyr], dim=1))
            mu = model.mu(h).cpu().numpy()
            all_mu.append(mu)
            all_ids.extend([int(t) for t in tid.numpy()])

    all_mu = np.vstack(all_mu)
    np.save(os.path.join(OUT_DIR, "latents_mu.npy"), all_mu)
    np.save(os.path.join(OUT_DIR, "track_ids.npy"), np.array(all_ids))
    print("Saved latents:", all_mu.shape)


if __name__ == "__main__":
    main()
