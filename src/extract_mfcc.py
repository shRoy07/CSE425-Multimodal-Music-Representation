import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

MANIFEST = "data/manifest/fma_small_manifest.csv"
OUT_DIR = "features/audio_ft"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(MANIFEST)

for _, row in tqdm(df.iterrows(), total=len(df)):
    track_id = int(row["track_id"])
    audio_path = row["audio_path"]

    try:
        # Load audio (mono, fixed sampling rate)
        y, sr = librosa.load(audio_path, sr=22050, mono=True)

        # Compute MFCCs
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=20
        )

        # Mean-pool over time
        mfcc_mean = mfcc.mean(axis=1)

        out_path = os.path.join(OUT_DIR, f"{track_id}.npy")
        np.save(out_path, mfcc_mean)

    except Exception as e:
        print(f"Failed on track {track_id}: {e}")
