import os
import numpy as np
import pandas as pd

GENRE_CSV = "data/manifest/track_genres.csv"

AUDIO_IDS = "results/audio_vae/track_ids.npy"
LYRICS_IDS = "results/lyrics_vae/track_ids.npy"
MULTI_IDS = "results/multimodal/track_ids.npy"

OUT_CSV = "results/eval_tracks.csv"

# --- Load IDs ---
audio_ids = set(map(int, np.load(AUDIO_IDS)))
lyrics_ids = set(map(int, np.load(LYRICS_IDS)))
multi_ids = set(map(int, np.load(MULTI_IDS)))

# --- Intersection ---
common_ids = audio_ids & lyrics_ids & multi_ids

print("Audio IDs:", len(audio_ids))
print("Lyrics IDs:", len(lyrics_ids))
print("Multimodal IDs:", len(multi_ids))
print("Common IDs:", len(common_ids))

# --- Attach genres ---
genres = pd.read_csv(GENRE_CSV)
genres = genres[genres["track_id"].isin(common_ids)]
genres = genres.dropna(subset=["genre"])

os.makedirs("results", exist_ok=True)
genres.sort_values("track_id").to_csv(OUT_CSV, index=False)

print("Eval subset saved to:", OUT_CSV)
print("Eval subset size:", len(genres))
print("Unique genres:", genres["genre"].nunique())
