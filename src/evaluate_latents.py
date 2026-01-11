import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score

EVAL_CSV = "results/eval_tracks.csv"

AUDIO_LAT = "results/audio_vae/latents.npy"
AUDIO_IDS = "results/audio_vae/track_ids.npy"

LYRICS_LAT = "results/lyrics_vae/latents.npy"
LYRICS_IDS = "results/lyrics_vae/track_ids.npy"

MULTI_LAT = "results/multimodal/latents.npy"
MULTI_IDS = "results/multimodal/track_ids.npy"

# --- Load evaluation set ---
eval_df = pd.read_csv(EVAL_CSV)
track_ids = eval_df["track_id"].astype(int).values

labels = LabelEncoder().fit_transform(eval_df["genre"])

def evaluate(lat_path, id_path, name):
    Z = np.load(lat_path)
    ids = np.load(id_path).astype(int)

    id_to_z = {tid: z for tid, z in zip(ids, Z)}
    Z_eval = np.stack([id_to_z[tid] for tid in track_ids])

    sil = silhouette_score(Z_eval, labels)
    ch = calinski_harabasz_score(Z_eval, labels)

    print(f"{name:12s} | Silhouette: {sil:.3f} | CH: {ch:.1f}")

print("\n=== Quantitative Evaluation ===")
evaluate(AUDIO_LAT, AUDIO_IDS, "Audio")
evaluate(LYRICS_LAT, LYRICS_IDS, "Lyrics")
evaluate(MULTI_LAT, MULTI_IDS, "Multimodal")
