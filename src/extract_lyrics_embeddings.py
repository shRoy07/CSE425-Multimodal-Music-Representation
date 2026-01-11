import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MANIFEST = "data/manifest/fma_small_lyrics_manifest.csv"
OUT_DIR = "features/lyrics_ft"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(MANIFEST)

model = SentenceTransformer("all-MiniLM-L6-v2")

for _, row in tqdm(df.iterrows(), total=len(df)):
    track_id = int(row["track_id"])
    text_path = row["text_path"]

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if len(text) == 0:
        emb = np.zeros(384)
    else:
        emb = model.encode(text, normalize_embeddings=True)

    np.save(os.path.join(OUT_DIR, f"{track_id}.npy"), emb)

print("Lyric embeddings extraction complete.")
