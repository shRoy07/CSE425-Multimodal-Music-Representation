import os
import pandas as pd

AUDIO_DIR = "features/audio_ft"
LYRICS_DIR = "features/lyrics_ft"
OUT_CSV = "data/manifest/paired_manifest.csv"

audio_ids = {int(f.replace(".npy", "")) for f in os.listdir(AUDIO_DIR) if f.endswith(".npy")}
lyrics_ids = {int(f.replace(".npy", "")) for f in os.listdir(LYRICS_DIR) if f.endswith(".npy")}

paired = sorted(list(audio_ids & lyrics_ids))

os.makedirs("data/manifest", exist_ok=True)
pd.DataFrame({"track_id": paired}).to_csv(OUT_CSV, index=False)

print("Audio features:", len(audio_ids))
print("Lyrics features:", len(lyrics_ids))
print("Paired tracks:", len(paired))
print("Saved:", OUT_CSV)
