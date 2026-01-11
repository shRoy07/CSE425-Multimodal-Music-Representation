import os
import pandas as pd

AUDIO_ROOT = "data/audio/fma_small"
METADATA_CSV = "data/metadata/fma_metadata/tracks.csv"
OUTPUT_CSV = "data/manifest/fma_small_manifest.csv"

# Load metadata (multi-index header)
tracks = pd.read_csv(METADATA_CSV, header=[0, 1])

rows = []

for track_id in tracks.index:
    tid = f"{track_id:06d}"
    folder = tid[:3]
    audio_path = os.path.join(AUDIO_ROOT, folder, f"{tid}.mp3")

    if not os.path.exists(audio_path):
        continue

    genre = tracks.loc[track_id, ("track", "genre_top")]
    split = tracks.loc[track_id, ("set", "split")]

    rows.append({
        "track_id": track_id,
        "audio_path": audio_path,
        "genre": genre,
        "split": split
    })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved {len(df)} tracks to {OUTPUT_CSV}")
