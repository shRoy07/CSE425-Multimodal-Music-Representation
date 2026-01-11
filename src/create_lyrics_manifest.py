import os
import pandas as pd

TEXT_DIR = "data/whisper/text"
OUT_CSV = "data/manifest/fma_small_lyrics_manifest.csv"

rows = []

for fname in os.listdir(TEXT_DIR):
    if not fname.endswith(".txt"):
        continue
    if fname.startswith("_"):  # skip _test_write.txt
        continue

    track_id = int(fname.replace(".txt", ""))
    rows.append({
        "track_id": track_id,
        "text_path": os.path.join(TEXT_DIR, fname)
    })

df = pd.DataFrame(rows).sort_values("track_id")
df.to_csv(OUT_CSV, index=False)

print(f"Saved {len(df)} lyric entries to {OUT_CSV}")
