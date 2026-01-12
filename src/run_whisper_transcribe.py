import os
import pandas as pd
import whisper
from tqdm import tqdm

# =======================
# Paths
# =======================
MANIFEST = "data/manifest/fma_small_manifest.csv"
OUT_DIR = "data/whisper/text"
LOG_CSV = "results/whisper/transcription_status.csv"
FEATURE_DIR = "features/audio_ft"

print("Current working directory:", os.getcwd())
print("Whisper output directory:", os.path.abspath(OUT_DIR))

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)

# =======================
# Load manifest
# =======================
df = pd.read_csv(MANIFEST)

df["audio_path"] = df["audio_path"].apply(
    lambda p: os.path.normpath(os.path.abspath(p))
)

# Keep only tracks that have audio features (alignment guarantee)
usable = []
for _, row in df.iterrows():
    tid = int(row["track_id"])
    if os.path.exists(os.path.join(FEATURE_DIR, f"{tid}.npy")):
        usable.append(row)

df = pd.DataFrame(usable)

print(f"Tracks with audio features: {len(df)}")

# =======================
# Load Whisper ONCE (IMPORTANT)
# =======================
# base is a strong balance between speed and recall on CPU
model = whisper.load_model("base")

status_rows = []

# =======================
# Transcription loop
# =======================
for _, row in tqdm(df.iterrows(), total=len(df)):
    track_id = int(row["track_id"])
    audio_path = row["audio_path"]
    out_txt = os.path.join(OUT_DIR, f"{track_id}.txt")

    # Skip already processed files
    if os.path.exists(out_txt) and os.path.getsize(out_txt) > 0:
        status_rows.append({
            "track_id": track_id,
            "audio_path": audio_path,
            "status": "skipped_exists"
        })
        continue

    try:
        result = model.transcribe(
            audio_path,
            language="en",
            fp16=False,
            duration=30,                       # ⬅ longer window
            condition_on_previous_text=True,  # ⬅ improves recall
            verbose=False
        )

        text = (result.get("text") or "").strip()

        # Keep weak but non-empty outputs
        if len(text) < 5:
            status = "too_short"
        else:
            status = "ok"

        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(text)

        status_rows.append({
            "track_id": track_id,
            "audio_path": audio_path,
            "status": status,
            "n_chars": len(text)
        })

    except Exception as e:
        status_rows.append({
            "track_id": track_id,
            "audio_path": audio_path,
            "status": f"fail:{type(e).__name__}"
        })

# =======================
# Save log
# =======================
pd.DataFrame(status_rows).to_csv(LOG_CSV, index=False)

print(f"Done. Wrote transcripts to {OUT_DIR}")
print(f"Log saved to {LOG_CSV}")
print(f"Total tracks attempted: {len(df)}")

