import os
import pandas as pd
import whisper
from tqdm import tqdm

MANIFEST = "data/manifest/fma_small_manifest.csv"
OUT_DIR = "data/whisper/text"
LOG_CSV = "results/whisper/transcription_status.csv"

print("Current working directory:", os.getcwd())
print("Whisper output directory:", os.path.abspath(OUT_DIR))


os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)

with open(os.path.join(OUT_DIR, "_test_write.txt"), "w") as f:
    f.write("test")


# Load the same manifest used for MFCC/VAE
df = pd.read_csv(MANIFEST)


df["audio_path"] = df["audio_path"].apply(
    lambda p: os.path.normpath(os.path.abspath(p))
)

# Use the same set of track_ids that actually have MFCC features (so alignment is guaranteed)
FEATURE_DIR = "features/audio_ft"
usable = []
for _, row in df.iterrows():
    tid = int(row["track_id"])
    if os.path.exists(os.path.join(FEATURE_DIR, f"{tid}.npy")):
        usable.append(row)

df = pd.DataFrame(usable)

# Load Whisper model (CPU-friendly; change to "small" later if you want better quality)
model = whisper.load_model("tiny.en")

status_rows = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    track_id = int(row["track_id"])
    audio_path = row["audio_path"]

    out_txt = os.path.join(OUT_DIR, f"{track_id}.txt")

    # Skip if already transcribed
    if os.path.exists(out_txt) and os.path.getsize(out_txt) > 0:
        status_rows.append({"track_id": track_id, "audio_path": audio_path, "status": "skipped_exists"})
        continue

    try:

        result = model.transcribe(
        audio_path,
        language="en",
        fp16=False,
        duration=15,
        verbose=False,
        condition_on_previous_text=False)

        text = (result.get("text") or "").strip()
        if len(text) == 0:
            text = "[NO_VOCALS_DETECTED]"
        print("WRITING:", out_txt)

        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(text)

        status_rows.append({"track_id": track_id, "audio_path": audio_path, "status": "ok", "n_chars": len(text)})

    except Exception as e:
        status_rows.append({"track_id": track_id, "audio_path": audio_path, "status": f"fail: {type(e).__name__}"})

# Save transcription log
pd.DataFrame(status_rows).to_csv(LOG_CSV, index=False)
print(f"Done. Wrote transcripts to {OUT_DIR} and log to {LOG_CSV}")
print(f"Total tracks attempted: {len(df)}")
