import os
import pandas as pd

TRACKS_CSV = "data/metadata/fma_metadata/tracks.csv"
OUT_CSV = "data/manifest/track_genres.csv"

# Try reading with 2-row header first (official FMA format)
try:
    tracks = pd.read_csv(TRACKS_CSV, header=[0, 1], low_memory=False)
    cols = list(tracks.columns)

    # Find "id" and "genre_top" columns regardless of exact top-level label
    id_col = None
    genre_col = None

    for c in cols:
        # c is a tuple like ('track','id') if MultiIndex
        if isinstance(c, tuple):
            if c[1] == "id":
                id_col = c
            if c[1] == "genre_top":
                genre_col = c

    if id_col is None or genre_col is None:
        raise KeyError("Expected MultiIndex columns with second level 'id' and 'genre_top' not found.")

    df = tracks[[id_col, genre_col]].copy()
    df.columns = ["track_id", "genre"]

except Exception:
    # Fallback: read as single header row and search by string match
    tracks = pd.read_csv(TRACKS_CSV, header=0, low_memory=False)

    # Try common FMA column naming patterns
    possible_id = [c for c in tracks.columns if "id" in str(c).lower() and "track" in str(c).lower()]
    possible_genre = [c for c in tracks.columns if "genre_top" in str(c).lower()]

    if not possible_id or not possible_genre:
        raise RuntimeError(
            "Could not locate track id / genre_top columns. "
            "Print columns to inspect: python -c \"import pandas as pd; "
            "df=pd.read_csv('data/metadata/fma_metadata/tracks.csv', nrows=1); print(df.columns)\""
        )

    df = tracks[[possible_id[0], possible_genre[0]]].copy()
    df.columns = ["track_id", "genre"]

# Clean and save
df = df.dropna()
df["track_id"] = df["track_id"].astype(int)

os.makedirs("data/manifest", exist_ok=True)
df.to_csv(OUT_CSV, index=False)

print(f"Saved {len(df)} genre labels to {OUT_CSV}")
