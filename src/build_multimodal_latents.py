import numpy as np
import pandas as pd
import os

# Load audio latents
audio_lat = np.load("results/audio_vae/latents.npy")
audio_ids = np.load("results/audio_vae/track_ids.npy")

# Load lyrics latents
lyric_lat = np.load("results/lyrics_vae/latents.npy")
lyric_ids = np.load("results/lyrics_vae/track_ids.npy")

# Build dicts
audio_dict = {tid: z for tid, z in zip(audio_ids, audio_lat)}
lyric_dict = {tid: z for tid, z in zip(lyric_ids, lyric_lat)}

# Intersection
common_ids = sorted(set(audio_dict) & set(lyric_dict))

Z_audio = np.stack([audio_dict[tid] for tid in common_ids])
Z_lyrics = np.stack([lyric_dict[tid] for tid in common_ids])

# Concatenate (multimodal fusion)
Z_multi = np.concatenate([Z_audio, Z_lyrics], axis=1)

os.makedirs("results/multimodal", exist_ok=True)

np.save("results/multimodal/latents.npy", Z_multi)
np.save("results/multimodal/track_ids.npy", np.array(common_ids))

print("Multimodal latents shape:", Z_multi.shape)
