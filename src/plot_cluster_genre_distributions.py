import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_counts(assign_csv, out_png, title):
    df = pd.read_csv(assign_csv)

    # cluster x genre counts
    ctab = pd.crosstab(df["cluster"], df["genre_top"])

    # stacked bar plot
    ax = ctab.plot(kind="bar", stacked=True, figsize=(12, 6))
    ax.set_title(title)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    plot_counts(
        "results/hard_eval/audio_eval_assignments.csv",
        "results/hard_eval/plot_audio_cluster_vs_genre.png",
        "Audio VAE clusters vs Genre (Eval subset)"
    )
    plot_counts(
        "results/hard_eval/lyrics_eval_assignments.csv",
        "results/hard_eval/plot_lyrics_cluster_vs_genre.png",
        "Lyrics VAE clusters vs Genre (Paired set)"
    )
    plot_counts(
        "results/hard_eval/multimodal_eval_assignments.csv",
        "results/hard_eval/plot_multimodal_cluster_vs_genre.png",
        "Multimodal clusters vs Genre (Paired set)"
    )
    print("Saved cluster-vs-genre plots into results/hard_eval/")

if __name__ == "__main__":
    main()
