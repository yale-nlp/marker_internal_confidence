import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from adjustText import adjust_text
import os
import argparse

# ─── CONFIG TOGGLES ───────────────────────────────────────────────────────────
SHOW_TITLE       = True
SHOW_XLABEL      = True
SHOW_YLABEL      = True
SHOW_XTICKS      = True
SHOW_YTICKS      = True
SHOW_RUG         = True
SHOW_HEDGE_LABELS = False
SHOW_GRID        = True
MAX_HEDGES       = 30          # set to None to include all
KDE_BW           = 0.15        # bandwidth for KDE; lower = more detail
FIG_SIZE         = (8, 3)
DPI              = 150
SAVE_GRID_PLOT   = True   # save a 6x2 grid of all dataset KDE subplots
GRID_FIG_SIZE    = (14, 18)
# ──────────────────────────────────────────────────────────────────────────────


def plot_kde_for_dataset(mic_values, hedge_labels, dataset_name, model_name, csv_path):
    """
    Plot KDE of MIC values for a single dataset.

    Parameters
    ----------
    mic_values   : array-like of float, MIC per hedge
    hedge_labels : array-like of str, hedge name per MIC value
    dataset_name : str
    model_name   : str
    output_dir   : str, directory to save the plot
    """
    mic_values   = np.array(mic_values, dtype=float)
    hedge_labels = np.array(hedge_labels)

    # drop NaNs
    valid = ~np.isnan(mic_values)
    mic_values   = mic_values[valid]
    hedge_labels = hedge_labels[valid]

    if len(mic_values) < 2:
        print(f"  Skipping {dataset_name}: fewer than 2 valid MIC values.")
        return

    # optionally cap number of hedges (keep top-N by MIC value for visibility)
    if MAX_HEDGES is not None and len(mic_values) > MAX_HEDGES:
        idx = np.argsort(mic_values)[-MAX_HEDGES:]
        mic_values   = mic_values[idx]
        hedge_labels = hedge_labels[idx]

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    kde = gaussian_kde(mic_values, bw_method=KDE_BW)
    x   = np.linspace(0, 1, 500)
    y   = kde(x)
    ax.plot(x, y, color="steelblue", linewidth=2)
    ax.fill_between(x, y, alpha=0.2, color="steelblue")

    if SHOW_RUG:
        rug_y = -0.04 * y.max()
        ax.scatter(
            mic_values,
            np.full_like(mic_values, rug_y),
            marker="|", color="steelblue", s=200, linewidth=1.5, clip_on=False, zorder=3
        )

    if SHOW_HEDGE_LABELS:
        texts = []
        label_y = -0.10 * y.max()
        for val, label in zip(mic_values, hedge_labels):
            t = ax.text(val, label_y, label, fontsize=7, ha="center", va="top", rotation=45)
            texts.append(t)
        try:
            adjust_text(
                texts,
                ax=ax,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                expand_text=(1.2, 1.4),
                force_text=(0.5, 0.8),
            )
        except Exception:
            pass  # adjustText is optional; falls back to raw placement

    ax.set_xlim(0, 1)
    bottom = -0.25 * y.max() if (SHOW_RUG or SHOW_HEDGE_LABELS) else 0
    ax.set_ylim(bottom, y.max() * 1.15)

    if SHOW_TITLE:
        ax.set_title(f"KDE Plot of MICs for {model_name} on {dataset_name}", fontsize=14)
    if SHOW_XLABEL:
        ax.set_xlabel("MIC Value", fontsize=10)
    if SHOW_YLABEL:
        ax.set_ylabel("Density", fontsize=10)
    if not SHOW_XTICKS:
        ax.set_xticklabels([])
    if not SHOW_YTICKS:
        ax.set_yticklabels([])
    if SHOW_GRID:
        ax.grid(axis="y", alpha=0.3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = os.path.join(csv_path.replace("_dfs", "_plots").replace("shared_mic.csv", "").replace("mic.csv", ""), f"kde_{dataset_name}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved to out_path for ds {dataset_name}!")

def plot_kde_grid(df, model_name, csv_path):
    datasets = df.columns.tolist()
    n_rows, n_cols = 6, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=GRID_FIG_SIZE)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= len(datasets):
            ax.set_visible(False)
            continue

        dataset      = datasets[i]
        mic_values   = np.array(df[dataset].values, dtype=float)
        hedge_labels = np.array(df.index.tolist())

        valid = ~np.isnan(mic_values)
        mic_values   = mic_values[valid]
        hedge_labels = hedge_labels[valid]

        if len(mic_values) < 2:
            ax.set_visible(False)
            continue

        if MAX_HEDGES is not None and len(mic_values) > MAX_HEDGES:
            idx = np.argsort(mic_values)[-MAX_HEDGES:]
            mic_values   = mic_values[idx]
            hedge_labels = hedge_labels[idx]

        kde = gaussian_kde(mic_values, bw_method=KDE_BW)
        x   = np.linspace(0, 1, 500)
        y   = kde(x)
        ax.plot(x, y, color="steelblue", linewidth=1.5)
        ax.fill_between(x, y, alpha=0.2, color="steelblue")

        if SHOW_RUG:
            rug_y = -0.04 * y.max()
            ax.scatter(mic_values, np.full_like(mic_values, rug_y),
                       marker="|", color="steelblue", s=100, linewidth=1.2,
                       clip_on=False, zorder=3)

        if SHOW_HEDGE_LABELS:
            texts = []
            label_y = -0.10 * y.max()
            for val, label in zip(mic_values, hedge_labels):
                t = ax.text(val, label_y, label, fontsize=5.5, ha="center", va="top", rotation=45)
                texts.append(t)
            try:
                adjust_text(texts, ax=ax,
                            arrowprops=dict(arrowstyle="-", color="gray", lw=0.4),
                            expand_text=(1.2, 1.4), force_text=(0.5, 0.8))
            except Exception:
                pass

        ax.set_xlim(0, 1)
        bottom = -0.25 * y.max() if (SHOW_RUG or SHOW_HEDGE_LABELS) else 0
        ax.set_ylim(bottom, y.max() * 1.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if SHOW_TITLE:
            ax.set_title(dataset, fontsize=14)
        if SHOW_XLABEL:
            ax.set_xlabel("MIC", fontsize=14)
        if SHOW_YLABEL:
            ax.set_ylabel("Density", fontsize=14)
        if not SHOW_XTICKS:
            ax.set_xticklabels([])
        if not SHOW_YTICKS:
            ax.set_yticklabels([])
        if SHOW_GRID:
            ax.grid(axis="y", alpha=0.3)

    if SHOW_TITLE:
        fig.suptitle(f"Per-Dataset KDE Plots of MICs for {model_name}", fontsize=20, y=1.01)

    out_path = os.path.join(csv_path.replace("_dfs", "_plots").replace("shared_mic.csv", "").replace("mic.csv", ""), f"kde_all.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved to out_path for model {model_name}!")

def main(csv_path):
    """
    Load a model-specific CSV where:
      - rows    = hedges (index = hedge label)
      - columns = datasets (values = MIC per hedge per dataset)
    """
    df = pd.read_csv(csv_path, index_col=0)

    # infer model name from filename
    model_name = os.path.splitext(os.path.basename(csv_path))[0]
    print(f"Model: {model_name}")
    print(f"Hedges: {len(df)}  |  Datasets: {len(df.columns)}\n")

    for dataset in df.columns:
        mic_values   = df[dataset].values
        hedge_labels = df.index.tolist()
        print(f"Plotting {dataset} ...")
        plot_kde_for_dataset(mic_values, hedge_labels, dataset, args.model_name, csv_path)

    if SAVE_GRID_PLOT:
        print("Plotting grid ...")
        plot_kde_grid(df, args.model_name, csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot per-dataset KDE of marker MIC values.")
    parser.add_argument("--csv_path", type=str, help="Path to model-specific MIC CSV.")
    parser.add_argument("--model_name", type=str)
    args = parser.parse_args()
    main(args.csv_path)
