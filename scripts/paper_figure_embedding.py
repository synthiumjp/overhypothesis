"""
paper_figure_embedding.py -- Novel Noun Embedding Similarity (Figure B3)
========================================================================
Violin plot showing cosine similarity distributions at L6:
  - Within-trained (30 nouns pairwise)
  - Cross (trained x novel)
  - Within-novel (8 nouns pairwise)

Also: layer-wise trajectory panel showing WT, CR, WN across all layers.

Uses raw distributions saved by study1_novel_embedding.py.

Output: figures/fig_embedding_similarity.pdf
Author: JP Cacioli | Date: 2026-03-05 | Session 11
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = r"C:\overhypothesis"
RESULTS_DIR = os.path.join(BASE_DIR, "results", "study1")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
RAW_PATH = os.path.join(RESULTS_DIR, "novel_noun_embedding_raw.npz")
CSV_PATH = os.path.join(RESULTS_DIR, "novel_noun_embedding_results.csv")
OUTPUT_PATH = os.path.join(FIGURE_DIR, "fig_embedding_similarity.pdf")

SEEDS = [42, 123, 456, 789, 1001]
FOCUS_LAYER = "L6"  # Peak linear probe accuracy

# Colours consistent with paper figures
COL_WT = "#2171B5"   # blue — within-trained
COL_CR = "#6A51A3"   # purple — cross
COL_WN = "#D94801"   # orange — within-novel
COL_NORM_FO = "#2171B5"
COL_NORM_SO = "#D94801"


def load_raw_data():
    """Load raw cosine similarity distributions from .npz file."""
    data = np.load(RAW_PATH, allow_pickle=True)
    return data


def load_csv_summary():
    """Load per-seed per-layer summary from CSV."""
    import csv
    rows = []
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def make_figure():
    raw = load_raw_data()
    csv_rows = load_csv_summary()

    # ── Parse layer-wise means across seeds ──
    layers_ordered = ["embed"] + [f"L{i}" for i in range(1, 9)]
    layer_labels = ["Emb"] + [f"L{i}" for i in range(1, 9)]

    wt_by_layer = {l: [] for l in layers_ordered}
    cr_by_layer = {l: [] for l in layers_ordered}
    wn_by_layer = {l: [] for l in layers_ordered}

    for row in csv_rows:
        layer = row["layer"]
        if layer in wt_by_layer:
            wt_by_layer[layer].append(float(row["within_trained"]))
            cr_by_layer[layer].append(float(row["cross"]))
            wn_by_layer[layer].append(float(row["within_novel"]))

    wt_means = [np.mean(wt_by_layer[l]) for l in layers_ordered]
    cr_means = [np.mean(cr_by_layer[l]) for l in layers_ordered]
    wn_means = [np.mean(wn_by_layer[l]) for l in layers_ordered]

    wt_sds = [np.std(wt_by_layer[l], ddof=1) if len(wt_by_layer[l]) > 1 else 0
              for l in layers_ordered]
    cr_sds = [np.std(cr_by_layer[l], ddof=1) if len(cr_by_layer[l]) > 1 else 0
              for l in layers_ordered]

    # ── Parse norms for secondary axis ──
    fn_by_layer = {l: [] for l in layers_ordered}
    sn_by_layer = {l: [] for l in layers_ordered}
    for row in csv_rows:
        layer = row["layer"]
        if layer in fn_by_layer:
            fn_by_layer[layer].append(float(row["fo_norm"]))
            sn_by_layer[layer].append(float(row["so_norm"]))

    fn_means = [np.mean(fn_by_layer[l]) for l in layers_ordered]
    sn_means = [np.mean(sn_by_layer[l]) for l in layers_ordered]

    # ── Collect raw distributions at focus layer ──
    focus_wt_all = []
    focus_cr_all = []
    focus_wn_all = []
    for seed in SEEDS:
        key_wt = f"seed{seed}_{FOCUS_LAYER}_wt"
        key_cr = f"seed{seed}_{FOCUS_LAYER}_cr"
        key_wn = f"seed{seed}_{FOCUS_LAYER}_wn"
        if key_wt in raw:
            focus_wt_all.extend(raw[key_wt])
        if key_cr in raw:
            focus_cr_all.extend(raw[key_cr])
        if key_wn in raw:
            focus_wn_all.extend(raw[key_wn])

    # ============================================================
    # FIGURE: 2-panel layout
    # Left: Violin plot at L6
    # Right: Layer-wise trajectory
    # ============================================================
    fig = plt.figure(figsize=(11, 4.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.4], wspace=0.35)

    # ── Panel A: Violin plot at L6 ──
    ax1 = fig.add_subplot(gs[0])

    violin_data = [focus_wt_all, focus_cr_all, focus_wn_all]
    violin_colors = [COL_WT, COL_CR, COL_WN]
    violin_labels = [
        f"Within-trained\n(n={len(focus_wt_all)})",
        f"Cross\n(n={len(focus_cr_all)})",
        f"Within-novel\n(n={len(focus_wn_all)})"
    ]

    parts = ax1.violinplot(violin_data, positions=[1, 2, 3],
                            showmeans=True, showmedians=False, showextrema=False)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(violin_colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.8)

    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(1.5)

    # Add individual seed means as points
    for seed in SEEDS:
        for i, suffix in enumerate(["wt", "cr", "wn"]):
            key = f"seed{seed}_{FOCUS_LAYER}_{suffix}"
            if key in raw:
                seed_mean = np.mean(raw[key])
                ax1.scatter(i + 1, seed_mean, color='white', edgecolor='black',
                           s=25, zorder=5, linewidth=0.8)

    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(violin_labels, fontsize=8)
    ax1.set_ylabel("Cosine Similarity", fontsize=10)
    ax1.set_title(f"(a) Similarity Distributions at {FOCUS_LAYER}", fontsize=11, fontweight='bold')
    ax1.set_ylim(0.3, 1.05)
    ax1.axhline(y=1.0, color='grey', linestyle=':', linewidth=0.5, alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Annotate the key finding
    ax1.annotate("Novel nouns\ncollapse to\nsingle point",
                xy=(3, np.mean(focus_wn_all)), xytext=(2.4, 0.55),
                fontsize=7, fontstyle='italic', color=COL_WN,
                arrowprops=dict(arrowstyle='->', color=COL_WN, lw=1.2),
                ha='center')

    # ── Panel B: Layer-wise trajectory ──
    ax2 = fig.add_subplot(gs[1])
    x = np.arange(len(layers_ordered))

    ax2.plot(x, wt_means, 'o-', color=COL_WT, label='Within-trained', markersize=5, linewidth=1.5)
    ax2.fill_between(x,
                     np.array(wt_means) - np.array(wt_sds),
                     np.array(wt_means) + np.array(wt_sds),
                     color=COL_WT, alpha=0.15)

    ax2.plot(x, cr_means, 's-', color=COL_CR, label='Cross (trained-novel)', markersize=5, linewidth=1.5)
    ax2.fill_between(x,
                     np.array(cr_means) - np.array(cr_sds),
                     np.array(cr_means) + np.array(cr_sds),
                     color=COL_CR, alpha=0.15)

    ax2.plot(x, wn_means, '^-', color=COL_WN, label='Within-novel', markersize=5, linewidth=1.5)
    # WN SD too small to see, skip fill

    ax2.set_xticks(x)
    ax2.set_xticklabels(layer_labels, fontsize=8)
    ax2.set_ylabel("Mean Cosine Similarity", fontsize=10)
    ax2.set_xlabel("Layer", fontsize=10)
    ax2.set_title("(b) Similarity Across Layers", fontsize=11, fontweight='bold')
    ax2.set_ylim(0.55, 1.02)
    ax2.legend(fontsize=8, loc='lower right', framealpha=0.9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Mark L6
    ax2.axvline(x=6, color='grey', linestyle='--', linewidth=0.7, alpha=0.5)
    ax2.text(6.1, 0.58, 'L6\n(probe peak)', fontsize=7, color='grey', va='bottom')

    # Add norm ratio as secondary annotation
    ax2_twin = ax2.twinx()
    norm_ratio = [sn / fn if fn > 0 else 1.0 for fn, sn in zip(fn_means, sn_means)]
    ax2_twin.plot(x, norm_ratio, 'D--', color='grey', markersize=3, linewidth=1.0, alpha=0.6)
    ax2_twin.set_ylabel("SO/FO Norm Ratio", fontsize=8, color='grey')
    ax2_twin.tick_params(axis='y', labelcolor='grey', labelsize=7)
    ax2_twin.set_ylim(0.9, 1.6)
    ax2_twin.spines['top'].set_visible(False)

    plt.tight_layout()

    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {OUTPUT_PATH}")

    # Also save PNG for quick preview
    png_path = OUTPUT_PATH.replace('.pdf', '.png')
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Preview: {png_path}")

    plt.close()


if __name__ == "__main__":
    make_figure()
