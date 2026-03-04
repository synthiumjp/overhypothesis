r"""
Study 1 Paper Figures — v3 (minimal fixes)
===========================================
Run from: C:\overhypothesis\
Usage:    python scripts\paper_figures_v3.py

Only regenerates Fig 3 (2x4 grid) and Fig 4 (clean labels).
Figs 1 and 2 are unchanged from v1 — already pre-reg compliant.

Reads:    results\study1\master_item_level.csv
Writes:   results\figures\fig3_seed_violins_v3.pdf/.png
          results\figures\fig4_seed42_spotlight_v3.pdf/.png
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BASE = Path(r"C:\overhypothesis")
ITEM_CSV = BASE / "results" / "study1" / "master_item_level.csv"
FIG_DIR = BASE / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

C_SEED42 = "#FF7F00"
C_OTHER = "#999999"
C_CHANCE = "#AAAAAA"
SIZE_ORDER = ["tiny", "small", "medium"]


def load():
    df = pd.read_csv(ITEM_CSV)
    if "correct_01" in df.columns:
        df["acc"] = pd.to_numeric(df["correct_01"], errors="coerce")
    elif "correct_binary" in df.columns:
        df["acc"] = pd.to_numeric(df["correct_binary"], errors="coerce")
    else:
        df["acc"] = pd.to_numeric(df["correct"], errors="coerce")
    if "size" in df.columns and "model_size" not in df.columns:
        df = df.rename(columns={"size": "model_size"})
    return df


# ====================================================================
# FIGURE 3 v3: Seed Violins — 2x4 grid
# ====================================================================
def fig3_v3(df):
    so = df[df["item_type"] == "second_order"].copy()

    seed_acc = so.groupby(["condition", "model_size", "seed"])["acc"].mean() * 100
    seed_df = seed_acc.reset_index()
    seed_df.columns = ["condition", "model_size", "seed", "accuracy"]

    conditions = ["regular", "scrambled", "noise_injection", "feature_swap",
                  "weak_label", "paraphrased_nolabel", "bare_nolabel", "frequency_matched"]
    conditions = [c for c in conditions if c in seed_df["condition"].unique()]

    cond_labels = {
        "regular": "Regular",
        "scrambled": "Scrambled",
        "noise_injection": "Noise Injection",
        "feature_swap": "Feature Swap",
        "weak_label": "Weak Label",
        "paraphrased_nolabel": "Paraph. No-Label",
        "bare_nolabel": "Bare No-Label",
        "frequency_matched": "Freq. Matched",
    }

    size_colors = {"tiny": "#66C2A5", "small": "#FC8D62", "medium": "#8DA0CB"}

    ncols = 4
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 6.5), sharey=True)
    axes_flat = axes.flatten()

    for i, cond in enumerate(conditions):
        ax = axes_flat[i]
        cond_data = seed_df[seed_df["condition"] == cond]

        data_groups, positions, colors_list = [], [], []
        for j, size in enumerate(SIZE_ORDER):
            sub = cond_data[cond_data["model_size"] == size]["accuracy"].values
            if len(sub) > 0:
                data_groups.append(sub)
                positions.append(j)
                colors_list.append(size_colors[size])

        if data_groups:
            vp = ax.violinplot(data_groups, positions=positions,
                               showmedians=True, widths=0.7)
            for body, col in zip(vp["bodies"], colors_list):
                body.set_facecolor(col)
                body.set_alpha(0.6)
            for partname in ("cmins", "cmaxes", "cmedians", "cbars"):
                if partname in vp:
                    vp[partname].set_color("#333333")
                    vp[partname].set_linewidth(0.8)

            # Seed dots
            for j, size in enumerate(SIZE_ORDER):
                sub = cond_data[cond_data["model_size"] == size]
                for _, row in sub.iterrows():
                    is_42 = (row["seed"] == 42)
                    jit = np.random.default_rng(int(row["seed"]) + j * 100).uniform(-0.12, 0.12)
                    ax.scatter(
                        j + jit, row["accuracy"],
                        c=C_SEED42 if is_42 else "#333333",
                        s=30 if is_42 else 14,
                        alpha=0.9 if is_42 else 0.5,
                        edgecolors="black" if is_42 else "none",
                        linewidths=0.5 if is_42 else 0,
                        marker="D" if is_42 else "o",
                        zorder=5,
                    )

        ax.axhline(50, color=C_CHANCE, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Tiny", "Small", "Med."], fontsize=8)
        ax.set_title(cond_labels.get(cond, cond), fontsize=10)
        ax.set_ylim(30, 75)

        if i % ncols == 0:
            ax.set_ylabel("Second-Order\nAccuracy (%)")

    # Hide unused axes
    for j in range(len(conditions), len(axes_flat)):
        axes_flat[j].set_visible(False)

    legend_elements = [
        mpatches.Patch(facecolor=size_colors["tiny"], alpha=0.6, label="Tiny (~3.4M)"),
        mpatches.Patch(facecolor=size_colors["small"], alpha=0.6, label="Small (~10M)"),
        mpatches.Patch(facecolor=size_colors["medium"], alpha=0.6, label="Medium (~25.6M)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_SEED42,
               markeredgecolor="black", markersize=5, label="Seed 42"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Second-Order Accuracy by Condition and Model Size",
                 fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    path = FIG_DIR / "fig3_seed_violins_v3.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"[SAVED] {path}")


# ====================================================================
# FIGURE 4 v3: Seed 42 Spotlight — clean labels, no annotation box
# ====================================================================
def fig4_v3(df):
    foso = df[df["item_type"].isin(["first_order", "second_order"])].copy()

    seed_acc = foso.groupby(["condition", "model_size", "seed", "item_type"])["acc"].mean() * 100
    seed_df = seed_acc.reset_index()
    seed_df.columns = ["condition", "model_size", "seed", "item_type", "accuracy"]

    pivot = seed_df.pivot_table(
        index=["condition", "model_size", "seed"],
        columns="item_type", values="accuracy"
    ).reset_index()

    if "first_order" not in pivot.columns or "second_order" not in pivot.columns:
        print("[SKIP] fig4")
        return

    focus_conditions = ["regular", "scrambled", "noise_injection", "weak_label",
                        "feature_swap", "frequency_matched"]
    focus_conditions = [c for c in focus_conditions if c in pivot["condition"].unique()]

    fig, ax = plt.subplots(figsize=(6, 5))

    # Other seeds
    others = pivot[(pivot["seed"] != 42) & (pivot["condition"].isin(focus_conditions))]
    ax.scatter(others["first_order"], others["second_order"],
               c=C_OTHER, s=18, alpha=0.35, edgecolors="none",
               label="Seeds 123\u20131001", zorder=2)

    # Seed 42
    s42 = pivot[(pivot["seed"] == 42) & (pivot["condition"].isin(focus_conditions))]
    ax.scatter(s42["first_order"], s42["second_order"],
               c=C_SEED42, s=50, alpha=0.9, edgecolors="black", linewidths=0.5,
               marker="D", label="Seed 42", zorder=4)

    # Smart labels — greedy non-overlapping, short codes
    label_map = {
        "regular": "Reg", "scrambled": "Scr", "noise_injection": "Noi",
        "weak_label": "Wkl", "feature_swap": "Swp", "frequency_matched": "Frq"
    }
    size_map = {"tiny": "T", "small": "S", "medium": "M"}

    placed = []
    for _, row in s42.sort_values("first_order", ascending=False).iterrows():
        if row["first_order"] < 62:
            continue
        lbl = f"{label_map.get(row['condition'], row['condition'][:3])}/{size_map.get(row['model_size'], '?')}"
        x, y = row["first_order"], row["second_order"]

        too_close = any(abs(x - px) < 5 and abs(y - py) < 3.5 for px, py in placed)
        if not too_close:
            ax.annotate(lbl, (x, y), textcoords="offset points",
                        xytext=(7, 3), fontsize=6.5, color="#555555", alpha=0.85)
            placed.append((x, y))

    # Reference lines
    ax.axhline(50, color=C_CHANCE, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(50, color=C_CHANCE, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.plot([20, 105], [20, 105], color="#EEEEEE", linestyle=":", linewidth=0.8, zorder=1)

    ax.set_xlabel("First-Order Accuracy (%)")
    ax.set_ylabel("Second-Order Accuracy (%)")
    ax.set_xlim(20, 105)
    ax.set_ylim(25, 75)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax.set_title("FO/SO Dissociation Across Conditions and Seeds")

    path = FIG_DIR / "fig4_seed42_spotlight_v3.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"[SAVED] {path}")


# ====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("STUDY 1 PAPER FIGURES v3 (Fig 3 + Fig 4 only)")
    print("=" * 60)

    df = load()
    print(f"Loaded {len(df):,} rows")

    # Filter to full corpus
    if "corpus_fraction" in df.columns:
        df2 = df[(df["corpus_fraction"] == 1.0) | (df["corpus_fraction"].isna())].copy()
        if len(df2) > 0:
            print(f"Filtered to full corpus: {len(df2):,} rows")
            df = df2

    print()
    fig3_v3(df)
    fig4_v3(df)
    print("\nDone. Figs 1 & 2 from v1 remain unchanged.")
