r"""
Study 1 Paper Figures — v2
==========================
Run from: C:\overhypothesis\
Usage:    python scripts\paper_figures_v2.py

Changes from v1:
  - Fig 1: Shows all-seeds mean AND seed-42-excluded mean; uses IQR not CI
  - Fig 3: 2x4 layout for readability at column width
  - Fig 4: Reduced label clutter, removed annotation box
  - Fig 2: Unchanged (publication-ready)

Reads:    results\study1\master_item_level.csv
Writes:   results\figures\fig1_fo_so_dissociation_v2.pdf
          results\figures\fig2_hswap_frame_vs_noun_v2.pdf
          results\figures\fig3_seed_violins_v2.pdf
          results\figures\fig4_seed42_spotlight_v2.pdf
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

# -- Style ---------------------------------------------------------------
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

# -- Paths ---------------------------------------------------------------
BASE = Path(r"C:\overhypothesis")
ITEM_CSV = BASE / "results" / "study1" / "master_item_level.csv"
FIG_DIR = BASE / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -- Colours -------------------------------------------------------------
C_FO = "#2166AC"
C_FO_EX42 = "#67A9CF"   # lighter blue for ex-seed42 line
C_SO = "#B2182B"
C_SEED42 = "#FF7F00"
C_OTHER = "#999999"
C_FRAME = "#4DAF4A"
C_NOUN = "#984EA3"
C_CHANCE = "#AAAAAA"

SIZE_ORDER = ["tiny", "small", "medium"]
SIZE_LABELS = {"tiny": "Tiny\n(~3.4M)", "small": "Small\n(~10M)", "medium": "Medium\n(~25.6M)"}
SIZE_X = {"tiny": 0, "small": 1, "medium": 2}


# -- Load ----------------------------------------------------------------
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
    if "delta_lp_norm" in df.columns:
        df = df.rename(columns={"delta_lp_norm": "delta_lp"})
    return df


# ====================================================================
# FIGURE 1 v2: FO/SO Dissociation — dual mean lines
# ====================================================================
def fig1_v2(df):
    reg = df[
        (df["condition"] == "regular") &
        (df["item_type"].isin(["first_order", "second_order"]))
    ].copy()

    # Filter dose-response if column exists
    if "corpus_fraction" in reg.columns:
        reg2 = reg[(reg["corpus_fraction"] == 1.0) | (reg["corpus_fraction"].isna())]
        if len(reg2) > 0:
            reg = reg2

    seed_acc = reg.groupby(["model_size", "item_type", "seed"])["acc"].mean() * 100
    seed_df = seed_acc.reset_index()
    seed_df.columns = ["model_size", "item_type", "seed", "accuracy"]

    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    offset = 0.10

    for itype, color, label, direction in [
        ("first_order", C_FO, "First-order (all seeds)", -1),
        ("second_order", C_SO, "Second-order (all seeds)", 1),
    ]:
        sub_seeds = seed_df[seed_df["item_type"] == itype]

        # --- All-seeds median + IQR ---
        xs, medians, q25s, q75s = [], [], [], []
        for size in SIZE_ORDER:
            vals = sub_seeds[sub_seeds["model_size"] == size]["accuracy"].values
            if len(vals) == 0:
                continue
            xs.append(SIZE_X[size] + direction * offset)
            medians.append(np.median(vals))
            q25s.append(np.percentile(vals, 25))
            q75s.append(np.percentile(vals, 75))

        err_low = [m - q for m, q in zip(medians, q25s)]
        err_high = [q - m for m, q in zip(medians, q75s)]

        ax.errorbar(xs, medians, yerr=[err_low, err_high], fmt="o-",
                    color=color, markersize=7, linewidth=1.8,
                    capsize=4, capthick=1.2, label=label, zorder=4)

        # --- Seed-42-excluded mean (FO only) ---
        if itype == "first_order":
            xs_ex, means_ex = [], []
            for size in SIZE_ORDER:
                vals_ex = sub_seeds[
                    (sub_seeds["model_size"] == size) & (sub_seeds["seed"] != 42)
                ]["accuracy"].values
                if len(vals_ex) == 0:
                    continue
                xs_ex.append(SIZE_X[size] + direction * offset)
                means_ex.append(np.mean(vals_ex))

            ax.plot(xs_ex, means_ex, "s--", color=C_FO_EX42, markersize=5,
                    linewidth=1.3, alpha=0.85, label="First-order (excl. seed 42)",
                    zorder=3)

        # --- Individual seeds ---
        for size in SIZE_ORDER:
            x_base = SIZE_X[size] + direction * offset
            size_seeds = sub_seeds[sub_seeds["model_size"] == size]
            for _, row in size_seeds.iterrows():
                is_42 = (row["seed"] == 42)
                jit = np.random.default_rng(int(row["seed"])).uniform(-0.04, 0.04)
                ax.scatter(
                    x_base + jit, row["accuracy"],
                    c=C_SEED42 if is_42 else color,
                    s=30 if is_42 else 15,
                    alpha=0.9 if is_42 else 0.4,
                    edgecolors="black" if is_42 else "none",
                    linewidths=0.5 if is_42 else 0,
                    marker="D" if is_42 else "o",
                    zorder=5 if is_42 else 3,
                )

    ax.axhline(50, color=C_CHANCE, linestyle="--", linewidth=1, alpha=0.7,
               label="Chance (50%)")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([SIZE_LABELS[s] for s in SIZE_ORDER])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(15, 105)
    ax.set_xlim(-0.5, 2.5)

    handles, labels = ax.get_legend_handles_labels()
    seed42_handle = Line2D([0], [0], marker="D", color="w",
                           markerfacecolor=C_SEED42, markeredgecolor="black",
                           markersize=6, label="Seed 42")
    handles.append(seed42_handle)
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.9,
              fontsize=7.5)

    ax.set_title("First-Order vs. Second-Order Accuracy\n(Regular Condition)")

    path = FIG_DIR / "fig1_fo_so_dissociation_v2.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"[SAVED] {path}")


# ====================================================================
# FIGURE 2 v2: H_swap (unchanged from v1 — already publication-ready)
# ====================================================================
def fig2_v2(df):
    swap = df[
        (df["condition"] == "feature_swap") &
        (df["item_type"].isin(["domain_b_frame_cued", "domain_b_noun_only"]))
    ].copy()

    if len(swap) == 0:
        print("[SKIP] fig2 — no feature_swap domain_b items")
        return

    seed_acc = swap.groupby(["model_size", "item_type", "seed"])["acc"].mean() * 100
    seed_df = seed_acc.reset_index()
    seed_df.columns = ["model_size", "item_type", "seed", "accuracy"]

    summary = seed_df.groupby(["model_size", "item_type"]).agg(
        mean=("accuracy", "mean"),
        sem=("accuracy", "sem"),
    ).reset_index()
    summary["ci95"] = summary["sem"] * 2.776

    fig, ax = plt.subplots(figsize=(5.5, 4))

    offset = 0.12
    for itype, color, label, direction in [
        ("domain_b_frame_cued", C_FRAME, "Frame-cued (Domain B)", -1),
        ("domain_b_noun_only", C_NOUN, "Noun-only (Domain B)", 1),
    ]:
        sub = summary[summary["item_type"] == itype]
        seeds_sub = seed_df[seed_df["item_type"] == itype]

        xs, ys, errs = [], [], []
        for size in SIZE_ORDER:
            s = sub[sub["model_size"] == size]
            if len(s) == 0:
                continue
            xs.append(SIZE_X[size] + direction * offset)
            ys.append(s["mean"].values[0])
            errs.append(s["ci95"].values[0])

        ax.errorbar(xs, ys, yerr=errs, fmt="o-", color=color, markersize=7,
                    linewidth=1.8, capsize=4, capthick=1.2, label=label, zorder=4)

        for size in SIZE_ORDER:
            x_base = SIZE_X[size] + direction * offset
            size_seeds = seeds_sub[seeds_sub["model_size"] == size]
            for _, row in size_seeds.iterrows():
                jit = np.random.default_rng(int(row["seed"])).uniform(-0.04, 0.04)
                ax.scatter(x_base + jit, row["accuracy"],
                           c=color, s=15, alpha=0.45, edgecolors="none", zorder=3)

    ax.axhline(50, color=C_CHANCE, linestyle="--", linewidth=1, alpha=0.7,
               label="Chance (50%)")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([SIZE_LABELS[s] for s in SIZE_ORDER])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(-5, 105)
    ax.set_xlim(-0.5, 2.5)
    ax.legend(loc="center left", frameon=True, framealpha=0.9)
    ax.set_title("Feature-Swap Domain B: Frame-Cued vs. Noun-Only")

    path = FIG_DIR / "fig2_hswap_frame_vs_noun_v2.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"[SAVED] {path}")


# ====================================================================
# FIGURE 3 v2: Seed Violins — 2x4 grid layout
# ====================================================================
def fig3_v2(df):
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

    n_cond = len(conditions)
    ncols = 4
    nrows = int(np.ceil(n_cond / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.5 * nrows), sharey=True)
    axes = axes.flatten()

    for i, cond in enumerate(conditions):
        ax = axes[i]
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
        ax.set_title(cond_labels.get(cond, cond), fontsize=9.5)
        ax.set_ylim(30, 75)

        if i % ncols == 0:
            ax.set_ylabel("Second-Order\nAccuracy (%)")

    # Hide unused axes
    for j in range(n_cond, len(axes)):
        axes[j].set_visible(False)

    legend_elements = [
        mpatches.Patch(facecolor=size_colors["tiny"], alpha=0.6, label="Tiny"),
        mpatches.Patch(facecolor=size_colors["small"], alpha=0.6, label="Small"),
        mpatches.Patch(facecolor=size_colors["medium"], alpha=0.6, label="Medium"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_SEED42,
               markeredgecolor="black", markersize=5, label="Seed 42"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               frameon=True, fontsize=8.5, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Second-Order Accuracy by Condition and Model Size",
                 fontsize=12, y=1.01)
    plt.tight_layout()

    path = FIG_DIR / "fig3_seed_violins_v2.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"[SAVED] {path}")


# ====================================================================
# FIGURE 4 v2: Seed 42 Spotlight — cleaned labels
# ====================================================================
def fig4_v2(df):
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

    others = pivot[(pivot["seed"] != 42) & (pivot["condition"].isin(focus_conditions))]
    ax.scatter(others["first_order"], others["second_order"],
               c=C_OTHER, s=18, alpha=0.35, edgecolors="none",
               label="Seeds 123-1001", zorder=2)

    s42 = pivot[(pivot["seed"] == 42) & (pivot["condition"].isin(focus_conditions))]
    ax.scatter(s42["first_order"], s42["second_order"],
               c=C_SEED42, s=50, alpha=0.9, edgecolors="black", linewidths=0.5,
               marker="D", label="Seed 42", zorder=4)

    # Smart labelling: only label non-overlapping points
    # Use a simple greedy algorithm to avoid overlap
    label_map = {
        "regular": "Reg", "scrambled": "Scr", "noise_injection": "Noi",
        "weak_label": "Wkl", "feature_swap": "Swp", "frequency_matched": "Frq"
    }
    size_map = {"tiny": "T", "small": "S", "medium": "M"}

    placed_labels = []
    for _, row in s42.sort_values("first_order", ascending=False).iterrows():
        if row["first_order"] < 65:
            continue  # only label high-FO points
        lbl = f"{label_map.get(row['condition'], row['condition'][:3])}/{size_map.get(row['model_size'], '?')}"
        x, y = row["first_order"], row["second_order"]

        # Check overlap with previously placed labels
        too_close = False
        for px, py in placed_labels:
            if abs(x - px) < 4 and abs(y - py) < 3:
                too_close = True
                break

        if not too_close:
            ax.annotate(lbl, (x, y), textcoords="offset points",
                        xytext=(7, 3), fontsize=6.5, color="#555555", alpha=0.8)
            placed_labels.append((x, y))

    ax.axhline(50, color=C_CHANCE, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(50, color=C_CHANCE, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.plot([20, 105], [20, 105], color="#EEEEEE", linestyle=":", linewidth=0.8, zorder=1)

    ax.set_xlabel("First-Order Accuracy (%)")
    ax.set_ylabel("Second-Order Accuracy (%)")
    ax.set_xlim(20, 105)
    ax.set_ylim(25, 75)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax.set_title("FO/SO Dissociation Across Conditions and Seeds")

    path = FIG_DIR / "fig4_seed42_spotlight_v2.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"[SAVED] {path}")


# ====================================================================
# MAIN
# ====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("STUDY 1 PAPER FIGURES v2")
    print("=" * 60)

    df = load()
    print(f"Loaded {len(df):,} rows")

    if "corpus_fraction" in df.columns:
        n_before = len(df)
        df_full = df[(df["corpus_fraction"] == 1.0) | (df["corpus_fraction"].isna())].copy()
        if len(df_full) > 0:
            print(f"Filtered to full corpus: {len(df_full):,} rows")
            df = df_full

    print(f"\nGenerating figures in {FIG_DIR} ...\n")

    fig1_v2(df)
    fig2_v2(df)
    fig3_v2(df)
    fig4_v2(df)

    print(f"\nAll figures saved to {FIG_DIR}")
    print("Done.")
