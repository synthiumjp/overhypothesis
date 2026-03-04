r"""
Study 1 Paper Figures
=====================
Run from: C:\overhypothesis\
Usage:    python scripts\paper_figures.py

Generates publication-ready figures for the Study 1 paper.
Reads:    results\study1\master_item_level.csv
Writes:   results\figures\fig1_fo_so_dissociation.pdf
          results\figures\fig2_hswap_frame_vs_noun.pdf
          results\figures\fig3_seed_violins.pdf
          results\figures\fig4_seed42_spotlight.pdf

Author: Claude (figures for JP)
Date: 2026-03-04
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

# ── Style ──────────────────────────────────────────────────────────
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

# ── Paths ──────────────────────────────────────────────────────────
BASE = Path(r"C:\overhypothesis")
ITEM_CSV = BASE / "results" / "study1" / "master_item_level.csv"
FIG_DIR = BASE / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Colours ────────────────────────────────────────────────────────
C_FO = "#2166AC"       # blue — first-order
C_SO = "#B2182B"       # red — second-order
C_SEED42 = "#FF7F00"   # orange — seed 42 highlight
C_OTHER = "#999999"    # grey — other seeds
C_FRAME = "#4DAF4A"    # green — frame-cued
C_NOUN = "#984EA3"     # purple — noun-only
C_CHANCE = "#AAAAAA"   # light grey — chance line

SIZE_ORDER = ["tiny", "small", "medium"]
SIZE_LABELS = {"tiny": "Tiny\n(~3.4M)", "small": "Small\n(~10M)", "medium": "Medium\n(~25.6M)"}
SIZE_X = {"tiny": 0, "small": 1, "medium": 2}


# ── Load & Prep ────────────────────────────────────────────────────
def load():
    df = pd.read_csv(ITEM_CSV)
    # Use numeric correct column
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
# FIGURE 1: FO/SO Dissociation by Model Size (Regular condition)
# ====================================================================
def fig1_fo_so_dissociation(df):
    """
    Main result figure. Shows first-order accuracy climbing with model
    size while second-order stays at chance. Individual seeds shown as
    points; seed 42 highlighted. Error bars = 95% CI across seeds.
    """
    reg = df[
        (df["condition"] == "regular") &
        (df["item_type"].isin(["first_order", "second_order"])) &
        (df["corpus_fraction"] == 1.0) if "corpus_fraction" in df.columns else True
    ].copy()
    
    # If corpus_fraction filtering removed everything, skip it
    if len(reg) == 0:
        reg = df[
            (df["condition"] == "regular") &
            (df["item_type"].isin(["first_order", "second_order"]))
        ].copy()
    
    # Aggregate by seed
    seed_acc = reg.groupby(["model_size", "item_type", "seed"])["acc"].mean() * 100
    seed_df = seed_acc.reset_index()
    seed_df.columns = ["model_size", "item_type", "seed", "accuracy"]
    
    # Mean and CI across seeds
    summary = seed_df.groupby(["model_size", "item_type"]).agg(
        mean=("accuracy", "mean"),
        sem=("accuracy", "sem"),
        n=("accuracy", "count")
    ).reset_index()
    summary["ci95"] = summary["sem"] * 2.776  # t(4) for 5 seeds, two-tailed 95%
    
    fig, ax = plt.subplots(figsize=(5.5, 4))
    
    offset = 0.12
    for itype, color, label, direction in [
        ("first_order", C_FO, "First-order (exemplar retrieval)", -1),
        ("second_order", C_SO, "Second-order (overhypothesis)", 1),
    ]:
        sub = summary[summary["item_type"] == itype]
        seeds_sub = seed_df[seed_df["item_type"] == itype]
        
        xs = [SIZE_X[s] + direction * offset for s in SIZE_ORDER]
        ys = [sub[sub["model_size"] == s]["mean"].values[0] for s in SIZE_ORDER]
        errs = [sub[sub["model_size"] == s]["ci95"].values[0] for s in SIZE_ORDER]
        
        # Mean line + error bars
        ax.errorbar(xs, ys, yerr=errs, fmt="o-", color=color, markersize=7,
                    linewidth=1.8, capsize=4, capthick=1.2, label=label, zorder=4)
        
        # Individual seeds
        for size in SIZE_ORDER:
            x_base = SIZE_X[size] + direction * offset
            size_seeds = seeds_sub[seeds_sub["model_size"] == size]
            for _, row in size_seeds.iterrows():
                is_42 = (row["seed"] == 42)
                jitter = np.random.default_rng(int(row["seed"])).uniform(-0.04, 0.04)
                ax.scatter(
                    x_base + jitter, row["accuracy"],
                    c=C_SEED42 if is_42 else color,
                    s=30 if is_42 else 15,
                    alpha=0.9 if is_42 else 0.4,
                    edgecolors="black" if is_42 else "none",
                    linewidths=0.5 if is_42 else 0,
                    marker="D" if is_42 else "o",
                    zorder=5 if is_42 else 3,
                )
    
    ax.axhline(50, color=C_CHANCE, linestyle="--", linewidth=1, alpha=0.7, label="Chance (50%)")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([SIZE_LABELS[s] for s in SIZE_ORDER])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(15, 105)
    ax.set_xlim(-0.5, 2.5)
    
    # Custom legend with seed 42 marker
    handles, labels = ax.get_legend_handles_labels()
    seed42_patch = plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=C_SEED42,
                               markeredgecolor="black", markersize=6, label="Seed 42")
    handles.append(seed42_patch)
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.9)
    
    ax.set_title("First-Order vs. Second-Order Accuracy\n(Regular Condition)")
    
    path = FIG_DIR / "fig1_fo_so_dissociation.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"[SAVED] {path}")


# ====================================================================
# FIGURE 2: H_swap — Frame-Cued vs Noun-Only (Feature-Swap condition)
# ====================================================================
def fig2_hswap(df):
    """
    Diagnostic figure showing frame-cued Domain B at 98-100% vs
    noun-only at 14-24%. Demonstrates frame→feature mapping, not
    noun→domain→feature.
    """
    swap = df[
        (df["condition"] == "feature_swap") &
        (df["item_type"].isin(["domain_b_frame_cued", "domain_b_noun_only"]))
    ].copy()
    
    if len(swap) == 0:
        print("[SKIP] fig2 — no feature_swap domain_b items found")
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
        
        # Individual seeds
        for size in SIZE_ORDER:
            x_base = SIZE_X[size] + direction * offset
            size_seeds = seeds_sub[seeds_sub["model_size"] == size]
            for _, row in size_seeds.iterrows():
                jitter = np.random.default_rng(int(row["seed"])).uniform(-0.04, 0.04)
                ax.scatter(
                    x_base + jitter, row["accuracy"],
                    c=color, s=15, alpha=0.45, edgecolors="none", zorder=3,
                )
    
    ax.axhline(50, color=C_CHANCE, linestyle="--", linewidth=1, alpha=0.7, label="Chance (50%)")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([SIZE_LABELS[s] for s in SIZE_ORDER])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(-5, 105)
    ax.set_xlim(-0.5, 2.5)
    ax.legend(loc="center left", frameon=True, framealpha=0.9)
    ax.set_title("Feature-Swap Domain B: Frame-Cued vs. Noun-Only")
    
    path = FIG_DIR / "fig2_hswap_frame_vs_noun.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"[SAVED] {path}")


# ====================================================================
# FIGURE 3: Seed-Level Violin Plots — SO accuracy across conditions
# ====================================================================
def fig3_seed_violins(df):
    """
    Shows per-seed second-order accuracy distributions for each condition
    at each model size. Demonstrates that SO hovers at chance across
    all conditions regardless of seed variance.
    """
    so = df[df["item_type"] == "second_order"].copy()
    
    seed_acc = so.groupby(["condition", "model_size", "seed"])["acc"].mean() * 100
    seed_df = seed_acc.reset_index()
    seed_df.columns = ["condition", "model_size", "seed", "accuracy"]
    
    conditions = ["regular", "scrambled", "noise_injection", "feature_swap",
                  "weak_label", "paraphrased_nolabel", "bare_nolabel", "frequency_matched"]
    # Only use conditions present in data
    conditions = [c for c in conditions if c in seed_df["condition"].unique()]
    
    n_cond = len(conditions)
    fig, axes = plt.subplots(1, n_cond, figsize=(2 * n_cond, 4), sharey=True)
    if n_cond == 1:
        axes = [axes]
    
    cond_labels = {
        "regular": "Regular",
        "scrambled": "Scrambled",
        "noise_injection": "Noise\nInjection",
        "feature_swap": "Feature\nSwap",
        "weak_label": "Weak\nLabel",
        "paraphrased_nolabel": "Paraph.\nNo-Label",
        "bare_nolabel": "Bare\nNo-Label",
        "frequency_matched": "Freq.\nMatched",
    }
    
    size_colors = {"tiny": "#66C2A5", "small": "#FC8D62", "medium": "#8DA0CB"}
    
    for i, cond in enumerate(conditions):
        ax = axes[i]
        cond_data = seed_df[seed_df["condition"] == cond]
        
        data_groups = []
        positions = []
        colors_list = []
        
        for j, size in enumerate(SIZE_ORDER):
            sub = cond_data[cond_data["model_size"] == size]["accuracy"].values
            if len(sub) > 0:
                data_groups.append(sub)
                positions.append(j)
                colors_list.append(size_colors[size])
        
        if data_groups:
            vp = ax.violinplot(data_groups, positions=positions, showmedians=True, widths=0.7)
            for body, col in zip(vp["bodies"], colors_list):
                body.set_facecolor(col)
                body.set_alpha(0.6)
            for partname in ("cmins", "cmaxes", "cmedians", "cbars"):
                if partname in vp:
                    vp[partname].set_color("#333333")
                    vp[partname].set_linewidth(0.8)
            
            # Overlay seeds as strip
            for j, size in enumerate(SIZE_ORDER):
                sub = cond_data[cond_data["model_size"] == size]
                for _, row in sub.iterrows():
                    is_42 = (row["seed"] == 42)
                    jit = np.random.default_rng(int(row["seed"]) + j * 100).uniform(-0.12, 0.12)
                    ax.scatter(
                        j + jit, row["accuracy"],
                        c=C_SEED42 if is_42 else "#333333",
                        s=25 if is_42 else 12,
                        alpha=0.9 if is_42 else 0.5,
                        edgecolors="black" if is_42 else "none",
                        linewidths=0.5 if is_42 else 0,
                        marker="D" if is_42 else "o",
                        zorder=5,
                    )
        
        ax.axhline(50, color=C_CHANCE, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["T", "S", "M"], fontsize=8)
        ax.set_title(cond_labels.get(cond, cond), fontsize=9)
        ax.set_ylim(25, 80)
        
        if i == 0:
            ax.set_ylabel("Second-Order Accuracy (%)")
    
    # Shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        mpatches.Patch(facecolor=size_colors["tiny"], alpha=0.6, label="Tiny"),
        mpatches.Patch(facecolor=size_colors["small"], alpha=0.6, label="Small"),
        mpatches.Patch(facecolor=size_colors["medium"], alpha=0.6, label="Medium"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_SEED42,
               markeredgecolor="black", markersize=5, label="Seed 42"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               frameon=True, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    
    fig.suptitle("Second-Order Accuracy by Condition and Model Size", fontsize=12, y=1.02)
    plt.tight_layout()
    
    path = FIG_DIR / "fig3_seed_violins.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"[SAVED] {path}")


# ====================================================================
# FIGURE 4: Seed 42 Spotlight — FO vs SO dissociation
# ====================================================================
def fig4_seed42_spotlight(df):
    """
    Paired comparison: For each condition × size, plot FO vs SO accuracy.
    Seed 42 shown in orange, other seeds in grey. Demonstrates that
    even when FO is near-ceiling (seed 42), SO remains at chance.
    """
    foso = df[
        df["item_type"].isin(["first_order", "second_order"])
    ].copy()
    
    seed_acc = foso.groupby(["condition", "model_size", "seed", "item_type"])["acc"].mean() * 100
    seed_df = seed_acc.reset_index()
    seed_df.columns = ["condition", "model_size", "seed", "item_type", "accuracy"]
    
    # Pivot to get FO and SO side by side
    pivot = seed_df.pivot_table(
        index=["condition", "model_size", "seed"],
        columns="item_type",
        values="accuracy"
    ).reset_index()
    
    if "first_order" not in pivot.columns or "second_order" not in pivot.columns:
        print("[SKIP] fig4 — missing FO or SO columns in pivot")
        return
    
    # Focus on conditions with enough data
    focus_conditions = ["regular", "scrambled", "noise_injection", "weak_label",
                        "feature_swap", "frequency_matched"]
    focus_conditions = [c for c in focus_conditions if c in pivot["condition"].unique()]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # All non-42 seeds as grey
    others = pivot[(pivot["seed"] != 42) & (pivot["condition"].isin(focus_conditions))]
    ax.scatter(others["first_order"], others["second_order"],
               c=C_OTHER, s=18, alpha=0.35, edgecolors="none", label="Seeds 123–1001", zorder=2)
    
    # Seed 42 as orange diamonds
    s42 = pivot[(pivot["seed"] == 42) & (pivot["condition"].isin(focus_conditions))]
    ax.scatter(s42["first_order"], s42["second_order"],
               c=C_SEED42, s=50, alpha=0.9, edgecolors="black", linewidths=0.5,
               marker="D", label="Seed 42", zorder=4)
    
    # Annotate a few seed 42 points
    for _, row in s42.iterrows():
        if row["first_order"] > 85:
            ax.annotate(
                f"{row['condition'][:6]}/{row['model_size'][0].upper()}",
                (row["first_order"], row["second_order"]),
                textcoords="offset points", xytext=(6, 4),
                fontsize=6.5, color=C_SEED42, alpha=0.85,
            )
    
    # Reference lines
    ax.axhline(50, color=C_CHANCE, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(50, color=C_CHANCE, linestyle="--", linewidth=0.8, alpha=0.6)
    
    # Diagonal (if FO=SO, no dissociation)
    ax.plot([20, 105], [20, 105], color="#DDDDDD", linestyle=":", linewidth=0.8, zorder=1)
    
    ax.set_xlabel("First-Order Accuracy (%)")
    ax.set_ylabel("Second-Order Accuracy (%)")
    ax.set_xlim(20, 105)
    ax.set_ylim(25, 80)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax.set_title("FO/SO Dissociation: Exemplar Learning Without Meta-Regularity")
    
    # Annotation box
    ax.text(0.97, 0.03,
            "Seed 42 achieves near-ceiling FO\nbut SO remains at chance (~50%)",
            transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C_SEED42, alpha=0.15),
            )
    
    path = FIG_DIR / "fig4_seed42_spotlight.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"[SAVED] {path}")


# ====================================================================
# MAIN
# ====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("STUDY 1 PAPER FIGURES")
    print("=" * 60)
    
    df = load()
    print(f"Loaded {len(df):,} rows")
    print(f"Conditions: {sorted(df['condition'].unique())}")
    print(f"Item types: {sorted(df['item_type'].unique())}")
    print(f"Model sizes: {sorted(df['model_size'].unique())}")
    print()
    
    # Filter to full corpus (exclude dose-response partial fractions)
    if "corpus_fraction" in df.columns:
        n_before = len(df)
        # Keep rows where corpus_fraction is 1.0 or NaN (for conditions without dose-response)
        df_full = df[(df["corpus_fraction"] == 1.0) | (df["corpus_fraction"].isna())].copy()
        if len(df_full) > 0:
            print(f"Filtered to full corpus: {len(df_full):,} rows (dropped {n_before - len(df_full):,} dose-response)")
            df = df_full
        else:
            print("[WARN] corpus_fraction filter removed all rows — using full dataset")
    
    print(f"\nGenerating figures in {FIG_DIR} ...\n")
    
    fig1_fo_so_dissociation(df)
    fig2_hswap(df)
    fig3_seed_violins(df)
    fig4_seed42_spotlight(df)
    
    print(f"\nAll figures saved to {FIG_DIR}")
    print("Done.")
