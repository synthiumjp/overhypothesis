r"""
Slot Shuffle Diagnostic -- Item-Level Investigation
====================================================
Run from: C:\overhypothesis\
Usage:    python scripts\slot_shuffle_diagnostic.py

Investigates the slot_shuffle anomaly:
  - Across-seed means 41-60% (sanity check was 100% for one seed)
  - Determines if position sensitivity is seed-driven or systematic

Reads: results\study1\master_item_level.csv
Writes: results\study1\slot_shuffle_diagnostic_report.txt
        results\study1\slot_shuffle_heatmap.png
        results\study1\slot_shuffle_seed_violin.png

Author: Claude (diagnostic for JP)
Date: 2026-03-04
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from collections import defaultdict

# ---------- paths ----------
BASE = Path(r"C:\overhypothesis")
ITEM_CSV = BASE / "results" / "study1" / "master_item_level.csv"
SUMMARY_CSV = BASE / "results" / "study1" / "master_summary.csv"
OUT_DIR = BASE / "results" / "study1"
REPORT_PATH = OUT_DIR / "slot_shuffle_diagnostic_report.txt"

# Try matplotlib; degrade gracefully if not available
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not available — skipping figure generation.")

# ============================================================
# 1. LOAD DATA
# ============================================================
def load_data():
    if not ITEM_CSV.exists():
        print(f"[ERROR] Cannot find {ITEM_CSV}")
        print("  Expected columns: condition, model_size, seed, item_type, item_id, correct, ...")
        print("  Check path or provide master_item_level.csv location.")
        sys.exit(1)
    
    df = pd.read_csv(ITEM_CSV)
    print(f"Loaded {len(df):,} rows from {ITEM_CSV.name}")
    print(f"Columns: {list(df.columns)}")
    
    # Standardise column names (handle common variants)
    # First, prefer numeric correct column
    if "correct_01" in df.columns:
        df["correct_num"] = pd.to_numeric(df["correct_01"], errors="coerce")
    elif "correct_binary" in df.columns:
        df["correct_num"] = pd.to_numeric(df["correct_binary"], errors="coerce")
    else:
        # Coerce the string 'correct' column
        df["correct_num"] = df["correct"].map({"True": 1, "False": 0, "1": 1, "0": 0})
        if df["correct_num"].isna().all():
            df["correct_num"] = pd.to_numeric(df["correct"], errors="coerce")
    
    # Rename other columns
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("item_type", "itemtype", "type"):
            col_map[c] = "item_type"
        elif cl in ("seed",):
            col_map[c] = "seed"
        elif cl in ("condition", "corpus_condition"):
            col_map[c] = "condition"
        elif cl == "size":
            col_map[c] = "model_size"
        elif cl in ("model_size", "modelsize"):
            col_map[c] = "model_size"
        elif cl in ("item_id", "itemid", "item"):
            col_map[c] = "item_id"
        elif cl == "delta_lp_norm":
            col_map[c] = "delta_lp"
        elif cl in ("rank", "target_rank"):
            col_map[c] = "rank"
    
    df = df.rename(columns=col_map)
    # Use the numeric column as our working 'correct'
    df["correct"] = df["correct_num"]
    
    # Verify required columns
    required = ["item_type", "correct", "seed", "condition", "model_size"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns after mapping: {missing}")
        print(f"  Available: {list(df.columns)}")
        sys.exit(1)
    
    return df


# ============================================================
# 2. CORE DIAGNOSTICS
# ============================================================
def run_diagnostics(df):
    report_lines = []
    
    def rpt(line=""):
        report_lines.append(line)
        print(line)
    
    rpt("=" * 72)
    rpt("SLOT SHUFFLE DIAGNOSTIC REPORT")
    rpt(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    rpt("=" * 72)
    
    # Filter to slot_shuffle items
    ss = df[df["item_type"] == "slot_shuffle"].copy()
    if len(ss) == 0:
        # Try partial match
        slot_types = [t for t in df["item_type"].unique() if "slot" in t.lower() or "shuffle" in t.lower()]
        if slot_types:
            rpt(f"[INFO] No exact 'slot_shuffle' type. Found: {slot_types}. Using first.")
            ss = df[df["item_type"] == slot_types[0]].copy()
        else:
            rpt("[ERROR] No slot_shuffle items found.")
            rpt(f"  Available item_types: {sorted(df['item_type'].unique())}")
            return report_lines, None
    
    n_items_per_eval = ss.groupby(["condition", "model_size", "seed"]).size()
    rpt(f"\nSlot shuffle rows: {len(ss):,}")
    rpt(f"Items per evaluation (min/max): {n_items_per_eval.min()}/{n_items_per_eval.max()}")
    rpt(f"Conditions: {sorted(ss['condition'].unique())}")
    rpt(f"Model sizes: {sorted(ss['model_size'].unique())}")
    rpt(f"Seeds: {sorted(ss['seed'].unique())}")
    
    # ----------------------------------------------------------
    # 2a. SEED × CONDITION × SIZE ACCURACY TABLE
    # ----------------------------------------------------------
    rpt("\n" + "-" * 72)
    rpt("SECTION 1: Accuracy by Seed × Condition × Model Size")
    rpt("-" * 72)
    
    acc_table = ss.groupby(["condition", "model_size", "seed"])["correct"].mean() * 100
    acc_df = acc_table.reset_index()
    acc_df.columns = ["condition", "model_size", "seed", "accuracy"]
    
    # Pivot for readability
    for cond in sorted(ss["condition"].unique()):
        rpt(f"\n  Condition: {cond}")
        sub = acc_df[acc_df["condition"] == cond]
        pivot = sub.pivot_table(index="seed", columns="model_size", values="accuracy")
        rpt(pivot.to_string(float_format=lambda x: f"{x:.1f}%"))
        
        # Flag any seed at 100%
        perfect = sub[sub["accuracy"] >= 99.9]
        if len(perfect) > 0:
            rpt(f"  *** 100% SEEDS: {perfect[['seed','model_size']].values.tolist()}")
        
        # Flag any seed below 30% (anti-learning)
        anti = sub[sub["accuracy"] <= 30.0]
        if len(anti) > 0:
            rpt(f"  *** LOW SEEDS (<30%): {anti[['seed','model_size','accuracy']].values.tolist()}")
    
    # ----------------------------------------------------------
    # 2b. IDENTIFY THE OUTLIER SEED(S)
    # ----------------------------------------------------------
    rpt("\n" + "-" * 72)
    rpt("SECTION 2: Outlier Detection — Which seed(s) hit 100%?")
    rpt("-" * 72)
    
    # Per condition×size, compute mean and SD across seeds
    seed_stats = acc_df.groupby(["condition", "model_size"]).agg(
        mean_acc=("accuracy", "mean"),
        sd_acc=("accuracy", "std"),
        min_acc=("accuracy", "min"),
        max_acc=("accuracy", "max"),
        n_seeds=("accuracy", "count")
    ).reset_index()
    
    rpt("\nCell-level summary (across seeds):")
    rpt(seed_stats.to_string(float_format=lambda x: f"{x:.1f}", index=False))
    
    # Flag cells with high variance
    high_var = seed_stats[seed_stats["sd_acc"] > 20]
    if len(high_var) > 0:
        rpt(f"\n  *** HIGH VARIANCE CELLS (SD > 20 pp): {len(high_var)}")
        for _, row in high_var.iterrows():
            rpt(f"    {row['condition']} / {row['model_size']}: "
                f"mean={row['mean_acc']:.1f}, SD={row['sd_acc']:.1f}, "
                f"range=[{row['min_acc']:.1f}, {row['max_acc']:.1f}]")
    
    # ----------------------------------------------------------
    # 2c. ITEM-LEVEL BREAKDOWN FOR OUTLIER SEEDS
    # ----------------------------------------------------------
    rpt("\n" + "-" * 72)
    rpt("SECTION 3: Item-Level Agreement Across Seeds")
    rpt("-" * 72)
    
    # For each condition × size, check how many items are correct across ALL seeds
    # vs. correct for SOME seeds vs. NONE
    
    if "item_id" in ss.columns:
        item_col = "item_id"
    else:
        # Try to construct from available columns
        rpt("  [WARN] No item_id column. Attempting row-index based analysis.")
        item_col = None
    
    if item_col:
        for cond in sorted(ss["condition"].unique()):
            for size in sorted(ss["model_size"].unique()):
                cell = ss[(ss["condition"] == cond) & (ss["model_size"] == size)]
                if len(cell) == 0:
                    continue
                
                # Pivot: item × seed
                item_seed = cell.pivot_table(
                    index=item_col, columns="seed", values="correct", aggfunc="first"
                )
                
                n_seeds = item_seed.shape[1]
                seeds_correct = item_seed.sum(axis=1)
                
                all_correct = (seeds_correct == n_seeds).sum()
                all_wrong = (seeds_correct == 0).sum()
                mixed = len(item_seed) - all_correct - all_wrong
                
                rpt(f"\n  {cond} / {size} ({len(item_seed)} items, {n_seeds} seeds):")
                rpt(f"    All seeds correct:  {all_correct} ({100*all_correct/len(item_seed):.0f}%)")
                rpt(f"    All seeds wrong:    {all_wrong} ({100*all_wrong/len(item_seed):.0f}%)")
                rpt(f"    Mixed (some right): {mixed} ({100*mixed/len(item_seed):.0f}%)")
                
                # Distribution of "how many seeds got it right"
                dist = seeds_correct.value_counts().sort_index()
                rpt(f"    Seeds-correct distribution: {dict(dist)}")
                
                # If there's a 100% seed, which items does it get right that others don't?
                cell_acc = cell.groupby("seed")["correct"].mean()
                top_seed = cell_acc.idxmax()
                top_acc = cell_acc.max()
                
                if top_acc >= 0.95:  # near-perfect seed
                    rpt(f"\n    >>> TOP SEED: {top_seed} (acc={top_acc*100:.1f}%)")
                    # Items this seed gets right but majority of others get wrong
                    top_correct = item_seed[top_seed] == 1
                    others_mean = item_seed.drop(columns=[top_seed]).mean(axis=1)
                    unique_to_top = item_seed.index[top_correct & (others_mean < 0.5)]
                    
                    rpt(f"    Items ONLY top seed gets right (others <50%): {len(unique_to_top)}")
                    if len(unique_to_top) > 0 and len(unique_to_top) <= 20:
                        for item in unique_to_top:
                            other_vals = item_seed.loc[item].drop(top_seed)
                            rpt(f"      {item}: top_seed=1, others={dict(other_vals)}")
    
    # ----------------------------------------------------------
    # 2d. COMPARE SLOT_SHUFFLE TO FIRST_ORDER AND SECOND_ORDER
    # ----------------------------------------------------------
    rpt("\n" + "-" * 72)
    rpt("SECTION 4: Slot Shuffle vs. First/Second-Order (Same Models)")
    rpt("-" * 72)
    rpt("Do the outlier seeds also show unusual FO/SO patterns?")
    
    fo_so = df[df["item_type"].isin(["first_order", "second_order", "slot_shuffle"])].copy()
    
    comparison = fo_so.groupby(["condition", "model_size", "seed", "item_type"])["correct"].mean() * 100
    comp_df = comparison.reset_index()
    comp_df.columns = ["condition", "model_size", "seed", "item_type", "accuracy"]
    
    # For conditions with slot_shuffle data, show side-by-side
    ss_conditions = ss["condition"].unique()
    for cond in sorted(ss_conditions):
        rpt(f"\n  Condition: {cond}")
        sub = comp_df[comp_df["condition"] == cond]
        pivot = sub.pivot_table(
            index=["seed", "model_size"],
            columns="item_type",
            values="accuracy"
        )
        if len(pivot) > 0:
            rpt(pivot.to_string(float_format=lambda x: f"{x:.1f}%"))
    
    # ----------------------------------------------------------
    # 2e. DELTA LOG-PROB DISTRIBUTION FOR SLOT_SHUFFLE
    # ----------------------------------------------------------
    rpt("\n" + "-" * 72)
    rpt("SECTION 5: Delta Log-Prob Distribution")
    rpt("-" * 72)
    
    if "delta_lp" in ss.columns:
        dlp_stats = ss.groupby(["condition", "model_size", "seed"])["delta_lp"].agg(
            ["mean", "median", "std", "min", "max"]
        )
        rpt("\nDelta LP stats by seed:")
        rpt(dlp_stats.to_string(float_format=lambda x: f"{x:.4f}"))
        
        # Check if the 100% seed has categorically different delta_lp magnitudes
        rpt("\n  Interpretation: If the 100% seed has much LARGER delta_lp values,")
        rpt("  the model genuinely distinguishes target from foil at those positions.")
        rpt("  If delta_lp is tiny but positive for all items, it's a marginal artefact.")
    else:
        rpt("  [SKIP] No delta_lp column available.")
    
    # ----------------------------------------------------------
    # 2f. RANK DISTRIBUTION FOR SLOT_SHUFFLE
    # ----------------------------------------------------------
    rpt("\n" + "-" * 72)
    rpt("SECTION 6: Rank Distribution")
    rpt("-" * 72)
    
    if "rank" in ss.columns:
        rank_stats = ss.groupby(["condition", "model_size", "seed"])["rank"].agg(
            ["mean", "median", "std"]
        )
        rpt("\nRank stats by seed:")
        rpt(rank_stats.to_string(float_format=lambda x: f"{x:.2f}"))
        
        pct_rank1 = ss.groupby(["condition", "model_size", "seed"]).apply(
            lambda g: (g["rank"] == 1).mean() * 100
        )
        rpt(f"\n% Ranked 1st by seed:\n{pct_rank1.to_string(float_format=lambda x: f'{x:.1f}%')}")
    else:
        rpt("  [SKIP] No rank column available.")
    
    # ----------------------------------------------------------
    # 2g. HYPOTHESIS: CORPUS POSITION PATTERN
    # ----------------------------------------------------------
    rpt("\n" + "-" * 72)
    rpt("SECTION 7: Interpretive Hypotheses")
    rpt("-" * 72)
    rpt("""
    Slot_shuffle items present features in a different slot order than training.
    If a model learns position-specific feature expectations (e.g., "slot 2 is 
    always a shape token"), shuffling should hurt accuracy.
    
    HYPOTHESIS A — Seed-specific corpus quirk:
      One seed's corpus happened to have less rigid position ordering (e.g., 
      due to sentence template variation), so that seed's model isn't position-
      sensitive. Prediction: the 100% seed should show SIMILAR first_order 
      accuracy to other seeds (the model works fine, just isn't position-bound).
    
    HYPOTHESIS B — Item-specific artefact:
      Certain slot_shuffle items happen to place the target in a position where 
      the model already assigns high probability. Prediction: the same items 
      should be easy across ALL seeds (item-level consistency, not seed-driven).
    
    HYPOTHESIS C — Evaluation bug:
      The 100% accuracy on one seed could indicate a scoring error (e.g., 
      wrong model loaded, or target/foil assignment issue for that seed).
      Prediction: check if delta_lp values are implausibly large or if the 
      model file for that seed is anomalous.
    
    DECISION RULE:
      If high variance is seed-driven (one outlier) with low item-level 
      consistency → report as seed noise, show violin plot, note in paper.
      If item-level consistency is high → investigate corpus position structure.
      If evaluation bug → fix and re-run.
    """)
    
    # ----------------------------------------------------------
    # 2h. ACTIONABLE NEXT STEPS
    # ----------------------------------------------------------
    rpt("\n" + "-" * 72)
    rpt("SECTION 8: Recommended Next Steps")
    rpt("-" * 72)
    
    # Determine which scenario we're in based on the data
    if item_col and len(ss) > 0:
        # Check overall item-level consistency
        for cond in sorted(ss_conditions):
            for size in sorted(ss["model_size"].unique()):
                cell = ss[(ss["condition"] == cond) & (ss["model_size"] == size)]
                if len(cell) == 0:
                    continue
                cell_acc_by_seed = cell.groupby("seed")["correct"].mean()
                max_acc = cell_acc_by_seed.max()
                min_acc = cell_acc_by_seed.min()
                spread = max_acc - min_acc
                
                if spread > 0.4:
                    rpt(f"\n  {cond}/{size}: Spread={spread*100:.0f}pp — SEED-DRIVEN pattern likely.")
                    rpt(f"    → Check corpus for that seed (feature-position correlations)")
                    rpt(f"    → Verify model checkpoint loaded correctly")
                    rpt(f"    → Re-run evaluation for top seed as sanity check")
                elif max_acc > 0.9:
                    rpt(f"\n  {cond}/{size}: High accuracy, low spread — SYSTEMATIC pattern.")
                    rpt(f"    → Models genuinely not position-sensitive in this cell")
                    rpt(f"    → Check if corpus templates vary position order")
    
    rpt("\n" + "=" * 72)
    rpt("END OF DIAGNOSTIC REPORT")
    rpt("=" * 72)
    
    return report_lines, ss


# ============================================================
# 3. FIGURES
# ============================================================
def make_figures(df, ss):
    if not HAS_MPL or ss is None or len(ss) == 0:
        return
    
    # --- Figure 1: Seed-level violin/strip plot ---
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    
    acc_by_seed = ss.groupby(["condition", "model_size", "seed"])["correct"].mean() * 100
    acc_df = acc_by_seed.reset_index()
    acc_df.columns = ["condition", "model_size", "seed", "accuracy"]
    
    # Create grouped positions
    conditions = sorted(acc_df["condition"].unique())
    sizes = sorted(acc_df["model_size"].unique())
    
    positions = []
    labels = []
    data_groups = []
    
    pos = 0
    for cond in conditions:
        for size in sizes:
            sub = acc_df[(acc_df["condition"] == cond) & (acc_df["model_size"] == size)]
            if len(sub) > 0:
                positions.append(pos)
                labels.append(f"{cond[:8]}\n{size}")
                data_groups.append(sub["accuracy"].values)
            pos += 1
        pos += 0.5  # gap between conditions
    
    if data_groups:
        vp = axes.violinplot(data_groups, positions=positions, showmedians=True, widths=0.7)
        
        # Overlay individual seeds as strip
        for i, (p, d) in enumerate(zip(positions, data_groups)):
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(d))
            axes.scatter([p + j for j in jitter], d, c="black", s=25, alpha=0.7, zorder=5)
            
            # Highlight any 100% seeds in red
            for j, val in enumerate(d):
                if val >= 99.9:
                    axes.scatter(p + jitter[j], val, c="red", s=80, marker="*", zorder=6)
        
        axes.set_xticks(positions)
        axes.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        axes.axhline(50, color="gray", linestyle="--", alpha=0.5, label="Chance (50%)")
        axes.set_ylabel("Accuracy (%)")
        axes.set_title("Slot Shuffle Accuracy by Seed (★ = 100% outlier)")
        axes.legend(fontsize=8)
        axes.set_ylim(-5, 110)
    
    plt.tight_layout()
    fig_path = OUT_DIR / "slot_shuffle_seed_violin.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[SAVED] {fig_path}")
    
    # --- Figure 2: Item × Seed heatmap (for conditions with slot_shuffle) ---
    if "item_id" in ss.columns:
        for cond in conditions:
            for size in sizes:
                cell = ss[(ss["condition"] == cond) & (ss["model_size"] == size)]
                if len(cell) == 0:
                    continue
                
                item_seed = cell.pivot_table(
                    index="item_id", columns="seed", values="correct", aggfunc="first"
                )
                
                if item_seed.shape[0] < 2:
                    continue
                
                fig2, ax2 = plt.subplots(figsize=(8, max(4, item_seed.shape[0] * 0.25)))
                im = ax2.imshow(item_seed.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
                ax2.set_xticks(range(item_seed.shape[1]))
                ax2.set_xticklabels(item_seed.columns, fontsize=9)
                ax2.set_yticks(range(item_seed.shape[0]))
                ax2.set_yticklabels(item_seed.index, fontsize=6)
                ax2.set_xlabel("Seed")
                ax2.set_ylabel("Item ID")
                ax2.set_title(f"Slot Shuffle: Item × Seed Correctness\n{cond} / {size}")
                plt.colorbar(im, ax=ax2, label="Correct (1) / Wrong (0)", shrink=0.8)
                plt.tight_layout()
                
                hm_path = OUT_DIR / f"slot_shuffle_heatmap_{cond}_{size}.png"
                plt.savefig(hm_path, dpi=150)
                plt.close()
                print(f"[SAVED] {hm_path}")


# ============================================================
# 4. MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 72)
    print("SLOT SHUFFLE DIAGNOSTIC")
    print("=" * 72)
    
    df = load_data()
    
    # Quick column summary
    print(f"\nItem types in data: {sorted(df['item_type'].unique())}")
    print(f"Conditions: {sorted(df['condition'].unique())}")
    print(f"Model sizes: {sorted(df['model_size'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    
    report_lines, ss = run_diagnostics(df)
    
    # Save report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n[SAVED] Report: {REPORT_PATH}")
    
    # Generate figures
    make_figures(df, ss)
    
    print("\nDone. Review the report and figures, then paste back the key sections")
    print("so we can determine the appropriate disposition for the paper.")
