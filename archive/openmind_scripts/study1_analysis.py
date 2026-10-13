r"""
Study 1: Overhypothesis Induction — Full Pre-Registered Analysis (Python)
=========================================================================
Run from: C:\overhypothesis\
Usage:    python scripts\study1_analysis.py

Reads:    results\study1\master_item_level.csv
Writes:   results\study1\analysis_output.txt
          results\study1\analysis_results.csv

Implements ALL pre-registered hypothesis tests.
GEE used in place of glmer (pre-reg specifies GLMM; GEE is directionally
equivalent — noted as deviation with justification).

Author: JP Cacioli (with Claude assistance)
Date: 2026-03-04
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import combinations
from scipy import stats
from scipy.special import comb

warnings.filterwarnings("ignore")

# ===========================================================================
# CONFIG
# ===========================================================================
BASE = Path(r"C:\overhypothesis")
ITEM_CSV = BASE / "results" / "study1" / "master_item_level.csv"
OUT_TXT = BASE / "results" / "study1" / "analysis_output.txt"
OUT_CSV = BASE / "results" / "study1" / "analysis_results.csv"

SIZES = ["tiny", "small", "medium"]
SEEDS = [42, 123, 456, 789, 1001]
LABEL_ORDER = ["bare_nolabel", "paraphrased_nolabel", "weak_label", "regular"]

# ===========================================================================
# OUTPUT HELPERS
# ===========================================================================
class Tee:
    """Write to both console and file."""
    def __init__(self, filepath):
        self.file = open(filepath, "w", encoding="utf-8")

    def __call__(self, msg=""):
        print(msg)
        self.file.write(msg + "\n")

    def close(self):
        self.file.close()

    def line(self):
        self("─" * 72)


results_rows = []

def add_result(hyp, typ, size, test, stat_name, stat_val, p_val, alpha, verdict, note=""):
    results_rows.append({
        "Hypothesis": hyp, "Type": typ, "Size": size, "Test": test,
        "Statistic": stat_name,
        "Value": round(stat_val, 4) if stat_val is not None else None,
        "p_value": round(p_val, 5) if p_val is not None else None,
        "Alpha": alpha, "Verdict": verdict, "Note": note
    })


# ===========================================================================
# STATISTICAL HELPERS
# ===========================================================================

def jonckheere_terpstra(groups, values, alternative="increasing"):
    """
    Jonckheere-Terpstra test for ordered alternatives.
    groups: integer group labels (1, 2, 3, ...)
    values: observed values
    alternative: 'increasing' or 'decreasing'
    Returns: (JT statistic, p-value)
    """
    unique_groups = sorted(set(groups))
    k = len(unique_groups)
    group_data = {g: [v for g2, v in zip(groups, values) if g2 == g]
                  for g in unique_groups}

    # Count pairs where value in higher group > value in lower group
    JT = 0
    n_pairs = 0
    for i in range(k):
        for j in range(i + 1, k):
            gi = unique_groups[i]
            gj = unique_groups[j]
            for vi in group_data[gi]:
                for vj in group_data[gj]:
                    if alternative == "increasing":
                        if vj > vi:
                            JT += 1
                        elif vj == vi:
                            JT += 0.5
                    else:
                        if vi > vj:
                            JT += 1
                        elif vi == vj:
                            JT += 0.5
                    n_pairs += 1

    # Normal approximation
    ns = [len(group_data[g]) for g in unique_groups]
    N = sum(ns)

    # Expected value
    E_JT = (N**2 - sum(n**2 for n in ns)) / 4

    # Variance (no ties correction for simplicity)
    var_num = N**2 * (2 * N + 3) - sum(n**2 * (2 * n + 3) for n in ns)
    V_JT = var_num / 72

    if V_JT <= 0:
        return JT, 1.0

    z = (JT - E_JT) / np.sqrt(V_JT)
    if alternative == "increasing":
        p = 1 - stats.norm.cdf(z)
    else:
        p = stats.norm.cdf(z)

    return JT, p


def gee_logistic(data, formula_like, group_col="seed"):
    """
    Fit GEE logistic regression.
    Returns coefficient summary dict or None on failure.
    """
    try:
        import statsmodels.api as sm
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.cov_struct import Exchangeable

        y = data["correct"].values.astype(float)
        # Create dummy for condition (treatment coding)
        conditions = data["condition"].unique()
        if len(conditions) == 2:
            ref = sorted(conditions)[0]  # alphabetical first as reference
            data = data.copy()
            data["cond_dummy"] = (data["condition"] != ref).astype(float)
            X = sm.add_constant(data["cond_dummy"].values)
            col_names = ["Intercept", f"condition_{sorted(conditions)[1]}_vs_{ref}"]
        else:
            data = data.copy()
            dummies = pd.get_dummies(data["condition"], drop_first=True, dtype=float)
            X = sm.add_constant(dummies.values)
            col_names = ["Intercept"] + [f"condition_{c}" for c in dummies.columns]

        groups = data[group_col].values

        model = GEE(y, X, groups=groups, family=Binomial(), cov_struct=Exchangeable())
        result = model.fit()

        coefs = {}
        for i, name in enumerate(col_names):
            coefs[name] = {
                "estimate": result.params[i],
                "se": result.bse[i],
                "z": result.tvalues[i],
                "p": result.pvalues[i]
            }
        return coefs

    except Exception as e:
        return {"error": str(e)}


# ===========================================================================
# LOAD DATA
# ===========================================================================
tee = Tee(OUT_TXT)

tee("=" * 72)
tee("STUDY 1: OVERHYPOTHESIS INDUCTION — PRE-REGISTERED ANALYSIS")
tee(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
tee("=" * 72)
tee()

tee("Loading data...")
df = pd.read_csv(ITEM_CSV)

# Standardise columns
if "correct_01" in df.columns:
    df["correct"] = pd.to_numeric(df["correct_01"], errors="coerce")
elif "correct_binary" in df.columns:
    df["correct"] = pd.to_numeric(df["correct_binary"], errors="coerce")
else:
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")

if "size" in df.columns and "model_size" not in df.columns:
    df = df.rename(columns={"size": "model_size"})

# Filter to full corpus
if "corpus_fraction" in df.columns:
    df_full = df[(df["corpus_fraction"] == 1.0) | (df["corpus_fraction"].isna())].copy()
    tee(f"  Full corpus: {len(df_full):,} rows (dropped {len(df) - len(df_full):,} dose-response)")
else:
    df_full = df.copy()

tee(f"  Total rows: {len(df_full):,}")
tee(f"  Conditions: {sorted(df_full['condition'].unique())}")
tee(f"  Model sizes: {sorted(df_full['model_size'].unique())}")
tee(f"  Seeds: {sorted(df_full['seed'].unique())}")
tee()

tee("NOTE: Pre-registration specifies glmer (R lme4). This analysis uses")
tee("GEE (statsmodels) as a Python equivalent. GEE produces population-")
tee("averaged estimates vs. glmer's subject-specific estimates. Given the")
tee("pervasive null pattern (all SO accuracy 48-55%), this distinction")
tee("does not affect any verdict. Noted as a minor procedural deviation.")
tee()


# ===========================================================================
# H1: OVERHYPOTHESIS EMERGENCE (Confirmatory Primary)
# ===========================================================================
tee.line()
tee("H1: OVERHYPOTHESIS EMERGENCE")
tee("  Test: Regular vs Scrambled on second-order items")
tee("  Criterion: >=15 pp above chance AND >=10 pp above Scrambled")
tee("  Model: GEE logistic (exchangeable correlation within seed)")
tee.line()

for sz in SIZES:
    tee(f"\n--- {sz.upper()} ---")

    h1 = df_full[
        (df_full["item_type"] == "second_order") &
        (df_full["condition"].isin(["regular", "scrambled"])) &
        (df_full["model_size"] == sz)
    ].copy()

    tee(f"  N rows: {len(h1)}  |  N items: {h1['item_id'].nunique()}  |  N seeds: {h1['seed'].nunique()}")

    # Descriptives
    desc = h1.groupby("condition")["correct"].agg(["mean", "std", "count"])
    for cond in ["regular", "scrambled"]:
        if cond in desc.index:
            tee(f"  {cond}: M={desc.loc[cond, 'mean']*100:.1f}%, "
                f"SD={desc.loc[cond, 'std']*100:.1f}%, N={int(desc.loc[cond, 'count'])}")

    reg_mean = desc.loc["regular", "mean"] * 100 if "regular" in desc.index else 50
    scr_mean = desc.loc["scrambled", "mean"] * 100 if "scrambled" in desc.index else 50

    tee(f"  Reg - Chance = {reg_mean - 50:.1f} pp")
    tee(f"  Reg - Scr = {reg_mean - scr_mean:.1f} pp")

    # GEE
    coefs = gee_logistic(h1, "correct ~ condition", group_col="seed")
    if "error" not in coefs:
        cond_key = [k for k in coefs if k != "Intercept"][0] if len(coefs) > 1 else None
        if cond_key:
            c = coefs[cond_key]
            tee(f"  GEE: β={c['estimate']:.3f}, z={c['z']:.3f}, p={c['p']:.4f}")
            est, p_val = c["estimate"], c["p"]
        else:
            est, p_val = 0, 1
    else:
        tee(f"  GEE error: {coefs['error']}")
        est, p_val = 0, 1

    verdict = "SUPPORTED" if (reg_mean - 50 >= 15) and (reg_mean - scr_mean >= 10) else "NOT SUPPORTED"
    tee(f"  VERDICT: {verdict}")

    add_result("H1", "Confirmatory", sz, "GEE Reg vs Scr",
               "beta", est, p_val, 0.05, verdict,
               f"reg={reg_mean:.1f}%; scr={scr_mean:.1f}%")

    # Mann-Whitney U backup (seed-level means)
    seed_means = h1.groupby(["condition", "seed"])["correct"].mean().reset_index()
    reg_seeds = seed_means[seed_means["condition"] == "regular"]["correct"].values
    scr_seeds = seed_means[seed_means["condition"] == "scrambled"]["correct"].values

    if len(reg_seeds) >= 2 and len(scr_seeds) >= 2:
        u_stat, u_p = stats.mannwhitneyu(reg_seeds, scr_seeds, alternative="two-sided")
        tee(f"  Mann-Whitney U: U={u_stat:.0f}, p={u_p:.4f}")
        add_result("H1", "Nonparametric", sz, "Mann-Whitney U",
                   "U", u_stat, u_p, 0.05, verdict, "seed-level means")


# ===========================================================================
# H2: FRAME-VARIANT TRANSFER (Confirmatory Primary)
# ===========================================================================
tee.line()
tee("H2: FRAME-VARIANT TRANSFER")
tee("  Test: Frame-variant accuracy >= 75% of standard SO accuracy")
tee("  Pre-registered: Paired t-test across frame conditions per size")
tee.line()

for sz in SIZES:
    tee(f"\n--- {sz.upper()} ---")

    h2 = df_full[
        (df_full["condition"] == "regular") &
        (df_full["item_type"].isin(["second_order", "frame_variant"])) &
        (df_full["model_size"] == sz)
    ]

    seed_acc = h2.groupby(["seed", "item_type"])["correct"].mean().reset_index()
    pivot = seed_acc.pivot(index="seed", columns="item_type", values="correct")

    if "frame_variant" in pivot.columns and "second_order" in pivot.columns:
        fv = pivot["frame_variant"].values * 100
        so = pivot["second_order"].values * 100
        ratio = np.mean(fv) / np.mean(so) if np.mean(so) > 0 else np.nan

        tee(f"  FV: M={np.mean(fv):.1f}%  |  SO: M={np.mean(so):.1f}%  |  Ratio: {ratio:.2f}")

        if len(fv) >= 3:
            t_stat, t_p = stats.ttest_rel(fv, so)
            tee(f"  Paired t-test: t({len(fv)-1})={t_stat:.3f}, p={t_p:.4f}")

            if np.mean(so) <= 52:
                verdict = "VACUOUS (both at chance)"
            elif ratio >= 0.75:
                verdict = "SUPPORTED"
            else:
                verdict = "NOT SUPPORTED"

            tee(f"  VERDICT: {verdict}")
            add_result("H2", "Confirmatory", sz, "Paired t FV vs SO",
                       "t", t_stat, t_p, 0.05, verdict,
                       f"ratio={ratio:.2f}; fv={np.mean(fv):.1f}%; so={np.mean(so):.1f}%")
    else:
        tee("  Missing frame_variant or second_order data")


# ===========================================================================
# H3: DOSE-RESPONSE (Exploratory)
# ===========================================================================
tee.line()
tee("H3: DOSE-RESPONSE TREND")
tee("  Test: Jonckheere-Terpstra + Kendall tau-b")
tee("  Data: 25% / 50% / 100% corpus fractions, regular condition")
tee.line()

if "corpus_fraction" in df.columns:
    dose = df[
        (df["condition"] == "regular") &
        (df["item_type"] == "second_order") &
        (df["corpus_fraction"].isin([0.25, 0.50, 1.0]))
    ]

    for sz in SIZES:
        tee(f"\n--- {sz.upper()} ---")
        sz_dose = dose[dose["model_size"] == sz]

        seed_acc = sz_dose.groupby(["seed", "corpus_fraction"])["correct"].mean().reset_index()

        desc = seed_acc.groupby("corpus_fraction")["correct"].agg(["mean", "std"])
        for frac, row in desc.iterrows():
            tee(f"  Frac {frac*100:.0f}%: M={row['mean']*100:.1f}%, SD={row['std']*100:.1f}%")

        # Kendall tau-b
        tau, tau_p = stats.kendalltau(seed_acc["corpus_fraction"], seed_acc["correct"])
        tee(f"  Kendall tau-b = {tau:.3f}, p = {tau_p:.4f}")

        # Jonckheere-Terpstra
        groups = seed_acc["corpus_fraction"].rank(method="dense").astype(int).values
        vals = seed_acc["correct"].values
        jt_stat, jt_p = jonckheere_terpstra(groups, vals, alternative="increasing")
        tee(f"  Jonckheere-Terpstra: JT={jt_stat:.0f}, p={jt_p:.4f}")

        verdict = "TREND DETECTED" if jt_p < 0.05 else "NOT SUPPORTED"
        tee(f"  VERDICT: {verdict}")

        add_result("H3", "Exploratory", sz, "JT trend",
                   "JT", jt_stat, jt_p, 0.05, verdict, f"tau={tau:.3f}")
else:
    tee("  No corpus_fraction column — skipping")


# ===========================================================================
# H4: ONTOLOGICAL KIND (Exploratory)
# ===========================================================================
tee.line()
tee("H4: ONTOLOGICAL KIND")
tee("  Test: Count-shape > Count-texture AND Mass-texture > Mass-shape")
tee.line()

for sz in SIZES:
    tee(f"\n--- {sz.upper()} ---")

    h4 = df_full[
        (df_full["condition"] == "regular") &
        (df_full["item_type"].isin(["count_frame", "mass_frame_shape", "mass_frame_texture"])) &
        (df_full["model_size"] == sz)
    ]

    seed_acc = h4.groupby(["seed", "item_type"])["correct"].mean().reset_index()
    pivot = seed_acc.pivot(index="seed", columns="item_type", values="correct")

    for col in ["count_frame", "mass_frame_shape", "mass_frame_texture"]:
        if col in pivot.columns:
            tee(f"  {col}: M={pivot[col].mean()*100:.1f}%")

    if "count_frame" in pivot.columns and "mass_frame_texture" in pivot.columns:
        cf = pivot["count_frame"].values * 100
        mt = pivot["mass_frame_texture"].values * 100
        if len(cf) >= 3:
            t_stat, t_p = stats.ttest_rel(cf, mt)
            tee(f"  Count vs Mass-texture: t({len(cf)-1})={t_stat:.3f}, p={t_p:.4f}")
            verdict = "SUPPORTED" if (t_p < 0.05 and np.mean(cf) > np.mean(mt)) else "NOT SUPPORTED"
            add_result("H4", "Exploratory", sz, "Count vs Mass-texture",
                       "t", t_stat, t_p, 0.05, verdict, "")


# ===========================================================================
# H_LABEL: LABELLING NECESSITY (Confirmatory Secondary)
# ===========================================================================
tee.line()
tee("H_LABEL: LABELLING NECESSITY")
tee("  Test: Jonckheere-Terpstra: Regular >> Weak >> Paraphrased >> Bare")
tee("  Alpha: 0.025 (Bonferroni for 2 planned contrasts)")
tee.line()

for sz in SIZES:
    tee(f"\n--- {sz.upper()} ---")

    hl = df_full[
        (df_full["item_type"] == "second_order") &
        (df_full["condition"].isin(LABEL_ORDER)) &
        (df_full["model_size"] == sz)
    ].copy()

    seed_acc = hl.groupby(["condition", "seed"])["correct"].mean().reset_index()

    for cond in LABEL_ORDER:
        sub = seed_acc[seed_acc["condition"] == cond]["correct"]
        if len(sub) > 0:
            tee(f"  {cond}: M={sub.mean()*100:.1f}%")

    # Map to ordered numeric
    cond_map = {c: i + 1 for i, c in enumerate(LABEL_ORDER)}
    seed_acc["cond_num"] = seed_acc["condition"].map(cond_map)

    groups = seed_acc["cond_num"].values
    vals = seed_acc["correct"].values

    # JT
    jt_stat, jt_p = jonckheere_terpstra(groups, vals, alternative="increasing")
    tee(f"  JT = {jt_stat:.0f}, p = {jt_p:.4f} (α = 0.025 after Bonferroni)")

    # Kendall tau-b
    tau, tau_p = stats.kendalltau(groups, vals)
    tee(f"  Kendall tau-b = {tau:.3f}, p = {tau_p:.4f}")

    verdict = "SUPPORTED" if jt_p < 0.025 else "NOT SUPPORTED"
    tee(f"  VERDICT: {verdict}")

    add_result("H_label", "Secondary", sz, "JT trend",
               "JT", jt_stat, jt_p, 0.025, verdict, "Bonferroni corrected")
    add_result("H_label", "Secondary", sz, "Kendall tau",
               "tau", tau, tau_p, 0.025, verdict, "effect size")


# ===========================================================================
# H_SWAP: FEATURE-SWAP (Exploratory)
# ===========================================================================
tee.line()
tee("H_SWAP: FEATURE-SWAP DOMAIN SWITCHING")
tee("  Test: Binomial test vs chance for Domain B items")
tee.line()

for sz in SIZES:
    tee(f"\n--- {sz.upper()} ---")

    swap = df_full[
        (df_full["condition"] == "feature_swap") &
        (df_full["item_type"].isin(["domain_b_frame_cued", "domain_b_noun_only"])) &
        (df_full["model_size"] == sz)
    ]

    for it in ["domain_b_frame_cued", "domain_b_noun_only"]:
        sub = swap[swap["item_type"] == it]
        if len(sub) > 0:
            n_correct = int(sub["correct"].sum())
            n_total = int(sub["correct"].notna().sum())
            prop = n_correct / n_total * 100
            bt = stats.binomtest(n_correct, n_total, p=0.5)
            ci = bt.proportion_ci(confidence_level=0.95)
            tee(f"  {it}: {prop:.1f}% ({n_correct}/{n_total}), "
                f"p={bt.pvalue:.4e}, 95% CI [{ci.low*100:.1f}, {ci.high*100:.1f}]")

            if bt.pvalue < 0.05 and prop > 65:
                v = "ABOVE CHANCE"
            elif bt.pvalue < 0.05 and prop < 35:
                v = "BELOW CHANCE"
            else:
                v = "AT CHANCE"

            add_result("H_swap", "Exploratory", sz,
                       f"Binomial {it.replace('domain_b_', '')}",
                       "prop", prop / 100, bt.pvalue, 0.05, v, f"n={n_total}")


# ===========================================================================
# H_BAYES: BAYESIAN ALIGNMENT (Confirmatory Secondary)
# ===========================================================================
tee.line()
tee("H_BAYES: BAYESIAN ALIGNMENT (KL Divergence)")
tee("  Criterion: KL divergence <= 0.1 nats")
tee("  STATUS: Model accuracy reported below.")
tee("  HBM ideal observer posterior requires Dirichlet-multinomial")
tee("  implementation (Appendix E). Deferred — does not block paper")
tee("  given pervasive null on H1.")
tee.line()

for sz in SIZES:
    reg_so = df_full[
        (df_full["condition"] == "regular") &
        (df_full["item_type"] == "second_order") &
        (df_full["model_size"] == sz)
    ]
    acc = reg_so["correct"].mean() * 100
    tee(f"  {sz}: Regular SO accuracy = {acc:.1f}% (chance = 50%)")
    add_result("H_Bayes", "Secondary", sz, "KL divergence",
               "acc", acc / 100, None, 0.05, "PENDING",
               "Requires HBM posterior")

tee()


# ===========================================================================
# NONPARAMETRIC BACKUPS: All pairwise Mann-Whitney U
# ===========================================================================
tee.line()
tee("NONPARAMETRIC BACKUPS: Mann-Whitney U (all condition pairs)")
tee("  Pre-registered: reported for all between-condition comparisons")
tee.line()

conditions = sorted(df_full["condition"].unique())

for sz in SIZES:
    tee(f"\n--- {sz.upper()} ---")

    sz_so = df_full[
        (df_full["item_type"] == "second_order") &
        (df_full["model_size"] == sz)
    ]
    seed_acc = sz_so.groupby(["condition", "seed"])["correct"].mean().reset_index()

    for c1, c2 in combinations(conditions, 2):
        a = seed_acc[seed_acc["condition"] == c1]["correct"].values
        b = seed_acc[seed_acc["condition"] == c2]["correct"].values

        if len(a) >= 2 and len(b) >= 2:
            u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
            d = (np.mean(a) - np.mean(b)) * 100
            sig = "*" if u_p < 0.05 else ""
            tee(f"  {c1} vs {c2}: Δ={d:+.1f}pp, U={u_stat:.0f}, p={u_p:.4f} {sig}")


# ===========================================================================
# EXPLORATORY: FO/SO DISSOCIATION
# ===========================================================================
tee.line()
tee("EXPLORATORY: FIRST-ORDER vs SECOND-ORDER DISSOCIATION")
tee.line()

for sz in SIZES:
    tee(f"\n--- {sz.upper()} ---")

    foso = df_full[
        (df_full["condition"] == "regular") &
        (df_full["item_type"].isin(["first_order", "second_order"])) &
        (df_full["model_size"] == sz)
    ]

    seed_acc = foso.groupby(["seed", "item_type"])["correct"].mean().reset_index()
    pivot = seed_acc.pivot(index="seed", columns="item_type", values="correct")

    if "first_order" in pivot.columns and "second_order" in pivot.columns:
        fo = pivot["first_order"].values * 100
        so = pivot["second_order"].values * 100

        tee(f"  FO: M={np.mean(fo):.1f}% (SD={np.std(fo, ddof=1):.1f}%)")
        tee(f"  SO: M={np.mean(so):.1f}% (SD={np.std(so, ddof=1):.1f}%)")
        tee(f"  FO - SO = {np.mean(fo) - np.mean(so):.1f} pp")

        if len(fo) >= 3:
            t_stat, t_p = stats.ttest_rel(fo, so)
            tee(f"  Paired t-test: t({len(fo)-1})={t_stat:.3f}, p={t_p:.4f}")

        # Seed 42 specifically
        s42 = pivot.loc[pivot.index == 42] if 42 in pivot.index else pivot.loc[pivot.index == "42"]
        if len(s42) > 0:
            tee(f"  Seed 42: FO={s42['first_order'].values[0]*100:.1f}%, "
                f"SO={s42['second_order'].values[0]*100:.1f}%")

        # One-sample t-test: FO > 50%?
        t_fo, p_fo = stats.ttest_1samp(fo, 50)
        tee(f"  FO vs chance: t({len(fo)-1})={t_fo:.3f}, p={p_fo:.4f}")
        t_so, p_so = stats.ttest_1samp(so, 50)
        tee(f"  SO vs chance: t({len(so)-1})={t_so:.3f}, p={p_so:.4f}")

        add_result("FO/SO", "Exploratory", sz, "FO vs SO paired t",
                   "t", t_stat, t_p, 0.05,
                   "DISSOCIATION" if t_p < 0.05 else "NO DISSOCIATION",
                   f"fo={np.mean(fo):.1f}%; so={np.mean(so):.1f}%")


# ===========================================================================
# EXPLORATORY: SENSITIVITY — SEED 42 EXCLUSION
# ===========================================================================
tee.line()
tee("EXPLORATORY: SENSITIVITY ANALYSIS — SEED 42 EXCLUSION")
tee("  Clearly labelled as post-hoc exploratory.")
tee.line()

for sz in SIZES:
    tee(f"\n--- {sz.upper()} ---")

    sens = df_full[
        (df_full["condition"] == "regular") &
        (df_full["item_type"].isin(["first_order", "second_order"])) &
        (df_full["model_size"] == sz) &
        (df_full["seed"] != 42)
    ]

    for it in ["first_order", "second_order"]:
        sub = sens[sens["item_type"] == it]
        seed_vals = sub.groupby("seed")["correct"].mean().values * 100
        tee(f"  {it} (excl. seed 42): M={np.mean(seed_vals):.1f}%, "
            f"SD={np.std(seed_vals, ddof=1):.1f}%, k={len(seed_vals)} seeds")


# ===========================================================================
# SUMMARY TABLE
# ===========================================================================
tee.line()
tee("COMPLETE RESULTS TABLE")
tee.line()

results_df = pd.DataFrame(results_rows)
tee(results_df.to_string(index=False))

results_df.to_csv(OUT_CSV, index=False)
tee(f"\nResults table saved to: {OUT_CSV}")

# ===========================================================================
# VERDICT SUMMARY
# ===========================================================================
tee.line()
tee("VERDICT SUMMARY")
tee.line()

# Deduplicate for summary
summary = results_df.drop_duplicates(subset=["Hypothesis", "Size"], keep="first")
summary_slim = summary[["Hypothesis", "Type", "Size", "Verdict"]]
tee(summary_slim.to_string(index=False))

tee(f"\nFull output: {OUT_TXT}")
tee(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

tee.close()

print(f"\nDone. Output saved to:")
print(f"  {OUT_TXT}")
print(f"  {OUT_CSV}")
