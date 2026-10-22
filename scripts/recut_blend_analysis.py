#!/usr/bin/env python3
"""
recut_blend_analysis.py  |  Noun-familiarity curve (exploratory; not registered)

Combines results/recut/rescore_raw.csv (blend 0 = original, 1 = typical) with
results/recut/rescore_blend.csv (intermediate blends) and reports, for each condition and
blend, seed-level means of Delta shape, Delta colour, the index and the noun-specific index
(whole-continuation Delta, k = 1), plus per-seed linear slopes of each against the blend.

Usage:
  python scripts/recut_blend_analysis.py --out results/recut/BLEND.md
"""
import argparse, os
import numpy as np, pandas as pd
from scipy import stats

ORDER = ["regular_random", "regular_grouped", "objects_same", "objects_diff", "reassigned_grouped"]


def load(paths):
    d = pd.concat([pd.read_csv(p) for p in paths if os.path.exists(p)], ignore_index=True)
    d["match"] = d["match"].astype(str).isin(["True", "true", "1"])
    d["alpha"] = d["emb"].map(lambda e: 0.0 if e == "original" else 1.0 if e == "typical"
                              else float(e[5:]))
    d["condition"] = d["condition"].str.replace("regular_f_", "", regex=False)
    return d[d.condition.isin(ORDER)]


def per_seed(d):
    run = (d.groupby(["condition", "size", "seed", "alpha", "match", "probe_dim"])["delta_full"]
             .mean().unstack("probe_dim").reset_index())
    run["index"] = run["shape"] - run["colour"]
    same = run[run.match].set_index(["condition", "size", "seed", "alpha"])
    diff = run[~run.match].set_index(["condition", "size", "seed", "alpha"])
    same = same.assign(noun_specific=same["index"] - diff["index"]).reset_index()
    return (same.groupby(["condition", "seed", "alpha"])[["shape", "colour", "index", "noun_specific"]]
                .mean().reset_index())


def ci(x):
    x = np.asarray(x, float)
    m, n = x.mean(), len(x)
    h = stats.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return f"{m:.2f} [{m - h:.2f}, {m + h:.2f}]", stats.ttest_1samp(x, 0).pvalue


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+",
                   default=["results/recut/rescore_raw.csv", "results/recut/rescore_blend.csv"])
    p.add_argument("--out", default="results/recut/BLEND.md")
    a = p.parse_args()
    s = per_seed(load(a.inputs))
    s.to_csv(os.path.splitext(a.out)[0] + "_per_seed.csv", index=False)
    alphas = sorted(s.alpha.unique())
    L = ["# Noun-familiarity curve (exploratory)", "",
         "Novel-noun input embeddings blended between their original untrained values (0) and the typical "
         "embedding (1: mean training-noun embedding plus a distinct offset of typical size). Fresh-text "
         "models, k = 1, whole-continuation Delta. Means of seed means; slopes are per-seed least-squares "
         "slopes against the blend, with t-based 95% intervals over five seeds.", ""]
    for q, lab in [("index", "Index"), ("shape", "Delta shape"), ("colour", "Delta colour"),
                   ("noun_specific", "Noun-specific index")]:
        L += [f"## {lab}", "", "| Condition | " + " | ".join(f"{x:g}" for x in alphas) + " | Slope [95% CI] | p |",
              "|---|" + "---|" * (len(alphas) + 2)]
        for c in ORDER:
            sc = s[s.condition == c]
            if sc.empty:
                continue
            means = [sc[sc.alpha == x][q].mean() for x in alphas]
            slopes = [np.polyfit(g.alpha, g[q], 1)[0] for _, g in sc.groupby("seed") if g.alpha.nunique() > 1]
            sl, pv = ci(slopes) if len(slopes) > 1 else ("", float("nan"))
            L.append(f"| {c} | " + " | ".join(f"{m:.2f}" for m in means) + f" | {sl} | {pv:.3g} |")
        L.append("")
    L += ["## Grouped minus random (index) by blend", "", "| Blend | Estimate [95% CI] | p |", "|---|---|---|"]
    for x in alphas:
        g = s[(s.condition == "regular_grouped") & (s.alpha == x)].set_index("seed")["index"]
        r = s[(s.condition == "regular_random") & (s.alpha == x)].set_index("seed")["index"]
        dd = (g - r).dropna()
        if len(dd) > 1:
            e, pv = ci(dd)
            L.append(f"| {x:g} | {e} | {pv:.3g} |")
    norms = "results/recut/norms.csv"
    if os.path.exists(norms):
        n = pd.read_csv(norms)
        L += ["", "## Embedding norms (mean over models)", "",
              f"Trained nouns {n.trained_norm.mean():.3f}; novel nouns, original {n.novel_norm.mean():.3f}; "
              f"typical {n.typical_norm.mean():.3f}."]
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
