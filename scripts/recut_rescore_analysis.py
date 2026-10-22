#!/usr/bin/env python3
"""
recut_rescore_analysis.py  |  Summarise recut_rescore.py output (exploratory; not registered)

For each fresh-text condition, the one-exemplar shape-over-colour index and the
noun-specific index under four scorings:
  original / full   - as reported (must reproduce results/amendment8)
  typical  / full   - novel nouns given typical embeddings
  original / word   - Delta on the changed word only
  typical  / word   - both changes
and seed-level contrasts (sizes averaged within seed; paired t over five seeds).

Usage:
  python scripts/recut_rescore_analysis.py --raw results/recut/rescore_raw.csv --out results/recut/RESCORE.md
"""
import argparse, glob, os
import numpy as np, pandas as pd
from scipy import stats

ORDER = ["regular_random", "regular_b3p6", "regular_grouped", "nostable_grouped",
         "objects_same", "objects_diff", "reassigned_random", "reassigned_grouped"]
CONTRASTS = [  # label, a, b, quantity  (a minus b; b None = against zero)
    ("Grouped minus random", "regular_grouped", "regular_random", "index"),
    ("B 3.5 minus random", "regular_b3p6", "regular_random", "index"),
    ("Grouped Regular minus grouped No stable feature", "regular_grouped", "nostable_grouped", "index"),
    ("Different-object minus same-object runs", "objects_diff", "objects_same", "index"),
    ("Random-order index", "regular_random", None, "index"),
    ("No stable feature index", "nostable_grouped", None, "index"),
    ("Noun-specific index, Reassigned grouped", "reassigned_grouped", None, "noun_specific"),
    ("Noun-specific: Reassigned minus Regular, grouped", "reassigned_grouped", "regular_grouped", "noun_specific"),
]
SCORINGS = [("original", "full"), ("typical", "full"), ("original", "word"), ("typical", "word")]


def short(c):
    return c.replace("regular_f_", "")


def per_seed(raw):
    d = raw.melt(id_vars=["condition", "size", "seed", "emb", "item_id", "match", "probe_dim"],
                 value_vars=["delta_full", "delta_word"], var_name="measure", value_name="delta")
    d["measure"] = d["measure"].str.replace("delta_", "")
    run = (d.groupby(["condition", "size", "seed", "emb", "measure", "match", "probe_dim"])["delta"]
             .mean().unstack("probe_dim").reset_index())
    run["index"] = run["shape"] - run["colour"]
    w = run.pivot_table(index=["condition", "size", "seed", "emb", "measure"], columns="match",
                        values=["index", "shape", "colour"]).reset_index()
    w.columns = ["_".join(str(x) for x in c if str(x) != "").replace("_True", "").replace("_False", "_diffnoun")
                 if isinstance(c, tuple) else c for c in w.columns]
    w["noun_specific"] = w["index"] - w["index_diffnoun"]
    w["condition"] = w["condition"].map(short)
    return (w.groupby(["condition", "seed", "emb", "measure"])
              [["index", "noun_specific", "shape", "colour"]].mean().reset_index())


def ci(x):
    x = np.asarray(x, float)
    m, n = x.mean(), len(x)
    h = stats.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    p = stats.ttest_1samp(x, 0).pvalue if n > 1 else float("nan")
    return m, m - h, m + h, p, int((x > 0).sum()), n


def check_reproduction(raw):
    files = glob.glob("results/amendment8/raw/lm_*.csv*")
    if not files:
        return "Reference scores not found; reproduction not checked."
    ref = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ref = ref[ref.k == 1][["ckpt", "item_id", "delta"]]
    m = raw[raw.emb == "original"].merge(ref, on=["ckpt", "item_id"])
    if m.empty:
        return "No overlapping rows with results/amendment8/raw (checkpoint paths differ?)."
    return (f"Reproduction of results/amendment8/raw with original embeddings: {len(m)} rows, "
            f"max absolute difference {np.abs(m.delta_full - m.delta).max():.2e} nats.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="results/recut/rescore_raw.csv")
    p.add_argument("--out", default="results/recut/RESCORE.md")
    a = p.parse_args()
    raw = pd.read_csv(a.raw)
    raw["match"] = raw["match"].astype(str).isin(["True", "true", "1"])
    s = per_seed(raw)
    s.to_csv(os.path.splitext(a.out)[0] + "_per_seed.csv", index=False)

    L = ["# Exploratory rescoring for the ACL recut", "",
         "Not registered. Fresh-text models, one exemplar (k = 1). Seed-level values: sizes averaged "
         "within seed, t-based 95% intervals over the seeds, two-sided tests.", "",
         check_reproduction(raw), "",
         "Scorings: emb = original (untrained novel-noun embeddings, as reported) or typical "
         "(mean training-noun embedding plus a distinct offset of typical size); "
         "measure = full (whole continuation, as reported) or word (changed word only).", "",
         "## Index and noun-specific index by condition (mean of seed means)", "",
         "| Condition | " + " | ".join(f"Index {e}/{m}" for e, m in SCORINGS) + " | "
         + " | ".join(f"Noun-specific {e}/{m}" for e, m in SCORINGS) + " |",
         "|---|" + "---|" * (2 * len(SCORINGS))]
    for c in [c for c in ORDER if c in set(s.condition)]:
        cells = []
        for q in ("index", "noun_specific"):
            for e, m in SCORINGS:
                v = s[(s.condition == c) & (s.emb == e) & (s.measure == m)][q]
                cells.append(f"{v.mean():.2f}" if len(v) else "")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["", "## Seed-level contrasts", "",
          "| Contrast | Scoring | Estimate [95% CI] | p | Seeds > 0 |", "|---|---|---|---|---|"]
    for lab, ca, cb, q in CONTRASTS:
        for e, m in SCORINGS:
            A = s[(s.condition == ca) & (s.emb == e) & (s.measure == m)].set_index("seed")[q]
            if A.empty:
                continue
            x = A if cb is None else (A - s[(s.condition == cb) & (s.emb == e) & (s.measure == m)]
                                       .set_index("seed")[q]).dropna()
            mm, lo, hi, pv, pos, n = ci(x)
            L.append(f"| {lab} | {e}/{m} | {mm:.2f} [{lo:.2f}, {hi:.2f}] | {pv:.3g} | {pos}/{n} |")
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
