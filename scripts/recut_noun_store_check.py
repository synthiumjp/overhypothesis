"""Exploratory check for the ACL recut (23 Oct 2026; not registered).

Compares the preference for repeating the context's shape and colour (Delta, nats; one exemplar,
same noun in the continuation) between novel test nouns and nouns from training, for every
fresh-text condition of Amendment 8. Seed-level means (sizes averaged within seed), t-based
95% intervals over the five seeds, two-sided one-sample tests.
Usage: python scripts/recut_noun_store_check.py > results/recut/noun_store_check.txt
"""
import numpy as np, pandas as pd
from scipy import stats
nv = pd.read_csv("results/amendment8/summary_per_run.csv")
tr = pd.read_csv("results/amendment8/trained_per_run.csv")
def seed(df, cond, col, match=True):
    d = df[(df.condition == cond) & (df.k == 1) & (df.match == match)]
    return d.groupby("seed")[col].mean()
def ci(x):
    x = np.asarray(x, float); m = x.mean()
    h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{m:6.2f} [{m-h:6.2f}, {m+h:6.2f}] p={stats.ttest_1samp(x, 0).pvalue:.2g}"
print("condition | dimension | novel nouns | trained nouns | novel - trained")
for c in sorted(nv.condition.unique()):
    for dim in ["shape", "colour"]:
        a, b = seed(nv, c, dim), seed(tr, c, dim)
        print(f"{c.replace('regular_f_','')} | {dim} | {ci(a)} | {ci(b)} | {ci(a-b)}")
print("\nTrained nouns, shape, same vs different noun in the continuation:")
for c in sorted(tr.condition.unique()):
    s, d = seed(tr, c, "shape", True), seed(tr, c, "shape", False)
    print(f"{c.replace('regular_f_','')} | same {ci(s)} | different {ci(d)}")

print("\nTrained nouns, seed-level contrasts of the index (same noun):")
def idx(c):
    d = tr[(tr.condition == "regular_f_" + c) & (tr.k == 1) & (tr.match == True)]
    return d.groupby("seed")["OI_colour"].mean()
for lab, a, b in [("grouped - random", "regular_grouped", "regular_random"),
                  ("B 3.5 - random", "regular_b3p6", "regular_random"),
                  ("Regular - No stable feature, grouped", "regular_grouped", "nostable_grouped"),
                  ("different - same object", "objects_diff", "objects_same")]:
    print(f"{lab} | {ci(idx(a) - idx(b))}")
for c in ["regular_random", "nostable_grouped"]:
    print(f"index {c} | {ci(idx(c))}")
