#!/usr/bin/env python3
"""
study1b_tests.py  |  Registered tests for Study 1b (Amendment 1, osf.io/x6zhn).

Study 1b: one-shot test of the models trained on the original corpora (random
sentence order). Paper: Study 1, random order.

Confirmatory, k = 1, one-sided, 15 runs per condition (3 sizes x 5 seeds),
Holm-corrected across the three:
  H1b  Regular: OI_colour > 0                          (one-sample t)
  H2b  Regular > Scrambled on OI_colour                (Welch t)
  H3b  Regular: matched-noun OI_colour > mismatched    (paired t)

Equivalence (TOST, smallest effect of interest 0.2 nats) for the Regular
index and for the matched minus mismatched difference.

Exploratory: Feature-swap control, Regular > Feature-swap on OI_texture
(Welch t), as the ideal observer predicts.

Input is results/study1c/summary_per_run.csv, which holds every condition of
Studies 1b and 1c (see REPRODUCE.md for how it is regenerated).

Usage
  python scripts/study1b_tests.py --runs results/study1c/summary_per_run.csv \
         --out results/study1b/tests_output.txt
"""

import argparse

import pandas as pd
from scipy import stats

SESOI = 0.2


def holm(p):
    out, run = {}, 0.0
    for i, n in enumerate(sorted(p, key=p.get)):
        run = max(run, min(1.0, (len(p) - i) * p[n]))
        out[n] = run
    return out


def tost_one_sample(x, bound=SESOI):
    lo = stats.ttest_1samp(x, -bound, alternative="greater").pvalue
    hi = stats.ttest_1samp(x, bound, alternative="less").pvalue
    return max(lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/study1c/summary_per_run.csv")
    ap.add_argument("--out", default="results/study1b/tests_output.txt")
    a = ap.parse_args()

    d = pd.read_csv(a.runs)
    d["match"] = d["match"].astype(str).isin(["True", "true"])
    d = d[(d.source == "lm") & (d.k == 1)]
    key = ["condition", "size", "seed"]
    m = d[d.match].set_index(key)
    x = d[~d.match].set_index(key)

    reg = m.loc["regular", "OI_colour"]
    scr = m.loc["scrambled", "OI_colour"]
    bind = (m.loc["regular", "OI_colour"] - x.loc["regular", "OI_colour"])

    t1 = stats.ttest_1samp(reg, 0, alternative="greater")
    t2 = stats.ttest_ind(reg, scr, equal_var=False, alternative="greater")
    t3 = stats.ttest_1samp(bind, 0, alternative="greater")
    p = {"H1b": t1.pvalue, "H2b": t2.pvalue, "H3b": t3.pvalue}
    ph = holm(p)

    swap = m.loc["feature_swap", "OI_texture"]
    regt = m.loc["regular", "OI_texture"]
    t4 = stats.ttest_ind(regt, swap, equal_var=False, alternative="greater")

    lines = [
        "Study 1b registered tests (k = 1, one-sided, 15 runs per condition, Holm across 3)",
        f"H1b Regular OI_colour > 0: mean {reg.mean():.4f} (SD {reg.std():.4f}), "
        f"t = {t1.statistic:.3f}, p = {t1.pvalue:.4f}, Holm p = {ph['H1b']:.4f}",
        f"H2b Regular > Scrambled: {reg.mean():.4f} vs {scr.mean():.4f} (SD {scr.std():.4f}), "
        f"t = {t2.statistic:.3f}, p = {t2.pvalue:.4f}, Holm p = {ph['H2b']:.4f}",
        f"H3b matched > mismatched (Regular): diff {bind.mean():.4f} (SD {bind.std():.4f}), "
        f"t = {t3.statistic:.3f}, p = {t3.pvalue:.4f}, Holm p = {ph['H3b']:.4f}",
        "",
        f"TOST, Regular OI_colour within +/-{SESOI}: p = {tost_one_sample(reg):.4f}",
        f"TOST, matched minus mismatched within +/-{SESOI}: p = {tost_one_sample(bind):.3g}",
        "",
        "Exploratory: Feature-swap control, OI_texture",
        f"Regular {regt.mean():.4f} vs Feature-swap {swap.mean():.4f}, "
        f"Welch t = {t4.statistic:.3f}, p = {t4.pvalue:.4f}",
    ]
    text = "\n".join(lines)
    print(text)
    with open(a.out, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
