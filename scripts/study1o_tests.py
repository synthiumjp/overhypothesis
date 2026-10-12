#!/usr/bin/env python3
"""
study1o_tests.py  |  Registered tests for Amendment 7 (same-object vs different-object
clustering).

Conditions (labels from make-manifest --conditions regular --label_suffix _o_<name>),
15 models each (3 sizes x 5 seeds), corpora in data/corpora_o:
  regular_o_random, regular_o_same_grouped, regular_o_diff_grouped,
  regular_o_same_b3p6, regular_o_diff_b3p6

Primary measure: OI_colour at k = 1, same noun (as in Study 1).

Confirmatory, one-sided, Holm across the three:
  H20  diff_grouped > same_grouped
  H21  diff_b3p6    > same_b3p6
  H22  diff_b3p6    > random
Test: as in study1x_tests.py (mixed model OI ~ condition + size, random intercept for
corpus seed; one-sided z), with Welch, paired and seed-paired t tests alongside.

Secondary (no correction):
  S29  same_b3p6 vs random, two-sided, and TOST at +/-0.2 nats
  S30  diff_grouped vs random and same_grouped vs random, two-sided
  S31  components (shape, colour, texture deltas) per condition
  S32  H20 to H22 on OI_texture
  S33  kind-specific index with trained nouns per condition (if --trained)
  S34  regular_o_random vs Study 1 regular (random order), two-sided: do the new
       corpora behave like Study 1's?
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study1x_tests import contrast, holm, load  # noqa: E402

warnings.filterwarnings("ignore")
P = "regular_o_"
CONTRASTS = [("H20", P + "diff_grouped", P + "same_grouped"),
             ("H21", P + "diff_b3p6", P + "same_b3p6"),
             ("H22", P + "diff_b3p6", P + "random")]
NEW = [P + c for c in ("random", "same_grouped", "diff_grouped", "same_b3p6", "diff_b3p6")]


def two_sided(k1, x, y, col="OI_colour", bound=None):
    r = contrast(k1, x, y, col)
    z = r["mlm_est"] / r["mlm_se"]
    r["p_two_sided"] = float(2 * stats.norm.sf(abs(z)))
    if bound:
        lo = stats.norm.sf((r["mlm_est"] + bound) / r["mlm_se"])
        hi = stats.norm.cdf((r["mlm_est"] - bound) / r["mlm_se"])
        r["tost_p"] = float(max(lo, hi))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--trained", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = load(a.runs)
    k1 = d[(d.k == 1) & d.match]
    for c in NEW:
        n = int((k1.condition == c).sum())
        if n != 15:
            print(f"WARNING: {c} has {n} runs, expected 15")
    res = {name: contrast(k1, x, y, "OI_colour") for name, x, y in CONTRASTS}
    h = holm({k: v["p_one_sided"] for k, v in res.items()})
    for k in res:
        res[k]["p_holm"] = h[k]
        res[k]["supported"] = bool(h[k] < 0.05)

    sec = {"S29_same_b3p6_vs_random": two_sided(k1, P + "same_b3p6", P + "random", bound=0.2),
           "S30_diff_grouped_vs_random": two_sided(k1, P + "diff_grouped", P + "random"),
           "S30_same_grouped_vs_random": two_sided(k1, P + "same_grouped", P + "random")}
    g = k1[k1.condition.isin(NEW)].groupby("condition")
    sec["S31_components"] = g[["shape", "colour", "texture", "OI_colour", "OI_texture"]].mean().round(4).to_dict("index")
    sec["S31_by_size"] = k1[k1.condition.isin(NEW)].pivot_table(
        index="condition", columns="size", values="OI_colour").round(4).to_dict("index")
    for name, x, y in CONTRASTS:
        sec["S32_" + name + "_texture"] = contrast(k1, x, y, "OI_texture")
    if a.trained:
        t = load(a.trained)
        t1 = t[(t.k == 1) & t.condition.isin(NEW)]
    if a.trained and len(t1):
        wt = t1.pivot_table(index=["condition", "size", "seed"], columns="match", values="OI_colour")
        diff = (wt[True] - wt[False]).reset_index(name="ks")
        sec["S33_trained_nouns_kind_specific"] = {
            c: {"mean": gg.ks.mean(), "sd": gg.ks.std(),
                "p_one_sided": stats.ttest_1samp(gg.ks, 0, alternative="greater").pvalue, "n_runs": len(gg)}
            for c, gg in diff.groupby("condition")}
    if (k1.condition == "regular").any():
        x = k1[k1.condition == P + "random"].OI_colour
        y = k1[k1.condition == "regular"].OI_colour
        sec["S34_o_random_vs_study1_random"] = {"mean_o": x.mean(), "mean_study1": y.mean(),
                                                "welch_p_two_sided": stats.ttest_ind(x, y, equal_var=False).pvalue}
    json.dump({"confirmatory": res, "secondary": sec}, open(a.out + ".json", "w"), indent=2, default=float)
    print(json.dumps({"confirmatory": {k: {m: v[m] for m in ("mean_x", "mean_y", "mlm_est", "mlm_ci95", "p_holm",
                                                            "supported", "runs_positive")} for k, v in res.items()},
                      "S29": {m: sec["S29_same_b3p6_vs_random"][m] for m in ("mlm_est", "p_two_sided", "tost_p")}},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
