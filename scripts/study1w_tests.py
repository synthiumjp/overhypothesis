#!/usr/bin/env python3
"""
study1w_tests.py  |  Registered tests for the Study 1 window test (Amendment 6).

New models (15 each, 3 sizes x 5 seeds, trained exactly as in Study 1 except
--seq_len 512 --keep_tokens_per_step), labelled by make-manifest --label_suffix:
  regular_w512                    Regular, random order (data/corpora)
  regular_graded_b3p6_w512        Regular, B = 3.6 (data/corpora_x/graded_b3p6)
  regular_nostable_random_w512    NoStable, random order (data/corpora_x/nostable_random)
Comparison models trained with 128-token windows (Study 1 and Amendment 5):
  regular, regular_graded_b3p6

Primary measure: OI_colour at k = 1, same noun (as in Studies 1 and 1x).

Confirmatory, one-sided, Holm across the three:
  H17  regular_w512 > regular                       (longer window helps random order)
  H18  regular_graded_b3p6_w512 > regular_graded_b3p6  (child-level clustering with a longer window)
  H19  regular_w512 > regular_nostable_random_w512  (the gain still needs a stable dimension)
Test: linear mixed model OI ~ condition + size, random intercept for corpus seed,
one-sided z on the condition coefficient (as in study1x_tests.py); Welch, paired
(size x seed) and seed-paired t tests reported alongside.

Secondary (no correction):
  S24  co-occurrence prediction: for regular_w512 and regular_graded_b3p6_w512, the
       observed mean with its 95% t interval over the 15 models against the value
       predicted before training (results/study1w/prediction.json). Reading:
       "consistent" if the interval contains the prediction, "below" if it lies
       wholly below, "above" if wholly above.
  S25  OI_texture for H17 to H19
  S26  indices after two and four exemplars, new conditions
  S27  size x window interaction for H17
  S28  kind-specific index with trained nouns, new conditions (if --trained given)
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
CONTRASTS = [("H17", "regular_w512", "regular"),
             ("H18", "regular_graded_b3p6_w512", "regular_graded_b3p6"),
             ("H19", "regular_w512", "regular_nostable_random_w512")]
NEW = ["regular_w512", "regular_graded_b3p6_w512", "regular_nostable_random_w512"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--prediction", default="results/study1w/prediction.json")
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

    sec = {}
    pred = json.load(open(a.prediction))["predictions_W512"]
    s24 = {}
    for c in ("regular_w512", "regular_graded_b3p6_w512"):
        x = k1[k1.condition == c].OI_colour
        m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
        t = stats.t.ppf(0.975, len(x) - 1)
        lo, hi, p = m - t * se, m + t * se, pred[c]["predicted_OI"]
        s24[c] = {"observed": m, "ci95": [lo, hi], "predicted": p,
                  "reading": "consistent" if lo <= p <= hi else ("below" if hi < p else "above")}
    sec["S24_cooccurrence_prediction"] = s24
    for name, x, y in CONTRASTS:
        sec["S25_" + name + "_texture"] = contrast(k1, x, y, "OI_texture")
    dm = d[d.match & d.condition.isin(NEW)]
    sec["S26_by_k"] = {f"{c}|k{k}": v for (c, k), v in
                       dm.groupby(["condition", "k"])[["OI_colour", "OI_texture"]].mean().round(4)
                       .to_dict("index").items()}
    s = k1[k1.condition.isin(["regular_w512", "regular"])].copy()
    s["tr"] = (s.condition == "regular_w512").astype(int)
    mi = smf.mixedlm("OI_colour ~ tr * C(size)", s, groups=s["seed"]).fit(reml=True)
    sec["S27_size_by_window"] = {k: [float(mi.params[k]), float(mi.pvalues[k])]
                                 for k in mi.params.index if "tr" in k}
    if a.trained:
        t = load(a.trained)
        t1 = t[(t.k == 1) & t.condition.isin(NEW)]
        wt = t1.pivot_table(index=["condition", "size", "seed"], columns="match", values="OI_colour")
        diff = (wt[True] - wt[False]).reset_index(name="ks")
        sec["S28_trained_nouns_kind_specific"] = {
            c: {"mean": g.ks.mean(), "sd": g.ks.std(),
                "t": stats.ttest_1samp(g.ks, 0, alternative="greater").statistic,
                "p_one_sided": stats.ttest_1samp(g.ks, 0, alternative="greater").pvalue, "n_runs": len(g)}
            for c, g in diff.groupby("condition")}
    json.dump({"confirmatory": res, "secondary": sec}, open(a.out + ".json", "w"), indent=2, default=float)
    print(json.dumps({"confirmatory": res, "S24": s24}, indent=2, default=float))


if __name__ == "__main__":
    main()
