#!/usr/bin/env python3
"""
study1x_tests.py  |  Registered tests for the Study 1 extension (Amendment 5).

Inputs
  --runs     per-run summaries (oneshot_eval.py summarise *_per_run.csv) holding
             the Study 1 conditions regular and regular_bursty and the new
             conditions regular_nostable_random, regular_nostable_grouped,
             regular_colourstable_random, regular_colourstable_grouped,
             regular_graded_b1p4, regular_graded_b3p6, regular_graded_b7,
             regular_graded_b12 (labels from make-manifest --label_suffix)
  --trained  per-run summaries of the trained-noun items (S14), optional
  --manifest data/corpora_x/manifest.json (measured B of every corpus)

Primary measure: OI_colour at k = 1, same noun (as in Study 1).

Confirmatory, one-sided, Holm across the four:
  H11   grouped Regular > grouped NoStable
  H12   grouped ColourStable < grouped NoStable
  H13a  across the six Regular orders (random, B 1.4, 3.6, 7, 12, grouped),
        OI_colour rises with log B
  H13b  Regular at B 3.6 (CHILDES level) > Regular in random order
Test for H11, H12, H13b: linear mixed model OI ~ condition + size with a
random intercept for corpus seed (the three sizes trained on one seed share
a corpus); one-sided z on the condition coefficient. H13a: OI ~ log B + size,
same random intercept, one-sided on the slope. Reported alongside: Welch t
over the 15 runs, paired t over the 15 matched runs, and paired t over the
five seed means.

Secondary (no correction): OI_texture for the same contrasts; the
kind-specific index (same minus different noun) for every condition; size by
condition interaction; S14, the kind-specific index with trained nouns.
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
ORDERS = {"regular": "random", "regular_graded_b1p4": "b1p4", "regular_graded_b3p6": "b3p6",
          "regular_graded_b7": "b7", "regular_graded_b12": "b12", "regular_bursty": "grouped"}
CORPUS_OF = {"regular": "regular_random", "regular_bursty": "regular_grouped",
             "regular_graded_b1p4": "graded_b1p4", "regular_graded_b3p6": "graded_b3p6",
             "regular_graded_b7": "graded_b7", "regular_graded_b12": "graded_b12"}
CONTRASTS = [("H11", "regular_bursty", "regular_nostable_grouped", "greater"),
             ("H12", "regular_nostable_grouped", "regular_colourstable_grouped", "greater"),
             ("H13b", "regular_graded_b3p6", "regular", "greater")]


def holm(p):
    out, run = {}, 0.0
    for i, n in enumerate(sorted(p, key=p.get)):
        run = max(run, min(1.0, (len(p) - i) * p[n]))
        out[n] = run
    return out


def load(path):
    d = pd.concat([pd.read_csv(p) for p in path], ignore_index=True)
    d["match"] = d["match"].astype(str).isin(["True", "true", "1"])
    return d[d.source == "lm"]


def contrast(k1, x_c, y_c, col):
    s = k1[k1.condition.isin([x_c, y_c])].copy()
    s["tr"] = (s.condition == x_c).astype(int)
    m = smf.mixedlm(f"{col} ~ tr + C(size)", s, groups=s["seed"]).fit(reml=True)
    x = s[s.tr == 1].set_index(["size", "seed"])[col]
    y = s[s.tr == 0].set_index(["size", "seed"])[col]
    x, y = x.sort_index(), y.sort_index()
    welch = stats.ttest_ind(x, y, equal_var=False, alternative="greater")
    paired = stats.ttest_rel(x, y, alternative="greater")
    seed_x, seed_y = x.groupby("seed").mean(), y.groupby("seed").mean()
    seedp = stats.ttest_rel(seed_x, seed_y, alternative="greater")
    sd = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    return {"mean_x": x.mean(), "mean_y": y.mean(), "diff": x.mean() - y.mean(),
            "d": (x.mean() - y.mean()) / sd,
            "mlm_est": m.params["tr"], "mlm_se": m.bse["tr"],
            "mlm_ci95": [m.params["tr"] - 1.96 * m.bse["tr"], m.params["tr"] + 1.96 * m.bse["tr"]],
            "p_one_sided": float(stats.norm.sf(m.params["tr"] / m.bse["tr"])),
            "welch_t": welch.statistic, "welch_p": welch.pvalue,
            "paired_t": paired.statistic, "paired_p": paired.pvalue,
            "seed_paired_t": seedp.statistic, "seed_paired_p": seedp.pvalue,
            "runs_positive": int(((x - y) > 0).sum()), "n_runs": len(x)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--trained", nargs="*", default=[])
    ap.add_argument("--manifest", default="data/corpora_x/manifest.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = load(a.runs)
    k1 = d[(d.k == 1) & d.match]
    res = {}
    for name, x_c, y_c, _ in CONTRASTS:
        res[name] = contrast(k1, x_c, y_c, "OI_colour")
    # H13a: dose-response over measured clustering
    man = pd.DataFrame(json.load(open(a.manifest)))
    B = man.groupby(["corpus", "seed"]).B.first()
    g = k1[k1.condition.isin(ORDERS)].copy()
    g["logB"] = [np.log(B[(CORPUS_OF[c], int(s))]) for c, s in zip(g.condition, g.seed)]
    m = smf.mixedlm("OI_colour ~ logB + C(size)", g, groups=g["seed"]).fit(reml=True)
    rho = stats.spearmanr(g.logB, g.OI_colour)[0]
    res["H13a"] = {"slope": m.params["logB"], "se": m.bse["logB"],
                   "p_one_sided": float(stats.norm.sf(m.params["logB"] / m.bse["logB"])),
                   "spearman_runs": rho,
                   "means_by_order": g.groupby("condition").OI_colour.mean().to_dict()}
    h = holm({k: v["p_one_sided"] for k, v in res.items()})
    for k in res:
        res[k]["p_holm"] = h[k]
        res[k]["supported"] = bool(h[k] < 0.05)

    sec = {}
    for name, x_c, y_c, _ in CONTRASTS:
        sec[name + "_texture"] = contrast(k1, x_c, y_c, "OI_texture")
    # Kind-specific index (same minus different noun) per condition
    k1all = d[d.k == 1]
    w = k1all.pivot_table(index=["condition", "size", "seed"], columns="match", values="OI_colour")
    ks = (w[True] - w[False]).groupby("condition").agg(["mean", "std"])
    sec["kind_specific_novel_nouns"] = ks.round(4).to_dict()
    # Size by condition interaction for the grouping effect
    s = k1[k1.condition.isin(["regular", "regular_bursty"])].copy()
    s["tr"] = (s.condition == "regular_bursty").astype(int)
    mi = smf.mixedlm("OI_colour ~ tr * C(size)", s, groups=s["seed"]).fit(reml=True)
    sec["size_by_grouping"] = {k: [float(mi.params[k]), float(mi.pvalues[k])]
                               for k in mi.params.index if "tr" in k}
    if a.trained:
        t = load(a.trained)
        t1 = t[t.k == 1]
        wt = t1.pivot_table(index=["condition", "size", "seed"], columns="match", values="OI_colour")
        diff = (wt[True] - wt[False]).reset_index(name="ks")
        out = {}
        for c, gg in diff.groupby("condition"):
            tt = stats.ttest_1samp(gg.ks, 0, alternative="greater")
            out[c] = {"mean": gg.ks.mean(), "sd": gg.ks.std(), "t": tt.statistic,
                      "p_one_sided": tt.pvalue, "n_runs": len(gg)}
        sec["S14_trained_nouns_kind_specific"] = out
    json.dump({"confirmatory": res, "secondary": sec}, open(a.out + ".json", "w"),
              indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
