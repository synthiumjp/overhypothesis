#!/usr/bin/env python3
"""
study1c_tests.py  |  Registered tests for Study 1c (Amendments 1 and 2).

Confirmatory, k = 1, matched noun, one-sided Welch t-tests over 15 runs per
condition (3 sizes x 5 seeds), Holm-corrected across the three:
  H4b          Regular-bursty > Regular            on OI_colour
  H5b colour   Regular-bursty > Scrambled-bursty   on OI_colour
  H5b texture  Regular-bursty > Scrambled-bursty   on OI_texture

Robustness (registered): mixed model, OI ~ condition + size, random intercept
for seed (runs with one seed share a corpus).

Exploratory: paired differences by size and seed, noun binding (matched minus
mismatched), the k trend, and the ideal observer on the same items.

Usage
  python scripts/study1c_tests.py --runs results/study1c/summary_per_run.csv \\
         --hbm "results/study1c/hbm/hbm_*.csv"
"""

import argparse
import glob
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

TESTS = [("H4b", "regular_bursty", "regular", "OI_colour"),
         ("H5b_colour", "regular_bursty", "scrambled_bursty", "OI_colour"),
         ("H5b_texture", "regular_bursty", "scrambled_bursty", "OI_texture")]


def holm(p):
    out, run = {}, 0.0
    for i, n in enumerate(sorted(p, key=p.get)):
        run = max(run, min(1.0, (len(p) - i) * p[n]))
        out[n] = run
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--hbm", default=None)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")

    d = pd.read_csv(a.runs)
    d["match"] = d["match"].astype(str).isin(["True", "true", "1"])
    k1 = d[(d.k == 1) & d.match]

    print("Confirmatory (one-sided Welch, k = 1, matched noun)")
    p = {}
    rows = []
    for name, x_c, y_c, col in TESTS:
        x = k1[k1.condition == x_c][col].to_numpy()
        y = k1[k1.condition == y_c][col].to_numpy()
        t, pv = stats.ttest_ind(x, y, equal_var=False, alternative="greater")
        sd = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
        p[name] = pv
        s = k1[k1.condition.isin([x_c, y_c])].copy()
        s["tr"] = (s.condition == x_c).astype(int)
        m = smf.mixedlm(f"{col} ~ tr + C(size)", s, groups=s["seed"]).fit(reml=True)
        w = s.pivot_table(index=["size", "seed"], columns="condition", values=col)
        rows.append({"test": name, "mean_x": x.mean(), "sd_x": x.std(ddof=1),
                     "mean_y": y.mean(), "sd_y": y.std(ddof=1), "t": t, "p": pv,
                     "d": (x.mean() - y.mean()) / sd,
                     "mlm_est": m.params["tr"], "mlm_se": m.bse["tr"],
                     "mlm_p_one_sided": stats.norm.sf(m.params["tr"] / m.bse["tr"]),
                     "pairs_positive": int(((w[x_c] - w[y_c]) > 0).sum())})
    h = holm(p)
    res = pd.DataFrame(rows)
    res["p_holm"] = res["test"].map(h)
    res["supported"] = res["p_holm"] < 0.05
    with pd.option_context("display.width", 200):
        print(res.round(4).to_string(index=False))

    print("\nNoun binding (matched minus mismatched OI_colour, k = 1)")
    k1all = d[d.k == 1]
    for c in ("regular", "regular_bursty", "scrambled_bursty"):
        w = k1all[k1all.condition == c].pivot_table(index=["size", "seed"], columns="match",
                                                     values="OI_colour")
        diff = w[True] - w[False]
        t, pv = stats.ttest_1samp(diff, 0)
        print(f"  {c:17s} diff {diff.mean():+.4f} (SD {diff.std():.4f}), t = {t:.2f}, p = {pv:.3f}")

    print("\nBy size, k = 1, matched")
    print(k1.groupby(["condition", "size"])[["OI_colour", "OI_texture", "shape", "colour",
                                              "texture"]].mean().round(3).to_string())
    print("\nk trend, matched")
    print(d[d.match].groupby(["condition", "k"])[["OI_colour", "OI_texture"]].mean()
          .round(3).to_string())

    if a.hbm:
        hb = pd.concat(pd.read_csv(f) for f in glob.glob(a.hbm))
        hb = hb[(hb.k == 1) & hb.match.astype(str).isin(["True", "true"])]
        w = hb.groupby(["condition", "seed", "probe_dim"]).delta.mean().unstack()
        w["OI_colour"] = w["shape"] - w["colour"]
        w["OI_texture"] = w["shape"] - w["texture"]
        print("\nIdeal observer, k = 1, matched (replication-corrected counts)")
        print(w.groupby("condition").mean().round(3).to_string())


if __name__ == "__main__":
    main()
