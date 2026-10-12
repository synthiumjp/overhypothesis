#!/usr/bin/env python3
"""
study1w_tost.py  |  Registered follow-up to H17 (Amendment 6): "If H17 is not
supported, equivalence to zero is tested with TOST at +/-0.2 nats on the same
model." Same mixed model as H17 (OI_colour ~ condition + size, random intercept
for corpus seed); paired differences over the 15 matched models reported alongside.
Also run, as exploratory, for H18.

Usage:  python scripts/study1w_tost.py --runs results/study1c/summary_per_run.csv \
            results/study1x/summary_per_run.csv results/study1w/summary_per_run.csv \
            --out results/study1w/tost.json
"""
import argparse
import json
import os
import sys
import warnings

import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study1x_tests import load  # noqa: E402

warnings.filterwarnings("ignore")
PAIRS = {"H17_registered": ("regular_w512", "regular"),
         "H18_exploratory": ("regular_graded_b3p6_w512", "regular_graded_b3p6")}

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--bound", type=float, default=0.2)
ap.add_argument("--out", required=True)
a = ap.parse_args()
d = load(a.runs)
k1 = d[(d.k == 1) & d.match]
b = a.bound
out = {"bound": b}
for name, (xc, yc) in PAIRS.items():
    s = k1[k1.condition.isin([xc, yc])].copy()
    s["tr"] = (s.condition == xc).astype(int)
    m = smf.mixedlm("OI_colour ~ tr + C(size)", s, groups=s["seed"]).fit(reml=True)
    est, se = m.params["tr"], m.bse["tr"]
    pl, pu = stats.norm.sf((est + b) / se), stats.norm.cdf((est - b) / se)
    x = s[s.tr == 1].set_index(["size", "seed"]).OI_colour.sort_index()
    y = s[s.tr == 0].set_index(["size", "seed"]).OI_colour.sort_index()
    diff = x - y
    n, md, sd = len(diff), diff.mean(), diff.std(ddof=1)
    tl = stats.t.sf((md + b) / (sd / n ** 0.5), n - 1)
    tu = stats.t.cdf((md - b) / (sd / n ** 0.5), n - 1)
    out[name] = {"mixed_model": {"est": est, "se": se, "p_tost": max(pl, pu),
                                 "equivalent": bool(max(pl, pu) < .05),
                                 "ci90": [est - 1.645 * se, est + 1.645 * se]},
                 "paired": {"mean_diff": md, "sd": sd, "n": n, "p_tost": max(tl, tu),
                            "equivalent": bool(max(tl, tu) < .05)}}
json.dump(out, open(a.out, "w"), indent=2, default=float)
print(json.dumps(out, indent=2, default=float))
