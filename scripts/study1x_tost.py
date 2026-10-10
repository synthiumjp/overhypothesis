#!/usr/bin/env python3
"""
study1x_tost.py  |  Registered follow-up to H13b (Amendment 5): "If H13b is not
supported, equivalence to zero is tested with TOST at +/-0.2 nats."

The rule was stated in the registration text but not coded in the locked
study1x_tests.py, so it is run here. Two one-sided tests on the same mixed model
as H13b (OI_colour ~ condition + size, random intercept for corpus seed), and,
alongside, on the paired differences over the 15 matched models.

Usage:  python scripts/study1x_tost.py --runs results/study1c/summary_per_run.csv \
            results/study1x/summary_per_run.csv --out results/study1x/tost_H13b.json
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
ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--bound", type=float, default=0.2)
ap.add_argument("--out", required=True)
a = ap.parse_args()

d = load(a.runs)
k1 = d[(d.k == 1) & d.match]
s = k1[k1.condition.isin(["regular_graded_b3p6", "regular"])].copy()
s["tr"] = (s.condition == "regular_graded_b3p6").astype(int)
m = smf.mixedlm("OI_colour ~ tr + C(size)", s, groups=s["seed"]).fit(reml=True)
est, se, b = m.params["tr"], m.bse["tr"], a.bound
p_lower = stats.norm.sf((est + b) / se)     # H0: diff <= -b
p_upper = stats.norm.cdf((est - b) / se)    # H0: diff >= +b
x = s[s.tr == 1].set_index(["size", "seed"]).OI_colour.sort_index()
y = s[s.tr == 0].set_index(["size", "seed"]).OI_colour.sort_index()
diff = x - y
n, md, sd = len(diff), diff.mean(), diff.std(ddof=1)
t_lo = stats.t.sf((md + b) / (sd / n ** 0.5), n - 1)
t_hi = stats.t.cdf((md - b) / (sd / n ** 0.5), n - 1)
out = {"bound": b,
       "mixed_model": {"est": est, "se": se, "p_lower": p_lower, "p_upper": p_upper,
                       "p_tost": max(p_lower, p_upper), "equivalent": bool(max(p_lower, p_upper) < .05),
                       "ci90": [est - 1.645 * se, est + 1.645 * se]},
       "paired": {"mean_diff": md, "sd": sd, "n": n, "p_lower": t_lo, "p_upper": t_hi,
                  "p_tost": max(t_lo, t_hi), "equivalent": bool(max(t_lo, t_hi) < .05)}}
json.dump(out, open(a.out, "w"), indent=2, default=float)
print(json.dumps(out, indent=2, default=float))
