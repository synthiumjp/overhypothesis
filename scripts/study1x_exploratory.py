#!/usr/bin/env python3
"""
study1x_exploratory.py  |  Exploratory summaries for the Study 1 extension
(Amendment 5, Part A). Not registered; written after the registered tests.

  <out>_by_condition.csv  OI_colour (k = 1, same noun) by condition and size, with
                          the run SD, runs above zero, OI_texture and the shape,
                          colour and texture components
  <out>_hbm.csv           ideal observer (k = 1, same noun) for every new corpus
  <out>_doseresponse.json H13a refitted with the registered model (OI ~ log B + size,
                          random intercept for seed) on (a) all six orders and
                          (b) the five orders without the fully grouped corpus, so
                          the slope does not rest on one end point

Usage:  python scripts/study1x_exploratory.py --runs results/study1c/summary_per_run.csv \
            results/study1x/summary_per_run.csv --hbm "results/study1x/hbm/hbm_*.csv" \
            --out results/study1x/exploratory
"""
import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from study1x_tests import CORPUS_OF, ORDERS, load  # noqa: E402

warnings.filterwarnings("ignore")
KEEP = ["regular", "regular_graded_b1p4", "regular_graded_b3p6", "regular_graded_b7",
        "regular_graded_b12", "regular_bursty", "regular_nostable_random", "regular_nostable_grouped",
        "regular_colourstable_random", "regular_colourstable_grouped", "scrambled", "scrambled_bursty"]

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--hbm", required=True)
ap.add_argument("--manifest", default="data/corpora_x/manifest.json")
ap.add_argument("--out", required=True)
a = ap.parse_args()

d = load(a.runs)
k1 = d[(d.k == 1) & d.match & d.condition.isin(KEEP)]
t = k1.pivot_table(index="condition", columns="size", values="OI_colour").reindex(KEEP)
g = k1.groupby("condition")
t["all"], t["sd"] = g.OI_colour.mean(), g.OI_colour.std()
t["runs_above_zero"] = g.OI_colour.apply(lambda x: int((x > 0).sum()))
t["n_runs"] = g.size()
t["OI_texture"] = g.OI_texture.mean()
for c in ("shape", "colour", "texture"):
    t[c] = g[c].mean()
t.round(4).to_csv(f"{a.out}_by_condition.csv")

h = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(a.hbm)) if f.endswith(".csv")])
h = h[(h.k == 1) & h.match.astype(str).isin(["True", "true", "1"])]
hb = h.pivot_table(index=["condition", "seed"], columns="probe_dim", values="delta", aggfunc="mean")
hb["OI_colour"], hb["OI_texture"] = hb["shape"] - hb["colour"], hb["shape"] - hb["texture"]
hb.groupby("condition").agg(["mean", "min", "max"]).round(3).to_csv(f"{a.out}_hbm.csv")

man = pd.DataFrame(json.load(open(a.manifest)))
B = man.groupby(["corpus", "seed"]).B.first()
gg = k1[k1.condition.isin(ORDERS)].copy()
gg["logB"] = [np.log(B[(CORPUS_OF[c], int(s))]) for c, s in zip(gg.condition, gg.seed)]
out = {}
for lab, sub in (("all_six", gg), ("without_grouped", gg[gg.condition != "regular_bursty"])):
    m = smf.mixedlm("OI_colour ~ logB + C(size)", sub, groups=sub["seed"]).fit(reml=True)
    out[lab] = {"slope": m.params["logB"], "se": m.bse["logB"],
                "p_one_sided": float(stats.norm.sf(m.params["logB"] / m.bse["logB"])),
                "spearman_runs": stats.spearmanr(sub.logB, sub.OI_colour)[0], "n": len(sub)}
json.dump(out, open(f"{a.out}_doseresponse.json", "w"), indent=2, default=float)
print(t[["tiny", "small", "medium", "all", "sd", "runs_above_zero"]].round(2).to_string())
print(json.dumps(out, indent=2, default=float))
