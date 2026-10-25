#!/usr/bin/env python3
"""
a11_tests.py  |  Registered tests of Amendment 11 (docs/AMENDMENT11.md).

  python scripts/a11_tests.py --scores results/amendment11/scores.csv --rows results/amendment11/reanalysis.csv \
      --crossed results/amendment11/crossed.csv --out results/amendment11/RESULTS.md

Inputs are produced by scripts/familiarity_eval.py, scripts/a10_reanalysis.py and scripts/a10_crossed.py on
the Amendment 11 final models (manifest from scripts/amendment9_manifest.py --design a11). Writes RESULTS.md and
RESULTS.json next to --out.

Conventions (as Amendments 9 and 10): final models only; cell value per model = mean over the cell's nouns;
sizes averaged within seed; t tests over seeds; t-based 95% intervals. Part A: Holm over H50, H51, H52 (difference),
H53, H55 and H56; a hypothesis is supported when its Holm threshold is met and its direction holds at two of the
three sizes. Equivalence (H52, H54): two one-sided tests against +-0.3 nats at alpha .05, not Holm-adjusted.
Part B (large models, ten seeds): one-sided t at alpha .05. Part C: one-sided t at alpha .05 and direction at two of
three sizes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

SIZES = ("tiny", "small", "medium")
SEEDS_A = [2029, 3037, 4049, 5051, 6067, 7079, 8081, 9091, 10103, 11117]
SEEDS_C = SEEDS_A[:5]
BOUND = 0.3


def final(df):
    return df[df.ckpt.str.endswith("model_final.pt")]


def per_size_seed(df, typ, e=None, q="carry", base="regular_grouped", sizes=SIZES):
    d = df[(df.base == base) & (df.type == typ) & df["size"].isin(sizes)]
    if "emb" in d:
        d = d[d.emb == "original"]
    if "condition" in d:
        d = d[d.condition == "own"]
    if e is not None:
        d = d[d.exposures.isin([e] if np.isscalar(e) else e)]
    # cell value per model: mean over nouns (and over exposure levels when pooled, each level weighted equally)
    g = d.groupby(["size", "seed", "exposures"])[q].mean().groupby(["size", "seed"]).mean()
    return g


def seed_level(g):
    return g.groupby("seed").mean()


def ttest(x, alternative):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(.975, len(x) - 1) * se
    r = stats.ttest_1samp(x, 0, alternative=alternative)
    return {"mean": m, "ci": [m - h, m + h], "t": float(r.statistic), "df": len(x) - 1, "p": float(r.pvalue),
            "n": len(x)}


def tost(x, b=BOUND):
    x = np.asarray(x, float)
    n = len(x)
    se = x.std(ddof=1) / np.sqrt(n)
    m = x.mean()
    return float(max(1 - stats.t.cdf((m + b) / se, n - 1), stats.t.cdf((m - b) / se, n - 1)))


def sizes_in_direction(g, sign):
    by = g.groupby("size").mean()
    return int(sum((by > 0) if sign > 0 else (by < 0)))


def contrast(df, a, b, e, q="carry", base_a="regular_grouped", base_b="regular_grouped", sizes=SIZES):
    ga = per_size_seed(df, a, e, q, base_a, sizes)
    gb = per_size_seed(df, b, e, q, base_b, sizes)
    g = (ga - gb).dropna()
    return g, seed_level(g)


def geometry_test(rows, e=1000, sizes=SIZES):
    o = rows[(rows.base == "regular_grouped") & (rows.condition == "own") & (rows.type != "unseen")
             & (rows.exposures == e) & rows["size"].isin(sizes)].copy()
    o["rot_run"] = (o.type == "rotating_run").astype(int)
    m = smf.ols("carry ~ cos_init + dist_mean_rel + row_norm + C(size) + C(seed) + rot_run", data=o).fit(
        cov_type="cluster", cov_kwds={"groups": o.seed})
    coef, se = float(m.params["rot_run"]), float(m.bse["rot_run"])
    p = float(1 - stats.norm.cdf(coef / se))
    per = {}
    for z in sizes:
        oz = o[o["size"] == z]
        mz = smf.ols("carry ~ cos_init + dist_mean_rel + row_norm + C(seed) + rot_run", data=oz).fit()
        per[z] = float(mz.params["rot_run"])
    return {"mean": coef, "ci": [coef - 1.96 * se, coef + 1.96 * se], "z": coef / se, "p": p,
            "sizes_in_direction": int(sum(v > 0 for v in per.values())), "by_size": per}


def crossed_test(cr, e=1000, sizes=SIZES):
    x = cr[(cr.base == "regular_grouped") & (cr.exposures == e) & (~cr.same_noun.astype(bool)) & cr["size"].isin(sizes)]
    fam = [t for t in x.ctx_type.unique() if t != "unseen"]
    x = x[x.ctx_type.isin(fam) & x.query_type.isin(fam)]
    out_by_size = {}
    vals = []
    for z in sizes:
        xz = x[x["size"] == z]
        q = xz.groupby(["seed", "query_type"]).carry.mean().unstack()
        c = xz.groupby(["seed", "ctx_type"]).carry.mean().unstack()
        d = (q.rotating_run - q.fixed_run) - (c.rotating_run - c.fixed_run)
        out_by_size[z] = float(d.mean())
        vals.append(d.rename(z))
    s = pd.concat(vals, axis=1).mean(axis=1)
    r = ttest(s, "greater")
    r["sizes_in_direction"] = int(sum(v > 0 for v in out_by_size.values()))
    return r


def holm(ps):
    order = sorted(ps.items(), key=lambda kv: kv[1])
    k = len(order)
    thr, reject, stop = {}, {}, False
    for i, (h, p) in enumerate(order):
        thr[h] = 0.05 / (k - i)
        if stop or p > thr[h]:
            stop = True
            reject[h] = False
        else:
            reject[h] = True
    return thr, reject


def fmt(r, label):
    s = f"{label}: {r['mean']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}], p = {r['p']:.3g}"
    for k in ("sizes_in_direction", "holm_threshold", "holm_reject", "tost_p", "outcome", "supported"):
        if k in r:
            v = r[k]
            s += f"; {k} = {v:.3g}" if isinstance(v, float) else f"; {k} = {v}"
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--crossed", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = final(pd.read_csv(a.scores))
    s["win"] = -s["compete"]
    rows = final(pd.read_csv(a.rows))
    cr = final(pd.read_csv(a.crossed))
    sA = s[s.seed.isin(SEEDS_A)]
    R = {}
    # Part A
    g = per_size_seed(sA, "unseen", None)
    R["M3"] = ttest(seed_level(g) - 1, "greater")
    R["M3"]["pass"] = R["M3"]["p"] < .05
    g, x = contrast(sA, "rotating_run", "fixed_run", 1000)
    R["H50"] = {**ttest(x, "greater"), "sizes_in_direction": sizes_in_direction(g, +1)}
    g, x = contrast(sA, "rotating_run", "rotating_single", [100, 1000])
    R["H51"] = {**ttest(x, "greater"), "sizes_in_direction": sizes_in_direction(g, +1)}
    g, x = contrast(sA, "fixed_single", "rotating_single", [100, 1000])
    r = ttest(x, "two-sided")
    R["H52"] = {**r, "tost_p": tost(x), "sizes_in_direction": max(sizes_in_direction(g, +1), sizes_in_direction(g, -1))}
    g, x = contrast(sA, "fixed_single", "rotating_single", [100, 1000], q="win")
    R["H53"] = {**ttest(x, "less"), "sizes_in_direction": sizes_in_direction(g, -1)}
    g = per_size_seed(sA, "rotating_run", 1000, q="noun_specific")
    x = seed_level(g)
    R["H54"] = {**ttest(x, "two-sided"), "tost_p": tost(x)}
    R["H54"]["outcome"] = "equivalent" if R["H54"]["tost_p"] < .05 else "not shown equivalent"
    R["H55"] = geometry_test(rows[rows.seed.isin(SEEDS_A)])
    R["H56"] = crossed_test(cr[cr.seed.isin(SEEDS_A)])
    thr, rej = holm({h: R[h]["p"] for h in ("H50", "H51", "H52", "H53", "H55", "H56")})
    for h in thr:
        R[h]["holm_threshold"], R[h]["holm_reject"] = thr[h], rej[h]
        R[h]["supported"] = bool(rej[h] and R[h]["sizes_in_direction"] >= 2)
    R["H52"]["outcome"] = ("equivalent" if R["H52"]["tost_p"] < .05 else
                           "different" if R["H52"]["supported"] else "inconclusive")
    R["H52"].pop("supported")
    # Part B: large models
    sB = s[(s["size"] == "large") & s.seed.isin(SEEDS_A)]
    if len(sB):
        ga = per_size_seed(sB, "rotating_run", 1000, sizes=("large",))
        gb = per_size_seed(sB, "fixed_run", 1000, sizes=("large",))
        x = seed_level(ga - gb)
        R["H57"] = ttest(x, "greater")
        R["H57"]["supported"] = R["H57"]["p"] < .05
        R["H57"]["n_models"] = int(len(x))
    # Part C: pure corpora
    sC = s[s.seed.isin(SEEDS_C)]
    if (sC.base == "regular_pure_changing").any() and (sC.base == "regular_pure_fixed").any():
        g, x = contrast(sC, "rotating_run", "fixed_run", 1000, base_a="regular_pure_changing",
                        base_b="regular_pure_fixed")
        R["H58"] = {**ttest(x, "greater"), "sizes_in_direction": sizes_in_direction(g, +1)}
        R["H58"]["supported"] = bool(R["H58"]["p"] < .05 and R["H58"]["sizes_in_direction"] >= 2)
        ex = {}
        for typ, base in (("rotating_run", "regular_pure_changing"), ("fixed_run", "regular_pure_fixed")):
            g, x = contrast(sC, typ, typ, 1000, base_a=base, base_b="regular_grouped")
            ex[f"{typ}_pure_minus_mixed"] = ttest(x, "two-sided")
        R["exploratory_interference"] = ex
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")
    lines = ["# Amendment 11 results", ""]
    for h in ("M3", "H50", "H51", "H52", "H53", "H54", "H55", "H56", "H57", "H58"):
        if h in R:
            lines.append("- " + fmt(R[h], h))
    if "exploratory_interference" in R:
        lines += ["", "Exploratory (interference, pure minus mixed corpora, same seeds, 1,000 exposures):"]
        for k, r in R["exploratory_interference"].items():
            lines.append("- " + fmt(r, k))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
