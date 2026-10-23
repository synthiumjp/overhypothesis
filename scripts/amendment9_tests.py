#!/usr/bin/env python3
"""
amendment9_tests.py  |  Registered tests of Amendment 9 (docs/AMENDMENT9.md section 4-5) and
the pilot report that fixes the number of nouns per cell (addendum item 2 and 13).

Inputs: scores CSV(s) from familiarity_eval.py; heads_summary.csv and per_noun_*.csv from
circuit_analysis.py.

  python scripts/amendment9_tests.py tests --scores results/amendment9/scores.csv \
      --circuits results/amendment9/circuits --out results/amendment9/RESULTS.md
  python scripts/amendment9_tests.py pilot --scores results/amendment9/scores_pilot.csv \
      --out results/amendment9/PILOT.md

Unit of replication: the seed (ten confirmatory seeds). Within a seed, each model gives one
value per cell (mean over the cell's nouns); sizes are averaged within seed. Exposure enters as
log10(exposures) over 10, 100, 1,000 and 10,000 (1 to 4); slopes are least-squares slopes per
model. One-sided t tests over seeds unless stated; Holm across H30 to H35 (H33 needs both of its
parts, so its p value is the larger of the two; H35 enters only if H33(a) is supported). A
supported hypothesis must also hold in direction at two of the three sizes.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_familiarity_corpora import CONF_SEEDS  # noqa: E402

LOGE = {10: 1, 100: 2, 1000: 3, 10000: 4}
SESOI = {"level": 1.0, "slope": 0.3, "attention": 0.02, "m1": 0.2}
ALPHA = 0.05
SIZES = ["tiny", "small", "medium"]


# ---------------------------------------------------------------- helpers
def t1(x, mu=0.0, side="greater"):
    x = np.asarray(x, float)
    r = stats.ttest_1samp(x, mu, alternative=side)
    m, n = x.mean(), len(x)
    h = stats.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return {"mean": m, "ci": [m - h, m + h], "t": float(r.statistic), "df": n - 1, "p": float(r.pvalue), "n": n}


def tost(x, bound):
    x = np.asarray(x, float)
    lo = stats.ttest_1samp(x, -bound, alternative="greater").pvalue
    hi = stats.ttest_1samp(x, bound, alternative="less").pvalue
    return float(max(lo, hi))


def slope(df, xcol, ycol):
    return np.polyfit(df[xcol], df[ycol], 1)[0] if df[xcol].nunique() > 1 else np.nan


def cell_means(s, measure):
    """Mean over the nouns of each cell, per model."""
    s = s.copy()
    s = s[s[measure].notna()]
    return (s.groupby(["base", "size", "seed", "type", "exposures"])[measure].mean().reset_index())


def per_model_slope(cm, typ, measure):
    d = cm[(cm.type == typ) & (cm.exposures > 0)].copy()
    d["x"] = d.exposures.map(LOGE)
    return d.groupby(["base", "size", "seed"]).apply(lambda g: slope(g, "x", measure)).rename(typ)


def seed_level(per_model):
    """per_model: Series indexed by (base, size, seed) or (size, seed). Returns size-averaged
    values per seed and the size-level means."""
    df = per_model.reset_index()
    val = df.columns[-1]
    seeds = df.groupby("seed")[val].mean()
    sizes = df.groupby("size")[val].mean()
    return seeds, sizes


def direction_ok(sizes, side):
    if side == "greater":
        return int((sizes > 0).sum())
    if side == "less":
        return int((sizes < 0).sum())
    return None


# ---------------------------------------------------------------- tests
def run_tests(scores, circ_dir):
    s = scores[scores.seed.isin(CONF_SEEDS)].copy()
    orig = s[s.emb == "original"]
    res = {}
    # M1
    cm_know = cell_means(orig, "know")
    m1 = {}
    for base in sorted(orig.base.unique()):
        d = cm_know[(cm_know.base == base) & (cm_know.type == "consistent")]
        at1000 = d[d.exposures == 1000].set_index(["size", "seed"])["know"]
        lvl, _ = seed_level(at1000)
        sl, _ = seed_level(per_model_slope(cm_know[cm_know.base == base], "consistent", "know"))
        tl, ts = t1(lvl, 0.0), t1(sl, 0.0)
        m1[base] = {"know_at_1000": tl, "slope": ts, "pass": bool(tl["mean"] >= SESOI["m1"] and ts["p"] < ALPHA)}
    res["M1"] = {"by_base": m1, "pass": all(v["pass"] for v in m1.values())}

    g = orig[orig.base == "regular_grouped"]
    cm = cell_means(g, "carry")
    # H30
    sc = per_model_slope(cm, "consistent", "carry")
    sa = per_model_slope(cm, "absent", "carry")
    diff = (sc - sa).rename("d")
    seeds, sizes = seed_level(diff.droplevel("base"))
    res["H30"] = {**t1(seeds, 0, "less"), "sizes_in_direction": direction_ok(sizes, "less"),
                  "by_size": sizes.to_dict(), "sesoi": SESOI["slope"]}
    # H31
    ab = cm[(cm.type == "absent") & cm.exposures.isin([1000, 10000])].groupby(["size", "seed"])["carry"].mean()
    ty = cell_means(s[(s.base == "regular_grouped") & (s.emb == "typical")], "carry")
    ty = ty[ty.type == "unseen"].groupby(["size", "seed"])["carry"].mean()
    seeds, sizes = seed_level((ab - ty).rename("d"))
    r = t1(seeds, 0, "two-sided")
    res["H31"] = {**r, "tost_p": tost(seeds, SESOI["level"]), "equivalence_bound": SESOI["level"],
                  "sizes_positive": int((sizes > 0).sum()), "by_size": sizes.to_dict()}
    # H32
    cs = cell_means(g, "stored")
    seeds, sizes = seed_level(per_model_slope(cs, "consistent", "stored").droplevel("base"))
    res["H32"] = {**t1(seeds, 0, "greater"), "sizes_in_direction": direction_ok(sizes, "greater"),
                  "by_size": sizes.to_dict(), "sesoi": SESOI["slope"]}
    # R1 (outside the family)
    u = cell_means(orig[orig.type == "unseen"], "carry")
    ug = u[u.base == "regular_grouped"].set_index(["size", "seed"])["carry"]
    ur = u[u.base == "regular_random"].set_index(["size", "seed"])["carry"]
    seeds, sizes = seed_level((ug - ur).rename("d"))
    res["R1"] = {**t1(seeds, 0, "greater"), "sizes_in_direction": direction_ok(sizes, "greater")}

    # heads
    hs_path = Path(circ_dir) / "heads_summary.csv"
    if hs_path.exists():
        hs = pd.read_csv(hs_path)
        hs = hs[hs.seed.isin(CONF_SEEDS)]
        hg = hs[hs.base == "regular_grouped"].set_index(["size", "seed"])
        seeds_a, sizes_a = seed_level(hg["unseen_reduction"])
        a = t1(seeds_a, 0.5, "greater")
        seeds_b, sizes_b = seed_level((hg["unseen_carry_random_mean"] - hg["unseen_carry_ablated"]).rename("d"))
        b = t1(seeds_b, 0, "greater")
        res["H33"] = {"a": {**a, "sizes_above_half": int((sizes_a > 0.5).sum())},
                      "b": {**b, "sizes_in_direction": direction_ok(sizes_b, "greater")},
                      "p": max(a["p"], b["p"])}
        hr = hs[hs.base == "reassigned_grouped"].set_index(["size", "seed"])
        seeds, sizes = seed_level((hr["trained_carry"] - hr["trained_carry_ablated"]).rename("d"))
        res["H34"] = {**t1(seeds, 0, "greater"), "sizes_in_direction": direction_ok(sizes, "greater")}
        pn = pd.concat([pd.read_csv(f) for f in glob.glob(str(Path(circ_dir) / "per_noun_*.csv"))])
        pn = pn[pn.seed.isin(CONF_SEEDS) & (pn.base == "regular_grouped")]
        cma = pn.groupby(["base", "size", "seed", "type", "exposures"])["attention"].mean().reset_index()
        d = (per_model_slope(cma, "consistent", "attention") - per_model_slope(cma, "absent", "attention"))
        seeds, sizes = seed_level(d.rename("d").droplevel("base"))
        res["H35"] = {**t1(seeds, 0, "less"), "sizes_in_direction": direction_ok(sizes, "less"),
                      "sesoi": SESOI["attention"]}
    # Holm
    fam = [("H30", res["H30"]["p"]), ("H31", res["H31"]["p"]), ("H32", res["H32"]["p"])]
    if "H33" in res:
        fam += [("H33", res["H33"]["p"]), ("H34", res["H34"]["p"])]
        if res["H33"]["a"]["p"] < ALPHA:
            fam.append(("H35", res["H35"]["p"]))
    fam.sort(key=lambda x: x[1])
    m, stop = len(fam), False
    for i, (h, p) in enumerate(fam):
        thr = ALPHA / (m - i)
        ok = (not stop) and p < thr
        stop = stop or not ok
        res[h]["holm_threshold"] = thr
        res[h]["holm_reject"] = bool(ok)
    return res


def mixed_models(scores):
    """Mixed models reported alongside (random intercepts for seed and noun)."""
    import statsmodels.formula.api as smf
    out = {}
    g = scores[(scores.seed.isin(CONF_SEEDS)) & (scores.emb == "original") & (scores.base == "regular_grouped")
               & scores.type.isin(["consistent", "absent"]) & (scores.exposures > 0)].copy()
    if len(g) < 20:
        return out
    g["x"] = g.exposures.map(LOGE)
    g["cons"] = (g.type == "consistent").astype(int)
    for y, form in (("carry", "carry ~ x * cons + C(size)"), ("stored", "stored ~ x * cons + C(size)")):
        try:
            md = smf.mixedlm(form, g, groups=g["seed"], vc_formula={"noun": "0 + C(noun)"}).fit()
            out[y] = {k: {"coef": float(md.params[k]), "p": float(md.pvalues[k])} for k in md.params.index
                      if k in ("x", "cons", "x:cons")}
        except Exception as e:  # report, do not fail
            out[y] = {"error": str(e)}
    return out


def fmt(r):
    return f"{r['mean']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}], t({r['df']}) = {r['t']:.2f}, p = {r['p']:.3g}"


def cmd_tests(a):
    scores = pd.concat([pd.read_csv(f) for f in a.scores])
    res = run_tests(scores, a.circuits)
    res["mixed_models"] = mixed_models(scores)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1, default=float)
    L = ["# Amendment 9 results", "", f"Seeds: {sorted(set(scores.seed) & set(CONF_SEEDS))}", "",
         f"M1 (manipulation check): {'pass' if res['M1']['pass'] else 'FAIL'}"]
    for b, v in res["M1"]["by_base"].items():
        L.append(f"- {b}: knowledge at 1,000 {fmt(v['know_at_1000'])}; slope {fmt(v['slope'])}")
    L.append("")
    for h in ("H30", "H31", "H32", "H33", "H34", "H35", "R1"):
        if h not in res:
            continue
        r = res[h]
        if h == "H33":
            L.append(f"{h}(a) reduction > 0.5: {fmt(r['a'])}; (b) random minus candidate: {fmt(r['b'])}")
        else:
            L.append(f"{h}: {fmt(r)}" + (f"; TOST p = {r['tost_p']:.3g}" if h == "H31" else ""))
        if "holm_reject" in r:
            L.append(f"  Holm threshold {r['holm_threshold']:.4f}: {'supported' if r['holm_reject'] else 'not supported'}")
        for k in ("sizes_in_direction", "sizes_positive", "sizes_above_half"):
            if k in r:
                L.append(f"  {k}: {r[k]} of 3")
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- pilot report
def power_one_sided(effect, sd, n, alpha):
    tc = stats.t.ppf(1 - alpha, n - 1)
    return float(1 - stats.nct.cdf(tc, n - 1, effect / (sd / np.sqrt(n))))


def cmd_pilot(a):
    """Addendum item 13: noun-level standard deviations from the pilot, combined with the
    Amendment 8 seed-level standard deviation (0.81 nats), give the expected seed-level SD of the
    H31 contrast for two and three nouns per cell; the smallest number with power >= 0.8 for a
    1-nat effect at alpha 0.05/6 (two-sided, ten seeds) is used."""
    s = pd.concat([pd.read_csv(f) for f in a.scores])
    o = s[s.emb == "original"]
    L = ["# Amendment 9 pilot", ""]
    sd_rows = []
    for base in sorted(o.base.unique()):
        for q in ("carry", "stored", "know"):
            d = o[(o.base == base) & o[q].notna() & (o.exposures > 0)]
            resid = d[q] - d.groupby(["type", "exposures"])[q].transform("mean")
            sd = float(np.sqrt((resid ** 2).sum() / max(1, len(d) - d.groupby(["type", "exposures"]).ngroups)))
            sd_rows.append((base, q, sd))
            L.append(f"- {base}: noun-level SD of {q} within cells = {sd:.3f} nats")
    sd_n = max(r[2] for r in sd_rows if r[0] == "regular_grouped" and r[1] == "carry")
    sd_u = sd_n
    L += ["", "Manipulation check (consistent nouns, knowledge by exposure, regular_grouped):"]
    k = o[(o.base == "regular_grouped") & (o.type == "consistent")].groupby("exposures")["know"].mean()
    L.append("  " + ", ".join(f"{e}: {v:.2f}" for e, v in k.items()))
    L += ["", "Power for the H31 contrast (1-nat effect, two-sided alpha 0.05/6, ten seeds); seed SD 0.81 "
              "from Amendment 8 combined with the pilot's noun SD:"]
    choice = 3
    for m in (2, 3):
        sd = np.sqrt(0.81 ** 2 + sd_n ** 2 / (2 * m) + sd_u ** 2 / 4)
        pw = power_one_sided(1.0, sd, 10, ALPHA / 6 / 2)
        L.append(f"- {m} nouns per cell: expected SD {sd:.3f}, power {pw:.2f}")
        if pw >= 0.8 and m < choice:
            choice = m
    L += ["", f"Nouns per cell: {choice}"]
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tests")
    t.add_argument("--scores", nargs="+", required=True)
    t.add_argument("--circuits", required=True)
    t.add_argument("--out", required=True)
    p = sub.add_parser("pilot")
    p.add_argument("--scores", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"tests": cmd_tests, "pilot": cmd_pilot}[a.cmd](a)
