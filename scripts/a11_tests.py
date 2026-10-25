#!/usr/bin/env python3
"""
a11_tests.py  |  Registered tests of Amendment 11 (docs/AMENDMENT11.md).

  python scripts/a11_tests.py --scores results/amendment11/scores.csv --rows results/amendment11/reanalysis.csv \
      --crossed results/amendment11/crossed.csv --out results/amendment11/RESULTS.md

Inputs are produced by scripts/familiarity_eval.py, scripts/a10_reanalysis.py and scripts/a10_crossed.py on
the Amendment 11 final models (manifest from scripts/amendment9_manifest.py --design a11). Writes RESULTS.md and
RESULTS.json next to --out.

Conventions (as Amendments 9 and 10): final models only; cell value per model = mean over the cell's nouns;
sizes averaged within seed; t tests over seeds; t-based 95% intervals. Models marked fail by
scripts/a11_model_check.py (--model_check) and models with any non-finite score are dropped; a seed enters a part
only when all its models for that part are present (Part A at least 8 seeds, Part B at least 5, Part C at least 4).
Part A: Holm over H50, H51, H52 (difference), H53, H55 and H56; a hypothesis is supported when its Holm threshold
is met and its direction holds at two of the three sizes. Equivalence (H52, H54, H59): two one-sided tests against
+-0.3 nats at alpha .05, not Holm-adjusted. Part B: one-sided t at alpha .05. Part C: H58 one-sided t at alpha .05
with direction at two of three sizes; H59 two-sided t and equivalence at alpha .05 each. Each part is interpreted
only if its manipulation check (M3, M3B, M3C) passes.
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
    if n < 2 or not se > 0:
        return float("nan")  # undefined; read as not equivalent
    return float(max(1 - stats.t.cdf((m + b) / se, n - 1), stats.t.cdf((m - b) / se, n - 1)))


def equiv(x):
    """TOST p and the matching 90% interval (equivalence holds when the 90% interval lies inside +-BOUND)."""
    x = np.asarray(x, float)
    n, m = len(x), x.mean()
    h = stats.t.ppf(.95, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return {"tost_p": tost(x), "ci90": [m - h, m + h]}


def sizes_in_direction(g, sign):
    """Sizes whose mean (over seeds) has the sign `sign`; for two-sided tests pass the sign of the pooled mean."""
    by = g.groupby("size").mean()
    return int(((by > 0) if sign > 0 else (by < 0)).sum())


def contrast(df, a, b, e, q="carry", base_a="regular_grouped", base_b="regular_grouped", sizes=SIZES):
    ga = per_size_seed(df, a, e, q, base_a, sizes)
    gb = per_size_seed(df, b, e, q, base_b, sizes)
    g = (ga - gb).dropna()
    return g, seed_level(g)


def _per_model_coef(o):
    out = []
    for (z, sd), om in o.groupby(["size", "seed"]):
        mm = smf.ols("carry ~ cos_init + dist_mean_rel + row_norm + rot_run", data=om).fit()
        out.append((z, sd, float(mm.params["rot_run"])))
    return pd.DataFrame(out, columns=["size", "seed", "b"]).set_index(["size", "seed"]).b


def geometry_test(rows, e=1000, sizes=SIZES):
    """H55. Registered test: per model, OLS of the copy score of the 18 familiar nouns at 1,000 exposures on cosine
    with the initial row, relative distance from the trained kinds' mean row, row norm and a changing-burst
    indicator; the indicator's coefficient is averaged over sizes within seed; one-sided t over seeds. The pooled
    noun-level regression with seed-clustered errors (t reference, G - 1 df) and a sensitivity analysis without
    no-shape nouns are reported alongside."""
    o = rows[(rows.base == "regular_grouped") & (rows.condition == "own") & (rows.type != "unseen")
             & (rows.exposures == e) & rows["size"].isin(sizes)].copy()
    o["rot_run"] = (o.type == "rotating_run").astype(int)
    pm = _per_model_coef(o)
    r = ttest(pm.groupby("seed").mean(), "greater")
    by = pm.groupby("size").mean()
    r["by_size"] = {z: float(v) for z, v in by.items()}
    r["sizes_in_direction"] = int((by > 0).sum())
    r["n_models"] = int(len(pm))
    r["n_nouns_per_model"] = int(o.groupby(["size", "seed"]).size().min())
    m = smf.ols("carry ~ cos_init + dist_mean_rel + row_norm + C(size) + C(seed) + rot_run", data=o).fit(
        cov_type="cluster", cov_kwds={"groups": o.seed})
    coef, se, G = float(m.params["rot_run"]), float(m.bse["rot_run"]), o.seed.nunique()
    h = stats.t.ppf(.975, G - 1) * se
    r["clustered"] = {"mean": coef, "ci": [coef - h, coef + h], "t": coef / se, "df": G - 1,
                      "p": float(stats.t.sf(coef / se, G - 1))}
    o2 = o[~o.type.str.startswith("absent")]
    r["without_no_shape"] = ttest(_per_model_coef(o2).groupby("seed").mean(), "greater")
    return r


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
    order = sorted(ps.items(), key=lambda kv: (not np.isfinite(kv[1]), kv[1]))
    k = len(order)
    thr, reject, stop = {}, {}, False
    for i, (h, p) in enumerate(order):
        thr[h] = 0.05 / (k - i)
        if stop or not np.isfinite(p) or not p < thr[h]:
            stop = True
            reject[h] = False
        else:
            reject[h] = True
    return thr, reject


def fmt(r, label):
    s = f"{label}: {r['mean']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}], p = {r['p']:.3g}"
    if "ci90" in r:
        s += f", 90% [{r['ci90'][0]:.3f}, {r['ci90'][1]:.3f}]"
    for k in ("pass", "n", "n_models", "sizes_in_direction", "holm_threshold", "holm_reject", "tost_p", "outcome",
              "supported", "note"):
        if k in r:
            v = r[k]
            s += f"; {k} = {v:.3g}" if isinstance(v, float) else f"; {k} = {v}"
    return s


def complete_seeds(s, base, sizes, seeds):
    have = set(zip(s.base, s["size"], s.seed))
    return [sd for sd in seeds if all((base, z, sd) in have for z in sizes)]


def two_sided_outcome(r, below="equivalent"):
    """Reading of a two-sided difference plus equivalence test, as registered (both results are reported)."""
    eq, diff = r["tost_p"] < .05, r["p"] < .05
    if eq and diff:
        return "different from zero but within +-0.3"
    if eq:
        return below
    if diff:
        return "different (above zero)" if r["mean"] > 0 else "different (below zero)"
    return "inconclusive"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--crossed", required=True)
    ap.add_argument("--model_check", nargs="+", required=True,
                    help="model_check.csv files from scripts/a11_model_check.py; models marked fail are dropped")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = final(pd.read_csv(a.scores))
    rows = final(pd.read_csv(a.rows))
    cr = final(pd.read_csv(a.crossed))
    R = {}
    # ---- completeness and the registered failure rule (report, never impute)
    excluded = set()
    for f in a.model_check:
        mc = pd.read_csv(f)
        excluded |= set(mc.ckpt[mc.status == "fail"])
    # non-finite scores: registered measures only (compete and noun_specific are undefined for trained kinds)
    sc = s[(s.emb == "original") & (s.type != "trained")] if "emb" in s else s[s.type != "trained"]
    nonfinite = set(sc.ckpt[~np.isfinite(sc[["carry", "compete", "noun_specific"]]).all(axis=1)])
    ro = rows[rows.condition == "own"]
    nonfinite |= set(rows.ckpt[~np.isfinite(rows.carry)])
    nonfinite |= set(ro.ckpt[~np.isfinite(ro[["row_norm", "dist_mean_rel"]]).all(axis=1)])
    nonfinite |= set(cr.ckpt[~np.isfinite(cr.carry)])
    m = s.drop_duplicates("ckpt")
    dup = m[m.duplicated(["base", "size", "seed"], keep=False)]
    if len(dup):
        raise SystemExit("more than one model for the same design, size and seed:\n" + "\n".join(dup.ckpt))
    R["models"] = {f"{b}/{z}": int(((m.base == b) & (m["size"] == z)).sum())
                   for b in ("regular_grouped", "regular_pure_changing", "regular_pure_fixed")
                   for z in SIZES + ("large",)}
    R["models"]["model_check_files"] = len(a.model_check)
    R["models"]["excluded_failed_loss"] = sorted(excluded)
    R["models"]["excluded_nonfinite_score"] = sorted(nonfinite)
    drop = excluded | nonfinite
    s, rows, cr = s[~s.ckpt.isin(drop)], rows[~rows.ckpt.isin(drop)], cr[~cr.ckpt.isin(drop)]
    # a model missing from any scoring output counts as missing (Part A uses all three outputs)
    fam_rows = rows[(rows.condition == "own") & (rows.type != "unseen") & (rows.exposures == 1000)]
    n_rows = fam_rows.groupby("ckpt").noun.nunique()
    crf = cr[(cr.exposures == 1000) & (~cr.same_noun.astype(bool)) & (cr.ctx_type != "unseen")
             & (cr.query_type != "unseen")]
    n_cr = crf.groupby("ckpt").apply(lambda d: len(set(zip(d.ctx_type, d.query_type))))
    full = set(n_rows.index[n_rows == 18]) & set(n_cr.index[n_cr == 36])
    sA_ok = s[(s.base != "regular_grouped") | s["size"].isin(("large",)) | s.ckpt.isin(full)]
    R["models"]["mixed_models_missing_rows_or_crossed"] = sorted(
        set(s.ckpt[(s.base == "regular_grouped") & s["size"].isin(SIZES)]) - full)
    seeds_a = complete_seeds(sA_ok, "regular_grouped", SIZES, SEEDS_A)
    R["models"]["partA_complete_seeds"] = seeds_a
    # ---- Part A
    if len(seeds_a) < 8:
        R["PartA"] = {"interpreted": False, "reason": f"{len(seeds_a)} complete seeds; at least 8 are required"}
    else:
        sA = s[s.seed.isin(seeds_a)]
        g = per_size_seed(sA, "unseen", None)
        R["M3"] = ttest(seed_level(g) - 1, "greater")
        R["M3"]["pass"] = bool(R["M3"]["p"] < .05)
        R["PartA"] = {"interpreted": R["M3"]["pass"], "n_seeds": len(seeds_a),
                      "reason": "" if R["M3"]["pass"] else "M3 failed"}
        g, x = contrast(sA, "rotating_run", "fixed_run", 1000)
        R["H50"] = {**ttest(x, "greater"), "sizes_in_direction": sizes_in_direction(g, +1)}
        g, x = contrast(sA, "rotating_run", "rotating_single", [100, 1000])
        R["H51"] = {**ttest(x, "greater"), "sizes_in_direction": sizes_in_direction(g, +1)}
        g, x = contrast(sA, "fixed_single", "rotating_single", [100, 1000])
        r = ttest(x, "two-sided")
        R["H52"] = {**r, **equiv(x), "sizes_in_direction": sizes_in_direction(g, 1 if r["mean"] > 0 else -1)}
        # H53: compete = L(s*) - L(s), L = log probability; above zero means the noun's own shape is preferred
        g = per_size_seed(sA, "fixed_single", [100, 1000], q="compete")
        R["H53"] = {**ttest(seed_level(g), "greater"), "sizes_in_direction": sizes_in_direction(g, +1)}
        g = per_size_seed(sA, "rotating_run", 1000, q="noun_specific")
        x = seed_level(g)
        R["H54"] = {**ttest(x, "two-sided"), **equiv(x)}
        R["H54"]["outcome"] = two_sided_outcome(R["H54"], "any noun-specific part is below 0.3 nats")
        ri = rows[rows.seed.isin(seeds_a) & (rows.base == "regular_grouped")]
        fam_ri = ri[(ri.condition == "own") & (ri.type != "unseen") & (ri.exposures == 1000)]
        bad = set(fam_ri.seed[pd.to_numeric(fam_ri.cos_init, errors="coerce").isna()])
        seeds_55 = [sd for sd in seeds_a if sd not in bad]
        if len(seeds_55) < 8:
            R["H55"] = {"mean": float("nan"), "ci": [float("nan")] * 2, "p": float("nan"), "sizes_in_direction": 0,
                        "note": f"not interpreted: initial rows reproduced for {len(seeds_55)} complete seeds"}
        else:
            ri = ri[ri.seed.isin(seeds_55)].copy()
            ri["cos_init"] = pd.to_numeric(ri.cos_init, errors="coerce")
            R["H55"] = geometry_test(ri)
            if bad:
                R["H55"]["note"] = f"seeds without reproduced initial rows left out: {sorted(bad)}"
        R["H56"] = crossed_test(cr[cr.seed.isin(seeds_a)])
        thr, rej = holm({h: R[h]["p"] for h in ("H50", "H51", "H52", "H53", "H55", "H56")})
        for h in thr:
            R[h]["holm_threshold"], R[h]["holm_reject"] = thr[h], rej[h]
            R[h]["supported"] = bool(rej[h] and R[h]["sizes_in_direction"] >= 2)
        eq, diff = R["H52"]["tost_p"] < .05, R["H52"]["supported"]
        R["H52"]["outcome"] = ("different but within +-0.3" if eq and diff else "equivalent" if eq else
                               ("different (fixed above)" if R["H52"]["mean"] > 0 else "different (fixed below)")
                               if diff else "inconclusive")
        R["H52"].pop("supported")
    # ---- Part B: large models (seeds trained in the listed order; every completed model is used; at least 5)
    seeds_b = complete_seeds(s, "regular_grouped", ("large",), SEEDS_A)
    R["models"]["partB_complete_seeds"] = seeds_b
    if 0 < len(seeds_b) < 5:
        R["PartB"] = {"interpreted": False, "reason": f"{len(seeds_b)} large models; at least 5 are required"}
    elif seeds_b:
        sB = s[(s["size"] == "large") & s.seed.isin(seeds_b)]
        R["M3B"] = ttest(seed_level(per_size_seed(sB, "unseen", None, sizes=("large",))) - 1, "greater")
        R["M3B"]["pass"] = bool(R["M3B"]["p"] < .05)
        ga = per_size_seed(sB, "rotating_run", 1000, sizes=("large",))
        gb = per_size_seed(sB, "fixed_run", 1000, sizes=("large",))
        R["H57"] = ttest(seed_level(ga - gb), "greater")
        R["H57"]["supported"] = bool(R["H57"]["p"] < .05)
        R["H57"]["n_models"] = len(seeds_b)
        R["PartB"] = {"interpreted": R["M3B"]["pass"], "reason": "" if R["M3B"]["pass"] else "M3B failed"}
    # ---- Part C: single-history corpora (at least 4 seeds with both pure designs at all three sizes)
    pc = set(complete_seeds(s, "regular_pure_changing", SIZES, SEEDS_C))
    seeds_c = [sd for sd in SEEDS_C if sd in pc and sd in complete_seeds(s, "regular_pure_fixed", SIZES, SEEDS_C)]
    R["models"]["partC_complete_seeds"] = seeds_c
    if (s.base == "regular_pure_changing").any() or (s.base == "regular_pure_fixed").any():
        if len(seeds_c) < 4:
            R["PartC"] = {"interpreted": False, "reason": f"{len(seeds_c)} complete seeds; at least 4 are required"}
        else:
            sC = s[s.seed.isin(seeds_c)]
            ok = True
            for b, k in (("regular_pure_changing", "M3C_changing"), ("regular_pure_fixed", "M3C_fixed")):
                R[k] = ttest(seed_level(per_size_seed(sC, "unseen", None, base=b)) - 1, "greater")
                R[k]["pass"] = bool(R[k]["p"] < .05)
                ok &= R[k]["pass"]
            R["PartC"] = {"interpreted": bool(ok), "n_seeds": len(seeds_c), "reason": "" if ok else "M3C failed"}
            g, x = contrast(sC, "rotating_run", "fixed_run", 1000, base_a="regular_pure_changing",
                            base_b="regular_pure_fixed")
            R["H58"] = {**ttest(x, "greater"), "sizes_in_direction": sizes_in_direction(g, +1)}
            R["H58"]["supported"] = bool(R["H58"]["p"] < .05 and R["H58"]["sizes_in_direction"] >= 2)
            # H59 (registered, secondary): pure minus mixed corpora, same history, same seed and size
            seeds_cm = [sd for sd in seeds_c if sd in seeds_a]
            sCM = s[s.seed.isin(seeds_cm)]
            R["models"]["H59_seeds"] = seeds_cm
            pairs = (("rotating_run", "regular_pure_changing", "H59_changing_bursts"),
                     ("fixed_run", "regular_pure_fixed", "H59_fixed_bursts"))
            for typ, base, k in pairs:
                if len(seeds_cm) < 4:
                    R[k] = {"mean": float("nan"), "ci": [float("nan")] * 2, "p": float("nan"),
                            "note": f"not run: {len(seeds_cm)} seeds with both pure and mixed models; at least 4"}
                    continue
                g, x = contrast(sCM, typ, typ, 1000, base_a=base, base_b="regular_grouped")
                R[k] = {**ttest(x, "two-sided"), **equiv(x)}
                R[k]["outcome"] = (two_sided_outcome(R[k]).replace("(above zero)", "(pure above mixed)")
                                   .replace("(below zero)", "(pure below mixed)"))
    # parts not run, and conclusions withheld when a part is not interpreted (values stay in RESULTS.json)
    for part, hs in (("PartA", ("H50", "H51", "H52", "H53", "H54", "H55", "H56")), ("PartB", ("H57",)),
                     ("PartC", ("H58", "H59_changing_bursts", "H59_fixed_bursts"))):
        if part not in R:
            R[part] = {"interpreted": False, "reason": "not run (no models)"}
        if not R[part]["interpreted"]:
            for h in hs:
                if h in R:
                    for k in ("supported", "outcome"):
                        if k in R[h]:
                            R[h][f"{k}_if_interpreted"] = R[h].pop(k)
                    R[h]["note"] = (R[h].get("note", "") + f" {part} not interpreted; no conclusion drawn").strip()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")
    lines = ["# Amendment 11 results", "", "Models: " + json.dumps(R["models"], default=str), ""]
    for part in ("PartA", "PartB", "PartC"):
        if part in R:
            lines.append(f"{part}: interpreted = {R[part]['interpreted']}" +
                         (f" ({R[part]['reason']})" if R[part]["reason"] else ""))
    lines.append("")
    for h in ("M3", "H50", "H51", "H52", "H53", "H54", "H55", "H56", "M3B", "H57", "M3C_changing", "M3C_fixed",
              "H58", "H59_changing_bursts", "H59_fixed_bursts"):
        if h in R:
            lines.append("- " + fmt(R[h], h))
            if h == "H55":
                lines.append("  - " + fmt(R[h]["clustered"], "H55 pooled noun-level, seed-clustered (reported)"))
                lines.append("  - " + fmt(R[h]["without_no_shape"], "H55 without no-shape nouns (sensitivity)"))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
