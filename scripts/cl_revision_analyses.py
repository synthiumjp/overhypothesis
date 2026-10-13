#!/usr/bin/env python3
"""
cl_revision_analyses.py | Analyses added for the Computational Linguistics short paper
in response to pre-submission review (13 Oct 2026). All exploratory or robustness
checks of registered tests; none replaces a registered test.

1. Seed-level tests for every Study 1 row of Table 2: average the three model sizes
   within each corpus seed, take the paired difference per seed (5 values), and report
   the mean difference, a t-based 95% CI (df = 4), a one-sided t p (two-sided where the
   registered test was two-sided or the result reversed), and the number of seeds in the
   predicted direction.
2. Noun-specific index (same noun minus different noun) per Study 1 condition, with a
   seed-level TOST against +/-0.2 nats.
3. Increments of the clustering curve per unit of log B.
4. Hierarchical bootstrap (sources, then documents) for B in each natural corpus and for
   the registered differences (speech to children vs written text, vs adult conversation).
5. Effective clustering of different objects, B_eff = 1 + (B - 1) d, with a bootstrap that
   resamples sources/documents for B and coded pairs for d.

Usage:
  python scripts/cl_revision_analyses.py --out results/cl_revision
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RUNS = ["results/rescore/study1bc_per_run.csv", "results/rescore/study1x_per_run.csv",
        "results/rescore/study1w_per_run.csv", "results/study1o/summary_per_run.csv"]
COUNTS = "results/study3x/counts.csv"
CODES = "results/study3x/codes_first.csv"
KEY = "results/study3x/coding_key.csv"
W, MIN_WORDS, SEED, NBOOT = 100, 400, 20261013, 4000

# Table 2 rows (Study 1): (row, a, b, index, sidedness). b = None tests a against zero.
ROWS = [
    (1, "regular_bursty", "regular", "OI_colour", "greater"),
    (2, "regular", None, "OI_colour", "greater"),
    (3, "regular", "scrambled", "OI_colour", "greater"),
    (4, "regular_bursty", "regular_nostable_grouped", "OI_colour", "greater"),
    (5, "regular_nostable_grouped", "regular_colourstable_grouped", "OI_colour", "greater"),
    (6, "regular_bursty", "scrambled_bursty", "OI_colour", "greater"),
    (7, "regular_bursty", "scrambled_bursty", "OI_texture", "greater"),
    (9, "regular_graded_b3p6", "regular", "OI_colour", "greater"),
    (10, "regular_o_diff_b3p6", "regular_o_random", "OI_colour", "greater"),
    (11, "regular_w512", "regular", "OI_colour", "greater"),
    (12, "regular_graded_b3p6_w512", "regular_graded_b3p6", "OI_colour", "two-sided"),
    (13, "regular_w512", "regular_nostable_random_w512", "OI_colour", "greater"),
    (14, "regular_o_diff_grouped", "regular_o_same_grouped", "OI_colour", "greater"),
    (15, "regular_o_diff_b3p6", "regular_o_same_b3p6", "OI_colour", "greater"),
    ("S", "regular_o_same_b3p6", "regular_o_random", "OI_colour", "two-sided"),
]
GRADED = [("regular", 0.912), ("regular_graded_b1p4", 1.369), ("regular_graded_b3p6", 3.61),
          ("regular_graded_b7", 6.989), ("regular_graded_b12", 12.144), ("regular_bursty", 23.082)]


def runs():
    d = pd.concat([pd.read_csv(f) for f in RUNS])
    return d[(d.k == 1) & (d.source == "lm")]


def seed_means(d, cond, col, match=True):
    g = d[(d.condition == cond) & (d.match == match)]
    assert len(g) == 15, (cond, len(g))
    return g.groupby("seed")[col].mean()


def ttest(x, side):
    x = np.asarray(x, float)
    n, m, se = len(x), x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    t = m / se
    p = stats.t.sf(t, n - 1) if side == "greater" else 2 * stats.t.sf(abs(t), n - 1)
    h = stats.t.ppf(0.975, n - 1) * se
    return {"diff": m, "lo": m - h, "hi": m + h, "t": t, "df": n - 1, "p": p,
            "seeds_in_direction": int((x > 0).sum()) if side == "greater" else int((x > 0).sum())}


def seed_rows(d):
    out = []
    for row, a, b, col, side in ROWS:
        xa = seed_means(d, a, col)
        x = xa - seed_means(d, b, col) if b else xa
        r = ttest(x.values, side)
        # 15 matched models (size x seed), paired
        ga = d[(d.condition == a) & d.match].set_index(["size", "seed"])[col]
        gb = d[(d.condition == b) & d.match].set_index(["size", "seed"])[col] if b else 0 * ga
        m = (ga - gb).dropna()
        r15 = ttest(m.values, side)
        out.append({"row": row, "a": a, "b": b, "index": col, "side": side,
                    **{k: r[k] for k in ("diff", "lo", "hi", "t", "df", "p", "seeds_in_direction")},
                    "p_15_matched": r15["p"], "models_in_direction": int((m > 0).sum())})
    return pd.DataFrame(out)


def trend_by_seed(d):
    B = dict(GRADED)
    g = d[d.condition.isin(B) & d.match].copy()
    g["logB"] = np.log(g.condition.map(B))
    slopes = [np.polyfit(gs.logB, gs.OI_colour, 1)[0] for _, gs in g.groupby("seed")]
    r = ttest(slopes, "greater")
    means = g.groupby("condition").OI_colour.mean().reindex([c for c, _ in GRADED])
    inc = []
    for (c0, b0), (c1, b1) in zip(GRADED[:-1], GRADED[1:]):
        inc.append({"from_B": b0, "to_B": b1,
                    "per_log_unit": (means[c1] - means[c0]) / (np.log(b1) - np.log(b0)),
                    "per_unit_B": (means[c1] - means[c0]) / (b1 - b0)})
    return {"slope_by_seed": r, "slopes": slopes, "means": means.round(3).to_dict(), "increments": inc}


def noun_specific(d):
    out = []
    for c, g in d.groupby("condition"):
        p = g.pivot_table(index=["size", "seed"], columns="match", values="OI_colour")
        ns = (p[True] - p[False]).groupby(level="seed").mean().values
        n, m, se = len(ns), ns.mean(), ns.std(ddof=1) / np.sqrt(len(ns))
        p_lower = stats.t.sf((m + 0.2) / se, n - 1)
        p_upper = stats.t.cdf((m - 0.2) / se, n - 1)
        h = stats.t.ppf(0.95, n - 1) * se
        out.append({"condition": c, "noun_specific": m, "ci90_lo": m - h, "ci90_hi": m + h,
                    "tost_p": max(p_lower, p_upper), "index_same_noun": g[g.match].OI_colour.mean()})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- corpora

def counts():
    c = pd.read_csv(COUNTS, keep_default_na=False)
    c = c[(c.n_mentions > 0) & (c.n_words >= MIN_WORDS) & (c.speakers == "all")]
    return {s: c[c.source == s] for s in ("childes", "cabank", "pile10k")}


def B_of(g):
    return (g[f"same_{W}"].sum() / g[f"pairs_{W}"].sum()) / (g[f"same0_{W}"].sum() / g[f"pairs0_{W}"].sum())


def hier_sample(g, rng):
    groups = [gc for _, gc in g.groupby("corpus")]
    parts = []
    for i in rng.integers(0, len(groups), len(groups)):
        gc = groups[i]
        parts.append(gc.iloc[rng.integers(0, len(gc), len(gc))])
    return pd.concat(parts)


def coded():
    k = pd.read_csv(KEY)
    c = pd.read_csv(CODES)
    return c.merge(k, on="pair_id")


def corpora(rng):
    C = counts()
    cod = coded()
    point = {s: B_of(g) for s, g in C.items()}
    dshare = {}
    for s in C:
        x = cod[cod.source == s].referent
        dshare[s] = {"different": int((x == "different").sum()), "same": int((x == "same").sum()),
                     "n": int(len(x))}
    boots = {s: [] for s in C}
    beff = {s: {"main": [], "strict": []} for s in C}
    for _ in range(NBOOT):
        for s, g in C.items():
            b = B_of(hier_sample(g, rng))
            boots[s].append(b)
            x = cod[cod.source == s].referent.to_numpy()
            xs = x[rng.integers(0, len(x), len(x))]
            nd, nsame = (xs == "different").sum(), (xs == "same").sum()
            dm = nd / (nd + nsame) if nd + nsame else np.nan
            beff[s]["main"].append(1 + (b - 1) * dm)
            beff[s]["strict"].append(1 + (b - 1) * nd / len(xs))
    res = {}
    for s in C:
        bs = np.array(boots[s])
        dd = dshare[s]
        dm = dd["different"] / (dd["different"] + dd["same"])
        ds = dd["different"] / dd["n"]
        res[s] = {"B": point[s], "B_lo": np.percentile(bs, 2.5), "B_hi": np.percentile(bs, 97.5),
                  "coded": dd,
                  "B_eff": 1 + (point[s] - 1) * dm,
                  "B_eff_lo": np.nanpercentile(beff[s]["main"], 2.5),
                  "B_eff_hi": np.nanpercentile(beff[s]["main"], 97.5),
                  "B_eff_strict": 1 + (point[s] - 1) * ds,
                  "B_eff_strict_lo": np.percentile(beff[s]["strict"], 2.5),
                  "B_eff_strict_hi": np.percentile(beff[s]["strict"], 97.5)}
    for a, b in (("childes", "pile10k"), ("childes", "cabank"), ("cabank", "pile10k")):
        diff = np.array(boots[a]) - np.array(boots[b])
        res[f"{a}_minus_{b}"] = {"diff": point[a] - point[b], "lo": np.percentile(diff, 2.5),
                                 "hi": np.percentile(diff, 97.5),
                                 "share_boot_le_0": float((diff <= 0).mean())}
        de = np.array(beff[a]["main"]) - np.array(beff[b]["main"])
        res[f"{a}_minus_{b}_B_eff"] = {"diff": res[a]["B_eff"] - res[b]["B_eff"],
                                       "lo": np.nanpercentile(de, 2.5), "hi": np.nanpercentile(de, 97.5)}
    return res


def interp_index(B, curve):
    xs = np.log([b for _, b in curve])
    ys = [curve_means[c] for c, _ in curve]
    return float(np.interp(np.log(B), xs, ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/cl_revision"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = runs()
    rows = seed_rows(d)
    rows.round(4).to_csv(a.out / "seed_level_tests.csv", index=False)
    print(rows.round(3).to_string(index=False))
    tr = trend_by_seed(d)
    ns = noun_specific(d)
    ns.round(4).to_csv(a.out / "noun_specific_by_condition.csv", index=False)
    print(ns.round(3).to_string(index=False))
    global curve_means
    curve_means = tr["means"]
    cor = corpora(rng)
    for s in ("childes", "cabank", "pile10k"):
        for k in ("B_eff", "B_eff_lo", "B_eff_hi", "B_eff_strict", "B_eff_strict_lo", "B_eff_strict_hi"):
            cor[s]["index_at_" + k] = interp_index(cor[s][k], GRADED)
    out = {"trend": tr, "corpora": cor,
           "notes": "Seed-level tests average the three sizes within each corpus seed. B_eff index "
                    "values interpolate the six Regular-order means linearly in log B."}
    (a.out / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    main()
