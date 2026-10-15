#!/usr/bin/env python3
"""
amendment8_tests.py  |  Registered tests for Amendment 8 (fresh order; shape reassignment).
Written before any Amendment 8 model was trained.

Conditions. 15 models each (3 sizes x 5 seeds), trained with train_model.py
--fresh_windows on the corpora of make_fresh_corpora.py. Labels from
  oneshot_eval.py make-manifest --conditions regular --label_suffix _f_<cond>
  study1_trained_nouns.py eval --label regular_f_<cond>
so a condition is labelled regular_f_<cond>, with <cond> one of
  regular_random, regular_b3p6, regular_grouped, nostable_grouped,
  objects_same, objects_diff, reassigned_grouped, reassigned_random.

Measures (as in Study 1). OI = Delta shape - Delta colour at k = 1 (the "OI_colour"
column). Index: OI after the same noun (match True). Noun-specific index: OI after the
same noun minus OI after a different noun, per model.

Primary test. Seed level: the three sizes are averaged within each corpus seed, and the
five seed means are compared with a one-sided paired t test (two conditions) or a
one-sided one-sample t test against 0 (4 degrees of freedom). Holm within each of the two
registered families: Part A (H23 to H26) and Part B (H27 to H29).
Alongside (not used for the decision): mixed model with size as a fixed effect and a
random intercept for seed, as in Studies 1 and 1x.

Confirmatory (one-sided; Holm within Part A, H23 to H26, and within Part B, H27 to H29):
  H23  index: regular_grouped    > regular_random
  H24  index: regular_b3p6       > regular_random
  H25  index: regular_grouped    > nostable_grouped
  H26  index: objects_diff       > objects_same
  H27  noun-specific index, novel nouns:  reassigned_grouped > 0
  H28  noun-specific index, novel nouns:  reassigned_grouped > regular_grouped
  H29  noun-specific index, trained nouns: reassigned_grouped > 0

Secondary (registered, no correction; two-sided where a direction is not stated):
  S35  memorisation: loss on own training file vs held-out arrangement (heldout_loss.py),
       fresh models and the matching fixed-corpus Study 1 models; gap by condition and
       the seed-level difference in gap, fixed minus fresh
  S36  each fresh condition vs its fixed-corpus counterpart (index), and H23 to H26
       computed on the fixed-corpus counterparts, with the difference in effect
  S37  reassigned_random: index and both noun-specific indices, and reassigned_grouped
       vs reassigned_random on each
  S38  components (Delta shape, colour, texture) per condition; H23 to H29 by size
  S39  index: reassigned_grouped vs regular_grouped (fresh), two-sided
  S40  corpus conditionals of the held-out files (corpus_conditionals.py counts)

Usage:
  python scripts/amendment8_tests.py --runs results/amendment8/summary_per_run.csv results/rescore/study1bc_per_run.csv \
      results/rescore/study1x_per_run.csv results/study1o/summary_per_run.csv \
      --trained results/amendment8/trained_per_run.csv --loss results/amendment8/heldout_loss.csv \
      --heldout_dir E:/overhypothesis_fresh/heldout --out results/amendment8/tests
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
P = "regular_f_"
FRESH = [P + c for c in ("regular_random", "regular_b3p6", "regular_grouped", "nostable_grouped",
                         "objects_same", "objects_diff", "reassigned_grouped", "reassigned_random")]
# fixed-corpus counterparts (labels in the Study 1, 1x and Amendment 7 per-run files)
FIXED = {P + "regular_random": "regular", P + "regular_b3p6": "regular_graded_b3p6",
         P + "regular_grouped": "regular_bursty", P + "nostable_grouped": "regular_nostable_grouped",
         P + "objects_same": "regular_o_same_grouped", P + "objects_diff": "regular_o_diff_grouped"}
INDEX_H = [("H23", P + "regular_grouped", P + "regular_random"),
           ("H24", P + "regular_b3p6", P + "regular_random"),
           ("H25", P + "regular_grouped", P + "nostable_grouped"),
           ("H26", P + "objects_diff", P + "objects_same")]
RG, RR, FG = P + "reassigned_grouped", P + "reassigned_random", P + "regular_grouped"
FAMILIES = {"A": ["H23", "H24", "H25", "H26"], "B": ["H27", "H28", "H29"]}


def seed_means(df, col):
    return df.groupby("seed")[col].mean().sort_index()


def seed_paired(x, y, two_sided=False):
    """x, y: per-model frames (size, seed, value) of two conditions; value column 'v'."""
    sx, sy = seed_means(x, "v"), seed_means(y, "v")
    if list(sx.index) != list(sy.index) or len(sx) != 5:
        raise SystemExit(f"seed mismatch: {list(sx.index)} vs {list(sy.index)}")
    d = (sx - sy).values
    return one_sample(d, two_sided)


def one_sample(d, two_sided=False):
    d = np.asarray(d, float)
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    t = m / se if se > 0 else np.inf * np.sign(m)
    q = stats.t.ppf(0.975, n - 1)
    p = 2 * stats.t.sf(abs(t), n - 1) if two_sided else stats.t.sf(t, n - 1)
    return {"mean": m, "ci95": [m - q * se, m + q * se], "t": t, "df": n - 1,
            "p_two_sided" if two_sided else "p_one_sided": float(p),
            "seeds_positive": int((d > 0).sum()), "n_seeds": n}


def mlm_intercept(df):
    """Mixed model v ~ 1 + size (sum-coded), random intercept for seed: grand mean > 0."""
    m = smf.mixedlm("v ~ C(size, Sum)", df, groups=df["seed"]).fit(reml=True)
    est, se = m.params["Intercept"], m.bse["Intercept"]
    return {"mlm_est": est, "mlm_se": se, "mlm_p_one_sided": float(stats.norm.sf(est / se))}


def mlm_contrast(x, y):
    s = pd.concat([x.assign(tr=1), y.assign(tr=0)])
    m = smf.mixedlm("v ~ tr + C(size)", s, groups=s["seed"]).fit(reml=True)
    est, se = m.params["tr"], m.bse["tr"]
    return {"mlm_est": est, "mlm_se": se, "mlm_p_one_sided": float(stats.norm.sf(est / se))}


def index_frame(k1, cond):
    return k1[k1.condition == cond][["size", "seed", "OI_colour"]].rename(columns={"OI_colour": "v"})


def ns_frame(d, cond):
    """Noun-specific index per model: OI(match) - OI(no match), k = 1."""
    s = d[(d.k == 1) & (d.condition == cond)]
    if s.empty:
        return pd.DataFrame(columns=["size", "seed", "v"])
    w = s.pivot_table(index=["size", "seed"], columns="match", values="OI_colour")
    return (w[True] - w[False]).reset_index(name="v")


def by_size(fx, fy=None):
    out = {}
    for z in ("tiny", "small", "medium"):
        a = fx[fx["size"] == z].set_index("seed").v.sort_index()
        if fy is None:
            out[z] = {"mean": a.mean(), "models_positive": int((a > 0).sum())}
        else:
            b = fy[fy["size"] == z].set_index("seed").v.sort_index()
            dd = (a - b).dropna()
            out[z] = {"mean_diff": dd.mean(), "models_positive": int((dd > 0).sum())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="per-run summaries, novel-noun items")
    ap.add_argument("--trained", nargs="+", required=True, help="per-run summaries, trained-noun items")
    ap.add_argument("--loss", default=None, help="heldout_loss.py output (S35)")
    ap.add_argument("--heldout_dir", default=None, help="held-out corpora folder (S40)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = load(a.runs)
    t = load(a.trained)
    k1 = d[(d.k == 1) & d.match]
    # Every condition used must have 15 models (3 sizes x 5 seeds) at k = 1 with both match levels.
    need = [(c, "novel", d) for c in FRESH + list(FIXED.values())] + [(c, "trained", t) for c in FRESH]
    for c, name, src in need:
        s1 = src[(src.k == 1) & (src.condition == c)]
        cells = s1.groupby(["size", "seed", "match"]).size()
        if len(cells) != 30 or s1.OI_colour.isna().any():
            raise SystemExit(f"{c} ({name} nouns): {len(cells)} size x seed x match cells, expected 30")

    # ---------------------------------------------------------------- confirmatory
    res = {}
    for name, xc, yc in INDEX_H:
        fx, fy = index_frame(k1, xc), index_frame(k1, yc)
        res[name] = {"measure": "index", "x": xc, "y": yc, "mean_x": fx.v.mean(), "mean_y": fy.v.mean(),
                     **seed_paired(fx, fy), **mlm_contrast(fx, fy)}
    ns_rg, ns_fg = ns_frame(d, RG), ns_frame(d, FG)
    res["H27"] = {"measure": "noun-specific index, novel nouns", "x": RG, **one_sample(seed_means(ns_rg, "v")),
                  **mlm_intercept(ns_rg)}
    res["H28"] = {"measure": "noun-specific index, novel nouns", "x": RG, "y": FG,
                  "mean_x": ns_rg.v.mean(), "mean_y": ns_fg.v.mean(), **seed_paired(ns_rg, ns_fg),
                  **mlm_contrast(ns_rg, ns_fg)}
    tns_rg = ns_frame(t, RG)
    res["H29"] = {"measure": "noun-specific index, trained nouns", "x": RG, **one_sample(seed_means(tns_rg, "v")),
                  **mlm_intercept(tns_rg)}
    if any(not np.isfinite(v["p_one_sided"]) for v in res.values()):
        raise SystemExit("a confirmatory p value is not finite")
    for fam in FAMILIES.values():
        h = holm({k: res[k]["p_one_sided"] for k in fam})
        for k in fam:
            res[k]["holm_family"] = fam
            res[k]["p_holm"] = h[k]
            res[k]["supported"] = bool(h[k] < 0.05)

    # ---------------------------------------------------------------- secondary
    sec = {}
    if a.loss and os.path.exists(a.loss):
        L = pd.read_csv(a.loss)
        L["regime"] = np.where(L.condition.astype(str).str.startswith(P), "fresh", "fixed")
        sec["S35_loss_by_condition"] = L.groupby(["regime", "condition"])[
            ["train_loss_last5", "loss_own", "loss_heldout", "gap"]].mean().round(4).reset_index().to_dict("records")
        pairs = {}
        for fc, xc in FIXED.items():
            gf = L[L.condition == fc].rename(columns={"gap": "v"})
            gx = L[L.condition == xc].rename(columns={"gap": "v"})
            if len(gf) and len(gx):
                pairs[fc] = {"fixed": xc, "gap_fixed": gx.v.mean(), "gap_fresh": gf.v.mean(),
                             **seed_paired(gx, gf, two_sided=True)}
        sec["S35_gap_fixed_minus_fresh"] = pairs

    s36 = {}
    for fc, xc in FIXED.items():
        fx, fy = index_frame(k1, fc), index_frame(k1, xc)
        if len(fx) and len(fy):
            s36[fc] = {"fixed": xc, "mean_fresh": fx.v.mean(), "mean_fixed": fy.v.mean(),
                       **seed_paired(fx, fy, two_sided=True)}
    for name, xc, yc in INDEX_H:
        fx, fy, gx, gy = (index_frame(k1, c) for c in (xc, yc, FIXED[xc], FIXED[yc]))
        if all(len(f) for f in (fx, fy, gx, gy)):
            fresh_d = seed_means(fx, "v") - seed_means(fy, "v")
            fixed_d = seed_means(gx, "v") - seed_means(gy, "v")
            s36[name + "_fixed_counterpart"] = {"fresh_effect": one_sample(fresh_d, True),
                                                "fixed_effect": one_sample(fixed_d, True),
                                                "fresh_minus_fixed": one_sample(fresh_d - fixed_d, True)}
    sec["S36_fresh_vs_fixed"] = s36

    ns_rr, tns_rr, tns_fg = ns_frame(d, RR), ns_frame(t, RR), ns_frame(t, FG)
    sec["S37_reassigned_random"] = {
        "index": one_sample(seed_means(index_frame(k1, RR), "v"), True),
        "ns_novel": one_sample(seed_means(ns_rr, "v"), True),
        "ns_trained": one_sample(seed_means(tns_rr, "v"), True),
        "grouped_minus_random_index": seed_paired(index_frame(k1, RG), index_frame(k1, RR), True),
        "grouped_minus_random_ns_novel": seed_paired(ns_rg, ns_rr, True),
        "grouped_minus_random_ns_trained": seed_paired(tns_rg, tns_rr, True)}

    g = k1[k1.condition.isin(FRESH)].groupby("condition")
    sec["S38_components"] = g[["shape", "colour", "texture", "OI_colour", "OI_texture"]].mean().round(4).to_dict("index")
    sec["S38_by_size"] = {name: by_size(index_frame(k1, xc), index_frame(k1, yc)) for name, xc, yc in INDEX_H}
    sec["S38_by_size"]["H27"] = by_size(ns_rg)
    sec["S38_by_size"]["H28"] = by_size(ns_rg, ns_fg)
    sec["S38_by_size"]["H29"] = by_size(tns_rg)
    sec["S38_ns_trained_all_fresh"] = {c: one_sample(seed_means(ns_frame(t, c), "v"), True)
                                       for c in FRESH if len(ns_frame(t, c))}
    sec["S38_ns_novel_all_fresh"] = {c: one_sample(seed_means(ns_frame(d, c), "v"), True)
                                     for c in FRESH if len(ns_frame(d, c))}

    sec["S39_index_reassigned_vs_regular_grouped"] = seed_paired(index_frame(k1, RG), index_frame(k1, FG), True)

    if a.heldout_dir:
        import corpus_conditionals as cc
        rows = []
        for c in FRESH:
            cond = c[len(P):]
            tot = None
            for s in cc.SEEDS:
                path = os.path.join(a.heldout_dir, cond, f"regular_seed{s}.txt")
                if not os.path.exists(path):
                    break
                x = cc.counts(path)
                tot = x if tot is None else {k: tot[k] + x[k] for k in tot}
            if tot:
                s_all = (tot["s_rep_same"] + tot["s_rep_diff"], tot["s_n_same"] + tot["s_n_diff"])
                c_all = (tot["c_rep_same"] + tot["c_rep_diff"], tot["c_n_same"] + tot["c_n_diff"])
                rows.append({"condition": c, "share_same_kind_next": tot["same"] / tot["pairs"],
                             "P_shape_rep_same_kind": tot["s_rep_same"] / max(tot["s_n_same"], 1),
                             "P_shape_rep_diff_kind": tot["s_rep_diff"] / max(tot["s_n_diff"], 1),
                             "P_colour_rep_same_kind": tot["c_rep_same"] / max(tot["c_n_same"], 1),
                             "P_colour_rep_diff_kind": tot["c_rep_diff"] / max(tot["c_n_diff"], 1),
                             "noun_blind_index": cc.D(*s_all) - cc.D(*c_all),
                             "noun_aware_index_same_kind": cc.D(tot["s_rep_same"], tot["s_n_same"])
                             - cc.D(tot["c_rep_same"], tot["c_n_same"]),
                             "noun_aware_index_diff_kind": cc.D(tot["s_rep_diff"], tot["s_n_diff"])
                             - cc.D(tot["c_rep_diff"], tot["c_n_diff"])})
        sec["S40_corpus_conditionals"] = rows

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"confirmatory": res, "secondary": sec}, open(a.out + ".json", "w"), indent=2, default=float)
    print(json.dumps({k: {m: v.get(m) for m in ("mean", "ci95", "p_one_sided", "p_holm", "supported",
                                                "seeds_positive", "mlm_est", "mlm_p_one_sided")}
                      for k, v in res.items()}, indent=2, default=float))


if __name__ == "__main__":
    main()
