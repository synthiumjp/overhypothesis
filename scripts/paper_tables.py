#!/usr/bin/env python3
"""
paper_tables.py | Rebuild Tables 2, D1, E2 and F1 of the paper from the per-model score files.

Inputs (corrected scores; see results/rescore/RESULTS.md):
  results/rescore/study1bc_per_run.csv   Regular, Mixed-slot and further controls; grouped Regular and Mixed-slot
  results/rescore/study1x_per_run.csv    graded orders, No stable feature, Colour stable
  results/rescore/study1w_per_run.csv    512-token windows
  results/study1o/summary_per_run.csv    Objects corpora
  results/rescore/trained_per_run.csv, results/study1o/trained_per_run.csv   trained nouns (Table F1)
Each per-run file has one row per model x number of exemplars (k) x noun match, with the mean
Δ for shape, colour and texture and the indices OI_colour (shape over colour) and OI_texture.

Usage: python scripts/paper_tables.py [--out results/paper_tables]
"""
import argparse
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd
from scipy import stats

FILES = ["results/rescore/study1bc_per_run.csv", "results/rescore/study1x_per_run.csv",
         "results/rescore/study1w_per_run.csv", "results/study1o/summary_per_run.csv"]
TRAINED = ["results/rescore/trained_per_run.csv", "results/study1o/trained_per_run.csv"]
MAIN = [("Regular, random", "regular"), ("Regular, B 1.4", "regular_graded_b1p4"),
        ("Regular, B 3.6", "regular_graded_b3p6"), ("Regular, B 7", "regular_graded_b7"),
        ("Regular, B 12", "regular_graded_b12"), ("Regular, grouped", "regular_bursty"),
        ("Mixed-slot, random", "scrambled"), ("Mixed-slot, grouped", "scrambled_bursty"),
        ("No stable feature, random", "regular_nostable_random"), ("No stable feature, grouped", "regular_nostable_grouped"),
        ("Colour stable, random", "regular_colourstable_random"), ("Colour stable, grouped", "regular_colourstable_grouped"),
        ("Regular, random, 512", "regular_w512"), ("Regular, B 3.6, 512", "regular_graded_b3p6_w512"),
        ("No stable feature, random, 512", "regular_nostable_random_w512"),
        ("Objects, random", "regular_o_random"), ("Objects, same-object, B 3.6", "regular_o_same_b3p6"),
        ("Objects, different-object, B 3.6", "regular_o_diff_b3p6"), ("Objects, same-object, grouped", "regular_o_same_grouped"),
        ("Objects, different-object, grouped", "regular_o_diff_grouped")]
CONTROLS = [("Regular (for comparison)", "regular"), ("No kind label", "bare_nolabel"),
            ("Paraphrased", "paraphrased_nolabel"), ("Weak label", "weak_label"),
            ("Frequency matched", "frequency_matched"), ("Added noise", "noise_injection"), ("Feature swap", "feature_swap")]
TRAINED_ROWS = [("Regular, random", "regular"), ("Regular, B 3.6", "regular_graded_b3p6"),
                ("Regular, B 7", "regular_graded_b7"), ("Regular, B 12", "regular_graded_b12"),
                ("Regular, grouped", "regular_bursty"), ("Objects, same-object, grouped", "regular_o_same_grouped"),
                ("Objects, different-object, grouped", "regular_o_diff_grouped")]


def r(v, n=2):
    q = Decimal(1).scaleb(-n)
    return float(Decimal(repr(float(v))).quantize(q, rounding=ROUND_HALF_UP))


def load(files):
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d[d.source == "lm"].copy()
    d["match"] = d["match"].astype(str) == "True"
    return d


def noun_specific(d, cond):
    p = d[(d.condition == cond) & (d.k == 1)].pivot_table(index=["size", "seed"], columns="match", values="OI_colour")
    return p[True] - p[False], p[True]


def table2(d):
    rows = []
    for label, cond in MAIN:
        g = d[(d.condition == cond) & (d.k == 1) & d.match]
        rows.append({"condition": label, "d_shape": r(g["shape"].mean()), "d_colour": r(g.colour.mean()),
                     "index_mean": r(g.OI_colour.mean()), "index_sd": r(g.OI_colour.std()),
                     "models_above_zero": int((g.OI_colour > 0).sum()), "n": len(g),
                     "shape_over_texture": r(g.OI_texture.mean())})
    return pd.DataFrame(rows)


def tableD1(d):
    rows = []
    for label, cond in CONTROLS:
        g = d[(d.condition == cond) & (d.k == 1) & d.match]
        ns, _ = noun_specific(d, cond)
        rows.append({"corpus": label, "index_mean": r(g.OI_colour.mean()), "index_sd": r(g.OI_colour.std()),
                     "shape_over_texture": r(g.OI_texture.mean()), "noun_specific": r(ns.mean(), 3), "n": len(g)})
    return pd.DataFrame(rows)


def tableE2(d):
    rows = []
    for label, cond in MAIN:
        g = d[(d.condition == cond) & d.match]
        k1 = g[g.k == 1].groupby("size").OI_colour.mean()
        rows.append({"condition": label, "3.4M": r(k1["tiny"]), "11.0M": r(k1["small"]), "25.6M": r(k1["medium"]),
                     "two_exemplars": r(g[g.k == 2].OI_colour.mean()), "four_exemplars": r(g[g.k == 4].OI_colour.mean())})
    return pd.DataFrame(rows)


def tableF1(t):
    rows = []
    for label, cond in TRAINED_ROWS:
        ns, same = noun_specific(t, cond)
        test = stats.ttest_1samp(ns, 0)
        rows.append({"condition": label, "index_same_noun": r(same.mean()), "noun_specific": r(ns.mean()),
                     "models_above_zero": int((ns > 0).sum()), "n": len(ns), "p_two_sided": r(test.pvalue, 3),
                     "p_one_sided_greater": r(test.pvalue / 2 if ns.mean() > 0 else 1 - test.pvalue / 2, 3)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/paper_tables"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d, t = load(FILES), load(TRAINED)
    for name, df in (("table2", table2(d)), ("tableD1", tableD1(d)), ("tableE2", tableE2(d)), ("tableF1", tableF1(t))):
        df.to_csv(a.out / f"{name}.csv", index=False)
        print(f"\n{name}\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()
