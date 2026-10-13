#!/usr/bin/env python3
"""
corpus_conditionals.py | What the Study 1 corpora themselves say about the next sentence
(exploratory; added for the CL revision, 13 Oct 2026).

For every pair of consecutive sentences in a training corpus that both name a shape (and,
for colour, both name a colour), count how often the next sentence repeats the previous
sentence's shape or colour, separately for pairs about the same kind and about different
kinds. From these counts:

  noun-blind reference index  = D(shape) - D(colour), with D(dim) = ln p - ln((1 - p) / 9),
                                p = P(next sentence repeats the value) over all pairs.
     This is what a learner that predicts each feature from the previous sentence alone,
     ignoring the noun, would show on a one-exemplar item (10 values per dimension).
  noun-aware reference        = the same, computed separately for same-kind and
                                different-kind pairs (add-0.5 smoothing).

Usage: python scripts/corpus_conditionals.py --out results/cl_revision/corpus_conditionals.csv
"""
import argparse
import math
import re
from pathlib import Path

import pandas as pd

SHAPES = {"zibby", "krannish", "tova", "felk", "mundi", "prex", "sallo", "wemb", "oofy", "drant"}
COLOURS = {"lavo", "renu", "vassa", "temil", "dorvi", "pulek", "sorba", "navri", "kelun", "morit"}
SEEDS = (42, 123, 456, 789, 1001)
CORPORA = {
    "Regular, random": "data/corpora/regular_seed{s}.txt",
    "Regular, B 1.4": "data/corpora_x/graded_b1p4/regular_seed{s}.txt",
    "Regular, B 3.6": "data/corpora_x/graded_b3p6/regular_seed{s}.txt",
    "Regular, B 7": "data/corpora_x/graded_b7/regular_seed{s}.txt",
    "Regular, B 12": "data/corpora_x/graded_b12/regular_seed{s}.txt",
    "Regular, grouped": "data/corpora_bursty/regular_seed{s}.txt",
    "No stable feature, random": "data/corpora_x/nostable_random/regular_seed{s}.txt",
    "No stable feature, grouped": "data/corpora_x/nostable_grouped/regular_seed{s}.txt",
    "Colour stable, random": "data/corpora_x/colourstable_random/regular_seed{s}.txt",
    "Colour stable, grouped": "data/corpora_x/colourstable_grouped/regular_seed{s}.txt",
    "Objects, random": "data/corpora_o/random/regular_seed{s}.txt",
    "Objects, same-object B 3.6": "data/corpora_o/same_b3p6/regular_seed{s}.txt",
    "Objects, different-object B 3.6": "data/corpora_o/diff_b3p6/regular_seed{s}.txt",
    "Objects, same-object grouped": "data/corpora_o/same_grouped/regular_seed{s}.txt",
    "Objects, different-object grouped": "data/corpora_o/diff_grouped/regular_seed{s}.txt",
}
NOUN = re.compile(r"\b(?:a|the|another)\s+(\w+)\b", re.I)


def parse(line):
    words = re.findall(r"[a-z]+", line.lower())
    noun = None
    m = NOUN.search(line)
    if m:
        noun = m.group(1).lower()
    shape = next((w for w in words if w in SHAPES), None)
    colour = next((w for w in words if w in COLOURS), None)
    return noun, shape, colour


def D(rep, n):
    p = (rep + 0.5) / (n + 1.0)
    return math.log(p) - math.log((1 - p) / 9)


def counts(path):
    sents = [parse(l) for l in open(path, encoding="utf-8") if l.strip()]
    c = {k: 0 for k in ("pairs", "same", "s_n_same", "s_rep_same", "s_n_diff", "s_rep_diff",
                        "c_n_same", "c_rep_same", "c_n_diff", "c_rep_diff")}
    for (n1, s1, c1), (n2, s2, c2) in zip(sents, sents[1:]):
        c["pairs"] += 1
        same = n1 == n2
        c["same"] += same
        k = "same" if same else "diff"
        if s1 and s2:
            c[f"s_n_{k}"] += 1
            c[f"s_rep_{k}"] += s1 == s2
        if c1 and c2:
            c[f"c_n_{k}"] += 1
            c[f"c_rep_{k}"] += c1 == c2
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/cl_revision/corpus_conditionals.csv"))
    a = ap.parse_args()
    rows = []
    for name, pat in CORPORA.items():
        tot = None
        for s in SEEDS:
            c = counts(pat.format(s=s))
            tot = c if tot is None else {k: tot[k] + c[k] for k in tot}
        t = tot
        s_all = (t["s_rep_same"] + t["s_rep_diff"], t["s_n_same"] + t["s_n_diff"])
        c_all = (t["c_rep_same"] + t["c_rep_diff"], t["c_n_same"] + t["c_n_diff"])
        rows.append({
            "corpus": name,
            "share_same_kind_next": t["same"] / t["pairs"],
            "P_shape_rep_same_kind": t["s_rep_same"] / max(t["s_n_same"], 1),
            "P_shape_rep_diff_kind": t["s_rep_diff"] / max(t["s_n_diff"], 1),
            "P_colour_rep_same_kind": t["c_rep_same"] / max(t["c_n_same"], 1),
            "P_colour_rep_diff_kind": t["c_rep_diff"] / max(t["c_n_diff"], 1),
            "noun_blind_index": D(*s_all) - D(*c_all),
            "noun_aware_index_same_kind": D(t["s_rep_same"], t["s_n_same"]) - D(t["c_rep_same"], t["c_n_same"]),
            "noun_aware_index_diff_kind": D(t["s_rep_diff"], t["s_n_diff"]) - D(t["c_rep_diff"], t["c_n_diff"]),
        })
    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(a.out, index=False)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
