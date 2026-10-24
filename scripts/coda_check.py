#!/usr/bin/env python3
"""
coda_check.py  |  Checks the colour-typicality codes of the Experiment 3 nouns (Amendment 10 Part B)
against human colour norms: CoDa (Paik et al., 2021, EMNLP; https://github.com/nala-cub/coda),
in which crowd workers rated how often objects of each class have each of 11 basic colours.
Exploratory; added after the Part B results.

For each CoDa object: normalise each included annotation over the 11 colours, average over
annotators, and take the entropy (nats) and the most likely colour. Nouns of data/amendment10/nouns.csv
are matched to CoDa objects by lower-case singular name.

  git clone --depth 1 https://github.com/nala-cub/coda.git /tmp/coda
  python scripts/coda_check.py --coda_dir /tmp/coda/projects/coda/data/coda --out results/amendment10/coda_check.csv
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

C = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coda_dir", required=True)
    ap.add_argument("--nouns", default="data/amendment10/nouns.csv")
    ap.add_argument("--pythia", default="results/amendment10/pythia_scores.csv")
    ap.add_argument("--out", default="results/amendment10/coda_check.csv")
    a = ap.parse_args()
    obj = [json.loads(x) for x in open(f"{a.coda_dir}/objects.jsonl", encoding="utf-8")]
    ann = [json.loads(x) for x in open(f"{a.coda_dir}/annotations.jsonl", encoding="utf-8")]
    rows = []
    for r in ann:
        if not r.get("included"):
            continue
        v = np.array([r.get(c, 0) for c in C], float)
        if v.sum() > 0:
            rows.append((r["class_id"], *(v / v.sum())))
    A = pd.DataFrame(rows, columns=["class_id"] + C)
    n_annot = A.groupby("class_id").size()
    A = A.groupby("class_id").mean()
    names = {}
    for o in obj:
        for k in ("object_s", "ngram", "display_name"):
            if o.get(k):
                names[o["class_id"]] = o[k].lower()
                break
    A["noun"] = A.index.map(names)
    A["n_annot"] = n_annot
    A["entropy"] = -(A[C] * np.log(A[C].clip(lower=1e-12))).sum(axis=1)
    A["top"] = A[C].idxmax(axis=1).str.replace("gray", "grey")
    A["p_top"] = A[C].max(axis=1)
    nouns = pd.read_csv(a.nouns)
    m = nouns.merge(A[["noun", "entropy", "top", "p_top", "n_annot"]], on="noun", how="left")
    m.to_csv(a.out, index=False)
    mm = m.dropna(subset=["entropy"])
    t = mm[mm.group == "typical"]
    v = mm[mm.group == "variable"]
    print(f"matched {len(mm)} of {len(m)} nouns")
    print(f"human colour entropy: colour-typical {t.entropy.mean():.2f} (n = {len(t)}), "
          f"colour-variable {v.entropy.mean():.2f} (n = {len(v)}); Mann-Whitney p = "
          f"{stats.mannwhitneyu(t.entropy, v.entropy).pvalue:.2g}")
    print(f"colour-typical nouns whose most frequent human colour is the coded colour: "
          f"{(t.top == t.colour).sum()} of {len(t)}")
    p = pd.read_csv(a.pythia)
    p = p[p.condition == "original"]
    for model in sorted(p.model.unique()):
        d = p[p.model == model].merge(mm[["noun", "entropy"]], on="noun")
        r = stats.spearmanr(d.entropy, d.carry)
        print(f"{model}: Spearman(human entropy, carry) = {r.statistic:.2f}, p = {r.pvalue:.3f}, n = {len(d)}")


if __name__ == "__main__":
    main()
