#!/usr/bin/env python3
"""
exploratory_analyses.py  |  Exploratory and robustness analyses for Studies 1b to 1d.

Everything here is labelled exploratory in the paper, except where noted as
promised in an amendment. Each output is a CSV under results/, written from
committed inputs only, so every number in the paper can be regenerated with

    python scripts/exploratory_analyses.py

Outputs
  results/study1b/hbm_corrected_summary.csv   ideal observer, replication-corrected
                                              (promised in Amendment 1)
  results/study1c/kind_specific_by_dim.csv    same-kind minus other-kind repetition,
                                              per dimension, Studies 1b and 1c
  results/study1d/kind_specific_final.csv     the same for every final checkpoint
  results/study1d/kind_specific_trajectory.csv  Pythia, across training
  results/study1d/robustness_leave_one_noun_out.csv  (promised in Amendment 3)
  results/study1d/robustness_raw_delta.csv           (promised in Amendment 3)
  results/study1d/robustness_sentence_order.csv      (promised in Amendment 3)
  results/study1d/robustness_word_effects.csv

Kind-specific index. For each dimension d, the repetition advantage
delta_d is measured twice on the same words: once when the query names the
same noun as the context (matched) and once when it names a different noun
(mismatched). Their difference removes how easily a word is repeated in
general and keeps only what is tied to the kind. Shape minus colour on this
measure equals the registered noun-binding analysis (S6 in Amendment 3).
"""

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SEED = 2027
N_BOOT = 10_000
PYTHIA_ORDER = ["14m", "70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]


def boot_ci(noun_means, rng):
    b = noun_means[rng.integers(0, len(noun_means), (N_BOOT, len(noun_means)))].mean(1)
    return float(noun_means.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def hbm_summary():
    rows = []
    for f in sorted(glob.glob("results/study1b/hbm_corrected/hbm_*_alpha.json")):
        a = json.load(open(f))
        base = os.path.basename(f).replace("hbm_", "").replace("_alpha.json", "")
        cond, seed = base.rsplit("_", 1)
        d = pd.read_csv(f.replace("_alpha.json", ".csv"))
        d = d[(d.k == 1) & d.match.astype(str).isin(["True", "true"])]
        m = d.groupby("probe_dim").delta.mean()
        rows.append({"condition": cond, "seed": int(seed),
                     **{f"alpha_{k}": v["map_alpha"] for k, v in a["alpha"].items()},
                     "OI_colour_k1": m["shape"] - m["colour"],
                     "OI_texture_k1": m["shape"] - m["texture"]})
    per_seed = pd.DataFrame(rows)
    out = per_seed.groupby("condition").agg(["mean", "std"]).drop(columns="seed")
    out.columns = ["_".join(c) for c in out.columns]
    out.round(4).to_csv("results/study1b/hbm_corrected_summary.csv")
    per_seed.round(5).to_csv("results/study1b/hbm_corrected_per_seed.csv", index=False)
    return out


def kind_specific_synthetic(rng):
    d = pd.read_csv("results/study1c/summary_per_run.csv")
    d["match"] = d["match"].astype(str).isin(["True", "true"])
    d = d[(d.source == "lm") & (d.k == 1)]
    w = d.pivot_table(index=["condition", "size", "seed"], columns="match",
                      values=["shape", "colour", "texture"])
    ks = pd.DataFrame({dim: w[(dim, True)] - w[(dim, False)] for dim in ("shape", "colour", "texture")})
    ks["shape_minus_colour"] = ks["shape"] - ks["colour"]
    ks["shape_minus_texture"] = ks["shape"] - ks["texture"]
    ks = ks.reset_index()
    ks.round(5).to_csv("results/study1c/kind_specific_by_dim_per_run.csv", index=False)
    summ = ks.groupby("condition")[["shape", "colour", "texture", "shape_minus_colour",
                                     "shape_minus_texture"]].agg(["mean", "std"])
    summ.columns = ["_".join(c) for c in summ.columns]
    summ.round(4).to_csv("results/study1c/kind_specific_by_dim.csv")
    return summ


def load_1d():
    d = pd.read_csv("results/study1d/scores_merged.csv", dtype={"revision": str}, keep_default_na=False)
    d["match"] = d["match"].astype(str).isin(["True", "true"])
    it = pd.read_csv("data/wug_tests/english_items.csv")
    d = d.merge(it[["item_id", "order", "rep_value", "new_value"]], on="item_id")
    d["label"] = d.model_id.str.split("/").str[-1]
    return d


def is_final(df):
    return ((df.family == "pythia") & (df.revision == "step143000")) | (df.family != "pythia")


def kind_specific_1d(d, rng):
    k1 = d[d.k == 1]
    w = k1.pivot_table(index=["label", "family", "revision", "noun", "draw", "probe_dim"],
                       columns="match", values="delta").reset_index()
    w["kind"] = w[True] - w[False]
    t = w.pivot_table(index=["label", "family", "revision", "noun", "draw"],
                      columns="probe_dim", values="kind").reset_index()
    t["shape_minus_colour"] = t["shape"] - t["colour"]
    t["shape_minus_texture"] = t["shape"] - t["texture"]
    rows = []
    for (lab, fam, rev), g in t.groupby(["label", "family", "revision"]):
        r = {"model": lab, "family": fam, "revision": rev}
        for c in ("shape", "colour", "texture", "shape_minus_colour", "shape_minus_texture"):
            m, lo, hi = boot_ci(g.groupby("noun")[c].mean().to_numpy(), rng)
            r[c], r[c + "_lo"], r[c + "_hi"] = m, lo, hi
        rows.append(r)
    res = pd.DataFrame(rows)
    res["step"] = res.revision.str.extract(r"step(\d+)", expand=False).astype(float)
    final = res[(res.family != "pythia") | (res.revision == "step143000")]
    final.round(4).to_csv("results/study1d/kind_specific_final.csv", index=False)
    traj = res[res.family == "pythia"].sort_values(["model", "step"])
    traj.round(4).to_csv("results/study1d/kind_specific_trajectory.csv", index=False)
    return final, traj


def robustness_1d(d):
    k1m = d[(d.k == 1) & d.match & is_final(d)]
    # Raw delta by dimension (promised in Amendment 3)
    raw = k1m.groupby(["label", "probe_dim"]).delta.mean().unstack().round(4)
    raw.to_csv("results/study1d/robustness_raw_delta.csv")
    # Leave one noun out, H6 statistic (promised in Amendment 3)
    py = k1m[k1m.family == "pythia"]
    sizes = py.groupby("label").params.first().astype(float).sort_values()

    def oi(g):
        w = g.pivot_table(index=["noun", "draw"], columns="probe_dim", values="delta")
        return (w["shape"] - w["colour"]).mean()
    lono = []
    for n in sorted(py.noun.unique()):
        v = [oi(py[(py.label == m) & (py.noun != n)]) for m in sizes.index]
        lono.append({"dropped_noun": n, "H6_rho": spearmanr(sizes.values, v)[0]})
    pd.DataFrame(lono).round(4).to_csv("results/study1d/robustness_leave_one_noun_out.csv", index=False)
    # Sentence order: position of the shape sentence (promised in Amendment 3)
    k = k1m.copy()
    k["shape_position"] = k.order.str.split("-").apply(lambda o: o.index("shape") + 1)
    w = k.pivot_table(index=["label", "noun", "draw", "shape_position"],
                      columns="probe_dim", values="delta").reset_index()
    w["OI_colour"] = w["shape"] - w["colour"]
    w.groupby(["label", "shape_position"]).OI_colour.mean().unstack().round(4) \
        .to_csv("results/study1d/robustness_sentence_order.csv")
    # Word effects: mean repetition advantage per repeated word
    k1m.groupby(["label", "probe_dim", "rep_value"]).delta.mean().unstack(["probe_dim", "rep_value"]) \
        .round(4).to_csv("results/study1d/robustness_word_effects.csv")
    return pd.DataFrame(lono)


def main():
    rng = np.random.default_rng(SEED)
    print("Ideal observer (corrected), k = 1:\n", hbm_summary()[["OI_colour_k1_mean", "OI_texture_k1_mean"]].round(3))
    print("\nKind-specific, Studies 1b and 1c:\n",
          kind_specific_synthetic(rng)[["shape_minus_colour_mean", "shape_minus_texture_mean"]].round(3))
    d = load_1d()
    final, _ = kind_specific_1d(d, rng)
    print("\nKind-specific, Study 1d final checkpoints:\n",
          final[["model", "shape_minus_colour", "shape_minus_colour_lo", "shape_minus_colour_hi",
                 "shape_minus_texture"]].round(2).to_string(index=False))
    print("\nLeave one noun out, H6 rho:", robustness_1d(d).H6_rho.round(2).tolist())


if __name__ == "__main__":
    main()
