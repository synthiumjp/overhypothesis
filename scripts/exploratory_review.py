#!/usr/bin/env python3
"""
exploratory_review.py  |  Exploratory analyses of EXISTING data requested in
review (Amendment 5, Part D). Every output here is labelled exploratory.

  results/review/study1_seed_level.csv     registered Study 1 contrasts at the level
                                           of corpus seeds (n = 5), and by model size
  results/review/study1_size_interaction.csv  condition x size mixed models
  results/review/study2_components.csv     the four parts of the kind-specific index
  results/review/study2_crossed.json       crossed random-effects model on the original
                                           items; size correlation of the kind-specific index
  results/review/study2_28b_fix.csv        every pythia-2.8b checkpoint before and after
                                           the stale-weights fix
  results/review/study3_rates.csv          written by study3_extension.py rates

Usage:  python scripts/exploratory_review.py
"""

import importlib.util
import json
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
OUT = "results/review"
CONTRASTS = [("H2b", "regular", "scrambled"), ("H4b", "regular_bursty", "regular"),
             ("H5b", "regular_bursty", "scrambled_bursty")]
PYTHIA = {"pythia-14m": 14e6, "pythia-70m": 70e6, "pythia-160m": 160e6, "pythia-410m": 410e6,
          "pythia-1b": 1e9, "pythia-1.4b": 1.4e9, "pythia-2.8b": 2.8e9, "pythia-6.9b": 6.9e9,
          "pythia-12b": 12e9}


def study1():
    d = pd.read_csv("results/study1c/summary_per_run.csv")
    d["match"] = d["match"].astype(str).isin(["True", "true"])
    k1 = d[(d.k == 1) & d.match]
    rows, inter = [], []
    for name, x_c, y_c in CONTRASTS:
        for col in ("OI_colour", "OI_texture"):
            x = k1[k1.condition == x_c].set_index(["size", "seed"])[col].sort_index()
            y = k1[k1.condition == y_c].set_index(["size", "seed"])[col].sort_index()
            sx, sy = x.groupby("seed").mean(), y.groupby("seed").mean()
            t = stats.ttest_rel(sx, sy, alternative="greater")
            diff = sx - sy
            r = {"test": name, "measure": col, "seed_mean_diff": diff.mean(),
                 "seed_diff_sd": diff.std(), "seed_t": t.statistic, "seed_p_one_sided": t.pvalue,
                 "seeds_positive": int((diff > 0).sum())}
            for z in ("tiny", "small", "medium"):
                r[f"diff_{z}"] = (x.xs(z) - y.xs(z)).mean()
            rows.append(r)
            s = k1[k1.condition.isin([x_c, y_c])].copy()
            s["tr"] = (s.condition == x_c).astype(int)
            m = smf.mixedlm(f"{col} ~ tr * C(size, Treatment('medium'))", s, groups=s["seed"]).fit(reml=True)
            m0 = smf.mixedlm(f"{col} ~ tr + C(size)", s, groups=s["seed"]).fit(reml=False)
            m1 = smf.mixedlm(f"{col} ~ tr * C(size)", s, groups=s["seed"]).fit(reml=False)
            lr = 2 * (m1.llf - m0.llf)
            inter.append({"test": name, "measure": col,
                          "interaction_LR": lr, "interaction_p": stats.chi2.sf(lr, 2),
                          **{k: v for k, v in m.params.items() if k.startswith("tr")}})
    pd.DataFrame(rows).round(4).to_csv(f"{OUT}/study1_seed_level.csv", index=False)
    pd.DataFrame(inter).round(4).to_csv(f"{OUT}/study1_size_interaction.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(pd.DataFrame(inter).round(3).to_string(index=False))


def study2():
    f = pd.read_csv("results/study1d/kind_specific_final.csv")
    f.round(4).to_csv(f"{OUT}/study2_components.csv", index=False)
    py = f[f.model.isin(PYTHIA)].copy()
    py["params"] = py.model.map(PYTHIA)
    rho, p = stats.spearmanr(py.params, py.shape_minus_colour)
    rng = np.random.default_rng(2027)
    perm = [stats.spearmanr(py.params, rng.permutation(py.shape_minus_colour.to_numpy()))[0]
            for _ in range(10_000)]
    res = {"kind_specific_size_rho": rho, "perm_p_one_sided": (1 + sum(q >= rho for q in perm)) / 10_001}
    # Crossed random effects on the original items (k = 1), pooled Pythia 410M to 12B final.
    d = pd.read_csv("results/study1d/scores_merged.csv", dtype={"revision": str}, keep_default_na=False)
    d["match"] = d["match"].astype(str).isin(["True", "true"])
    it = pd.read_csv("data/wug_tests/english_items.csv")
    d = d[(d.k == 1)].merge(it[["item_id", "rep_value"]], on="item_id")
    d["model"] = d.model_id.str.split("/").str[-1]
    d = d[d.model.isin([m for m, s in PYTHIA.items() if s >= 4e8]) & (d.revision == "step143000")]
    w = d.pivot_table(index=["model", "noun", "draw"], columns=["probe_dim", "match"], values="delta")
    words = d[d.match].pivot_table(index=["model", "noun", "draw"], columns="probe_dim",
                                   values="rep_value", aggfunc="first")
    ks = w.index.to_frame(index=False)
    ks["ks_colour"] = ((w[("shape", True)] - w[("shape", False)]) -
                       (w[("colour", True)] - w[("colour", False)])).to_numpy()
    ks["shape_word"] = words["shape"].to_numpy()
    ks["colour_word"] = words["colour"].to_numpy()
    s2x = importlib.util.spec_from_file_location("s2x", "scripts/study2_extension.py")
    mod = importlib.util.module_from_spec(s2x)
    s2x.loader.exec_module(mod)
    est, se, pm, conv = mod.crossed_model(ks)
    m, lo, hi, pb = mod.pigeonhole_boot(ks, rng)
    res["crossed_original_items_410m_to_12b"] = {
        "estimate": est, "se": se, "p_one_sided": pm, "converged": conv,
        "pigeonhole_mean": m, "pigeonhole_ci95": [lo, hi], "pigeonhole_p": pb,
        "n_rows": len(ks), "note": "original items: same noun with 'another', different noun with 'a'"}
    json.dump(res, open(f"{OUT}/study2_crossed.json", "w"), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))


def study2_28b():
    def oi(path):
        d = pd.read_csv(path, dtype={"revision": str}, keep_default_na=False)
        d = d[d.model_id.str.contains("pythia-2.8b") & (d.k == 1) &
              d.match.astype(str).isin(["True", "true"])]
        w = d.groupby(["revision", "probe_dim"]).delta.mean().unstack()
        return (w["shape"] - w["colour"]).rename("OI_colour")
    before, after = oi("results/study1d/raw/scores.csv"), oi("results/study1d/fix28b/scores_28b_fix.csv")
    merged = pd.read_csv("results/study1d/scores_merged.csv", dtype={"revision": str}, keep_default_na=False)
    used = set(merged[merged.model_id.str.contains("pythia-2.8b")].revision)
    t = pd.DataFrame({"OI_colour_before": before, "OI_colour_after": after})
    t["in_analysis"] = [r in used for r in t.index]
    t["step"] = t.index.str.replace("step", "").astype(int)
    t = t.sort_values("step")
    t.round(4).to_csv(f"{OUT}/study2_28b_fix.csv")
    print(t.round(3).to_string())


def main():
    os.makedirs(OUT, exist_ok=True)
    study1()
    study2()
    study2_28b()


if __name__ == "__main__":
    main()
