#!/usr/bin/env python3
"""
cl_round3_analyses.py | Exploratory checks added after the third round of reviews.

1. Study 3 token balance. In the English items, the repeated and the new feature
   word can differ in how many tokens they take. An untrained model pays about
   ln(vocabulary) per extra token, so these items give large deltas before any
   training (the untrained index of about 0.5 nats). We find the word pairs whose
   two words have the same token count by their untrained deltas (|mean delta|
   over the seven step-0 Pythia models; the split is bimodal: 98 pairs below
   1 nat, 70 above 8) and recompute the shape-over-colour index on those items.
   Token counts are inferred from Pythia's step-0 models, so the split applies
   to Pythia (other families use other tokenisers).

2. Study 1 condition-level agreement between the model index and the noun-blind
   one-sentence reference of Table E3 (descriptive; conditions share seeds).

Outputs: results/cl_revision/study3_token_balanced.csv,
         results/cl_revision/study3_token_balanced.json,
         results/cl_revision/reference_agreement.json
Run from the repo root: python scripts/cl_round3_analyses.py
"""

import json
import re

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

OUT = "results/cl_revision"
rng = np.random.default_rng(2031)


def load_study3():
    d = pd.read_csv("results/study1d/scores_merged.csv", dtype={"revision": str}, keep_default_na=False)
    d["match"] = d["match"].astype(str).isin(["True", "true"])
    it = pd.read_csv("data/wug_tests/english_items.csv")
    d = d.merge(it[["item_id", "rep_value", "new_value"]], on="item_id")
    d["label"] = d.model_id.str.split("/").str[-1]
    return d


def index_ci(g, n_boot=4000):
    """Shape-over-colour index with a 95% interval resampling nouns."""
    s = g[g.probe_dim == "shape"].groupby("noun").delta.mean()
    c = g[g.probe_dim == "colour"].groupby("noun").delta.mean()
    v = (s - c).dropna().to_numpy()
    b = [rng.choice(v, len(v)).mean() for _ in range(n_boot)]
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def study3_token_balance():
    d = load_study3()
    u = d[(d.family == "pythia") & (d.revision == "step0")]
    pair = u.groupby(["probe_dim", "rep_value", "new_value"]).delta.mean()
    balanced = set(pair[pair.abs() < 3].index)
    d["balanced"] = [(a, b, c) in balanced for a, b, c in zip(d.probe_dim, d.rep_value, d.new_value)]
    k = d[(d.k == 1) & d.match]
    final = k[((k.family == "pythia") & (k.revision == "step143000")) | (k.family != "pythia")]
    rows = []
    for lab, g in final.groupby("label"):
        a, alo, ahi = index_ci(g)
        b, blo, bhi = index_ci(g[g.balanced])
        rows.append({"model": lab, "family": g.family.iloc[0], "params": int(g.params.iloc[0]),
                     "index_all": a, "all_lo": alo, "all_hi": ahi,
                     "index_balanced": b, "balanced_lo": blo, "balanced_hi": bhi})
    res = pd.DataFrame(rows).sort_values("params")
    res.round(4).to_csv(f"{OUT}/study3_token_balanced.csv", index=False)
    u0 = k[(k.family == "pythia") & (k.revision == "step0")]
    untrained_all = {l: index_ci(g, 1)[0] for l, g in u0.groupby("label")}
    untrained_bal = {l: index_ci(g[g.balanced], 1)[0] for l, g in u0.groupby("label")}
    py = res[res.family == "pythia"]
    summary = {
        "pairs_total": int(len(pair)),
        "pairs_balanced_abs_delta_below_1": int((pair.abs() < 1).sum()),
        "pairs_unbalanced_abs_delta_above_8": int((pair.abs() > 8).sum()),
        "share_items_balanced_by_dim": k.groupby("probe_dim").balanced.mean().round(3).to_dict(),
        "untrained_index_all_items": {l: round(v, 3) for l, v in untrained_all.items()},
        "untrained_index_balanced_items": {l: round(v, 3) for l, v in untrained_bal.items()},
        "pythia_size_rho_all": float(spearmanr(py.params, py.index_all)[0]),
        "pythia_size_rho_balanced": float(spearmanr(py.params, py.index_balanced)[0]),
        "pythia_size_p_balanced": float(spearmanr(py.params, py.index_balanced)[1]),
    }
    json.dump(summary, open(f"{OUT}/study3_token_balanced.json", "w"), indent=1)
    print(res.round(2).to_string(index=False))
    print(json.dumps(summary, indent=1))


def reference_agreement():
    """Model index against the noun-blind reference, read from the Table E3 sources."""
    ref = pd.read_csv(f"{OUT}/corpus_conditionals.csv").set_index("corpus").noun_blind_index
    t2 = pd.read_csv("results/paper_tables/table2.csv").set_index("condition").index_mean
    # the 12 conditions of Table E3 (corpus label -> Table 2 label)
    names = {"Regular, random": "Regular, random", "Regular, B 1.4": "Regular, B 1.4",
             "Regular, B 3.6": "Regular, B 3.6", "Regular, B 7": "Regular, B 7",
             "Regular, B 12": "Regular, B 12", "Regular, grouped": "Regular, grouped",
             "No stable feature, grouped": "No stable feature, grouped",
             "Colour stable, grouped": "Colour stable, grouped",
             "Objects, same-object B 3.6": "Objects, same-object, B 3.6",
             "Objects, different-object B 3.6": "Objects, different-object, B 3.6",
             "Objects, same-object grouped": "Objects, same-object, grouped",
             "Objects, different-object grouped": "Objects, different-object, grouped"}
    model = {c: float(t2[n]) for c, n in names.items()}
    x = np.array([ref[c] for c in model]); y = np.array(list(model.values()))
    r, p = pearsonr(x, y)
    rho, prho = spearmanr(x, y)
    keep = [c for c in model if c != "Objects, same-object grouped"]
    r2, _ = pearsonr([ref[c] for c in keep], [model[c] for c in keep])
    out = {"n_conditions": len(model), "pearson_r": round(float(r), 3), "pearson_p": float(p),
           "spearman_rho": round(float(rho), 3), "spearman_p": float(prho),
           "pearson_r_without_same_object_grouped": round(float(r2), 3)}
    json.dump(out, open(f"{OUT}/reference_agreement.json", "w"), indent=1)
    print(out)


if __name__ == "__main__":
    study3_token_balance()
    reference_agreement()
