#!/usr/bin/env python3
"""
study1d_tests.py  |  Confirmatory and secondary tests for Study 1d
(pre-registered in Amendment 3). Fixed before any model was scored.

Input: the scores CSV written by `english_oneshot.py eval`.

Primary index (all confirmatory tests): OI_colour at k = 1 with a matched noun.
Per model checkpoint, the index is the mean over 96 cells (8 nouns x 12 draws).

Confirmatory tests, one-sided, Holm-corrected together at alpha = .05
  H6   Pythia final checkpoint (step143000): Spearman rho between parameter
       count and OI_colour across the nine sizes. Exact permutation p.
  H7   Pythia, all checkpoints: mean within-size Spearman rho between training
       step and OI_colour. p from 10,000 permutations of OI across steps
       within each size.
  H8a  BabyLM 10M baseline: OI_colour > 0. Cluster bootstrap over nouns.
  H8b  BabyLM 100M baseline: same.

Equivalence (for H8a, H8b): the 90% bootstrap CI inside [-0.2, 0.2] nats.

Secondary (reported, no correction)
  S5   H6 to H8 with OI_texture
  S6   noun binding: matched minus mismatched OI_colour at k = 1
  S7   k = 2 and 4: mean of shape-stable and colour-stable OI_colour (copying
       cancels); evidence use = shape-stable minus colour-stable
  S8   onset: first Pythia step from which every later checkpoint has a 95%
       CI for OI_colour above zero

Usage
  python scripts/study1d_tests.py --scores results/study1d/scores.csv \\
         --out results/study1d/tests
"""

import argparse
import itertools
import json

import numpy as np
import pandas as pd

SESOI = 0.2
N_BOOT = 10_000
N_PERM = 10_000
SEED = 2027
KEYS = ["family", "model_id", "revision", "params", "train_tokens"]


def load_cells(path):
    d = pd.read_csv(path, dtype={"revision": str}, keep_default_na=False)
    d["delta"] = pd.to_numeric(d["delta"])
    d["match"] = d["match"].astype(str).isin(["True", "true", "1"])
    d["k"] = d["k"].astype(int)
    d["params"] = pd.to_numeric(d["params"])
    d["train_tokens"] = pd.to_numeric(d["train_tokens"])
    cell = ["noun", "draw", "k", "match", "stable_dim"]
    w = d.pivot_table(index=KEYS + cell, columns="probe_dim", values="delta").reset_index()
    w["OI_colour"] = w["shape"] - w["colour"]
    w["OI_texture"] = w["shape"] - w["texture"]
    w["step"] = w["revision"].str.extract(r"step(\d+)", expand=False).astype(float)
    return w


def noun_means(g, col):
    return g.groupby("noun")[col].mean().to_numpy()


def boot(nm, rng, n=N_BOOT):
    return nm[rng.integers(0, len(nm), (n, len(nm)))].mean(1)


def spearman(x, y):
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def holm(pvals):
    names = sorted(pvals, key=pvals.get)
    m, out, running = len(names), {}, 0.0
    for i, n in enumerate(names):
        running = max(running, min(1.0, (m - i) * pvals[n]))
        out[n] = running
    return out


def per_model(w, col, k=1, match=True, stable="none"):
    g = w[(w.k == k) & (w.match == match) & (w.stable_dim == stable)]
    return g.groupby(KEYS + ["step"], dropna=False)[col].mean().reset_index()


def h6(w, col):
    pm = per_model(w, col)
    f = pm[(pm.family == "pythia") & (pm.revision == "step143000")].sort_values("params")
    x, y = f["params"].to_numpy(), f[col].to_numpy()
    rho = spearman(x, y)
    ry = pd.Series(y).rank().to_numpy()
    rx = pd.Series(x).rank().to_numpy()
    null = [np.corrcoef(rx, np.array(p))[0, 1] for p in itertools.permutations(ry)]
    p = (np.sum(np.array(null) >= rho - 1e-12)) / len(null)
    return {"rho": rho, "p_one_sided": float(p), "n_sizes": int(len(f)),
            "values": dict(zip(f["model_id"], np.round(y, 4)))}


def h7(w, col, rng):
    pm = per_model(w, col)
    f = pm[pm.family == "pythia"]
    groups = [g.sort_values("step") for _, g in f.groupby("model_id") if len(g) >= 3]
    obs_each = {g["model_id"].iloc[0]: spearman(g["step"], g[col]) for g in groups}
    obs = float(np.mean(list(obs_each.values())))
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        null[i] = np.mean([spearman(g["step"], rng.permutation(g[col].to_numpy())) for g in groups])
    p = (np.sum(null >= obs - 1e-12) + 1) / (N_PERM + 1)
    return {"mean_rho": obs, "p_one_sided": float(p), "per_size_rho": obs_each}


def h8(w, col, model_id, rng):
    g = w[(w.model_id == model_id) & (w.k == 1) & (w.match) & (w.stable_dim == "none")]
    if g.empty:
        return {"missing": True}
    nm = noun_means(g, col)
    b = boot(nm, rng)
    lo95, hi95 = np.percentile(b, [2.5, 97.5])
    lo90, hi90 = np.percentile(b, [5, 95])
    return {"mean": float(nm.mean()), "ci95": [float(lo95), float(hi95)],
            "ci90": [float(lo90), float(hi90)],
            "p_one_sided": float((np.sum(b <= 0) + 1) / (N_BOOT + 1)),
            "equivalent_within_sesoi": bool(lo90 > -SESOI and hi90 < SESOI)}


def secondary(w, rng):
    rows = []
    for key, g in w.groupby(KEYS + ["step"], dropna=False):
        r = dict(zip(KEYS + ["step"], key))
        k1m = g[(g.k == 1) & g.match]
        k1x = g[(g.k == 1) & ~g.match]
        for col in ("OI_colour", "OI_texture"):
            nm = noun_means(k1m, col)
            b = boot(nm, rng, 2000)
            r[col] = nm.mean()
            r[col + "_lo"], r[col + "_hi"] = np.percentile(b, [2.5, 97.5])
        # S6 binding: matched minus mismatched, paired by noun
        diff = noun_means(k1m, "OI_colour") - noun_means(k1x, "OI_colour")
        b = boot(diff, rng, 2000)
        r["binding"] = diff.mean()
        r["binding_lo"], r["binding_hi"] = np.percentile(b, [2.5, 97.5])
        # S7 evidence: k = 2 and 4, matched
        for k in (2, 4):
            gk = g[(g.k == k) & g.match]
            s = noun_means(gk[gk.stable_dim == "shape"], "OI_colour")
            c = noun_means(gk[gk.stable_dim == "colour"], "OI_colour")
            prior = (s + c) / 2
            use = s - c
            for name, v in ((f"k{k}_prior", prior), (f"k{k}_evidence_use", use)):
                b = boot(v, rng, 2000)
                r[name] = v.mean()
                r[name + "_lo"], r[name + "_hi"] = np.percentile(b, [2.5, 97.5])
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["family", "params", "train_tokens"])


def onset(sec):
    out = {}
    f = sec[sec.family == "pythia"]
    for mid, g in f.groupby("model_id"):
        g = g.sort_values("step")
        above = (g.OI_colour_lo > 0).to_numpy()
        # Sustained onset: the first step from which every later checkpoint
        # also has a 95% CI above zero. Guards against one noisy checkpoint.
        idx = next((i for i in range(len(g)) if above[i:].all()), None)
        out[mid] = None if idx is None else {"step": int(g.step.iloc[idx]),
                                             "tokens": int(g.train_tokens.iloc[idx])}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--babylm_10m", default="BabyLM-community/babylm-baseline-10m-gpt2")
    ap.add_argument("--babylm_100m", default="BabyLM-community/babylm-baseline-100m-gpt2")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    w = load_cells(args.scores)
    res = {}
    for col, tag in (("OI_colour", "primary"), ("OI_texture", "S5_texture")):
        r = {"H6": h6(w, col), "H7": h7(w, col, rng),
             "H8a": h8(w, col, args.babylm_10m, rng),
             "H8b": h8(w, col, args.babylm_100m, rng)}
        ps = {k: v["p_one_sided"] for k, v in r.items() if "p_one_sided" in v}
        if tag == "primary":
            for k, p in holm(ps).items():
                r[k]["p_holm"] = p
                r[k]["supported"] = p < 0.05
        res[tag] = r

    sec = secondary(w, rng)
    sec.round(4).to_csv(args.out + "_secondary.csv", index=False)
    res["S8_onset"] = onset(sec)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(res["primary"], indent=2, default=str))
    print(f"\nWrote {args.out}.json and {args.out}_secondary.csv")


if __name__ == "__main__":
    main()
