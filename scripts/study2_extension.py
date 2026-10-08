#!/usr/bin/env python3
"""
study2_extension.py  |  Study 2 (registered "Study 1d") extension, Amendment 5.

Fixed before any model is scored on these items. Scoring itself uses the
unchanged scorer in english_oneshot.py (eval), so model loading and log
probabilities are computed exactly as in Study 2.

What changes from the original items
  The original kind-specific contrast compared "Here is another snerg." (same
  noun) with "Here is a zull." (different noun). It therefore confounded the
  noun with the word "another". Every set below uses the same wording for the
  same-noun and different-noun continuation.

Item sets (k = 1 only, 16 novel nouns, 12 draws, 3 probed dimensions, same
and different continuation noun; 1,152 items per set)
  M   matched article, frame A, word list 1   "Here is a snerg. It is oval. ..."
      continuation "Here is a snerg." / "Here is a brell."         [primary]
  M2  matched "another", frame A, word list 1
      continuation "Here is another snerg." / "Here is another brell."
  F   frame B, word list 1
      "I found a snerg. The snerg is oval. The snerg is amber. ..."
  W   matched article, frame A, word list 2 (new, frequency matched)

Subcommands
  make-items      write the item file (deterministic, seed 2029)
  subset          write only some sets (same item ids), for the slowest models
  word-stats      Zipf frequency of both word lists (needs wordfreq)
  word-tokens     number of tokens per feature word, per tokenizer (needs transformers)
  make-manifests  model lists: final checkpoints, dense Pythia checkpoints, PolyPythia seeds
  analyse         registered tests H14a, H14b, H15 and secondary analyses S15 to S18

Usage (from the repo root)
  python scripts/study2_extension.py make-items --out data/wug_tests/english_items_v2.csv
  python scripts/study2_extension.py make-manifests --out_dir results/study2x
  python scripts/english_oneshot.py eval --items data/wug_tests/english_items_v2.csv \\
         --manifest results/study2x/models_final.csv --out results/study2x/scores_final.csv
  (likewise models_dense.csv and models_poly.csv; add --backend mlx for the 70B models)
  python scripts/study2_extension.py analyse --out results/study2x/tests
"""

import argparse
import csv
import itertools
import json
import random

WORDS = {
    1: {"shape": ["pointed", "oval", "circular", "curved",
                  "rectangular", "triangular", "spherical", "cylindrical"],
        "colour": ["grey", "purple", "amber", "violet",
                   "scarlet", "crimson", "maroon", "turquoise"],
        "texture": ["smooth", "shiny", "sticky", "fuzzy",
                    "fluffy", "furry", "glossy", "silky"]},
    2: {"shape": ["oblong", "hexagonal", "conical", "tapered",
                  "elliptical", "tubular", "cubic", "rounded"],
        "colour": ["magenta", "lilac", "teal", "beige",
                   "indigo", "lavender", "ivory", "coral"],
        "texture": ["velvety", "waxy", "prickly", "bumpy",
                    "gritty", "coarse", "slippery", "rough"]},
}
DIM_ORDER = ["shape", "colour", "texture"]
ORDERS = list(itertools.permutations(DIM_ORDER))
NOUNS = ["snerg", "twick", "vulf", "wazz", "glorp", "yeff", "zull", "kleb",
         "brell", "sprock", "fendle", "plim", "skeb", "tulver", "grisp", "vosh"]
SETS = {"M": ("A", 1, "a"), "M2": ("A", 1, "another"),
        "F": ("B", 1, None), "W": ("A", 2, "a")}
DRAWS = 12
SEED = 2029

FIELDS = ["item_id", "set", "frame", "wordlist", "noun", "query_noun", "match",
          "k", "draw", "stable_dim", "probe_dim", "order", "rep_value",
          "new_value", "context", "query_rep", "query_new"]


def passage(frame, noun, vals, order, article):
    if frame == "A":
        return " ".join([f"Here is {article} {noun}."] +
                        [f"It is {vals[d]}." for d in order])
    return " ".join([f"I found a {noun}."] +
                    [f"The {noun} is {vals[d]}." for d in order])


def make_items(args):
    rng = random.Random(SEED)
    rows, iid = [], 0
    for set_name, (frame, wl, art) in SETS.items():
        dims = WORDS[wl]
        for noun in NOUNS:
            others = [n for n in NOUNS if n != noun]
            for draw in range(DRAWS):
                order = ORDERS[rng.randrange(len(ORDERS))]
                ctx = {d: rng.choice(dims[d]) for d in DIM_ORDER}
                context = passage(frame, noun, ctx, order, "a")
                other = rng.choice(others)
                for probe in DIM_ORDER:
                    rep = ctx[probe]
                    new = rng.choice([v for v in dims[probe] if v != rep])
                    fill = {d: rng.choice([v for v in dims[d] if v != ctx[d]])
                            for d in DIM_ORDER if d != probe}
                    for qnoun, match in ((noun, True), (other, False)):
                        def q(val):
                            vals = dict(fill)
                            vals[probe] = val
                            return passage(frame, qnoun, vals, order, art)
                        rows.append({"item_id": iid, "set": set_name, "frame": frame,
                                     "wordlist": wl, "noun": noun, "query_noun": qnoun,
                                     "match": match, "k": 1, "draw": draw,
                                     "stable_dim": "none", "probe_dim": probe,
                                     "order": "-".join(order), "rep_value": rep,
                                     "new_value": new, "context": context,
                                     "query_rep": q(rep), "query_new": q(new)})
                        iid += 1
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} items to {args.out}")


def subset(args):
    """Write only some item sets, keeping item ids (for the slowest models)."""
    with open(args.items, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["set"] in args.sets]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} items ({', '.join(args.sets)}) to {args.out}")


def word_stats(args):
    from wordfreq import zipf_frequency as z
    for wl, dims in WORDS.items():
        for d, ws in dims.items():
            print(f"list {wl} {d:8s} mean Zipf {sum(z(w, 'en') for w in ws) / len(ws):.2f}")
    print("novel nouns (Zipf, 0 = unattested):",
          {n: round(z(n, "en"), 2) for n in NOUNS})


def word_tokens(args):
    from transformers import AutoTokenizer
    out = []
    for name in args.tokenizers:
        tok = AutoTokenizer.from_pretrained(name)
        for wl, dims in WORDS.items():
            for d, ws in dims.items():
                for w in ws:
                    out.append({"tokenizer": name, "wordlist": wl, "dim": d, "word": w,
                                "n_tokens": len(tok(" " + w, add_special_tokens=False)["input_ids"])})
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    print(f"Wrote {len(out)} rows to {args.out}")


PYTHIA = [("14m", 14_000_000), ("70m", 70_000_000), ("160m", 160_000_000),
          ("410m", 410_000_000), ("1b", 1_000_000_000), ("1.4b", 1_400_000_000),
          ("2.8b", 2_800_000_000), ("6.9b", 6_900_000_000), ("12b", 12_000_000_000)]
TOK_PER_STEP = 2_097_152


def _row(family, mid, rev, params, tokens):
    big = params > 1_000_000_000
    return {"family": family, "model_id": mid, "revision": rev, "params": params,
            "train_tokens": tokens, "dtype": "float16" if big else "float32",
            "batch_size": 8 if big else 64}


def make_manifests(args):
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    final = [_row("pythia", f"EleutherAI/pythia-{n}", "step143000", p, 143000 * TOK_PER_STEP)
             for n, p in PYTHIA]
    final += [_row("babylm", "BabyLM-community/babylm-baseline-10m-gpt2", "", 124_000_000, 100_000_000),
              _row("babylm", "BabyLM-community/babylm-baseline-100m-gpt2", "", 124_000_000, 1_000_000_000),
              _row("large", "Qwen/Qwen2.5-72B", "", 72_000_000_000, 18_000_000_000_000),
              _row("large", "meta-llama/Llama-3.1-70B", "", 70_000_000_000, 15_000_000_000_000)]
    # Dense checkpoints around the onset seen in Study 2 (between steps 4,000 and 16,000).
    dense_steps = [2000, 3000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000,
                   13000, 14000, 15000, 18000, 20000, 24000, 32000]
    dense = [_row("pythia", f"EleutherAI/pythia-{n}", f"step{s}", p, s * TOK_PER_STEP)
             for n, p in PYTHIA if n in ("410m", "1b", "1.4b", "2.8b") for s in dense_steps]
    dense += [_row("pythia", f"EleutherAI/pythia-{n}", f"step{s}", p, s * TOK_PER_STEP)
              for n, p in PYTHIA if n in ("6.9b", "12b") for s in (4000, 8000, 12000, 32000)]
    # PolyPythias: nine further seeds of the five smallest sizes (van der Wal et al.).
    poly_sizes = [("14m", 14_000_000), ("31m", 31_000_000), ("70m", 70_000_000),
                  ("160m", 160_000_000), ("410m", 410_000_000)]
    poly = [_row("polypythia", f"EleutherAI/pythia-{n}-seed{s}", rev, p,
                 int(rev[4:]) * TOK_PER_STEP)
            for n, p in poly_sizes for s in range(1, 10)
            for rev in ("step143000", "step4000", "step16000", "step64000")]
    poly += [_row("pythia", "EleutherAI/pythia-31m", rev, 31_000_000, int(rev[4:]) * TOK_PER_STEP)
             for rev in ("step143000", "step4000", "step16000", "step64000")]
    small = [r for r in final if r["family"] != "large"]
    large = [r for r in final if r["family"] == "large"]
    for name, rows in (("models_final", final), ("models_final_small", small),
                       ("models_final_large", large), ("models_dense", dense), ("models_poly", poly)):
        path = f"{args.out_dir}/{name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{path}: {len(rows)} models")


# --------------------------------------------------------------------------
# analyse
# --------------------------------------------------------------------------

def _load(paths, items_path):
    import pandas as pd
    it = pd.read_csv(items_path)
    frames = []
    for p in paths:
        d = pd.read_csv(p, dtype={"revision": str}, keep_default_na=False)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["match"] = d["match"].astype(str).isin(["True", "true"])
    d = d.merge(it[["item_id", "set", "rep_value", "new_value", "draw"]].rename(
        columns={"draw": "draw_it"}), on="item_id")
    d["model"] = d.model_id.str.split("/").str[-1] + "@" + d.revision.astype(str)
    return d


def kind_specific(d):
    """Per model, set, noun and draw: (shape same - shape diff) - (colour same - colour diff)."""
    w = d.pivot_table(index=["model", "family", "params", "train_tokens", "set", "noun", "draw"],
                      columns=["probe_dim", "match"], values="delta")
    out = w.index.to_frame(index=False)
    for dim in DIM_ORDER:
        out[f"{dim}_same"] = w[(dim, True)].to_numpy()
        out[f"{dim}_diff"] = w[(dim, False)].to_numpy()
        out[f"{dim}_kind"] = out[f"{dim}_same"] - out[f"{dim}_diff"]
    out["ks_colour"] = out["shape_kind"] - out["colour_kind"]
    out["ks_texture"] = out["shape_kind"] - out["texture_kind"]
    out["raw_colour"] = out["shape_same"] - out["colour_same"]
    return out


def boot(x_by_noun, rng, n=10_000):
    import numpy as np
    x = np.asarray(x_by_noun)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def crossed_model(sub):
    """Item-level ks_colour with crossed random intercepts for noun, item
    (noun x draw, shared across models), repeated shape word and repeated colour word (variance components, REML, Powell
    optimiser). Returns the fixed intercept, its SE, one-sided p (H: > 0) and
    whether the fit converged."""
    import warnings
    import statsmodels.formula.api as smf
    from scipy import stats
    sub = sub.copy()
    sub["g"] = 1
    sub["item"] = sub["noun"] + "_" + sub["draw"].astype(str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm("ks_colour ~ 1", sub, groups="g",
                        vc_formula={"noun": "0 + C(noun)", "item": "0 + C(item)",
                                    "shape_word": "0 + C(shape_word)",
                                    "colour_word": "0 + C(colour_word)"}).fit(reml=True, method="powell")
    est, se = float(m.fe_params.iloc[0]), float(m.bse.iloc[0])
    return est, se, float(stats.norm.sf(est / se)), bool(m.converged)


def pigeonhole_boot(sub, rng, n=10_000):
    """Crossed (pigeonhole) bootstrap: resample nouns, shape words and colour
    words independently; each row is weighted by the product of its three
    multiplicities. Returns mean, 95% CI and one-sided p (share of resampled
    means <= 0)."""
    import numpy as np
    keys = {}
    for col in ("noun", "shape_word", "colour_word"):
        levels = np.unique(sub[col].to_numpy())
        keys[col] = (levels, np.searchsorted(levels, sub[col].to_numpy()))
    y = sub["ks_colour"].to_numpy()
    means = np.empty(n)
    for i in range(n):
        w = np.ones(len(y))
        for levels, idx in keys.values():
            counts = rng.multinomial(len(levels), np.full(len(levels), 1 / len(levels)))
            w *= counts[idx]
        means[i] = (w * y).sum() / w.sum() if w.sum() > 0 else np.nan
    means = means[~np.isnan(means)]
    return float(y.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), \
        float((means <= 0).mean())


def analyse(args):
    import numpy as np
    import pandas as pd
    from scipy import stats
    rng = np.random.default_rng(2029)
    d = _load(args.scores, args.items)
    # Feature words of each item, for the crossed model.
    it = pd.read_csv(args.items)
    words = it[it.match.astype(str).isin(["True", "true"])].pivot_table(
        index=["set", "noun", "draw"], columns="probe_dim", values="rep_value", aggfunc="first")
    ks = kind_specific(d).merge(words.rename(columns={"shape": "shape_word", "colour": "colour_word",
                                                      "texture": "texture_word"}).reset_index(),
                                on=["set", "noun", "draw"])
    ks.to_csv(f"{args.out}_kind_specific_items.csv", index=False)
    rows = []
    for (model, fam, params, tokens, s), g in ks.groupby(["model", "family", "params", "train_tokens", "set"]):
        r = {"model": model, "family": fam, "params": params, "train_tokens": tokens, "set": s}
        for c in ("ks_colour", "ks_texture", "raw_colour", "shape_kind", "colour_kind", "texture_kind",
                  "shape_same", "shape_diff", "colour_same", "colour_diff"):
            m, lo, hi = boot(g.groupby("noun")[c].mean(), rng)
            r[c], r[c + "_lo"], r[c + "_hi"] = m, lo, hi
        rows.append(r)
    summ = pd.DataFrame(rows)
    summ.round(4).to_csv(f"{args.out}_summary.csv", index=False)

    res = {}
    nine = [p for _, p in PYTHIA]
    fin = ks[(ks.set == "M") & (ks.model.str.endswith("@step143000")) & (ks.family == "pythia")
             & ks.params.isin(nine)]
    # H14a: kind-specific index (set M) rises with Pythia size, final checkpoints.
    per_model = fin.groupby("params").ks_colour.mean()
    rho = stats.spearmanr(per_model.index, per_model.values)[0]
    perm = [stats.spearmanr(per_model.index, rng.permutation(per_model.values))[0] for _ in range(10_000)]
    res["H14a"] = {"rho": rho, "p_one_sided": (1 + sum(p >= rho for p in perm)) / 10_001}
    # H14b: pooled Pythia 410M to 12B, set M, crossed random effects, > 0.
    big = fin[fin.params >= 400_000_000]
    est, se, p, conv = crossed_model(big)
    m, lo, hi, pb = pigeonhole_boot(big, rng)
    # Registered rule: the mixed model decides if it converged; otherwise the
    # pigeonhole bootstrap decides. Both are always reported.
    res["H14b"] = {"estimate": est, "se": se, "p_mixed": p, "converged": conv,
                   "boot_mean": m, "boot_ci95": [lo, hi], "p_boot": pb,
                   "p_one_sided": p if conv else pb}
    # H15: PolyPythia seeds at step 143,000: 410M > 160M on the kind-specific index (set M).
    poly = ks[(ks.set == "M") & ks.model.str.endswith("@step143000") &
              ks.family.isin(["polypythia", "pythia"])]
    seed_means = poly.groupby(["params", "model"]).ks_colour.mean().reset_index()
    a = seed_means[seed_means.params == 410_000_000].ks_colour
    b = seed_means[seed_means.params == 160_000_000].ks_colour
    t = stats.ttest_ind(a, b, equal_var=False, alternative="greater")
    res["H15"] = {"mean_410m": float(a.mean()), "mean_160m": float(b.mean()), "n_410m": len(a),
                  "n_160m": len(b), "t": float(t.statistic), "p_one_sided": float(t.pvalue),
                  "seeds_410m_positive": int((a > 0).sum())}
    ps = {k: v["p_one_sided"] for k, v in res.items()}
    order = sorted(ps, key=ps.get)
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[k]))
        res[k]["p_holm"] = run
        res[k]["supported"] = bool(run < 0.05)
    json.dump(res, open(f"{args.out}.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("make-items"); p.add_argument("--out", required=True)
    sp.add_parser("word-stats")
    p = sp.add_parser("subset"); p.add_argument("--items", required=True)
    p.add_argument("--sets", nargs="+", required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("word-tokens"); p.add_argument("--out", required=True)
    p.add_argument("--tokenizers", nargs="+", default=[
        "EleutherAI/pythia-160m", "BabyLM-community/babylm-baseline-10m-gpt2",
        "Qwen/Qwen2.5-72B", "meta-llama/Llama-3.1-70B"])
    p = sp.add_parser("make-manifests"); p.add_argument("--out_dir", required=True)
    p = sp.add_parser("analyse")
    p.add_argument("--scores", nargs="+", required=True)
    p.add_argument("--items", default="data/wug_tests/english_items_v2.csv")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"make-items": make_items, "subset": subset, "word-stats": word_stats, "word-tokens": word_tokens,
     "make-manifests": make_manifests, "analyse": analyse}[a.cmd](a)


if __name__ == "__main__":
    main()
