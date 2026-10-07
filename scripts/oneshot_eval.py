#!/usr/bin/env python3
"""
oneshot_eval.py  |  Study 1 few-shot overhypothesis test

Replaces the zero-shot SO measure with the test Smith et al. (2002) and
Kemp et al. (2007) actually used: after k exemplars of a novel noun, does the
learner expect the SHAPE to repeat more than the COLOUR or TEXTURE?

For each item and each probe dimension d, two query sentences differ only in
dimension d. One repeats a value seen in context, the other uses an unseen
value. The other two dimensions take unseen values in both options.

    delta_d = log P(query_repeat) - log P(query_new)      (nats)

Copying or frame priming raises delta for every dimension alike.
An overhypothesis raises delta for the stable dimension selectively.

    OI_colour  = delta_shape - delta_colour
    OI_texture = delta_shape - delta_texture

k = 1 is the pure prior test (the context cannot show which dimension is
stable). k >= 2 lets in-context statistics contribute.

Control rows (match = False) query a DIFFERENT novel noun after the same
context. An overhypothesis bound to the noun should vanish there; the ideal
observer predicts delta = 0 for every dimension.

Subcommands
-----------
  make-items   generate the item file (deterministic, run once)
  eval-hbm     ideal-observer predictions on the same items, per corpus
  eval-lm      score items with trained checkpoints (listed in a manifest)
  summarise    aggregate result files into per-run and per-cell tables

Typical run (from C:\\overhypothesis, defaults match this repo)
-----------
  python oneshot_eval.py make-items --out items.csv
  python oneshot_eval.py eval-hbm --items items.csv --corpus data/corpora/regular_seed42.txt \
         --condition regular --seed 42 --out hbm_regular_42.csv
  python oneshot_eval.py make-manifest --out manifest.csv
  python oneshot_eval.py eval-lm --items items.csv --manifest manifest.csv \
         --scripts_dir scripts --out lm_results.csv
  python oneshot_eval.py summarise --inputs lm_results.csv "hbm_*.csv" --out summary

manifest.csv columns: ckpt,tokenizer,condition,size,seed

Dependencies: numpy, scipy, pandas. eval-lm also needs torch and tokenizers.
Inference is CPU only (DirectML is avoided for evaluation, as elsewhere).
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

# Vocabulary from scripts/corpus_generator.py (confirmed in Session 5).
SHAPES = ["zibby", "krannish", "tova", "felk", "mundi",
          "prex", "sallo", "wemb", "oofy", "drant"]
COLOURS = ["lavo", "renu", "vassa", "temil", "dorvi",
           "pulek", "sorba", "navri", "kelun", "morit"]
TEXTURES = ["shorfi", "glaven", "frimel", "tuskan", "merdo",
            "pliven", "druska", "halven", "skorli", "tramel"]
DIMS = {"shape": SHAPES, "colour": COLOURS, "texture": TEXTURES}
DIM_ORDER = ["shape", "colour", "texture"]
FEATURE_TO_DIM = {v: d for d, vals in DIMS.items() for v in vals}

# Labelled Frame 1, the canonical training frame.
FRAME = "A {noun} is a {shape} {colour} {texture} thing."

ITEM_FIELDS = ["item_id", "noun", "query_noun", "match", "k", "draw",
               "stable_dim", "probe_dim", "ctx_shape", "ctx_colour",
               "ctx_texture", "rep_value", "new_value", "context",
               "query_rep", "query_new"]


# --------------------------------------------------------------------------
# make-items
# --------------------------------------------------------------------------

def make_items(args):
    rng = random.Random(args.seed)
    nouns = [n.strip() for n in args.novel_nouns.split(",") if n.strip()]
    if not args.no_control and len(nouns) < 2:
        sys.exit("Need at least two novel nouns for the mismatch control.")
    ks = [int(x) for x in args.ks.split(",")]
    if max(ks) > 9:
        sys.exit("k must be <= 9 so an unseen value always exists.")

    rows, iid = [], 0
    for noun in nouns:
        others = [n for n in nouns if n != noun]
        for draw in range(args.draws):
            for k in ks:
                ctx = {}
                for d in DIM_ORDER:
                    if d == args.stable_dim:
                        ctx[d] = [rng.choice(DIMS[d])] * k
                    else:
                        ctx[d] = rng.sample(DIMS[d], k)  # distinct values
                context = args.sep.join(
                    FRAME.format(noun=noun, shape=ctx["shape"][i],
                                 colour=ctx["colour"][i],
                                 texture=ctx["texture"][i])
                    for i in range(k))

                queries = [(noun, True)]
                if not args.no_control:
                    queries.append((rng.choice(others), False))

                for probe in DIM_ORDER:
                    rep = ctx[probe][-1]
                    new = rng.choice([v for v in DIMS[probe] if v not in ctx[probe]])
                    fill = {d: rng.choice([v for v in DIMS[d] if v not in ctx[d]])
                            for d in DIM_ORDER if d != probe}
                    for qnoun, match in queries:
                        def sentence(val):
                            vals = dict(fill)
                            vals[probe] = val
                            return FRAME.format(noun=qnoun, **vals)
                        rows.append({
                            "item_id": iid, "noun": noun, "query_noun": qnoun,
                            "match": match, "k": k, "draw": draw,
                            "stable_dim": args.stable_dim, "probe_dim": probe,
                            "ctx_shape": "|".join(ctx["shape"]),
                            "ctx_colour": "|".join(ctx["colour"]),
                            "ctx_texture": "|".join(ctx["texture"]),
                            "rep_value": rep, "new_value": new,
                            "context": context,
                            "query_rep": sentence(rep),
                            "query_new": sentence(new),
                        })
                        iid += 1

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ITEM_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} item rows to {args.out} "
          f"({len(nouns)} nouns x {args.draws} draws x k={ks} x 3 probes"
          f"{'' if args.no_control else ' x 2 query nouns'}).")


def read_items(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["k"] = int(r["k"])
        r["match"] = r["match"] in ("True", "true", "1")
    return rows


# --------------------------------------------------------------------------
# eval-hbm  (Kemp et al., 2007 style Dirichlet-multinomial, per dimension)
# --------------------------------------------------------------------------

NOUN_PATTERNS = [re.compile(p) for p in (
    r"\ba (\w+) is a\b",            # Frame 1
    r"\blook at the (\w+)\b",        # Frame 2
    r"\bhere is another (\w+)\b",    # Frame 3
    r"\bthe (\w+) i showed you\b",   # Frame 4
    r"\bthat is a (\w+) because\b",  # Frame 5
)]
NOT_NOUNS = set(FEATURE_TO_DIM) | {"one", "different", "new", "thing"}


def parse_corpus(path):
    """Return {kind: {dim: Counter}} from labelled lines of a corpus file."""
    kinds = defaultdict(lambda: {d: Counter() for d in DIM_ORDER})
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    # corpus_generator.py replicates the base exemplar set with fresh shuffles
    # until the token target is met, so every sentence appears R times (R=26
    # for the main corpora). Counting raw lines would treat replicas as new
    # observations and inflate within-kind concentration. Divide them out.
    from functools import reduce
    line_counts = Counter(lines)
    R = reduce(math.gcd, line_counts.values())
    print(f"Replication factor R = {R}: {len(lines)} lines, "
          f"{sum(line_counts.values()) // R} base exemplar sentences.")
    n_lines, n_used = len(lines), 0
    for line, cnt in line_counts.items():
            w = cnt // R
            low = line.lower()
            noun = None
            for pat in NOUN_PATTERNS:
                m = pat.search(low)
                if m and m.group(1) not in NOT_NOUNS:
                    noun = m.group(1)
                    break
            if noun is None:
                continue
            feats = [t for t in re.findall(r"[a-z]+", low) if t in FEATURE_TO_DIM]
            if not feats:
                continue
            n_used += w
            for t in feats:
                kinds[noun][FEATURE_TO_DIM[t]][t] += w
    return dict(kinds), n_lines, n_used


def fit_alpha_posterior(kinds, grid, prior):
    """Posterior weights over the alpha grid, separately for each dimension."""
    import numpy as np
    from scipy.special import gammaln
    post = {}
    for d in DIM_ORDER:
        V = len(DIMS[d])
        idx = {v: i for i, v in enumerate(DIMS[d])}
        counts = np.zeros((len(kinds), V))
        for j, kc in enumerate(kinds.values()):
            for v, c in kc[d].items():
                counts[j, idx[v]] = c
        counts = counts[counts.sum(1) > 0]
        N = counts.sum(1)
        a = grid[:, None]
        lml = (gammaln(a) - gammaln(a + N[None, :])).sum(1)
        lml += (gammaln(counts[None, :, :] + a[:, :, None] / V)
                - gammaln(a[:, :, None] / V)).sum((1, 2))
        # Grid is uniform in log(alpha); add the log-density of each prior on log(alpha).
        if prior == "exponential":
            logprior = -grid + np.log(grid)     # Exp(1) on alpha, Jacobian alpha
        else:
            logprior = np.zeros_like(grid)      # flat on log(alpha)
        lp = lml + logprior
        w = np.exp(lp - lp.max())
        post[d] = w / w.sum()
    return post


def hbm_delta(row, dim, grid, weights):
    """log P(repeat) - log P(new) under the posterior predictive."""
    import numpy as np
    V = len(DIMS[dim])
    seen = row["ctx_" + dim].split("|") if row["match"] else []
    cnt = Counter(seen)
    N = len(seen)

    def pred(v):
        return float(np.sum(weights * (cnt[v] + grid / V) / (N + grid)))

    return math.log(pred(row["rep_value"])) - math.log(pred(row["new_value"]))


def eval_hbm(args):
    import numpy as np
    kinds, n_lines, n_used = parse_corpus(args.corpus)
    print(f"{args.corpus}: {n_lines} lines, {n_used} labelled lines used, "
          f"{len(kinds)} kinds found.")
    if len(kinds) < 8:
        print("WARNING: few kinds parsed. No-label corpora carry no nouns and "
              "cannot be fitted this way; check NOUN_PATTERNS for others.")
    grid = np.logspace(-3, 3, 241)
    post = fit_alpha_posterior(kinds, grid, args.alpha_prior)
    summary = {d: {"map_alpha": float(grid[np.argmax(post[d])]),
                   "post_mean_log10_alpha": float(np.sum(post[d] * np.log10(grid)))}
               for d in DIM_ORDER}
    print(json.dumps(summary, indent=2))

    rows = read_items(args.items)
    out = []
    for r in rows:
        out.append({"source": "hbm", "condition": args.condition, "size": "hbm",
                    "seed": args.seed, "ckpt": os.path.basename(args.corpus),
                    "item_id": r["item_id"], "k": r["k"], "match": r["match"],
                    "probe_dim": r["probe_dim"],
                    "delta": hbm_delta(r, r["probe_dim"], grid, post[r["probe_dim"]])})
    write_results(args.out, out)
    with open(os.path.splitext(args.out)[0] + "_alpha.json", "w") as f:
        json.dump({"corpus": args.corpus, "prior": args.alpha_prior,
                   "alpha": summary}, f, indent=2)


# --------------------------------------------------------------------------
# eval-lm
# --------------------------------------------------------------------------

def load_model(ckpt_path, scripts_dir, module_name, class_name, random_init=None):
    """Build OverhypothesisLM (scripts/train_model.py) from model_final.pt.

    model_final.pt stores model_state, vocab_size, model_cfg
    (n_layer, n_head, d_model, d_ff) and train_cfg (seq_len, ...).
    Intermediate ckpt_stepNNNNN.pt files lack model_cfg, so for those the
    config is read from the sibling model_config.json.
    """
    import importlib
    import torch
    if scripts_dir and scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    if random_init is not None:
        # Untrained baseline: same architecture, fresh weights, no checkpoint.
        side = os.path.join(os.path.dirname(ckpt_path), "model_config.json")
        with open(side, encoding="utf-8") as f:
            meta = json.load(f)
        cls = getattr(importlib.import_module(module_name), class_name)
        torch.manual_seed(int(random_init))
        model = cls(vocab_size=meta["vocab_size"], seq_len=meta["train_cfg"]["seq_len"],
                    **meta["model_cfg"])
        model.eval()
        return model, meta["train_cfg"]["seq_len"]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_cfg" in ckpt:
        vocab, mcfg, tcfg = ckpt["vocab_size"], ckpt["model_cfg"], ckpt["train_cfg"]
    else:
        side = os.path.join(os.path.dirname(ckpt_path), "model_config.json")
        with open(side, encoding="utf-8") as f:
            meta = json.load(f)
        vocab, mcfg, tcfg = meta["vocab_size"], meta["model_cfg"], meta["train_cfg"]
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls(vocab_size=vocab, seq_len=tcfg["seq_len"], **mcfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tcfg["seq_len"]


def make_logprob_fn(model):
    import torch

    def logprob(ids):
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        logits = getattr(logits, "logits", logits)
        lp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        return lp.gather(1, x[0, 1:].unsqueeze(1)).sum().item()

    return logprob


def score_items(rows, encode, logprob, sep, bos_id=None, block=None):
    """Return delta per row. The shared context contributes equally to both
    options, so the difference of full-sequence log-probs is the
    difference of the query continuations."""
    cache, deltas = {}, []
    for r in rows:
        vals = []
        for q in (r["query_rep"], r["query_new"]):
            text = r["context"] + sep + q
            if text not in cache:
                ids = encode(text)
                if bos_id is not None:
                    ids = [bos_id] + ids
                if block and len(ids) > block:
                    raise ValueError(f"Item {r['item_id']}: {len(ids)} tokens "
                                     f"exceeds block size {block}.")
                cache[text] = logprob(ids)
            vals.append(cache[text])
        deltas.append(vals[0] - vals[1])
    return deltas


def eval_lm(args):
    from tokenizers import Tokenizer
    rows = read_items(args.items)
    with open(args.manifest, encoding="utf-8") as f:
        runs = list(csv.DictReader(f))
    out = []
    for i, run in enumerate(runs, 1):
        tok = Tokenizer.from_file(run["tokenizer"])
        model, block = load_model(run["ckpt"], args.scripts_dir,
                                  args.model_module, args.model_class,
                                  run["seed"] if args.random_init else None)
        bos = tok.token_to_id(args.bos_token) if args.bos_token else None
        deltas = score_items(rows, lambda s: tok.encode(s).ids,
                             make_logprob_fn(model), args.sep, bos, block)
        for r, d in zip(rows, deltas):
            out.append({"source": "random" if args.random_init else "lm",
                        "condition": run["condition"],
                        "size": run["size"], "seed": run["seed"],
                        "ckpt": run["ckpt"], "item_id": r["item_id"],
                        "k": r["k"], "match": r["match"],
                        "probe_dim": r["probe_dim"], "delta": d})
        print(f"[{i}/{len(runs)}] {run['condition']} {run['size']} "
              f"seed {run['seed']}: {len(rows)} items scored")
    write_results(args.out, out)


# --------------------------------------------------------------------------
# make-manifest
# --------------------------------------------------------------------------

CONDITIONS = ["regular", "scrambled", "feature_swap", "weak_label",
              "paraphrased_nolabel", "bare_nolabel", "noise_injection",
              "frequency_matched"]
SIZES = ["tiny", "small", "medium"]
SEEDS = ["42", "123", "456", "789", "1001"]


def make_manifest(args):
    """Scan the checkpoint folder for the 120 primary runs."""
    rows, missing = [], []
    for c in (args.conditions or CONDITIONS):
        tok = os.path.join(args.base_dir, "data", "tokenizers", f"{c}_tokenizer.json")
        for z in SIZES:
            for sd in SEEDS:
                ck = os.path.join(args.ckpt_dir, f"{c}_{z}_seed{sd}", args.ckpt_name)
                if os.path.exists(ck) and os.path.exists(tok):
                    rows.append({"ckpt": ck, "tokenizer": tok,
                                 "condition": c + args.label_suffix,
                                 "size": z, "seed": sd})
                else:
                    missing.append(ck)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ckpt", "tokenizer", "condition", "size", "seed"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} runs to {args.out}; {len(missing)} missing.")
    for m in missing[:10]:
        print("  missing:", m)


# --------------------------------------------------------------------------
# summarise
# --------------------------------------------------------------------------

RESULT_FIELDS = ["source", "condition", "size", "seed", "ckpt", "item_id",
                 "k", "match", "probe_dim", "delta"]


def write_results(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def summarise(args):
    import pandas as pd
    files = [p for pat in args.inputs for p in glob.glob(pat)]
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    keys = ["source", "condition", "size", "seed", "k", "match"]

    # Average within each run and cell, then contrast probe dimensions.
    run = (df.groupby(keys + ["probe_dim"])["delta"].mean()
             .unstack("probe_dim").reset_index())
    run["OI_colour"] = run["shape"] - run["colour"]
    run["OI_texture"] = run["shape"] - run["texture"]
    run.to_csv(args.out + "_per_run.csv", index=False)

    cell_keys = ["source", "condition", "size", "k", "match"]
    agg = run.groupby(cell_keys).agg(
        n_runs=("seed", "nunique"),
        d_shape=("shape", "mean"), d_colour=("colour", "mean"),
        d_texture=("texture", "mean"),
        OI_colour=("OI_colour", "mean"), OI_colour_sd=("OI_colour", "std"),
        OI_texture=("OI_texture", "mean"), OI_texture_sd=("OI_texture", "std"),
    ).reset_index()
    agg.to_csv(args.out + "_per_cell.csv", index=False)
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(agg.round(3).to_string(index=False))


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make-items")
    m.add_argument("--novel_nouns",
                   default="snerg,twick,vulf,wazz,xorn,yeff,zull,bruck",
                   help="Held-out TEST_NOUNS from corpus_generator.py")
    m.add_argument("--ks", default="1,2,4")
    m.add_argument("--draws", type=int, default=25)
    m.add_argument("--stable_dim", default="shape", choices=DIM_ORDER,
                   help="Dimension held fixed across context exemplars (k>=2)")
    m.add_argument("--sep", default=" ",
                   help="Separator between sentences; match training data format")
    m.add_argument("--no_control", action="store_true")
    m.add_argument("--seed", type=int, default=2026)
    m.add_argument("--out", required=True)

    h = sub.add_parser("eval-hbm")
    h.add_argument("--items", required=True)
    h.add_argument("--corpus", required=True)
    h.add_argument("--condition", required=True)
    h.add_argument("--seed", required=True)
    h.add_argument("--alpha_prior", default="exponential",
                   choices=["exponential", "loguniform"])
    h.add_argument("--out", required=True)

    e = sub.add_parser("eval-lm")
    e.add_argument("--items", required=True)
    e.add_argument("--manifest", required=True,
                   help="CSV with columns ckpt,tokenizer,condition,size,seed")
    e.add_argument("--scripts_dir", default="scripts")
    e.add_argument("--model_module", default="train_model")
    e.add_argument("--model_class", default="OverhypothesisLM")
    e.add_argument("--bos_token", default=None)
    e.add_argument("--sep", default=" ")
    e.add_argument("--random_init", action="store_true",
                   help="Score untrained models of the same architecture (baseline)")
    e.add_argument("--out", required=True)

    mf = sub.add_parser("make-manifest")
    mf.add_argument("--base_dir", default=r"C:\overhypothesis")
    mf.add_argument("--ckpt_dir", default=r"C:\overhypothesis\models\checkpoints")
    mf.add_argument("--ckpt_name", default="model_final.pt")
    mf.add_argument("--out", default="manifest.csv")
    mf.add_argument("--conditions", nargs="+", default=None,
                    help="Subset of conditions (default: all eight)")
    mf.add_argument("--label_suffix", default="",
                    help="Appended to condition labels, e.g. _bursty for Study 1c")

    s = sub.add_parser("summarise")
    s.add_argument("--inputs", nargs="+", required=True)
    s.add_argument("--out", required=True, help="Output prefix")

    args = p.parse_args()
    {"make-items": make_items, "eval-hbm": eval_hbm, "eval-lm": eval_lm,
     "make-manifest": make_manifest, "summarise": summarise}[args.cmd](args)


if __name__ == "__main__":
    main()
