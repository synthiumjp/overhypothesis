#!/usr/bin/env python3
"""
study1_trained_nouns.py  |  Study 1 extension (Amendment 5, S14): the same
one-exemplar test with nouns the models were trained on.

Why. With the eight held-out test nouns, the kind-specific index was zero in
every Study 1 model. One explanation is that held-out nouns have untrained
embeddings, so a model cannot tell "same noun" from "different noun". Trained
nouns have trained embeddings, so this test asks whether the models tie the
repetition of shape to the noun when they can tell nouns apart.

Items. Context: one sentence in the canonical frame about a trained noun A,
"A wug is a <shape> <colour> <texture> thing." Continuation: the same frame
about A (same noun) or about another trained noun B (different noun). No
shape word that is the stable word of A or B in that corpus (its canonical
shape in Regular; its stable word in Scrambled) appears anywhere in the item,
so remembered noun-shape pairs cannot favour either option. 16 context nouns
x 10 draws x 3 probed dimensions x 2 continuation nouns = 960 items per file.
One file per corpus seed and corpus family (Regular-derived, Scrambled),
because the stable words differ by seed.

Usage (from the repo root on the training PC)
  python scripts/study1_trained_nouns.py make-items --out_dir data/wug_tests/trained_nouns
  python scripts/study1_trained_nouns.py eval --items_dir data/wug_tests/trained_nouns \\
      --ckpt_dir <checkpoint folder> --family regular --label <label> --out <results.csv>
  (family = regular for Regular, Regular grouped, the controls and the graded
   corpora; scrambled for the Scrambled models)
"""

import argparse
import csv
import importlib.util
import json
import os
import random

SEEDS = [42, 123, 456, 789, 1001]
SIZES = ["tiny", "small", "medium"]
N_CONTEXT_NOUNS, DRAWS = 16, 10
SEED = 2030


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


oe = _load("oe", os.path.join(os.path.dirname(__file__), "oneshot_eval.py"))


def make_items(args):
    os.makedirs(args.out_dir, exist_ok=True)
    for family in ("regular", "scrambled"):
        for seed in SEEDS:
            meta = json.load(open(f"data/corpora/{family}_seed{seed}_meta.json"))
            stable = meta["noun_stable_feature"]
            nouns = sorted(stable)
            rng = random.Random(SEED + seed + (0 if family == "regular" else 1))
            ctx_nouns = rng.sample(nouns, N_CONTEXT_NOUNS)
            rows, iid = [], 0
            for a in ctx_nouns:
                for draw in range(DRAWS):
                    b = rng.choice([n for n in nouns if n != a])
                    banned = {stable[a], stable[b]}
                    shapes = [s for s in oe.SHAPES if s not in banned]
                    ctx = {"shape": rng.choice(shapes), "colour": rng.choice(oe.COLOURS),
                           "texture": rng.choice(oe.TEXTURES)}
                    allowed = {"shape": shapes, "colour": oe.COLOURS, "texture": oe.TEXTURES}
                    context = oe.FRAME.format(noun=a, **ctx)
                    for probe in oe.DIM_ORDER:
                        rep = ctx[probe]
                        new = rng.choice([v for v in allowed[probe] if v != rep])
                        fill = {d: rng.choice([v for v in allowed[d] if v != ctx[d]])
                                for d in oe.DIM_ORDER if d != probe}
                        for qnoun, match in ((a, True), (b, False)):
                            def sent(val):
                                vals = dict(fill)
                                vals[probe] = val
                                return oe.FRAME.format(noun=qnoun, **vals)
                            rows.append({"item_id": iid, "noun": a, "query_noun": qnoun,
                                         "match": match, "k": 1, "draw": draw,
                                         "stable_dim": "shape", "probe_dim": probe,
                                         "ctx_shape": ctx["shape"], "ctx_colour": ctx["colour"],
                                         "ctx_texture": ctx["texture"], "rep_value": rep,
                                         "new_value": new, "context": context,
                                         "query_rep": sent(rep), "query_new": sent(new)})
                            iid += 1
            path = os.path.join(args.out_dir, f"trained_{family}_seed{seed}.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=oe.ITEM_FIELDS)
                w.writeheader()
                w.writerows(rows)
            print(f"{path}: {len(rows)} items")


def evaluate(args):
    """Scores every final checkpoint in ckpt_dir with the item file for its
    seed, using oneshot_eval.eval_lm unchanged."""
    tok = os.path.join("data", "tokenizers", f"{args.tokenizer}_tokenizer.json")
    for seed in SEEDS:
        man = os.path.join(args.tmp_dir, f"manifest_{args.label}_seed{seed}.csv")
        os.makedirs(args.tmp_dir, exist_ok=True)
        rows = []
        for size in SIZES:
            ck = os.path.join(args.ckpt_dir, f"{args.run_prefix}_{size}_seed{seed}", "model_final.pt")
            if os.path.exists(ck):
                rows.append({"ckpt": ck, "tokenizer": tok, "condition": args.label,
                             "size": size, "seed": seed})
            else:
                print("missing:", ck)
        with open(man, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["ckpt", "tokenizer", "condition", "size", "seed"])
            w.writeheader()
            w.writerows(rows)
        ns = argparse.Namespace(items=os.path.join(args.items_dir, f"trained_{args.family}_seed{seed}.csv"),
                                manifest=man, scripts_dir="scripts", model_module="train_model",
                                model_class="OverhypothesisLM", bos_token=None, sep=" ", ks="",
                                random_init=False, out=args.out)
        oe.eval_lm(ns)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("make-items")
    p.add_argument("--out_dir", default="data/wug_tests/trained_nouns")
    p = sp.add_parser("eval")
    p.add_argument("--items_dir", default="data/wug_tests/trained_nouns")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--family", required=True, choices=["regular", "scrambled"])
    p.add_argument("--run_prefix", default=None,
                   help="Run folder prefix; default = family (e.g. regular_small_seed42)")
    p.add_argument("--tokenizer", default=None, help="Default = family")
    p.add_argument("--label", required=True, help="Condition label written to the results")
    p.add_argument("--tmp_dir", default="results/study1x/manifests")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "eval":
        a.run_prefix = a.run_prefix or a.family
        a.tokenizer = a.tokenizer or a.family
        evaluate(a)
    else:
        make_items(a)


if __name__ == "__main__":
    main()
