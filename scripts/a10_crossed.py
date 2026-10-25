#!/usr/bin/env python3
"""
a10_crossed.py  |  Exploratory: whose history sets copying, the context noun's or the queried noun's?
Inference only, on the Amendment 10 Part A (Experiment 2) final models. Added after the results.

Item: context "A X is a s c t thing." about a noun X of one history, then the query "A Y is a" naming a
noun Y of another history (or the same history, a different noun). Copy score = L(s) - mean L over
the other shapes, leaving out the shapes stored for X and Y. For every ordered pair of histories
(the six familiar histories and unseen), at each exposure level (unseen nouns pair with both levels),
24 items per model, drawn with a fixed seed.

  python scripts/a10_crossed.py --manifest results/amendment10/manifest_a.csv --out results/amendment10/crossed.csv
  (resumable)
"""
import argparse
import csv
import os
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus_generator as cg  # noqa: E402
from a10_reanalysis import score  # noqa: E402
from familiarity_eval import check_forward  # noqa: E402
from oneshot_eval import load_model  # noqa: E402

SEED = 91_000
N_ITEMS = 24
FIELDS = ["base", "size", "seed", "ckpt", "exposures", "ctx_type", "query_type", "same_noun", "carry", "n_items"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    import make_familiarity_corpora_a10 as m10
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    runs = [r for r in csv.DictReader(open(a.manifest, encoding="utf-8")) if r["ckpt"].endswith("model_final.pt")]
    done = set()
    if os.path.exists(a.out):
        done = {r["ckpt"] for r in csv.DictReader(open(a.out, encoding="utf-8"))}
    for i, run in enumerate(runs, 1):
        if run["ckpt"] in done:
            continue
        seed = int(run["seed"])
        tok = Tokenizer.from_file(run["tokenizer"])
        model, _ = load_model(run["ckpt"], None, "train_model", "OverhypothesisLM")
        model.eval()
        check_forward(model, tok)
        if run.get("design", "a10").startswith("a11_"):
            import make_familiarity_corpora_a11 as m11
            rows = m11.load_assignment(seed, run["design"][4:])
        else:
            rows = m10.load_assignment(seed)
        types = sorted({r["type"] for r in rows if r["type"] != "unseen"})
        rng = random.Random(SEED + seed)
        items, keys = [], []
        for e in (100, 1000):
            cells = {t: [r for r in rows if r["type"] == t and r["exposures"] == e] for t in types}
            cells["unseen"] = [r for r in rows if r["type"] == "unseen"]
            for tx, X in cells.items():
                for ty, Y in cells.items():
                    for _ in range(N_ITEMS):
                        x = rng.choice(X)
                        y = rng.choice([r for r in Y if r["noun"] != x["noun"]] or Y)
                        stars = {x["s_star"], y["s_star"]}
                        s = rng.choice([z for z in cg.SHAPE_TOKENS if z not in stars])
                        text = (f"A {x['noun']} is a {s} {rng.choice(cg.COLOUR_TOKENS)} "
                                f"{rng.choice(cg.TEXTURE_TOKENS)} thing. A {y['noun']} is a")
                        items.append({"dim": "shape", "target": s, "exclude": list(stars), "text": text})
                        keys.append((e, tx, ty, x["noun"] == y["noun"]))
        res = score(model, tok, items)
        acc = {}
        for k, r in zip(keys, res):
            acc.setdefault(k, []).append(r[0])
        out = [{"base": run["base"], "size": run["size"], "seed": seed, "ckpt": run["ckpt"], "exposures": e,
                "ctx_type": tx, "query_type": ty, "same_noun": same, "carry": sum(v) / len(v), "n_items": len(v)}
               for (e, tx, ty, same), v in acc.items()]
        new = not os.path.exists(a.out)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(out)
        print(f"[{i}/{len(runs)}] {run['base']} {run['size']} seed {seed}", flush=True)


if __name__ == "__main__":
    main()
