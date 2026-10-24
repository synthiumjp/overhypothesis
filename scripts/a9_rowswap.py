#!/usr/bin/env python3
"""
a9_rowswap.py  |  Amendment 10 Part C: what about a noun's input row matters, in the Amendment 9
models (docs/AMENDMENT10.md). Inference only; no model is changed on disk.

Every noun-specific input to the carry items enters through the noun's input row, so carry is a
function of that row. Part C asks which property of the row matters. For each final model in
grouped Regular and Reassigned corpora, carry (same-noun items, Amendment 9 measure) is scored for
  unseen nouns:  own      their own (untrained) rows
                 randnorm a random direction scaled to the mean norm of the 32 trained kinds'
                          rows, with no shift towards their mean (H48)
                 ownnorm  their own direction scaled to that norm
                 mean     the mean row of the trained kinds, no offset
  familiar shape-absent nouns at 10,000 exposures:
                 own      their trained rows
                 shrunk   their own direction scaled to the unseen nouns' mean norm
Random directions use a fixed seed per model and noun.

  python scripts/a9_rowswap.py --manifest results/amendment9/manifest.csv --out results/amendment10/rowswap.csv
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus_generator as cg  # noqa: E402
from familiarity_eval import check_forward, measures, run_items, token_id  # noqa: E402
from familiarity_items import make_items  # noqa: E402
from oneshot_eval import load_model  # noqa: E402

SEED = 73_000
FIELDS = ["base", "size", "seed", "ckpt", "noun", "type", "exposures", "condition", "carry", "row_norm"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    runs = [r for r in csv.DictReader(open(a.manifest, encoding="utf-8"))
            if r["base"] in ("regular_grouped", "reassigned_grouped") and r["ckpt"].endswith("model_final.pt")]
    done = set()
    if os.path.exists(a.out):
        done = {r["ckpt"] for r in csv.DictReader(open(a.out, encoding="utf-8"))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for i, run in enumerate(runs, 1):
        if run["ckpt"] in done:
            continue
        tok = Tokenizer.from_file(run["tokenizer"])
        model, _ = load_model(run["ckpt"], None, "train_model", "OverhypothesisLM")
        model.eval()
        check_forward(model, tok)
        items = [it for it in make_items(int(run["seed"]), int(run["per_cell"]))
                 if it["kind"] == "carry_same" and (it["type"] == "unseen" or
                                                    (it["type"] == "absent" and it["exposures"] == 10000))]
        E = model.tok_embed.weight
        orig = E.detach().clone()
        train_ids = [token_id(tok, n) for n in cg.TRAIN_NOUNS]
        kind_norm = orig[train_ids].norm(dim=1).mean()
        mu = orig[train_ids].mean(0)
        nouns = sorted({it["noun"] for it in items})
        ids = {n: token_id(tok, n) for n in nouns}
        unseen = [n for n in nouns if any(it["noun"] == n and it["type"] == "unseen" for it in items)]
        unseen_norm = torch.stack([orig[ids[n]] for n in unseen]).norm(dim=1).mean()
        g = torch.Generator().manual_seed(SEED + int(run["seed"]))
        rand = {n: torch.randn(E.shape[1], generator=g) for n in nouns}
        conds = {"unseen": ["own", "randnorm", "ownnorm", "mean"], "absent": ["own", "shrunk"]}
        out = []
        for typ, cl in conds.items():
            its = [it for it in items if it["type"] == typ]
            for cond in cl:
                with torch.no_grad():
                    E.copy_(orig)
                    for n in {it["noun"] for it in its}:
                        v = orig[ids[n]]
                        if cond == "randnorm":
                            E[ids[n]] = rand[n] / rand[n].norm() * kind_norm
                        elif cond == "ownnorm":
                            E[ids[n]] = v / v.norm() * kind_norm
                        elif cond == "mean":
                            E[ids[n]] = mu
                        elif cond == "shrunk":
                            E[ids[n]] = v / v.norm() * unseen_norm
                L = run_items(model, tok, its)
                per = {}
                for it, m in zip(its, measures(its, L)):
                    per.setdefault(it["noun"], []).append(m["carry"])
                for n, v in per.items():
                    out.append({"base": run["base"], "size": run["size"], "seed": run["seed"], "ckpt": run["ckpt"],
                                "noun": n, "type": typ, "exposures": 0 if typ == "unseen" else 10000,
                                "condition": cond, "carry": sum(v) / len(v), "row_norm": E[ids[n]].norm().item()})
        with torch.no_grad():
            E.copy_(orig)
        new = not os.path.exists(a.out)
        with open(a.out, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(out)
        print(f"[{i}/{len(runs)}] {run['base']} {run['size']} seed {run['seed']}", flush=True)


if __name__ == "__main__":
    main()
