#!/usr/bin/env python3
"""
loss_check.py  |  Does the corrected loader return the trained model?

For each checkpoint in a manifest (oneshot_eval.py format), load the model twice
(corrected loading, and --legacy_tied_load as before 12 Oct 2026), and compute the
mean next-token loss on its own training corpus over a fixed set of windows. The
corpus, tokenizer and window length are read from the checkpoint, exactly as
train_model.py used them. The last training losses in train_log.json are reported
alongside. If the loader is right, the corrected loss should match the end of
training (a little lower, since dropout is off) and the legacy loss should be worse.

Usage:  python scripts/loss_check.py --manifest results/study1b/raw/manifest.csv \
            --n_models 6 --out results/rescore/loss_check.csv
"""
import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oneshot_eval as oe  # noqa: E402
import train_model as tm  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", nargs="+", required=True)
ap.add_argument("--n_models", type=int, default=0, help="0 = all; otherwise a fixed random subset")
ap.add_argument("--n_windows", type=int, default=200)
ap.add_argument("--out", required=True)
a = ap.parse_args()

runs = []
for m in a.manifest:
    with open(m, encoding="utf-8") as f:
        runs += list(csv.DictReader(f))
if a.n_models:
    runs = random.Random(2026).sample(runs, min(a.n_models, len(runs)))

rows = []
for r in runs:
    ck = torch.load(r["ckpt"], map_location="cpu", weights_only=False)
    args = ck["args"]
    corpus = Path(args["corpus_dir"]) / f"{args['condition']}_seed{args['seed']}.txt"
    tok_path = Path(args.get("tok_dir", "data/tokenizers")) / f"{args['condition']}_tokenizer.json"
    if not tok_path.exists():
        tok_path = Path(args.get("tok_dir", "data/tokenizers")) / "regular_tokenizer.json"
    ids = tm.load_and_tokenize(corpus, Tokenizer.from_file(str(tok_path)), args.get("corpus_fraction", 1.0))
    losses = {}
    for legacy in (False, True):
        model, T = oe.load_model(r["ckpt"], "scripts", "train_model", "OverhypothesisLM", legacy_tied_load=legacy)
        rng = random.Random(7)
        starts = [rng.randrange(0, len(ids) - T - 1) for _ in range(a.n_windows)]
        tot, n = 0.0, 0
        with torch.no_grad():
            for s in starts:
                x = torch.tensor([ids[s:s + T]])
                y = torch.tensor([ids[s + 1:s + T + 1]])
                out = model(x)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                tot += torch.nn.functional.cross_entropy(logits[0], y[0], reduction="sum").item()
                n += T
        losses["legacy" if legacy else "corrected"] = tot / n
    log = Path(r["ckpt"]).parent / "train_log.json"
    last = None
    if log.exists():
        L = json.load(open(log))
        if L.get("losses"):
            last = sum(L["losses"][-5:]) / len(L["losses"][-5:])
    rows.append({"ckpt": r["ckpt"], "condition": r["condition"], "size": r["size"], "seed": r["seed"],
                 "train_loss_last5": last, "loss_corrected": round(losses["corrected"], 4),
                 "loss_legacy": round(losses["legacy"], 4)})
    print(rows[-1], flush=True)

with open(a.out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
