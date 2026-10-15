#!/usr/bin/env python3
"""
heldout_loss.py  |  Amendment 8, S35: how far did each model memorise its training text?

For each checkpoint in a manifest (oneshot_eval.py format), compute the mean next-token
loss (nats per token) on
  own      windows from the model's own training file, and
  heldout  windows from a held-out file with the same exemplars (or objects) in fresh
           arrangements, written by make_fresh_corpora.py and never used in training.
The gap (heldout minus own) measures memorisation of the particular text. Both files are
read up to --max_lines lines and tokenised line by line (the same ids as train_model.py
uses). The corpus, tokeniser and window length are read from the checkpoint. Models are
loaded with the corrected loader.

Usage (one call per manifest; rows are appended to --out, and checkpoints already in
--out are skipped, so a job can be resumed):
  python scripts/heldout_loss.py --manifest results/amendment8/manifest_regular_grouped.csv \
      --heldout_dir E:/overhypothesis_fresh/heldout/regular_grouped --out results/amendment8/heldout_loss.csv
"""
import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oneshot_eval as oe  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


def read_ids(path, tok, max_lines):
    lines = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line:
                lines.append(line)
            if len(lines) >= max_lines:
                break
    return np.fromiter((i for e in tok.encode_batch(lines) for i in e.ids), dtype=np.int64)


def mean_loss(model, ids, T, n_windows, seed):
    rng = random.Random(seed)
    starts = [rng.randrange(0, len(ids) - T - 1) for _ in range(n_windows)]
    tot = 0.0
    with torch.no_grad():
        for s in starts:
            x = torch.from_numpy(ids[s:s + T].copy())[None]
            y = torch.from_numpy(ids[s + 1:s + T + 1].copy())
            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            tot += torch.nn.functional.cross_entropy(logits[0], y, reduction="sum").item()
    return tot / (n_windows * T)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--heldout_dir", required=True, help="folder holding regular_seed<s>.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_windows", type=int, default=300)
    ap.add_argument("--max_lines", type=int, default=44_800)
    ap.add_argument("--conditions", nargs="*", default=None, help="keep only these manifest conditions")
    a = ap.parse_args()

    with open(a.manifest, encoding="utf-8") as f:
        runs = [r for r in csv.DictReader(f) if not a.conditions or r["condition"] in a.conditions]
    done = set()
    if Path(a.out).exists():
        with open(a.out, encoding="utf-8") as f:
            done = {r["ckpt"] for r in csv.DictReader(f)}
    runs = [r for r in runs if r["ckpt"] not in done]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    fields = ["ckpt", "condition", "size", "seed", "own_file", "heldout_file", "train_loss_last5",
              "loss_own", "loss_heldout", "gap"]
    with open(out, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        for r in runs:
            ck = torch.load(r["ckpt"], map_location="cpu", weights_only=False)
            args = ck["args"]
            seed = int(args["seed"])
            own = Path(args["corpus_dir"]) / f"{args['condition']}_seed{seed}.txt"
            held = Path(a.heldout_dir) / f"regular_seed{seed}.txt"
            tok_path = Path(args.get("tok_dir", "data/tokenizers")) / f"{args['condition']}_tokenizer.json"
            if not tok_path.exists():
                tok_path = Path(args.get("tok_dir", "data/tokenizers")) / "regular_tokenizer.json"
            tok = Tokenizer.from_file(str(tok_path))
            model, T = oe.load_model(r["ckpt"], "scripts", "train_model", "OverhypothesisLM")
            l_own = mean_loss(model, read_ids(own, tok, a.max_lines), T, a.n_windows, 7)
            l_held = mean_loss(model, read_ids(held, tok, a.max_lines), T, a.n_windows, 7)
            log = Path(r["ckpt"]).parent / "train_log.json"
            last = None
            if log.exists():
                L = json.load(open(log))
                if L.get("losses"):
                    last = round(sum(L["losses"][-5:]) / len(L["losses"][-5:]), 4)
            row = {"ckpt": r["ckpt"], "condition": r["condition"], "size": r["size"], "seed": seed,
                   "own_file": str(own), "heldout_file": str(held), "train_loss_last5": last,
                   "loss_own": round(l_own, 4), "loss_heldout": round(l_held, 4),
                   "gap": round(l_held - l_own, 4)}
            w.writerow(row)
            fh.flush()
            print({k: row[k] for k in ("condition", "size", "seed", "loss_own", "loss_heldout", "gap")}, flush=True)


if __name__ == "__main__":
    main()
