#!/usr/bin/env python3
"""
noun_copy_check.py | Can the Study 1 models tell the novel test nouns apart?
(exploratory diagnostic for the CL revision, 13 Oct 2026)

The noun-specific index is zero in every Study 1 condition. That is only informative if
the models can tell one novel noun from another. The test nouns never occur in training,
so their input embeddings keep their random initial values (weight decay shrinks them by
under 10%). This script asks whether a model, after a one-sentence context about a novel
noun, expects that same noun again more than another novel noun:

  copy preference = log P(same noun | context + " A") - mean over the 7 other novel nouns
                    of log P(other noun | context + " A")

computed on the 200 one-exemplar contexts of the Study 1 test items (8 nouns x 25 draws).
A clearly positive value means the model tracks which novel noun it has just read.

Usage (PC, checkpoints on E:):
  python scripts/noun_copy_check.py --manifest results/study1b/raw/manifest.csv \
      results/study1c/raw/manifest.csv --conditions regular regular_bursty \
      --out results/cl_revision/noun_copy_check.csv
"""
import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oneshot_eval as oe  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", nargs="+", required=True)
ap.add_argument("--conditions", nargs="*", default=None)
ap.add_argument("--items", default="data/wug_tests/oneshot_items.csv")
ap.add_argument("--out", required=True)
a = ap.parse_args()

items = [r for r in oe.read_items(a.items) if str(r["k"]) == "1" and str(r["match"]) == "True" and r["probe_dim"] == "shape"]
nouns = sorted({r["noun"] for r in items})
runs = []
for m in a.manifest:
    with open(m, encoding="utf-8") as f:
        runs += [r for r in csv.DictReader(f) if not a.conditions or r["condition"] in a.conditions]

out = []
for r in runs:
    tok = Tokenizer.from_file(r["tokenizer"])
    noun_ids = {n: tok.encode(n).ids for n in nouns}
    assert all(len(v) == 1 for v in noun_ids.values()), "novel nouns must be single tokens"
    model, _ = oe.load_model(r["ckpt"], "scripts", "train_model", "OverhypothesisLM")
    prefs = []
    with torch.no_grad():
        for it in items:
            ids = tok.encode(it["context"] + " A").ids
            logits = model(torch.tensor([ids]))
            logits = logits[0] if isinstance(logits, (tuple, list)) else logits
            lp = torch.log_softmax(logits[0, -1].float(), dim=-1)
            same = lp[noun_ids[it["noun"]][0]].item()
            others = [lp[noun_ids[n][0]].item() for n in nouns if n != it["noun"]]
            prefs.append(same - sum(others) / len(others))
    row = {"condition": r["condition"], "size": r["size"], "seed": r["seed"],
           "copy_preference": sum(prefs) / len(prefs),
           "share_items_positive": sum(p > 0 for p in prefs) / len(prefs)}
    out.append(row)
    print(row, flush=True)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
