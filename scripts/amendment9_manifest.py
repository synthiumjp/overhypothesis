#!/usr/bin/env python3
"""
amendment9_manifest.py  |  Lists trained Amendment 9 models for scoring.

Writes one row per model_final.pt (or, with --weights, per saved weights_step*.pt) found under
<ckpt_dir>/<base>/<size>_seed<s>/, with the paths familiarity_eval.py and circuit_analysis.py
need.

  python scripts/amendment9_manifest.py --ckpt_dir "E:/AI Checkpoints/overhypothesis_a9" \
      --corpus_dir E:/overhypothesis_a9 --per_cell 3 --seeds 2027 --out results/amendment9/manifest_pilot.csv
"""
import argparse
import csv
import glob
import os
from pathlib import Path

WINDOWS = {"tiny": 160_000, "medium": 160_000, "small": 256_000}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--corpus_dir", required=True)
    ap.add_argument("--per_cell", type=int, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--tokenizer", default="data/tokenizers/amendment9_tokenizer.json")
    ap.add_argument("--design", default="a9", choices=["a9", "a10"], help="a10: Amendment 10 Part A models")
    ap.add_argument("--weights", action="store_true", help="list saved weights (formation) instead of final models")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for base in (("regular_grouped", "regular_random", "reassigned_grouped") if a.design == "a9"
                 else ("regular_grouped", "reassigned_grouped")):
        for size in ("tiny", "small", "medium"):
            for seed in a.seeds:
                run = Path(a.ckpt_dir) / base / f"{size}_seed{seed}"
                files = (sorted(glob.glob(str(run / "weights_step*.pt"))) if a.weights
                         else [str(run / "model_final.pt")] if (run / "model_final.pt").exists() else [])
                cdir = Path(a.corpus_dir) / base / f"w{WINDOWS[size] // 1000}k"
                for f in files:
                    label = Path(f).stem.replace("weights_", "") if a.weights else ""
                    rows.append({"base": base, "size": size, "seed": seed, "per_cell": a.per_cell, "design": a.design, "ckpt": f,
                                 "label": label, "tokenizer": a.tokenizer,
                                 "corpus": str(cdir / f"seed{seed}.txt"), "order": str(cdir / f"order_seed{seed}.npy"),
                                 "placements": str(cdir / f"placements_seed{seed}.csv")})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["base", "size", "seed", "per_cell", "design", "ckpt", "label", "tokenizer",
                                           "corpus", "order", "placements"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} models listed in {a.out}")


if __name__ == "__main__":
    main()
