#!/usr/bin/env python3
"""
a11_model_check.py  |  Registered model-failure rule of Amendment 11 (docs/AMENDMENT11.md, section 3).

  python scripts/a11_model_check.py --manifest results/amendment11/manifest.csv --out results/amendment11/model_check.csv

Reads each final model's train_log.json (next to model_final.pt). A model fails when its final training loss is
not finite, or is more than 0.5 nats above the median final loss of the models with the same design and size.
A run stopped by a non-finite loss leaves FAILED_nonfinite_loss.txt and no model_final.pt; it never enters the
manifest and is counted as missing by scripts/a11_tests.py. Failed models are retrained once with the same seed;
if the retrained model also fails, it is excluded. scripts/a11_tests.py drops every model listed here as "fail".
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

MARGIN = 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.manifest, encoding="utf-8")) if r["ckpt"].endswith("model_final.pt")]
    seen, recs = set(), []
    for r in rows:
        if r["ckpt"] in seen:
            continue
        seen.add(r["ckpt"])
        log = Path(r["ckpt"]).parent / "train_log.json"
        loss = float("nan")
        if log.exists():
            loss = float(json.loads(log.read_text(encoding="utf-8")).get("final_loss", float("nan")))
        recs.append({"base": r["base"], "size": r["size"], "seed": r["seed"], "ckpt": r["ckpt"], "final_loss": loss})
    for rec in recs:
        same = [x["final_loss"] for x in recs if x["base"] == rec["base"] and x["size"] == rec["size"]
                and math.isfinite(x["final_loss"])]
        med = float(np.median(same)) if same else float("nan")
        rec["median_same_cell"] = med
        ok = math.isfinite(rec["final_loss"]) and rec["final_loss"] <= med + MARGIN
        rec["status"] = "pass" if ok else "fail"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["base", "size", "seed", "ckpt", "final_loss", "median_same_cell", "status"])
        w.writeheader()
        w.writerows(recs)
    bad = [x for x in recs if x["status"] == "fail"]
    print(f"{len(recs)} models checked, {len(bad)} failed")
    for x in bad:
        print(f"  FAIL {x['base']} {x['size']} seed {x['seed']}: final loss {x['final_loss']:.4f} "
              f"(median {x['median_same_cell']:.4f}); retrain once with the same seed")


if __name__ == "__main__":
    main()
