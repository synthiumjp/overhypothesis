#!/usr/bin/env python3
"""
amendment9_geometry.py  |  Exploratory (registered as exploratory): geometry of each familiarity
noun's embedding rows, as a possible mediator of carry (docs/AMENDMENT9.md section 5).

For every model in the manifest and every familiarity noun:
  in_norm        norm of the noun's input row
  in_dist        distance of the input row from the mean input row of the 32 trained kinds,
                 divided by the mean distance of the trained kinds' rows from that mean (1 = typical)
  in_cos         cosine between the input row and that mean
  out_norm       norm of the output row
  logit_prior    mean logit of the noun at the start of a sentence (after 20 neutral base sentences)
Unseen nouns are also given the typical embedding of the Amendment 8 rescoring (emb = typical).

  python scripts/amendment9_geometry.py --manifest results/amendment9/manifest.csv --out results/amendment9/geometry.csv
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
import make_familiarity_corpora as mfc  # noqa: E402
from familiarity_eval import forward, token_id, typical_rows  # noqa: E402
from oneshot_eval import load_model  # noqa: E402

FIELDS = ["base", "size", "seed", "ckpt", "label", "noun", "type", "exposures", "emb", "in_norm", "in_dist", "in_cos",
          "out_norm", "logit_prior"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    runs = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
    rows = []
    for i, run in enumerate(runs, 1):
        tok = Tokenizer.from_file(run["tokenizer"])
        model, _ = load_model(run["ckpt"], None, "train_model", "OverhypothesisLM")
        model.eval()
        assign = mfc.load_assignment(int(run["per_cell"]), int(run["seed"]))
        E = model.tok_embed.weight.detach()
        U = model.lm_head.weight.detach()
        train_ids = [token_id(tok, n) for n in cg.TRAIN_NOUNS]
        mu = E[train_ids].mean(0)
        r = (E[train_ids] - mu).norm(dim=1).mean()
        exemplars, _ = mfc.base_exemplars(int(run["seed"]))
        rng = random.Random(53_000 + int(run["seed"]))
        prefixes = [tok.encode(rng.choice(exemplars)).ids for _ in range(20)]
        with torch.no_grad():
            logits = [forward(model, torch.tensor([p]))[0] for p in prefixes]
        unseen = [x for x in assign if x["type"] == "unseen"]
        typ = typical_rows(E, train_ids, len(unseen))
        typ_of = {x["noun"]: typ[j] for j, x in enumerate(unseen)}
        for x in assign:
            nid = token_id(tok, x["noun"])
            for emb in (["original", "typical"] if x["type"] == "unseen" else ["original"]):
                v = E[nid] if emb == "original" else typ_of[x["noun"]]
                rows.append({"base": run["base"], "size": run["size"], "seed": run["seed"], "ckpt": run["ckpt"],
                             "label": run.get("label", ""), "noun": x["noun"], "type": x["type"],
                             "exposures": x["exposures"], "emb": emb, "in_norm": v.norm().item(),
                             "in_dist": ((v - mu).norm() / r).item(),
                             "in_cos": torch.nn.functional.cosine_similarity(v, mu, dim=0).item(),
                             "out_norm": U[nid].norm().item(),
                             "logit_prior": sum(l[nid].item() for l in logits) / len(logits)})
        print(f"[{i}/{len(runs)}] {run['base']} {run['size']} seed {run['seed']}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows written to {a.out}")


if __name__ == "__main__":
    main()
