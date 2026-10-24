#!/usr/bin/env python3
"""
familiarity_items.py  |  Test items for Amendment 9 (docs/AMENDMENT9.md, addendum item 4).

Every item ends with "A {noun} is a", so the next word is a shape; all measures are read from
the log-softmax at that last position. Item kinds:
  carry      context "A {noun} is a {s} {colour} {texture} thing." then "A {q} is a";
             q = noun (same) or the next noun of the same cell (different); s never s*;
             each of the nine shapes other than s* four times (36 draws per noun)
  agree      as carry with s = s* (4 draws per noun; exploratory)
  know       one base sentence about a trained kind, then "A {noun} is a" (36 per noun)
  trained    for each of the 32 trained kinds: context with shape s, then the same kind
             (each of the ten shapes four times; used in Reassigned models, H34)
Items depend only on the seed (and nouns per cell), not on the model.

Usage:  python scripts/familiarity_items.py --seed 2027 --per_cell 2 --out results/amendment9/items/items_seed2027.csv
"""
import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus_generator as cg  # noqa: E402
import make_familiarity_corpora as mfc  # noqa: E402

ITEM_SEED = 29_000
FIELDS = ["item_id", "kind", "noun", "query_noun", "type", "exposures", "s_star", "q_s_star",
          "ctx_shape", "ctx_shape_pos", "text"]


def frame1(noun, shape, rng):
    return f"A {noun} is a {shape} {rng.choice(cg.COLOUR_TOKENS)} {rng.choice(cg.TEXTURE_TOKENS)} thing."


def make_items(seed, per_cell, design="a9"):
    if design == "a10":
        import make_familiarity_corpora_a10 as m10
        rows = m10.load_assignment(seed)
    else:
        rows = mfc.load_assignment(per_cell, seed)
    exemplars, _ = mfc.base_exemplars(seed)
    rng = random.Random(ITEM_SEED + seed)
    by_cell = {}
    for r in rows:
        by_cell.setdefault((r["type"], r["exposures"]), []).append(r)
    items = []

    def add(kind, r, q, ctx_shape, text, pos):
        items.append({"item_id": len(items), "kind": kind, "noun": r["noun"], "query_noun": q["noun"],
                      "type": r["type"], "exposures": r["exposures"], "s_star": r["s_star"],
                      "q_s_star": q["s_star"], "ctx_shape": ctx_shape, "ctx_shape_pos": pos, "text": text})

    for r in rows:
        cell = by_cell[(r["type"], r["exposures"])]
        partner = cell[(cell.index(r) + 1) % len(cell)]
        shapes = [s for s in cg.SHAPE_TOKENS if s != r["s_star"]] * 4
        rng.shuffle(shapes)
        for s in shapes:
            ctx = frame1(r["noun"], s, rng)
            add("carry_same", r, r, s, f"{ctx} A {r['noun']} is a", 4)
            add("carry_diff", r, partner, s, f"{ctx} A {partner['noun']} is a", 4)
        for _ in range(4):
            add("agree", r, r, r["s_star"], f"{frame1(r['noun'], r['s_star'], rng)} A {r['noun']} is a", 4)
        for _ in range(36):
            add("know", r, r, "", f"{rng.choice(exemplars)} A {r['noun']} is a", -1)
    for noun in cg.TRAIN_NOUNS:
        shapes = cg.SHAPE_TOKENS * 4
        rng.shuffle(shapes)
        r = {"noun": noun, "type": "trained", "exposures": -1, "s_star": ""}
        for s in shapes:
            add("trained", r, r, s, f"{frame1(noun, s, rng)} A {noun} is a", 4)
    return items


def write_items(items, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(items)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--per_cell", type=int, default=2, choices=[2, 3])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    it = make_items(a.seed, a.per_cell)
    write_items(it, a.out)
    print(f"{len(it)} items written to {a.out}")
