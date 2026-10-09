#!/usr/bin/env python3
"""
study3_reliability.py  |  Second-coder subset and agreement for the Study 3 hand coding
(Amendment 5, S23; rubric docs/exemplar_coding_rubric.md).

  subset  : draw 25% of pair_ids with seed 2031 and write a blank sheet for the
            second coder (same columns, codes empty).
  agree   : Cohen's kappa for `referent` and percentage agreement for `property`
            (exact match of the set of property codes) on the shared rows.

The rubric fixes the seed (2031) and the share (25%) but not the draw itself;
this script makes it explicit: random.Random(2031).sample(range(n), round(n/4)).

Usage:
  python scripts/study3_reliability.py subset --sheet results/study3x/coding_sheet.csv \
         --out results/study3x/coding_sheet_second.csv
  python scripts/study3_reliability.py agree --first results/study3x/coding_sheet_coded.csv \
         --second results/study3x/coding_sheet_second_coded.csv --out results/study3x/reliability.json
"""

import argparse
import csv
import json
import random
from collections import Counter

SEED = 2031


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def subset(a):
    rows = read(a.sheet)
    ids = sorted(random.Random(SEED).sample(range(len(rows)), round(len(rows) / 4)))
    keep = [r for r in rows if int(r["pair_id"]) in set(ids)]
    for r in keep:
        r["referent"] = r["property"] = r["notes"] = ""
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(keep)
    print(f"{len(keep)} of {len(rows)} rows to {a.out}")


def props(s):
    s = (s or "").strip().lower()
    return frozenset() if s in ("", "none") else frozenset(p.strip() for p in s.split(";") if p.strip())


def kappa(x, y):
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = Counter(x), Counter(y)
    pe = sum(cx[k] * cy[k] for k in set(cx) | set(cy)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else float("nan"), po


def agree(a):
    first = {r["pair_id"]: r for r in read(a.first)}
    second = {r["pair_id"]: r for r in read(a.second)}
    ids = sorted(set(first) & set(second), key=int)
    ref1 = [first[i]["referent"].strip().lower() for i in ids]
    ref2 = [second[i]["referent"].strip().lower() for i in ids]
    k, po = kappa(ref1, ref2)
    prop = sum(props(first[i]["property"]) == props(second[i]["property"]) for i in ids) / len(ids)
    res = {"n_rows": len(ids), "referent_kappa": k, "referent_agreement": po,
           "property_agreement": prop,
           "confusion": {f"{p}->{q}": c for (p, q), c in Counter(zip(ref1, ref2)).items()}}
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("subset"); p.add_argument("--sheet", required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("agree"); p.add_argument("--first", required=True)
    p.add_argument("--second", required=True); p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"subset": subset, "agree": agree}[a.cmd](a)


if __name__ == "__main__":
    main()
