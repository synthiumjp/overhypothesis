#!/usr/bin/env python3
"""
study3_coding_summary.py  |  S23 (Amendment 5): per-corpus shares of hand-coded
referent and property codes, with Wilson 95% intervals.

Registered: share coded "different" and share with a stated property, per corpus.
Also reported (descriptive): same, generic, unclear, and "not object" notes;
the same shares among pairs that refer to a physical object (exploratory).

Usage: python scripts/study3_coding_summary.py --codes results/study3x/codes_first.csv \
         --key results/study3x/coding_key.csv --out results/study3x/coding_summary.csv
"""
import argparse
import csv
import math


def wilson(k, n, z=1.959964):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    key = {r["pair_id"]: r["source"] for r in csv.DictReader(open(a.key, encoding="utf-8-sig"))}
    rows = list(csv.DictReader(open(a.codes, encoding="utf-8-sig")))
    out = []
    for subset in ("all", "objects_only"):
        for src in ("childes", "cabank", "pile10k"):
            g = [r for r in rows if key[r["pair_id"]] == src]
            if subset == "objects_only":
                g = [r for r in g if "not object" not in r["notes"].lower()]
            n = len(g)
            ref = lambda v: sum(r["referent"].strip().lower() == v for r in g)
            prop = sum(r["property"].strip().lower() not in ("", "none") for r in g)
            notobj = sum("not object" in r["notes"].lower() for r in g)
            for name, k in (("different", ref("different")), ("same", ref("same")),
                            ("generic", ref("generic")), ("unclear", ref("unclear")),
                            ("any_property", prop), ("not_object", notobj)):
                p, lo, hi = wilson(k, n)
                out.append({"subset": subset, "source": src, "measure": name, "k": k, "n": n,
                            "share": round(p, 3), "lo": round(lo, 3), "hi": round(hi, 3)})
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(f"{r['subset']:12s} {r['source']:8s} {r['measure']:12s} {r['k']:3d}/{r['n']:3d} "
              f"{r['share']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")


if __name__ == "__main__":
    main()
