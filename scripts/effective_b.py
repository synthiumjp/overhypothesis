#!/usr/bin/env python3
"""
effective_b.py  |  Clustering of different objects in natural corpora (exploratory;
uses the Study 3 measures and the hand coding, both already reported).

B counts nearby repetitions of an object label, whether the two mentions refer to
the same object or to different objects of the kind. Study 1's graded corpora repeat
different exemplars. To put natural corpora on that scale, the excess clustering
(B - 1) is scaled by the share of coded pairs that refer to different objects:

  B_eff = 1 + (B - 1) * d

  d (main)    = different / (different + same)       pairs about objects, either way
  d (strict)  = different / all coded pairs           generic and unclear count as not different

This assumes the pairs that make up the excess have the coded mix. It is a rough
scale, reported as exploratory, with both versions of d.

The Study 1 index at each B_eff is interpolated on the corrected Study 1 curve
(results/rescore, six Regular orders), linear in log B.

Usage:  python scripts/effective_b.py --out results/rescore/effective_b.json
"""
import argparse
import json
import math

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--coding", default="results/study3x/coding_summary.csv")
ap.add_argument("--tests3x", default="results/study3x/tests.json")
ap.add_argument("--curve", default="results/rescore/study1x_exploratory_by_condition.csv")
ap.add_argument("--manifest", default="data/corpora_x/manifest.json")
ap.add_argument("--out", required=True)
a = ap.parse_args()

t = json.load(open(a.tests3x))
B = {"childes": t["childes"]["B"], "cabank": t["cabank"]["B"], "pile10k": 1.43}
c = pd.read_csv(a.coding)
c = c[c.subset == "all"].set_index(["source", "measure"]).k

curve = pd.read_csv(a.curve, index_col=0)["all"]
man = pd.DataFrame(json.load(open(a.manifest))).groupby("corpus").B.mean()
pts = sorted([(man["regular_random"], curve["regular"]), (man["graded_b1p4"], curve["regular_graded_b1p4"]),
              (man["graded_b3p6"], curve["regular_graded_b3p6"]), (man["graded_b7"], curve["regular_graded_b7"]),
              (man["graded_b12"], curve["regular_graded_b12"]), (man["regular_grouped"], curve["regular_bursty"])])


def interp(b):
    for (b0, y0), (b1, y1) in zip(pts, pts[1:]):
        if b0 <= b <= b1:
            w = (math.log(b) - math.log(b0)) / (math.log(b1) - math.log(b0))
            return y0 + w * (y1 - y0)
    return None


out = {"study1_curve": [{"B": round(b, 3), "index": round(y, 3)} for b, y in pts]}
for s in ("childes", "cabank", "pile10k"):
    diff, same, n = c[(s, "different")], c[(s, "same")], sum(c[(s, m)] for m in ("different", "same", "generic", "unclear"))
    r = {"B": round(B[s], 3), "different": int(diff), "same": int(same), "n_coded": int(n)}
    for name, d in (("main", diff / (diff + same)), ("strict", diff / n)):
        be = 1 + (B[s] - 1) * d
        r[name] = {"d": round(d, 3), "B_eff": round(be, 3), "study1_index_at_B_eff": round(interp(be), 3) if interp(be) else None}
    out[s] = r
json.dump(out, open(a.out, "w"), indent=2)
print(json.dumps(out, indent=2))
