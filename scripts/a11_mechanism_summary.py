#!/usr/bin/env python3
"""
a11_mechanism_summary.py  |  Summarises results/amendment11/mechanism.csv (scripts/a11_mechanism.py). Exploratory.

  python scripts/a11_mechanism_summary.py --out results/amendment11/MECHANISM.md --tex paper_acl/figures/tab_a11_mech.tex

Seed-level: per model, sizes averaged within seed, t-based 95% intervals over ten seeds.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

LAB = {"fixed_run": "Fixed, bursts", "rotating_run": "Changing, bursts", "fixed_single": "Fixed, spaced",
       "rotating_single": "Changing, spaced", "absent_single": "No shape, spaced", "absent_run": "No shape, bursts",
       "unseen": "Unseen, typical embedding"}


def ci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def tex(v):
    s = f"{v:.2f}"
    return s.replace("-", r"\textminus{}") if v < 0 and float(s) != 0 else s.lstrip("-")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="results/amendment11/mechanism.csv")
    ap.add_argument("--scores", default="results/amendment11/scores.csv")
    ap.add_argument("--out", default="results/amendment11/MECHANISM.md")
    ap.add_argument("--tex", default="paper_acl/figures/tab_a11_mech.tex")
    a = ap.parse_args()
    d = pd.read_csv(a.inp)
    md = ["# Experiment 3 mechanism analyses (exploratory; added after the results)", ""]
    # steering
    st = d[d.analysis == "steer"].pivot_table(index=["size", "seed", "type"], columns="cond", values="value")
    md += ["## Steering (change in copy score, nats)", ""]
    tl = [r"\begin{tabular}{@{}llrlrl@{}}", r"\toprule",
          r"Nouns & Baseline & Direction & 95\% CI & Random & 95\% CI \\", r"\midrule"]
    plan = [("fixed_single", "own", "+v1000", "+random1000"), ("rotating_single", "own", "+v1000", "+random1000"),
            ("absent_single", "own", "+v1000", "+random1000"), ("absent_run", "own", "+v1000", "+random1000"),
            ("unseen", "typical", "+v1000", "+random1000"), ("fixed_run", "own", "+v100", "+random100"),
            ("rotating_run", "own", "-v100", "-random100")]
    for t, base, cv, cr in plan:
        x = st.xs(t, level="type")
        b = ci(x[base].groupby("seed").mean())
        dv = ci((x[cv] - x[base]).groupby("seed").mean())
        dr = ci((x[cr] - x[base]).groupby("seed").mean())
        bys = (x[cv] - x[base]).groupby("size").mean()
        md.append(f"- {LAB[t]} ({cv}): baseline {b[0]:.2f}; direction {dv[0]:.2f} [{dv[1]:.2f}, {dv[2]:.2f}] "
                  f"(tiny {bys['tiny']:.2f}, small {bys['small']:.2f}, medium {bys['medium']:.2f}); random "
                  f"{dr[0]:.2f} [{dr[1]:.2f}, {dr[2]:.2f}]")
        note = {"+v100": "$^a$", "-v100": "$^b$"}.get(cv, "")
        tl.append(f"{LAB[t]}{note} & {tex(b[0])} & {tex(dv[0])} & [{tex(dv[1])}, {tex(dv[2])}] & {tex(dr[0])} & "
                  f"[{tex(dr[1])}, {tex(dr[2])}] \\\\")
    tl += [r"\bottomrule", r"\end{tabular}"]
    Path(a.tex).write_text("\n".join(tl) + "\n", encoding="utf-8")
    # patching
    p = d[d.analysis == "patch"].copy()
    p["dir"] = p.cond.str.split(":").str[0]
    p["kind"] = p.cond.str.split(":").str[1]
    md += ["", "## Activation patching (copy score of the target; fraction of the source-minus-clean difference)", ""]
    for dr in ("R_into_F", "F_into_R"):
        for z in ("tiny", "small", "medium"):
            q = p[(p.dir == dr) & (p["size"] == z)]
            c = q[q.kind == "clean"].value.mean()
            s = q[q.kind == "source"].value.mean()
            pp = q[q.kind == "patch"].groupby(["pos", "layer"]).value.mean().unstack("pos")
            line = f"- {dr}, {z}: clean {c:.2f}, source {s:.2f}; " + "; ".join(
                f"after layer {int(l)}: noun {(r['noun'] - c) / (s - c):.2f}, last {(r['last'] - c) / (s - c):.2f}"
                for l, r in pp.iterrows())
            md.append(line)
    # attention
    at = d[d.analysis == "attention"]
    x = at[(at.exposures == 1000) & (at.layer == 0)].pivot_table(index=["size", "seed"], columns="type", values="value")
    m = ci((x.rotating_run - x.fixed_run).groupby("seed").mean())
    md += ["", "## Attention from the last position to the context's shape word", "",
           f"- Layer 0, changing minus fixed bursts at 1,000: {m[0]:.3f} [{m[1]:.3f}, {m[2]:.3f}]"]
    s = pd.read_csv(a.scores)
    s = s[(s.base == "regular_grouped") & (s.emb == "original")]
    c = s.groupby(["size", "seed", "type", "exposures"]).carry.mean()
    for L in sorted(at.layer.unique()):
        aa = at[at.layer == L].set_index(["size", "seed", "type", "exposures"]).value
        j = pd.concat([c, aa], axis=1, join="inner")
        md.append(f"- Layer {int(L)}: Spearman with copy score over cells x models {stats.spearmanr(j.carry, j.value).statistic:.2f} (n = {len(j)})")
    Path(a.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
