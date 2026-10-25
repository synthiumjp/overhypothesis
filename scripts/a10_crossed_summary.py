#!/usr/bin/env python3
"""
a10_crossed_summary.py  |  Summarises results/amendment10/crossed.csv (scripts/a10_crossed.py): copy score when
the context sentence is about a noun of one history and the query names a noun of another. Exploratory.

  python scripts/a10_crossed_summary.py --out results/amendment10/CROSSED.md --tex paper_acl/figures/tab_a10_crossed.tex

Seed-level: mean over items per model, sizes averaged within seed, t-based 95% CI over ten seeds. Pairs with the
same noun are excluded. Main effects are averaged over the six familiar histories of the other noun.
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

ORDER = [("unseen", "Unseen"), ("fixed_single", "Fixed, spaced"), ("fixed_run", "Fixed, bursts"),
         ("absent_single", "No shape, spaced"), ("absent_run", "No shape, bursts"),
         ("rotating_single", "Changing, spaced"), ("rotating_run", "Changing, bursts")]


def ci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{m:.2f} [{m - h:.2f}, {m + h:.2f}]"


def tex(v):
    s = f"{v:.2f}"
    return s.replace("-", r"\textminus{}") if v < 0 and float(s) != 0 else s.lstrip("-")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="results/amendment10/crossed.csv")
    ap.add_argument("--out", default="results/amendment10/CROSSED.md")
    ap.add_argument("--tex", default="paper_acl/figures/tab_a10_crossed.tex")
    a = ap.parse_args()
    d = pd.read_csv(a.inp)
    d = d[~d.same_noun.astype(bool)]
    md = ["# Crossed histories (exploratory; added after the results)", "",
          "Rows: history of the noun the context sentence is about. Columns: history of the queried noun.", ""]
    tex_lines = None
    for base in ("regular_grouped", "reassigned_grouped"):
        for e in (100, 1000):
            x = d[(d.base == base) & (d.exposures == e)]
            s = (x.groupby(["size", "seed", "ctx_type", "query_type"]).carry.mean()
                 .groupby(["seed", "ctx_type", "query_type"]).mean().reset_index())
            m = s.groupby(["ctx_type", "query_type"]).carry.mean().unstack()
            md += [f"## {base}, {e} exposures", "", "| context \\ query | " + " | ".join(l for _, l in ORDER) + " |",
                   "|---" * (len(ORDER) + 1) + "|"]
            for t, l in ORDER:
                md.append(f"| {l} | " + " | ".join(f"{m.loc[t, q]:.2f}" for q, _ in ORDER) + " |")
            fam = [t for t, _ in ORDER if t != "unseen"]
            sf = s[s.ctx_type.isin(fam) & s.query_type.isin(fam)]
            q = sf.groupby(["seed", "query_type"]).carry.mean().unstack()
            c = sf.groupby(["seed", "ctx_type"]).carry.mean().unstack()
            md += ["", f"- queried noun changing bursts minus fixed bursts (over familiar contexts): "
                       f"{ci(q.rotating_run - q.fixed_run)}",
                   f"- context noun changing bursts minus fixed bursts (over familiar queries): "
                   f"{ci(c.rotating_run - c.fixed_run)}", ""]
            if base == "regular_grouped" and e == 1000:
                tex_lines = [r"\begin{tabular}{@{}l" + "r" * len(ORDER) + r"@{}}", r"\toprule",
                             r"Context noun $\backslash$ queried noun & " + " & ".join(
                                 l.replace(", ", r",\newline ") if False else l for _, l in ORDER) + r" \\",
                             r"\midrule"]
                for t, l in ORDER:
                    tex_lines.append(l + " & " + " & ".join(tex(m.loc[t, qq]) for qq, _ in ORDER) + r" \\")
                tex_lines += [r"\bottomrule", r"\end{tabular}"]
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    open(a.tex, "w", encoding="utf-8").write("\n".join(tex_lines) + "\n")
    print("written", a.out, a.tex)


if __name__ == "__main__":
    main()
