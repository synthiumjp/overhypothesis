#!/usr/bin/env python3
"""
a10_reanalysis_summary.py  |  Summarises results/amendment10/reanalysis.csv (scripts/a10_reanalysis.py):
exploratory analyses of the Experiment 2 models, added after the results.

  python scripts/a10_reanalysis_summary.py --in results/amendment10/reanalysis.csv \
      --out results/amendment10/REANALYSIS.md --tex paper_acl/figures/tab_a10_reanalysis.tex

Seed-level values: mean over a cell's nouns per model, sizes averaged within seed, t-based 95% CI over
ten seeds. The geometry regression is noun-level (familiar nouns, own rows, one exposure level):
copy score ~ cosine with the initial row + relative distance from the trained kinds' mean + row norm
+ size + seed + changing-shape-bursts indicator, with standard errors clustered by seed.
"""
import argparse

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

BASES = [("regular_grouped", "stable"), ("reassigned_grouped", "shifting")]
TYPES = [("unseen", "Unseen"), ("fixed_single", "Fixed, spaced"), ("fixed_run", "Fixed, bursts"),
         ("absent_single", "No shape, spaced"), ("absent_run", "No shape, bursts"),
         ("rotating_single", "Changing, spaced"), ("rotating_run", "Changing, bursts")]


def seedv(df, q="carry"):
    return df.groupby(["size", "seed"])[q].mean().groupby("seed").mean()


def ci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h, stats.ttest_1samp(x, 0).pvalue


def fmt(t, d=2):
    m, lo, hi, p = t
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def tex(v, d=2):
    s = f"{v:.{d}f}"
    return s.replace("-", r"\textminus{}") if v < 0 and float(s) != 0 else s.lstrip("-")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="results/amendment10/reanalysis.csv")
    ap.add_argument("--out", default="results/amendment10/REANALYSIS.md")
    ap.add_argument("--tex", default="paper_acl/figures/tab_a10_reanalysis.tex")
    a = ap.parse_args()
    d = pd.read_csv(a.inp)
    md = ["# Experiment 2 reanalysis (exploratory; added after the results)", ""]
    rows_tex = []
    for base, bl in BASES:
        x = d[d.base == base]

        def cell(t, e, c, q="carry"):
            sel = (x.type == t) & (x.condition == c)
            if t != "unseen":
                sel &= x.exposures == e
            return seedv(x[sel], q)

        md += [f"## {bl}-shape corpora", ""]
        for e in (100, 1000):
            md.append(f"### {e} exposures")
            md.append("| Noun | own | typical | gap | frame2 | colour | cos_init | att_l0 |")
            md.append("|---|---|---|---|---|---|---|---|")
            for t, tl in TYPES:
                vals = []
                for c in ("own", "typical", "gap", "frame2", "colour"):
                    s = cell(t, e, c)
                    vals.append(f"{s.mean():.2f}" if len(s) else "")
                vals += [f"{cell(t, e, 'own', 'cos_init').mean():.3f}", f"{cell(t, e, 'own', 'att_l0_mean').mean():.3f}"]
                md.append(f"| {tl} | " + " | ".join(vals) + " |")
                if base == "regular_grouped" and e == 1000:
                    rows_tex.append((tl, *[cell(t, e, c).mean() if len(cell(t, e, c)) else np.nan
                                           for c in ("own", "typical", "gap", "frame2", "colour")],
                                     cell(t, e, "own", "cos_init").mean(), cell(t, e, "own", "att_l0_mean").mean()))
            md.append("")
            md.append(f"- swap: changing bursts own minus with fixed-burst rows: "
                      f"{fmt(ci(cell('rotating_run', e, 'own') - cell('rotating_run', e, 'swap_fixed_run')))}")
            md.append(f"- swap: fixed bursts with changing-burst rows minus own: "
                      f"{fmt(ci(cell('fixed_run', e, 'swap_rotating_run') - cell('fixed_run', e, 'own')))}")
            for c in ("gap", "frame2", "colour"):
                md.append(f"- {c}: changing bursts {fmt(ci(cell('rotating_run', e, c)))}; minus fixed bursts "
                          f"{fmt(ci(cell('rotating_run', e, c) - cell('fixed_run', e, c)))}; minus changing spaced "
                          f"{fmt(ci(cell('rotating_run', e, c) - cell('rotating_single', e, c)))}")
            md.append(f"- cos_init, changing bursts minus fixed bursts: "
                      f"{fmt(ci(cell('rotating_run', e, 'own', 'cos_init') - cell('fixed_run', e, 'own', 'cos_init')), 3)}")
            md.append(f"- layer-0 attention to the context shape, changing bursts minus fixed bursts: "
                      f"{fmt(ci(cell('rotating_run', e, 'own', 'att_l0_mean') - cell('fixed_run', e, 'own', 'att_l0_mean')), 3)}")
            o = x[(x.condition == "own") & (x.type != "unseen") & (x.exposures == e)].copy()
            o["rot_run"] = (o.type == "rotating_run").astype(int)
            m = smf.ols("carry ~ cos_init + dist_mean_rel + row_norm + C(size) + C(seed) + rot_run", data=o).fit(
                cov_type="cluster", cov_kwds={"groups": o.seed})
            lo, hi = m.conf_int().loc["rot_run"]
            m0 = smf.ols("carry ~ cos_init + dist_mean_rel + row_norm + C(size)", data=o[o.rot_run == 0]).fit()
            pred = m0.predict(o[o.rot_run == 1]).mean()
            md.append(f"- geometry regression: changing-bursts coefficient {m.params['rot_run']:.2f} [{lo:.2f}, {hi:.2f}] "
                      f"(p = {m.pvalues['rot_run']:.2g}); observed {o[o.rot_run == 1].carry.mean():.2f}, predicted from "
                      f"the other histories' geometry {pred:.2f}; geometry-only R2 {m0.rsquared:.2f}")
            md.append("")
        own = x[x.condition == "own"]
        r = stats.spearmanr(own.att_l0_mean, own.carry)
        md.append(f"- noun-level Spearman correlation of layer-0 attention with copy score: {r.statistic:.2f}")
        md.append("")
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r"Noun & Own row & Typical row & Gap & Other frame & Colour & Cos.\ initial & Attention \\", r"\midrule"]
    for t in rows_tex:
        lines.append(t[0] + " & " + " & ".join("" if (isinstance(v, float) and np.isnan(v)) else
                                                tex(v, 3 if i >= 5 else 2) for i, v in enumerate(t[1:])) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(a.tex, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("written", a.out, a.tex)


if __name__ == "__main__":
    main()
