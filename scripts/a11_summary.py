#!/usr/bin/env python3
"""
a11_summary.py  |  Descriptive summaries of Amendment 11 (Experiment 3) for the paper. The registered tests are in
scripts/a11_tests.py; everything here is either a registered value restated or exploratory.

  python scripts/a11_summary.py --out results/amendment11/SUMMARY.md --tex_dir paper_acl/figures

Seed-level: mean over a cell's nouns per model, exposure levels weighted equally when pooled, sizes averaged within
seed, t-based 95% intervals over seeds. Writes SUMMARY.md, tab_a11_cells.tex (cell means by size) and
tab_a11_crossed.tex (crossed histories at 1,000 exposures).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SIZES = ("tiny", "small", "medium")
ORDER = [("unseen", "Unseen"), ("fixed_single", "Fixed, spaced"), ("fixed_run", "Fixed, bursts"),
         ("absent_single", "No shape, spaced"), ("absent_run", "No shape, bursts"),
         ("rotating_single", "Changing, spaced"), ("rotating_run", "Changing, bursts")]
LABEL = dict(ORDER)


def ci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def f(x):
    m, lo, hi = ci(x)
    return f"{m:.2f} [{lo:.2f}, {hi:.2f}]"


def tex(v):
    s = f"{v:.2f}"
    return s.replace("-", r"\textminus{}") if v < 0 and float(s) != 0 else s.lstrip("-")


def cell(s, base, typ, e, q="carry", emb="original", sizes=SIZES):
    d = s[(s.base == base) & (s.type == typ) & (s.emb == emb) & s["size"].isin(sizes)]
    if e is not None:
        d = d[d.exposures.isin([e] if np.isscalar(e) else e)]
    return d.groupby(["size", "seed", "exposures"])[q].mean().groupby(["size", "seed"]).mean()


def seed(g):
    return g.groupby("seed").mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default="results/amendment11/scores.csv")
    ap.add_argument("--crossed", default="results/amendment11/crossed.csv")
    ap.add_argument("--rows", default="results/amendment11/reanalysis.csv")
    ap.add_argument("--out", default="results/amendment11/SUMMARY.md")
    ap.add_argument("--tex_dir", default="paper_acl/figures")
    a = ap.parse_args()
    s = pd.read_csv(a.scores)
    s = s[s.ckpt.str.endswith("model_final.pt")]
    s["win"] = -s["compete"]
    G = "regular_grouped"
    md = ["# Amendment 11 (Experiment 3): descriptive summary", "",
          "Seed-level means with t-based 95% intervals; sizes averaged within seed. Registered tests: RESULTS_AC.md.", ""]
    # key comparisons, mixed design
    md += ["## Key comparisons, mixed design (pooled 100 and 1,000; and 1,000 alone)", ""]
    comps = [("Changing: bursts - spaced", "rotating_run", "rotating_single"),
             ("Bursts: changing - fixed", "rotating_run", "fixed_run"),
             ("No shape: bursts - spaced", "absent_run", "absent_single"),
             ("Fixed: bursts - spaced", "fixed_run", "fixed_single"),
             ("Spaced: fixed - changing", "fixed_single", "rotating_single"),
             ("Spaced: fixed - no shape", "fixed_single", "absent_single")]
    for lab, x, y in comps:
        pooled = seed(cell(s, G, x, [100, 1000]) - cell(s, G, y, [100, 1000]))
        k = seed(cell(s, G, x, 1000) - cell(s, G, y, 1000))
        h = seed(cell(s, G, x, 100) - cell(s, G, y, 100))
        md.append(f"- {lab}: pooled {f(pooled)}; 100 {f(h)}; 1,000 {f(k)}")
    for e in ([100, 1000], 100, 1000):
        x = seed(cell(s, G, "absent_single", e) - cell(s, G, "unseen", None, emb="typical"))
        md.append(f"- Spaced no shape - unseen typical ({e}): {f(x)}")
    md.append(f"- Unseen, own rows: {f(seed(cell(s, G, 'unseen', None)))}; typical embedding: "
              f"{f(seed(cell(s, G, 'unseen', None, emb='typical')))}")
    md.append(f"- Unseen typical minus own: {f(seed(cell(s, G, 'unseen', None, emb='typical') - cell(s, G, 'unseen', None)))}")
    # noun-specific
    ns = {t: seed(cell(s, G, t, 1000, q="noun_specific")).mean() for t, _ in ORDER if t != "unseen"}
    ns["unseen"] = seed(cell(s, G, "unseen", None, q="noun_specific")).mean()
    md.append("- Noun-specific copy score (same noun minus next noun of the cell), 1,000 / unseen: "
              + ", ".join(f"{LABEL[t]} {v:.2f}" for t, v in ns.items()))
    # behaviour
    for t in ("fixed_single", "rotating_single", "fixed_run", "rotating_run"):
        md.append(f"- {LABEL[t]}: L(s) - L(own) at 100 {f(seed(cell(s, G, t, 100, q='win')))}; "
                  f"at 1,000 {f(seed(cell(s, G, t, 1000, q='win')))}")
    for t in ("fixed_single", "fixed_run"):
        md.append(f"- {LABEL[t]}: knowledge at 100 {f(seed(cell(s, G, t, 100, q='know')))}; "
                  f"at 1,000 {f(seed(cell(s, G, t, 1000, q='know')))}")
    # per size
    md += ["", "## By size (means over seeds)", ""]
    for lab, x, y, e in (("Bursts: changing - fixed, 1,000", "rotating_run", "fixed_run", 1000),
                         ("Changing: bursts - spaced, pooled", "rotating_run", "rotating_single", [100, 1000]),
                         ("Spaced: fixed - changing, pooled", "fixed_single", "rotating_single", [100, 1000])):
        g = (cell(s, G, x, e) - cell(s, G, y, e)).groupby("size").mean()
        md.append(f"- {lab}: " + ", ".join(f"{z} {g[z]:.2f}" for z in SIZES))
    g = (cell(s, G, "unseen", None, emb="typical")).groupby("size").mean()
    md.append("- Unseen typical: " + ", ".join(f"{z} {g[z]:.2f}" for z in SIZES))
    # pure designs
    md += ["", "## Single-history corpora (seeds 2029 to 6067)", ""]
    s5 = s[s.seed.isin(sorted(s[s.base == "regular_pure_fixed"].seed.unique()))]
    for base, typs in (("regular_pure_changing", ("rotating_run", "rotating_single", "unseen")),
                       ("regular_pure_fixed", ("fixed_run", "fixed_single", "unseen")),
                       (G, ("rotating_run", "rotating_single", "fixed_run", "fixed_single", "unseen"))):
        for t in typs:
            e = None if t == "unseen" else 1000
            md.append(f"- {base} {LABEL[t]} ({'own rows' if t == 'unseen' else '1,000'}): {f(seed(cell(s5, base, t, e)))}")
    # cell table by size
    rows = []
    for base, blab in ((G, "Mixed"), ("regular_pure_changing", "Changing only"), ("regular_pure_fixed", "Fixed only")):
        for t, l in ORDER:
            for e in ((0,) if t == "unseen" else (100, 1000)):
                d = s[(s.base == base) & (s.type == t) & (s.emb == "original") & (s.exposures == e)]
                if not len(d):
                    continue
                g = d.groupby(["size", "seed"]).carry.mean()
                by = g.groupby("size").mean()
                m, lo, hi = ci(g.groupby("seed").mean())
                rows.append((blab, l, "--" if e == 0 else f"{e:,}", [by.get(z, np.nan) for z in SIZES], m, lo, hi,
                             g.index.get_level_values("seed").nunique()))
    tl = [r"\begin{tabular}{@{}lllrrrrl@{}}", r"\toprule",
          r"Corpus & History & Exp. & 3.4M & 10.9M & 25.6M & Mean & 95\% CI \\", r"\midrule"]
    last = None
    for blab, l, e, by, m, lo, hi, n in rows:
        if last is not None and blab != last:
            tl.append(r"\midrule")
        tl.append(f"{blab if blab != last else ''} & {l} & {e} & " + " & ".join(tex(v) for v in by)
                  + f" & {tex(m)} & [{tex(lo)}, {tex(hi)}] \\\\")
        last = blab
    tl += [r"\bottomrule", r"\end{tabular}"]
    Path(a.tex_dir, "tab_a11_cells.tex").write_text("\n".join(tl) + "\n", encoding="utf-8")
    # crossed
    cr = pd.read_csv(a.crossed)
    cr = cr[(cr.base == G) & (~cr.same_noun.astype(bool)) & cr.ckpt.str.endswith("model_final.pt")]
    md += ["", "## Crossed histories, mixed design", ""]
    for e in (100, 1000):
        x = cr[cr.exposures == e]
        sm = (x.groupby(["size", "seed", "ctx_type", "query_type"]).carry.mean()
              .groupby(["seed", "ctx_type", "query_type"]).mean().reset_index())
        m = sm.groupby(["ctx_type", "query_type"]).carry.mean().unstack()
        fam = [t for t, _ in ORDER if t != "unseen"]
        sf = sm[sm.ctx_type.isin(fam) & sm.query_type.isin(fam)]
        q = sf.groupby(["seed", "query_type"]).carry.mean().unstack()
        c = sf.groupby(["seed", "ctx_type"]).carry.mean().unstack()
        md += [f"- {e}: queried changing - fixed bursts {f(q.rotating_run - q.fixed_run)}; context changing - fixed "
               f"bursts {f(c.rotating_run - c.fixed_run)}; both changing bursts {m.loc['rotating_run', 'rotating_run']:.2f}"]
        if e == 1000:
            tl = [r"\begin{tabular}{@{}l" + "r" * len(ORDER) + r"@{}}", r"\toprule",
                  r"Context $\backslash$ query & " + " & ".join(
                      r"\shortstack{" + l.replace(", ", r"\\ ") + "}" for _, l in ORDER) + r" \\", r"\midrule"]
            for t, l in ORDER:
                tl.append(l + " & " + " & ".join(tex(m.loc[t, qq]) for qq, _ in ORDER) + r" \\")
            tl += [r"\bottomrule", r"\end{tabular}"]
            Path(a.tex_dir, "tab_a11_crossed.tex").write_text("\n".join(tl) + "\n", encoding="utf-8")
    # rows: geometry, exploratory descriptive
    r = pd.read_csv(a.rows)
    r = r[(r.base == G) & r.ckpt.str.endswith("model_final.pt")]
    o = r[(r.condition == "own") & (r.exposures == 1000)]
    md += ["", "## Rows, mixed design, 1,000 exposures (exploratory)", ""]
    for t in ("fixed_run", "rotating_run", "fixed_single", "rotating_single"):
        oo = o[o.type == t].groupby(["size", "seed"])[["cos_init", "dist_mean_rel", "row_norm"]].mean().groupby("seed").mean()
        md.append(f"- {LABEL[t]}: cos_init {oo.cos_init.mean():.3f}, rel. distance {oo.dist_mean_rel.mean():.3f}, "
                  f"norm {oo.row_norm.mean():.3f}")
    for cond in ("swap_fixed_run", "swap_rotating_run", "typical", "gap", "frame2", "colour"):
        for t in ("fixed_run", "rotating_run"):
            x = r[(r.condition == cond) & (r.type == t) & (r.exposures == 1000)]
            if len(x):
                md.append(f"- {cond}, {LABEL[t]}: {f(x.groupby(['size', 'seed']).carry.mean().groupby('seed').mean())}")
    if "att_l0_mean" in r:
        oa = r[(r.condition == "own") & r.att_l0_mean.notna() & (r.type != "unseen")]
        if len(oa):
            per = oa.groupby(["size", "seed", "type", "exposures"])[["carry", "att_l0_mean"]].mean()
            rho = stats.spearmanr(per.carry, per.att_l0_mean).statistic
            md.append(f"- layer-0 attention vs copy score (cells x models): Spearman {rho:.2f}")
    Path(a.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
