#!/usr/bin/env python3
"""
make_paper_figures.py | Rebuild the data behind Figures 2 to 4 of the paper, and draw them.

Figure 1 is a schematic with illustrative sentences and has no data.

Inputs (all in the repository):
  Figure 2  results/rescore/study1x_exploratory_by_condition.csv  (index per size, corrected scores)
            data/corpora_x/manifest.json                          (B of each corpus, mean over seeds)
            results/cl_revision/summary.json                      (effective clustering of natural corpora)
  Figure 3  results/study1o/summary_per_run.csv                    (Objects corpora, k = 1, same noun)
  Figure 4  results/study1d/tests_final_secondary.csv              (Pythia, original items; step-0 level)
            results/study1d/tests_final.json                       (BabyLM baselines, registered H8a/H8b)
            results/study2x/secondary_sets_final.csv               (matched wording: set M = "a" in both,
                                                                   set M2 = "another" in both)
Outputs: results/paper_figures/figure{2,3,4a,4b}.csv and figure{2,3,4}.png

Usage: python scripts/make_paper_figures.py [--out results/paper_figures] [--no-png]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from decimal import Decimal, ROUND_HALF_UP


def r2(v):
    """Round half up to 2 decimals, as in the paper (Python's round() sends 0.835 to 0.83)."""
    return float(Decimal(repr(float(v))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

ORDERS = [("Random", "regular", "regular_random"), ("B 1.4", "regular_graded_b1p4", "graded_b1p4"),
          ("B 3.6", "regular_graded_b3p6", "graded_b3p6"), ("B 7", "regular_graded_b7", "graded_b7"),
          ("B 12", "regular_graded_b12", "graded_b12"), ("Grouped", "regular_bursty", "regular_grouped")]
SIZES = [("3.4M", "tiny"), ("11.0M", "small"), ("25.6M", "medium")]
OBJECTS = [("Random", "regular_o_random"), ("Same object, B 3.6", "regular_o_same_b3p6"),
           ("Different objects, B 3.6", "regular_o_diff_b3p6"), ("Same object, grouped", "regular_o_same_grouped"),
           ("Different objects, grouped", "regular_o_diff_grouped")]
NATURAL = [("Written text", "pile10k"), ("Adult conversation", "cabank"), ("Speech to children", "childes")]


def figure2():
    idx = pd.read_csv("results/rescore/study1x_exploratory_by_condition.csv").set_index("condition")
    bsum = defaultdict(list)
    for r in json.load(open("data/corpora_x/manifest.json")):
        bsum[r["corpus"]].append(r["B"])
    rows = []
    for label, cond, corpus in ORDERS:
        B = sum(bsum[corpus]) / len(bsum[corpus])
        for size, col in SIZES:
            rows.append({"order": label, "B": round(B, 3), "size": size, "index": r2(idx.loc[cond, col])})
        rows.append({"order": label, "B": round(B, 3), "size": "all 15", "index": r2(idx.loc[cond, "all"])})
    corp = json.load(open("results/cl_revision/summary.json"))["corpora"]
    refs = [{"corpus": n, "B_eff": r2(corp[k]["B_eff"]), "lo": r2(corp[k]["B_eff_lo"]),
             "hi": r2(corp[k]["B_eff_hi"])} for n, k in NATURAL]
    return pd.DataFrame(rows), pd.DataFrame(refs)


def figure3():
    d = pd.read_csv("results/study1o/summary_per_run.csv")
    d = d[(d.source == "lm") & (d.k == 1) & (d.match.astype(str) == "True")]
    rows = []
    for label, cond in OBJECTS:
        g = d[d.condition == cond]
        for dim in ("shape", "colour", "texture"):
            rows.append({"run": label, "dimension": dim, "delta": r2(g[dim].mean()), "n_models": len(g)})
    return pd.DataFrame(rows)


def figure4():
    s = pd.read_csv("results/study1d/tests_final_secondary.csv")
    py = s[s.family == "pythia"]
    final = py.loc[py.groupby("model_id").step.idxmax()]
    untrained = py[py.step == 0].OI_colour.mean()
    a = [{"model": m.split("-")[-1].upper(), "params": p, "set": "Original items", "index": r2(v)}
         for m, p, v in zip(final.model_id, final.params, final.OI_colour)]
    sec = pd.read_csv("results/study2x/secondary_sets_final.csv")
    pm = sec[(sec.set == "M") & sec.base.str.startswith("pythia")]
    a += [{"model": b.split("-")[-1].upper(), "params": p, "set": "Matched wording (a in both)", "index": r2(v)}
          for b, p, v in zip(pm.base, pm.params, pm.raw_colour)]
    a = pd.DataFrame(a).sort_values(["set", "params"])
    t = json.load(open("results/study1d/tests_final.json"))["primary"]
    babylm = pd.DataFrame([{"model": n, "index": r2(t[k]["mean"]), "lo": r2(t[k]["ci95"][0]),
                            "hi": r2(t[k]["ci95"][1])} for n, k in (("BabyLM 10M words", "H8a"),
                                                                          ("BabyLM 100M words", "H8b"))])
    b = sec[sec.set.isin(["M", "M2"]) & sec.base.str.startswith("pythia")]
    b = pd.DataFrame({"model": b.base.str.split("-").str[-1].str.upper(), "params": b.params,
                      "set": b.set.map({"M": "a in both", "M2": "another in both"}),
                      "noun_specific": b.ks_colour.map(r2), "lo": b.ks_colour_lo.map(r2),
                      "hi": b.ks_colour_hi.map(r2)}).sort_values(["set", "params"])
    return a, babylm, r2(untrained), b


def draw(out, f2, refs, f3, f4a, babylm, untrained, f4b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for _, r in refs.iterrows():
        ax.axvspan(r.lo, r.hi, alpha=0.08, color="grey")
        ax.axvline(r.B_eff, ls="--", color="grey", lw=0.8)
        ax.text(r.B_eff, 6.8, f" {r.corpus}", fontsize=7, rotation=90, va="top")
    per = f2[f2["size"] != "all 15"]
    ax.scatter(per.B, per["index"], color="grey", s=12)
    m = f2[f2["size"] == "all 15"]
    ax.plot(m.B, m["index"], "-o", color="tab:blue")
    ax.set_xscale("log"); ax.set_xlabel("Clustering, B (log scale)"); ax.set_ylabel("Index (nats)")
    fig.tight_layout(); fig.savefig(out / "figure2.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    runs = list(dict.fromkeys(f3.run))
    for j, dim in enumerate(("shape", "colour", "texture")):
        v = f3[f3.dimension == dim].set_index("run").loc[runs, "delta"]
        ax.bar([i + (j - 1) * 0.27 for i in range(len(runs))], v, width=0.27, label=dim)
    ax.axhline(0, color="black", lw=0.6); ax.set_xticks(range(len(runs))); ax.set_xticklabels(runs, fontsize=7)
    ax.set_ylabel("Δ (nats)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "figure3.png", dpi=200); plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for s, g in f4a.groupby("set"):
        a1.plot(g.params, g["index"], "-o", label=s)
    a1.axhline(untrained, ls="--", color="grey", lw=0.8)
    for i, r in babylm.iterrows():
        a1.errorbar(124e6 * (0.9 + 0.2 * i), r["index"], yerr=[[r["index"] - r.lo], [r.hi - r["index"]]], fmt="s", color="black")
    a1.set_xscale("log"); a1.set_xlabel("Parameters"); a1.set_ylabel("Index (nats)"); a1.legend(fontsize=7)
    for s, g in f4b.groupby("set"):
        a2.errorbar(g.params, g.noun_specific, yerr=[g.noun_specific - g.lo, g.hi - g.noun_specific], fmt="-o", label=s)
    a2.axhline(0, color="black", lw=0.6); a2.set_xscale("log"); a2.set_xlabel("Parameters")
    a2.set_ylabel("Noun-specific index (nats)"); a2.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "figure4.png", dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/paper_figures"))
    ap.add_argument("--no-png", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    f2, refs = figure2()
    f3 = figure3()
    f4a, babylm, untrained, f4b = figure4()
    f2.to_csv(a.out / "figure2.csv", index=False)
    refs.to_csv(a.out / "figure2_natural_corpora.csv", index=False)
    f3.to_csv(a.out / "figure3.csv", index=False)
    f4a.to_csv(a.out / "figure4a.csv", index=False)
    babylm.assign(untrained_pythia_level=untrained).to_csv(a.out / "figure4a_babylm.csv", index=False)
    f4b.to_csv(a.out / "figure4b.csv", index=False)
    for name, df in (("Figure 2", f2), ("Figure 2 overlay", refs), ("Figure 3", f3), ("Figure 4a", f4a),
                     ("Figure 4a BabyLM", babylm), ("Figure 4b", f4b)):
        print(f"\n{name}\n{df.to_string(index=False)}")
    print(f"\nUntrained Pythia level (mean of step-0 checkpoints): {untrained}")
    if not a.no_png:
        draw(a.out, f2, refs, f3, f4a, babylm, untrained, f4b)


if __name__ == "__main__":
    main()
