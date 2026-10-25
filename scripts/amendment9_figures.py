#!/usr/bin/env python3
"""
amendment9_figures.py  |  Figures and tables for the Amendment 9 results.

  python scripts/amendment9_figures.py --scores results/amendment9/scores.csv \
      --circuits results/amendment9/circuits --out_dir paper_acl/figures
  (optional) --formation results/amendment9/scores_formation.csv

Values are means over the ten confirmatory seeds of size-averaged cell means (mean over a
cell's nouns), with t-based 95% intervals over seeds, as in the registered tests. Only final
models enter the main figures (amendment9_tests.final_only).

Writes fig_a9_carry.pdf, fig_a9_knowledge.pdf, fig_a9_heads.pdf, tab_a9_cells.tex and, with
--formation, fig_a9_formation.pdf.
"""
import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from amendment9_tests import final_only  # noqa: E402
from make_familiarity_corpora import CONF_SEEDS  # noqa: E402

# Reference categorical palette, first three slots (validated all-pairs), plus ink and grid
COL = {"consistent": "#2a78d6", "uninformative": "#eb6834", "absent": "#1baf7a"}
MARK = {"consistent": "o", "uninformative": "s", "absent": "^"}
NAME = {"consistent": "Fixed shape", "uninformative": "Changing shape", "absent": "No shape"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BASES = [("regular_grouped", "Stable shapes"), ("reassigned_grouped", "Shifting shapes"),
         ("regular_random", "Stable shapes, random order")]
EXPO = [10, 100, 1000, 10000]

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "lines.linewidth": 1.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (x.mean() if len(x) else np.nan), 0.0
    return x.mean(), stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def seed_cells(s, measure):
    """Seed-level values (size-averaged cell means) per (base, emb, type, exposures)."""
    d = s[s[measure].notna()]
    cm = d.groupby(["base", "emb", "size", "seed", "type", "exposures"])[measure].mean().reset_index()
    return cm.groupby(["base", "emb", "seed", "type", "exposures"])[measure].mean().reset_index()


def style(ax, ylabel=None):
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["0", "10", "100", "1k", "10k"])
    ax.set_xlim(-0.4, 4.4)
    if ylabel:
        ax.set_ylabel(ylabel)


def exposure_panel(ax, sc, measure, base, show_unseen=True, show_typical=True):
    d = sc[(sc.base == base) & (sc.emb == "original")]
    for typ in ("consistent", "uninformative", "absent"):
        m = [ci(d[(d.type == typ) & (d.exposures == e)][measure]) for e in EXPO]
        ax.errorbar(range(1, 5), [a for a, _ in m], yerr=[b for _, b in m], color=COL[typ], marker=MARK[typ],
                    markersize=4.5, capsize=2, elinewidth=0.8, markeredgecolor="white", markeredgewidth=0.6,
                    label=NAME[typ], zorder=3)
    if show_unseen:
        a, b = ci(d[d.type == "unseen"][measure])
        ax.errorbar([0], [a], yerr=[b], color=INK, marker="D", markersize=4.5, capsize=2, elinewidth=0.8,
                    label="Unseen", zorder=3)
    if show_typical:
        t = sc[(sc.base == base) & (sc.emb == "typical") & (sc.type == "unseen")][measure]
        if len(t):
            a, _ = ci(t)
            ax.axhline(a, color=INK2, linestyle=(0, (3, 2)), linewidth=0.9, label="Unseen, typical embedding",
                       zorder=2)
    ax.axhline(0, color=INK2, linewidth=0.5, zorder=1)


def fig_carry(sc, out):
    fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.2), sharey=True)
    for ax, (b, lab) in zip(axs, BASES):
        exposure_panel(ax, sc, "carry", b)
        style(ax, "Copy score for shape (nats)" if ax is axs[0] else None)
        ax.set_title(lab, fontsize=8, color=INK)
        ax.set_xlabel("Exposures in training")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "fig_a9_carry.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_knowledge(sk, ss, out):
    fig, axs = plt.subplots(1, 2, figsize=(4.4, 2.1))
    for ax, sc, m, lab in ((axs[0], sk, "know", "Knowledge (nats)"), (axs[1], ss, "stored", "Stored preference (nats)")):
        exposure_panel(ax, sc, m, "regular_grouped", show_typical=False)
        style(ax, lab)
        ax.set_xlabel("Exposures in training")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "fig_a9_knowledge.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_heads(hs, pn, out):
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.2), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axs[0]
    rows = [("regular_grouped", "unseen", "Regular\nunseen nouns"),
            ("reassigned_grouped", "unseen", "Reassigned\nunseen nouns"),
            ("reassigned_grouped", "trained", "Reassigned\ntrained kinds")]
    conds = [("carry", "Intact", "#8a8984"), ("carry_ablated", "Candidates ablated", "#2a78d6"),
             ("carry_random_mean", "Random heads ablated", "#eb6834")]
    w = 0.26
    for i, (b, who, lab) in enumerate(rows):
        d = hs[hs.base == b]
        for j, (col, clab, c) in enumerate(conds):
            v = d.groupby("seed")[f"{who}_{col}"].mean()
            a, e = ci(v)
            ax.bar(i + (j - 1) * w, a, width=w - 0.03, color=c, yerr=e, capsize=2,
                   error_kw={"elinewidth": 0.8, "ecolor": INK2}, label=clab if i == 0 else None)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[2] for r in rows], fontsize=7)
    ax.axhline(0, color=INK2, linewidth=0.5)
    ax.set_ylabel("Copy score for shape (nats)")
    ax.set_xlabel("Grouped models")
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0),
              handlelength=1.2, columnspacing=0.8)
    ax = axs[1]
    d = pn[pn.base == "regular_grouped"]
    sc = (d.groupby(["size", "seed", "type", "exposures"])["attention"].mean().reset_index()
            .groupby(["seed", "type", "exposures"])["attention"].mean().reset_index())
    sc["base"], sc["emb"] = "regular_grouped", "original"
    exposure_panel(ax, sc, "attention", "regular_grouped", show_typical=False)
    style(ax, "Attention to context shape")
    ax.set_xlabel("Exposures in training")
    ax.set_title("Stable shapes", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig_a9_heads.pdf", bbox_inches="tight")
    plt.close(fig)


def tab_cells(sc_c, sc_s, sc_k, out):
    L = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
         r"Noun type & Measure & 10 & 100 & 1,000 & 10,000 \\", r"\midrule"]
    for typ in ("consistent", "uninformative", "absent"):
        for sc, m, lab in ((sc_c, "carry", "Carry"), (sc_s, "stored", "Stored"), (sc_k, "know", "Knowledge")):
            d = sc[(sc.base == "regular_grouped") & (sc.emb == "original") & (sc.type == typ)]
            vals = [ci(d[d.exposures == e][m])[0] for e in EXPO]
            L.append(f"{NAME[typ] if m == 'carry' else ''} & {lab} & " +
                     " & ".join(f"{v:.2f}".replace("-", r"\textminus{}") for v in vals) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_a9_cells.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def fig_formation(path, out):
    """Carry against the exposures a noun had received by each saved checkpoint (exploratory).
    Exposures are spread evenly from 5% to 95% of training, so a noun with E exposures has had
    about E * (t - 0.05) / 0.9 by fraction t of training."""
    f = pd.read_csv(path)
    f = f[f.seed.isin(CONF_SEEDS) & (f.emb == "original") & (f.type.isin(["consistent", "uninformative", "absent"]))].copy()
    f["step"] = f.label.str.extract(r"(\d+)").astype(float)
    f["frac"] = f.step / f["size"].map({"tiny": 5000, "small": 8000, "medium": 10000})
    f["cum"] = f.exposures * np.clip((f.frac - 0.05) / 0.9, 0, 1)
    edges = [0.5, 3, 10, 30, 100, 300, 1000, 3000, 10001]
    mids = [np.sqrt(a * b) for a, b in zip(edges[:-1], edges[1:])]
    f["bin"] = pd.cut(f.cum, edges, labels=range(len(mids)))
    fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.1), sharey=True)
    for ax, (b, lab) in zip(axs, BASES[:2]):
        d = f[f.base == b]
        for typ in ("consistent", "uninformative", "absent"):
            g = (d[d.type == typ].groupby(["size", "seed", "bin"], observed=True).carry.mean()
                   .groupby(["seed", "bin"], observed=True).mean())
            m = [ci(g.xs(k, level="bin")) if k in g.index.get_level_values("bin") else (np.nan, 0)
                 for k in range(len(mids))]
            ax.errorbar(mids, [a for a, _ in m], yerr=[e for _, e in m], color=COL[typ], marker=MARK[typ],
                        markersize=4, capsize=2, elinewidth=0.8, markeredgecolor="white", markeredgewidth=0.6,
                        label=NAME[typ])
        ax.set_xscale("log")
        ax.axhline(0, color=INK2, linewidth=0.5)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_title(lab, fontsize=8, color=INK)
        ax.set_xlabel("Exposures so far")
    axs[0].set_ylabel("Copy score for shape (nats)")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "fig_a9_formation.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--circuits", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--formation", default=None)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = final_only(pd.concat([pd.read_csv(f) for f in a.scores]), ["base", "size", "seed", "emb", "noun"])
    s = s[s.seed.isin(CONF_SEEDS)]
    sc_c, sc_s, sc_k = seed_cells(s, "carry"), seed_cells(s, "stored"), seed_cells(s, "know")
    fig_carry(sc_c, out)
    fig_knowledge(sc_k, sc_s, out)
    tab_cells(sc_c, sc_s, sc_k, out)
    hs = final_only(pd.read_csv(Path(a.circuits) / "heads_summary.csv"), ["base", "size", "seed"])
    hs = hs[hs.seed.isin(CONF_SEEDS)]
    pn = final_only(pd.concat([pd.read_csv(f) for f in glob.glob(str(Path(a.circuits) / "per_noun_*.csv"))]),
                    ["base", "size", "seed", "noun"])
    pn = pn[pn.seed.isin(CONF_SEEDS)]
    fig_heads(hs, pn, out)
    if a.formation:
        fig_formation(a.formation, out)
    print(f"figures and table written to {out}")


if __name__ == "__main__":
    main()
