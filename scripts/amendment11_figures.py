#!/usr/bin/env python3
"""
amendment11_figures.py  |  Figure for Experiment 3 (Amendment 11): the replication in mixed corpora and the
single-history corpora.

  python scripts/amendment11_figures.py --scores results/amendment11/scores.csv --out paper_acl/figures/fig_a11.pdf

Values are means over seeds of size-averaged cell means (final models), with t-based 95% intervals over seeds.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import amendment10_figures as f10  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def pure_panel(ax, sc):
    off = {"fixed": -0.04, "rotating": 0.04}
    for h, base in (("fixed", "regular_pure_fixed"), ("rotating", "regular_pure_changing")):
        o = sc[(sc.base == base) & (sc.emb == "original")]
        for pres, ls, fill in (("run", "-", True), ("single", (0, (2, 1.5)), False)):
            m = [f10.ci(o[(o.type == f"{h}_{pres}") & (o.exposures == e)].carry) for e in f10.EXPO]
            x = np.array([0, 1]) + off[h] + (0.02 if pres == "run" else -0.02)
            ax.errorbar(x, [v for v, _ in m], yerr=[e for _, e in m], color=f10.COL[h], marker=f10.MARK[h], ls=ls,
                        markersize=4.5, capsize=2, elinewidth=0.8, mfc=f10.COL[h] if fill else "white",
                        mec=f10.COL[h], mew=0.9, zorder=3)
        u = o[o.type == "unseen"]
        a, _ = f10.ci(u.carry)
        ax.plot([1.22, 1.6], [a, a], color=f10.COL[h], lw=1.2, zorder=1)
        ax.text(1.41, a + 0.12, "unseen", color=f10.COL[h], fontsize=6.5, ha="center", va="bottom")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["100", "1,000"])
    ax.set_xlim(-0.35, 1.65)
    ax.set_xlabel("Exposures (sentences)")
    ax.set_title("Single-history corpora (5 seeds)", fontsize=8, color=f10.INK)
    ax.grid(axis="y", color=f10.GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default="results/amendment11/scores.csv")
    ap.add_argument("--scores_a10", default="results/amendment10/scores_a.csv")
    ap.add_argument("--out", default="paper_acl/figures/fig_a11.pdf")
    a = ap.parse_args()
    s = pd.read_csv(a.scores)
    s = s[s.ckpt.str.endswith("model_final.pt") & (s.type != "trained")]
    sc = f10.seed_cells(s, "carry")
    s10 = pd.read_csv(a.scores_a10)
    s10 = s10[s10.ckpt.str.endswith("model_final.pt") & (s10.type != "trained")]
    sc10 = f10.seed_cells(s10, "carry")
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.45), sharey=True, gridspec_kw={"width_ratios": [1.2, 1, 1.2]})
    f10.part_a_panel(axes[0], sc, "regular_grouped", "(a) Stable kinds, Exp. 3", legend=True)
    pure_panel(axes[1], sc)
    axes[1].set_title("(b) Single history, Exp. 3", fontsize=8, color=f10.INK)
    f10.part_a_panel(axes[2], sc10, "reassigned_grouped", "(c) Shifting kinds, Exp. 2", legend=False)
    for ax in axes:
        for tx in list(ax.texts):
            if tx.get_text() == "unseen, typical embedding":
                tx.set_text("typical emb.")
    axes[0].set_ylabel("Copy score (nats)")
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(min(lo, -0.6), hi)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=6.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(a.out, bbox_inches="tight")
    print("written", a.out)


if __name__ == "__main__":
    main()
