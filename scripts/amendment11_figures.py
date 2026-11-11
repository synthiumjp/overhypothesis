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


def control_panel(ax, sc):
    """Redrawn-burst control: copy score against how often a burst sentence repeats the previous one's shape."""
    q = 0.102
    rate = {"redrawn_run": q, "half_run": 0.5 + 0.5 * q, "rotating_run": 1.0}
    o = sc[sc.emb == "original"]
    for e, fill, ls in ((100, False, (0, (2, 1.5))), (1000, True, "-")):
        xs, ms, es = [], [], []
        for t in ("redrawn_run", "half_run", "rotating_run"):
            m, h = f10.ci(o[(o.type == t) & (o.exposures == e)].carry)
            xs.append(rate[t] + (-0.012 if e == 100 else 0.012))
            ms.append(m)
            es.append(h)
        ax.errorbar(xs, ms, yerr=es, color=f10.COL["rotating"], marker="s", ls=ls, markersize=4.5, capsize=2,
                    elinewidth=0.8, mfc=f10.COL["rotating"] if fill else "white", mec=f10.COL["rotating"], mew=0.9,
                    zorder=3, label=f"{e:,} exposures")
    for t, col, lab in (("rotating_single", f10.COL["rotating"], "changing, spaced, 1,000"),
                        ("fixed_run", f10.COL["fixed"], "fixed, bursts, 1,000")):
        m, _ = f10.ci(o[(o.type == t) & (o.exposures == 1000)].carry)
        ax.axhline(m, color=col, lw=0.9, ls=(0, (1, 1.2)), zorder=1, label=lab)
    ax.set_xticks([q, 0.5 + 0.5 * q, 1.0])
    ax.set_xticklabels(["redrawn\n0.10", "half-kept\n0.55", "changing\n1.00"])
    ax.set_xlim(-0.05, 1.12)
    ax.set_xlabel("Share of burst sentences\nrepeating the last shape")
    ax.legend(frameon=False, fontsize=5.8, loc="upper left", handletextpad=0.3, handlelength=1.8,
              borderaxespad=0.2, labelspacing=0.25)
    ax.grid(axis="y", color=f10.GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default="results/amendment11/scores.csv")
    ap.add_argument("--scores_a10", default="results/amendment10/scores_a.csv")
    ap.add_argument("--scores_ctl", default="results/control_redrawn/scores.csv")
    ap.add_argument("--out", default="paper_acl/figures/fig_a11.pdf")
    a = ap.parse_args()
    s = pd.read_csv(a.scores)
    s = s[s.ckpt.str.endswith("model_final.pt") & (s.type != "trained")]
    sc = f10.seed_cells(s, "carry")
    s10 = pd.read_csv(a.scores_a10)
    s10 = s10[s10.ckpt.str.endswith("model_final.pt") & (s10.type != "trained")]
    sc10 = f10.seed_cells(s10, "carry")
    sct = pd.read_csv(a.scores_ctl)
    sct = sct[sct.ckpt.str.endswith("model_final.pt") & (sct.type != "trained")]
    scc = f10.seed_cells(sct, "carry")
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.5), sharey=True,
                             gridspec_kw={"width_ratios": [1.15, 1.05, 0.95, 1.15]})
    f10.part_a_panel(axes[0], sc, "regular_grouped", "(a) Stable kinds, Exp. 3", legend=True)
    control_panel(axes[1], scc)
    axes[1].set_title("(b) Control", fontsize=8, color=f10.INK)
    pure_panel(axes[2], sc)
    axes[2].set_title("(c) Single history", fontsize=8, color=f10.INK)
    f10.part_a_panel(axes[3], sc10, "reassigned_grouped", "(d) Shifting kinds, Exp. 2", legend=False)
    for ax in axes:
        for tx in list(ax.texts):
            if tx.get_text() == "unseen, typical embedding":
                tx.set_text("typical emb.")
    axes[0].set_ylabel("Copy score (nats)")
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(min(lo, -0.6), hi)
    for ax in axes:
        ax.tick_params(labelsize=6.5)
        ax.xaxis.label.set_size(7)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=6.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(a.out, bbox_inches="tight")
    print("written", a.out)


if __name__ == "__main__":
    main()
