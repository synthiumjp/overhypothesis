#!/usr/bin/env python3
"""
amendment10_figures.py  |  Figures and a cell table for the Amendment 10 results.

  python scripts/amendment10_figures.py --scores_a results/amendment10/scores_a.csv \
      --pythia results/amendment10/pythia_scores.csv --checkpoints results/amendment10/pythia_checkpoints.csv \
      --out_dir paper_acl/figures

Part A values are means over the ten seeds of size-averaged cell means (final models only), with
t-based 95% intervals over seeds, as in the registered tests. Part B values are means over the 44
nouns with t-based 95% intervals over nouns.

Writes fig_a10.pdf (Part A, both bases, and Part B) and tab_a10_cells.tex.
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

COL = {"fixed": "#2a78d6", "rotating": "#eb6834", "absent": "#1baf7a"}
MARK = {"fixed": "o", "rotating": "s", "absent": "^"}
NAME = {"fixed": "Fixed shape", "rotating": "Rotating shape", "absent": "Shape-absent"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BASES = [("regular_grouped", "Regular, grouped (confirmatory)"), ("reassigned_grouped", "Reassigned, grouped (exploratory)")]
EXPO = [100, 1000]
MODELS = ["70m", "160m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
COND = [("original", "Original row", INK, "o"), ("atypical_init", "Random row, initial norm", "#2a78d6", "s"),
        ("typical", "Typical row", "#eb6834", "^")]

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "lines.linewidth": 1.4,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (x.mean() if len(x) else np.nan), 0.0
    return x.mean(), stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def seed_cells(s, measure):
    d = s[s[measure].notna()]
    cm = d.groupby(["base", "emb", "size", "seed", "type", "exposures"])[measure].mean().reset_index()
    return cm.groupby(["base", "emb", "seed", "type", "exposures"])[measure].mean().reset_index()


def part_a_panel(ax, sc, base, title, legend):
    d = sc[sc.base == base]
    o = d[d.emb == "original"]
    a, b = ci(o[o.type == "unseen"].carry)
    ax.axhspan(a - b, a + b, color=GRID, alpha=0.8, lw=0, zorder=0)
    ax.axhline(a, color=INK2, lw=0.8, zorder=1)
    ax.text(1.62, a + 0.12, "unseen", color=INK2, fontsize=6.5, ha="right", va="bottom")
    t = d[(d.emb == "typical") & (d.type == "unseen")]
    a, _ = ci(t.carry)
    ax.axhline(a, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(-0.32, a - 0.15, "unseen, typical row", color=INK2, fontsize=6.5, ha="left", va="top")
    off = {"fixed": -0.06, "absent": 0.0, "rotating": 0.06}
    for h in ("fixed", "absent", "rotating"):
        for pres, ls, fill in (("run", "-", True), ("single", (0, (2, 1.5)), False)):
            m = [ci(o[(o.type == f"{h}_{pres}") & (o.exposures == e)].carry) for e in EXPO]
            x = np.array([0, 1]) + off[h] + (0.02 if pres == "run" else -0.02)
            ax.errorbar(x, [v for v, _ in m], yerr=[e for _, e in m], color=COL[h], marker=MARK[h], ls=ls,
                        markersize=4.5, capsize=2, elinewidth=0.8, mfc=COL[h] if fill else "white",
                        mec=COL[h], mew=0.9, zorder=3, label=f"{NAME[h]}, {pres}s" if legend else None)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["100", "1,000"])
    ax.set_xlim(-0.35, 1.65)
    ax.set_xlabel("Exposures (sentences)")
    ax.set_title(title, fontsize=8, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def part_b_panel(ax, p):
    xs = np.arange(len(MODELS))
    for k, (cond, lab, col, mk) in enumerate(COND):
        m = [ci(p[(p.size_lab == z) & (p.condition == cond)].carry) for z in MODELS]
        ax.errorbar(xs + (k - 1) * 0.16, [v for v, _ in m], yerr=[e for _, e in m], color=col, marker=mk, ls="none",
                    markersize=4, capsize=1.5, elinewidth=0.8, mec="white", mew=0.5, label=lab, zorder=3)
    ax.axvspan(1.5, 6.5, color=GRID, alpha=0.35, lw=0, zorder=0)
    ax.text(4, 3.25, "primary models", color=INK2, fontsize=6.5, ha="center")
    ax.set_xticks(xs)
    ax.set_xticklabels([z.upper() for z in MODELS], fontsize=6.5)
    ax.set_xlabel("Pythia (deduplicated)")
    ax.set_ylabel("Carry of the context's colour (nats)")
    ax.set_ylim(0, 3.4)
    ax.set_title("Pretrained models, real nouns (Part B)", fontsize=8, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right", handletextpad=0.3, borderaxespad=0.2)


def cell_table(sc, out):
    rows = []
    for base, lab in BASES:
        d = sc[(sc.base == base) & (sc.emb == "original")]
        for h in ("fixed", "absent", "rotating"):
            for pres in ("single", "run"):
                vals = []
                for e in EXPO:
                    a, b = ci(d[(d.type == f"{h}_{pres}") & (d.exposures == e)].carry)
                    vals.append(f"{a:.2f} [{a - b:.2f}, {a + b:.2f}]")
                rows.append((lab.split(" (")[0], f"{NAME[h]}, {pres}s", *vals))
        a, b = ci(d[d.type == "unseen"].carry)
        rows.append((lab.split(" (")[0], "Unseen", f"{a:.2f} [{a - b:.2f}, {a + b:.2f}]", ""))
        t = sc[(sc.base == base) & (sc.emb == "typical") & (sc.type == "unseen")]
        a, b = ci(t.carry)
        rows.append((lab.split(" (")[0], "Unseen, typical row", f"{a:.2f} [{a - b:.2f}, {a + b:.2f}]", ""))
    lines = [r"\begin{tabular}{@{}llll@{}}", r"\toprule", r"Base & Noun & 100 exposures & 1,000 exposures \\",
             r"\midrule"]
    last = None
    for r in rows:
        if last is not None and r[0] != last:
            lines.append(r"\midrule")
        lines.append(" & ".join([r[0] if r[0] != last else ""] + list(r[1:])).replace("[", "{[}").replace("]", "{]}"))
        lines[-1] = re.sub(r"(?<![A-Za-z])-(?=\d)", r"\\textminus{}", lines[-1]) + r" \\"
        last = r[0]
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores_a", default="results/amendment10/scores_a.csv")
    ap.add_argument("--pythia", default="results/amendment10/pythia_scores.csv")
    ap.add_argument("--out_dir", default="paper_acl/figures")
    a = ap.parse_args()
    out = Path(a.out_dir)
    s = pd.read_csv(a.scores_a)
    s = s[s.ckpt.str.endswith("model_final.pt")]
    sc = seed_cells(s, "carry")
    p = pd.read_csv(a.pythia)
    p["size_lab"] = p.model.str.extract(r"pythia-([0-9.]+[mb])-")[0]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.45), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    lo = min(-0.6, sc[sc.emb == "original"].groupby(["base", "type", "exposures"]).carry.mean().min() - 0.6)
    hi = sc[sc.emb == "original"].groupby(["base", "type", "exposures"]).carry.mean().max() + 1.2
    for k, (base, title) in enumerate(BASES):
        part_a_panel(axs[k], sc, base, title, legend=(k == 0))
        axs[k].set_ylim(lo, hi)
    axs[0].set_ylabel("Carry of the context's shape (nats)")
    part_b_panel(axs[2], p)
    h, l = axs[0].get_legend_handles_labels()
    order = [0, 1, 2, 3, 4, 5]
    fig.legend([h[i] for i in order], [l[i] for i in order], frameon=False, fontsize=6.5, ncol=3, loc="lower center",
               bbox_to_anchor=(0.31, -0.02), handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.11, 1, 1), w_pad=1.0)
    fig.savefig(out / "fig_a10.pdf", bbox_inches="tight")
    cell_table(sc, out / "tab_a10_cells.tex")
    print("written", out / "fig_a10.pdf", out / "tab_a10_cells.tex")


if __name__ == "__main__":
    main()
