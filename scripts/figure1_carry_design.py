#!/usr/bin/env python3
"""
figure1_carry_design.py  |  Figure 1 of the paper: the carry test and the Experiment 2 histories.

  python scripts/figure1_carry_design.py --out paper_acl/figures/figure1_carry.pdf

Schematic only (no data). Top: a test item and how carry is read. Bottom: four training
histories drawn as successive 128-token windows; grey cells are sentences about trained kinds,
coloured cells are sentences about one familiarity noun, labelled with the shape they name.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#d9d8d4", "#f3f2ef"
SHAPE_COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#8a5cc2", "-": "white"}
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "pdf.fonttype": 42, "mathtext.fontset": "stix"})


def window(ax, x, y, w, h, cells, ncell=6):
    ax.add_patch(Rectangle((x, y), w, h, fc=BG, ec=INK2, lw=0.5))
    cw = w / ncell
    for i in range(ncell):
        lab = cells.get(i)
        if lab is None:
            ax.add_patch(Rectangle((x + i * cw + 0.004, y + 0.012), cw - 0.008, h - 0.024, fc=GRID, ec="none"))
        else:
            fc = SHAPE_COL[lab]
            ax.add_patch(Rectangle((x + i * cw + 0.004, y + 0.012), cw - 0.008, h - 0.024, fc=fc,
                                   ec=INK if lab == "-" else "none", lw=0.6))
            ax.text(x + (i + 0.5) * cw, y + h / 2, "no\nshape" if lab == "-" else lab, ha="center", va="center",
                    fontsize=4.2 if lab == "-" else 6.5, color=INK if lab == "-" else "white", fontweight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper_acl/figures/figure1_carry.pdf")
    a = ap.parse_args()
    fig = plt.figure(figsize=(3.3, 3.05))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0.03, 1)
    ax.axis("off")

    # (a) test item
    ax.text(0.01, 0.975, "(a) Test item", fontsize=8, fontweight="bold", va="top")
    ax.add_patch(FancyBboxPatch((0.02, 0.775), 0.96, 0.15, boxstyle="round,pad=0.008", fc=BG, ec=INK2, lw=0.5))
    ax.text(0.04, 0.895, "Context:", color=INK2, va="center")
    ax.text(0.20, 0.895, r"A $\it{dax}$ is a $\bf{zibby}$ lavo shorfi thing.", va="center")
    ax.text(0.04, 0.848, "Query:", color=INK2, va="center")
    ax.text(0.20, 0.848, r"A $\it{dax}$ is a  ___", va="center")
    ax.text(0.04, 0.800, r"Carry $= L(\mathrm{zibby})\, -$ mean $L$ of other shapes (nats)", va="center",
            fontsize=6.8, color=INK)
    ax.text(0.02, 0.745, r"Control: the query names another noun (A $\it{blick}$ is a ___).",
            fontsize=6.3, color=INK2, va="center")

    # (b) histories
    ax.text(0.01, 0.705, "(b) Experiment 2: four histories of one noun", fontsize=8, fontweight="bold", va="top")
    rows = [("Fixed, singles", [{2: "A"}, {4: "A"}, {1: "A"}]),
            ("Rotating, singles", [{2: "A"}, {4: "C"}, {1: "B"}]),
            ("Fixed, runs", [{1: "A", 2: "A", 3: "A", 4: "A"}, {}, {0: "A", 1: "A", 2: "A", 3: "A"}]),
            ("Rotating, runs", [{1: "B", 2: "B", 3: "B", 4: "B"}, {}, {0: "D", 1: "D", 2: "D", 3: "D"}])]
    y0, dy, h = 0.585, 0.128, 0.085
    for k, (lab, wins) in enumerate(rows):
        y = y0 - k * dy
        ax.text(0.02, y + h / 2, lab, va="center", fontsize=7)
        for j, cells in enumerate(wins):
            window(ax, 0.31 + j * 0.232, y, 0.21, h, cells)
    ax.text(0.02, 0.165, "Boxes: successive 128-token windows. Grey: sentences about trained\n"
            "kinds. Letters: the shape each sentence about the noun names.\n"
            "Shape-absent nouns (not shown) name none.", fontsize=6.2, color=INK2,
            va="top", linespacing=1.3)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.02)
    print("written", a.out)


if __name__ == "__main__":
    main()
