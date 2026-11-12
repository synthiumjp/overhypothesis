#!/usr/bin/env python3
"""
paper_numbers.py  |  Numbers quoted in the paper that no other results file reports directly. Exploratory.
Uses the registered test functions of scripts/a11_tests.py on existing outputs.

  python scripts/paper_numbers.py --out results/PAPER_NUMBERS.md
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a11_tests as t  # noqa: E402


def fmt(r):
    return f"{r['mean']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}] (n = {r['n']} seeds)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/PAPER_NUMBERS.md")
    a = ap.parse_args()
    md = ["# Numbers quoted in the paper without another results file (exploratory)", ""]
    # 1. Experiment 2: changing-burst copying beyond geometry, seed-level per-model regression (the H55 method)
    r10 = pd.read_csv("results/amendment10/reanalysis.csv")
    g = t.geometry_test(r10)
    md.append(f"- Experiment 2, beyond geometry at 1,000 (per-model regression, sizes averaged within seed): {fmt(g)}; "
              f"by size {', '.join(f'{k} {v:.2f}' for k, v in g['by_size'].items())}; pooled clustered "
              f"{g['clustered']['mean']:.2f} [{g['clustered']['ci'][0]:.2f}, {g['clustered']['ci'][1]:.2f}]")
    g100 = t.geometry_test(r10, e=100)
    md.append(f"- Experiment 2, beyond geometry at 100: {fmt(g100)}")
    # 2. Same noun minus other noun (noun_specific), changing-shape bursts
    s10 = t.final(pd.read_csv("results/amendment10/scores_a.csv"))
    for lab, e in (("pooled", [100, 1000]), ("1,000", 1000)):
        x = t.seed_level(t.per_size_seed(s10, "rotating_run", e, q="noun_specific"))
        md.append(f"- Experiment 2, changing-shape bursts, same noun minus other noun ({lab}): {fmt(t.ttest(x, 'two-sided'))}")
    s11 = t.final(pd.read_csv("results/amendment11/scores.csv"))
    for base in ("regular_grouped", "regular_pure_changing"):
        for lab, e in (("pooled", [100, 1000]), ("1,000", 1000)):
            x = t.seed_level(t.per_size_seed(s11, "rotating_run", e, q="noun_specific", base=base))
            md.append(f"- Experiment 3 {base}, changing-shape bursts, same noun minus other noun ({lab}): "
                      f"{fmt(t.ttest(x, 'two-sided'))}")
    Path(a.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
