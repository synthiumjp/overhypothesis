#!/usr/bin/env python3
"""
study2x_secondary.py  |  Secondary analyses of the Study 2 extension (Amendment 5,
S15 to S17), from results/study2x/tests_summary.csv (written by
study2_extension.py analyse).

  secondary_onset.csv       S15: per Pythia size, first checkpoint from which the 95%
                            interval of the kind-specific index (and of the raw index)
                            stays above zero at every later scored checkpoint (set M)
  secondary_trajectory_M.csv  kind-specific index on set M at every scored checkpoint
  secondary_sets_final.csv  S16 and S17: every set (M, M2, F, W), final checkpoints,
                            kind-specific index, raw index, texture index and the
                            components (shape/colour, same/different noun)
  secondary_poly_seeds.csv  PolyPythia: mean, min, max and number of positive seeds per
                            size and checkpoint (set M)

Usage:  python scripts/study2x_secondary.py --summary results/study2x/tests_summary.csv \
            --out results/study2x/secondary
"""
import argparse

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--summary", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

s = pd.read_csv(a.summary)
s["base"] = s.model.str.split("@").str[0]
s["step"] = pd.to_numeric(s.model.str.split("@step").str[1], errors="coerce")
orig = s[(s.family == "pythia") & ~s.base.str.contains("seed")]

# S15 onset (set M)
rows = []
for base, g in orig[orig.set == "M"].groupby("base"):
    g = g.sort_values("step")
    for col in ("ks_colour", "raw_colour"):
        lo, steps = g[col + "_lo"].to_numpy(), g.step.to_numpy()
        onset = next((int(steps[i]) for i in range(len(steps)) if (lo[i:] > 0).all()), None)
        rows.append({"model": base, "measure": col, "checkpoints": len(g), "onset_step": onset})
pd.DataFrame(rows).to_csv(f"{a.out}_onset.csv", index=False)

# Trajectory on set M
traj = orig[orig.set == "M"].sort_values(["base", "step"])[
    ["base", "step", "ks_colour", "ks_colour_lo", "ks_colour_hi", "raw_colour", "raw_colour_lo", "raw_colour_hi"]]
traj.round(4).to_csv(f"{a.out}_trajectory_M.csv", index=False)

# Final checkpoints, every set (S16, S17)
fin = s[((s.step == 143000) & ~s.base.str.contains("seed")) | s.family.isin(["babylm", "large"])]
cols = ["base", "set", "params", "ks_colour", "ks_colour_lo", "ks_colour_hi", "raw_colour", "raw_colour_lo",
        "raw_colour_hi", "ks_texture", "shape_same", "shape_diff", "colour_same", "colour_diff",
        "shape_kind", "colour_kind"]
fin.sort_values(["set", "params"])[cols].round(4).to_csv(f"{a.out}_sets_final.csv", index=False)

# PolyPythia seeds (set M)
poly = s[(s.set == "M") & s.base.str.contains("seed")].copy()
poly["size"] = poly.base.str.extract(r"pythia-(\d+m)")[0]
agg = poly.groupby(["size", "step"]).ks_colour.agg(
    mean="mean", min="min", max="max", seeds="count", positive=lambda x: int((x > 0).sum()))
agg.round(4).to_csv(f"{a.out}_poly_seeds.csv")
print(pd.DataFrame(rows).to_string(index=False))
