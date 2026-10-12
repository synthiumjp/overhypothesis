#!/usr/bin/env python3
"""
rescore_study1.py  |  Study 1 analyses on the corrected scores (results/rescore/).

Scores before commit 483ad6b used the output matrix as the input embedding (see
docs/RUN_RESCORE.md). This script rebuilds, from results/rescore/raw/, the two Study 1
outputs of scripts/exploratory_analyses.py that depend on model scores (checkpoint
tracking; kind-specific index by dimension), with the same code. The registered tests
are rerun with the locked scripts themselves (commands in results/rescore/RESULTS.md).

Usage:  python scripts/rescore_study1.py --dir results/rescore
"""
import argparse

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results/rescore")
a = ap.parse_args()
R = a.dir

# Checkpoint tracking (as exploratory_analyses.checkpoint_tracking)
d = pd.concat([pd.read_csv(f"{R}/raw/track_regular.csv.gz"), pd.read_csv(f"{R}/raw/track_bursty.csv.gz")])
d["step"] = d.ckpt.str.extract(r"ckpt_step(\d+)", expand=False).astype(int)
d["match"] = d["match"].astype(str).isin(["True", "true"])
w = d.groupby(["condition", "size", "seed", "step", "match", "probe_dim"]).delta.mean().unstack("probe_dim")
w["OI_colour"] = w["shape"] - w["colour"]
w["OI_texture"] = w["shape"] - w["texture"]
w = w.reset_index()
m = w[w.match].set_index(["condition", "size", "seed", "step"])
x = w[~w.match].set_index(["condition", "size", "seed", "step"])
per_run = m[["OI_colour", "OI_texture"]].copy()
per_run["kind_specific_shape_minus_colour"] = m["OI_colour"] - x["OI_colour"]
per_run = per_run.reset_index()
per_run.round(5).to_csv(f"{R}/tracking_per_run.csv", index=False)
traj = per_run.groupby(["condition", "size", "step"]).agg(
    n_seeds=("seed", "nunique"), OI_colour=("OI_colour", "mean"), OI_colour_sd=("OI_colour", "std"),
    OI_texture=("OI_texture", "mean"),
    kind_specific=("kind_specific_shape_minus_colour", "mean")).reset_index()
traj.round(4).to_csv(f"{R}/tracking_trajectory.csv", index=False)

# Kind-specific index by dimension (as exploratory_analyses.kind_specific_synthetic)
s = pd.concat([pd.read_csv(f"{R}/study1bc_per_run.csv"), pd.read_csv(f"{R}/study1x_per_run.csv"),
               pd.read_csv(f"{R}/study1w_per_run.csv")])
s["match"] = s["match"].astype(str).isin(["True", "true"])
s = s[(s.source == "lm") & (s.k == 1)]
p = s.pivot_table(index=["condition", "size", "seed"], columns="match", values=["shape", "colour", "texture"])
ks = pd.DataFrame({dim: p[(dim, True)] - p[(dim, False)] for dim in ("shape", "colour", "texture")})
ks["shape_minus_colour"] = ks["shape"] - ks["colour"]
ks["shape_minus_texture"] = ks["shape"] - ks["texture"]
ks = ks.reset_index()
ks.round(5).to_csv(f"{R}/kind_specific_by_dim_per_run.csv", index=False)
summ = ks.groupby("condition")[["shape", "colour", "texture", "shape_minus_colour",
                                 "shape_minus_texture"]].agg(["mean", "std"])
summ.columns = ["_".join(c) for c in summ.columns]
summ.round(4).to_csv(f"{R}/kind_specific_by_dim.csv")

# When the grouped models reach their level (first step within 90% of final, per size)
fin = traj[traj.step == traj.groupby(["condition", "size"]).step.transform("max")].set_index(["condition", "size"]).OI_colour
traj["frac_final"] = traj.apply(lambda r: r.OI_colour / fin[(r.condition, r["size"])], axis=1)
print(traj[traj.condition == "regular_bursty"].pivot_table(index="step", columns="size", values="OI_colour").round(2).iloc[::4])
print(traj[traj.condition == "regular"].pivot_table(index="step", columns="size", values="OI_colour").round(2).iloc[::4])
print("kind-specific range over all checkpoints:", traj.kind_specific.min().round(3), traj.kind_specific.max().round(3))
print(summ.round(3))
