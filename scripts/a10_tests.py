#!/usr/bin/env python3
"""
a10_tests.py  |  Registered tests of Amendment 10 (docs/AMENDMENT10.md).

  python scripts/a10_tests.py partA --scores results/amendment10/scores_a.csv --out results/amendment10/RESULTS_A.md
  python scripts/a10_tests.py partB --scores results/amendment10/pythia_scores.csv --pilot results/amendment10/pythia_pilot.csv --out results/amendment10/RESULTS_B.md
  python scripts/a10_tests.py partC --rowswap results/amendment10/rowswap.csv --out results/amendment10/RESULTS_C.md
  python scripts/a10_tests.py pilotB --scores results/amendment10/pythia_pilot.csv   (prints the smallest effect of interest)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_familiarity_corpora import CONF_SEEDS  # noqa: E402

ALPHA = 0.05
SESOI_A = 0.3
PRIMARY = ["pythia-1b-deduped", "pythia-1.4b-deduped", "pythia-2.8b-deduped", "pythia-6.9b-deduped",
           "pythia-12b-deduped"]


def t1(x, side):
    x = np.asarray(x, float)
    r = stats.ttest_1samp(x, 0.0, alternative=side)
    m, n = x.mean(), len(x)
    h = stats.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return {"mean": float(m), "ci": [float(m - h), float(m + h)], "t": float(r.statistic), "df": n - 1,
            "p": float(r.pvalue), "n": n}


def tost(x, b):
    x = np.asarray(x, float)
    return float(max(stats.ttest_1samp(x, -b, alternative="greater").pvalue,
                     stats.ttest_1samp(x, b, alternative="less").pvalue))


def holm(res, keys):
    fam = sorted(((k, res[k]["p"]) for k in keys), key=lambda x: x[1])
    stop = False
    for i, (k, p) in enumerate(fam):
        thr = ALPHA / (len(fam) - i)
        ok = (not stop) and p < thr
        stop = stop or not ok
        res[k]["holm_threshold"], res[k]["holm_reject"] = thr, bool(ok)


def fmt(r):
    return f"{r['mean']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}], t({r['df']}) = {r['t']:.2f}, p = {r['p']:.3g}"


def final_only(df, keys):
    df = df[df.ckpt.astype(str).str.endswith("model_final.pt")].copy()
    if df.duplicated(keys).any():
        sys.exit("duplicate rows")
    return df


# ---------------------------------------------------------------- Part A
def seed_val(s, base, typ, expos=(100, 1000), emb="original"):
    d = s[(s.base == base) & (s.type == typ) & (s.emb == emb)]
    if expos is not None:
        d = d[d.exposures.isin(expos)]
    per_model = d.groupby(["size", "seed", "exposures"]).carry.mean().groupby(["size", "seed"]).mean()
    return per_model


def contrast(a, b):
    d = (a - b).dropna()
    return d.groupby("seed").mean(), d.groupby("size").mean()


def part_a(a):
    s = final_only(pd.concat([pd.read_csv(f) for f in a.scores]), ["base", "size", "seed", "emb", "noun"])
    s = s[s.seed.isin(CONF_SEEDS)]
    B = "regular_grouped"
    res = {}
    un = seed_val(s, B, "unseen", None)
    res["M2"] = {**t1(un.groupby("seed").mean() - 1.0, "greater"), "note": "unseen carry minus 1 nat"}
    specs = {"H40": (("absent_run", "absent_single"), "two-sided"),
             "H41": (("fixed_run", "fixed_single"), "two-sided"),
             "H42": (("fixed_single", "absent_single"), "less")}
    for h, ((x, y), side) in specs.items():
        seeds, sizes = contrast(seed_val(s, B, x), seed_val(s, B, y))
        r = t1(seeds, side)
        r["sizes_in_direction"] = int((np.sign(sizes) == (np.sign(r["mean"]) if side == "two-sided" else -1)).sum())
        if side == "two-sided":
            r["tost_p"] = tost(seeds, SESOI_A)
            r["equivalent"] = bool(r["tost_p"] < ALPHA)
        res[h] = r
    ty = seed_val(s, B, "unseen", None, emb="typical")
    seeds, sizes = contrast(seed_val(s, B, "absent_single"), ty)
    r = t1(seeds, "two-sided")
    r.update({"sizes_in_direction": int((np.sign(sizes) == np.sign(r["mean"])).sum()), "tost_p": tost(seeds, SESOI_A)})
    r["equivalent"] = bool(r["tost_p"] < ALPHA)
    res["H43"] = r
    holm(res, ["H40", "H41", "H42", "H43"])
    for h in ("H40", "H41", "H42", "H43"):
        res[h]["supported"] = bool(res[h]["holm_reject"] and res[h]["sizes_in_direction"] >= 2)
    # exploratory: the same contrasts in Reassigned grouped models
    expl = {}
    for h, ((x, y), side) in specs.items():
        seeds, _ = contrast(seed_val(s, "reassigned_grouped", x), seed_val(s, "reassigned_grouped", y))
        if len(seeds) > 1:
            expl[h] = t1(seeds, "two-sided")
    res["exploratory_reassigned"] = expl
    L = ["# Amendment 10 Part A results", "", f"M2 (unseen carry above 1 nat): {fmt(res['M2'])}", ""]
    for h in ("H40", "H41", "H42", "H43"):
        r = res[h]
        L.append(f"{h}: {fmt(r)}; Holm threshold {r['holm_threshold']:.4f} "
                 f"({'met' if r['holm_reject'] else 'not met'}); sizes in direction {r['sizes_in_direction']} of 3; "
                 f"{'SUPPORTED' if r['supported'] else 'not supported'}" +
                 (f"; TOST(+-{SESOI_A}) p = {r['tost_p']:.3g}, equivalent: {r['equivalent']}" if "tost_p" in r else ""))
    L += ["", "Exploratory, Reassigned grouped:"] + [f"- {h}: {fmt(r)}" for h, r in expl.items()]
    write(a.out, res, L)


# ---------------------------------------------------------------- Part B
def sesoi_b(pilot_path):
    p = pd.read_csv(pilot_path)
    o = p[p.condition == "original"].carry.mean()
    return max(0.3, 0.2 * abs(o)), float(o)


def part_b(a):
    s = pd.concat([pd.read_csv(f) for f in a.scores])
    s["short"] = s.model.str.split("/").str[-1]
    s = s[s.revision == "step143000"]
    ses, pilot_mean = sesoi_b(a.pilot)
    res = {"sesoi": ses, "pilot_original_carry": pilot_mean, "models": {}}
    for m in PRIMARY:
        d = s[s.short == m]
        if d.empty:
            continue
        w = d.pivot_table(index=["noun", "group", "pair"], columns="condition", values="carry").reset_index()
        r = {"H44": t1(w.atypical - w.original, "greater"), "H45": t1(w.atypical - w.typical, "greater")}
        pr = w.pivot_table(index="pair", columns="group", values="original")
        r["H46"] = t1(pr.typical - pr.variable, "less")
        inc = (w.assign(inc=w.atypical - w.original).pivot_table(index="pair", columns="group", values="inc"))
        r["H47"] = t1(inc.typical - inc.variable, "two-sided")
        r["H47"]["tost_p"] = tost(inc.typical - inc.variable, ses)
        r["H47"]["equivalent"] = bool(r["H47"]["tost_p"] < ALPHA)
        holm(r, ["H44", "H45", "H46", "H47"])
        res["models"][m] = r
    for h in ("H44", "H45", "H46", "H47"):
        n = sum(res["models"][m][h]["holm_reject"] for m in res["models"])
        res[h] = {"models_holm": n, "of": len(res["models"]), "supported": n >= 4}
    res["H47"]["models_equivalent"] = sum(res["models"][m]["H47"]["equivalent"] for m in res["models"])
    L = ["# Amendment 10 Part B results", "", f"Smallest effect of interest (H47): {ses:.3f} nats "
         f"(pilot original carry {pilot_mean:.3f})", ""]
    for m, r in res["models"].items():
        L.append(f"## {m}")
        for h in ("H44", "H45", "H46", "H47"):
            L.append(f"- {h}: {fmt(r[h])}; Holm {'met' if r[h]['holm_reject'] else 'not met'}" +
                     (f"; TOST p = {r[h]['tost_p']:.3g}" if h == "H47" else ""))
    L += [""] + [f"{h}: Holm met in {res[h]['models_holm']} of {res[h]['of']} primary models; "
                 f"{'SUPPORTED' if res[h]['supported'] else 'not supported'}" for h in ("H44", "H45", "H46", "H47")]
    L.append(f"H47 equivalence in {res['H47']['models_equivalent']} of {res['H47']['of']} models")
    write(a.out, res, L)


def pilot_b(a):
    ses, o = sesoi_b(a.scores)
    print(f"pilot mean original carry {o:.3f}; smallest effect of interest for H47: {ses:.3f} nats")


# ---------------------------------------------------------------- Part C
def part_c(a):
    r = final_only(pd.read_csv(a.rowswap), ["base", "size", "seed", "noun", "condition"])
    r = r[r.seed.isin(CONF_SEEDS)]
    res = {}
    for b in ("regular_grouped", "reassigned_grouped"):
        d = r[(r.base == b) & (r.type == "unseen")]
        w = d.groupby(["size", "seed", "condition"]).carry.mean().unstack("condition")
        diff = (w.randnorm - w.own)
        seeds, sizes = diff.groupby("seed").mean(), diff.groupby("size").mean()
        key = "H48" if b == "regular_grouped" else "H48_reassigned_exploratory"
        res[key] = {**t1(seeds, "less"), "sizes_in_direction": int((sizes < 0).sum())}
        res[f"means_{b}"] = {c: float(w[c].groupby("seed").mean().mean()) for c in w.columns}
        da = r[(r.base == b) & (r.type == "absent")].groupby(["size", "seed", "condition"]).carry.mean().unstack()
        res[f"means_absent_{b}"] = {c: float(da[c].groupby("seed").mean().mean()) for c in da.columns}
    res["H48"]["supported"] = bool(res["H48"]["p"] < ALPHA and res["H48"]["sizes_in_direction"] >= 2)
    L = ["# Amendment 10 Part C results", "", f"H48 (random direction at trained norm minus own row, unseen, grouped "
         f"Regular): {fmt(res['H48'])}; sizes {res['H48']['sizes_in_direction']} of 3; "
         f"{'SUPPORTED' if res['H48']['supported'] else 'not supported'}", "",
         "Means (exploratory):"] + [f"- {k}: " + ", ".join(f"{c} {v:.2f}" for c, v in res[k].items())
                                    for k in res if k.startswith("means")]
    write(a.out, res, L)


def write(out, res, L):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(os.path.splitext(out)[0] + ".json", "w"), indent=1, default=float)
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("partA")
    pa.add_argument("--scores", nargs="+", required=True)
    pa.add_argument("--out", required=True)
    pb = sub.add_parser("partB")
    pb.add_argument("--scores", nargs="+", required=True)
    pb.add_argument("--pilot", required=True)
    pb.add_argument("--out", required=True)
    pc = sub.add_parser("partC")
    pc.add_argument("--rowswap", required=True)
    pc.add_argument("--out", required=True)
    pp = sub.add_parser("pilotB")
    pp.add_argument("--scores", required=True)
    a = ap.parse_args()
    {"partA": part_a, "partB": part_b, "partC": part_c, "pilotB": pilot_b}[a.cmd](a)
