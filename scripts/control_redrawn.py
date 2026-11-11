#!/usr/bin/env python3
"""
control_redrawn.py  |  Redrawn-burst control (not pre-registered; design and predictions fixed in docs/CONTROL_REDRAWN.md
before any model was trained). Added after the results in answer to a review.

Question. Changing-shape bursts keep copying and fixed-shape bursts lose it. Is that because a burst makes the previous
sentence a useful source of the shape, or only because the noun recurs? Fixed-shape bursts cannot tell: they also give
memory a shape to store. This design adds bursts in which the noun recurs but its shape is drawn anew for every sentence,
so the previous sentence is no guide (redrawn bursts), and bursts in which each sentence keeps the previous shape half the
time (half-kept bursts).

Design (base text, frames, placement, tokenizer, nouns, seeds and model sizes exactly as the Experiment 3 mixed design;
scripts/make_familiarity_corpora_a10.py build_one is used unchanged). Cells, three nouns each, at 100 and 1,000
exposures:
  fixed_single     fixed shape, spaced                       (as Exp. 3)
  fixed_run        fixed shape, bursts of four               (as Exp. 3)
  rotating_single  shape drawn anew for every sentence, spaced   (as Exp. 3 "changing, spaced")
  rotating_run     one shape per burst, changing between bursts  (as Exp. 3 "changing, bursts")
  redrawn_run      shape drawn anew for every sentence, in bursts of four (the same sentences as rotating_single,
                   only grouped)
  half_run         first sentence of a burst drawn anew; each later sentence keeps the previous shape with
                   probability 0.5, else is drawn anew
plus four unseen nouns: 40 nouns, 18 with 100 exposures and 18 with 1,000, as in Exp. 3. Shapes are drawn from the
corpus's shape distribution. The share of sentences whose shape repeats the previous sentence in the burst is 1 for
rotating_run, about 0.5 + 0.5 q for half_run and about q for redrawn_run, where q is the chance that two independent
draws agree (reported by `check`).

Subcommands (from the repository root):
  assign                              write docs/redrawn_assignment.json
  build   --out_dir DIR [--seeds ...] [--windows ...]
  check   --out_dir DIR [--against docs/redrawn_corpus_manifest.json]
  corpus_manifest --out_dir DIR       write docs/redrawn_corpus_manifest.json
  manifest --ckpt_dir D --corpus_dir C --out results/control_redrawn/manifest.csv
  score   --manifest M --out S        familiarity_eval.py on these models (same items and measures as Exp. 3)
  summary --scores S --out MD [--tex T]
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus_generator as cg  # noqa: E402
import make_familiarity_corpora as m9  # noqa: E402
import make_familiarity_corpora_a10 as m10  # noqa: E402
import make_familiarity_corpora_a11 as m11  # noqa: E402

BASE = "regular_redrawn"
DESIGN = "redrawn"
SEEDS = m11.SEEDS_A
TYPES = [("fixed", "single"), ("fixed", "run"), ("rotating", "single"), ("rotating", "run"), ("redrawn", "run"),
         ("half", "run")]
PER_CELL = 3
P_KEEP = 0.5
ASSIGN_PATH = Path("docs/redrawn_assignment.json")
S_STAR_SEED = 98_000
WINDOWS = {"tiny": 160_000, "medium": 160_000, "small": 256_000}
m10.BASE_ID[BASE] = 9


def cells():
    return [(h, p, e) for h, p in TYPES for e in m10.EXPOSURES]


def assignment():
    cs = cells()
    slots = [c for c in cs for _ in range(PER_CELL)] + [("unseen", "none", 0)] * m10.N_UNSEEN
    nouns = m9.FAM_NOUNS[:len(slots)]
    assert len(nouns) == len(slots) == 40
    by_seed = {}
    for i, seed in enumerate(SEEDS):
        rows = []
        for k, noun in enumerate(nouns):
            h, p, e = slots[(k + 3 * i) % len(slots)]
            rows.append({"noun": noun, "history": h, "presentation": p, "exposures": e,
                         "type": "unseen" if h == "unseen" else f"{h}_{p}", "s_star": None})
        rng = random.Random(S_STAR_SEED + seed)
        for c in cs + [("unseen", "none", 0)]:
            members = [r for r in rows if (r["history"], r["presentation"], r["exposures"]) == c]
            pool = []
            while len(pool) < len(members):
                pool += rng.sample(cg.SHAPE_TOKENS, len(cg.SHAPE_TOKENS))
            for r, s in zip(members, pool):
                r["s_star"] = s
        c = Counter((r["history"], r["presentation"], r["exposures"]) for r in rows)
        assert all(c[x] == PER_CELL for x in cs) and c[("unseen", "none", 0)] == m10.N_UNSEEN
        assert sum(r["exposures"] == 100 for r in rows) == 18 and sum(r["exposures"] == 1000 for r in rows) == 18
        by_seed[str(seed)] = rows
    return {"nouns": nouns, "seeds": by_seed}


def load_assignment(seed):
    return json.loads(ASSIGN_PATH.read_text(encoding="utf-8"))["seeds"][str(seed)]


def fam_units(rows, rng, marginal):
    """As make_familiarity_corpora_a10.fam_units, plus the redrawn and half-kept histories (frame 1 only)."""
    shapes, weights = list(marginal), list(marginal.values())
    out = []
    for r in rows:
        if r["history"] == "unseen":
            continue
        size = m10.RUN if r["presentation"] == "run" else 1
        for k in range(r["exposures"] // size):
            run_shape = rng.choices(shapes, weights)[0]
            sents, prev = [], None
            for j in range(size):
                c, t = rng.choice(cg.COLOUR_TOKENS), rng.choice(cg.TEXTURE_TOKENS)
                h = r["history"]
                if h == "fixed":
                    sh = r["s_star"]
                elif h == "rotating":
                    sh = run_shape
                elif h == "redrawn":
                    sh = rng.choices(shapes, weights)[0]
                elif h == "half":
                    sh = prev if (j > 0 and rng.random() < P_KEEP) else rng.choices(shapes, weights)[0]
                else:
                    raise ValueError(h)
                prev = sh
                sents.append(m10.FRAME1.format(noun=r["noun"], shape=sh, colour=c, texture=t))
            out.append((r["noun"], k, sents, 1))
    return out


def _patch():
    m10.load_assignment = load_assignment
    m10.fam_units = fam_units
    orig = m11.load_assignment

    def la(seed, design="mixed"):
        return load_assignment(seed) if design == DESIGN else orig(seed, design)
    m11.load_assignment = la


def cmd_assign(a):
    A = assignment()
    ASSIGN_PATH.write_text(json.dumps({"note": "Redrawn-burst control: noun-to-cell rotation and own shapes.", **A},
                                      indent=1), encoding="utf-8")
    print(f"written {ASSIGN_PATH}: {len(A['seeds'])} seeds")


def cmd_build(a):
    from tokenizers import Tokenizer
    _patch()
    tok = Tokenizer.from_file(str(m9.TOK_PATH))
    if m9.check_tokenizer(tok):
        sys.exit("tokenizer check failed")
    for seed in a.seeds:
        for n in a.windows:
            m10.build_one(BASE, seed, n, a.out_dir, tok)


def within_burst(d, seed, tok, ids):
    """Per history: share of burst sentences (after the first) whose shape repeats the previous sentence's, and the
    number of distinct shapes per noun."""
    shape_of = {tok.token_to_id(s): s for s in cg.SHAPE_TOKENS}
    man = json.loads((d / f"manifest_seed{seed}.json").read_text(encoding="utf-8"))
    hist = {r["noun"]: (r["history"], r["presentation"]) for r in man["assignment"]}
    same, n, distinct = Counter(), Counter(), {}
    for p in csv.DictReader(open(d / f"placements_seed{seed}.csv", encoding="utf-8")):
        start, L = int(p["start"]), int(p["n_tokens"])
        sh = [shape_of[int(x)] for x in ids[start:start + L] if int(x) in shape_of]
        h = hist[p["noun"]]
        distinct.setdefault(p["noun"], set()).update(sh)
        if h[1] == "run":
            for x, y in zip(sh, sh[1:]):
                same[h[0]] += x == y
                n[h[0]] += 1
    rate = {h: same[h] / n[h] for h in n}
    return rate, {k: len(v) for k, v in distinct.items()}, hist


def cmd_check(a):
    from tokenizers import Tokenizer
    _patch()
    tok = Tokenizer.from_file(str(m9.TOK_PATH))
    cache = {}
    orig_tok = m9.tokenise_file

    def tf(path, t):
        cache["ids"] = orig_tok(path, t)
        return cache["ids"]
    m9.tokenise_file = tf
    ref = {}
    if a.against:
        ref = {(r["base"], r["seed"], r["windows"]): r for r in json.load(open(a.against))["corpora"]}
    marg_q = {}
    n = bad = 0
    for seed in a.seeds:
        _, stable = m9.base_exemplars(seed)
        marg = m9.shape_marginal(BASE, stable)
        marg_q[seed] = sum(v * v for v in marg.values())
        for nw in a.windows:
            d = Path(a.out_dir) / BASE / f"w{nw // 1000}k"
            n += 1
            if not (d / f"manifest_seed{seed}.json").exists():
                print(f"MISSING {d} seed {seed}")
                bad += 1
                continue
            fail, man = m10.check_one(d, seed, tok)
            if man["assignment"] != load_assignment(seed):
                fail.append(f"assignment differs from {ASSIGN_PATH}")
            rate, distinct, hist = within_burst(d, seed, tok, cache["ids"])
            q = marg_q[seed]
            if abs(rate["rotating"] - 1) > 1e-9:
                fail.append(f"changing bursts repeat {rate['rotating']:.3f}")
            if abs(rate["fixed"] - 1) > 1e-9:
                fail.append(f"fixed bursts repeat {rate['fixed']:.3f}")
            if not (abs(rate["redrawn"] - q) < 0.05):
                fail.append(f"redrawn bursts repeat {rate['redrawn']:.3f}, expected about {q:.3f}")
            if not (abs(rate["half"] - (P_KEEP + (1 - P_KEEP) * q)) < 0.05):
                fail.append(f"half-kept bursts repeat {rate['half']:.3f}, expected about {P_KEEP + (1 - P_KEEP) * q:.3f}")
            few = [k for k, v in distinct.items() if hist[k][0] in ("rotating", "redrawn", "half") and v < 3]
            if few:
                fail.append(f"too few shapes: {few}")
            r = ref.get((BASE, seed, nw))
            if a.against and (r is None or r["md5_text"] != man["md5_text"] or r["md5_order"] != man["md5_order"]):
                fail.append(f"not the corpus recorded in {a.against}")
            bad += bool(fail)
            print(f"{'OK  ' if not fail else 'FAIL'} {BASE} w{nw // 1000}k seed {seed}: repeats within bursts "
                  f"changing {rate['rotating']:.2f}, half-kept {rate['half']:.3f}, redrawn {rate['redrawn']:.3f} "
                  f"(q = {q:.3f})" + ("" if not fail else ": " + "; ".join(fail[:6])), flush=True)
    print(f"{n - bad} of {n} corpora passed every check")
    if bad:
        sys.exit(1)


def cmd_corpus_manifest(a):
    rows = []
    for seed in a.seeds:
        for nw in a.windows:
            d = Path(a.out_dir) / BASE / f"w{nw // 1000}k"
            man = json.loads((d / f"manifest_seed{seed}.json").read_text(encoding="utf-8"))
            rows.append({k: man[k] for k in ("base", "seed", "windows", "tokens", "md5_text", "md5_order",
                                             "familiarity_units", "familiarity_sentences", "base_frame7_share")})
    Path(a.out).write_text(json.dumps({"note": "Redrawn-burst control corpora as built, before any model was trained.",
                                       "corpora": rows}, indent=1), encoding="utf-8")
    print(f"written {a.out}: {len(rows)} corpora")


def cmd_manifest(a):
    rows = []
    for size in ("tiny", "small", "medium"):
        for seed in SEEDS:
            run = Path(a.ckpt_dir) / BASE / f"{size}_seed{seed}"
            if not (run / "model_final.pt").exists():
                continue
            cdir = Path(a.corpus_dir) / BASE / f"w{WINDOWS[size] // 1000}k"
            rows.append({"base": BASE, "size": size, "seed": seed, "per_cell": PER_CELL, "design": f"a11_{DESIGN}",
                         "ckpt": str(run / "model_final.pt"), "label": "", "tokenizer": a.tokenizer,
                         "corpus": str(cdir / f"seed{seed}.txt"), "order": str(cdir / f"order_seed{seed}.npy"),
                         "placements": str(cdir / f"placements_seed{seed}.csv")})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} models listed in {a.out}")


def cmd_score(a):
    _patch()
    import familiarity_eval
    sys.argv = ["familiarity_eval.py", "--manifest", a.manifest, "--out", a.out]
    familiarity_eval.main()


# ---------------------------------------------------------------- summary
LAB = {"fixed_single": "Fixed, spaced", "fixed_run": "Fixed, bursts", "rotating_single": "Changing, spaced",
       "rotating_run": "Changing, bursts", "half_run": "Half-kept, bursts", "redrawn_run": "Redrawn, bursts",
       "unseen": "Unseen"}
ORDER = ["rotating_run", "half_run", "redrawn_run", "fixed_run", "rotating_single", "fixed_single"]
CONTRASTS = [("rotating_run", "redrawn_run"), ("rotating_run", "half_run"), ("half_run", "redrawn_run"),
             ("redrawn_run", "rotating_single"), ("redrawn_run", "fixed_run"), ("rotating_run", "fixed_run")]


def cmd_summary(a):
    import numpy as np
    import pandas as pd
    from scipy import stats

    def ci(x):
        x = np.asarray(x, float)
        m = x.mean()
        h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        return m, m - h, m + h

    def tex(v):
        s = f"{v:.2f}"
        return s.replace("-", r"\textminus{}") if v < 0 and float(s) != 0 else s.lstrip("-")

    s = pd.read_csv(a.scores)
    s = s[(s.base == BASE) & (s.emb == "original") & (s.type != "trained")]
    if a.model_check:
        mc = pd.read_csv(a.model_check)
        bad = mc[mc.status.str.lower() != "pass"]
        if len(bad):
            print("excluded by the model check:", bad[["size", "seed"]].values.tolist())
        s = s[~s.set_index(["size", "seed"]).index.isin(bad.set_index(["size", "seed"]).index)]
    g = s.groupby(["size", "seed", "type", "exposures"]).carry.mean()
    sizes = sorted(s["size"].unique(), key=["tiny", "small", "medium"].index)
    seeds_complete = [sd for sd in s.seed.unique() if all((z, sd) in g.droplevel([2, 3]).index for z in sizes)]

    def cell(t, e):
        x = g.xs((t, e), level=("type", "exposures")).unstack("size")[sizes]
        return x.loc[seeds_complete]

    md = ["# Redrawn-burst control (exploratory; not pre-registered)", "",
          f"Models: {s.groupby(['size', 'seed']).ngroups} ({', '.join(f'{z} {s[s['size'] == z].seed.nunique()}' for z in sizes)}); "
          f"{len(seeds_complete)} seeds with every size. Seed-level: sizes averaged within seed; t-based 95% intervals.", "",
          "## Copy score by history (nats)", ""]
    tl = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
          r"& \multicolumn{2}{c}{100 exposures} & \multicolumn{2}{c}{1,000 exposures} \\",
          r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"History & Mean & 95\% CI & Mean & 95\% CI \\", r"\midrule"]
    for t in ORDER:
        parts, tparts = [], []
        for e in (100, 1000):
            x = cell(t, e)
            m, lo, hi = ci(x.mean(axis=1))
            bys = ", ".join(f"{z} {x[z].mean():.2f}" for z in sizes)
            parts.append(f"{e}: {m:.2f} [{lo:.2f}, {hi:.2f}] ({bys})")
            tparts.append(f"{tex(m)} & [{tex(lo)}, {tex(hi)}]")
        md.append(f"- {LAB[t]}: " + "; ".join(parts))
        tl.append(f"{LAB[t]} & " + " & ".join(tparts) + r" \\")
    u = cell("unseen", 0)
    m, lo, hi = ci(u.mean(axis=1))
    md.append(f"- Unseen: {m:.2f} [{lo:.2f}, {hi:.2f}]")
    md += ["", "## Contrasts (paired within model, nats)", ""]
    tl += [r"\midrule", r"\multicolumn{5}{@{}l}{\emph{Differences}} \\"]
    for x, y in CONTRASTS:
        parts, tparts = [], []
        for e in (100, 1000):
            dd = cell(x, e) - cell(y, e)
            m, lo, hi = ci(dd.mean(axis=1))
            bys = ", ".join(f"{z} {dd[z].mean():.2f}" for z in sizes)
            parts.append(f"{e}: {m:.2f} [{lo:.2f}, {hi:.2f}] ({bys})")
            tparts.append(f"{tex(m)} & [{tex(lo)}, {tex(hi)}]")
        md.append(f"- {LAB[x]} minus {LAB[y]}: " + "; ".join(parts))
        tl.append(f"{LAB[x].split(',')[0]} $-$ {LAB[y].split(',')[0].lower()}"
                  f"{' bursts' if y.endswith('_run') and x.endswith('_run') else ''} & " + " & ".join(tparts) + r" \\")
    # dose-response: copy score at 1,000 against the within-burst repeat rate, per seed
    md += ["", "## Fall from 100 to 1,000 exposures (nats)", ""]
    for t in ORDER:
        dd = cell(t, 100) - cell(t, 1000)
        m, lo, hi = ci(dd.mean(axis=1))
        md.append(f"- {LAB[t]}: {m:.2f} [{lo:.2f}, {hi:.2f}]")
    tl += [r"\bottomrule", r"\end{tabular}"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    if a.tex:
        Path(a.tex).write_text("\n".join(tl) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("assign")
    for name in ("build", "check", "corpus_manifest"):
        b = sub.add_parser(name)
        b.add_argument("--out_dir", required=True)
        b.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
        b.add_argument("--windows", nargs="+", type=int, default=[160_000, 256_000])
        if name == "check":
            b.add_argument("--against", default=None)
        if name == "corpus_manifest":
            b.add_argument("--out", default="docs/redrawn_corpus_manifest.json")
    b = sub.add_parser("manifest")
    b.add_argument("--ckpt_dir", required=True)
    b.add_argument("--corpus_dir", required=True)
    b.add_argument("--tokenizer", default="data/tokenizers/amendment9_tokenizer.json")
    b.add_argument("--out", required=True)
    b = sub.add_parser("score")
    b.add_argument("--manifest", required=True)
    b.add_argument("--out", required=True)
    b = sub.add_parser("summary")
    b.add_argument("--scores", required=True)
    b.add_argument("--model_check", default=None)
    b.add_argument("--out", required=True)
    b.add_argument("--tex", default=None)
    a = ap.parse_args()
    {"assign": cmd_assign, "build": cmd_build, "check": cmd_check, "corpus_manifest": cmd_corpus_manifest,
     "manifest": cmd_manifest, "score": cmd_score, "summary": cmd_summary}[a.cmd](a)
