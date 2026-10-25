#!/usr/bin/env python3
"""
make_familiarity_corpora_a11.py  |  Corpora for Amendment 11 (docs/AMENDMENT11.md).

Reuses the Amendment 10 Part A builder (scripts/make_familiarity_corpora_a10.py) unchanged, with
new seeds and two new noun designs:
  mixed            the Amendment 10 design: fixed, no-shape (absent) and changing (rotating) nouns,
                   each spaced (single) or in bursts (runs), at 100 and 1,000 exposures, three nouns
                   per cell, plus four unseen nouns (base name regular_grouped).
  pure_changing    every familiar noun has the changing-shape history: spaced and bursts at 100 and
                   1,000 exposures, nine nouns per cell, plus four unseen (base regular_pure_changing).
  pure_fixed       every familiar noun has the fixed-shape history, likewise (base regular_pure_fixed).
In all three, 18 familiar nouns have 100 exposures and 18 have 1,000, so the new nouns fill the
same share of the text. Base text: stable-shape (Regular) kinds in runs of four, exactly as in
Amendment 10 (frame 7 in one sentence in fifteen). Nouns rotate over cells across seeds.

  python scripts/make_familiarity_corpora_a11.py assign
  python scripts/make_familiarity_corpora_a11.py build --out_dir E:/overhypothesis_a11 [--designs ...] [--seeds ...]
  python scripts/make_familiarity_corpora_a11.py check --out_dir E:/overhypothesis_a11 --against docs/amendment11_corpus_manifest.json
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus_generator as cg  # noqa: E402
import make_familiarity_corpora as m9  # noqa: E402
import make_familiarity_corpora_a10 as m10  # noqa: E402

SEEDS_A = [2029, 3037, 4049, 5051, 6067, 7079, 8081, 9091, 10103, 11117]   # Part A (and B: first five)
SEEDS_C = SEEDS_A[:5]                                                       # Part C
DESIGNS = {"mixed": ("regular_grouped", m10.HISTORIES, 3),
           "pure_changing": ("regular_pure_changing", ["rotating"], 9),
           "pure_fixed": ("regular_pure_fixed", ["fixed"], 9)}
BASE_OF = {d: v[0] for d, v in DESIGNS.items()}
DESIGN_OF = {v[0]: d for d, v in DESIGNS.items()}
ASSIGN_PATH = Path("docs/amendment11_assignment.json")
S_STAR_SEED = 97_000
m10.BASE_ID.update({"regular_pure_changing": 5, "regular_pure_fixed": 7})


def cells(histories):
    return [(h, p, e) for h in histories for p in m10.PRESENTATIONS for e in m10.EXPOSURES]


def assignment():
    out = {"nouns": None, "designs": {}}
    for design, (base, hist, per_cell) in DESIGNS.items():
        cs = cells(hist)
        slots = [c for c in cs for _ in range(per_cell)] + [("unseen", "none", 0)] * m10.N_UNSEEN
        nouns = m9.FAM_NOUNS[:len(slots)]
        out["nouns"] = nouns
        seeds = SEEDS_A if design == "mixed" else SEEDS_C
        by_seed = {}
        for i, seed in enumerate(seeds):
            rows = []
            for k, noun in enumerate(nouns):
                h, p, e = slots[(k + 3 * i) % len(slots)]
                rows.append({"noun": noun, "history": h, "presentation": p, "exposures": e,
                             "type": "unseen" if h == "unseen" else f"{h}_{p}", "s_star": None})
            rng = random.Random(S_STAR_SEED + 101 * list(DESIGNS).index(design) + seed)
            for c in cs + [("unseen", "none", 0)]:
                members = [r for r in rows if (r["history"], r["presentation"], r["exposures"]) == c]
                # own shapes: sampled without replacement where possible, else with as few repeats as possible
                pool = []
                while len(pool) < len(members):
                    pool += rng.sample(cg.SHAPE_TOKENS, len(cg.SHAPE_TOKENS))
                for r, s in zip(members, pool):
                    r["s_star"] = s
            c = Counter((r["history"], r["presentation"], r["exposures"]) for r in rows)
            assert all(c[x] == per_cell for x in cs) and c[("unseen", "none", 0)] == m10.N_UNSEEN
            by_seed[str(seed)] = rows
        out["designs"][design] = by_seed
    return out


def load_assignment(seed, design="mixed"):
    return json.loads(ASSIGN_PATH.read_text(encoding="utf-8"))["designs"][design][str(seed)]


def cmd_assign(a):
    A = assignment()
    ASSIGN_PATH.write_text(json.dumps({"note": "Amendment 11 noun-to-cell rotation and own shapes, by design.", **A},
                                      indent=1), encoding="utf-8")
    print(f"written {ASSIGN_PATH}: designs {list(A['designs'])}")


def _use(design):
    m10.load_assignment = lambda seed: load_assignment(seed, design)


def cmd_build(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(m9.TOK_PATH))
    if m9.check_tokenizer(tok):
        sys.exit("tokenizer check failed")
    for design in a.designs:
        _use(design)
        seeds = a.seeds or (SEEDS_A if design == "mixed" else SEEDS_C)
        for seed in seeds:
            for n in a.windows:
                m10.build_one(BASE_OF[design], seed, n, a.out_dir, tok)


def cmd_check(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(m9.TOK_PATH))
    ref = {}
    if a.against:
        ref = {(r["base"], r["seed"], r["windows"]): r for r in json.load(open(a.against))["corpora"]}
    n = bad = 0
    for design in a.designs:
        seeds = a.seeds or (SEEDS_A if design == "mixed" else SEEDS_C)
        for seed in seeds:
            for nw in a.windows:
                d = Path(a.out_dir) / BASE_OF[design] / f"w{nw // 1000}k"
                n += 1
                if not (d / f"manifest_seed{seed}.json").exists():
                    print(f"MISSING {d} seed {seed}")
                    bad += 1
                    continue
                fail, man = m10.check_one(d, seed, tok)
                if man["assignment"] != load_assignment(seed, design):
                    fail.append("assignment differs from docs/amendment11_assignment.json")
                r = ref.get((BASE_OF[design], seed, nw))
                if a.against and (r is None or r["md5_text"] != man["md5_text"] or r["md5_order"] != man["md5_order"]):
                    fail.append(f"not the corpus recorded in {a.against}")
                bad += bool(fail)
                print(f"{'OK  ' if not fail else 'FAIL'} {BASE_OF[design]} w{nw // 1000}k seed {seed}"
                      + ("" if not fail else ": " + "; ".join(fail[:6])), flush=True)
    print(f"{n - bad} of {n} corpora passed every check")
    if bad:
        sys.exit(1)


def cmd_manifest(a):
    rows = []
    for design in a.designs:
        seeds = SEEDS_A if design == "mixed" else SEEDS_C
        for seed in seeds:
            for nw in a.windows:
                d = Path(a.out_dir) / BASE_OF[design] / f"w{nw // 1000}k"
                man = json.loads((d / f"manifest_seed{seed}.json").read_text(encoding="utf-8"))
                rows.append({k: man[k] for k in ("base", "seed", "windows", "tokens", "md5_text", "md5_order",
                                                 "familiarity_units", "familiarity_sentences", "base_frame7_share")})
    Path(a.out).write_text(json.dumps({"note": "Amendment 11 corpora as built, before any Amendment 11 model was "
                                               "trained.", "corpora": rows}, indent=1), encoding="utf-8")
    print(f"written {a.out}: {len(rows)} corpora")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("assign")
    for name in ("build", "check", "manifest"):
        b = sub.add_parser(name)
        b.add_argument("--out_dir", required=True)
        b.add_argument("--designs", nargs="+", default=list(DESIGNS), choices=list(DESIGNS))
        b.add_argument("--seeds", nargs="+", type=int, default=None)
        b.add_argument("--windows", nargs="+", type=int, default=[160_000, 256_000])
        if name == "check":
            b.add_argument("--against", default=None)
        if name == "manifest":
            b.add_argument("--out", default="docs/amendment11_corpus_manifest.json")
    a = ap.parse_args()
    {"assign": cmd_assign, "build": cmd_build, "check": cmd_check, "manifest": cmd_manifest}[a.cmd](a)
