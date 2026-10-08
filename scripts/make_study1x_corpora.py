#!/usr/bin/env python3
"""
make_study1x_corpora.py  |  Study 1 extension corpora (Amendment 5).

All new corpora are built from the five Regular corpora of Study 1, so every
one keeps the 32 kinds, 14 exemplars per kind, the five frames, the 26 passes
and the token count. Each corpus is written as regular_seed<s>.txt in its own
folder, so train_model.py runs unchanged with the Regular tokenizer:

  python scripts/train_model.py --condition regular --size <size> --seed <s> \
      --corpus_dir data/corpora_x/<name> --output_dir models/checkpoints_x/<name>

Controls (what is stable within a kind), each in random and grouped order
  nostable_random, nostable_grouped
      every exemplar gets a fresh shape word; no dimension is stable within a kind
  colourstable_random, colourstable_grouped
      shape varies as in nostable; each kind keeps one colour word wherever a
      frame mentions colour
The substitution is made once per base exemplar, so all 26 passes repeat the
same 448 exemplars, as in Regular. Random order keeps the exact sentence
order of the Regular source; grouped order uses the Study 1c procedure
(make_bursty_corpus.build) with the same seed.

Graded clustering (Regular content, order only)
  graded_f<f>: within each pass, each kind's exemplars are shuffled and cut
  into runs of four (4, 4, 4, 2); each run is kept together with probability
  f and otherwise split into single sentences; all units are then shuffled.
  f = 0 is random order and f = 1 is the Study 1c grouped order. The values
  of f were chosen, before any model was trained, so that clustering B
  (cds_burstiness.py, W = 100) lands near 1.4 (Pile), 3.6 (CHILDES), 7 and 12.

Usage (from the repo root)
  python scripts/make_study1x_corpora.py controls
  python scripts/make_study1x_corpora.py tune      # prints B for a grid of f
  python scripts/make_study1x_corpora.py graded
  python scripts/make_study1x_corpora.py check     # B of every corpus + MD5s
"""

import argparse
import hashlib
import importlib.util
import json
import random
import re
from collections import Counter, defaultdict
from functools import reduce
from math import gcd
from pathlib import Path

SEEDS = [42, 123, 456, 789, 1001]
SRC = Path("data/corpora")
OUT = Path("data/corpora_x")
SUB_SEED = 9_000       # substitution RNG, separate from every earlier RNG
GRADED_SEED = 11_000   # ordering RNG for graded corpora
GRADED_F = {"b1p4": 0.02, "b3p6": 0.12, "b7": 0.27, "b12": 0.5}   # fixed by `tune`


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


mb = _load("mb", "scripts/make_bursty_corpus.py")
cg = _load("cg", "scripts/corpus_generator.py")
cds = _load("cds", "scripts/cds_burstiness.py")
SHAPES, COLOURS = set(cg.SHAPE_TOKENS), set(cg.COLOUR_TOKENS)


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def read_lines(path):
    return [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def write(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def substitute(sentence, shape=None, colour=None):
    toks = re.findall(r"\w+|[^\w\s]", sentence)
    out = []
    for t in toks:
        if shape and t in SHAPES:
            t = shape
        elif colour and t in COLOURS:
            t = colour
        out.append(t)
    s = " ".join(out)
    return re.sub(r" ([.,])", r"\1", s)


def controls(args):
    for seed in SEEDS:
        lines = read_lines(SRC / f"regular_seed{seed}.txt")
        counts = Counter(lines)
        R = reduce(gcd, counts.values())
        n = len(lines) // R
        passes = [lines[i * n:(i + 1) * n] for i in range(R)]
        base_counts = Counter({s: c // R for s, c in counts.items()})
        for p in passes:  # each pass is one shuffle of the 448 base exemplars
            assert Counter(p) == base_counts
        # Regular repeats some exemplar sentences word for word (a kind's shape
        # never changes). Each copy is a separate exemplar, so each copy gets its
        # own substitution: copy j of sentence s is the j-th time s occurs in a pass.
        copies = sorted((s, j) for s, c in base_counts.items() for j in range(c))
        rng = random.Random(seed + SUB_SEED)
        new_shape = {sj: rng.choice(cg.SHAPE_TOKENS) for sj in copies}
        kinds = sorted({mb.kind_of(s) for s in base_counts})
        pool = cg.COLOUR_TOKENS[:]
        rng.shuffle(pool)
        kind_colour = {k: pool[i % len(pool)] for i, k in enumerate(kinds)}
        maps = {
            "nostable": {sj: substitute(sj[0], shape=new_shape[sj]) for sj in copies},
            "colourstable": {sj: substitute(sj[0], shape=new_shape[sj],
                                            colour=kind_colour[mb.kind_of(sj[0])]) for sj in copies},
        }
        for name, m in maps.items():
            for sj in copies:  # the substitution must not change anything else
                assert len(m[sj].split()) == len(sj[0].split()), (sj, m[sj])
            rand = []
            for p in passes:
                seen = Counter()
                for s in p:
                    rand.append(m[(s, seen[s])])
                    seen[s] += 1
            p = OUT / f"{name}_random" / f"regular_seed{seed}.txt"
            write(p, rand)
            text, _ = mb.build(p, seed)
            write(OUT / f"{name}_grouped" / f"regular_seed{seed}.txt", text.split("\n"))
        print(f"seed {seed}: controls written")


def graded_lines(lines, f, seed):
    counts = Counter(lines)
    R = reduce(gcd, counts.values())
    base = [s for s, c in counts.items() for _ in range(c // R)]
    by_kind = defaultdict(list)
    for s in base:
        by_kind[mb.kind_of(s)].append(s)
    rng = random.Random(seed + GRADED_SEED)
    out = []
    for _ in range(R):
        units = []
        for k in sorted(by_kind):
            ex = by_kind[k][:]
            rng.shuffle(ex)
            for i in range(0, len(ex), mb.BURST_SIZE):
                run = ex[i:i + mb.BURST_SIZE]
                if rng.random() < f:
                    units.append(run)
                else:
                    units += [[s] for s in run]
        rng.shuffle(units)
        out += [s for u in units for s in u]
    assert Counter(out) == counts
    return out


def burstiness(lines):
    units = [cds.WORD.findall(l.lower()) for l in lines]
    kinds = [mb.kind_of(l) for l in lines]
    r = cds.measure_doc(units, random.Random(cds.SEED), kinds)
    w = cds.PRIMARY_W
    P = r[f"same_{w}"] / r[f"pairs_{w}"]
    P0 = r[f"same0_{w}"] / r[f"pairs0_{w}"]
    return P / P0, P


def tune(args):
    for f in [0.0, 0.01, 0.02, 0.05, 0.1, 0.115, 0.13, 0.2, 0.25, 0.3, 0.4, 0.46, 0.5, 1.0]:
        bs = [burstiness(graded_lines(read_lines(SRC / f"regular_seed{s}.txt"), f, s))[0]
              for s in SEEDS]
        print(f"f = {f:5.3f}: B = {sum(bs) / len(bs):6.2f} (range {min(bs):.2f} to {max(bs):.2f})")


def graded(args):
    for name, f in GRADED_F.items():
        for seed in SEEDS:
            lines = graded_lines(read_lines(SRC / f"regular_seed{seed}.txt"), f, seed)
            write(OUT / f"graded_{name}" / f"regular_seed{seed}.txt", lines)
        print(f"graded_{name} (f = {f}) written")


def check(args):
    rows = []
    folders = [("regular_random", SRC), ("regular_grouped", Path("data/corpora_bursty"))] + \
              [(p.name, p) for p in sorted(OUT.iterdir()) if p.is_dir()]
    for name, folder in folders:
        for seed in SEEDS:
            p = folder / f"regular_seed{seed}.txt"
            lines = read_lines(p)
            B, P = burstiness(lines)
            rows.append({"corpus": name, "seed": seed, "file": str(p), "md5": md5(p),
                         "sentences": len(lines), "words": sum(len(l.split()) for l in lines),
                         "B": round(B, 3), "P_same_kind_neighbour": round(P, 4)})
    json.dump(rows, open(OUT / "manifest.json", "w"), indent=1)
    by = defaultdict(list)
    for r in rows:
        by[r["corpus"]].append(r["B"])
    for k, v in by.items():
        print(f"{k:24s} B = {sum(v) / len(v):6.2f}  (seeds {min(v):.2f} to {max(v):.2f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("controls", "tune", "graded", "check"):
        sp.add_parser(c)
    a = ap.parse_args()
    {"controls": controls, "tune": tune, "graded": graded, "check": check}[a.cmd](a)


if __name__ == "__main__":
    main()
