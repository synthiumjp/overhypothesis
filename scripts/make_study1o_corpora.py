#!/usr/bin/env python3
"""
make_study1o_corpora.py  |  Corpora for Amendment 7: does nearby repetition teach the
regularity when it re-describes the same object, or only when it brings different
objects of the kind?

One set of sentences per seed, arranged in five orders. Every order contains exactly
the same sentences, each as often; same-object and different-object runs have the
same kind-level clustering. Only whether a run re-describes one object or describes
four different objects differs.

Sentences. 32 kinds (the Study 1 nouns), each with its Study 1 stable shape (from
data/corpora/regular_seed<s>_meta.json). Each kind has 13 objects; each object has
that shape and its own colour and texture (drawn uniformly from the Study 1 word
lists). Each object is described in four Study 1 frames:
  F1  A {noun} is a {shape} {colour} {texture} thing.
  F2  Look at the {noun}. It is {shape} and {colour}.
  F3  Here is another {noun}. This one is {shape}.
  F5  That is a {noun} because it is {shape}.
(The Study 1 frame "The {noun} I showed you is {shape}, but a different {colour}."
is left out because it states that the colour differs.) 32 x 13 x 4 = 1,664 sentences
per pass; 7 passes = 11,648 sentences, as in Study 1.

Orders (within each pass; units are shuffled across kinds):
  random        every sentence a unit
  same_<f>      runs of four = the four descriptions of one object (random frame order)
  diff_<f>      runs of four = four different objects of one kind, one per frame
                (a Latin square over objects and frames, so each object-frame
                sentence is used once per pass)
  Each run is kept with probability f and otherwise split into single sentences.
  f = 1 is fully grouped; f for the B 3.6 level is chosen from the corpora alone
  (`tune`), before any model is trained.

Usage:
  python scripts/make_study1o_corpora.py tune
  python scripts/make_study1o_corpora.py build
  python scripts/make_study1o_corpora.py check
"""
import argparse
import hashlib
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path

SEEDS = (42, 123, 456, 789, 1001)
OUT = Path("data/corpora_o")
N_OBJ, PASSES, SEED_OFFSET = 13, 7, 7000
F_B36 = 0.11   # fixed by `tune` before training (B 3.58 same, 3.60 diff)
FRAMES = ["A {noun} is a {shape} {colour} {texture} thing.",
          "Look at the {noun}. It is {shape} and {colour}.",
          "Here is another {noun}. This one is {shape}.",
          "That is a {noun} because it is {shape}."]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cg = _load("cg", "scripts/corpus_generator.py")
x = _load("x", "scripts/make_study1x_corpora.py")


def base(seed):
    meta = json.load(open(f"data/corpora/regular_seed{seed}_meta.json"))
    stable = meta["noun_stable_feature"]
    rng = random.Random(seed + SEED_OFFSET)
    sent = {}
    for noun in sorted(stable):
        for o in range(N_OBJ):
            col, tex = rng.choice(cg.COLOUR_TOKENS), rng.choice(cg.TEXTURE_TOKENS)
            for fi, fr in enumerate(FRAMES):
                sent[(noun, o, fi)] = fr.format(noun=noun, shape=stable[noun], colour=col, texture=tex)
    return sent, sorted(stable)


def arrange(sent, nouns, kind, f, seed):
    """Returns the ordered keys (noun, object, frame); sentences are sent[key]."""
    rng = random.Random(seed + SEED_OFFSET + {"random": 1, "same": 2, "diff": 3}[kind] * 100 + int(f * 1000))
    out = []
    for _ in range(PASSES):
        units = []
        for n in nouns:
            if kind == "random":
                units += [[(n, o, fi)] for o in range(N_OBJ) for fi in range(4)]
                continue
            if kind == "same":
                runs = [[(n, o, fi) for fi in range(4)] for o in range(N_OBJ)]
            else:
                perm = list(range(N_OBJ))
                rng.shuffle(perm)
                # Latin square: run i takes object perm[(i + 3j) mod 13] in frame j, so every
                # object-frame sentence is used once per pass and the four objects differ
                runs = [[(n, perm[(i + 3 * j) % N_OBJ], j) for j in range(4)] for i in range(N_OBJ)]
            for r in runs:
                rng.shuffle(r)
                if rng.random() < f:
                    units.append(r)
                else:
                    units += [[k] for k in r]
        rng.shuffle(units)
        out += [k for u in units for k in u]
    return out


def conditions(f_b36):
    return {"random": ("random", 0.0), "same_grouped": ("same", 1.0), "diff_grouped": ("diff", 1.0),
            "same_b3p6": ("same", f_b36), "diff_b3p6": ("diff", f_b36)}


def same_object_share(keys):
    """Of neighbouring sentence pairs about the same kind, the share about the same object."""
    same_kind = same_obj = 0
    for a, b in zip(keys, keys[1:]):
        if a[0] == b[0]:
            same_kind += 1
            same_obj += a[1] == b[1]
    return same_obj / max(1, same_kind)


def tune(a):
    for f in (0.05, 0.08, 0.1, 0.12, 0.15, 0.2):
        bs = {k: [] for k in ("same", "diff")}
        for s in SEEDS:
            sent, nouns = base(s)
            for k in bs:
                bs[k].append(x.burstiness([sent[q] for q in arrange(sent, nouns, k, f, s)])[0])
        print(f"f = {f:.3f}: B same {sum(bs['same']) / 5:.2f}, diff {sum(bs['diff']) / 5:.2f}")


def build(a):
    rows = []
    for s in SEEDS:
        sent, nouns = base(s)
        ref = None
        for name, (kind, f) in conditions(a.f_b36).items():
            keys = arrange(sent, nouns, kind, f, s)
            lines = [sent[q] for q in keys]
            c = Counter(keys)
            ref = ref or c
            assert c == ref and len(lines) == 11648
            p = OUT / name / f"regular_seed{s}.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
            B, P = x.burstiness(lines)
            rows.append({"corpus": name, "seed": s, "f": f, "file": str(p).replace("\\", "/"),
                         "md5": hashlib.md5(p.read_bytes()).hexdigest(), "sentences": len(lines),
                         "words": sum(len(l.split()) for l in lines), "B": round(B, 3),
                         "same_object_share_neighbours": round(same_object_share(keys), 3)})
            print(rows[-1])
    json.dump({"f_b36": a.f_b36, "corpora": rows}, open(OUT / "manifest.json", "w"), indent=1)


def check(a):
    m = json.load(open(OUT / "manifest.json"))
    for r in m["corpora"]:
        assert hashlib.md5(Path(r["file"]).read_bytes()).hexdigest() == r["md5"], r["file"]
    print(f"{len(m['corpora'])} corpora match manifest.json")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
sub.add_parser("tune")
b = sub.add_parser("build")
b.add_argument("--f_b36", type=float, required=True)
sub.add_parser("check")
a = ap.parse_args()
{"tune": tune, "build": build, "check": check}[a.cmd](a)
