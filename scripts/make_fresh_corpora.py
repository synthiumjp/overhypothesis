#!/usr/bin/env python3
"""
make_fresh_corpora.py  |  Corpora for Amendment 8.

Why. Study 1 models were trained by sampling 128-token windows from one fixed file
(26 passes of 448 exemplars), so each position was seen 160 to 290 times and the
models came close to memorising the file (loss 0.07 to 0.16 nats per token on it).
Amendment 8 asks two things:
  Part A  Do the main arrangement effects hold when the arrangement is generated
          afresh for every pass, so that no stretch of text recurs in training?
  Part B  When a kind's shape is reassigned between passes, the noun alone no longer
          predicts the shape, but within a pass the shape still stays fixed. Do models
          then tie the expectation to the noun?

Every corpus keeps the Study 1 exemplars (or the Amendment 7 objects) of its seed, the
five frames, the regular tokenizer and the arrangement rules of the matching Study 1
corpus. Only the number of passes changes (each pass arranged with its own draw). With
train_model.py --fresh_windows, training reads non-overlapping 128-token windows, so a
whole run (at most 32.8 million tokens, small models) never returns to the same text:
7,500 passes for the 448-exemplar corpora (4,636 to 4,693 tokens per pass, depending on
the seed; 34.8 to 35.2 million tokens) and 2,100 passes for the Objects corpora (1,664
sentences, 16,224 tokens per pass; 34.1 million tokens).

Conditions (folder names; file in each = regular_seed<s>.txt, so train_model.py runs
with --condition regular):
  A  regular_random      Regular exemplars, each pass fully shuffled
     regular_b3p6        Regular, graded order with f = 0.12 (as graded_b3p6)
     regular_grouped     Regular, runs of four by kind (as Study 1c)
     nostable_grouped    No stable feature exemplars (as corpora_x), grouped
     objects_same        Objects, same-object runs, grouped (as Amendment 7)
     objects_diff        Objects, different-object runs, grouped (as Amendment 7)
  B  reassigned_grouped  Regular exemplars; in each pass every kind gets a new shape
                         word drawn uniformly from the ten, used in all of that kind's
                         sentences in that pass; grouped as regular_grouped
     reassigned_random   the same, fully shuffled

Held-out files (heldout/<condition>/regular_seed<s>.txt): 100 further passes with an
independent random generator, never used in training; used for the loss check.

Usage (from the repo root):
  python scripts/make_fresh_corpora.py build --out_dir "E:/overhypothesis_fresh"
  python scripts/make_fresh_corpora.py check --out_dir "E:/overhypothesis_fresh" --against docs/amendment8_corpus_manifest.json
  python scripts/make_fresh_corpora.py build --out_dir <dir> --passes_scale 0.01   (quick test)
"""
import argparse
import hashlib
import importlib.util
import json
import random
from collections import Counter, defaultdict
from functools import reduce
from math import gcd
from pathlib import Path

SEEDS = (42, 123, 456, 789, 1001)
PASSES_REGULAR, PASSES_OBJECTS, PASSES_HELDOUT_REGULAR, PASSES_HELDOUT_OBJECTS = 7500, 2100, 100, 27
FRESH_SEED, HELDOUT_SEED = 13_000, 17_000
F = {"random": 0.0, "b3p6": 0.12, "grouped": 1.0}
CONDITIONS = ["regular_random", "regular_b3p6", "regular_grouped", "nostable_grouped",
              "objects_same", "objects_diff", "reassigned_grouped", "reassigned_random"]
COND_ID = {c: i + 1 for i, c in enumerate(CONDITIONS)}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


x = _load("x", "scripts/make_study1x_corpora.py")   # loads make_bursty_corpus (mb) and corpus_generator (cg)
mb, cg = x.mb, x.cg

# ---------------------------------------------------------------- exemplar sets

def regular_base(seed):
    """The 448 base exemplars of the Regular corpus for this seed, with copies."""
    lines = x.read_lines(Path("data/corpora") / f"regular_seed{seed}.txt")
    counts = Counter(lines)
    R = reduce(gcd, counts.values())
    return sorted(s for s, c in counts.items() for _ in range(c // R))


def nostable_base(seed):
    """The No stable feature exemplars of make_study1x_corpora.py `controls` (same draw)."""
    lines = x.read_lines(Path("data/corpora") / f"regular_seed{seed}.txt")
    counts = Counter(lines)
    R = reduce(gcd, counts.values())
    base_counts = Counter({s: c // R for s, c in counts.items()})
    copies = sorted((s, j) for s, c in base_counts.items() for j in range(c))
    rng = random.Random(seed + x.SUB_SEED)
    new_shape = {sj: rng.choice(cg.SHAPE_TOKENS) for sj in copies}
    return sorted(x.substitute(s, shape=new_shape[(s, j)]) for s, j in copies)


O_FRAMES = ["A {noun} is a {shape} {colour} {texture} thing.",
            "Look at the {noun}. It is {shape} and {colour}.",
            "Here is another {noun}. This one is {shape}.",
            "That is a {noun} because it is {shape}."]
N_OBJ, O_SEED_OFFSET = 13, 7000


def objects_base(seed):
    """The Amendment 7 objects (make_study1o_corpora.py base(), same draw)."""
    meta = json.load(open(f"data/corpora/regular_seed{seed}_meta.json"))
    stable = meta["noun_stable_feature"]
    rng = random.Random(seed + O_SEED_OFFSET)
    sent = {}
    for noun in sorted(stable):
        for o in range(N_OBJ):
            col, tex = rng.choice(cg.COLOUR_TOKENS), rng.choice(cg.TEXTURE_TOKENS)
            for fi, fr in enumerate(O_FRAMES):
                sent[(noun, o, fi)] = fr.format(noun=noun, shape=stable[noun], colour=col, texture=tex)
    return sent, sorted(stable)

# ---------------------------------------------------------------- one pass

def arrange_pass(exemplars, f, rng):
    """One pass: each kind's exemplars shuffled and cut into runs of four; each run kept
    with probability f, otherwise split; all units shuffled (f = 0 random, 1 grouped)."""
    by_kind = defaultdict(list)
    for s in exemplars:
        by_kind[mb.kind_of(s)].append(s)
    units = []
    for k in sorted(by_kind):
        ex = by_kind[k][:]
        rng.shuffle(ex)
        for i in range(0, len(ex), mb.BURST_SIZE):
            run = ex[i:i + mb.BURST_SIZE]
            if f >= 1 or (f > 0 and rng.random() < f):
                units.append(run)
            else:
                units += [[s] for s in run]
    rng.shuffle(units)
    return [s for u in units for s in u]


def objects_pass(sent, nouns, kind, rng):
    """One pass of Amendment 7 grouped runs (f = 1): same = the four descriptions of one
    object; diff = four different objects of the kind, one per frame (Latin square)."""
    units = []
    for n in nouns:
        if kind == "same":
            runs = [[(n, o, fi) for fi in range(4)] for o in range(N_OBJ)]
        else:
            perm = list(range(N_OBJ))
            rng.shuffle(perm)
            runs = [[(n, perm[(i + 3 * j) % N_OBJ], j) for j in range(4)] for i in range(N_OBJ)]
        for r in runs:
            rng.shuffle(r)
            units.append(r)
    rng.shuffle(units)
    return [sent[q] for u in units for q in u]


def reassign(exemplars, rng):
    """New shape word for every kind, used in all of that kind's sentences in this pass."""
    shape = {k: rng.choice(cg.SHAPE_TOKENS) for k in sorted({mb.kind_of(s) for s in exemplars})}
    return [x.substitute(s, shape=shape[mb.kind_of(s)]) for s in exemplars]


def passes(cond, seed, n, rng):
    """Yields the passes (lists of sentences) of one corpus."""
    if cond.startswith("objects_"):
        sent, nouns = objects_base(seed)
        for _ in range(n):
            yield objects_pass(sent, nouns, cond.split("_")[1], rng)
        return
    base = nostable_base(seed) if cond.startswith("nostable") else regular_base(seed)
    f = F[cond.split("_")[1]]
    for _ in range(n):
        ex = reassign(base, rng) if cond.startswith("reassigned") else base
        yield arrange_pass(ex, f, rng)


def write_corpus(path, cond, seed, n, rng):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, n_lines, n_words, first = hashlib.md5(), 0, 0, []
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for i, p in enumerate(passes(cond, seed, n, rng)):
            chunk = ("\n" if i else "") + "\n".join(p)
            fh.write(chunk)
            h.update(chunk.encode("utf-8"))
            n_lines += len(p)
            n_words += sum(len(s.split()) for s in p)
            if len(first) < 11_648:
                first += p
    return {"md5": h.hexdigest(), "sentences": n_lines, "words": n_words, "first_11648": first[:11_648]}


def build(a):
    out = Path(a.out_dir)
    rows = []
    for cond in a.conditions:
        objects = cond.startswith("objects_")
        n_train = max(1, round((PASSES_OBJECTS if objects else PASSES_REGULAR) * a.passes_scale))
        n_held = PASSES_HELDOUT_OBJECTS if objects else PASSES_HELDOUT_REGULAR
        for seed in SEEDS:
            for part, n, base_seed in (("train", n_train, FRESH_SEED), ("heldout", n_held, HELDOUT_SEED)):
                if part not in a.parts:
                    continue
                rng = random.Random(seed + base_seed + 100 * COND_ID[cond])
                p = (out / cond if part == "train" else out / "heldout" / cond) / f"regular_seed{seed}.txt"
                r = write_corpus(p, cond, seed, n, rng)
                B, P = x.burstiness(r.pop("first_11648"))
                rows.append({"condition": cond, "part": part, "seed": seed, "passes": n,
                             "file": str(p.relative_to(out)).replace("\\", "/"), **r,
                             "B_first_11648_sentences": round(B, 3)})
                print({k: v for k, v in rows[-1].items() if k != "file"}, flush=True)
    name = "manifest.json" if set(a.parts) == {"train", "heldout"} else f"manifest_{'_'.join(a.parts)}.json"
    json.dump({"passes_scale": a.passes_scale, "corpora": rows}, open(out / name, "w"), indent=1)


def check(a):
    out = Path(a.out_dir)
    m = json.load(open(out / "manifest.json"))
    bad = 0
    for r in m["corpora"]:
        h = hashlib.md5()
        with open(out / r["file"], "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        ok = h.hexdigest() == r["md5"]
        bad += not ok
        if not ok:
            print("MISMATCH", r["file"])
    print(f"{len(m['corpora']) - bad} of {len(m['corpora'])} corpora match manifest.json")
    if a.against:
        reg = {r["file"]: r["md5"] for r in json.load(open(a.against))["corpora"]}
        mine = {r["file"]: r["md5"] for r in m["corpora"]}
        same = sum(mine.get(f) == h for f, h in reg.items())
        print(f"{same} of {len(reg)} registered corpora reproduced exactly ({a.against})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out_dir", required=True)
    b.add_argument("--conditions", nargs="+", default=CONDITIONS)
    b.add_argument("--passes_scale", type=float, default=1.0, help="< 1 only for a quick test")
    b.add_argument("--parts", nargs="+", default=["train", "heldout"], choices=["train", "heldout"],
                   help="heldout alone rebuilds only the held-out files (for S40), with manifest_heldout.json")
    c = sub.add_parser("check")
    c.add_argument("--out_dir", required=True)
    c.add_argument("--against", default=None, help="registered manifest to compare MD5s with")
    a = ap.parse_args()
    {"build": build, "check": check}[a.cmd](a)
