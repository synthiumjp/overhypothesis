#!/usr/bin/env python3
"""
make_bursty_corpus.py  |  Study 1c (pre-registered in Amendment 1, osf.io/x6zhn)

Builds a bursty version of an existing Study 1 corpus. Only the ORDER of
sentences changes. The exemplar sentences, the number of replicated passes
and the token count are identical to the source corpus.

Source corpora are 448 base exemplar sentences replicated in R full passes,
each pass freshly shuffled (corpus_generator.py). This script:
  1. recovers the base exemplars by dividing out the replication factor R
  2. groups them by kind (noun)
  3. for each of the R passes: shuffles each kind's exemplars, cuts them into
     bursts of 4 consecutive sentences (14 exemplars -> 4, 4, 4, 2), and
     shuffles the bursts across kinds

A 128-token training window then holds about three bursts, so the model sees
several exemplars of one kind together, and several kinds side by side.

Output goes to its own folder (default data/corpora_bursty) under the SAME
file names as the source (regular_seed42.txt, ...), so train_model.py runs
unmodified: point --corpus_dir at the new folder and --output_dir at a new
checkpoint folder. The source condition's tokenizer is reused as is.

Usage (from C:\\overhypothesis):
  python scripts/make_bursty_corpus.py
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from functools import reduce
from math import gcd
from pathlib import Path

BURST_SIZE = 4          # fixed by the pre-registration
SEED_OFFSET = 7_000     # ordering RNG is separate from the generator's RNG

NOUN_PATTERNS = [re.compile(p) for p in (
    r"^a (\w+) is a\b",            # A {noun} is a ...
    r"^look at the (\w+)\.",        # Look at the {noun}.
    r"^here is another (\w+)\.",    # Here is another {noun}.
    r"^the (\w+) i showed you\b",   # The {noun} I showed you ...
    r"^that is a (\w+) because\b",  # That is a {noun} because ...
)]


def kind_of(sentence):
    low = sentence.lower()
    for pat in NOUN_PATTERNS:
        m = pat.search(low)
        if m:
            return m.group(1)
    raise ValueError(f"No kind found in sentence: {sentence!r}")


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def build(src_path, seed):
    lines = [l.strip() for l in Path(src_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    counts = Counter(lines)
    R = reduce(gcd, counts.values())
    base = [s for s, c in counts.items() for _ in range(c // R)]
    by_kind = defaultdict(list)
    for s in base:
        by_kind[kind_of(s)].append(s)

    rng = random.Random(seed + SEED_OFFSET)
    kinds = sorted(by_kind)
    passes = []
    for _ in range(R):
        bursts = []
        for k in kinds:
            ex = by_kind[k][:]
            rng.shuffle(ex)
            bursts += [ex[i:i + BURST_SIZE] for i in range(0, len(ex), BURST_SIZE)]
        rng.shuffle(bursts)
        passes.append([s for b in bursts for s in b])

    out_lines = [s for p in passes for s in p]
    # Integrity: identical sentence multiset and token count to the source.
    assert Counter(out_lines) == counts, "sentence multiset differs from source"
    assert sum(len(s.split()) for s in out_lines) == sum(len(s.split()) for s in lines)
    stats = {
        "replication_factor": R,
        "base_exemplars": len(base),
        "kinds": len(kinds),
        "exemplars_per_kind": sorted({len(v) for v in by_kind.values()}),
        "sentences": len(out_lines),
        "whitespace_tokens": sum(len(s.split()) for s in out_lines),
        "same_kind_adjacent_pairs": sum(kind_of(a) == kind_of(b) for a, b in zip(out_lines, out_lines[1:])),
        "source_same_kind_adjacent_pairs": sum(kind_of(a) == kind_of(b) for a, b in zip(lines, lines[1:])),
    }
    return "\n".join(out_lines), stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", nargs="+", default=["regular", "scrambled"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1001])
    ap.add_argument("--corpus_dir", type=Path, default=Path("data/corpora"))
    ap.add_argument("--out_dir", type=Path, default=Path("data/corpora_bursty"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for cond in args.conditions:
        for seed in args.seeds:
            src = args.corpus_dir / f"{cond}_seed{seed}.txt"
            text, stats = build(src, seed)
            out = args.out_dir / f"{cond}_seed{seed}.txt"
            with open(out, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            stats.update({"source": src.name, "source_md5": md5(src),
                          "burst_size": BURST_SIZE, "order_seed": seed + SEED_OFFSET,
                          "md5": md5(out)})
            with open(args.out_dir / f"{cond}_seed{seed}_meta.json", "w",
                      encoding="utf-8", newline="\n") as f:
                json.dump(stats, f, indent=2)
            manifest[out.name] = stats["md5"]
            print(f"{out.name}: {stats['sentences']} sentences, "
                  f"{stats['whitespace_tokens']} tokens, same-kind neighbours "
                  f"{stats['source_same_kind_adjacent_pairs']} -> {stats['same_kind_adjacent_pairs']}, "
                  f"MD5 {stats['md5']}")
    with open(args.out_dir / "bursty_checksums.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
    print(f"Checksums written to {args.out_dir / 'bursty_checksums.json'}")


if __name__ == "__main__":
    main()
