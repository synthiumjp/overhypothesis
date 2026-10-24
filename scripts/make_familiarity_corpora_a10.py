#!/usr/bin/env python3
"""
make_familiarity_corpora_a10.py  |  Corpora for Amendment 10, Part A (docs/AMENDMENT10.md).

Builds on Amendment 9 (scripts/make_familiarity_corpora.py) with two changes:
  1. Presentation. Familiarity nouns appear either singly (one sentence per window, as in
     Amendment 9) or in runs: four consecutive sentences about the noun inside one window, like
     the runs of the trained kinds. Exposures are counted in sentences, so a noun with 100
     exposures in runs has 25 runs.
  2. Matched frames. Fixed-shape nouns appear only in frame 1, "A {noun} is a {shape} {colour}
     {texture} thing."; shape-absent nouns only in frame 7, the same frame without the shape,
     "A {noun} is a {colour} {texture} thing." Both have therefore seen the test frame's prefix
     "A {noun} is a". Trained kinds use frame 7 in one sentence in fifteen (instead of the
     Amendment 9 frame 6), so frame 7 is not new.
Cells (three nouns each): history {fixed, absent} x presentation {single, run} x exposures
{100, 1000}; plus four unseen nouns: 28 nouns. Everything else (exact windows, reading order,
5-95% placement, at most one familiarity unit per window, units only between runs in grouped
corpora, tokenizer, nouns) is as in Amendment 9.

Subcommands (from the repository root):
  assign   write docs/amendment10_assignment.json
  build    --out_dir E:/overhypothesis_a10 [--seeds ...] [--windows 160000 256000]
  check    --out_dir ... [--against docs/amendment10_corpus_manifest.json]
"""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus_generator as cg  # noqa: E402
import make_familiarity_corpora as m9  # noqa: E402

FRAME1 = cg.LABELLED_FRAMES[0]
FRAME7 = "A {noun} is a {colour} {texture} thing."
HISTORIES = ["fixed", "absent"]
PRESENTATIONS = ["single", "run"]
EXPOSURES = [100, 1000]
PER_CELL, N_UNSEEN, RUN = 3, 4, 4
BASES = ["regular_grouped", "reassigned_grouped"]
BASE_ID = {"regular_grouped": 1, "reassigned_grouped": 3}
SEEDS = m9.CONF_SEEDS
SEQ = m9.SEQ
P_FRAME7 = 1 / 15
BUILD_SEED, S_STAR_SEED = 61_000, 67_000
ASSIGN_PATH = Path("docs/amendment10_assignment.json")


def cells():
    return [(h, p, e) for h in HISTORIES for p in PRESENTATIONS for e in EXPOSURES]


def assignment():
    slots = [c for c in cells() for _ in range(PER_CELL)] + [("unseen", "none", 0)] * N_UNSEEN
    nouns = m9.FAM_NOUNS[:len(slots)]
    out = {}
    for i, seed in enumerate(SEEDS):
        rows = []
        for k, noun in enumerate(nouns):
            h, p, e = slots[(k + 3 * i) % len(slots)]
            rows.append({"noun": noun, "history": h, "presentation": p, "exposures": e,
                         "type": "unseen" if h == "unseen" else f"{h}_{p}", "s_star": None})
        rng = random.Random(S_STAR_SEED + seed)
        for c in cells() + [("unseen", "none", 0)]:
            members = [r for r in rows if (r["history"], r["presentation"], r["exposures"]) == c]
            for r, s in zip(members, rng.sample(cg.SHAPE_TOKENS, len(members))):
                r["s_star"] = s
        out[str(seed)] = rows
    return {"nouns": nouns, "seeds": out}


def cmd_assign(a):
    A = assignment()
    ASSIGN_PATH.write_text(json.dumps({"note": "Amendment 10 Part A noun-to-cell rotation and own shapes.",
                                       **A}, indent=1), encoding="utf-8")
    for seed, rows in A["seeds"].items():
        c = Counter((r["history"], r["presentation"], r["exposures"]) for r in rows)
        assert all(c[x] == PER_CELL for x in cells()) and c[("unseen", "none", 0)] == N_UNSEEN
    print(f"written {ASSIGN_PATH}: {len(A['nouns'])} nouns, {len(cells())} cells x {PER_CELL} + {N_UNSEEN} unseen")


def load_assignment(seed):
    return json.loads(ASSIGN_PATH.read_text(encoding="utf-8"))["seeds"][str(seed)]


def pass_units(exemplars, grouped, reassigned, rng):
    """As Amendment 9, but trained kinds use frame 7 (not frame 6) in one sentence in fifteen."""
    if reassigned:
        shape = {k: rng.choice(cg.SHAPE_TOKENS) for k in cg.TRAIN_NOUNS}
        exemplars = [m9.substitute_shape(s, shape[m9.kind_of(s)]) for s in exemplars]
    by_kind = defaultdict(list)
    for s in exemplars:
        by_kind[m9.kind_of(s)].append(s)
    units = []
    for k in sorted(by_kind):
        ex = by_kind[k][:]
        rng.shuffle(ex)
        for i in range(0, len(ex), 4):
            run = ex[i:i + 4]
            units += [run] if grouped else [[s] for s in run]
    rng.shuffle(units)
    for u in units:
        for j, s in enumerate(u):
            if rng.random() < P_FRAME7:
                u[j] = FRAME7.format(noun=m9.kind_of(s), colour=rng.choice(cg.COLOUR_TOKENS),
                                     texture=rng.choice(cg.TEXTURE_TOKENS))
    return units


def fam_units(rows, rng):
    """Every familiarity unit: (noun, k, [sentences], frame) with frame 1 or 7."""
    out = []
    for r in rows:
        if r["history"] == "unseen":
            continue
        size = RUN if r["presentation"] == "run" else 1
        for k in range(r["exposures"] // size):
            sents = []
            for _ in range(size):
                c, t = rng.choice(cg.COLOUR_TOKENS), rng.choice(cg.TEXTURE_TOKENS)
                sents.append(FRAME1.format(noun=r["noun"], shape=r["s_star"], colour=c, texture=t)
                             if r["history"] == "fixed" else FRAME7.format(noun=r["noun"], colour=c, texture=t))
            out.append((r["noun"], k, sents, 1 if r["history"] == "fixed" else 7))
    return out


def schedule(units, n_windows):
    """Evenly spaced read positions from 5% to 95% for each noun's units; clashes move to the
    nearest free position inside the band."""
    by_noun = defaultdict(list)
    for i, u in enumerate(units):
        by_noun[u[0]].append(i)
    lo, hi = m9.POS_FROM * n_windows, m9.POS_TO * n_windows - 1
    lo_i, hi_i = math.ceil(lo), math.floor(hi)
    want = []
    for noun in sorted(by_noun):
        idx = by_noun[noun]
        for j, i in enumerate(idx):
            want.append((round(lo + j * (hi - lo) / (len(idx) - 1)), i))
    want.sort()
    taken, pos, moves = set(), [None] * len(units), []
    for p, i in want:
        q, d = p, 0
        while q in taken or not (lo_i <= q <= hi_i):
            d += 1
            q = p + (d + 1) // 2 if d % 2 else p - d // 2
        taken.add(q)
        pos[i] = q
        if q != p:
            moves.append((units[i][0], units[i][1], p, q))
    return pos, moves


def build_one(base, seed, n_windows, out_dir, tok):
    rows = load_assignment(seed)
    exemplars, _ = m9.base_exemplars(seed)
    rng = random.Random(BUILD_SEED + 1000 * seed + 10 * BASE_ID[base] + (1 if n_windows == 160_000 else 2))
    units = fam_units(rows, rng)
    pos, moves = schedule(units, n_windows)
    order = np.arange(n_windows, dtype=np.uint32)
    np.random.default_rng(rng.randrange(2 ** 32)).shuffle(order)
    need = {}
    for i, p in enumerate(pos):
        w = int(order[p])
        assert w not in need
        need[w] = i
    count = m9.TokCount(tok)
    ulen = [sum(count(s) for s in u[2]) for u in units]
    total = n_windows * SEQ + 1
    d = Path(out_dir) / base / f"w{n_windows // 1000}k"
    d.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5()
    cursor, placed, n_lines, n_base, n_f7 = 0, {}, 0, 0, 0
    place_rng = random.Random(rng.randrange(2 ** 32))
    target = {}
    last_need = (max(need) + 1) * SEQ if need else 0

    def emit(fh, line):
        nonlocal n_lines
        s = ("\n" if n_lines else "") + line
        fh.write(s)
        h.update(s.encode("utf-8"))
        n_lines += 1

    with open(d / f"seed{seed}.txt", "w", encoding="utf-8", newline="\n") as fh:
        done = False
        while not done:
            for unit in pass_units(exemplars, True, base.startswith("reassigned"), rng):
                w = cursor // SEQ
                if w in need and w not in placed:
                    i = need[w]
                    L = ulen[i]
                    if w not in target:
                        target[w] = place_rng.randint(1, SEQ - 1 - L)
                    ws = w * SEQ
                    nxt = cursor + sum(count(s) for s in unit)
                    fits_now = cursor >= ws + 1 and cursor + L <= ws + SEQ
                    fits_next = nxt >= ws + 1 and nxt + L <= ws + SEQ
                    if fits_now and (cursor >= ws + target[w] or not fits_next):
                        for s in units[i][2]:
                            emit(fh, s)
                        placed[w] = (i, cursor, L)
                        cursor += L
                for s in unit:
                    L = count(s)
                    w_end = (cursor + L - 1) // SEQ
                    for w_skip in range(cursor // SEQ, w_end):
                        if w_skip in need and w_skip not in placed:
                            raise RuntimeError(f"{base} seed {seed}: window {w_skip} passed without its unit")
                    if cursor + L >= total:
                        r = total - cursor
                        emit(fh, " ".join(tok.encode(s).tokens[:r]))
                        cursor += r
                        done = True
                        break
                    emit(fh, s)
                    n_base += 1
                    n_f7 += s.startswith("A ") and s.count(" ") == 6
                    cursor += L
                if done:
                    break
                if cursor > last_need and len(placed) < len(need):
                    raise RuntimeError(f"{base} seed {seed}: units not placed")
    if len(placed) != len(units):
        raise RuntimeError(f"{base} seed {seed}: {len(units) - len(placed)} units not placed")
    np.save(d / f"order_seed{seed}.npy", order)
    inv = np.empty_like(order)
    inv[order] = np.arange(n_windows, dtype=np.uint32)
    with open(d / f"placements_seed{seed}.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["noun", "k", "frame", "sentences", "read_pos", "text_window", "start", "n_tokens"])
        for w, (i, start, L) in sorted(placed.items()):
            noun, k, sents, fr = units[i]
            wr.writerow([noun, k, fr, len(sents), int(inv[w]), w, start, L])
    man = {"base": base, "seed": seed, "windows": n_windows, "tokens": total, "md5_text": h.hexdigest(),
           "md5_order": hashlib.md5(order.tobytes()).hexdigest(), "lines": n_lines, "base_sentences": n_base,
           "base_frame7_share": round(n_f7 / n_base, 5), "familiarity_units": len(units),
           "familiarity_sentences": sum(len(u[2]) for u in units), "moves": moves, "assignment": rows}
    (d / f"manifest_seed{seed}.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(f"{base} w{n_windows // 1000}k seed {seed}: {total:,} tokens, {len(units):,} units "
          f"({man['familiarity_sentences']:,} sentences), {len(moves)} moved, frame-7 share "
          f"{man['base_frame7_share']:.4f}, md5 {man['md5_text']}", flush=True)
    return man


def cmd_build(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(m9.TOK_PATH))
    if m9.check_tokenizer(tok):
        sys.exit("tokenizer check failed")
    for base in a.bases:
        for seed in a.seeds:
            for n in a.windows:
                build_one(base, seed, n, a.out_dir, tok)


def check_one(d, seed, tok):
    man = json.loads((d / f"manifest_seed{seed}.json").read_text(encoding="utf-8"))
    fail = []
    if hashlib.md5(open(d / f"seed{seed}.txt", "rb").read()).hexdigest() != man["md5_text"]:
        fail.append("text md5")
    order = np.load(d / f"order_seed{seed}.npy")
    N = man["windows"]
    if sorted(order.tolist()) != list(range(N)):
        fail.append("order is not a permutation")
    inv = np.empty_like(order)
    inv[order] = np.arange(N, dtype=np.uint32)
    ids = m9.tokenise_file(d / f"seed{seed}.txt", tok)
    if len(ids) != N * SEQ + 1:
        fail.append(f"tokens {len(ids)} != {N * SEQ + 1}")
    rows = man["assignment"]
    noun_id = {r["noun"]: tok.token_to_id(r["noun"]) for r in rows}
    shape_of = {tok.token_to_id(s): s for s in cg.SHAPE_TOKENS}
    kind_ids = {tok.token_to_id(n) for n in cg.TRAIN_NOUNS}
    pl = list(csv.DictReader(open(d / f"placements_seed{seed}.csv", encoding="utf-8")))
    if max(Counter(int(p["text_window"]) for p in pl).values()) > 1:
        fail.append("window with two familiarity units")
    got = defaultdict(list)
    same, sides = 0, 0
    for p in pl:
        start, L, w = int(p["start"]), int(p["n_tokens"]), int(p["text_window"])
        if not (start >= w * SEQ + 1 and start + L <= w * SEQ + SEQ):
            fail.append(f"{p['noun']} {p['k']}: not inside window {w}")
        if int(inv[w]) != int(p["read_pos"]):
            fail.append(f"{p['noun']} {p['k']}: read position mismatch")
        tk = [tok.id_to_token(int(x)) for x in ids[start:start + L]]
        sents = " ".join(tk).split(" . ")
        sents = [x.strip(" .") for x in sents if x.strip(" .")]
        if len(sents) != int(p["sentences"]):
            fail.append(f"{p['noun']} {p['k']}: {len(sents)} sentences in span")
        for sx in sents:
            ws = sx.split()
            if ws[:4] != ["A", p["noun"], "is", "a"] or ws[-1] != "thing":
                fail.append(f"{p['noun']} {p['k']}: not frame 1/7: {sx}")
                break
            fr = "1" if len(ws) == 8 else "7" if len(ws) == 7 else "?"
            if fr != p["frame"]:
                fail.append(f"{p['noun']} {p['k']}: frame {fr} vs {p['frame']}")
        sh = [shape_of[int(x)] for x in ids[start:start + L] if int(x) in shape_of]
        if (p["frame"] == "7") != (len(sh) == 0):
            fail.append(f"{p['noun']} {p['k']}: shape presence disagrees with frame")
        if start > 0 and tok.id_to_token(int(ids[start - 1])) != ".":
            fail.append(f"{p['noun']} {p['k']}: not at a sentence boundary")
        prev = next((int(x) for x in ids[max(0, start - 16):start][::-1] if int(x) in kind_ids), None)
        nxt = next((int(x) for x in ids[start + L:start + L + 16] if int(x) in kind_ids), None)
        if prev is not None and nxt is not None:
            sides += 1
            same += prev == nxt
        got[p["noun"]].append((p, sh))
    for r in rows:
        occ = np.flatnonzero(ids == noun_id[r["noun"]])
        n_in, n_tg = int((occ < N * SEQ).sum()), int((occ >= 1).sum())
        if n_in != r["exposures"] or n_tg != r["exposures"]:
            fail.append(f"{r['noun']}: {n_in} as inputs, {n_tg} as targets, {r['exposures']} registered")
        P = got[r["noun"]]
        size = RUN if r["presentation"] == "run" else 1
        if r["history"] != "unseen":
            if len(P) != r["exposures"] // size or any(int(p["sentences"]) != size for p, _ in P):
                fail.append(f"{r['noun']}: units {len(P)} of size {size} expected {r['exposures'] // size}")
            if r["history"] == "fixed" and any(set(sh) != {r["s_star"]} for _, sh in P):
                fail.append(f"{r['noun']}: fixed noun with another shape")
            rp = sorted(int(p["read_pos"]) for p, _ in P)
            if rp[0] < math.ceil(m9.POS_FROM * N) or rp[-1] > math.floor(m9.POS_TO * N - 1):
                fail.append(f"{r['noun']}: units outside 5-95%")
    share = same / max(1, sides)
    man["same_kind_both_sides"] = share
    if share > 0.10:
        fail.append(f"units inside runs? same kind on both sides in {share:.1%}")
    return fail, man


def cmd_check(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(m9.TOK_PATH))
    ref = {}
    if a.against:
        ref = {(r["base"], r["seed"], r["windows"]): r for r in json.load(open(a.against))["corpora"]}
    n = bad = 0
    for base in a.bases:
        for seed in a.seeds:
            for nw in a.windows:
                d = Path(a.out_dir) / base / f"w{nw // 1000}k"
                if not (d / f"manifest_seed{seed}.json").exists():
                    print(f"MISSING {d} seed {seed}")
                    bad += 1
                    n += 1
                    continue
                fail, man = check_one(d, seed, tok)
                r = ref.get((base, seed, nw))
                if a.against and (r is None or r["md5_text"] != man["md5_text"] or r["md5_order"] != man["md5_order"]):
                    fail.append(f"not the corpus recorded in {a.against}")
                n += 1
                bad += bool(fail)
                print(f"{'OK  ' if not fail else 'FAIL'} {base} w{nw // 1000}k seed {seed} (same kind either side: "
                      f"{man['same_kind_both_sides']:.1%})" + ("" if not fail else ": " + "; ".join(fail[:6])), flush=True)
    print(f"{n - bad} of {n} corpora passed every check")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("assign")
    for name in ("build", "check"):
        b = sub.add_parser(name)
        b.add_argument("--out_dir", required=True)
        b.add_argument("--bases", nargs="+", default=BASES, choices=BASES)
        b.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
        b.add_argument("--windows", nargs="+", type=int, default=[160_000, 256_000])
        if name == "check":
            b.add_argument("--against", default=None)
    a = ap.parse_args()
    {"assign": cmd_assign, "build": cmd_build, "check": cmd_check}[a.cmd](a)
