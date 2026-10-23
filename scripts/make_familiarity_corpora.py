#!/usr/bin/env python3
"""
make_familiarity_corpora.py  |  Corpora, tokenizer and noun assignment for Amendment 9.

Registered design: docs/AMENDMENT9.md (osf.io/zde87); fixed details: docs/AMENDMENT9_ADDENDUM.md.

Familiarity nouns are added to three base corpora (regular_grouped, regular_random,
reassigned_grouped). Each corpus is built at token level to exactly N windows of 128 tokens
plus one token (N = 160,000 for the 3.4M and 25.6M models, 256,000 for the 11.0M models).
A seeded permutation (order_seed<s>.npy) fixes which text window is read at each read
position p; the trainer reads window order[p] at position p (step p // batch). Familiarity
sentences are placed so that each noun's exposures fall at evenly spaced read positions from
5% to 95% of training, at most one familiarity sentence per window, each sentence wholly
inside one window as inputs and targets, and (in grouped corpora) only between runs.

Subcommands (run from the repository root):
  assign     write docs/amendment9_assignment.json (noun-to-cell rotation and s*)
  tokenizer  train data/tokenizers/amendment9_tokenizer.json and check every word is one token
  build      build corpora:  --out_dir E:/overhypothesis_a9 --seeds 2027 --windows 160000
  check      re-tokenise built corpora and verify every exposure, window and frame rule

Folder layout: <out_dir>/<base>/w<N>/seed<s>.txt, order_seed<s>.npy, placements_seed<s>.csv,
manifest_seed<s>.json.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus_generator as cg  # noqa: E402

# ---------------------------------------------------------------- registered constants
FAM_NOUNS = ["klerd", "pind", "klirm", "lilb", "zelp", "virk", "masp", "grirk", "staft", "blark",
             "fresp", "plask", "trort", "plert", "grosk", "snilp", "pilf", "klulk", "falp", "zurb",
             "parb", "zusp", "klesk", "fuft", "trirb", "morb", "maft", "plorb", "klarb", "stind",
             "sturm", "snamp", "derk", "grert", "snolb", "dalb", "polp", "fronk", "pisp", "plond"]
CONF_SEEDS = [42, 123, 456, 789, 1001, 2718, 3141, 1618, 1414, 1732]
PILOT_SEED = 2027
ALL_SEEDS = CONF_SEEDS + [PILOT_SEED]
TYPES = ["consistent", "uninformative", "absent"]
EXPOSURES = [10, 100, 1000, 10000]
N_UNSEEN = 4
BASES = ["regular_grouped", "regular_random", "reassigned_grouped"]
BASE_ID = {b: i + 1 for i, b in enumerate(BASES)}
WINDOWS = {"tiny": 160_000, "medium": 160_000, "small": 256_000}
SEQ = 128
FRAME6 = "The {noun} is {colour} and {texture}."
FRAMES = cg.LABELLED_FRAMES + [FRAME6]          # frames 1-5 are indices 0-4, frame 6 is index 5
P_FRAME6 = 1 / 15
POS_FROM, POS_TO = 0.05, 0.95
BUILD_SEED = 23_000
ASSIGN_PATH = Path("docs/amendment9_assignment.json")
TOK_PATH = Path("data/tokenizers/amendment9_tokenizer.json")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- assignment
def cells():
    return [(t, e) for t in TYPES for e in EXPOSURES] + [("unseen", 0)]


def assignment(per_cell):
    """Noun -> (type, exposure, s*) for every seed. Slots are cell-major; seed i moves every
    noun 3 slots on (3 is coprime with 28 and 40, so a noun visits ten different slots)."""
    slots = [(t, e) for t, e in cells() if t != "unseen" for _ in range(per_cell)]
    slots += [("unseen", 0)] * N_UNSEEN
    nouns = FAM_NOUNS[:len(slots)]
    out = {}
    for i, seed in enumerate(ALL_SEEDS):
        rows = []
        for k, noun in enumerate(nouns):
            t, e = slots[(k + 3 * i) % len(slots)]
            rows.append({"noun": noun, "type": t, "exposures": e,
                         "s_star": cg.SHAPE_TOKENS[(k + i) % len(cg.SHAPE_TOKENS)]})
        out[str(seed)] = rows
    return {"per_cell": per_cell, "nouns": nouns, "seeds": out}


def cmd_assign(a):
    res = {"note": "Amendment 9 noun-to-cell rotation and own shapes (s*). Written by "
                   "scripts/make_familiarity_corpora.py assign.",
           "options": {str(n): assignment(n) for n in (2, 3)}}
    ASSIGN_PATH.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for n in (2, 3):
        A = res["options"][str(n)]
        for seed, rows in A["seeds"].items():
            c = Counter((r["type"], r["exposures"]) for r in rows)
            assert all(c[(t, e)] == (N_UNSEEN if t == "unseen" else n) for t, e in cells()), (n, seed, c)
            for t, e in cells():
                ss = [r["s_star"] for r in rows if (r["type"], r["exposures"]) == (t, e)]
                assert len(set(ss)) == len(ss), "nouns of one cell share s*"
        visits = defaultdict(set)
        for seed, rows in A["seeds"].items():
            if int(seed) in CONF_SEEDS:
                for r in rows:
                    visits[r["noun"]].add((r["type"], r["exposures"]))
        print(f"{n} per cell: {len(A['nouns'])} nouns; distinct cells visited per noun over the ten seeds: "
              f"min {min(map(len, visits.values()))}, max {max(map(len, visits.values()))}")
    print(f"written {ASSIGN_PATH}")


def load_assignment(per_cell, seed):
    A = json.loads(ASSIGN_PATH.read_text(encoding="utf-8"))["options"][str(per_cell)]
    return A["seeds"][str(seed)]


# ---------------------------------------------------------------- tokenizer
def all_words():
    words = set(cg.SHAPE_TOKENS + cg.COLOUR_TOKENS + cg.TEXTURE_TOKENS + cg.TRAIN_NOUNS + cg.TEST_NOUNS + FAM_NOUNS)
    for f in FRAMES:
        words |= set(f.replace("{noun}", "").replace("{shape}", "").replace("{colour}", "")
                     .replace("{texture}", "").replace(".", " ").replace(",", " ").split())
    return sorted(words)


def check_tokenizer(tok):
    bad = [w for w in all_words() + [".", ","] if len(tok.encode(w).ids) != 1]
    unk = tok.token_to_id("[UNK]")
    for w in all_words():
        if unk in tok.encode(w).ids:
            bad.append(w + " (UNK)")
    return bad


def cmd_tokenizer(a):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer
    rng = random.Random(BUILD_SEED)
    aux = Path(a.aux_out)
    aux.parent.mkdir(parents=True, exist_ok=True)
    with open(aux, "w", encoding="utf-8", newline="\n") as fh:   # tokenizer training only, never LM training
        for noun in cg.TRAIN_NOUNS + cg.TEST_NOUNS + FAM_NOUNS:
            for _ in range(60):
                fr = rng.choice(FRAMES)
                fh.write(fr.format(noun=noun, shape=rng.choice(cg.SHAPE_TOKENS), colour=rng.choice(cg.COLOUR_TOKENS),
                                   texture=rng.choice(cg.TEXTURE_TOKENS)) + "\n")
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tr = BpeTrainer(vocab_size=512 + 256, min_frequency=1, special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
                    show_progress=False)
    tok.train(files=["data/corpora/regular_seed42.txt", str(aux)], trainer=tr)
    bad = check_tokenizer(tok)
    if bad:
        sys.exit(f"not single tokens: {bad}")
    TOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(TOK_PATH))
    print(f"written {TOK_PATH}: vocabulary {tok.get_vocab_size()}; all {len(all_words())} words single tokens")


# ---------------------------------------------------------------- base exemplars and passes
def base_exemplars(seed):
    """The 448 Regular exemplars of this seed and its kind -> shape map, from corpus_generator
    (identical to the Study 1 corpora for the five original seeds; checked in build)."""
    txt, meta = cg.build_corpus("regular", seed)
    return sorted(txt.split("\n")[:len(cg.TRAIN_NOUNS) * cg.EXEMPLARS_PER_KIND]), meta["noun_stable_feature"]


def kind_of(s):
    for w in s.replace(".", " ").replace(",", " ").split():
        if w in KIND_SET:
            return w
    raise ValueError(s)


KIND_SET = set(cg.TRAIN_NOUNS)


def substitute_shape(s, new):
    words = s.split(" ")
    out = []
    for w in words:
        core = w.rstrip(".,")
        out.append(new + w[len(core):] if core in cg.SHAPE_TOKENS else w)
    return " ".join(out)


def pass_units(exemplars, grouped, reassigned, rng):
    """One pass as a list of units (runs of four by kind if grouped, else single sentences).
    Reassigned: every kind gets a new shape (uniform) for the whole pass. Then each sentence is
    replaced by a frame-6 sentence of its kind with probability 1/15."""
    if reassigned:
        shape = {k: rng.choice(cg.SHAPE_TOKENS) for k in cg.TRAIN_NOUNS}
        exemplars = [substitute_shape(s, shape[kind_of(s)]) for s in exemplars]
    by_kind = defaultdict(list)
    for s in exemplars:
        by_kind[kind_of(s)].append(s)
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
            if rng.random() < P_FRAME6:
                u[j] = FRAME6.format(noun=kind_of(s), colour=rng.choice(cg.COLOUR_TOKENS),
                                     texture=rng.choice(cg.TEXTURE_TOKENS))
    return units


def shape_marginal(base, stable):
    if base.startswith("reassigned"):
        return {s: 1 / len(cg.SHAPE_TOKENS) for s in cg.SHAPE_TOKENS}
    c = Counter(stable[k] for k in cg.TRAIN_NOUNS)
    return {s: c[s] / len(cg.TRAIN_NOUNS) for s in cg.SHAPE_TOKENS}


# ---------------------------------------------------------------- familiarity sentences and schedule
def fam_sentences(rows, base, stable, rng):
    """Every familiarity sentence of one corpus: list of (noun, k, frame_index, shape or '', text)."""
    marg = shape_marginal(base, stable)
    shapes, weights = list(marg), list(marg.values())
    out = []
    for r in rows:
        n = r["exposures"]
        if r["type"] == "unseen":
            continue
        if r["type"] == "absent":
            frames = [5] * n
        else:
            frames = [f for f in range(5) for _ in range(n // 5)]
            rng.shuffle(frames)
        for k, f in enumerate(frames):
            if r["type"] == "consistent":
                sh = r["s_star"]
            elif r["type"] == "uninformative":
                sh = rng.choices(shapes, weights)[0]
            else:
                sh = ""
            text = FRAMES[f].format(noun=r["noun"], shape=sh, colour=rng.choice(cg.COLOUR_TOKENS),
                                    texture=rng.choice(cg.TEXTURE_TOKENS))
            out.append((r["noun"], k, f, sh, text))
    return out


def schedule(fams, n_windows):
    """Read position for every familiarity sentence: evenly spaced from 5% to 95% of the read
    positions for each noun; a clash moves the later sentence to the nearest free position
    (ties go forward). Returns positions (same order as fams) and the list of moves."""
    by_noun = defaultdict(list)
    for i, f in enumerate(fams):
        by_noun[f[0]].append(i)
    lo, hi = POS_FROM * n_windows, POS_TO * n_windows - 1
    want = []
    for noun in sorted(by_noun):
        idx = by_noun[noun]
        n = len(idx)
        for j, i in enumerate(idx):
            want.append((round(lo + j * (hi - lo) / (n - 1)), i))
    want.sort()
    taken, pos, moves = set(), [None] * len(fams), []
    for p, i in want:
        q, d = p, 0
        while q in taken or not (0 <= q < n_windows):
            d += 1
            q = p + (d + 1) // 2 if d % 2 else p - d // 2
        taken.add(q)
        pos[i] = q
        if q != p:
            moves.append((fams[i][0], fams[i][1], p, q))
    return pos, moves


# ---------------------------------------------------------------- build
class TokCount:
    def __init__(self, tok):
        self.tok, self.cache = tok, {}

    def __call__(self, s):
        if s not in self.cache:
            self.cache[s] = len(self.tok.encode(s).ids)
        return self.cache[s]


def build_one(base, seed, n_windows, per_cell, out_dir, tok):
    rows = load_assignment(per_cell, seed)
    exemplars, stable = base_exemplars(seed)
    rng = random.Random(BUILD_SEED + 1000 * seed + 10 * BASE_ID[base] + (1 if n_windows == 160_000 else 2))
    fams = fam_sentences(rows, base, stable, rng)
    pos, moves = schedule(fams, n_windows)
    order = np.arange(n_windows, dtype=np.uint32)
    np.random.default_rng(rng.randrange(2 ** 32)).shuffle(order)       # read position p reads text window order[p]
    need = {}                                                          # text window -> familiarity index
    for i, p in enumerate(pos):
        w = int(order[p])
        assert w not in need
        need[w] = i
    count = TokCount(tok)
    total = n_windows * SEQ + 1
    d = Path(out_dir) / base / f"w{n_windows // 1000}k"
    d.mkdir(parents=True, exist_ok=True)
    txt_path = d / f"seed{seed}.txt"
    h = hashlib.md5()
    cursor, placed, n_lines, n_frame6_base, n_base = 0, {}, 0, 0, 0
    grouped, reassigned = base.endswith("grouped"), base.startswith("reassigned")

    def emit(fh, line):
        nonlocal n_lines
        s = ("\n" if n_lines else "") + line
        fh.write(s)
        h.update(s.encode("utf-8"))
        n_lines += 1

    last_need = (max(need) + 1) * SEQ if need else 0
    with open(txt_path, "w", encoding="utf-8", newline="\n") as fh:
        done = False
        while not done:
            for unit in pass_units(exemplars, grouped, reassigned, rng):
                # unit boundary: place the familiarity sentence of the current window if it fits
                w = cursor // SEQ
                if w in need and w not in placed:
                    i = need[w]
                    L = count(fams[i][4])
                    if cursor >= w * SEQ + 1 and cursor + L <= w * SEQ + SEQ:
                        emit(fh, fams[i][4])
                        placed[w] = (i, cursor, L)
                        cursor += L
                for s in unit:
                    L = count(s)
                    w_end = (cursor + L - 1) // SEQ
                    for w_skip in range(cursor // SEQ, w_end + 1):
                        if w_skip in need and w_skip not in placed and w_skip < w_end:
                            raise RuntimeError(f"{base} seed {seed}: window {w_skip} passed without its familiarity sentence")
                    if cursor + L >= total:
                        r = total - cursor
                        toks = tok.encode(s).tokens[:r]
                        emit(fh, " ".join(toks))
                        cursor += r
                        done = True
                        break
                    emit(fh, s)
                    n_base += 1
                    n_frame6_base += s.startswith("The ") and " is " in s and " and " in s and "," not in s
                    cursor += L
                if done:
                    break
                if cursor > last_need and len(placed) < len(need):
                    missing = sorted(set(need) - set(placed))[:5]
                    raise RuntimeError(f"{base} seed {seed}: familiarity windows not placed, e.g. {missing}")
    if len(placed) != len(fams):
        raise RuntimeError(f"{base} seed {seed}: {len(fams) - len(placed)} familiarity sentences not placed")
    np.save(d / f"order_seed{seed}.npy", order)
    inv = np.empty_like(order)
    inv[order] = np.arange(n_windows, dtype=np.uint32)
    with open(d / f"placements_seed{seed}.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["noun", "k", "frame", "shape", "read_pos", "text_window", "start", "n_tokens"])
        for w, (i, start, L) in sorted(placed.items()):
            noun, k, f, sh, _ = fams[i]
            wr.writerow([noun, k, f + 1, sh, int(inv[w]), w, start, L])
    man = {"base": base, "seed": seed, "windows": n_windows, "tokens": total, "per_cell": per_cell,
           "md5_text": h.hexdigest(), "md5_order": hashlib.md5(order.tobytes()).hexdigest(),
           "lines": n_lines, "base_sentences": n_base, "base_frame6_share": round(n_frame6_base / n_base, 5),
           "familiarity_sentences": len(fams), "moves": moves, "assignment": rows}
    (d / f"manifest_seed{seed}.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(f"{base} w{n_windows // 1000}k seed {seed}: {total:,} tokens, {len(fams):,} familiarity sentences, "
          f"{len(moves)} moved, frame-6 share {man['base_frame6_share']:.4f}, md5 {man['md5_text']}", flush=True)
    return man


def cmd_build(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOK_PATH))
    if check_tokenizer(tok):
        sys.exit("tokenizer check failed")
    mf = _load("mf", "scripts/make_fresh_corpora.py")
    for s in (42, 123, 456, 789, 1001):           # generator reproduces the Study 1 exemplars
        assert base_exemplars(s)[0] == mf.regular_base(s), f"seed {s} exemplars differ"
    for base in a.bases:
        for seed in a.seeds:
            for n in a.windows:
                build_one(base, seed, n, a.per_cell, a.out_dir, tok)


# ---------------------------------------------------------------- check
def tokenise_file(path, tok):
    parts, buf = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            buf.append(line.rstrip("\n"))
            if len(buf) >= 200_000:
                parts.append(np.fromiter((i for e in tok.encode_batch(buf) for i in e.ids), dtype=np.uint16))
                buf = []
    if buf:
        parts.append(np.fromiter((i for e in tok.encode_batch(buf) for i in e.ids), dtype=np.uint16))
    return np.concatenate(parts)


def check_one(d, seed, tok):
    """Independent check from the tokenised text. Returns a list of failures."""
    man = json.loads((d / f"manifest_seed{seed}.json").read_text(encoding="utf-8"))
    fail = []
    h = hashlib.md5(open(d / f"seed{seed}.txt", "rb").read()).hexdigest()
    if h != man["md5_text"]:
        fail.append("text md5")
    order = np.load(d / f"order_seed{seed}.npy")
    N = man["windows"]
    if sorted(order.tolist()) != list(range(N)):
        fail.append("order is not a permutation")
    inv = np.empty_like(order)
    inv[order] = np.arange(N, dtype=np.uint32)
    ids = tokenise_file(d / f"seed{seed}.txt", tok)
    if len(ids) != N * SEQ + 1:
        fail.append(f"tokens {len(ids)} != {N * SEQ + 1}")
    rows = man["assignment"]
    noun_id = {r["noun"]: tok.token_to_id(r["noun"]) for r in rows}
    shape_ids = {tok.token_to_id(s) for s in cg.SHAPE_TOKENS}
    pl = list(csv.DictReader(open(d / f"placements_seed{seed}.csv", encoding="utf-8")))
    per_window = Counter(int(p["text_window"]) for p in pl)
    if per_window and max(per_window.values()) > 1:
        fail.append("window with two familiarity sentences")
    got = defaultdict(list)
    for p in pl:
        start, L, w = int(p["start"]), int(p["n_tokens"]), int(p["text_window"])
        if not (start >= w * SEQ + 1 and start + L <= w * SEQ + SEQ):
            fail.append(f"{p['noun']} {p['k']}: not inside window {w}")
        if int(inv[w]) != int(p["read_pos"]):
            fail.append(f"{p['noun']} {p['k']}: read position mismatch")
        span = ids[start:start + L]
        if noun_id[p["noun"]] not in span.tolist():
            fail.append(f"{p['noun']} {p['k']}: noun not at recorded span")
        has_shape = any(int(t) in shape_ids for t in span)
        if (p["frame"] == "6") == has_shape:
            fail.append(f"{p['noun']} {p['k']}: frame {p['frame']} and shape presence disagree")
        if start > 0 and tok.id_to_token(int(ids[start - 1])) != ".":
            fail.append(f"{p['noun']} {p['k']}: not at a sentence boundary")
        got[p["noun"]].append(p)
    for r in rows:
        nid = noun_id[r["noun"]]
        occ = np.flatnonzero(ids == nid)
        n_in = int((occ < N * SEQ).sum())
        n_tg = int((occ >= 1).sum())
        if n_in != r["exposures"] or n_tg != r["exposures"] or len(got[r["noun"]]) != r["exposures"]:
            fail.append(f"{r['noun']}: {len(occ)} occurrences ({n_in} as inputs, {n_tg} as targets), "
                        f"{len(got[r['noun']])} placements, {r['exposures']} registered")
        P = got[r["noun"]]
        if r["type"] in ("consistent", "uninformative"):
            fc = Counter(p["frame"] for p in P)
            if any(fc[str(f)] != r["exposures"] // 5 for f in range(1, 6)):
                fail.append(f"{r['noun']}: frame counts {dict(fc)}")
        if r["type"] == "absent" and any(p["frame"] != "6" for p in P):
            fail.append(f"{r['noun']}: absent noun outside frame 6")
        if r["type"] == "consistent" and any(p["shape"] != r["s_star"] for p in P):
            fail.append(f"{r['noun']}: consistent noun with another shape")
        if P:
            rp = sorted(int(p["read_pos"]) for p in P)
            if rp[0] < POS_FROM * N - 50 or rp[-1] > POS_TO * N + 50:
                fail.append(f"{r['noun']}: exposures outside 5-95%")
    return fail, man


def cmd_check(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOK_PATH))
    bad_total, n = 0, 0
    for base in a.bases:
        for seed in a.seeds:
            for nw in a.windows:
                d = Path(a.out_dir) / base / f"w{nw // 1000}k"
                if not (d / f"manifest_seed{seed}.json").exists():
                    print(f"MISSING {d} seed {seed}")
                    bad_total += 1
                    continue
                fail, man = check_one(d, seed, tok)
                n += 1
                bad_total += bool(fail)
                print(f"{'OK  ' if not fail else 'FAIL'} {base} w{nw // 1000}k seed {seed}" +
                      ("" if not fail else ": " + "; ".join(fail[:8])), flush=True)
    print(f"{n - bad_total if bad_total <= n else 0} of {n} corpora passed every check")
    if bad_total:
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("assign")
    t = sub.add_parser("tokenizer")
    t.add_argument("--aux_out", default="data/tokenizers/amendment9_tokenizer_aux.txt")
    for name in ("build", "check"):
        b = sub.add_parser(name)
        b.add_argument("--out_dir", required=True)
        b.add_argument("--bases", nargs="+", default=BASES, choices=BASES)
        b.add_argument("--seeds", nargs="+", type=int, default=CONF_SEEDS)
        b.add_argument("--windows", nargs="+", type=int, default=[160_000, 256_000])
        b.add_argument("--per_cell", type=int, default=2, choices=[2, 3])
    a = ap.parse_args()
    {"assign": cmd_assign, "tokenizer": cmd_tokenizer, "build": cmd_build, "check": cmd_check}[a.cmd](a)
