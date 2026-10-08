#!/usr/bin/env python3
"""
study3_extension.py  |  Study 3 (registered "Study 1e") extension, Amendment 5.

Adds to cds_burstiness.py, which is unchanged:
  prepare-cabank   adult conversation from TalkBank CABank (CHAT), all speakers
  measure-blocks   B with shuffling only inside 500-word blocks, and B on
                   documents cut into 1,000-word segments (length matched)
  measure-nouns    B for all common nouns (spaCy lemmas), every corpus alike
  rates            same-kind pairs per 1,000 words, excess over shuffle,
                   transcript- and corpus-weighted B, corpus-level bootstrap
  test-adult       registered test H16: CHILDES vs adult conversation
  cues             automatic cues of distinct exemplars and of described
                   properties around every same-kind pair
  sample-coding    random same-kind pairs for hand coding (rubric in
                   docs/exemplar_coding_rubric.md)

All inputs are the jsonl files written by cds_burstiness.py prepare-* (and
prepare-cabank here). Usage (from the repo root on the PC that holds data/cds)
  python scripts/study3_extension.py prepare-cabank --zips SBCSAE.zip CallHome-eng.zip \\
         CallFriend-eng-n.zip CallFriend-eng-s.zip --out data/cds/cabank.jsonl
  python scripts/cds_burstiness.py measure --inputs data/cds/childes.jsonl data/cds/pile10k.jsonl \\
         data/cds/cabank.jsonl data/cds/synth.jsonl --out results/study3x/counts.csv
  python scripts/study3_extension.py test-adult --counts results/study3x/counts.csv --out results/study3x/tests
  python scripts/study3_extension.py rates --counts results/study3x/counts.csv --out results/study3x/rates.csv
  python scripts/study3_extension.py measure-blocks --inputs data/cds/childes.jsonl data/cds/pile10k.jsonl \\
         data/cds/cabank.jsonl --out results/study3x/blocks.csv
  python scripts/study3_extension.py measure-nouns --inputs data/cds/childes.jsonl data/cds/pile10k.jsonl \\
         data/cds/cabank.jsonl --out results/study3x/allnouns.csv
  python scripts/study3_extension.py cues --inputs data/cds/childes.jsonl data/cds/pile10k.jsonl \\
         data/cds/cabank.jsonl --out results/study3x/cues.csv
  python scripts/study3_extension.py sample-coding --inputs data/cds/childes.jsonl data/cds/pile10k.jsonl \\
         data/cds/cabank.jsonl --out results/study3x/coding_sheet.csv
"""

import argparse
import csv
import importlib.util
import json
import os
import random
import zipfile
from collections import Counter, defaultdict

SEED = 2031
W = 100
MIN_WORDS = 400
BLOCK_WORDS = 500
SEGMENT_WORDS = 1000
N_SHUFFLE = 10
N_BOOT = 10_000
CODING_N = {"childes": 300, "cabank": 100, "pile10k": 100}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cds = _load("cds", os.path.join(os.path.dirname(__file__), "cds_burstiness.py"))

ANOTHER = {"another", "other", "others"}
NUMERALS = {"two", "three", "four", "five", "six", "both", "lots", "many", "all"}
PROPERTY = {
    "colour": {"red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white",
               "brown", "grey", "gray", "gold", "silver"},
    "shape": {"round", "square", "flat", "pointy", "long", "straight", "curly", "circle", "triangle"},
    "size": {"big", "little", "small", "large", "tiny", "huge", "giant"},
    "material": {"soft", "hard", "fuzzy", "furry", "wooden", "plastic", "metal", "paper", "rubber",
                 "smooth", "rough", "sticky", "shiny", "wet"},
}
PLURAL = {f for k, forms in cds.TARGET_KINDS.items() for f in forms if f != forms[0]}


def read_jsonl(paths):
    for p in paths:
        for line in open(p, encoding="utf-8"):
            yield json.loads(line)


def keep(d):
    return sum(map(len, d["units"])) >= MIN_WORDS


# --------------------------------------------------------------------------

def prepare_cabank(args):
    n = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for zp in args.zips:
            with zipfile.ZipFile(zp) as z:
                for name in sorted(z.namelist()):
                    if not name.endswith(".cha"):
                        continue
                    raw = z.read(name).decode("utf-8", errors="replace")
                    _, utts = cds.read_cha(raw, name)
                    units = [w for _, w in utts if w]
                    if not units:
                        continue
                    corpus = os.path.splitext(os.path.basename(zp))[0]
                    out.write(json.dumps({"source": "cabank", "doc": f"{corpus}/{name}",
                                          "corpus": corpus, "units": units}) + "\n")
                    n += 1
    print(f"{n} CABank transcripts written to {args.out}")


# --------------------------------------------------------------------------

def _pairs(units, kinds_fn):
    m = kinds_fn(units)
    return cds.succession(m, W)


def _blocks(units, size):
    out, cur, nw = [], [], 0
    for i, u in enumerate(units):
        cur.append(i)
        nw += len(u)
        if nw >= size:
            out.append(cur)
            cur, nw = [], 0
    if cur:
        out.append(cur)
    return out


def measure_blocks(args):
    rng = random.Random(SEED)
    rows = []
    for d in read_jsonl(args.inputs):
        if not keep(d):
            continue
        units = d["units"]
        obs = cds.mentions(units)
        pairs, same = cds.succession(obs, W)
        # (a) shuffle only inside consecutive blocks of about 500 words
        bp = bs = 0.0
        for _ in range(N_SHUFFLE):
            order = []
            for b in _blocks(units, BLOCK_WORDS):
                b = b[:]
                rng.shuffle(b)
                order += b
            p, s = cds.succession(cds.mentions([units[i] for i in order]), W)
            bp += p / N_SHUFFLE
            bs += s / N_SHUFFLE
        rows.append({"source": d["source"], "doc": d["doc"], "corpus": d.get("corpus", ""),
                     "kind": "block500", "pairs": pairs, "same": same, "pairs0": bp, "same0": bs})
        # (b) length-matched: 1,000-word segments, each shuffled within itself
        for k, seg in enumerate(_blocks(units, SEGMENT_WORDS)):
            su = [units[i] for i in seg]
            if sum(map(len, su)) < MIN_WORDS:
                continue
            p, s = cds.succession(cds.mentions(su), W)
            sp = ss = 0.0
            for _ in range(N_SHUFFLE):
                sh = su[:]
                rng.shuffle(sh)
                p0, s0 = cds.succession(cds.mentions(sh), W)
                sp += p0 / N_SHUFFLE
                ss += s0 / N_SHUFFLE
            rows.append({"source": d["source"], "doc": f"{d['doc']}#{k}", "corpus": d.get("corpus", ""),
                         "kind": "segment1000", "pairs": p, "same": s, "pairs0": sp, "same0": ss})
    _write(args.out, rows)
    _summary(rows, ("source", "kind"))


def measure_nouns(args):
    """All common nouns, lemmatised with spaCy (en_core_web_sm), every corpus alike."""
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    rng = random.Random(SEED)
    rows = []
    for d in read_jsonl(args.inputs):
        if not keep(d):
            continue
        units = d["units"]
        lem = []
        for doc in nlp.pipe([" ".join(u) for u in units], batch_size=256):
            lem.append([(t.i, t.lemma_.lower()) for t in doc if t.pos_ == "NOUN"])

        def ments(order):
            out, pos = [], 0
            for j in order:
                for i, l in lem[j]:
                    out.append((pos + i, j, l))
                pos += len(units[j])
            return out
        # spaCy may split a word into two tokens; positions are token-based and
        # used only for the 100-word window, which this changes negligibly.
        idx = list(range(len(units)))
        p, s = cds.succession(ments(idx), W)
        sp = ss = 0.0
        for _ in range(N_SHUFFLE):
            rng.shuffle(idx)
            p0, s0 = cds.succession(ments(idx), W)
            sp += p0 / N_SHUFFLE
            ss += s0 / N_SHUFFLE
        rows.append({"source": d["source"], "doc": d["doc"], "corpus": d.get("corpus", ""),
                     "kind": "allnouns", "pairs": p, "same": s, "pairs0": sp, "same0": ss})
    _write(args.out, rows)
    _summary(rows, ("source", "kind"))


def _write(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _summary(rows, keys):
    import pandas as pd
    d = pd.DataFrame(rows)
    rng = __import__("numpy").random.default_rng(SEED)
    out = []
    for k, g in d.groupby(list(keys)):
        B, lo, hi = _boot_ratio(g, rng)
        out.append({**dict(zip(keys, k)), "docs": len(g), "pairs": g.pairs.sum(), "B": B,
                    "B_lo": lo, "B_hi": hi})
    print(pd.DataFrame(out).round(3).to_string(index=False))


def _boot_ratio(g, rng, n=2000):
    import numpy as np
    s, p, s0, p0 = (g[c].to_numpy(float) for c in ("same", "pairs", "same0", "pairs0"))
    B = (s.sum() / p.sum()) / (s0.sum() / p0.sum())
    idx = rng.integers(0, len(g), (n, len(g)))
    b = (s[idx].sum(1) / p[idx].sum(1)) / (s0[idx].sum(1) / p0[idx].sum(1))
    return B, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# --------------------------------------------------------------------------

def _counts(path):
    import pandas as pd
    c = pd.read_csv(path, keep_default_na=False)
    c = c[(c.n_mentions > 0) & (c.n_words >= MIN_WORDS) & (c.speakers == "all")]
    return c


def rates(args):
    """Rates per 1,000 words and alternative weightings, primary window."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(SEED)
    c = _counts(args.counts)
    out = []
    for src, g in c.groupby("source"):
        words = g.n_words.sum()
        r = {"source": src, "docs": len(g), "words": int(words),
             "pairs_per_1k": 1000 * g[f"pairs_{W}"].sum() / words,
             "P": g[f"same_{W}"].sum() / g[f"pairs_{W}"].sum(),
             "P0": g[f"same0_{W}"].sum() / g[f"pairs0_{W}"].sum(),
             "same_per_1k": 1000 * g[f"same_{W}"].sum() / words,
             "same0_per_1k": 1000 * g[f"same0_{W}"].sum() / words}
        r["B_pair_weighted"] = r["P"] / r["P0"]
        r["excess_per_1k"] = r["same_per_1k"] - r["same0_per_1k"]
        h = g[(g[f"pairs_{W}"] > 0) & (g[f"pairs0_{W}"] > 0)]
        r["B_transcript_weighted"] = (h[f"same_{W}"] / h[f"pairs_{W}"]).mean() / \
            (h[f"same0_{W}"] / h[f"pairs0_{W}"]).mean()
        if g.corpus.nunique() > 1:
            per = []
            for _, gc in g.groupby("corpus"):
                if gc[f"pairs0_{W}"].sum() > 0 and gc[f"same0_{W}"].sum() > 0:
                    per.append((gc[f"same_{W}"].sum() / gc[f"pairs_{W}"].sum()) /
                               (gc[f"same0_{W}"].sum() / gc[f"pairs0_{W}"].sum()))
            r["B_corpus_mean"] = float(np.mean(per))
            r["corpora"] = len(per)
            r["corpora_B_above_1"] = int(sum(b > 1 for b in per))
            # two-stage bootstrap: corpora, then documents within corpus
            corp = g.corpus.unique()
            groups = {k: gc for k, gc in g.groupby("corpus")}
            bs = []
            for _ in range(2000):
                parts = []
                for k in rng.choice(corp, len(corp)):
                    gc = groups[k]
                    parts.append(gc.iloc[rng.integers(0, len(gc), len(gc))])
                d = pd.concat(parts)
                bs.append((d[f"same_{W}"].sum() / d[f"pairs_{W}"].sum()) /
                          (d[f"same0_{W}"].sum() / d[f"pairs0_{W}"].sum()))
            r["B_hier_lo"], r["B_hier_hi"] = np.percentile(bs, [2.5, 97.5])
        out.append(r)
    pd.DataFrame(out).round(4).to_csv(args.out, index=False)
    print(pd.DataFrame(out).round(3).T.to_string())


def test_adult(args):
    """H16: B (70 target kinds, W = 100) differs between CHILDES and adult
    conversation (CABank). Two-sided; percentile bootstrap over documents."""
    import numpy as np
    rng = np.random.default_rng(SEED)
    c = _counts(args.counts)
    a, b = c[c.source == "childes"], c[c.source == "cabank"]
    ba, bb = cds.boot_B(a, W, rng), cds.boot_B(b, W, rng)
    diff = ba - bb
    p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
    res = {"childes": cds.corpus_stats(a, W), "cabank": cds.corpus_stats(b, W),
           "n_docs": {"childes": len(a), "cabank": len(b)},
           "difference": float(np.median(diff)), "ci95": list(np.percentile(diff, [2.5, 97.5])),
           "p_two_sided": float(max(p, 1 / (len(diff) + 1))),
           "cabank_B_ci95": list(np.percentile(bb, [2.5, 97.5]))}
    json.dump(res, open(args.out + ".json", "w"), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))


# --------------------------------------------------------------------------

def _same_kind_pairs(d):
    m = cds.mentions(d["units"])
    out = []
    for (p1, j1, k1), (p2, j2, k2) in zip(m, m[1:]):
        if j1 != j2 and p2 - p1 <= W and k1 == k2:
            out.append((j1, j2, k1))
    return out


def _cue_flags(units, j1, j2, kind):
    span = [w for u in units[j1:j2 + 1] for w in u]
    forms = set(cds.TARGET_KINDS[kind])
    f = {"another": any(w in ANOTHER for w in span),
         "numeral": any(w in NUMERALS for w in span),
         "plural": any(w in PLURAL and w in forms for w in span)}
    for dim, words in PROPERTY.items():
        f[f"prop_{dim}"] = any(w in words for w in span)
    f["any_property"] = any(f[f"prop_{d}"] for d in PROPERTY)
    f["any_distinct_cue"] = f["another"] or f["numeral"] or f["plural"]
    return f


def cues(args):
    import pandas as pd
    rows = []
    for d in read_jsonl(args.inputs):
        if not keep(d):
            continue
        for j1, j2, k in _same_kind_pairs(d):
            rows.append({"source": d["source"], "doc": d["doc"], "kind": k,
                         **_cue_flags(d["units"], j1, j2, k)})
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.groupby("source").mean(numeric_only=True).round(3).T.to_string())


def sample_coding(args):
    rng = random.Random(SEED)
    pool = defaultdict(list)
    for d in read_jsonl(args.inputs):
        if not keep(d) or d["source"] not in CODING_N:
            continue
        for j1, j2, k in _same_kind_pairs(d):
            u = d["units"]
            pool[d["source"]].append({
                "source": d["source"], "doc": d["doc"], "kind": k,
                "before": " ".join(u[j1 - 1]) if j1 > 0 else "",
                "span": " || ".join(" ".join(x) for x in u[j1:j2 + 1]),
                "after": " ".join(u[j2 + 1]) if j2 + 1 < len(u) else ""})
    rows = []
    for src, n in CODING_N.items():
        rows += rng.sample(pool[src], min(n, len(pool[src])))
    rng.shuffle(rows)  # coders do not see which corpus a pair came from
    key = []
    for i, r in enumerate(rows):
        r["pair_id"] = i
        key.append({"pair_id": i, "source": r.pop("source"), "doc": r.pop("doc")})
        r.update({"referent": "", "property": "", "notes": ""})
    _write(args.out, [{k: r[k] for k in ("pair_id", "kind", "before", "span", "after",
                                          "referent", "property", "notes")} for r in rows])
    _write(args.out.replace(".csv", "_key.csv"), key)
    print(f"{len(rows)} pairs to {args.out}; corpus key kept separately")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prepare-cabank"); p.add_argument("--zips", nargs="+", required=True)
    p.add_argument("--out", required=True)
    for c in ("measure-blocks", "measure-nouns", "cues", "sample-coding"):
        p = sp.add_parser(c); p.add_argument("--inputs", nargs="+", required=True)
        p.add_argument("--out", required=True)
    for c in ("rates", "test-adult"):
        p = sp.add_parser(c); p.add_argument("--counts", required=True); p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"prepare-cabank": prepare_cabank, "measure-blocks": measure_blocks,
     "measure-nouns": measure_nouns, "rates": rates, "test-adult": test_adult,
     "cues": cues, "sample-coding": sample_coding}[a.cmd](a)


if __name__ == "__main__":
    main()
