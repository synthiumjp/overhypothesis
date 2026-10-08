#!/usr/bin/env python3
"""
cds_burstiness.py  |  Study 1e (pre-registered in Amendment 4).

Do mentions of the same object kind cluster more in speech to children than
in adult text? Fixed before any CHILDES transcript or adult text was read.

Primary measure. A document is a sequence of units (utterances or
sentences). Take each pair of consecutive target-noun mentions that sit in
different units and at most W words apart. P = the share of those pairs that
name the same kind. P0 = the same share after shuffling the units within each
document (10 shuffles, averaged): same words, same documents, no local order.
Burstiness B = P / P0. B = 1 means talk about one kind is no more clustered
than the document's content predicts. This is the manipulation of Study 1c:
same sentences, different order. Documents under 400 words are excluded.

Secondary measure. R = share of mentions whose previous mention of the same
kind lies in an earlier unit within W words; R0 and BR = R / R0 as above. R
saturates when mentions are dense, so it is secondary.

Subcommands
  prepare-childes  CHILDES CHAT zip  -> jsonl of transcripts (child input)
  prepare-adult    Hugging Face text dataset -> jsonl of documents
  prepare-synth    Study 1 synthetic corpora -> jsonl (reference points)
  measure          per-document counts for one or more corpora
  test             registered tests H9, H10 and secondary analyses

Usage (from the repo root)
  python scripts/cds_burstiness.py prepare-childes --zip Eng-NA.zip --out data/cds/childes.jsonl
  python scripts/cds_burstiness.py prepare-adult --dataset NeelNanda/pile-10k --out data/cds/pile10k.jsonl
  python scripts/cds_burstiness.py prepare-synth --out data/cds/synth.jsonl
  python scripts/cds_burstiness.py measure --inputs data/cds/childes.jsonl data/cds/pile10k.jsonl data/cds/synth.jsonl --out results/study1e/counts.csv
  python scripts/cds_burstiness.py test --counts results/study1e/counts.csv --out results/study1e/tests
"""

import argparse
import io
import json
import os
import random
import re
import zipfile
from collections import defaultdict

SEED = 2028
N_SHUFFLE = 10
N_BOOT = 10_000
WINDOWS = (25, 50, 100, 200)
PRIMARY_W = 100           # about 128 BPE tokens, the Study 1c training window
MIN_WORDS = 4 * PRIMARY_W  # documents shorter than this are excluded
MAX_AGE_MONTHS = 36

# Solid, inanimate, count-noun kinds from the MacArthur-Bates CDI (Words and
# Sentences). Words that are common verbs or adjectives in child-directed
# speech (brush, comb, dress, watch, light, glass, orange, sink, can, plant)
# are left out. Each kind lists the surface forms counted as that kind.
TARGET_KINDS = {
    # toys
    "ball": ["ball", "balls"], "balloon": ["balloon", "balloons"],
    "block": ["block", "blocks"], "book": ["book", "books"],
    "crayon": ["crayon", "crayons"], "doll": ["doll", "dolls"],
    "puzzle": ["puzzle", "puzzles"], "toy": ["toy", "toys"],
    # vehicles
    "airplane": ["airplane", "airplanes", "plane", "planes"],
    "bicycle": ["bicycle", "bicycles", "bike", "bikes"], "boat": ["boat", "boats"],
    "bus": ["bus", "buses", "busses"], "car": ["car", "cars"],
    "motorcycle": ["motorcycle", "motorcycles"], "stroller": ["stroller", "strollers"],
    "tractor": ["tractor", "tractors"], "train": ["train", "trains"],
    "truck": ["truck", "trucks"],
    # food (solid count nouns)
    "apple": ["apple", "apples"], "banana": ["banana", "bananas"],
    "carrot": ["carrot", "carrots"], "cookie": ["cookie", "cookies"],
    "cracker": ["cracker", "crackers"], "egg": ["egg", "eggs"],
    "grape": ["grape", "grapes"], "raisin": ["raisin", "raisins"],
    "sandwich": ["sandwich", "sandwiches"],
    # clothing
    "boot": ["boot", "boots"], "button": ["button", "buttons"],
    "coat": ["coat", "coats"], "diaper": ["diaper", "diapers"],
    "hat": ["hat", "hats"], "jacket": ["jacket", "jackets"],
    "mitten": ["mitten", "mittens"], "necklace": ["necklace", "necklaces"],
    "pajamas": ["pajamas"], "pants": ["pants"], "shirt": ["shirt", "shirts"],
    "shoe": ["shoe", "shoes"], "sock": ["sock", "socks"],
    "sweater": ["sweater", "sweaters"], "zipper": ["zipper", "zippers"],
    # small household items
    "basket": ["basket", "baskets"], "blanket": ["blanket", "blankets"],
    "bottle": ["bottle", "bottles"], "bowl": ["bowl", "bowls"],
    "box": ["box", "boxes"], "broom": ["broom", "brooms"],
    "clock": ["clock", "clocks"], "cup": ["cup", "cups"],
    "dish": ["dish", "dishes"], "fork": ["fork", "forks"],
    "hammer": ["hammer", "hammers"], "key": ["key", "keys"],
    "lamp": ["lamp", "lamps"], "pencil": ["pencil", "pencils"],
    "pillow": ["pillow", "pillows"], "plate": ["plate", "plates"],
    "purse": ["purse", "purses"], "scissors": ["scissors"],
    "spoon": ["spoon", "spoons"], "telephone": ["telephone", "telephones", "phone", "phones"],
    "toothbrush": ["toothbrush", "toothbrushes"], "towel": ["towel", "towels"],
    # furniture
    "bed": ["bed", "beds"], "chair": ["chair", "chairs"],
    "couch": ["couch", "couches"], "crib": ["crib", "cribs"],
    "table": ["table", "tables"], "television": ["television", "televisions", "tv"],
}
FORM_TO_KIND = {f: k for k, forms in TARGET_KINDS.items() for f in forms}

WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


# --------------------------------------------------------------------------
# CHAT parsing
# --------------------------------------------------------------------------

def parse_age(s):
    """CHAT age 'y;mm.dd' -> months (float), or None."""
    m = re.match(r"\s*(\d+);(\d{1,2})?(?:\.(\d{1,2}))?", s or "")
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0)
    return y * 12 + mo + d / 30.0


def clean_chat(text):
    """CHAT utterance -> lowercase word list."""
    t = re.sub(r"\[[^\]]*\]", " ", text)          # [= ...], [/], [*], [+ ...]
    t = re.sub(r"\x15[^\x15]*\x15", " ", t)        # media bullets
    t = re.sub(r"&\S+", " ", t)                    # &=laughs, &-uh, &+fr
    t = re.sub(r"@\S+", "", t)                     # word@s -> word
    t = re.sub(r"\(([a-z]*)\)", r"\1", t)          # (be)cause -> because
    t = t.replace("_", " ").replace("+", " ").replace("<", " ").replace(">", " ")
    words = []
    for w in t.lower().split():
        if w in ("xxx", "yyy", "www") or w.startswith("0"):
            continue
        words += WORD.findall(w)
    return words


def read_cha(raw, path):
    """Return (age_months, [(speaker, words)]) for one transcript."""
    lines, cur = [], None
    for line in raw.splitlines():
        if line.startswith("\t") and cur is not None:
            lines[-1] = lines[-1] + " " + line.strip()
        else:
            lines.append(line)
            cur = line
    age, utts = None, []
    for line in lines:
        if line.startswith("@ID:"):
            f = line.split(":", 1)[1].strip().split("|")
            if len(f) > 7 and f[2] == "CHI" and "Target_Child" in f[7]:
                age = parse_age(f[3])
        elif line.startswith("*"):
            spk, _, text = line[1:].partition(":")
            utts.append((spk.strip(), clean_chat(text)))
    return age, utts


def prepare_childes(args):
    n_in = n_out = 0
    with zipfile.ZipFile(args.zip) as z, open(args.out, "w", encoding="utf-8") as out:
        for name in sorted(z.namelist()):
            if not name.endswith(".cha"):
                continue
            n_in += 1
            raw = z.read(name).decode("utf-8", errors="replace")
            age, utts = read_cha(raw, name)
            if age is None or age >= MAX_AGE_MONTHS:
                continue
            units = [w for s, w in utts if s != "CHI" and w]
            parents = [w for s, w in utts if s in ("MOT", "FAT") and w]
            if not units:
                continue
            corpus = name.split("/")[1] if "/" in name else ""
            out.write(json.dumps({"source": "childes", "doc": name, "corpus": corpus,
                                  "age": round(age, 2), "units": units,
                                  "units_parents": parents}) + "\n")
            n_out += 1
    print(f"{n_in} transcripts read, {n_out} kept (target child under {MAX_AGE_MONTHS} months)")


# --------------------------------------------------------------------------
# Adult text and synthetic corpora
# --------------------------------------------------------------------------

SENT = re.compile(r"(?<=[.!?])\s+")


def prepare_adult(args):
    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train")
    n = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for i, row in enumerate(ds):
            meta = row.get("meta") or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except ValueError:
                    meta = {}
            units = [WORD.findall(s.lower()) for s in SENT.split(row["text"])]
            units = [u for u in units if u]
            if sum(map(len, units)) < 50:
                continue
            out.write(json.dumps({"source": args.label, "doc": f"{args.label}_{i}",
                                  "corpus": meta.get("pile_set_name", ""), "units": units}) + "\n")
            n += 1
    print(f"{n} documents written to {args.out}")


def prepare_synth(args):
    import importlib.util
    spec = importlib.util.spec_from_file_location("mb", os.path.join("scripts", "make_bursty_corpus.py"))
    mb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mb)
    with open(args.out, "w", encoding="utf-8") as out:
        for label, folder in (("synthetic_standard", "data/corpora"),
                              ("synthetic_bursty", "data/corpora_bursty")):
            for seed in (42, 123, 456, 789, 1001):
                lines = [l.strip() for l in open(f"{folder}/regular_seed{seed}.txt", encoding="utf-8") if l.strip()]
                # In the synthetic corpora the kind is the nonce noun of each sentence.
                units = [WORD.findall(l.lower()) for l in lines]
                kinds = [mb.kind_of(l) for l in lines]
                out.write(json.dumps({"source": label, "doc": f"{label}_{seed}", "corpus": "regular",
                                      "units": units, "kinds": kinds}) + "\n")
    print(f"synthetic reference corpora written to {args.out}")


# --------------------------------------------------------------------------
# Measure
# --------------------------------------------------------------------------

def mentions(units, kinds=None):
    """List of (word_position, unit_index, kind) for target-noun mentions."""
    pos, out = 0, []
    for j, u in enumerate(units):
        for w in u:
            if kinds is not None:
                if w == kinds[j]:
                    out.append((pos, j, w))
            else:
                k = FORM_TO_KIND.get(w)
                if k:
                    out.append((pos, j, k))
            pos += 1
    return out


def local_repeats(ments, w):
    """Mentions whose previous mention of the same kind lies in an EARLIER unit
    within the preceding w words. Repeats inside one utterance or sentence are
    not counted: shuffling cannot move them, and sentences are longer than
    utterances, so counting them would pull adult text towards B = 1."""
    last, hits = {}, 0
    for p, j, k in ments:
        if k in last and p - last[k][0] <= w and last[k][1] != j:
            hits += 1
        last[k] = (p, j)
    return hits


def succession(ments, w):
    """Consecutive target mentions in different units, at most w words apart:
    (pairs, pairs naming the same kind). Unlike the repeat rate, this does not
    saturate when mentions are dense, because it conditions on proximity."""
    pairs = same = 0
    for (p1, j1, k1), (p2, j2, k2) in zip(ments, ments[1:]):
        if j1 != j2 and p2 - p1 <= w:
            pairs += 1
            same += k1 == k2
    return pairs, same


def measure_doc(units, rng, kinds=None):
    obs = mentions(units, kinds)
    res = {"n_words": sum(map(len, units)), "n_units": len(units), "n_mentions": len(obs)}
    for w in WINDOWS:
        res[f"hits_{w}"] = local_repeats(obs, w)
        res[f"pairs_{w}"], res[f"same_{w}"] = succession(obs, w)
    shuf = defaultdict(float)
    idx = list(range(len(units)))
    for _ in range(N_SHUFFLE):
        rng.shuffle(idx)
        u2 = [units[i] for i in idx]
        k2 = [kinds[i] for i in idx] if kinds is not None else None
        m2 = mentions(u2, k2)
        for w in WINDOWS:
            shuf[f"hits0_{w}"] += local_repeats(m2, w) / N_SHUFFLE
            pr, sm = succession(m2, w)
            shuf[f"pairs0_{w}"] += pr / N_SHUFFLE
            shuf[f"same0_{w}"] += sm / N_SHUFFLE
    res.update(shuf)
    return res


def measure(args):
    import csv
    rng = random.Random(SEED)
    fields = None
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        for path in args.inputs:
            for line in open(path, encoding="utf-8"):
                d = json.loads(line)
                variants = [("all", d["units"])]
                if "units_parents" in d:
                    variants.append(("parents", d["units_parents"]))
                for speakers, units in variants:
                    r = measure_doc(units, rng, d.get("kinds"))
                    row = {"source": d["source"], "doc": d["doc"], "corpus": d.get("corpus", ""),
                           "age": d.get("age", ""), "speakers": speakers, **r}
                    if fields is None:
                        fields = list(row)
                        w = csv.DictWriter(f, fieldnames=fields)
                        w.writeheader()
                    w.writerow(row)
    print(f"Per-document counts written to {args.out}")


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def corpus_stats(g, w):
    """P = share of nearby consecutive mentions naming the same kind; P0 under
    shuffling; B = P / P0 (primary). R, R0, BR = the repeat-rate version."""
    def ratio(a, b):
        return a / b if b else float("nan")
    P = ratio(g[f"same_{w}"].sum(), g[f"pairs_{w}"].sum())
    P0 = ratio(g[f"same0_{w}"].sum(), g[f"pairs0_{w}"].sum())
    R = ratio(g[f"hits_{w}"].sum(), g.n_mentions.sum())
    R0 = ratio(g[f"hits0_{w}"].sum(), g.n_mentions.sum())
    return {"P": P, "P0": P0, "B": ratio(P, P0), "R": R, "R0": R0, "BR": ratio(R, R0),
            "pairs": float(g[f"pairs_{w}"].sum())}


def boot_B(g, w, rng, n=N_BOOT):
    import numpy as np
    s, p = g[f"same_{w}"].to_numpy(), g[f"pairs_{w}"].to_numpy()
    s0, p0 = g[f"same0_{w}"].to_numpy(), g[f"pairs0_{w}"].to_numpy()
    idx = rng.integers(0, len(g), (n, len(g)))
    P = s[idx].sum(1) / np.maximum(p[idx].sum(1), 1e-9)
    P0 = s0[idx].sum(1) / np.maximum(p0[idx].sum(1), 1e-9)
    return P / np.maximum(P0, 1e-9)


def test(args):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(SEED)
    c = pd.read_csv(args.counts, keep_default_na=False)
    # Documents shorter than 4 windows leave shuffling little room to move
    # anything, which pulls B towards 1. Excluded from all corpora alike.
    n_short = int((c.n_words < MIN_WORDS).sum())
    c = c[(c.n_mentions > 0) & (c.n_words >= MIN_WORDS)]
    cds = c[(c.source == "childes") & (c.speakers == "all")]
    adult = c[(c.source != "childes") & (~c.source.str.startswith("synthetic")) & (c.speakers == "all")]
    if cds.empty or adult.empty:
        raise SystemExit("No child-directed or adult documents left after exclusions.")
    w = PRIMARY_W
    res = {"window_words": w, "min_words": MIN_WORDS, "docs_excluded_short": n_short,
           "n_docs": {"childes": len(cds), "adult": len(adult)},
           "n_mentions": {"childes": int(cds.n_mentions.sum()), "adult": int(adult.n_mentions.sum())},
           "childes": corpus_stats(cds, w), "adult": corpus_stats(adult, w)}
    bc, ba = boot_B(cds, w, rng), boot_B(adult, w, rng)
    p9 = (np.sum(bc <= 1) + 1) / (N_BOOT + 1)
    diff = bc - ba
    p10 = (np.sum(diff <= 0) + 1) / (N_BOOT + 1)
    holm = sorted([("H9", p9), ("H10", p10)], key=lambda x: x[1])
    adj, run = {}, 0.0
    for i, (k, p) in enumerate(holm):
        run = max(run, min(1.0, (2 - i) * p))
        adj[k] = run
    res["H9"] = {"B_childes": res["childes"]["B"], "ci95": list(np.percentile(bc, [2.5, 97.5])),
                 "p_one_sided": p9, "p_holm": adj["H9"], "supported": adj["H9"] < 0.05}
    res["H10"] = {"B_difference": res["childes"]["B"] - res["adult"]["B"],
                  "ci95": list(np.percentile(diff, [2.5, 97.5])),
                  "p_one_sided": p10, "p_holm": adj["H10"], "supported": adj["H10"] < 0.05}
    # Secondary: every corpus, every window; parents only; age bins; synthetic reference
    rows = []
    for (src, spk), g in c.groupby(["source", "speakers"]):
        for ww in WINDOWS:
            s = corpus_stats(g, ww)
            b = boot_B(g, ww, rng, 2000) if len(g) > 1 else np.array([s["B"]])
            rows.append({"source": src, "speakers": spk, "window": ww, "docs": len(g),
                         "mentions": int(g.n_mentions.sum()), **s,
                         "B_lo": np.percentile(b, 2.5), "B_hi": np.percentile(b, 97.5)})
    ages = cds.copy()
    ages["age"] = pd.to_numeric(ages.age)
    ages["age_bin"] = pd.cut(ages.age, [0, 18, 24, 36], right=False, labels=["<18", "18-24", "24-36"])
    for b_, g in ages.groupby("age_bin", observed=True):
        s = corpus_stats(g, w)
        rows.append({"source": f"childes_age_{b_}", "speakers": "all", "window": w, "docs": len(g),
                     "mentions": int(g.n_mentions.sum()), **s})
    if "corpus" in adult and adult.corpus.ne("").any():
        for sub, g in adult.groupby("corpus"):
            s = corpus_stats(g, w)
            rows.append({"source": f"adult_{sub}", "speakers": "all", "window": w, "docs": len(g),
                         "mentions": int(g.n_mentions.sum()), **s})
    pd.DataFrame(rows).round(4).to_csv(args.out + "_secondary.csv", index=False)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare-childes"); p.add_argument("--zip", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("prepare-adult"); p.add_argument("--dataset", default="NeelNanda/pile-10k")
    p.add_argument("--label", default="pile10k"); p.add_argument("--out", required=True)
    p = sub.add_parser("prepare-synth"); p.add_argument("--out", required=True)
    p = sub.add_parser("measure"); p.add_argument("--inputs", nargs="+", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("test"); p.add_argument("--counts", required=True); p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"prepare-childes": prepare_childes, "prepare-adult": prepare_adult, "prepare-synth": prepare_synth,
     "measure": measure, "test": test}[a.cmd](a)


if __name__ == "__main__":
    main()
