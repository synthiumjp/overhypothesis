#!/usr/bin/env python3
"""
pythia_prep.py  |  Amendment 10 Part B: noun and colour lists for the Pythia study.

Run once on a machine that can reach huggingface.co and api.infini-gram.io (the training PC):
  pip install transformers requests
  python scripts/pythia_prep.py --out_dir data/amendment10

Steps (all deterministic; nothing here runs a language model):
 1. Nouns coded by two independent passes (data/amendment10/coding_pass_A.csv, _B.csv; fixed prompt,
    coded before any model output was seen). Colour-typical: both passes rate 1 and name the same
    colour. Colour-variable: both rate 3. Nouns that are themselves colour words, or that are also used
    as colour terms (orange, gold, cream, cherry, ...), are excluded (EXCLUDE below).
 2. Keep nouns that are one token with a leading space in the Pythia (GPT-NeoX) tokenizer.
 3. Colour words: the eleven basic colour terms, kept if one token with a leading space.
 4. Nouns that both kind-coding passes (data/amendment10/kind_pass_A.csv, _B.csv) flag as
    polysemous are excluded; their kind (natural/artifact) and count/mass codes are kept as
    covariates (agreed codes only).
 5. Pile counts of "noun" from the infini-gram API (index v4_piletrain_llama; Llama-2
    tokenizer, full Pile; lower-case singular; proportional estimates only). The token ids the
    API used are logged, and a noun whose query is not tokenised as expected is dropped.
 6. Frequency matching: each colour-typical noun, in decreasing order of count, is paired with
    the unused colour-variable noun of the same count/mass type closest in log10 count, if
    within 0.3; at most 40 pairs.
Writes nouns.csv (noun, group, colour, pile_count, pair), colours.csv and prep_log.json.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

COLOURS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "grey"]
EXCLUDE = {"orange", "violet", "lavender", "gold", "silver", "cream", "ivory", "chocolate", "lime", "plum", "mustard",
           "pearl", "charcoal", "copper", "bronze", "rust", "olive", "peach", "apricot", "cherry", "ash", "coral", "slate"}
TOKENIZER = "EleutherAI/pythia-1b-deduped"
CALIPER, MAX_PAIRS = 0.3, 40


def count(term, session, cache, cache_path):
    """Pile count of term (infini-gram). Results are cached in cache_path so that an interrupted
    run resumes; failures are retried with growing pauses."""
    if term in cache:
        return cache[term]["count"], cache[term]["token_ids"], cache[term]["tokens"]
    last = None
    for attempt in range(12):
        try:
            r = session.post("https://api.infini-gram.io/", json={"index": "v4_piletrain_llama", "query_type": "count",
                                                                 "query": term}, timeout=90)
            j = r.json()
            if "count" in j:
                cache[term] = {"count": int(j["count"]), "token_ids": j.get("token_ids"), "tokens": j.get("tokens")}
                cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
                time.sleep(0.5)
                return cache[term]["count"], cache[term]["token_ids"], cache[term]["tokens"]
            last = j
        except Exception as e:
            last = repr(e)
        time.sleep(min(60, 3 * (attempt + 1)))
    raise SystemExit(f"infini-gram count failed for {term!r}: {last}. Run the script again; finished counts are kept.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default="data/amendment10")
    a = ap.parse_args()
    import requests
    from transformers import AutoTokenizer
    out = Path(a.out_dir)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    def one_token(w):
        return len(tok(" " + w, add_special_tokens=False)["input_ids"]) == 1

    KA = {r["noun"]: r for r in csv.DictReader(open(out / "kind_pass_A.csv", encoding="utf-8"))}
    KB = {r["noun"]: r for r in csv.DictReader(open(out / "kind_pass_B.csv", encoding="utf-8"))}
    A = {r["noun"]: r for r in csv.DictReader(open(out / "coding_pass_A.csv", encoding="utf-8"))}
    B = {r["noun"]: r for r in csv.DictReader(open(out / "coding_pass_B.csv", encoding="utf-8"))}
    log = {"tokenizer": TOKENIZER, "excluded_colour_or_material": [], "multi_token": [], "disagreed": 0,
           "polysemous": [], "count_query_dropped": []}
    groups = {"typical": [], "variable": []}
    for n in sorted(A):
        ra, rb = A[n], B[n]
        if n in EXCLUDE:
            log["excluded_colour_or_material"].append(n)
            continue
        if ra["rating"] == rb["rating"] == "1" and ra["colour"] == rb["colour"]:
            g, col = "typical", ra["colour"]
        elif ra["rating"] == rb["rating"] == "3":
            g, col = "variable", ""
        else:
            log["disagreed"] += 1
            continue
        if KA[n]["polysemous"] == KB[n]["polysemous"] == "1":
            log["polysemous"].append(n)
            continue
        if not one_token(n):
            log["multi_token"].append(n)
            continue
        groups[g].append((n, col))
    colours = [c for c in COLOURS if one_token(c)]
    log["colours"] = colours
    s = requests.Session()
    cache_path = out / "count_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    counts, qlog = {}, {}
    for g in groups.values():
        for n, _ in g:
            c, ids, toks = count(n, s, cache, cache_path)
            qlog[n] = {"count": c, "token_ids": ids, "tokens": toks}
            if not toks or "".join(t.replace("\u2581", "") for t in toks) != n:
                log["count_query_dropped"].append(n)
                continue
            counts[n] = c
    for g in groups:
        groups[g] = [(n, col) for n, col in groups[g] if n in counts]
    ktype = {n: (KA[n]["kind"] if KA[n]["kind"] == KB[n]["kind"] else "", KA[n]["count"]
                 if KA[n]["count"] == KB[n]["count"] else "") for n in counts}
    typ = sorted(groups["typical"], key=lambda x: (-counts[x[0]], x[0]))
    var = list(groups["variable"])
    pairs, used = [], set()
    for n, col in typ:
        cands = [(abs(math.log10(counts[n] + 1) - math.log10(counts[v] + 1)), v) for v, _ in var
                 if v not in used and ktype[v][1] == ktype[n][1] and ktype[n][1]]
        if not cands:
            break
        d, v = min(cands)
        if d <= CALIPER:
            used.add(v)
            pairs.append((n, col, v))
        if len(pairs) == MAX_PAIRS:
            break
    with open(out / "nouns.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["noun", "group", "colour", "pile_count", "pair", "kind", "count_type"])
        for i, (n, col, v) in enumerate(pairs):
            w.writerow([n, "typical", col, counts[n], i, *ktype[n]])
            w.writerow([v, "variable", "", counts[v], i, *ktype[v]])
    with open(out / "colours.csv", "w", newline="", encoding="utf-8") as fh:
        fh.write("colour\n" + "\n".join(colours) + "\n")
    log.update({"n_typical_candidates": len(groups["typical"]), "n_variable_candidates": len(groups["variable"]),
                "pairs": len(pairs), "counts": counts, "count_queries": qlog})
    (out / "prep_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"{len(pairs)} matched pairs; colours: {colours}; written to {out}")


if __name__ == "__main__":
    main()
