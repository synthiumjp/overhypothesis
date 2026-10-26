#!/usr/bin/env python3
"""
natural_regime.py  |  Which of the two training regimes does natural text resemble? Exploratory, added after the
results in answer to a review.

Corpus: WikiText-103 (raw, train split; Merity et al. 2017), split into articles at level-1 headings (" = Title = ").
Nouns: the 44 nouns of the pretrained test (data/amendment10/nouns.csv, 22 colour-typical and 22 colour-variable,
fixed before any model was run). Colours: the 11 basic colour words (data/amendment10/colours.csv).

For each noun:
  - recurrence: of its occurrences, the share followed by another occurrence within the next 128 words, and the share
    expected if the noun were spread evenly at its overall rate (1 - exp(-128 * rate)); their ratio is its burstiness;
  - colour mentions: a basic colour word immediately before the noun ("red truck"); overall agreement = chance that two
    random mentions name the same colour (sum of squared shares); within-article agreement = share of pairs of mentions
    in the same article that name the same colour (pairs weighted equally).
In the paper's terms, a stable-shape kind has high overall agreement; a shifting-shape kind has low overall agreement
but high agreement within a burst (here, within an article).

  pip install datasets
  python scripts/natural_regime.py --out results/natural_regime.csv --md results/NATURAL_REGIME.md
"""
import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

HEAD = re.compile(r"^ = [^=].* = $")
WORD = re.compile(r"[a-z]+")


def articles(lines):
    cur = []
    for ln in lines:
        if HEAD.match(ln) and cur:
            yield cur
            cur = []
        cur.extend(WORD.findall(ln.lower()))
    if cur:
        yield cur


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/natural_regime.csv")
    ap.add_argument("--md", default="results/NATURAL_REGIME.md")
    ap.add_argument("--window", type=int, default=128)
    a = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    nouns = list(csv.DictReader(open("data/amendment10/nouns.csv", encoding="utf-8")))
    group = {r["noun"]: r["group"] for r in nouns}
    colours = [r["colour"] for r in csv.DictReader(open("data/amendment10/colours.csv", encoding="utf-8"))]
    cset = set(colours)
    total = 0
    count = Counter()
    rec = Counter()
    col = defaultdict(list)          # noun -> list of (article index, colour)
    n_art = 0
    for ai, words in enumerate(articles(ds["text"])):
        n_art += 1
        total += len(words)
        pos = defaultdict(list)
        for i, w in enumerate(words):
            if w in group:
                pos[w].append(i)
                if i > 0 and words[i - 1] in cset:
                    col[w].append((ai, words[i - 1]))
        for w, ps in pos.items():
            count[w] += len(ps)
            for k, p in enumerate(ps[:-1]):
                if ps[k + 1] - p <= a.window:
                    rec[w] += 1
    rows = []
    for w in group:
        rate = count[w] / total if total else 0
        obs = rec[w] / count[w] if count[w] else float("nan")
        exp_ = 1 - math.exp(-a.window * rate)
        cs = col[w]
        n = len(cs)
        shares = Counter(c for _, c in cs)
        overall = sum((v / n) ** 2 for v in shares.values()) if n else float("nan")
        by_art = defaultdict(list)
        for ai, c in cs:
            by_art[ai].append(c)
        same = pairs = 0
        for lst in by_art.values():
            m = len(lst)
            if m < 2:
                continue
            cc = Counter(lst)
            same += sum(v * (v - 1) / 2 for v in cc.values())
            pairs += m * (m - 1) / 2
        within = same / pairs if pairs else float("nan")
        top = shares.most_common(1)[0][0] if n else ""
        rows.append({"noun": w, "group": group[w], "count": count[w], "recur_obs": obs, "recur_exp": exp_,
                     "burstiness": obs / exp_ if exp_ > 0 else float("nan"), "colour_mentions": n, "top_colour": top,
                     "agree_overall": overall, "within_pairs": int(pairs), "agree_within": within})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    def med(v):
        v = sorted(x for x in v if x == x)
        return v[len(v) // 2] if v else float("nan")

    md = ["# Natural text: burstiness and colour stability (WikiText-103; exploratory)", "",
          f"{n_art:,} articles, {total:,} words; window {a.window} words.", ""]
    for g in ("typical", "variable"):
        r = [x for x in rows if x["group"] == g and x["count"] > 0]
        md.append(f"- {g}: median burstiness {med([x['burstiness'] for x in r]):.1f} (observed recurrence "
                  f"{med([x['recur_obs'] for x in r]):.3f} vs expected {med([x['recur_exp'] for x in r]):.4f}); "
                  f"colour mentions {sum(x['colour_mentions'] for x in r)}; median agreement overall "
                  f"{med([x['agree_overall'] for x in r]):.2f}, within articles "
                  f"{med([x['agree_within'] for x in r if x['within_pairs'] >= 3]):.2f} "
                  f"(nouns with at least 3 within-article pairs: {sum(x['within_pairs'] >= 3 for x in r)})")
    pooled = {}
    for g in ("typical", "variable"):
        r = [x for x in rows if x["group"] == g and x["within_pairs"] > 0]
        s = sum(x["agree_within"] * x["within_pairs"] for x in r)
        p = sum(x["within_pairs"] for x in r)
        o = sum(x["agree_overall"] * x["colour_mentions"] for x in rows if x["group"] == g and x["colour_mentions"])
        n = sum(x["colour_mentions"] for x in rows if x["group"] == g and x["colour_mentions"])
        pooled[g] = (s / p if p else float("nan"), o / n if n else float("nan"), p)
        md.append(f"- {g}, pooled over nouns: within-article agreement {pooled[g][0]:.2f} over {p} pairs; "
                  f"overall agreement (mention-weighted) {pooled[g][1]:.2f}")
    Path(a.md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
