#!/usr/bin/env python3
"""
a10_reanalysis.py  |  Exploratory reanalysis of the Amendment 10 Part A (Experiment 2) models, in
answer to a review. Inference only; no model is changed on disk. Added after the results.

For every final model in the manifest (grouped Regular and Reassigned), and every familiarity noun:

1. Row diagnostics (condition 'own'): norm of the noun's input row; its distance from the mean row of
   the 32 trained kinds, divided by the trained kinds' mean distance from that mean; its cosine with
   that mean; and its cosine with the noun's row at initialisation. The initial rows are rebuilt by
   seeding torch with the run's seed and constructing the model exactly as train_model.py does; the
   rebuild is checked on the unseen nouns, whose rows receive no gradient and so only shrink under
   weight decay (cosine with their initial rows must be 1 to 1e-4). If the check fails, the cosine
   columns are left empty.

2. Row swaps (carry with the registered test items, same-noun context):
     own       the noun's own row (reproduces scores_a.csv; checked)
     typical   a typical row (trained kinds' mean plus an orthonormal offset of typical length, as for
               the unseen nouns in the paper)
     swap      the row of the noun in the paired cell with the same index and exposure level:
               rotating runs <-> fixed runs, rotating runs <-> rotating singles (rotating-run nouns
               are scored with each partner's row, recorded as swap_fixed_run and swap_rotating_single)
   Only the input row changes; output rows and every other weight stay as trained.

3. Generalisation items (own rows; carry = L(target) - mean L over the other words of the probed
   dimension, excluding the noun's own shape s* for shapes):
     base       "A n is a s c t thing. A n is a"                       (the test item)
     gap        the same context, then one sentence about a trained kind, then the query
     frame2     context "Look at the n. It is s and c." then "A n is a"
     colour     context "A n is a s c t thing." then "A n is a s' " (s' a random shape other than s
                and s*), probing the colour c
   36 items per noun for each, drawn with a fixed seed.

4. Attention (base items, own rows): attention from the query position to the context's shape word,
   mean over heads of the first layer, and the largest single head anywhere.

  python scripts/a10_reanalysis.py --manifest results/amendment10/manifest_a.csv --out results/amendment10/reanalysis.csv
  (resumable: models already in the output are skipped)
"""
import argparse
import csv
import os
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus_generator as cg  # noqa: E402
import make_familiarity_corpora as mfc  # noqa: E402
from familiarity_eval import check_forward, forward, token_id, typical_rows  # noqa: E402
from familiarity_items import make_items  # noqa: E402
from oneshot_eval import load_model  # noqa: E402

ITEM_SEED = 83_000
FIELDS = ["base", "size", "seed", "ckpt", "noun", "type", "exposures", "condition", "carry", "n_items",
          "row_norm", "dist_mean_rel", "cos_mean", "cos_init", "att_l0_mean", "att_max_head"]
PAIRS = [("rotating_run", "fixed_run"), ("rotating_run", "rotating_single")]


def init_rows(run, vocab_size):
    """The input embedding at initialisation, rebuilt as train_model.py builds it."""
    import train_model as tm
    torch.manual_seed(int(run["seed"]))
    mc = tm.MODEL_CONFIGS[run["size"]]
    m = tm.OverhypothesisLM(vocab_size=vocab_size, d_model=mc["d_model"], n_head=mc["n_head"],
                            n_layer=mc["n_layer"], d_ff=mc["d_ff"], seq_len=tm.TRAIN_CONFIG[run["size"]]["seq_len"])
    return m.tok_embed.weight.detach().clone()


def score(model, tok, items, want_att=False):
    """items: dicts with text, target (word), dim ('shape'|'colour'), exclude (list of words).
    Returns carry per item and, if want_att, (att_l0_mean, att_max_head) at the target's context position."""
    ids_by_dim = {"shape": [token_id(tok, s) for s in cg.SHAPE_TOKENS],
                  "colour": [token_id(tok, c) for c in cg.COLOUR_TOKENS]}
    words = {"shape": cg.SHAPE_TOKENS, "colour": cg.COLOUR_TOKENS}
    out = []
    groups = {}
    enc = [tok.encode(it["text"]).ids for it in items]
    for i, e in enumerate(enc):
        groups.setdefault(len(e), []).append(i)
    res = [None] * len(items)
    with torch.no_grad():
        for n, idx in groups.items():
            for j in range(0, len(idx), 256):
                part = idx[j:j + 256]
                x = torch.tensor([enc[i] for i in part])
                r = forward(model, x, None, want_att)
                logits, att = r if want_att else (r, None)
                lp = torch.log_softmax(logits.float(), -1)
                for k, i in enumerate(part):
                    it = items[i]
                    dim = it["dim"]
                    L = {w: lp[k, t].item() for w, t in zip(words[dim], ids_by_dim[dim])}
                    others = [L[w] for w in words[dim] if w != it["target"] and w not in it["exclude"]]
                    c = L[it["target"]] - sum(others) / len(others)
                    a0 = amax = None
                    if want_att:
                        pos = enc[i].index(token_id(tok, it["target"]))
                        a0 = att[0][k, :, pos].mean().item()
                        amax = max(a[k, :, pos].max().item() for a in att)
                    res[i] = (c, a0, amax)
    return res


def make_gen_items(rows, seed, exemplars):
    rng = random.Random(ITEM_SEED + seed)
    items = []
    for r in rows:
        n, star = r["noun"], r["s_star"]
        shapes = [s for s in cg.SHAPE_TOKENS if s != star] * 4
        rng.shuffle(shapes)
        for s in shapes:
            c, t = rng.choice(cg.COLOUR_TOKENS), rng.choice(cg.TEXTURE_TOKENS)
            ctx = f"A {n} is a {s} {c} {t} thing."
            base = {"noun": n, "dim": "shape", "target": s, "exclude": [star]}
            items.append({**base, "cond": "gap", "text": f"{ctx} {rng.choice(exemplars)} A {n} is a"})
            items.append({**base, "cond": "frame2", "text": f"Look at the {n}. It is {s} and {c}. A {n} is a"})
            s2 = rng.choice([x for x in cg.SHAPE_TOKENS if x not in (s, star)])
            items.append({"noun": n, "dim": "colour", "target": c, "exclude": [], "cond": "colour",
                          "text": f"{ctx} A {n} is a {s2}"})
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check_scores", default="results/amendment10/scores_a.csv")
    a = ap.parse_args()
    import pandas as pd
    from tokenizers import Tokenizer
    import make_familiarity_corpora_a10 as m10
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    ref = None
    if os.path.exists(a.check_scores):
        ref = pd.read_csv(a.check_scores)
        ref = ref[ref.emb == "original"].set_index(["ckpt", "noun"])["carry"]
    runs = [r for r in csv.DictReader(open(a.manifest, encoding="utf-8")) if r["ckpt"].endswith("model_final.pt")]
    done = set()
    if os.path.exists(a.out):
        done = {r["ckpt"] for r in csv.DictReader(open(a.out, encoding="utf-8"))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for i, run in enumerate(runs, 1):
        if run["ckpt"] in done:
            continue
        seed = int(run["seed"])
        tok = Tokenizer.from_file(run["tokenizer"])
        model, _ = load_model(run["ckpt"], None, "train_model", "OverhypothesisLM")
        model.eval()
        check_forward(model, tok)
        if run.get("design", "a10").startswith("a11_"):
            import make_familiarity_corpora_a11 as m11
            rows = m11.load_assignment(seed, run["design"][4:])
        else:
            rows = m10.load_assignment(seed)
        info = {r["noun"]: r for r in rows}
        reg = [it for it in make_items(seed, 3, run.get("design") or "a10") if it["kind"] == "carry_same"]
        E = model.tok_embed.weight
        orig = E.detach().clone()
        train_ids = [token_id(tok, n) for n in cg.TRAIN_NOUNS]
        mu = orig[train_ids].mean(0)
        spread = (orig[train_ids] - mu).norm(dim=1).mean()
        nid = {n: token_id(tok, n) for n in info}
        init = init_rows(run, tok.get_vocab_size())
        unseen = [n for n in info if info[n]["type"] == "unseen"]
        cos_un = min(torch.nn.functional.cosine_similarity(orig[nid[n]], init[nid[n]], dim=0).item() for n in unseen)
        init_ok = cos_un > 1 - 1e-4
        if not init_ok:
            print(f"  initial rows not reproduced (min cosine on unseen nouns {cos_un:.5f}); cos_init left empty")
        out = []

        def base_items(nouns):
            return [{"noun": it["noun"], "dim": "shape", "target": it["ctx_shape"], "exclude": [it["s_star"]],
                     "text": it["text"]} for it in reg if it["noun"] in nouns]

        def emit(n, cond, vals, att=None):
            v = E[nid[n]].detach()
            rec = {"base": run["base"], "size": run["size"], "seed": seed, "ckpt": run["ckpt"], "noun": n,
                   "type": info[n]["type"], "exposures": info[n]["exposures"], "condition": cond,
                   "carry": sum(vals) / len(vals), "n_items": len(vals)}
            if cond == "own":
                rec.update({"row_norm": v.norm().item(), "dist_mean_rel": ((v - mu).norm() / spread).item(),
                            "cos_mean": torch.nn.functional.cosine_similarity(v, mu, dim=0).item(),
                            "cos_init": (torch.nn.functional.cosine_similarity(v, init[nid[n]], dim=0).item()
                                         if init_ok else "")})
            if att:
                rec.update({"att_l0_mean": sum(x[0] for x in att) / len(att),
                            "att_max_head": sum(x[1] for x in att) / len(att)})
            out.append(rec)

        # own rows: base items with attention, then generalisation items
        its = base_items(set(info))
        res = score(model, tok, its, want_att=True)
        per = {}
        for it, r in zip(its, res):
            per.setdefault(it["noun"], []).append(r)
        for n, rr in per.items():
            emit(n, "own", [x[0] for x in rr], [(x[1], x[2]) for x in rr])
            if ref is not None and (run["ckpt"], n) in ref.index:
                if abs(ref[(run["ckpt"], n)] - out[-1]["carry"]) > 1e-3:
                    sys.exit(f"own row does not reproduce scores_a.csv for {n} in {run['ckpt']}")
        exemplars, _ = mfc.base_exemplars(seed)
        gen = make_gen_items([info[n] for n in info], seed, exemplars)
        res = score(model, tok, gen)
        per = {}
        for it, r in zip(gen, res):
            per.setdefault((it["noun"], it["cond"]), []).append(r[0])
        for (n, cond), vals in per.items():
            emit(n, cond, vals)
        # typical rows for every familiarity noun
        fam = [n for n in info if info[n]["type"] != "unseen"]
        with torch.no_grad():
            E.copy_(orig)
            T = typical_rows(orig, train_ids, len(fam))
            for k, n in enumerate(fam):
                E[nid[n]] = T[k]
        its = base_items(set(fam))
        per = {}
        for it, r in zip(its, score(model, tok, its)):
            per.setdefault(it["noun"], []).append(r[0])
        for n, vals in per.items():
            emit(n, "typical", vals)
        # swaps
        for t1, t2 in PAIRS:
            for e in (100, 1000):
                c1 = sorted(n for n in info if info[n]["type"] == t1 and info[n]["exposures"] == e)
                c2 = sorted(n for n in info if info[n]["type"] == t2 and info[n]["exposures"] == e)
                with torch.no_grad():
                    E.copy_(orig)
                    for x, y in zip(c1, c2):
                        E[nid[x]], E[nid[y]] = orig[nid[y]].clone(), orig[nid[x]].clone()
                its = base_items(set(c1) | set(c2))
                per = {}
                for it, r in zip(its, score(model, tok, its)):
                    per.setdefault(it["noun"], []).append(r[0])
                for n, vals in per.items():
                    partner = t2 if n in c1 else t1
                    emit(n, f"swap_{partner}", vals)
        with torch.no_grad():
            E.copy_(orig)
        new = not os.path.exists(a.out)
        with open(a.out, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(out)
        print(f"[{i}/{len(runs)}] {run['base']} {run['size']} seed {seed}; initial rows "
              f"{'reproduced' if init_ok else 'NOT reproduced'}", flush=True)


if __name__ == "__main__":
    main()
