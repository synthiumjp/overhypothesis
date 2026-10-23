#!/usr/bin/env python3
"""
circuit_analysis.py  |  Candidate heads, ablation and attention for Amendment 9
(docs/AMENDMENT9.md section 4 "Heads"; addendum items 10 and 11).

For each model, in this order:
 1. Candidate heads, chosen before any familiarity item is scored, from a task-independent
    signature (Olsson et al., 2022) on 100 sequences of 25 distinct words of the base language
    (no familiarity or test nouns), each repeated once:
      prefix matching  mean attention from a token in the second copy to the token that followed
                       its first occurrence;
      copying          sum of the real parts over the sum of the moduli of the eigenvalues of the
                       head's OV circuit restricted to those words, W_E W_V W_O W_U (layer norms
                       ignored, as in Olsson et al.).
    The k heads with the highest product are the candidates (k = 2, 4, 6 for 4, 6, 8 layers).
 2. Means for ablation: each head's output (before the output projection) at 2,000 positions
    that predict a shape in frame-1 sentences ("A <kind> is a | <shape>") of base windows of
    the model's own corpus, read in the training order, skipping windows with a familiarity
    sentence.
 3. Carry of the unseen familiarity nouns (same-noun items) intact, with the candidates
    mean-ablated at the last position, and with 200 draws of random heads (each candidate
    replaced by a non-candidate head of the same layer; H33). Carry of trained kinds intact and
    with the candidates ablated (H34), plus 20 random draws (exploratory).
 4. Attention of the candidate heads from the last position to the context's shape word, for
    every familiarity noun (H35), and the carry of every familiarity noun with candidates ablated
    (exploratory).
 5. Direct logit attribution: the change in carry that the candidates' direct paths predict
    under mean ablation (final layer norm frozen), against the observed change (exploratory).

Usage:
  python scripts/circuit_analysis.py --manifest results/amendment9/manifest_pilot.csv --out_dir results/amendment9/circuits
Manifest columns: base, size, seed, per_cell, ckpt, tokenizer, corpus, order, placements.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus_generator as cg  # noqa: E402
import make_familiarity_corpora as mfc  # noqa: E402
from familiarity_eval import (check_forward, forward, head_outputs, measures, run_items,  # noqa: E402
                              token_id)
from familiarity_items import make_items  # noqa: E402
from oneshot_eval import load_model  # noqa: E402

K_BY_LAYERS = {4: 2, 6: 4, 8: 6}
N_SEQ, SEQ_WORDS, SIG_SEED = 100, 25, 31_000
N_MEAN_POS = 2000
N_RANDOM, N_RANDOM_TRAINED, RAND_SEED = 200, 20, 37_000


def base_words():
    return [w for w in mfc.all_words() if w not in set(mfc.FAM_NOUNS) | set(cg.TEST_NOUNS)]


def attention_all(model, ids):
    B, T = ids.shape
    x = model.tok_embed(ids) + model.pos_embed(torch.arange(T).unsqueeze(0))
    out = []
    for blk in model.blocks:
        a = blk.attn
        q, k, v = a.qkv(blk.ln1(x)).split(model.d_model, dim=2)
        H, dh = a.n_head, a.d_head
        q, k, v = (z.view(B, T, H, dh).transpose(1, 2) for z in (q, k, v))
        att = ((q @ k.transpose(-2, -1)) / math.sqrt(dh)).masked_fill(a.mask[:, :, :T, :T] == 0,
                                                                       float("-inf")).softmax(-1)
        out.append(att)
        x = x + a.proj((att @ v).transpose(1, 2).reshape(B, T, model.d_model))
        x = x + blk.ff(blk.ln2(x))
    return out


def signature(model, tok, seed):
    words = base_words()
    wid = [token_id(tok, w) for w in words]
    rng = random.Random(SIG_SEED + seed)
    seqs = []
    for _ in range(N_SEQ):
        s = rng.sample(wid, SEQ_WORDS)
        seqs.append(s + s)
    ids = torch.tensor(seqs)
    L = SEQ_WORDS
    with torch.no_grad():
        atts = attention_all(model, ids)
    n_layer, H = len(model.blocks), model.blocks[0].attn.n_head
    dh = model.blocks[0].attn.d_head
    WE = model.tok_embed.weight.detach()[wid]
    WU = model.lm_head.weight.detach()[wid]
    rows = []
    for li in range(n_layer):
        a = model.blocks[li].attn
        Wv = a.qkv.weight.detach()[2 * model.d_model:]
        for h in range(H):
            qi = torch.arange(L, 2 * L)
            prefix = atts[li][:, h, qi, qi - L + 1].mean().item()
            ov = WE @ Wv[h * dh:(h + 1) * dh].T @ a.proj.weight.detach()[:, h * dh:(h + 1) * dh].T @ WU.T
            ev = np.linalg.eigvals(ov.double().numpy())
            copy = float(ev.real.sum() / np.abs(ev).sum())
            rows.append({"layer": li, "head": h, "prefix": prefix, "copy": copy, "score": prefix * copy})
    k = K_BY_LAYERS[n_layer]
    cand = sorted(rows, key=lambda r: -r["score"])[:k]
    return rows, [(r["layer"], r["head"]) for r in cand]


def ablation_means(model, tok, run):
    """Mean head outputs at shape-predicting positions of frame-1 base sentences."""
    ids = mfc.tokenise_file(run["corpus"], tok)
    order = np.load(run["order"])
    fam_w = {int(p["text_window"]) for p in csv.DictReader(open(run["placements"], encoding="utf-8"))}
    A, a_, is_ = token_id(tok, "A"), token_id(tok, "a"), token_id(tok, "is")
    kinds = {token_id(tok, n) for n in cg.TRAIN_NOUNS}
    shapes = {token_id(tok, s) for s in cg.SHAPE_TOKENS}
    wins, pos = [], []
    for p in range(len(order)):
        w = int(order[p])
        if w in fam_w:
            continue
        t = ids[w * 128: w * 128 + 129].astype(np.int64)
        hits = [i for i in range(3, 127) if t[i] == a_ and t[i - 1] == is_ and t[i - 2] in kinds
                and t[i - 3] == A and t[i + 1] in shapes]
        if hits:
            b = len(wins)
            wins.append(t[:128])
            pos += [(b, i) for i in hits]
        if len(pos) >= N_MEAN_POS:
            break
    pos = pos[:N_MEAN_POS]
    used = sorted({b for b, _ in pos})
    remap = {b: j for j, b in enumerate(used)}
    x = torch.tensor(np.stack([wins[b] for b in used]))
    with torch.no_grad():
        outs = []
        for j in range(0, len(pos), 500):
            part = pos[j:j + 500]
            bs = sorted({b for b, _ in part})
            sub = {b: n for n, b in enumerate(bs)}
            o = head_outputs(model, x[[remap[b] for b in bs]], [(sub[b], t) for b, t in part])
            outs.append(o)
    return torch.cat(outs, 1).mean(1), len(pos)                         # (n_layer, H, dh)


def ablate_spec(heads, means):
    spec = {}
    for li, h in heads:
        spec.setdefault(li, ([], means[li]))[0].append(h)
    return spec


def mean_carry(items, L, kind_key="carry"):
    m = [r[kind_key] for r in measures(items, L) if kind_key in r]
    return sum(m) / len(m)


def random_draws(cands, n_layer, H, n, rng):
    draws = []
    cand = set(cands)
    for _ in range(n):
        used, draw = set(), []
        for li, _ in cands:
            pool = [(li, h) for h in range(H) if (li, h) not in cand and (li, h) not in used]
            if not pool:
                pool = [(l2, h) for l2 in range(n_layer) for h in range(H)
                        if (l2, h) not in cand and (l2, h) not in used]
            pick = rng.choice(pool)
            used.add(pick)
            draw.append(pick)
        draws.append(draw)
    return draws


def dla_change(model, tok, items, heads, means):
    """Change in carry predicted by the candidates' direct paths when their outputs are
    replaced by the means (final layer norm frozen at the intact run)."""
    shape_ids = [token_id(tok, s) for s in cg.SHAPE_TOKENS]
    preds = []
    with torch.no_grad():
        for it in items:
            ids = torch.tensor([tok.encode(it["text"]).ids])
            T = ids.shape[1]
            x = model.tok_embed(ids) + model.pos_embed(torch.arange(T).unsqueeze(0))
            ys = {}
            for li, blk in enumerate(model.blocks):
                a = blk.attn
                q, k, v = a.qkv(blk.ln1(x)).split(model.d_model, dim=2)
                H, dh = a.n_head, a.d_head
                q, k, v = (z.view(1, T, H, dh).transpose(1, 2) for z in (q, k, v))
                att = ((q @ k.transpose(-2, -1)) / math.sqrt(dh)).masked_fill(a.mask[:, :, :T, :T] == 0,
                                                                               float("-inf")).softmax(-1)
                y = att @ v
                ys[li] = y[0, :, -1, :]
                x = x + a.proj(y.transpose(1, 2).reshape(1, T, model.d_model))
                x = x + blk.ff(blk.ln2(x))
            r = x[0, -1]
            sigma = r.var(unbiased=False).add(model.ln_f.eps).sqrt()
            delta = torch.zeros_like(r)
            for li, h in heads:
                dh = model.blocks[li].attn.d_head
                P = model.blocks[li].attn.proj.weight[:, h * dh:(h + 1) * dh]
                delta += P @ (means[li][h] - ys[li][h])
            dlog = model.lm_head.weight[shape_ids] @ ((delta - delta.mean()) / sigma * model.ln_f.weight)
            d = dict(zip(cg.SHAPE_TOKENS, dlog.tolist()))
            s, star = it["ctx_shape"], it["q_s_star"]
            others = [d[z] for z in cg.SHAPE_TOKENS if z not in (s, star)]
            preds.append(d[s] - sum(others) / len(others))
    return sum(preds) / len(preds)


def analyse(run, out_dir):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(run["tokenizer"])
    model, _ = load_model(run["ckpt"], None, "train_model", "OverhypothesisLM")
    model.eval()
    check_forward(model, tok)
    seed = int(run["seed"])
    n_layer, H = len(model.blocks), model.blocks[0].attn.n_head
    tag = f"{run['base']}_{run['size']}_seed{seed}" + (f"_{run['label']}" if run.get("label") else "")
    # 1. candidates (before any familiarity item is scored)
    sig, cands = signature(model, tok, seed)
    # 2. ablation means
    means, n_pos = ablation_means(model, tok, run)
    items = make_items(seed, int(run["per_cell"]))
    unseen = [it for it in items if it["kind"] == "carry_same" and it["type"] == "unseen"]
    fam = [it for it in items if it["kind"] == "carry_same" and it["type"] != "unseen"]
    trained = [it for it in items if it["kind"] == "trained"]
    rng = random.Random(RAND_SEED + seed)
    spec = ablate_spec(cands, means)
    # 3. ablation
    c0 = mean_carry(unseen, run_items(model, tok, unseen))
    c1 = mean_carry(unseen, run_items(model, tok, unseen, ablate=spec))
    rand = [mean_carry(unseen, run_items(model, tok, unseen, ablate=ablate_spec(d, means)))
            for d in random_draws(cands, n_layer, H, N_RANDOM, rng)]
    t0 = mean_carry(trained, run_items(model, tok, trained))
    t1 = mean_carry(trained, run_items(model, tok, trained, ablate=spec))
    trand = [mean_carry(trained, run_items(model, tok, trained, ablate=ablate_spec(d, means)))
             for d in random_draws(cands, n_layer, H, N_RANDOM_TRAINED, rng)]
    # 4. attention and per-noun ablated carry
    Lf, Af = run_items(model, tok, unseen + fam, want_att=True)
    Lfa = run_items(model, tok, unseen + fam, ablate=spec)
    per_noun = {}
    for it, l, la, att in zip(unseen + fam, Lf, Lfa, Af):
        d = per_noun.setdefault(it["noun"], {"noun": it["noun"], "type": it["type"], "exposures": it["exposures"],
                                             "att": [], "carry": [], "carry_ablated": []})
        d["att"].append(sum(att[li][h, it["ctx_shape_pos"]].item() for li, h in cands) / len(cands))
        d["carry"].append(measures([it], [l])[0]["carry"])
        d["carry_ablated"].append(measures([it], [la])[0]["carry"])
    # 5. direct logit attribution
    dla = dla_change(model, tok, unseen, cands, means)
    summary = {"base": run["base"], "size": run["size"], "seed": seed, "ckpt": run["ckpt"], "label": run.get("label", ""),
               "candidates": cands, "k": len(cands), "mean_positions": n_pos,
               "unseen_carry": c0, "unseen_carry_ablated": c1, "unseen_reduction": (c0 - c1) / c0 if c0 else None,
               "unseen_carry_random_mean": sum(rand) / len(rand), "unseen_random_draws": rand,
               "trained_carry": t0, "trained_carry_ablated": t1, "trained_carry_random_mean": sum(trand) / len(trand),
               "dla_predicted_change": dla, "observed_change": c1 - c0}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"heads_{tag}.json").write_text(json.dumps({"summary": summary, "signature": sig}, indent=1))
    with open(out / f"per_noun_{tag}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["base", "size", "seed", "ckpt", "label", "noun", "type", "exposures", "attention", "carry",
                    "carry_ablated"])
        for d in per_noun.values():
            w.writerow([run["base"], run["size"], seed, run["ckpt"], run.get("label", ""), d["noun"], d["type"],
                        d["exposures"],
                        np.mean(d["att"]), np.mean(d["carry"]), np.mean(d["carry_ablated"])])
    return summary


SUM_FIELDS = ["base", "size", "seed", "ckpt", "label", "k", "candidates", "mean_positions", "unseen_carry",
              "unseen_carry_ablated", "unseen_reduction", "unseen_carry_random_mean", "trained_carry",
              "trained_carry_ablated", "trained_carry_random_mean", "dla_predicted_change", "observed_change"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    runs = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
    summ = Path(a.out_dir) / "heads_summary.csv"
    done = set()
    if summ.exists():
        done = {r["ckpt"] for r in csv.DictReader(open(summ, encoding="utf-8"))}
    for i, run in enumerate(runs, 1):
        if run["ckpt"] in done:
            continue
        s = analyse(run, a.out_dir)
        new = not summ.exists()
        with open(summ, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=SUM_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({**s, "candidates": " ".join(f"L{l}H{h}" for l, h in s["candidates"])})
        print(f"[{i}/{len(runs)}] {run['base']} {run['size']} seed {run['seed']}: candidates "
              f"{s['candidates']}, reduction {s['unseen_reduction']}", flush=True)


if __name__ == "__main__":
    main()
