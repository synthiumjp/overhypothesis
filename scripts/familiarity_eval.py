#!/usr/bin/env python3
"""
familiarity_eval.py  |  Behavioural measures for Amendment 9 (docs/AMENDMENT9.md section 3.5).

For one model, every item of its seed (scripts/familiarity_items.py) is scored at the last
position, from the log-softmax over the whole vocabulary, keeping the ten shape words. With
L(x) the log probability of shape x:
  carry       L(s) - mean L over shapes not in {s, s*}              (same-noun items)
  carry_diff  L(s) - mean L over shapes not in {s, s*(query noun)}  (different-noun items; items
              with s = s*(query noun) are dropped from both, for the paired same-minus-different)
  stored      L(s*) - mean L over shapes not in {s, s*}
  compete     L(s*) - L(s)
  know        L(s*) - mean L over the other nine shapes              (knowledge items)
  agree       L(s*) - mean L over the other nine shapes, after a context with s = s*
  trained     L(s) - mean L over the other nine shapes               (trained kinds)
Unseen familiarity nouns are scored twice: as trained (emb = original) and with the typical
embedding of the Amendment 8 rescoring (emb = typical: mean input row of the 32 trained kinds
plus orthonormal offsets of typical length; output rows unchanged).

The forward pass is written out here (forward()) so that circuit_analysis.py can ablate heads
and read attention with exactly the same computation; it is checked against the model's own
forward on every model.

Usage:
  python scripts/familiarity_eval.py --manifest results/amendment9/manifest_pilot.csv --out results/amendment9/scores_pilot.csv
Manifest columns: base, size, seed, per_cell, ckpt, tokenizer.
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus_generator as cg  # noqa: E402
from familiarity_items import make_items  # noqa: E402
from oneshot_eval import load_model  # noqa: E402

TYPICAL_SEED = 2026


# ---------------------------------------------------------------- explicit forward
def forward(model, ids, ablate=None, want_att=False):
    """ids: LongTensor (B, T). ablate: {layer: (head_list, mean_tensor (n_head, d_head))};
    the listed heads' outputs at the last position are replaced by their means.
    Returns logits at the last position (B, V) and, if want_att, the attention of every layer
    from the last position (list of (B, n_head, T))."""
    B, T = ids.shape
    pos = torch.arange(T).unsqueeze(0)
    x = model.tok_embed(ids) + model.pos_embed(pos)
    atts = []
    for li, blk in enumerate(model.blocks):
        a = blk.attn
        h = blk.ln1(x)
        q, k, v = a.qkv(h).split(model.d_model, dim=2)
        H, dh = a.n_head, a.d_head
        q = q.view(B, T, H, dh).transpose(1, 2)
        k = k.view(B, T, H, dh).transpose(1, 2)
        v = v.view(B, T, H, dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        att = att.masked_fill(a.mask[:, :, :T, :T] == 0, float("-inf")).softmax(-1)
        y = att @ v                                                     # (B, H, T, dh)
        if ablate and li in ablate:
            heads, mean = ablate[li]
            y = y.clone()
            for hh in heads:
                y[:, hh, -1, :] = mean[hh]
        if want_att:
            atts.append(att[:, :, -1, :])
        x = x + a.proj(y.transpose(1, 2).reshape(B, T, model.d_model))
        x = x + blk.ff(blk.ln2(x))
    logits = model.lm_head(model.ln_f(x[:, -1, :]))
    return (logits, atts) if want_att else logits


def head_outputs(model, ids, positions):
    """Per-head outputs y (before the output projection) at the given positions.
    ids (B, T); positions: list of (b, t). Returns tensor (n_layer, len(positions), n_head, d_head)."""
    B, T = ids.shape
    x = model.tok_embed(ids) + model.pos_embed(torch.arange(T).unsqueeze(0))
    bi = torch.tensor([p[0] for p in positions])
    ti = torch.tensor([p[1] for p in positions])
    out = []
    for blk in model.blocks:
        a = blk.attn
        q, k, v = a.qkv(blk.ln1(x)).split(model.d_model, dim=2)
        H, dh = a.n_head, a.d_head
        q, k, v = (z.view(B, T, H, dh).transpose(1, 2) for z in (q, k, v))
        att = ((q @ k.transpose(-2, -1)) / math.sqrt(dh)).masked_fill(a.mask[:, :, :T, :T] == 0,
                                                                       float("-inf")).softmax(-1)
        y = att @ v
        out.append(y[bi, :, ti, :])                                      # (P, H, dh)
        x = x + a.proj(y.transpose(1, 2).reshape(B, T, model.d_model))
        x = x + blk.ff(blk.ln2(x))
    return torch.stack(out)


def check_forward(model, tok):
    ids = torch.tensor([tok.encode("A wug is a zibby lavo shorfi thing. A wug is a").ids])
    with torch.no_grad():
        ref = model(ids)[0, -1]
        mine = forward(model, ids)[0]
    err = (ref - mine).abs().max().item()
    if err > 1e-4:
        sys.exit(f"explicit forward differs from the model's forward by {err}")


# ---------------------------------------------------------------- helpers
def token_id(tok, w):
    ids = tok.encode(w).ids
    if len(ids) != 1:
        sys.exit(f"'{w}' is not one token")
    return ids[0]


def typical_rows(E, train_ids, n_new, seed=TYPICAL_SEED):
    T = E[train_ids]
    mu = T.mean(0)
    r = (T - mu).norm(dim=1).mean()
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(E.shape[1], n_new, generator=g))
    return mu + r * q.T


def run_items(model, tok, items, ablate=None, want_att=False, batch=256):
    """Log-probabilities of the ten shapes at the last position, per item (and attention)."""
    shape_ids = [token_id(tok, s) for s in cg.SHAPE_TOKENS]
    by_len = defaultdict(list)
    enc = [tok.encode(it["text"]).ids for it in items]
    for i, e in enumerate(enc):
        by_len[len(e)].append(i)
    L = [None] * len(items)
    A = [None] * len(items)
    with torch.no_grad():
        for n, idx in by_len.items():
            for j in range(0, len(idx), batch):
                part = idx[j:j + batch]
                ids = torch.tensor([enc[i] for i in part])
                res = forward(model, ids, ablate, want_att)
                logits, att = res if want_att else (res, None)
                lp = torch.log_softmax(logits.float(), -1)[:, shape_ids]
                for r, i in enumerate(part):
                    L[i] = lp[r].tolist()
                    if want_att:
                        A[i] = [a[r] for a in att]                         # per layer (H, T)
    return (L, A) if want_att else L


def measures(items, L):
    """Item-level measures from shape log-probabilities."""
    S = cg.SHAPE_TOKENS
    out = []
    for it, lp in zip(items, L):
        d = dict(zip(S, lp))
        k = it["kind"]
        rec = {"item_id": it["item_id"], "kind": k, "noun": it["noun"], "type": it["type"],
               "exposures": it["exposures"]}
        if k in ("carry_same", "carry_diff"):
            s, star = it["ctx_shape"], it["q_s_star"]
            others = [d[x] for x in S if x not in (s, star)]
            if s == star:
                rec["drop"] = 1
            rec["carry"] = d[s] - sum(others) / len(others)
            if k == "carry_same":
                rec["stored"] = d[star] - sum(others) / len(others)
                rec["compete"] = d[star] - d[s]
        elif k in ("know", "agree"):
            star = it["s_star"]
            others = [d[x] for x in S if x != star]
            rec["know" if k == "know" else "agree"] = d[star] - sum(others) / len(others)
        elif k == "trained":
            s = it["ctx_shape"]
            others = [d[x] for x in S if x != s]
            rec["carry"] = d[s] - sum(others) / len(others)
        out.append(rec)
    return out


def summarise(meas, items):
    """Noun-level means. Same and different carry use the same items (pairs with s = s* of the
    different-noun query dropped from both)."""
    drop = {m["item_id"] - 1 for m in meas if m["kind"] == "carry_diff" and m.get("drop")}
    acc = defaultdict(lambda: defaultdict(list))
    for m in meas:
        key = (m["noun"], m["type"], m["exposures"])
        if m["kind"] == "carry_same":
            acc[key]["carry"].append(m["carry"])
            acc[key]["stored"].append(m["stored"])
            acc[key]["compete"].append(m["compete"])
            if m["item_id"] not in drop:
                acc[key]["carry_paired"].append(m["carry"])
        elif m["kind"] == "carry_diff" and not m.get("drop"):
            acc[key]["carry_diff"].append(m["carry"])
        elif m["kind"] == "know":
            acc[key]["know"].append(m["know"])
        elif m["kind"] == "agree":
            acc[key]["agree"].append(m["agree"])
        elif m["kind"] == "trained":
            acc[key]["carry"].append(m["carry"])
    rows = []
    for (noun, typ, e), d in acc.items():
        r = {"noun": noun, "type": typ, "exposures": e}
        for q, v in d.items():
            r[q] = sum(v) / len(v)
            r["n_" + q] = len(v)
        if "carry_paired" in d and "carry_diff" in d:
            r["noun_specific"] = r["carry_paired"] - r["carry_diff"]
        rows.append(r)
    return rows


OUT_FIELDS = ["base", "size", "seed", "ckpt", "label", "emb", "noun", "type", "exposures", "carry", "stored", "compete",
              "know", "agree", "carry_paired", "carry_diff", "noun_specific", "n_carry", "n_know", "n_carry_diff"]


def score_model(run, emb_conditions=("original", "typical")):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(run["tokenizer"])
    model, _ = load_model(run["ckpt"], None, "train_model", "OverhypothesisLM")
    model.eval()
    check_forward(model, tok)
    items = make_items(int(run["seed"]), int(run["per_cell"]))
    rows_out = []
    E = model.tok_embed.weight
    orig = E.detach().clone()
    unseen = sorted({it["noun"] for it in items if it["type"] == "unseen"})
    unseen_ids = [token_id(tok, n) for n in unseen]
    train_ids = [token_id(tok, n) for n in cg.TRAIN_NOUNS]
    for emb in emb_conditions:
        with torch.no_grad():
            E.copy_(orig)
            if emb == "typical":
                E[unseen_ids] = typical_rows(orig, train_ids, len(unseen_ids))
        its = items if emb == "original" else [it for it in items if it["type"] == "unseen"]
        L = run_items(model, tok, its)
        for r in summarise(measures(its, L), its):
            rows_out.append({"base": run["base"], "size": run["size"], "seed": run["seed"], "ckpt": run["ckpt"],
                             "label": run.get("label", ""), "emb": emb, **r})
    with torch.no_grad():
        E.copy_(orig)
    return rows_out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    runs = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
    done = set()
    if os.path.exists(a.out):
        done = {r["ckpt"] for r in csv.DictReader(open(a.out, encoding="utf-8"))}
    new = not os.path.exists(a.out)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for i, run in enumerate(runs, 1):
        if run["ckpt"] in done:
            continue
        rows = score_model(run)
        with open(a.out, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=OUT_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
                new = False
            w.writerows(rows)
        print(f"[{i}/{len(runs)}] {run['base']} {run['size']} seed {run['seed']} scored", flush=True)


if __name__ == "__main__":
    main()
