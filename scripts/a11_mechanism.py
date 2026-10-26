#!/usr/bin/env python3
"""
a11_mechanism.py  |  Exploratory mechanism analyses of the Amendment 11 (Experiment 3) mixed-design models, added
after the results in answer to a review. Inference only; nothing is written to the models.

For each final model in the manifest (base regular_grouped):

1. Steering with a direction learned from other nouns. Input rows are changed and the registered copy items
   (same-noun context) are scored.
     v_e   = mean row of changing-shape bursts minus mean row of fixed-shape bursts, at exposure level e
     fixed_run at 1,000      + v_100 (direction from different nouns), and + a random vector of the same norm
     fixed_single, rotating_single, absent_single, absent_run at 1,000   + v_1000, and + random of the same norm
     unseen nouns with a typical row                                     + v_1000, and + random of the same norm
     rotating_run at 1,000   - v_100, and - random of the same norm
   Conditions 'own' / 'typical' give the unchanged baselines.

2. Activation patching. For each fixed-shape burst noun F and the changing-shape burst noun R with the same index
   and exposure level (1,000), the target item is the registered item "A F is a s c t thing. A F is a"; the source
   item names R only in the query, "A F is a s c t thing. A R is a". The residual stream after layer l (l = -1: the
   embedding sum, before any block) at one position is copied from the source run into the target run, and the
   target's copy score is read at the last position. Positions: 'noun' (the query noun) and 'last' (the final 'a').
   Also the reverse (R target, F in the source query). Reported with the clean target and full source scores.

3. Attention by layer: from the last position to the context's shape word, mean over heads, own rows, registered
   items, every layer.

  python scripts/a11_mechanism.py --manifest results/amendment11/manifest.csv --out results/amendment11/mechanism.csv
  (resumable)
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus_generator as cg  # noqa: E402
from a10_reanalysis import score  # noqa: E402
from familiarity_eval import check_forward, forward, token_id, typical_rows  # noqa: E402
from familiarity_items import make_items  # noqa: E402
from oneshot_eval import load_model  # noqa: E402

FIELDS = ["base", "size", "seed", "ckpt", "analysis", "cond", "type", "exposures", "layer", "pos", "value",
          "n_items"]
RAND_SEED = 91_000


def patched_logits(model, ids, patch=None):
    """ids (B, T). patch: (layer, pos, tensor (B, d)) replaces the residual after `layer` (-1 = embeddings) at
    `pos`. Returns (logits at the last position, list of residuals after each layer incl. -1) in eval mode."""
    B, T = ids.shape
    x = model.tok_embed(ids) + model.pos_embed(torch.arange(T).unsqueeze(0))
    res = []
    if patch and patch[0] == -1:
        x = x.clone()
        x[:, patch[1], :] = patch[2]
    res.append(x)
    for li, blk in enumerate(model.blocks):
        x = blk(x)
        if patch and patch[0] == li:
            x = x.clone()
            x[:, patch[1], :] = patch[2]
        res.append(x)
    return model.lm_head(model.ln_f(x[:, -1, :])), res


def copy_from_logits(lp, items, shape_ids):
    out = []
    for k, it in enumerate(items):
        L = {w: lp[k, t].item() for w, t in zip(cg.SHAPE_TOKENS, shape_ids)}
        others = [L[w] for w in cg.SHAPE_TOKENS if w != it["target"] and w not in it["exclude"]]
        out.append(L[it["target"]] - sum(others) / len(others))
    return out


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    import make_familiarity_corpora_a11 as m11
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    runs = [r for r in csv.DictReader(open(a.manifest, encoding="utf-8"))
            if r["ckpt"].endswith("model_final.pt") and r["base"] == "regular_grouped"]
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
        rows = m11.load_assignment(seed, run["design"][4:])
        info = {r["noun"]: r for r in rows}
        reg = [it for it in make_items(seed, 3, run["design"]) if it["kind"] == "carry_same"]
        E = model.tok_embed.weight
        orig = E.detach().clone()
        nid = {n: token_id(tok, n) for n in info}
        shape_ids = [token_id(tok, s) for s in cg.SHAPE_TOKENS]
        out = []

        def rec(analysis, cond, typ, e, value, n, layer="", pos=""):
            out.append({"base": run["base"], "size": run["size"], "seed": seed, "ckpt": run["ckpt"],
                        "analysis": analysis, "cond": cond, "type": typ, "exposures": e, "layer": layer, "pos": pos,
                        "value": value, "n_items": n})

        def items_for(nouns):
            return [{"noun": it["noun"], "dim": "shape", "target": it["ctx_shape"], "exclude": [it["s_star"]],
                     "text": it["text"]} for it in reg if it["noun"] in nouns]

        def cell(t, e):
            return sorted(n for n in info if info[n]["type"] == t and int(info[n]["exposures"]) == e)

        def score_cell(nouns, analysis, cond, t, e):
            its = items_for(set(nouns))
            vals = [r[0] for r in score(model, tok, its)]
            rec(analysis, cond, t, e, mean(vals), len(vals))

        g = torch.Generator().manual_seed(RAND_SEED + seed)
        v = {e: orig[[nid[n] for n in cell("rotating_run", e)]].mean(0) - orig[[nid[n] for n in cell("fixed_run", e)]].mean(0)
             for e in (100, 1000)}
        rnd = {e: (lambda r: r / r.norm() * v[e].norm())(torch.randn(orig.shape[1], generator=g)) for e in (100, 1000)}
        # 1. steering
        plan = [("fixed_run", 1000, +1, 100), ("rotating_run", 1000, -1, 100)] + \
               [(t, 1000, +1, 1000) for t in ("fixed_single", "rotating_single", "absent_single", "absent_run")]
        for t, e, sign, src in plan:
            nouns = cell(t, e)
            with torch.no_grad():
                E.copy_(orig)
            score_cell(nouns, "steer", "own", t, e)
            for lab, d in (("v", v[src]), ("random", rnd[src])):
                with torch.no_grad():
                    E.copy_(orig)
                    for n in nouns:
                        E[nid[n]] += sign * d
                score_cell(nouns, "steer", f"{'+' if sign > 0 else '-'}{lab}{src}", t, e)
        un = cell("unseen", 0)
        train_ids = [token_id(tok, n) for n in cg.TRAIN_NOUNS]
        for lab, d in (("typical", None), ("+v1000", v[1000]), ("+random1000", rnd[1000])):
            with torch.no_grad():
                E.copy_(orig)
                T = typical_rows(orig, train_ids, len(un))
                for k, n in enumerate(un):
                    E[nid[n]] = T[k] + (d if d is not None else 0)
            score_cell(un, "steer", lab, "unseen", 0)
        with torch.no_grad():
            E.copy_(orig)
        # 2. patching
        F, R = cell("fixed_run", 1000), cell("rotating_run", 1000)
        nL = len(model.blocks)
        for tgt_list, src_list, direction in ((F, R, "R_into_F"), (R, F, "F_into_R")):
            acc = {}
            for tn, sn in zip(tgt_list, src_list):
                its = items_for({tn})
                tgt_txt = [it["text"] for it in its]
                src_txt = [t[::-1].replace(tn[::-1], sn[::-1], 1)[::-1] for t in tgt_txt]  # replace last mention
                ti = torch.tensor([tok.encode(t).ids for t in tgt_txt])
                si = torch.tensor([tok.encode(t).ids for t in src_txt])
                if ti.shape != si.shape:
                    sys.exit("source and target items differ in length")
                qpos = ti.shape[1] - 3   # "... A F is a": noun, is, a
                if ti[0, qpos].item() != nid[tn] or si[0, qpos].item() != nid[sn]:
                    sys.exit("query noun not where expected")
                with torch.no_grad():
                    lt, _ = patched_logits(model, ti)
                    ls, res_s = patched_logits(model, si)
                    acc.setdefault(("clean", "", ""), []).extend(copy_from_logits(torch.log_softmax(lt, -1), its, shape_ids))
                    acc.setdefault(("source", "", ""), []).extend(copy_from_logits(torch.log_softmax(ls, -1), its, shape_ids))
                    for li in range(-1, nL):
                        for pname, p in (("noun", qpos), ("last", ti.shape[1] - 1)):
                            lp_, _ = patched_logits(model, ti, (li, p, res_s[li + 1][:, p, :]))
                            acc.setdefault(("patch", li, pname), []).extend(
                                copy_from_logits(torch.log_softmax(lp_, -1), its, shape_ids))
            for (cond, li, pname), vals in acc.items():
                rec("patch", f"{direction}:{cond}", "fixed_run" if direction == "R_into_F" else "rotating_run", 1000,
                    mean(vals), len(vals), li, pname)
        # 3. attention by layer
        for t in ("fixed_run", "rotating_run", "fixed_single", "rotating_single", "absent_single", "absent_run"):
            for e in (100, 1000):
                its = items_for(set(cell(t, e)))
                enc = [tok.encode(it["text"]).ids for it in its]
                x = torch.tensor(enc)
                with torch.no_grad():
                    _, att = forward(model, x, None, True)
                pos = [en.index(token_id(tok, it["target"])) for en, it in zip(enc, its)]
                for li, at in enumerate(att):
                    vals = [at[k, :, p].mean().item() for k, p in enumerate(pos)]
                    rec("attention", "own", t, e, mean(vals), len(vals), li, "shape")
        new = not os.path.exists(a.out)
        with open(a.out, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(out)
        print(f"[{i}/{len(runs)}] {run['base']} {run['size']} seed {seed} done", flush=True)


if __name__ == "__main__":
    main()
