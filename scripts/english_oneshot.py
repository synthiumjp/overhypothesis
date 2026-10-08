#!/usr/bin/env python3
"""
english_oneshot.py  |  Study 1d: the few-shot overhypothesis test in natural
English, run on pretrained language models (Pythia checkpoints, BabyLM
baselines, large open models). Designed for Apple Silicon (MPS) but runs on
CUDA or CPU.

The measure matches Study 1b (oneshot_eval.py). After k exemplars of a novel
noun, two query passages differ in one dimension only. One repeats a value
seen in context, the other uses an unseen value.

    delta_d    = log P(passage with repeated value) - log P(passage with new value)
    OI_colour  = delta_shape - delta_colour
    OI_texture = delta_shape - delta_texture

Each feature sits in its own sentence ("It is oval. It is amber. It is
silky.") so English adjective order cannot favour one dimension. The order of
the three sentences is counterbalanced across items.

Item types
  k = 1           : one exemplar; only a learned prior can favour shape.
  k = 2, 4        : the context holds one dimension constant (stable_dim =
                    shape or colour, counterbalanced). Comparing shape-stable
                    with colour-stable contexts removes the copying confound:
                    copying favours whichever dimension repeated, a shape prior
                    favours shape in both.
  match = False   : the query names a different novel noun.

Subcommands
  make-items     generate the item file (deterministic)
  word-stats     print frequency matching of the word lists (needs wordfreq)
  eval           score items with every model in a manifest
  summarise      per-model indices with bootstrap confidence intervals

Typical run (Mac, from the repo root, venv active)
  python scripts/english_oneshot.py make-items --out data/wug_tests/english_items.csv
  python scripts/english_oneshot.py eval --items data/wug_tests/english_items.csv \\
         --manifest results/study1d/models.csv --out results/study1d/scores.csv
  python scripts/english_oneshot.py summarise --scores results/study1d/scores.csv \\
         --out results/study1d/summary

Dependencies: torch, transformers, accelerate, pandas, numpy.
"""

import argparse
import csv
import gc
import itertools
import math
import os
import random
import shutil
import sys
import time

SHAPES = ["pointed", "oval", "circular", "curved",
          "rectangular", "triangular", "spherical", "cylindrical"]
COLOURS = ["grey", "purple", "amber", "violet",
           "scarlet", "crimson", "maroon", "turquoise"]
TEXTURES = ["smooth", "shiny", "sticky", "fuzzy",
            "fluffy", "furry", "glossy", "silky"]
DIMS = {"shape": SHAPES, "colour": COLOURS, "texture": TEXTURES}
DIM_ORDER = ["shape", "colour", "texture"]
NOUNS = ["snerg", "twick", "vulf", "wazz", "glorp", "yeff", "zull", "kleb"]
ORDERS = list(itertools.permutations(DIM_ORDER))  # 6 sentence orders

ITEM_FIELDS = ["item_id", "noun", "query_noun", "match", "k", "draw",
               "stable_dim", "probe_dim", "order", "rep_value", "new_value",
               "context", "query_rep", "query_new"]


def passage(noun, vals, order, first):
    head = f"Here is {'a' if first else 'another'} {noun}."
    return " ".join([head] + [f"It is {vals[d]}." for d in order])


# --------------------------------------------------------------------------
# make-items
# --------------------------------------------------------------------------

def make_items(args):
    rng = random.Random(args.seed)
    rows, iid = [], 0
    for noun in NOUNS:
        others = [n for n in NOUNS if n != noun]
        for draw in range(args.draws):
            for k in (1, 2, 4):
                stables = ["none"] if k == 1 else ["shape", "colour"]
                for stable in stables:
                    order = ORDERS[rng.randrange(len(ORDERS))]
                    ctx_vals = {}
                    for d in DIM_ORDER:
                        if d == stable:
                            ctx_vals[d] = [rng.choice(DIMS[d])] * k
                        else:
                            ctx_vals[d] = rng.sample(DIMS[d], k)
                    context = " ".join(
                        passage(noun, {d: ctx_vals[d][i] for d in DIM_ORDER},
                                order, first=(i == 0))
                        for i in range(k))
                    other = rng.choice(others)
                    for probe in DIM_ORDER:
                        rep = ctx_vals[probe][-1]
                        new = rng.choice([v for v in DIMS[probe] if v not in ctx_vals[probe]])
                        fill = {d: rng.choice([v for v in DIMS[d] if v not in ctx_vals[d]])
                                for d in DIM_ORDER if d != probe}
                        for qnoun, match in ((noun, True), (other, False)):
                            def q(val):
                                vals = dict(fill)
                                vals[probe] = val
                                # A different kind is introduced with "a", the same kind with "another".
                                return passage(qnoun, vals, order, first=not match)
                            rows.append({
                                "item_id": iid, "noun": noun, "query_noun": qnoun,
                                "match": match, "k": k, "draw": draw,
                                "stable_dim": stable, "probe_dim": probe,
                                "order": "-".join(order),
                                "rep_value": rep, "new_value": new,
                                "context": context,
                                "query_rep": q(rep), "query_new": q(new)})
                            iid += 1
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ITEM_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} items to {args.out}")


def word_stats(args):
    from wordfreq import zipf_frequency as z
    import statistics as st
    for d in DIM_ORDER:
        f = [z(w, "en") for w in DIMS[d]]
        print(f"{d:8s} mean Zipf {st.mean(f):.2f} (SD {st.stdev(f):.2f}): "
              + ", ".join(f"{w} {z(w, 'en'):.2f}" for w in DIMS[d]))
    print("nouns    " + ", ".join(f"{n} {z(n, 'en'):.2f}" for n in NOUNS))


def read_items(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["k"] = int(r["k"])
        r["match"] = r["match"] in ("True", "true", "1")
    return rows


# --------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------

def pick_device(name):
    import torch
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_source(model_id, revision):
    """Where to load a checkpoint from.

    Some Pythia branches (pythia-2.8b, steps 0 to 64000) hold a stale
    single-file copy of the final weights (model.safetensors and
    pytorch_model.bin, same SHA-256 as main) next to the correct sharded
    files for that step. transformers prefers the single file, so every such
    branch loads the final weights. When a branch has both, download only the
    shards and the branch's own config, and load from that folder.
    """
    if not revision:
        return model_id, None, None
    from huggingface_hub import HfApi, snapshot_download
    files = set(HfApi().list_repo_files(model_id, revision=revision))
    if {"model.safetensors", "model.safetensors.index.json"} <= files:
        import tempfile
        tmp = tempfile.mkdtemp(prefix="ovh_ckpt_")
        snapshot_download(model_id, revision=revision, local_dir=tmp,
                          allow_patterns=["config.json", "generation_config.json",
                                          "model.safetensors.index.json",
                                          "model-*-of-*.safetensors", "tokenizer*",
                                          "special_tokens_map.json"])
        print(f"    {model_id}@{revision}: loading sharded weights, stale single file ignored")
        return tmp, None, tmp
    return model_id, revision, None


def load(model_id, revision, dtype, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tdtype = {"float32": torch.float32, "float16": torch.float16,
              "bfloat16": torch.bfloat16}[dtype]
    src, rev, tmp = resolve_source(model_id, revision)
    try:
        tok = AutoTokenizer.from_pretrained(src, revision=rev)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                src, revision=rev, dtype=tdtype, low_cpu_mem_usage=True)
        except TypeError:  # older transformers
            model = AutoModelForCausalLM.from_pretrained(
                src, revision=rev, torch_dtype=tdtype, low_cpu_mem_usage=True)
        model.to(device).eval()
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    return model, tok


def seq_logprobs(model, tok, texts, device, batch_size):
    """Sum of token log-probabilities for each text (first token excluded)."""
    import torch
    bos = tok.bos_token_id
    enc = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    if bos is not None:
        enc = [[bos] + e for e in enc]
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    out = []
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    res = [None] * len(enc)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        L = max(len(enc[i]) for i in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            ids[j, :len(enc[i])] = torch.tensor(enc[i])
            mask[j, :len(enc[i])] = 1
        ids, mask = ids.to(device), mask.to(device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits.float()
        lp = torch.log_softmax(logits[:, :-1], dim=-1)
        tgt = ids[:, 1:]
        tok_lp = lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1) * mask[:, 1:]
        for j, i in enumerate(idx):
            res[i] = tok_lp[j].sum().item()
    return res


def eval_models(args):
    items = read_items(args.items)
    with open(args.manifest, encoding="utf-8") as f:
        runs = [r for r in csv.DictReader(f) if not r["model_id"].startswith("#")]
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = {(r["model_id"], r["revision"]) for r in csv.DictReader(f)}
    device = pick_device(args.device)
    # Models already in the Hugging Face cache before this run are never deleted.
    keep = set()
    if args.delete_cache:
        from huggingface_hub import scan_cache_dir
        try:
            keep = {(r.repo_id, rv.commit_hash) for r in scan_cache_dir().repos
                    for rv in r.revisions}
        except Exception:
            pass
        if keep:
            print(f"--delete_cache: {len(keep)} model revisions already cached will be kept")
    texts = sorted({r["context"] + " " + q for r in items for q in (r["query_rep"], r["query_new"])})
    new_file = not os.path.exists(args.out)
    for n, run in enumerate(runs, 1):
        key = (run["model_id"], run.get("revision", ""))
        if key in done:
            print(f"[{n}/{len(runs)}] skip {key} (already scored)")
            continue
        t0 = time.time()
        dtype = run.get("dtype") or args.dtype
        try:
            model, tok = load(run["model_id"], run.get("revision", ""), dtype, device)
            lp = dict(zip(texts, seq_logprobs(model, tok, texts, device,
                                              int(run.get("batch_size") or args.batch_size))))
        except Exception as e:  # keep going; report at the end of the line
            print(f"[{n}/{len(runs)}] FAILED {key}: {type(e).__name__}: {e}")
            continue
        with open(args.out, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["family", "model_id", "revision", "params",
                                              "train_tokens", "item_id", "noun", "draw", "k",
                                              "match", "stable_dim", "probe_dim", "delta"])
            if new_file:
                w.writeheader()
                new_file = False
            for r in items:
                c = r["context"] + " "
                w.writerow({"family": run["family"], "model_id": run["model_id"],
                            "revision": run.get("revision", ""), "params": run["params"],
                            "train_tokens": run["train_tokens"], "item_id": r["item_id"],
                            "noun": r["noun"], "draw": r["draw"], "k": r["k"],
                            "match": r["match"], "stable_dim": r["stable_dim"],
                            "probe_dim": r["probe_dim"],
                            "delta": lp[c + r["query_rep"]] - lp[c + r["query_new"]]})
        print(f"[{n}/{len(runs)}] {key}: {len(items)} items in {time.time() - t0:.0f}s on {device}")
        del model
        gc.collect()
        if device == "mps":
            import torch
            torch.mps.empty_cache()
        if args.delete_cache:
            from huggingface_hub import scan_cache_dir
            info = scan_cache_dir()
            for repo in info.repos:
                if repo.repo_id == run["model_id"]:
                    revs = [rv.commit_hash for rv in repo.revisions
                            if (repo.repo_id, rv.commit_hash) not in keep]
                    if revs:
                        info.delete_revisions(*revs).execute()


# --------------------------------------------------------------------------
# summarise
# --------------------------------------------------------------------------

def summarise(args):
    import numpy as np
    import pandas as pd
    d = pd.read_csv(args.scores, dtype={"revision": str}, keep_default_na=False)
    d["delta"] = pd.to_numeric(d["delta"])
    d["match"] = d["match"].astype(str).isin(["True", "true", "1"])
    keys = ["family", "model_id", "revision", "params", "train_tokens"]
    cell = ["noun", "draw", "k", "match", "stable_dim"]
    wide = d.pivot_table(index=keys + cell, columns="probe_dim", values="delta").reset_index()
    wide["OI_colour"] = wide["shape"] - wide["colour"]
    wide["OI_texture"] = wide["shape"] - wide["texture"]
    rng = np.random.default_rng(2026)
    out = []
    for gk, g in wide.groupby(keys + ["k", "match", "stable_dim"]):
        row = dict(zip(keys + ["k", "match", "stable_dim"], gk))
        for col in ("shape", "colour", "texture", "OI_colour", "OI_texture"):
            row[col] = g[col].mean()
        # Cluster bootstrap over the 8 nouns (items are balanced across nouns).
        for col in ("OI_colour", "OI_texture"):
            nm = g.groupby("noun")[col].mean().to_numpy()
            boots = nm[rng.integers(0, len(nm), (args.n_boot, len(nm)))].mean(1)
            row[col + "_lo"], row[col + "_hi"] = np.percentile(boots, [2.5, 97.5])
        row["n_items"] = len(g)
        out.append(row)
    s = pd.DataFrame(out)
    s.to_csv(args.out + "_per_model.csv", index=False)
    show = s[(s.match) & (s.k == 1)].sort_values(["family", "params", "train_tokens"])
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(show[["family", "model_id", "revision", "train_tokens", "shape", "colour",
                    "texture", "OI_colour", "OI_colour_lo", "OI_colour_hi",
                    "OI_texture"]].round(3).to_string(index=False))
    print(f"\nFull table: {args.out}_per_model.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-items")
    m.add_argument("--draws", type=int, default=12)
    m.add_argument("--seed", type=int, default=2027)
    m.add_argument("--out", required=True)
    sub.add_parser("word-stats")
    e = sub.add_parser("eval")
    e.add_argument("--items", required=True)
    e.add_argument("--manifest", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--device", default="auto")
    e.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    e.add_argument("--batch_size", type=int, default=32)
    e.add_argument("--delete_cache", action="store_true",
                   help="Delete each model's download after scoring (saves disk)")
    s = sub.add_parser("summarise")
    s.add_argument("--scores", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n_boot", type=int, default=2000)
    args = p.parse_args()
    {"make-items": make_items, "word-stats": word_stats, "eval": eval_models,
     "summarise": summarise}[args.cmd](args)


if __name__ == "__main__":
    main()
