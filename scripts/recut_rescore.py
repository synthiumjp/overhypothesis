#!/usr/bin/env python3
"""
recut_rescore.py  |  Exploratory rescoring for the ACL recut (23 Oct 2026; not registered)

Two checks on the 120 fresh-text models of Amendment 8, using the same one-exemplar
items (k = 1) and the same scoring as oneshot_eval.py:

1. Typical novel-noun embeddings. The eight novel test nouns have untrained input
   embeddings. Here each is replaced by the mean input embedding of the 32 training
   nouns plus a distinct offset of typical size: offset_j = r * u_j, where r is the mean
   distance of the training-noun embeddings from their mean and the u_j are orthonormal
   random directions (fixed seed). The novel nouns then look like ordinary nouns to the
   model but remain distinct from one another, so same- and different-noun items still
   differ. Output weights are not changed.

2. Changed-word Delta. Besides the usual Delta (difference in summed log probability of
   the whole continuation), the script records the difference in log probability of the
   changed word alone, given the identical prefix.

Each model is scored under both embedding conditions ("original", "typical"); both Delta
measures come from the same forward pass. With emb = original, delta_full reproduces
results/amendment8/raw (checked by recut_rescore_analysis.py).

Usage (from the repository folder, in the training venv):
  python scripts/recut_rescore.py --manifests results/amendment8/manifest_*.csv \
      --items data/wug_tests/oneshot_items.csv --out results/recut/rescore_raw.csv
Results are appended after every model, so the run can be stopped and resumed.
"""
import argparse, csv, glob, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oneshot_eval import read_items, load_model  # noqa: E402
from corpus_generator import TRAIN_NOUNS, TEST_NOUNS  # noqa: E402

FIELDS = ["condition", "size", "seed", "ckpt", "emb", "item_id", "k", "match",
          "probe_dim", "delta_full", "delta_word"]


def single_token_id(tok, word):
    ids = tok.encode("A " + word).ids
    if len(ids) != 2:
        sys.exit(f"'{word}' is not a single token ({tok.encode('A ' + word).tokens}).")
    return ids[1]


def typical_rows(E, train_ids, n_new, seed=2026):
    """Mean of training-noun rows plus orthonormal offsets of typical length."""
    import torch
    T = E[train_ids]
    mu = T.mean(0)
    r = (T - mu).norm(dim=1).mean()
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(E.shape[1], n_new, generator=g))
    return mu + r * q.T  # (n_new, d)


def token_logprobs(model, ids):
    import torch
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model(x)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    logits = getattr(logits, "logits", logits)
    lp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    return lp.gather(1, x[0, 1:].unsqueeze(1)).squeeze(1).tolist()  # lp of token t+1


def score(model, tok, rows, block):
    """Return (delta_full, delta_word) per row."""
    cache, out = {}, []
    for r in rows:
        seqs = []
        for q in (r["query_rep"], r["query_new"]):
            text = r["context"] + " " + q
            if text not in cache:
                ids = tok.encode(text).ids
                if len(ids) > block:
                    raise ValueError(f"item {r['item_id']}: {len(ids)} tokens > {block}")
                cache[text] = (ids, token_logprobs(model, ids))
            seqs.append(cache[text])
        (a_ids, a_lp), (b_ids, b_lp) = seqs
        if len(a_ids) != len(b_ids):
            raise ValueError(f"item {r['item_id']}: continuations differ in length")
        diff = [i for i, (x, y) in enumerate(zip(a_ids, b_ids)) if x != y]
        if not diff or diff[-1] - diff[0] + 1 != len(diff):
            raise ValueError(f"item {r['item_id']}: changed tokens not one contiguous span")
        # a_lp[i - 1] is the log probability of token i given tokens < i.
        word = sum(a_lp[i - 1] - b_lp[i - 1] for i in diff)
        out.append((sum(a_lp) - sum(b_lp), word))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifests", nargs="+", required=True)
    p.add_argument("--items", default="data/wug_tests/oneshot_items.csv")
    p.add_argument("--ks", default="1")
    p.add_argument("--out", default="results/recut/rescore_raw.csv")
    p.add_argument("--limit", type=int, default=0, help="score only the first N models (testing)")
    a = p.parse_args()

    import torch
    from tokenizers import Tokenizer
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

    keep = {int(k) for k in a.ks.split(",")}
    rows = [r for r in read_items(a.items) if r["k"] in keep]
    runs = []
    for pat in a.manifests:
        for m in sorted(glob.glob(pat)):
            with open(m, encoding="utf-8") as f:
                runs += list(csv.DictReader(f))
    if a.limit:
        runs = runs[:a.limit]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            done = {(r["ckpt"], r["emb"]) for r in csv.DictReader(f)}
    new_file = not os.path.exists(a.out)
    print(f"{len(runs)} models x 2 embedding conditions x {len(rows)} items")

    for i, run in enumerate(runs, 1):
        if (run["ckpt"], "original") in done and (run["ckpt"], "typical") in done:
            continue
        tok = Tokenizer.from_file(run["tokenizer"])
        model, block = load_model(run["ckpt"], None, "train_model", "OverhypothesisLM")
        train_ids = [single_token_id(tok, n) for n in TRAIN_NOUNS]
        test_ids = [single_token_id(tok, n) for n in TEST_NOUNS]
        E = model.tok_embed.weight
        orig = E.detach().clone()
        typical = typical_rows(orig, train_ids, len(test_ids))
        for emb in ("original", "typical"):
            if (run["ckpt"], emb) in done:
                continue
            with torch.no_grad():
                E.copy_(orig)
                if emb == "typical":
                    E[test_ids] = typical
            res = score(model, tok, rows, block)
            with open(a.out, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new_file:
                    w.writeheader()
                    new_file = False
                for r, (df, dw) in zip(rows, res):
                    w.writerow({"condition": run["condition"], "size": run["size"],
                                "seed": run["seed"], "ckpt": run["ckpt"], "emb": emb,
                                "item_id": r["item_id"], "k": r["k"], "match": r["match"],
                                "probe_dim": r["probe_dim"], "delta_full": df, "delta_word": dw})
        print(f"[{i}/{len(runs)}] {run['condition']} {run['size']} seed {run['seed']} done", flush=True)


if __name__ == "__main__":
    main()
