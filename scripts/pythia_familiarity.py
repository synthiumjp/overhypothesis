#!/usr/bin/env python3
"""
pythia_familiarity.py  |  Amendment 10 Part B: colour carry in pretrained models with the
nouns' input embeddings swapped (docs/AMENDMENT10.md). Inference only.

For each noun (data/amendment10/nouns.csv) and each embedding condition, the model reads short
contexts that give the noun a colour, then a query that mentions the noun again, and we read the
log probability of every colour word at the next position, renormalised over the colour set
(data/amendment10/colours.csv):
  carry = mean over frames and ordered colour pairs (c1, c2), c1 != c2, of
          log p(c1 | context with c1) - log p(c1 | context with c2).
The noun's own colour prior cancels in this contrast. Also recorded, from the neutral query
"The {noun} was" with no context: the entropy of the renormalised colour distribution and, for
colour-typical nouns, the log probability of the coded colour minus the mean over the others.

Embedding conditions (input embedding only; output embedding and all other weights unchanged):
  original   the trained row
  atypical   a random direction with the norm of the noun's own row (seed fixed per noun)
  atypical_init  the same random direction with the norm of a row at initialisation
             (initializer_range x sqrt(hidden size)), the counterpart of the untrained rows of
             the unseen nouns in Amendments 8 and 9
  typical    the mean row of all listed nouns plus a distinct orthonormal offset of typical
             length (the procedure of Amendments 8 and 9)

  python scripts/pythia_familiarity.py --model EleutherAI/pythia-1b-deduped --out results/amendment10/pythia_scores.csv
  (--revision step143000 by default; float32 on CPU)
"""
import argparse
import csv
import os
from pathlib import Path

import torch

FRAMES = [("The {noun} was {colour}. The {noun} was", "The {noun} was"),
          ("This {noun} is {colour}. This {noun} is", "This {noun} is"),
          ("I saw the {colour} {noun}. The {noun} was", "I saw the {noun}. The {noun} was")]
SWAP_SEED = 71_000
FIELDS = ["model", "revision", "noun", "group", "colour", "pile_count", "pair", "kind", "count_type", "condition", "carry",
          "prior_entropy", "prior_typical", "row_norm"]


def one_id(tok, w):
    ids = tok(" " + w, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise SystemExit(f"' {w}' is not one token")
    return ids[0]


def last_logprobs(model, tok, texts, colour_ids, batch=64):
    """Renormalised log probabilities of the colour words at the last token of each text."""
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False)
        with torch.no_grad():
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        last = enc["attention_mask"].sum(1) - 1
        lp = torch.log_softmax(logits[torch.arange(len(last)), last].float(), -1)[:, colour_ids]
        out.append(lp - torch.logsumexp(lp, -1, keepdim=True))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="step143000")
    ap.add_argument("--nouns", default="data/amendment10/nouns.csv")
    ap.add_argument("--colours", default="data/amendment10/colours.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--local_tokenizer", default=None, help="testing only")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as fh:
            if any(r["model"] == a.model and r["revision"] == a.revision for r in csv.DictReader(fh)):
                print(f"{a.model} {a.revision} already scored in {a.out}; skipped")
                return
    tok = AutoTokenizer.from_pretrained(a.local_tokenizer or a.model, revision=None if a.local_tokenizer else a.revision)
    tok.padding_side = "right"
    if tok.pad_token is None:
        # any id will do: padded positions are masked and only the last real token is read
        tok.pad_token = tok.eos_token or tok.unk_token or tok.convert_ids_to_tokens(0)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.float32)
    model.eval()
    nouns = list(csv.DictReader(open(a.nouns, encoding="utf-8")))
    colours = [r["colour"] for r in csv.DictReader(open(a.colours, encoding="utf-8"))]
    cid = [one_id(tok, c) for c in colours]
    nid = [one_id(tok, r["noun"]) for r in nouns]
    E = model.get_input_embeddings().weight
    orig = E.detach().clone()
    rows_n = orig[nid]
    mu = rows_n.mean(0)
    r = (rows_n - mu).norm(dim=1).mean()
    g = torch.Generator().manual_seed(SWAP_SEED)
    q, _ = torch.linalg.qr(torch.randn(E.shape[1], len(nid), generator=g))
    typical = mu + r * q.T
    atyp = []
    for j in range(len(nid)):
        gj = torch.Generator().manual_seed(SWAP_SEED + 1 + j)
        v = torch.randn(E.shape[1], generator=gj)
        atyp.append(v / v.norm() * orig[nid[j]].norm())
    atyp = torch.stack(atyp)
    init_norm = float(getattr(model.config, "initializer_range", 0.02)) * E.shape[1] ** 0.5
    atyp_init = atyp / atyp.norm(dim=1, keepdim=True) * init_norm
    out_rows = []
    for cond in ("original", "atypical", "atypical_init", "typical"):
        with torch.no_grad():
            E.copy_(orig)
            if cond == "atypical":
                E[nid] = atyp
            elif cond == "atypical_init":
                E[nid] = atyp_init
            elif cond == "typical":
                E[nid] = typical
        for j, n in enumerate(nouns):
            texts = [f.format(noun=n["noun"], colour=c) for f, _ in FRAMES for c in colours]
            lp = last_logprobs(model, tok, texts, cid).view(len(FRAMES), len(colours), len(colours))
            # lp[f, k, :] = log p(colour | context with colour k); carry over ordered pairs
            carry = []
            for f in range(len(FRAMES)):
                for k1 in range(len(colours)):
                    for k2 in range(len(colours)):
                        if k1 != k2:
                            carry.append((lp[f, k1, k1] - lp[f, k2, k1]).item())
            prior = last_logprobs(model, tok, [pq.format(noun=n["noun"]) for _, pq in FRAMES], cid)
            pr = prior.exp()
            ent = (-(pr * prior).sum(1)).mean().item()
            pt = ""
            if n["colour"] in colours:
                k = colours.index(n["colour"])
                others = [x for x in range(len(colours)) if x != k]
                pt = (prior[:, k] - prior[:, others].mean(1)).mean().item()
            out_rows.append({"model": a.model, "revision": a.revision, "noun": n["noun"], "group": n["group"],
                             "colour": n["colour"], "pile_count": n["pile_count"], "pair": n["pair"],
                             "kind": n.get("kind", ""), "count_type": n.get("count_type", ""),
                             "row_norm": E[nid[j]].norm().item(),
                             "condition": cond, "carry": sum(carry) / len(carry), "prior_entropy": ent,
                             "prior_typical": pt})
        print(f"{a.model} {a.revision} {cond} done", flush=True)
    with torch.no_grad():
        E.copy_(orig)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(out_rows)


if __name__ == "__main__":
    main()
