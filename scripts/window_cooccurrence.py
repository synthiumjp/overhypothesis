#!/usr/bin/env python3
"""
window_cooccurrence.py  |  How often a training window of Study 1 shows a model an
earlier sentence about the same kind (Amendment 6).

measure   For every corpus file and window length W, the training stream is
          tokenised exactly as in train_model.py (the whole file in one call) and a
          window starts at every token offset, as the data loader allows. For each
          window, among the sentences that begin inside it, the share preceded in the
          same window by a sentence about the same kind. Output: the mean over all
          windows, per corpus file and W.

predict   Fits OI_colour (k = 1, same noun) ~ share + size, random intercept for
          corpus seed, on the six Regular orders trained with W = 128 (Study 1 and
          its extension), and predicts the mean index of the conditions trained with
          W = 512 from their measured share. Written and run before any W = 512
          model was trained; the output is part of Amendment 6.

Usage:
  python scripts/window_cooccurrence.py measure --out results/study1w/cooccurrence.csv
  python scripts/window_cooccurrence.py predict --cooc results/study1w/cooccurrence.csv \
      --runs results/study1c/summary_per_run.csv results/study1x/summary_per_run.csv \
      --out results/study1w/prediction.json
"""
import argparse
import importlib.util
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SEEDS = (42, 123, 456, 789, 1001)
# condition label used in the summaries -> corpus file pattern
CORPORA = {
    "regular": "data/corpora/regular_seed{s}.txt",
    "regular_bursty": "data/corpora_bursty/regular_seed{s}.txt",
    "regular_graded_b1p4": "data/corpora_x/graded_b1p4/regular_seed{s}.txt",
    "regular_graded_b3p6": "data/corpora_x/graded_b3p6/regular_seed{s}.txt",
    "regular_graded_b7": "data/corpora_x/graded_b7/regular_seed{s}.txt",
    "regular_graded_b12": "data/corpora_x/graded_b12/regular_seed{s}.txt",
    "regular_nostable_random": "data/corpora_x/nostable_random/regular_seed{s}.txt",
    "regular_nostable_grouped": "data/corpora_x/nostable_grouped/regular_seed{s}.txt",
    "regular_colourstable_random": "data/corpora_x/colourstable_random/regular_seed{s}.txt",
    "regular_colourstable_grouped": "data/corpora_x/colourstable_grouped/regular_seed{s}.txt",
}
SIX = ["regular", "regular_graded_b1p4", "regular_graded_b3p6", "regular_graded_b7",
       "regular_graded_b12", "regular_bursty"]
NEW = {"regular": "regular_w512", "regular_graded_b3p6": "regular_graded_b3p6_w512",
       "regular_nostable_random": "regular_nostable_random_w512"}


def kind_of_module():
    spec = importlib.util.spec_from_file_location("mb", "scripts/make_bursty_corpus.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.kind_of


def window_share(path, W, tok, kind_of):
    text = open(path, encoding="utf-8").read()
    enc = tok.encode(text)
    lines = text.split("\n")
    # character offset where each sentence begins -> first token at or after it
    starts_char, pos = [], 0
    for ln in lines:
        if ln.strip():
            starts_char.append(pos + (len(ln) - len(ln.lstrip())))
        pos += len(ln) + 1
    tok_start = np.array([o[0] for o in enc.offsets])
    sent_tok = np.searchsorted(tok_start, np.array(starts_char), side="left")
    kinds = [kind_of(ln) for ln in lines if ln.strip()]
    prev = np.full(len(kinds), -1)
    last = {}
    for j, k in enumerate(kinds):
        prev[j] = sent_tok[last[k]] if k in last else -1
        last[k] = j
    n_tok = len(enc.ids)
    offsets = np.arange(0, n_tok - W)          # data loader: idx in [0, len - seq_len)
    lo = np.searchsorted(sent_tok, offsets, side="left")
    hi = np.searchsorted(sent_tok, offsets + W, side="left")
    hit = (prev >= 0).astype(np.int64)
    # a sentence j in [lo, hi) counts if prev[j] >= window start; do it per window
    shares = np.empty(len(offsets))
    for i, (s, a, b) in enumerate(zip(offsets, lo, hi)):
        shares[i] = (prev[a:b] >= s).mean() if b > a else 0.0
    return float(shares.mean()), int(n_tok), float(np.mean(hi - lo))


def measure(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("data/tokenizers/regular_tokenizer.json")
    kind_of = kind_of_module()
    rows = []
    for cond, pat in CORPORA.items():
        for s in SEEDS:
            for W in (128, 512):
                share, n_tok, n_sent = window_share(pat.format(s=s), W, tok, kind_of)
                rows.append({"condition": cond, "seed": s, "W": W, "share": round(share, 5),
                             "tokens": n_tok, "sentences_per_window": round(n_sent, 2)})
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


def predict(a):
    import statsmodels.formula.api as smf
    c = pd.read_csv(a.cooc)
    d = pd.concat([pd.read_csv(p) for p in a.runs], ignore_index=True)
    d["match"] = d["match"].astype(str).isin(["True", "true", "1"])
    d = d[(d.source == "lm") & (d.k == 1) & d.match & d.condition.isin(SIX)]
    d = d.merge(c[c.W == 128][["condition", "seed", "share"]], on=["condition", "seed"])
    m = smf.mixedlm("OI_colour ~ share + C(size)", d, groups=d["seed"]).fit(reml=True)
    out = {"fit": {"n_runs": len(d), "intercept": m.params["Intercept"], "slope_share": m.params["share"],
                   "slope_se": m.bse["share"], "size_small": m.params["C(size)[T.small]"],
                   "size_tiny": m.params["C(size)[T.tiny]"], "seed_var": float(m.cov_re.iloc[0, 0]),
                   "resid_var": m.scale},
           "observed_W128_means": d.groupby("condition").OI_colour.mean().to_dict(),
           "share_W128": c[c.W == 128].groupby("condition").share.mean().to_dict(),
           "share_W512": c[c.W == 512].groupby("condition").share.mean().to_dict(),
           "predictions_W512": {}}
    fe = m.fe_params
    cov = m.cov_params().loc[fe.index, fe.index]
    for cond, label in NEW.items():
        if cond not in SIX:
            continue  # the fit covers Regular orders only; no prediction for NoStable
        sh = c[(c.W == 512) & (c.condition == cond)].set_index("seed").share
        X = []
        for size in ("tiny", "small", "medium"):
            for s in SEEDS:
                X.append([1.0, float(size == "small"), float(size == "tiny"), sh[s]])
        X = pd.DataFrame(X, columns=fe.index).mean()
        pred = float(X @ fe)
        se_fix = float(np.sqrt(X @ cov @ X))
        out["predictions_W512"][label] = {"mean_share": float(sh.mean()), "predicted_OI": pred,
                                          "se_fixed": se_fix}
    json.dump(out, open(a.out, "w"), indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("measure")
p.add_argument("--out", required=True)
p = sub.add_parser("predict")
p.add_argument("--cooc", required=True)
p.add_argument("--runs", nargs="+", required=True)
p.add_argument("--out", required=True)
a = ap.parse_args()
{"measure": measure, "predict": predict}[a.cmd](a)
