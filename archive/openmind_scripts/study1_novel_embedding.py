"""
study1_novel_embedding.py -- Novel Noun Embedding Cosine Similarity Analysis
============================================================================
Purpose: Test whether novel noun representations at the critical prediction
position are structured (clustered near trained nouns) or effectively random.

Paper location: New S4.7, plus key paragraph in S5.2
Output: results/study1/novel_noun_embedding_results.csv
        results/study1/novel_noun_embedding_output.txt
        results/study1/novel_noun_embedding_raw.npz (for figure B3)

Author: JP Cacioli | Date: 2026-03-05 | Session: 11 (V9 revision)
"""

import csv
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from tokenizers import Tokenizer

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# CONFIG
# ============================================================
BASE_DIR = r"C:\overhypothesis"
MODEL_DIR = os.path.join(BASE_DIR, "models", "checkpoints")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "study1")

WUG_TEST_PATH = os.path.join(DATA_DIR, "wug_tests", "wug_test_seed42.csv")
TOKENIZER_DIR = os.path.join(DATA_DIR, "tokenizers")

OUTPUT_CSV = os.path.join(RESULTS_DIR, "novel_noun_embedding_results.csv")
OUTPUT_TXT = os.path.join(RESULTS_DIR, "novel_noun_embedding_output.txt")
OUTPUT_RAW = os.path.join(RESULTS_DIR, "novel_noun_embedding_raw.npz")

CONDITION = "regular"
SIZE = "medium"
SEEDS = [42, 123, 456, 789, 1001]

MEDIUM_SEQ_LEN = 128
DEVICE = torch.device("cpu")


# ============================================================
# MODEL ARCHITECTURE (exact copy from study1_linear_probe.py)
# Key differences from current train_model.py on disk:
#   - self.ff (not self.mlp)
#   - bias=False on qkv, proj, lm_head
#   - attention uses .split() not .reshape()
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, seq_len, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("mask", mask.view(1, 1, seq_len, seq_len))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj_drop(self.proj(y))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, seq_len, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class OverhypothesisLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, d_ff,
                 seq_len=128, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, seq_len, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight

    def forward(self, input_ids, return_hidden_states=False):
        B, T = input_ids.shape
        assert T <= self.seq_len, f"Sequence length {T} exceeds max {self.seq_len}"
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_embed(input_ids) + self.pos_embed(pos))
        hidden = []
        if return_hidden_states:
            hidden.append(x.detach().cpu())
        for block in self.blocks:
            x = block(x)
            if return_hidden_states:
                hidden.append(x.detach().cpu())
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if return_hidden_states:
            return logits, hidden
        return logits


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def find_checkpoint(models_dir, condition, size, seed):
    run_dir = Path(models_dir) / f"{condition}_{size}_seed{seed}"
    if not run_dir.exists():
        return None
    final = run_dir / "model_final.pt"
    if final.exists():
        return final
    ckpts = sorted(run_dir.glob("ckpt_step*.pt"))
    return ckpts[-1] if ckpts else None


def load_model(ckpt_path):
    checkpoint = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    vocab_size = checkpoint['vocab_size']
    cfg = checkpoint['model_cfg']
    train_cfg = checkpoint.get('train_cfg', {})
    seq_len = train_cfg.get('seq_len', MEDIUM_SEQ_LEN)

    model = OverhypothesisLM(
        vocab_size=vocab_size,
        d_model=cfg['d_model'],
        n_head=cfg['n_head'],
        n_layer=cfg['n_layer'],
        d_ff=cfg['d_ff'],
        seq_len=seq_len,
    )
    model.load_state_dict(checkpoint['model_state'])
    model.to(DEVICE)
    model.eval()

    print(f"  Loaded {ckpt_path.name}: vocab={vocab_size}, d={cfg['d_model']}, "
          f"L={cfg['n_layer']}, H={cfg['n_head']}, ff={cfg['d_ff']}, seq={seq_len}")
    return model


# ============================================================
# HIDDEN STATE EXTRACTION
# ============================================================

def extract_hidden_states(model, tokenizer, items):
    layer_states = defaultdict(list)
    with torch.no_grad():
        for item in items:
            encoding = tokenizer.encode(item["context"])
            input_ids = encoding.ids
            if len(input_ids) > model.seq_len:
                input_ids = input_ids[-model.seq_len:]
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
            _, hidden = model(input_tensor, return_hidden_states=True)
            for idx, h in enumerate(hidden):
                name = "embed" if idx == 0 else f"L{idx}"
                layer_states[name].append(h[0, -1, :].numpy())
    return {name: np.stack(vecs) for name, vecs in layer_states.items()}


# ============================================================
# WUG TEST LOADING
# ============================================================

def load_wug_items(wug_path):
    fo_items, so_items = [], []
    with open(wug_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        print(f"  Columns: {reader.fieldnames}")
        for row in reader:
            item_type = row.get("item_type", "").strip()
            context = None
            if "frame" in row and "___" in row.get("frame", ""):
                context = row["frame"].split("___")[0].strip()
            elif "context" in row and row["context"].strip():
                context = row["context"].strip()
            if not context:
                continue
            item = {"item_type": item_type, "context": context,
                    "noun": row.get("noun", ""), "target": row.get("target", "")}
            if item_type == "first_order":
                fo_items.append(item)
            elif item_type == "second_order":
                so_items.append(item)

    fo_nouns = sorted(set(i["noun"] for i in fo_items))
    so_nouns = sorted(set(i["noun"] for i in so_items))
    print(f"  {len(fo_items)} FO ({len(fo_nouns)} nouns), "
          f"{len(so_items)} SO ({len(so_nouns)} nouns)")
    return fo_items, so_items, fo_nouns, so_nouns


# ============================================================
# COSINE SIMILARITY
# ============================================================

def pairwise_cosine(X):
    norms = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    normed = X / norms
    sim = normed @ normed.T
    idx = np.triu_indices(sim.shape[0], k=1)
    return sim[idx], sim

def cross_cosine(A, B):
    nA = np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-8)
    nB = np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-8)
    sim = (A / nA) @ (B / nB).T
    return sim.flatten(), sim

def aggregate_by_noun(items, states, nouns):
    d = defaultdict(list)
    for item, h in zip(items, states):
        d[item["noun"]].append(h)
    result = [np.mean(d[n], axis=0) for n in nouns if n in d]
    return np.stack(result) if result else np.zeros((0, states.shape[1]))


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("NOVEL NOUN EMBEDDING COSINE SIMILARITY ANALYSIS")
    print(f"Condition: {CONDITION}, Size: {SIZE}, Seeds: {SEEDS}")
    print(f"Models: {MODEL_DIR}")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    print("\n--- Tokenizer ---")
    tok_path = os.path.join(TOKENIZER_DIR, f"{CONDITION}_tokenizer.json")
    tokenizer = Tokenizer.from_file(tok_path)
    print(f"  Vocab: {tokenizer.get_vocab_size()}")

    print("\n--- Wug test ---")
    fo_items, so_items, fo_nouns, so_nouns = load_wug_items(WUG_TEST_PATH)

    print("\n--- Tokenization check ---")
    for label, items in [("FO", fo_items[:2]), ("SO", so_items[:2])]:
        for item in items:
            enc = tokenizer.encode(item["context"])
            print(f"  {label} [{item['noun']}]: '{item['context'][:40]}...' -> {len(enc.ids)} tok")

    all_seed_results = {}

    for seed in SEEDS:
        print(f"\n{'=' * 40}")
        print(f"SEED {seed}")
        print(f"{'=' * 40}")

        ckpt = find_checkpoint(MODEL_DIR, CONDITION, SIZE, seed)
        if ckpt is None:
            print(f"  WARNING: No checkpoint found")
            continue

        model = load_model(ckpt)
        all_items = fo_items + so_items
        print(f"  Extracting hidden states for {len(all_items)} items...")
        layer_states = extract_hidden_states(model, tokenizer, all_items)
        del model

        n_fo = len(fo_items)
        results_by_layer = {}

        for layer_name in sorted(layer_states.keys(),
                                  key=lambda x: -1 if x == "embed" else int(x[1:])):
            S = layer_states[layer_name]
            fo_s, so_s = S[:n_fo], S[n_fo:]

            fo_noun_s = aggregate_by_noun(fo_items, fo_s, fo_nouns)
            so_noun_s = aggregate_by_noun(so_items, so_s, so_nouns)

            wt_vals, _ = pairwise_cosine(fo_noun_s)
            wn_vals, _ = pairwise_cosine(so_noun_s)
            cr_vals, _ = cross_cosine(fo_noun_s, so_noun_s)

            _, cross_so_fo = cross_cosine(so_noun_s, fo_noun_s)
            nearest = cross_so_fo.max(axis=1) if cross_so_fo.size > 0 else np.array([])

            fo_norms = np.linalg.norm(fo_noun_s, axis=1)
            so_norms = np.linalg.norm(so_noun_s, axis=1)

            results_by_layer[layer_name] = {
                "wt_mean": float(np.mean(wt_vals)), "wt_sd": float(np.std(wt_vals)),
                "wn_mean": float(np.mean(wn_vals)), "wn_sd": float(np.std(wn_vals)),
                "cr_mean": float(np.mean(cr_vals)), "cr_sd": float(np.std(cr_vals)),
                "nearest_mean": float(np.mean(nearest)) if nearest.size else float('nan'),
                "nearest_sd": float(np.std(nearest)) if nearest.size else float('nan'),
                "fo_norm": float(np.mean(fo_norms)), "so_norm": float(np.mean(so_norms)),
                "fo_norm_sd": float(np.std(fo_norms)), "so_norm_sd": float(np.std(so_norms)),
                "wt_vals": wt_vals, "wn_vals": wn_vals, "cr_vals": cr_vals,
            }

            print(f"  {layer_name}: WT={np.mean(wt_vals):.4f}  CR={np.mean(cr_vals):.4f}  "
                  f"WN={np.mean(wn_vals):.4f}  FO-norm={np.mean(fo_norms):.1f}  "
                  f"SO-norm={np.mean(so_norms):.1f}")

        all_seed_results[seed] = results_by_layer

    if not all_seed_results:
        print("\nERROR: No seeds produced results.")
        return

    # ── Aggregate ──
    print(f"\n{'=' * 70}")
    print("AGGREGATE ACROSS SEEDS")
    print(f"{'=' * 70}")

    first = next(iter(all_seed_results.values()))
    all_layers = sorted(first.keys(), key=lambda x: -1 if x == "embed" else int(x[1:]))

    rpt = []
    rpt.append("=" * 70)
    rpt.append("NOVEL NOUN EMBEDDING COSINE SIMILARITY ANALYSIS")
    rpt.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rpt.append(f"Condition: {CONDITION}, Size: {SIZE}")
    rpt.append(f"Seeds: {list(all_seed_results.keys())} ({len(all_seed_results)} ok)")
    rpt.append(f"Trained nouns: {len(fo_nouns)}, Novel nouns: {len(so_nouns)}")
    rpt.append(f"FO items: {len(fo_items)}, SO items: {len(so_items)}")
    rpt.append("NOTE: Exploratory (not pre-registered).")
    rpt.append("=" * 70)

    csv_rows = []

    for layer in all_layers:
        rpt.append(f"\n--- {layer} ---")
        m = defaultdict(list)
        for seed, sr in all_seed_results.items():
            if layer not in sr:
                continue
            r = sr[layer]
            m["wt"].append(r["wt_mean"]); m["wn"].append(r["wn_mean"])
            m["cr"].append(r["cr_mean"]); m["nt"].append(r["nearest_mean"])
            m["fn"].append(r["fo_norm"]); m["sn"].append(r["so_norm"])

        n = len(m["wt"])
        if n == 0:
            rpt.append("  No data."); continue

        wt, wn, cr = np.array(m["wt"]), np.array(m["wn"]), np.array(m["cr"])
        nt, fn, sn = np.array(m["nt"]), np.array(m["fn"]), np.array(m["sn"])

        rpt.append(f"  Within-trained:   M={np.mean(wt):.4f} (SD={np.std(wt,ddof=1):.4f})")
        rpt.append(f"  Cross (T<->N):    M={np.mean(cr):.4f} (SD={np.std(cr,ddof=1):.4f})")
        rpt.append(f"  Within-novel:     M={np.mean(wn):.4f} (SD={np.std(wn,ddof=1):.4f})")
        rpt.append(f"  Nearest-trained:  M={np.mean(nt):.4f} (SD={np.std(nt,ddof=1):.4f})")
        rpt.append(f"  FO norm: {np.mean(fn):.1f} (SD={np.std(fn,ddof=1):.1f}), "
                   f"SO norm: {np.mean(sn):.1f} (SD={np.std(sn,ddof=1):.1f})")

        if n >= 3:
            diff = wt - cr
            t_val, p_val = stats.ttest_rel(wt, cr)
            d_val = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else np.inf
            rpt.append(f"  WT vs CR: t({n-1})={t_val:.3f}, p={p_val:.4f}, d={d_val:.3f}")
            rpt.append(f"    diff = {np.mean(diff):.4f} (trained {'>' if np.mean(diff)>0 else '<'} novel)")

            t2, p2 = stats.ttest_rel(wn, cr)
            rpt.append(f"  WN vs CR: t({n-1})={t2:.3f}, p={p2:.4f}")

            t3, p3 = stats.ttest_rel(fn, sn)
            rpt.append(f"  FO vs SO norm: t({n-1})={t3:.3f}, p={p3:.4f}, diff={np.mean(fn-sn):.2f}")

        for seed in all_seed_results:
            if layer in all_seed_results[seed]:
                r = all_seed_results[seed][layer]
                csv_rows.append({
                    "seed": seed, "layer": layer,
                    "condition": CONDITION, "size": SIZE,
                    "within_trained": f"{r['wt_mean']:.6f}",
                    "within_novel": f"{r['wn_mean']:.6f}",
                    "cross": f"{r['cr_mean']:.6f}",
                    "nearest_trained": f"{r['nearest_mean']:.6f}",
                    "fo_norm": f"{r['fo_norm']:.4f}",
                    "so_norm": f"{r['so_norm']:.4f}",
                    "n_fo_nouns": len(fo_nouns),
                    "n_so_nouns": len(so_nouns),
                })

    # Key findings
    rpt.append(f"\n{'=' * 70}")
    rpt.append("KEY FINDINGS")
    rpt.append("=" * 70)

    for fl in ["L6", "L8"]:
        if fl not in all_layers:
            continue
        wt_v = [all_seed_results[s][fl]["wt_mean"] for s in all_seed_results if fl in all_seed_results[s]]
        cr_v = [all_seed_results[s][fl]["cr_mean"] for s in all_seed_results if fl in all_seed_results[s]]
        wn_v = [all_seed_results[s][fl]["wn_mean"] for s in all_seed_results if fl in all_seed_results[s]]
        if not wt_v:
            continue
        gap = np.mean(wt_v) - np.mean(cr_v)
        rpt.append(f"\n  {fl}: WT={np.mean(wt_v):.4f}, CR={np.mean(cr_v):.4f}, "
                   f"WN={np.mean(wn_v):.4f}, gap={gap:.4f}")
        if gap > 0.05:
            rpt.append(f"  -> Novel nouns NOT clustered with trained nouns.")
        elif gap > 0.01:
            rpt.append(f"  -> Moderate gap.")
        else:
            rpt.append(f"  -> Similar structure for novel and trained nouns.")

    rpt.append(f"\n{'=' * 70}")
    rpt.append("PAPER TEXT TEMPLATE")
    rpt.append("=" * 70)
    rpt.append(
        "S4.7: 'We compared cosine similarity of hidden-state representations "
        "at the prediction position for trained vs novel nouns (Medium, Regular, "
        "5 seeds). At L6, within-trained similarity (M=X.XXX) exceeded "
        "trained-novel cross-similarity (M=X.XXX; paired t(4)=X.XX, p=.XXX, "
        "d=X.XX), indicating novel nouns do not cluster with trained categories.'"
    )
    rpt.append(
        "\nS5.2: 'Novel nouns produce representations dissimilar to trained "
        "nouns at the prediction position, so the model cannot cue "
        "domain-specific feature preferences from novel input. Even with 99% "
        "probe accuracy on trained nouns, generalisation fails because novel "
        "inputs do not activate learned category structure.'"
    )

    rpt.append(f"\n{'=' * 70}")
    rpt.append("END")
    rpt.append("=" * 70)

    full = "\n".join(rpt)
    print("\n" + full)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        f.write(full)
    print(f"\nReport: {OUTPUT_TXT}")

    if csv_rows:
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            w.writeheader()
            w.writerows(csv_rows)
        print(f"CSV: {OUTPUT_CSV}")

    raw = {}
    for seed, sr in all_seed_results.items():
        for ln, r in sr.items():
            p = f"seed{seed}_{ln}"
            raw[f"{p}_wt"] = r["wt_vals"]
            raw[f"{p}_wn"] = r["wn_vals"]
            raw[f"{p}_cr"] = r["cr_vals"]
    if raw:
        np.savez(OUTPUT_RAW, **raw)
        print(f"Raw: {OUTPUT_RAW}")


if __name__ == "__main__":
    main()
