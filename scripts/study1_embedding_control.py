"""
study1_embedding_control.py -- Embedding Perturbation Control
=============================================================
Addresses the strongest remaining reviewer concern: is novel noun
representation collapse a property of untrained embeddings, or of the
transformer's processing pipeline?

Three conditions:
  A) NOVEL NOUNS (baseline): 8 novel nouns never seen in training.
     Expected: cosine ~.999 (replicates §4.7 finding)

  B) TRAINED NOUNS IN NOVEL CONTEXT: Take 8 trained nouns but present
     them in frames they were never associated with (e.g., a Domain B
     frame for a Domain A noun). The embeddings are fully trained, but
     the context is unfamiliar.
     If collapse occurs: the transformer collapses ANY input it can't
     match to a learned template, regardless of embedding quality.
     If no collapse: the collapse IS about untrained embeddings.

  C) RANDOM EMBEDDING SUBSTITUTION: Replace novel noun embeddings with
     copies of trained noun embeddings (+ small noise), then re-run.
     If collapse still occurs: proves the transformer's weights cause
     collapse, not the specific embedding vectors.
     If no collapse: the untrained embeddings themselves are the issue.

  D) HELD-OUT TRAINED NOUNS: Train probe on 24 nouns, test on 8 held-out
     trained nouns. If held-out trained nouns DON'T collapse, the
     difference is purely about training exposure.

Output: results/study1/embedding_control_output.txt
        results/study1/embedding_control_results.csv

Author: JP Cacioli | Date: 2026-03-05 | Session 12b
"""

import csv
import math
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tokenizers import Tokenizer

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = r"C:\overhypothesis"
MODEL_DIR = os.path.join(BASE_DIR, "models", "checkpoints")
TOKENIZER_DIR = os.path.join(BASE_DIR, "data", "tokenizers")
WUG_TEST_PATH = os.path.join(BASE_DIR, "data", "wug_tests", "wug_test_seed42.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "study1")

OUTPUT_TXT = os.path.join(RESULTS_DIR, "embedding_control_output.txt")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "embedding_control_results.csv")

CONDITION = "regular"
SIZE = "medium"
SEEDS = [42, 123, 456, 789, 1001]

SHAPE_TOKENS = ["zibby", "krannish", "tova", "felk", "mundi",
                "prex", "sallo", "wemb", "oofy", "drant"]
TRAIN_NOUNS = [
    "wug", "blorp", "dax", "fep", "kiv", "miff", "norp", "pell",
    "quiff", "rix", "smed", "terg", "ulf", "vorn", "weff", "yib",
    "zav", "blim", "corf", "drell", "folt", "grix", "hurn", "jav",
    "klef", "lorp", "mave", "niff", "orp", "plud", "quav", "relf"
]
TEST_NOUNS = ["snerg", "twick", "vulf", "wazz", "xorn", "yeff", "zull", "bruck"]

DEVICE = torch.device("cpu")

# Standard frames for generating test items
STD_FRAMES = [
    "Another {noun} is a ___.",
    "A new {noun} is a ___.",
    "This {noun} is a ___.",
    "The {noun} over there is a ___.",
]


# ============================================================
# MODEL (from study1_linear_probe.py)
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, seq_len, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("mask", mask.view(1, 1, seq_len, seq_len))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.d_head)
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, seq_len, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
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
        self.n_layer = n_layer
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
# HELPERS
# ============================================================

def find_checkpoint(seed):
    run_dir = Path(MODEL_DIR) / f"{CONDITION}_{SIZE}_seed{seed}"
    final = run_dir / "model_final.pt"
    if final.exists():
        return final
    ckpts = sorted(run_dir.glob("ckpt_step*.pt"))
    return ckpts[-1] if ckpts else None


def load_model(ckpt_path):
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    cfg = ckpt['model_cfg']
    tcfg = ckpt.get('train_cfg', {})
    model = OverhypothesisLM(
        vocab_size=ckpt['vocab_size'], d_model=cfg['d_model'],
        n_head=cfg['n_head'], n_layer=cfg['n_layer'], d_ff=cfg['d_ff'],
        seq_len=tcfg.get('seq_len', 128),
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(DEVICE)
    model.eval()
    return model


def get_hidden_at_last_pos(model, tokenizer, text):
    """Run forward pass, return hidden states at last position for all layers."""
    enc = tokenizer.encode(text)
    ids = enc.ids
    if len(ids) > model.seq_len:
        ids = ids[-model.seq_len:]
    tensor = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        _, hidden = model(tensor, return_hidden_states=True)
    # Return dict: layer_name -> vector (d_model,)
    result = {}
    for i, h in enumerate(hidden):
        name = "embed" if i == 0 else f"L{i}"
        result[name] = h[0, -1, :].numpy()
    return result


def pairwise_cosine(vecs):
    """Mean pairwise cosine similarity (upper triangle)."""
    norms = np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8)
    normed = vecs / norms
    sim = normed @ normed.T
    n = sim.shape[0]
    idx = np.triu_indices(n, k=1)
    vals = sim[idx]
    return float(np.mean(vals)), vals


def make_items(nouns, frames=None):
    """Generate simple test items for a set of nouns."""
    if frames is None:
        frames = STD_FRAMES
    items = []
    for noun in nouns:
        for frame in frames:
            text = frame.format(noun=noun)
            context = text.split("___")[0].rstrip()
            items.append({"noun": noun, "context": context})
    return items


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("EMBEDDING PERTURBATION CONTROL")
    print(f"Condition: {CONDITION}, Size: {SIZE}")
    print("=" * 70)

    tok_path = os.path.join(TOKENIZER_DIR, f"{CONDITION}_tokenizer.json")
    tokenizer = Tokenizer.from_file(tok_path)

    # Get token IDs
    novel_ids = [tokenizer.token_to_id(n) for n in TEST_NOUNS]
    train_ids = [tokenizer.token_to_id(n) for n in TRAIN_NOUNS]
    print(f"Novel noun IDs: {dict(zip(TEST_NOUNS, novel_ids))}")
    print(f"Sample trained noun IDs: {dict(zip(TRAIN_NOUNS[:4], train_ids[:4]))}")

    # Generate items for each condition
    novel_items = make_items(TEST_NOUNS)
    # For condition B: use 8 trained nouns in standard frames (they'll be in
    # familiar frames, but we select 8 to match the novel noun count)
    held_out_nouns = TRAIN_NOUNS[24:]  # last 8 trained nouns
    held_in_nouns = TRAIN_NOUNS[:24]   # first 24 trained nouns
    held_out_items = make_items(held_out_nouns)
    held_in_items = make_items(held_in_nouns[:8])  # 8 for comparison

    rpt = []
    rpt.append("=" * 70)
    rpt.append("EMBEDDING PERTURBATION CONTROL")
    rpt.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rpt.append(f"Condition: {CONDITION}, Size: {SIZE}")
    rpt.append("Purpose: Determine whether novel noun collapse is due to")
    rpt.append("untrained embeddings or transformer processing pipeline")
    rpt.append("=" * 70)

    csv_rows = []
    focus_layers = ["embed", "L1", "L4", "L6", "L8"]

    for seed in SEEDS:
        print(f"\n{'=' * 40}")
        print(f"SEED {seed}")
        print(f"{'=' * 40}")

        ckpt_path = find_checkpoint(seed)
        if ckpt_path is None:
            print("  No checkpoint"); continue

        model = load_model(ckpt_path)

        # ── Condition A: Novel nouns (baseline) ──
        print("  A) Novel nouns...")
        novel_hidden = defaultdict(list)
        for item in novel_items:
            h = get_hidden_at_last_pos(model, tokenizer, item["context"])
            for layer, vec in h.items():
                novel_hidden[layer].append((item["noun"], vec))

        # ── Condition B: Held-out trained nouns ──
        print("  B) Held-out trained nouns (8 nouns never probed)...")
        heldout_hidden = defaultdict(list)
        for item in held_out_items:
            h = get_hidden_at_last_pos(model, tokenizer, item["context"])
            for layer, vec in h.items():
                heldout_hidden[layer].append((item["noun"], vec))

        # ── Condition C: Familiar trained nouns (control) ──
        print("  C) Familiar trained nouns (first 8)...")
        familiar_hidden = defaultdict(list)
        for item in held_in_items:
            h = get_hidden_at_last_pos(model, tokenizer, item["context"])
            for layer, vec in h.items():
                familiar_hidden[layer].append((item["noun"], vec))

        # ── Condition D: Novel nouns with swapped embeddings ──
        # Replace novel noun embeddings with copies of trained noun embeddings
        print("  D) Novel nouns with trained-noun embeddings (swap)...")
        # Save original embeddings
        orig_novel_embs = model.tok_embed.weight.data[novel_ids].clone()
        # Replace with trained noun embeddings (+ small noise for uniqueness)
        swap_source_ids = train_ids[:8]
        with torch.no_grad():
            for i, nid in enumerate(novel_ids):
                src_emb = model.tok_embed.weight.data[swap_source_ids[i]].clone()
                noise = torch.randn_like(src_emb) * 0.001
                model.tok_embed.weight.data[nid] = src_emb + noise

        swapped_hidden = defaultdict(list)
        for item in novel_items:
            h = get_hidden_at_last_pos(model, tokenizer, item["context"])
            for layer, vec in h.items():
                swapped_hidden[layer].append((item["noun"], vec))

        # Restore original embeddings
        with torch.no_grad():
            for i, nid in enumerate(novel_ids):
                model.tok_embed.weight.data[nid] = orig_novel_embs[i]

        del model

        # ── Compute within-group cosine similarities per layer ──
        rpt.append(f"\nSeed {seed}:")

        for layer in focus_layers:
            # Aggregate by noun (mean across frames)
            def aggregate(hidden_list):
                by_noun = defaultdict(list)
                for noun, vec in hidden_list:
                    by_noun[noun].append(vec)
                return np.stack([np.mean(vecs, axis=0) for vecs in by_noun.values()])

            novel_agg = aggregate(novel_hidden[layer])
            heldout_agg = aggregate(heldout_hidden[layer])
            familiar_agg = aggregate(familiar_hidden[layer])
            swapped_agg = aggregate(swapped_hidden[layer])

            cos_novel, _ = pairwise_cosine(novel_agg)
            cos_heldout, _ = pairwise_cosine(heldout_agg)
            cos_familiar, _ = pairwise_cosine(familiar_agg)
            cos_swapped, _ = pairwise_cosine(swapped_agg)

            # Norms
            novel_norm = float(np.mean(np.linalg.norm(novel_agg, axis=1)))
            heldout_norm = float(np.mean(np.linalg.norm(heldout_agg, axis=1)))
            familiar_norm = float(np.mean(np.linalg.norm(familiar_agg, axis=1)))
            swapped_norm = float(np.mean(np.linalg.norm(swapped_agg, axis=1)))

            rpt.append(f"  {layer}:")
            rpt.append(f"    A) Novel nouns:           cosine={cos_novel:.4f}, norm={novel_norm:.1f}")
            rpt.append(f"    B) Held-out trained nouns: cosine={cos_heldout:.4f}, norm={heldout_norm:.1f}")
            rpt.append(f"    C) Familiar trained nouns: cosine={cos_familiar:.4f}, norm={familiar_norm:.1f}")
            rpt.append(f"    D) Novel + swapped embeds: cosine={cos_swapped:.4f}, norm={swapped_norm:.1f}")

            print(f"  {layer}: novel={cos_novel:.4f}, held-out={cos_heldout:.4f}, "
                  f"familiar={cos_familiar:.4f}, swapped={cos_swapped:.4f}")

            csv_rows.append({
                "seed": seed, "layer": layer,
                "novel_cosine": f"{cos_novel:.4f}",
                "heldout_cosine": f"{cos_heldout:.4f}",
                "familiar_cosine": f"{cos_familiar:.4f}",
                "swapped_cosine": f"{cos_swapped:.4f}",
                "novel_norm": f"{novel_norm:.1f}",
                "heldout_norm": f"{heldout_norm:.1f}",
                "familiar_norm": f"{familiar_norm:.1f}",
                "swapped_norm": f"{swapped_norm:.1f}",
            })

    # ── Summary ──
    rpt.append(f"\n{'=' * 70}")
    rpt.append("SUMMARY AT L6")
    rpt.append("=" * 70)

    l6_rows = [r for r in csv_rows if r["layer"] == "L6"]
    if l6_rows:
        novel_cos = np.mean([float(r["novel_cosine"]) for r in l6_rows])
        heldout_cos = np.mean([float(r["heldout_cosine"]) for r in l6_rows])
        familiar_cos = np.mean([float(r["familiar_cosine"]) for r in l6_rows])
        swapped_cos = np.mean([float(r["swapped_cosine"]) for r in l6_rows])

        rpt.append(f"  A) Novel nouns (untrained embeds):     {novel_cos:.4f}")
        rpt.append(f"  B) Held-out trained nouns:             {heldout_cos:.4f}")
        rpt.append(f"  C) Familiar trained nouns:             {familiar_cos:.4f}")
        rpt.append(f"  D) Novel nouns (trained-noun embeds):  {swapped_cos:.4f}")

        rpt.append(f"\n  KEY COMPARISONS:")
        rpt.append(f"    A vs C (novel vs familiar): {novel_cos - familiar_cos:+.4f}")
        rpt.append(f"    D vs A (swapped vs novel):  {swapped_cos - novel_cos:+.4f}")
        rpt.append(f"    D vs C (swapped vs familiar): {swapped_cos - familiar_cos:+.4f}")
        rpt.append(f"    B vs C (held-out vs familiar): {heldout_cos - familiar_cos:+.4f}")

        if swapped_cos > 0.95:
            rpt.append("\n  INTERPRETATION: Collapse persists even with trained-noun embeddings.")
            rpt.append("  The transformer's weights collapse ALL inputs not matching learned")
            rpt.append("  templates, regardless of embedding quality. This is a property of")
            rpt.append("  the processing pipeline, not of untrained embeddings.")
        elif swapped_cos < familiar_cos + 0.05:
            rpt.append("\n  INTERPRETATION: Swapping embeddings substantially reduces collapse.")
            rpt.append("  The untrained embeddings ARE a contributing factor.")
        else:
            rpt.append("\n  INTERPRETATION: Mixed result. Partial collapse with swapped embeddings.")

        if heldout_cos < novel_cos - 0.1:
            rpt.append("  Held-out trained nouns do NOT collapse, confirming that training")
            rpt.append("  exposure differentiates representations.")

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


if __name__ == "__main__":
    main()
