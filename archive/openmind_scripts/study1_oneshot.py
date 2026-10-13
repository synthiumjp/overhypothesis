"""
study1_oneshot.py -- One-Shot In-Context Learning Test
======================================================
Tests whether prepending a single exemplar sentence for a novel noun
changes the model's feature prediction.

Design:
  - Zero-shot baseline: "A new zull is a ___." (no prior context)
  - One-shot: "A zull is a mundi temil glaven thing. A new zull is a ___."
  - Control: "A zull is a mundi temil glaven thing. A new wazz is a ___."
    (exemplar for zull, query for different novel noun)

If the model can do in-context exemplar learning:
  - One-shot accuracy >> zero-shot accuracy for the exemplar's shape
  - Control accuracy ~ zero-shot (no transfer to different noun)

If the model can do in-context OVERHYPOTHESIS learning:
  - Both one-shot AND control should improve (the exemplar teaches
    "shape is the stable dimension" which should generalise)

Most likely outcome for our tiny models: no in-context learning at all
(which is itself informative — it means the model can't even do rapid
exemplar learning from context, let alone rapid abstraction).

Output: results/study1/oneshot_results.csv
        results/study1/oneshot_output.txt

Author: JP Cacioli | Date: 2026-03-05 | Session 12b
"""

import csv
import math
import os
import random
import re
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
CORPUS_DIR = os.path.join(BASE_DIR, "data", "corpora")
TOKENIZER_DIR = os.path.join(BASE_DIR, "data", "tokenizers")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "study1")

OUTPUT_TXT = os.path.join(RESULTS_DIR, "oneshot_output.txt")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "oneshot_results.csv")

CONDITION = "regular"
SIZE = "medium"
SEEDS = [42, 123, 456, 789, 1001]

SHAPE_TOKENS = ["zibby", "krannish", "tova", "felk", "mundi",
                "prex", "sallo", "wemb", "oofy", "drant"]
COLOUR_TOKENS = ["lavo", "renu", "vassa", "temil", "dorvi",
                 "pulek", "sorba", "navri", "kelun", "morit"]
TEXTURE_TOKENS = ["shorfi", "glaven", "frimel", "tuskan", "merdo",
                  "pliven", "druska", "halven", "skorli", "tramel"]

TEST_NOUNS = ["snerg", "twick", "vulf", "wazz", "xorn", "yeff", "zull", "bruck"]

# Use GPU for inference (not extracting hidden states)
try:
    import torch_directml
    DEVICE = torch_directml.device()
except ImportError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def get_shape_logprobs(model, tokenizer, context_text):
    """Get log-prob of each shape token at the prediction position."""
    enc = tokenizer.encode(context_text)
    ids = enc.ids
    if len(ids) > model.seq_len:
        ids = ids[-model.seq_len:]
    tensor = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits = model(tensor)
    last_lp = torch.log_softmax(logits[0, -1, :], dim=-1).cpu()

    result = {}
    for tok in SHAPE_TOKENS:
        tid = tokenizer.token_to_id(tok)
        if tid is not None:
            result[tok] = last_lp[tid].item()
    return result


def rank_target(shape_lps, target_shape):
    """Rank of target shape among all shape tokens (1=best)."""
    target_lp = shape_lps.get(target_shape, float("-inf"))
    rank = 1 + sum(1 for s, lp in shape_lps.items()
                   if lp > target_lp and s != target_shape)
    return rank


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("ONE-SHOT IN-CONTEXT LEARNING TEST")
    print(f"Condition: {CONDITION}, Size: {SIZE}")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    tok_path = os.path.join(TOKENIZER_DIR, f"{CONDITION}_tokenizer.json")
    tokenizer = Tokenizer.from_file(tok_path)

    # Assign each novel noun a "target" shape and exemplar sentence
    rng = random.Random(42)
    novel_assignments = {}
    for i, noun in enumerate(TEST_NOUNS):
        shape = SHAPE_TOKENS[i % len(SHAPE_TOKENS)]
        colour = rng.choice(COLOUR_TOKENS)
        texture = rng.choice(TEXTURE_TOKENS)
        exemplar = f"A {noun} is a {shape} {colour} {texture} thing."
        novel_assignments[noun] = {
            "shape": shape,
            "colour": colour,
            "texture": texture,
            "exemplar": exemplar,
        }
    print("\nNovel noun assignments:")
    for noun, info in novel_assignments.items():
        print(f"  {noun} -> {info['shape']} (exemplar: '{info['exemplar']}')")

    rpt = []
    rpt.append("=" * 70)
    rpt.append("ONE-SHOT IN-CONTEXT LEARNING TEST")
    rpt.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rpt.append(f"Condition: {CONDITION}, Size: {SIZE}")
    rpt.append("Purpose: Test whether models can learn from a single in-context exemplar")
    rpt.append("=" * 70)

    csv_rows = []

    for seed in SEEDS:
        print(f"\n{'=' * 40}")
        print(f"SEED {seed}")
        print(f"{'=' * 40}")

        ckpt_path = find_checkpoint(seed)
        if ckpt_path is None:
            print("  No checkpoint"); continue
        model = load_model(ckpt_path)

        seed_results = {"zero_shot": [], "one_shot": [], "control": []}

        for noun in TEST_NOUNS:
            info = novel_assignments[noun]
            target_shape = info["shape"]
            exemplar = info["exemplar"]

            # Pick a different novel noun for control
            other_nouns = [n for n in TEST_NOUNS if n != noun]
            control_noun = rng.choice(other_nouns)

            query = f"A new {noun} is a"

            # Zero-shot: just the query
            zero_context = query
            zero_lps = get_shape_logprobs(model, tokenizer, zero_context)
            zero_rank = rank_target(zero_lps, target_shape)
            zero_top = max(zero_lps, key=zero_lps.get)

            # One-shot: exemplar + query
            one_context = f"{exemplar} {query}"
            one_lps = get_shape_logprobs(model, tokenizer, one_context)
            one_rank = rank_target(one_lps, target_shape)
            one_top = max(one_lps, key=one_lps.get)

            # Control: exemplar for THIS noun + query for DIFFERENT noun
            control_query = f"A new {control_noun} is a"
            control_context = f"{exemplar} {control_query}"
            ctrl_lps = get_shape_logprobs(model, tokenizer, control_context)
            # For control, target is the SAME shape (testing overhypothesis transfer)
            ctrl_rank = rank_target(ctrl_lps, target_shape)
            ctrl_top = max(ctrl_lps, key=ctrl_lps.get)

            seed_results["zero_shot"].append(zero_rank)
            seed_results["one_shot"].append(one_rank)
            seed_results["control"].append(ctrl_rank)

            csv_rows.append({
                "seed": seed, "noun": noun, "target_shape": target_shape,
                "zero_rank": zero_rank, "zero_top": zero_top,
                "one_rank": one_rank, "one_top": one_top,
                "control_noun": control_noun,
                "control_rank": ctrl_rank, "control_top": ctrl_top,
            })

        del model

        # Summary for this seed
        zr = np.array(seed_results["zero_shot"])
        or_ = np.array(seed_results["one_shot"])
        cr = np.array(seed_results["control"])

        rpt.append(f"\nSeed {seed}:")
        rpt.append(f"  Zero-shot:  mean rank={np.mean(zr):.2f}, rank-1={np.mean(zr==1)*100:.0f}%")
        rpt.append(f"  One-shot:   mean rank={np.mean(or_):.2f}, rank-1={np.mean(or_==1)*100:.0f}%")
        rpt.append(f"  Control:    mean rank={np.mean(cr):.2f}, rank-1={np.mean(cr==1)*100:.0f}%")
        rpt.append(f"  One-shot improvement: {np.mean(zr) - np.mean(or_):+.2f} ranks")

        print(f"  Zero-shot: rank={np.mean(zr):.2f}, One-shot: rank={np.mean(or_):.2f}, "
              f"Control: rank={np.mean(cr):.2f}")

    # Overall summary
    rpt.append(f"\n{'=' * 70}")
    rpt.append("OVERALL SUMMARY")
    rpt.append("=" * 70)

    all_zero = [float(r["zero_rank"]) for r in csv_rows]
    all_one = [float(r["one_rank"]) for r in csv_rows]
    all_ctrl = [float(r["control_rank"]) for r in csv_rows]

    rpt.append(f"  Mean zero-shot rank:  {np.mean(all_zero):.2f} (chance = 5.5)")
    rpt.append(f"  Mean one-shot rank:   {np.mean(all_one):.2f}")
    rpt.append(f"  Mean control rank:    {np.mean(all_ctrl):.2f}")
    rpt.append(f"  Zero-shot rank-1:     {np.mean(np.array(all_zero)==1)*100:.1f}%")
    rpt.append(f"  One-shot rank-1:      {np.mean(np.array(all_one)==1)*100:.1f}%")
    rpt.append(f"  Control rank-1:       {np.mean(np.array(all_ctrl)==1)*100:.1f}%")

    # Statistical test
    from scipy import stats
    t_zero_one, p_zero_one = stats.ttest_rel(all_zero, all_one)
    rpt.append(f"\n  Paired t (zero vs one-shot): t={t_zero_one:.3f}, p={p_zero_one:.4f}")

    if np.mean(all_one) < np.mean(all_zero) - 1.0:
        rpt.append("\n  INTERPRETATION: One-shot context improves ranking of the target shape.")
        rpt.append("  Models show some in-context exemplar learning.")
        if np.mean(all_ctrl) < np.mean(all_zero) - 0.5:
            rpt.append("  Control also improves -> suggests overhypothesis-like transfer!")
        else:
            rpt.append("  Control does NOT improve -> exemplar-specific, not generalised.")
    else:
        rpt.append("\n  INTERPRETATION: No evidence of in-context learning.")
        rpt.append("  One-shot exemplar does not change the model's predictions.")
        rpt.append("  These models cannot do rapid exemplar learning from context,")
        rpt.append("  consistent with their small scale (8 layers, 25.6M params).")

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
