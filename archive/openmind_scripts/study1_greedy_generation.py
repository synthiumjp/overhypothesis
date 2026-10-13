"""
study1_greedy_generation.py -- Exploratory: Few-shot Greedy Generation
======================================================================

Pre-registration spec (PROJECT_PLAN_v1.5, Section 3.2):
    "Exploratory: Few-shot generation (Medium model only). Greedy decode
     from one-shot context; check whether the top generated feature token
     matches the correct shape token."

Scope: Medium models x Regular condition x 5 seeds x SO + FO items.
For each wug test item, tokenize context up to the critical position,
get model logits, check if argmax(logits) = correct shape token.

Usage:
    python study1_greedy_generation.py
        --models_dir C:\\overhypothesis\\models\\checkpoints
        --tokenizer_path C:\\overhypothesis\\data\\tokenizers\\regular_tokenizer.json
        --wug_test_path C:\\overhypothesis\\data\\wug_tests\\wug_test_seed42.csv
        --results_dir C:\\overhypothesis\\results\\study1
        --master_csv C:\\overhypothesis\\results\\study1\\master_item_level.csv

Author: JP Cacioli
Date: 2026-03-04
"""

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import torch_directml
    DML_AVAILABLE = torch_directml.is_available()
except ImportError:
    DML_AVAILABLE = False

from tokenizers import Tokenizer


# ============================================================
# Model architecture (must match train_model.py exactly)
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, seq_len, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
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
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


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

    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_embed(input_ids) + self.pos_embed(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


# ============================================================
# Constants
# ============================================================

SHAPE_TOKENS = [
    'zibby', 'krannish', 'tova', 'felk', 'mundi',
    'prex', 'sallo', 'wemb', 'oofy', 'drant'
]

COLOUR_TOKENS = [
    'lavo', 'renu', 'vassa', 'temil', 'dorvi',
    'pulek', 'sorba', 'navri', 'kelun', 'morit'
]

TEXTURE_TOKENS = [
    'shorfi', 'glaven', 'frimel', 'tuskan', 'merdo',
    'pliven', 'druska', 'halven', 'skorli', 'tramel'
]

ALL_FEATURE_TOKENS = SHAPE_TOKENS + COLOUR_TOKENS + TEXTURE_TOKENS

SEEDS = [42, 123, 456, 789, 1001]
TARGET_ITEM_TYPES = ['second_order', 'first_order']

# Medium model config (from train_model.py)
MEDIUM_CFG = dict(n_layer=8, n_head=8, d_model=512, d_ff=2048)
MEDIUM_SEQ_LEN = 128


# ============================================================
# Load model
# ============================================================

def load_model(ckpt_path, device):
    """Load a trained OverhypothesisLM from checkpoint."""
    checkpoint = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)

    vocab_size = checkpoint['vocab_size']
    cfg = checkpoint['model_cfg']

    # Get seq_len from checkpoint args or default
    train_cfg = checkpoint.get('train_cfg', {})
    seq_len = train_cfg.get('seq_len', MEDIUM_SEQ_LEN)

    model = OverhypothesisLM(
        vocab_size=vocab_size,
        d_model=cfg['d_model'],
        n_head=cfg['n_head'],
        n_layer=cfg['n_layer'],
        d_ff=cfg['d_ff'],
        seq_len=seq_len,
        dropout=0.0,  # No dropout for inference
    )

    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    return model


# ============================================================
# Load wug test items
# ============================================================

def load_wug_items(wug_test_path, target_types):
    """Load wug test items from CSV."""
    items = []
    with open(wug_test_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('item_type', '') in target_types:
                items.append(row)
    return items


# ============================================================
# Load forced-choice accuracy for comparison
# ============================================================

def load_forced_choice_accuracy(master_csv_path, seeds):
    """Load Medium/Regular FC accuracy per seed from master CSV."""
    data = defaultdict(lambda: defaultdict(list))

    with open(master_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get('condition', '') == 'regular' and
                    row.get('size', '') == 'medium'):
                try:
                    seed = int(row.get('seed', ''))
                except (ValueError, TypeError):
                    continue
                if seed not in seeds:
                    continue
                item_type = row.get('item_type', '')
                if item_type in TARGET_ITEM_TYPES:
                    try:
                        correct = int(row.get('correct_binary',
                                              row.get('correct_01', 0)))
                    except (ValueError, TypeError):
                        continue
                    data[seed][item_type].append(correct)

    result = {}
    for seed in data:
        result[seed] = {}
        for item_type in data[seed]:
            result[seed][item_type] = np.mean(data[seed][item_type])
    return result


# ============================================================
# Greedy generation evaluation
# ============================================================

def evaluate_greedy(model, tokenizer, items, device, shape_token_ids, feature_token_ids):
    """For each item, get model's greedy prediction at the critical position.

    Returns list of dicts with per-item results.
    """
    results = []

    for item in items:
        # Get the context text (everything before the critical position)
        # The wug test CSV should have a 'context' or 'target_text' field
        context_text = item.get('context', '')
        if not context_text:
            # Extract context from frame field: "Another pell is a ___."
            frame = item.get('frame', '')
            if '___' in frame:
                context_text = frame.split('___')[0].rstrip()
            elif frame:
                context_text = frame
            else:
                continue

        if not context_text:
            continue

        # Tokenize context
        encoding = tokenizer.encode(context_text)
        input_ids = encoding.ids

        if len(input_ids) == 0:
            continue

        # Truncate if needed
        max_len = model.seq_len
        if len(input_ids) > max_len:
            input_ids = input_ids[-max_len:]

        # Get model logits at last position
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_tensor)

        # Logits at the last position = prediction for next token
        last_logits = logits[0, -1, :].cpu().numpy()

        # Greedy prediction: argmax over full vocabulary
        greedy_id = int(np.argmax(last_logits))

        # Check if greedy prediction matches correct shape
        correct_shape = item.get('target', '')
        correct_id = tokenizer.encode(correct_shape).ids
        correct_shape_id = correct_id[0] if correct_id else -1

        greedy_matches_target = (greedy_id == correct_shape_id)

        # Check if greedy is ANY shape token
        greedy_is_shape = (greedy_id in shape_token_ids)

        # Check if greedy is ANY feature token
        greedy_is_feature = (greedy_id in feature_token_ids)

        # Get greedy token string
        greedy_token = tokenizer.decode([greedy_id]).strip()

        # Rank of correct token among all feature tokens
        feature_logits = {tid: last_logits[tid] for tid in feature_token_ids}
        sorted_features = sorted(feature_logits.items(), key=lambda x: -x[1])
        rank = None
        for r, (tid, _) in enumerate(sorted_features, 1):
            if tid == correct_shape_id:
                rank = r
                break

        results.append({
            'item_id': item.get('item_id', ''),
            'item_type': item.get('item_type', ''),
            'noun': item.get('noun', ''),
            'target': correct_shape,
            'target_id': correct_shape_id,
            'greedy_token': greedy_token,
            'greedy_id': greedy_id,
            'greedy_matches_target': greedy_matches_target,
            'greedy_is_shape': greedy_is_shape,
            'greedy_is_feature': greedy_is_feature,
            'rank_among_features': rank,
        })

    return results


# ============================================================
# Find checkpoint
# ============================================================

def find_checkpoint(models_dir, seed):
    """Find Medium/Regular model checkpoint for a given seed."""
    run_name = f"regular_medium_seed{seed}"
    run_dir = Path(models_dir) / run_name

    if not run_dir.exists():
        return None

    # Try model_final.pt first, then highest-numbered checkpoint
    for name in ['model_final.pt', 'checkpoint_final.pt', 'final_model.pt']:
        path = run_dir / name
        if path.exists():
            return path

    # Fall back to highest checkpoint
    ckpts = sorted(run_dir.glob('ckpt_step*.pt'))
    if ckpts:
        return ckpts[-1]

    # Try any .pt file
    pts = sorted(run_dir.glob('*.pt'))
    if pts:
        return pts[-1]

    return None


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Greedy Generation Evaluation (Pre-Registered Exploratory)')
    parser.add_argument('--models_dir', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--wug_test_path', type=str, required=True)
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--master_csv', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Device
    if DML_AVAILABLE:
        device = torch_directml.device()
        print("Using DirectML device")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA device")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    print(f"Device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer_path}")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    # Get feature token IDs
    shape_token_ids = set()
    feature_token_ids = set()
    for token in SHAPE_TOKENS:
        ids = tokenizer.encode(token).ids
        if ids:
            shape_token_ids.add(ids[0])
            feature_token_ids.add(ids[0])
    for token in COLOUR_TOKENS + TEXTURE_TOKENS:
        ids = tokenizer.encode(token).ids
        if ids:
            feature_token_ids.add(ids[0])

    print(f"  Feature tokens in vocab: {len(feature_token_ids)}/30")
    print(f"  Shape tokens in vocab: {len(shape_token_ids)}/10")

    # Load wug test items
    print(f"Loading wug test items: {args.wug_test_path}")
    items = load_wug_items(args.wug_test_path, TARGET_ITEM_TYPES)
    type_counts = defaultdict(int)
    for item in items:
        type_counts[item['item_type']] += 1
    print(f"  Loaded {len(items)} items across types: {sorted(type_counts.keys())}")
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c} items")

    # Load forced-choice accuracy for comparison
    print(f"Loading forced-choice accuracy from: {args.master_csv}")
    fc_acc = load_forced_choice_accuracy(args.master_csv, SEEDS)
    for seed in sorted(fc_acc.keys()):
        for item_type in sorted(fc_acc[seed].keys()):
            print(f"  Seed {seed} / {item_type}: FC acc = {fc_acc[seed][item_type]*100:.1f}%")

    # Run greedy evaluation per seed
    lines = []
    lines.append("=" * 70)
    lines.append("GREEDY GENERATION -- Exploratory Pre-Registered Metric")
    lines.append("Medium model, Regular condition, 5 seeds")
    lines.append("=" * 70)
    lines.append("")

    all_csv_rows = []

    for seed in SEEDS:
        lines.append(f"{'=' * 50}")
        lines.append(f"SEED {seed}")
        lines.append(f"{'=' * 50}")

        ckpt_path = find_checkpoint(args.models_dir, seed)
        if ckpt_path is None:
            lines.append(f"  WARNING: No checkpoint found for seed {seed}")
            lines.append("")
            continue

        print(f"\n{'=' * 50}")
        print(f"SEED {seed}")
        print(f"{'=' * 50}")
        print(f"  Loading model: {ckpt_path}")
        lines.append(f"  Checkpoint: {ckpt_path}")

        model = load_model(ckpt_path, device)

        # Run greedy evaluation
        results = evaluate_greedy(
            model, tokenizer, items, device,
            shape_token_ids, feature_token_ids)

        # Separate by item type
        for item_type in TARGET_ITEM_TYPES:
            type_results = [r for r in results if r['item_type'] == item_type]
            if not type_results:
                continue

            n = len(type_results)
            n_match = sum(1 for r in type_results if r['greedy_matches_target'])
            n_shape = sum(1 for r in type_results if r['greedy_is_shape'])
            n_feature = sum(1 for r in type_results if r['greedy_is_feature'])
            ranks = [r['rank_among_features'] for r in type_results if r['rank_among_features'] is not None]
            mean_rank = np.mean(ranks) if ranks else float('nan')
            pct_rank1 = sum(1 for r in ranks if r == 1) / len(ranks) * 100 if ranks else 0

            # FC comparison
            fc = fc_acc.get(seed, {}).get(item_type, float('nan'))

            lines.append(f"  {item_type} ({n} items):")
            lines.append(f"    Greedy top-1 match: {n_match}/{n} = {n_match/n*100:.1f}%")
            lines.append(f"    Greedy is shape: {n_shape}/{n} = {n_shape/n*100:.1f}%")
            lines.append(f"    Greedy is feature: {n_feature}/{n} = {n_feature/n*100:.1f}%")
            lines.append(f"    Mean rank of target among 30 features: {mean_rank:.1f}")
            lines.append(f"    Target ranked 1st: {pct_rank1:.1f}%")
            lines.append(f"    FC accuracy (comparison): {fc*100:.1f}%")

            for r in type_results:
                r['seed'] = seed
                all_csv_rows.append(r)

        # Free model memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        lines.append("")

    # Aggregate summary
    lines.append("=" * 70)
    lines.append("AGGREGATE SUMMARY (across 5 seeds)")
    lines.append("=" * 70)

    for item_type in TARGET_ITEM_TYPES:
        type_results = [r for r in all_csv_rows if r['item_type'] == item_type]
        if not type_results:
            continue

        # Per-seed accuracies
        seed_accs = {}
        for seed in SEEDS:
            seed_results = [r for r in type_results if r['seed'] == seed]
            if seed_results:
                seed_accs[seed] = np.mean([r['greedy_matches_target'] for r in seed_results])

        if seed_accs:
            vals = list(seed_accs.values())
            mean = np.mean(vals)
            se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            ci95 = 1.96 * se

            lines.append(f"\n  {item_type}:")
            lines.append(f"    Mean greedy top-1 accuracy: {mean*100:.1f}% +/- {ci95*100:.1f}%")
            lines.append(f"    Per-seed: {', '.join(f's{s}={a*100:.1f}%' for s, a in sorted(seed_accs.items()))}")
            lines.append(f"    Chance (any shape): 10.0% (1/10 shape tokens)")
            lines.append(f"    Chance (any feature): 3.3% (1/30 feature tokens)")

    lines.append("")
    lines.append("INTERPRETATION:")
    lines.append("  Greedy generation is a higher bar than forced-choice:")
    lines.append("  the model must rank the correct token as the single most")
    lines.append("  probable in the ENTIRE vocabulary (not just above one foil).")
    lines.append("  FO greedy may show signal (especially seed 42 outlier).")
    lines.append("  SO greedy at chance (~10%) confirms null H1.")
    lines.append("")

    # Save
    report_text = "\n".join(lines)
    report_path = os.path.join(args.results_dir, 'greedy_generation_output.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print("\n" + report_text)

    if all_csv_rows:
        csv_path = os.path.join(args.results_dir, 'greedy_generation_results.csv')
        fieldnames = list(all_csv_rows[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_csv_rows)
        print(f"\nCSV saved: {csv_path} ({len(all_csv_rows)} rows)")

    print(f"Report saved: {report_path}")


if __name__ == '__main__':
    main()
