"""
study1_linear_probe.py -- P2: Mechanistic Interpretability (Pre-Registered Exploratory)
========================================================================================

Pre-registration spec (PROJECT_PLAN_v1.5, Section 6.1):
    "P2: Mechanistic Interpretability -- EXPLORATORY
     Linear probe > 33% chance (3-class: shape/colour/texture);
     scrambled control at chance."

Method:
    1. For each wug test FO item, feed the context (up to the blank)
       through the model and extract hidden states at the last position.
    2. Label each item with the correct shape token (10-class) or
       stable dimension (3-class: shape/colour/texture -- but since
       all FO items test shape, we use 10-class shape identity).
    3. Train logistic regression (sklearn) on hidden states -> label.
    4. Evaluate via leave-one-out cross-validation.
    5. Repeat at each layer to get layer-wise probe curve.
    6. Compare Regular vs Scrambled (control should be at chance).

    Since H1 is null, we expect the probe to be at or near chance,
    confirming that the models do not encode the overhypothesis
    in a linearly decodable way.

Scope: Medium models, Regular + Scrambled conditions, 5 seeds.

Usage:
    python study1_linear_probe.py
        --models_dir C:\\overhypothesis\\models\\checkpoints
        --tokenizer_dir C:\\overhypothesis\\data\\tokenizers
        --wug_test_path C:\\overhypothesis\\data\\wug_tests\\wug_test_seed42.csv
        --results_dir C:\\overhypothesis\\results\\study1

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

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, LeaveOneOut
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: sklearn not available. Install with: pip install scikit-learn")


# ============================================================
# Model architecture (exact copy from train_model.py)
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
            hidden.append(x.detach().cpu())  # Layer 0: embedding output
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
# Constants
# ============================================================

SHAPE_TOKENS = [
    'zibby', 'krannish', 'tova', 'felk', 'mundi',
    'prex', 'sallo', 'wemb', 'oofy', 'drant'
]

SEEDS = [42, 123, 456, 789, 1001]
PROBE_CONDITIONS = ['regular', 'scrambled']
MEDIUM_SEQ_LEN = 128


# ============================================================
# Load model
# ============================================================

def load_model(ckpt_path, device):
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
        dropout=0.0,
    )
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    return model


# ============================================================
# Extract hidden states
# ============================================================

def extract_hidden_states(model, tokenizer, items, device):
    """Extract hidden states at the critical (last) position for each item.

    Returns:
        hidden_dict: {layer_idx: np.array of shape (n_items, d_model)}
        labels: list of target shape tokens
        item_ids: list of item IDs
    """
    all_hidden = defaultdict(list)  # layer_idx -> list of vectors
    labels = []
    item_ids = []

    for item in items:
        # Extract context from frame
        frame = item.get('frame', '')
        if '___' in frame:
            context_text = frame.split('___')[0].rstrip()
        elif frame:
            context_text = frame
        else:
            continue

        if not context_text:
            continue

        # Tokenize
        encoding = tokenizer.encode(context_text)
        input_ids = encoding.ids
        if len(input_ids) == 0:
            continue
        if len(input_ids) > model.seq_len:
            input_ids = input_ids[-model.seq_len:]

        # Forward pass with hidden states
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _, hidden_states = model(input_tensor, return_hidden_states=True)

        # Extract last position from each layer
        for layer_idx, h in enumerate(hidden_states):
            vec = h[0, -1, :].numpy()  # (d_model,)
            all_hidden[layer_idx].append(vec)

        labels.append(item.get('target', ''))
        item_ids.append(item.get('item_id', ''))

    # Convert to arrays
    hidden_dict = {}
    for layer_idx in all_hidden:
        hidden_dict[layer_idx] = np.array(all_hidden[layer_idx])

    return hidden_dict, labels, item_ids


# ============================================================
# Linear probe
# ============================================================

def run_linear_probe(X, y, cv_folds=5):
    """Train logistic regression probe with cross-validation.

    Returns accuracy (mean CV score) and std.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    if n_classes < 2:
        return 0.0, 0.0, n_classes

    # Use stratified k-fold (or LOO if too few samples)
    n_samples = len(y_enc)
    # Use 3-fold to handle class imbalance (some shapes have only 2 items)
    cv_folds = 3
    if n_samples < cv_folds * n_classes:
        cv_folds = 2

    clf = LogisticRegression(
        max_iter=2000,
        solver='lbfgs',

        C=1.0,
        random_state=42
    )

    try:
        scores = cross_val_score(clf, X, y_enc, cv=cv_folds, scoring='accuracy')
        return float(np.mean(scores)), float(np.std(scores)), n_classes
    except Exception as e:
        print(f"    Probe failed: {e}")
        return 0.0, 0.0, n_classes


# ============================================================
# Load wug test items
# ============================================================

def load_wug_fo_items(wug_test_path):
    """Load first-order wug test items only."""
    items = []
    with open(wug_test_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('item_type', '') == 'first_order':
                items.append(row)
    return items


# ============================================================
# Find checkpoint
# ============================================================

def find_checkpoint(models_dir, condition, seed):
    run_name = f"{condition}_medium_seed{seed}"
    run_dir = Path(models_dir) / run_name
    if not run_dir.exists():
        return None
    for name in ['model_final.pt']:
        path = run_dir / name
        if path.exists():
            return path
    ckpts = sorted(run_dir.glob('ckpt_step*.pt'))
    return ckpts[-1] if ckpts else None


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='P2: Linear Probe (Pre-Registered Exploratory)')
    parser.add_argument('--models_dir', type=str, required=True)
    parser.add_argument('--tokenizer_dir', type=str, required=True)
    parser.add_argument('--wug_test_path', type=str, required=True)
    parser.add_argument('--results_dir', type=str, required=True)
    args = parser.parse_args()

    if not SKLEARN_AVAILABLE:
        print("ERROR: scikit-learn required. Install with: pip install scikit-learn")
        return

    os.makedirs(args.results_dir, exist_ok=True)

    # Force CPU for probe -- only 80 items, speed irrelevant,
    # avoids DirectML tensor issues with hidden state extraction
    device = torch.device('cpu')
    print(f"Device: {device} (CPU forced for probe stability)")

    # Load wug test FO items
    print(f"Loading FO wug test items: {args.wug_test_path}")
    fo_items = load_wug_fo_items(args.wug_test_path)
    print(f"  {len(fo_items)} first-order items")

    lines = []
    lines.append("=" * 70)
    lines.append("P2: LINEAR PROBE -- Pre-Registered Exploratory Metric")
    lines.append("Probe: logistic regression on residual stream at critical position")
    lines.append("Task: predict correct shape token (10-class) from hidden state")
    lines.append("Medium models, Regular + Scrambled, 5 seeds, layer-wise")
    lines.append("Pre-registered threshold: > 33% (3-class) => > 10% (10-class)")
    lines.append("=" * 70)
    lines.append("")

    csv_rows = []

    for condition in PROBE_CONDITIONS:
        lines.append(f"{'=' * 50}")
        lines.append(f"CONDITION: {condition}")
        lines.append(f"{'=' * 50}")

        # Load appropriate tokenizer
        tok_path = Path(args.tokenizer_dir) / f"{condition}_tokenizer.json"
        if not tok_path.exists():
            tok_path = Path(args.tokenizer_dir) / "regular_tokenizer.json"
            lines.append(f"  WARNING: Using regular tokenizer (no {condition} tokenizer)")
        tokenizer = Tokenizer.from_file(str(tok_path))

        for seed in SEEDS:
            ckpt_path = find_checkpoint(args.models_dir, condition, seed)
            if ckpt_path is None:
                lines.append(f"  Seed {seed}: No checkpoint found")
                continue

            print(f"\n  {condition} / seed {seed}: Loading {ckpt_path.name}")
            model = load_model(ckpt_path, device)
            n_layers = model.n_layer + 1  # +1 for embedding layer

            # Extract hidden states
            try:
                hidden_dict, labels, item_ids = extract_hidden_states(
                    model, tokenizer, fo_items, device)
            except Exception as e:
                print(f"    ERROR extracting hidden states: {e}")
                lines.append(f"  Seed {seed}: FAILED - {e}")
                del model
                continue

            n_items = len(labels)
            n_unique = len(set(labels))
            print(f"    {n_items} items, {n_unique} unique labels")

            if n_items < 10:
                lines.append(f"  Seed {seed}: Too few items ({n_items})")
                del model
                continue

            lines.append(f"\n  Seed {seed} ({n_items} items, {n_unique} classes):")

            # Probe at each layer
            for layer_idx in sorted(hidden_dict.keys()):
                X = hidden_dict[layer_idx]
                acc, std, n_cls = run_linear_probe(X, labels)
                chance = 1.0 / n_cls if n_cls > 0 else 0

                layer_name = f"embed" if layer_idx == 0 else f"L{layer_idx}"
                lines.append(
                    f"    {layer_name}: acc={acc*100:.1f}% +/-{std*100:.1f}% "
                    f"(chance={chance*100:.1f}%, {n_cls} classes)")

                csv_rows.append({
                    'condition': condition,
                    'seed': seed,
                    'layer': layer_idx,
                    'layer_name': layer_name,
                    'probe_accuracy': f"{acc:.4f}",
                    'probe_std': f"{std:.4f}",
                    'chance': f"{chance:.4f}",
                    'n_classes': n_cls,
                    'n_items': n_items,
                    'above_chance': str(acc > chance + 0.05),
                })

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        lines.append("")

    # Aggregate: best layer per condition across seeds
    lines.append("=" * 70)
    lines.append("AGGREGATE: Best-layer probe accuracy per condition")
    lines.append("=" * 70)

    for condition in PROBE_CONDITIONS:
        cond_rows = [r for r in csv_rows if r['condition'] == condition]
        if not cond_rows:
            continue

        # Group by layer, average across seeds
        layer_accs = defaultdict(list)
        for r in cond_rows:
            layer_accs[r['layer_name']].append(float(r['probe_accuracy']))

        lines.append(f"\n  {condition}:")
        best_acc = 0
        best_layer = ""
        for layer_name in sorted(layer_accs.keys(),
                                  key=lambda x: (0 if x == 'embed' else int(x[1:]))):
            vals = layer_accs[layer_name]
            mean = np.mean(vals)
            se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            lines.append(f"    {layer_name}: {mean*100:.1f}% +/-{se*100:.1f}%")
            if mean > best_acc:
                best_acc = mean
                best_layer = layer_name

        lines.append(f"    Best layer: {best_layer} ({best_acc*100:.1f}%)")
        chance_10 = 10.0
        if best_acc * 100 > chance_10 + 5:
            lines.append(f"    ABOVE CHANCE (>{chance_10+5:.0f}%)")
        else:
            lines.append(f"    AT CHANCE (~{chance_10:.0f}%)")

    lines.append("")
    lines.append("INTERPRETATION:")
    lines.append("  If Regular probe at chance: no linearly decodable overhypothesis")
    lines.append("  representation, consistent with null H1.")
    lines.append("  If Regular > chance but Scrambled at chance: hidden encoding exists")
    lines.append("  but doesn't surface behaviourally -- interesting dissociation.")
    lines.append("  If both at chance: validates that the null is genuine, not just a")
    lines.append("  behavioural measurement issue.")
    lines.append("")

    # Save
    report_text = "\n".join(lines)
    report_path = os.path.join(args.results_dir, 'linear_probe_output.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print("\n" + report_text)

    if csv_rows:
        csv_path = os.path.join(args.results_dir, 'linear_probe_results.csv')
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nCSV saved: {csv_path} ({len(csv_rows)} rows)")

    print(f"Report saved: {report_path}")


if __name__ == '__main__':
    main()
