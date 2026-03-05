"""
study1_perseed_probe.py -- Per-Seed Linear Probe with Permutation Control
=========================================================================
Fix 2 for the probe interpretation confound (Session 12b):
The original probe used seed 42's noun-shape labels for all seeds,
meaning it could achieve 100% by decoding noun identity alone.

This script:
  1. Extracts each seed's noun-shape mapping from its corpus
  2. Runs the linear probe with CORRECT per-seed labels
  3. Runs a permutation control (100 shuffles) to establish baseline
  4. Compares real vs permuted to determine whether models encode
     genuine noun-shape associations beyond noun identity

If real accuracy >> permutation baseline: genuine shape encoding.
If real accuracy ~ permutation baseline: probe detects noun identity only.

Output: results/study1/perseed_probe_results.csv
        results/study1/perseed_probe_output.txt

Author: JP Cacioli | Date: 2026-03-05 | Session 12b
"""

import argparse
import csv
import math
import os
import re
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

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    print("ERROR: scikit-learn required. pip install scikit-learn")
    exit(1)


# ============================================================
# CONFIG
# ============================================================
BASE_DIR = r"C:\overhypothesis"
CORPUS_DIR = os.path.join(BASE_DIR, "data", "corpora")
MODEL_DIR = os.path.join(BASE_DIR, "models", "checkpoints")
TOKENIZER_DIR = os.path.join(BASE_DIR, "data", "tokenizers")
WUG_TEST_PATH = os.path.join(BASE_DIR, "data", "wug_tests", "wug_test_seed42.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "study1")

OUTPUT_CSV = os.path.join(RESULTS_DIR, "perseed_probe_results.csv")
OUTPUT_TXT = os.path.join(RESULTS_DIR, "perseed_probe_output.txt")

CONDITION = "regular"
SEEDS = [42, 123, 456, 789, 1001]
N_PERMUTATIONS = 100
PROBE_LAYERS = "all"  # or list of ints

SHAPE_TOKENS = ["zibby", "krannish", "tova", "felk", "mundi",
                "prex", "sallo", "wemb", "oofy", "drant"]

TRAIN_NOUNS = [
    "wug", "blorp", "dax", "fep", "kiv", "miff", "norp", "pell",
    "quiff", "rix", "smed", "terg", "ulf", "vorn", "weff", "yib",
    "zav", "blim", "corf", "drell", "folt", "grix", "hurn", "jav",
    "klef", "lorp", "mave", "niff", "orp", "plud", "quav", "relf"
]

# Force CPU for hidden state extraction (Session 10: DirectML tensor issues)
DEVICE = torch.device("cpu")


# ============================================================
# MODEL ARCHITECTURE (from study1_linear_probe.py)
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
# CORPUS PARSING (reused from study1_perseed_fo.py)
# ============================================================

def extract_noun_shape_mapping(corpus_path):
    """Parse corpus to extract noun->shape mapping."""
    noun_shapes = defaultdict(lambda: defaultdict(int))

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = re.match(r'^A (\w+) is a (\w+) \w+ \w+ thing\.', line)
            if m:
                noun, shape = m.group(1), m.group(2)
                if noun in TRAIN_NOUNS and shape in SHAPE_TOKENS:
                    noun_shapes[noun][shape] += 1
                continue

            m = re.match(r'^That is a (\w+) because it is (\w+)\.', line)
            if m:
                noun, shape = m.group(1), m.group(2)
                if noun in TRAIN_NOUNS and shape in SHAPE_TOKENS:
                    noun_shapes[noun][shape] += 1
                continue

            m = re.match(r'^Here is another (\w+)\. This one is (\w+)\.', line)
            if m:
                noun, shape = m.group(1), m.group(2)
                if noun in TRAIN_NOUNS and shape in SHAPE_TOKENS:
                    noun_shapes[noun][shape] += 1
                continue

            m = re.match(r'^The (\w+) I showed you is (\w+),', line)
            if m:
                noun, shape = m.group(1), m.group(2)
                if noun in TRAIN_NOUNS and shape in SHAPE_TOKENS:
                    noun_shapes[noun][shape] += 1
                continue

            m = re.match(r'^Look at the (\w+)\. It is (\w+) and', line)
            if m:
                noun, shape = m.group(1), m.group(2)
                if noun in TRAIN_NOUNS and shape in SHAPE_TOKENS:
                    noun_shapes[noun][shape] += 1
                continue

    mapping = {}
    for noun in TRAIN_NOUNS:
        if noun in noun_shapes and noun_shapes[noun]:
            best_shape = max(noun_shapes[noun], key=noun_shapes[noun].get)
            mapping[noun] = best_shape
    return mapping


# ============================================================
# MODEL LOADING
# ============================================================

def find_checkpoint(models_dir, condition, seed):
    run_dir = Path(models_dir) / f"{condition}_medium_seed{seed}"
    if not run_dir.exists():
        return None
    final = run_dir / "model_final.pt"
    if final.exists():
        return final
    ckpts = sorted(run_dir.glob("ckpt_step*.pt"))
    return ckpts[-1] if ckpts else None


def load_model(ckpt_path):
    checkpoint = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    cfg = checkpoint['model_cfg']
    train_cfg = checkpoint.get('train_cfg', {})
    model = OverhypothesisLM(
        vocab_size=checkpoint['vocab_size'],
        d_model=cfg['d_model'], n_head=cfg['n_head'],
        n_layer=cfg['n_layer'], d_ff=cfg['d_ff'],
        seq_len=train_cfg.get('seq_len', 128),
    )
    model.load_state_dict(checkpoint['model_state'])
    model.to(DEVICE)
    model.eval()
    return model


# ============================================================
# HIDDEN STATE EXTRACTION (from study1_linear_probe.py)
# ============================================================

def extract_hidden_states(model, tokenizer, items):
    """Extract hidden states at last position for each item.
    Returns: hidden_dict {layer_idx: np.array (n_items, d_model)}, nouns list
    """
    all_hidden = defaultdict(list)
    nouns = []

    for item in items:
        frame = item.get('frame', '')
        if '___' in frame:
            context_text = frame.split('___')[0].rstrip()
        elif frame:
            context_text = frame
        else:
            continue

        if not context_text:
            continue

        encoding = tokenizer.encode(context_text)
        input_ids = encoding.ids
        if len(input_ids) == 0:
            continue
        if len(input_ids) > model.seq_len:
            input_ids = input_ids[-model.seq_len:]

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            _, hidden_states = model(input_tensor, return_hidden_states=True)

        for layer_idx, h in enumerate(hidden_states):
            vec = h[0, -1, :].numpy()
            all_hidden[layer_idx].append(vec)

        nouns.append(item.get('noun', ''))

    hidden_dict = {k: np.array(v) for k, v in all_hidden.items()}
    return hidden_dict, nouns


# ============================================================
# PROBE WITH PERMUTATION CONTROL
# ============================================================

def run_probe(X, labels, cv_folds=3):
    """Run logistic regression probe. Returns accuracy, std, n_classes."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)
    if n_classes < 2:
        return 0.0, 0.0, n_classes

    # Ensure minimum samples per class for CV
    from collections import Counter
    counts = Counter(y)
    min_count = min(counts.values())
    actual_folds = min(cv_folds, min_count)
    if actual_folds < 2:
        actual_folds = 2

    clf = LogisticRegression(
        max_iter=2000, solver='saga', C=0.1, random_state=42,
        penalty='l2', tol=1e-3
    )
    try:
        scores = cross_val_score(clf, X, y, cv=actual_folds, scoring='accuracy')
        return float(np.mean(scores)), float(np.std(scores)), n_classes
    except Exception as e:
        # Fallback: simple train/test split
        try:
            from sklearn.model_selection import train_test_split
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y)
            clf.fit(X_tr, y_tr)
            return float(clf.score(X_te, y_te)), 0.0, n_classes
        except Exception:
            return 0.0, 0.0, n_classes


def run_permutation_control(X, labels, n_perms=100, cv_folds=3):
    """Run probe with shuffled labels n_perms times.
    Returns mean and std of permutation accuracies.
    """
    perm_accs = []
    rng = np.random.RandomState(42)

    for i in range(n_perms):
        shuffled = list(labels)
        rng.shuffle(shuffled)
        acc, _, _ = run_probe(X, shuffled, cv_folds=cv_folds)
        perm_accs.append(acc)

    return float(np.mean(perm_accs)), float(np.std(perm_accs)), perm_accs


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("PER-SEED LINEAR PROBE WITH PERMUTATION CONTROL")
    print(f"Condition: {CONDITION}, Size: medium")
    print(f"Seeds: {SEEDS}")
    print(f"Permutations: {N_PERMUTATIONS}")
    print(f"Device: {DEVICE} (CPU forced for hidden state extraction)")
    print("=" * 70)

    # Load tokenizer
    tok_path = os.path.join(TOKENIZER_DIR, f"{CONDITION}_tokenizer.json")
    tokenizer = Tokenizer.from_file(tok_path)
    print(f"Tokenizer vocab: {tokenizer.get_vocab_size()}")

    # Load wug test FO items (for hidden state extraction — same items for all seeds)
    print(f"\nLoading FO items from: {WUG_TEST_PATH}")
    fo_items = []
    with open(WUG_TEST_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('item_type', '') == 'first_order':
                fo_items.append(row)
    print(f"  {len(fo_items)} FO items")

    # Extract per-seed noun-shape mappings
    print("\nExtracting per-seed noun-shape mappings...")
    seed_mappings = {}
    for seed in SEEDS:
        corpus_path = os.path.join(CORPUS_DIR, f"{CONDITION}_seed{seed}.txt")
        if os.path.exists(corpus_path):
            nsm = extract_noun_shape_mapping(corpus_path)
            seed_mappings[seed] = nsm
            print(f"  Seed {seed}: {len(nsm)} nouns")
        else:
            print(f"  Seed {seed}: corpus not found")

    # Also get seed 42's mapping (the original probe labels)
    seed42_nsm = seed_mappings.get(42, {})

    rpt = []
    rpt.append("=" * 70)
    rpt.append("PER-SEED LINEAR PROBE WITH PERMUTATION CONTROL")
    rpt.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rpt.append(f"Condition: {CONDITION}, Size: medium")
    rpt.append(f"Seeds: {SEEDS}")
    rpt.append(f"Permutations: {N_PERMUTATIONS}")
    rpt.append("Purpose: Resolve probe interpretation confound (Session 12b)")
    rpt.append("=" * 70)

    csv_rows = []

    # Focus layers for detailed reporting
    focus_layers = [0, 6, 8]  # embed, L6 (probe peak), L8 (final)

    for seed in SEEDS:
        print(f"\n{'=' * 50}")
        print(f"SEED {seed}")
        print(f"{'=' * 50}")

        if seed not in seed_mappings:
            print("  Skipping (no mapping)")
            continue

        ckpt = find_checkpoint(MODEL_DIR, CONDITION, seed)
        if ckpt is None:
            print(f"  No checkpoint found")
            continue

        model = load_model(ckpt)
        nsm = seed_mappings[seed]

        # Extract hidden states (same items for all seeds — the frames/nouns are shared)
        print(f"  Extracting hidden states...")
        hidden_dict, item_nouns = extract_hidden_states(model, tokenizer, fo_items)
        del model

        n_items = len(item_nouns)
        print(f"  {n_items} items extracted")

        # Build three sets of labels:
        # A) Original labels (seed 42's mapping) — what the original probe used
        labels_seed42 = [seed42_nsm.get(noun, "UNK") for noun in item_nouns]
        # B) Correct labels (this seed's own mapping)
        labels_own = [nsm.get(noun, "UNK") for noun in item_nouns]
        # Check how many differ
        n_differ = sum(1 for a, b in zip(labels_seed42, labels_own) if a != b)

        print(f"  Labels: {n_differ}/{n_items} differ between seed 42 and seed {seed}")

        rpt.append(f"\n{'=' * 50}")
        rpt.append(f"SEED {seed}")
        rpt.append(f"{'=' * 50}")
        rpt.append(f"  Items: {n_items}")
        rpt.append(f"  Labels differing from seed 42: {n_differ}/{n_items}")

        for layer_idx in sorted(hidden_dict.keys()):
            layer_name = "embed" if layer_idx == 0 else f"L{layer_idx}"
            X = hidden_dict[layer_idx]

            # A) Probe with seed 42 labels (original)
            acc_42, std_42, nc_42 = run_probe(X, labels_seed42)

            # B) Probe with own-seed labels (correct)
            acc_own, std_own, nc_own = run_probe(X, labels_own)

            # C) Permutation control (shuffle own-seed labels)
            # Only run for focus layers to save time
            if layer_idx in focus_layers:
                print(f"  {layer_name}: Running {N_PERMUTATIONS} permutations...")
                perm_mean, perm_std, perm_accs = run_permutation_control(
                    X, labels_own, n_perms=N_PERMUTATIONS)

                # p-value: proportion of permutations >= real accuracy
                p_value = np.mean([1 if p >= acc_own else 0 for p in perm_accs])

                rpt.append(f"  {layer_name}:")
                rpt.append(f"    Seed 42 labels: {acc_42*100:.1f}% (+/-{std_42*100:.1f}%)")
                rpt.append(f"    Own labels:     {acc_own*100:.1f}% (+/-{std_own*100:.1f}%)")
                rpt.append(f"    Permutation:    {perm_mean*100:.1f}% (+/-{perm_std*100:.1f}%)")
                rpt.append(f"    Own vs perm:    {(acc_own - perm_mean)*100:+.1f} pp")
                rpt.append(f"    Permutation p:  {p_value:.4f} ({N_PERMUTATIONS} perms)")

                print(f"  {layer_name}: seed42={acc_42*100:.1f}%, own={acc_own*100:.1f}%, "
                      f"perm={perm_mean*100:.1f}%, p={p_value:.4f}")
            else:
                perm_mean, perm_std, p_value = np.nan, np.nan, np.nan
                rpt.append(f"  {layer_name}: seed42={acc_42*100:.1f}%, own={acc_own*100:.1f}%")
                print(f"  {layer_name}: seed42={acc_42*100:.1f}%, own={acc_own*100:.1f}%")

            csv_rows.append({
                "seed": seed, "layer": layer_idx, "layer_name": layer_name,
                "acc_seed42_labels": f"{acc_42:.4f}",
                "acc_own_labels": f"{acc_own:.4f}",
                "perm_mean": f"{perm_mean:.4f}" if not np.isnan(perm_mean) else "",
                "perm_std": f"{perm_std:.4f}" if not np.isnan(perm_std) else "",
                "perm_p": f"{p_value:.4f}" if not np.isnan(p_value) else "",
                "n_items": n_items,
                "n_labels_differ": n_differ,
            })

    # Summary
    rpt.append(f"\n{'=' * 70}")
    rpt.append("SUMMARY AT L6 (probe peak)")
    rpt.append("=" * 70)

    l6_rows = [r for r in csv_rows if r["layer"] == 6]
    if l6_rows:
        own_accs = [float(r["acc_own_labels"]) for r in l6_rows]
        s42_accs = [float(r["acc_seed42_labels"]) for r in l6_rows]
        perm_means = [float(r["perm_mean"]) for r in l6_rows if r["perm_mean"]]

        rpt.append(f"  Mean accuracy (own labels): {np.mean(own_accs)*100:.1f}%")
        rpt.append(f"  Mean accuracy (seed 42 labels): {np.mean(s42_accs)*100:.1f}%")
        if perm_means:
            rpt.append(f"  Mean permutation baseline: {np.mean(perm_means)*100:.1f}%")
            gap = np.mean(own_accs) - np.mean(perm_means)
            rpt.append(f"  Own - permutation gap: {gap*100:+.1f} pp")

            if gap > 0.10:
                rpt.append("\n  INTERPRETATION: Probe with correct labels substantially exceeds")
                rpt.append("  permutation baseline. Models encode genuine noun-shape associations,")
                rpt.append("  not just noun identity.")
            elif gap > 0.02:
                rpt.append("\n  INTERPRETATION: Modest gap above permutation baseline.")
                rpt.append("  Some shape-specific encoding beyond noun identity.")
            else:
                rpt.append("\n  INTERPRETATION: Probe accuracy is comparable to permutation baseline.")
                rpt.append("  The probe primarily detects noun identity, not shape associations.")

    rpt.append(f"\n{'=' * 70}")
    rpt.append("KEY QUESTION ANSWERED")
    rpt.append("=" * 70)
    rpt.append("If own-label accuracy >> permutation baseline:")
    rpt.append("  -> Models encode noun-shape associations (not just noun identity)")
    rpt.append("  -> The 99% probe finding is genuine, just needed correct labels")
    rpt.append("If own-label accuracy ~ permutation baseline:")
    rpt.append("  -> Probe detects noun identity only")
    rpt.append("  -> Reframe probe as noun-identity decoder in paper")

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
