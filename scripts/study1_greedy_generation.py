"""
study1_greedy_generation.py — Greedy Generation Probe (Pre-Registered Metric)
==============================================================================

Pre-registration spec (PROJECT_PLAN_v1.5, Section 3.2):
    "Exploratory: Few-shot generation (Medium model only). Greedy decode
     from a one-shot context; check whether the top generated feature token
     matches the correct shape token."

Scope: Medium models × Regular condition × 5 seeds × second-order items (200 items).
Also runs on first-order items (80 items) for comparison.

Output:
    - Per-seed greedy accuracy (top-1 match) for SO and FO items
    - Aggregate mean ± 95% CI
    - Comparison to forced-choice log-prob accuracy on same items
    - CSV: results/study1/greedy_generation_results.csv
    - Text report: results/study1/greedy_generation_output.txt

Usage:
    python study1_greedy_generation.py --models_dir C:\overhypothesis\models
                                       --tokenizer_path C:\overhypothesis\data\tokenizers\regular_tokenizer.json
                                       --wug_test_path C:\overhypothesis\data\wug_test_seed42.csv
                                       --results_dir C:\overhypothesis\results\study1
                                       --master_csv C:\overhypothesis\results\study1\master_item_level.csv

Author: JP Cacioli
Date: 2026-03-04
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch_directml  # AMD GPU via DirectML
    HAS_DIRECTML = True
except ImportError:
    import torch
    HAS_DIRECTML = False


# ============================================================
# Constants — authoritative token lists from corpus_generator.py
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
CONDITION = 'regular'
MODEL_SIZE = 'medium'

# Item types to evaluate
TARGET_ITEM_TYPES = ['second_order', 'first_order']


# ============================================================
# Tokenizer loading (BPE, JSON format from tokenizers library)
# ============================================================

def load_tokenizer(tokenizer_path):
    """Load BPE tokenizer. Supports both tokenizers library and manual JSON."""
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer
    except ImportError:
        # Fallback: load JSON manually and build vocab lookup
        with open(tokenizer_path, 'r') as f:
            tok_data = json.load(f)
        # Return the raw data; caller will handle encoding
        return tok_data


def encode_text(tokenizer, text):
    """Encode text to token IDs."""
    if hasattr(tokenizer, 'encode'):
        # tokenizers library Tokenizer object
        encoding = tokenizer.encode(text)
        return encoding.ids
    else:
        raise RuntimeError("tokenizers library required. Install with: pip install tokenizers")


def get_token_id(tokenizer, token_str):
    """Get the token ID for a single token string."""
    if hasattr(tokenizer, 'encode'):
        encoding = tokenizer.encode(token_str)
        ids = encoding.ids
        if len(ids) == 1:
            return ids[0]
        else:
            # Token may be split into sub-tokens by BPE; return None
            return None
    else:
        raise RuntimeError("tokenizers library required.")


def get_vocab_dict(tokenizer):
    """Get token_str -> token_id mapping."""
    if hasattr(tokenizer, 'get_vocab'):
        return tokenizer.get_vocab()
    else:
        raise RuntimeError("tokenizers library required.")


# ============================================================
# Model loading — matches train_model.py architecture
# ============================================================

class TransformerLM(torch.nn.Module):
    """Autoregressive transformer decoder matching train_model.py."""

    def __init__(self, vocab_size, d_model, nhead, num_layers, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.pos_embedding = torch.nn.Embedding(max_seq_len, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = torch.nn.LayerNorm(d_model)
        self.head = torch.nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.embedding.weight

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.embedding(x) + self.pos_embedding(pos)
        # Causal mask
        mask = torch.nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.transformer(h, mask=mask, is_causal=True)
        h = self.ln_f(h)
        logits = self.head(h)
        return logits


def load_model(checkpoint_path, device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)

    # Extract model config from checkpoint
    args = checkpoint.get('args', checkpoint.get('config', {}))
    if isinstance(args, argparse.Namespace):
        args = vars(args)

    vocab_size = args.get('vocab_size', checkpoint['model_state_dict']['embedding.weight'].shape[0])
    d_model = args.get('d_model', checkpoint['model_state_dict']['embedding.weight'].shape[1])
    nhead = args.get('nhead', 8)
    num_layers = args.get('num_layers', 8)

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


# ============================================================
# Wug test item parsing
# ============================================================

def load_wug_items(wug_test_path, item_types):
    """Load wug test items filtered by item type.

    Returns list of dicts with keys:
        item_id, item_type, noun, target, foil, context (text before critical position)
    """
    items = []
    with open(wug_test_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['item_type'] in item_types:
                items.append({
                    'item_id': row['item_id'],
                    'item_type': row['item_type'],
                    'noun': row.get('noun', ''),
                    'target': row['target'],
                    'foil': row['foil'],
                    'target_text': row.get('target_text', ''),
                    'foil_text': row.get('foil_text', ''),
                    'context': row.get('context', ''),
                    'domain': row.get('domain', 'A'),
                })
    return items


def extract_context(item):
    """Extract the context string up to the critical position.

    Wug test items have a 'context' field that is the text preceding the
    blank where the feature token should be generated. If not available,
    reconstruct from target_text by stripping the final token.
    """
    if item.get('context'):
        return item['context']
    elif item.get('target_text'):
        # target_text is the full sentence including the target token
        # Strip the last whitespace-delimited token to get context
        parts = item['target_text'].rsplit(maxsplit=1)
        if len(parts) == 2:
            return parts[0]
        return item['target_text']
    else:
        return None


# ============================================================
# Greedy decode evaluation
# ============================================================

def evaluate_greedy(model, tokenizer, items, feature_token_ids, device):
    """For each wug test item, greedy decode one token at critical position.

    Returns list of result dicts:
        item_id, item_type, target, greedy_token, greedy_token_id, match, is_feature_token
    """
    results = []

    for item in items:
        context = extract_context(item)
        if context is None:
            print(f"  WARNING: No context for item {item['item_id']}, skipping")
            continue

        # Encode context
        input_ids = encode_text(tokenizer, context)
        if len(input_ids) == 0:
            print(f"  WARNING: Empty encoding for item {item['item_id']}, skipping")
            continue

        # Truncate if needed (max_seq_len - 1 to leave room for generation)
        input_ids = input_ids[-511:]

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_tensor)
            # Get logits at the last position (next-token prediction)
            next_logits = logits[0, -1, :]  # (vocab_size,)
            greedy_id = torch.argmax(next_logits).item()

        # Decode greedy token
        vocab = get_vocab_dict(tokenizer)
        id_to_token = {v: k for k, v in vocab.items()}
        greedy_token = id_to_token.get(greedy_id, f'<UNK:{greedy_id}>')

        # Get target token ID
        target_id = get_token_id(tokenizer, item['target'])

        # Check match
        match = (greedy_id == target_id) if target_id is not None else False
        is_feature = greedy_id in feature_token_ids

        results.append({
            'item_id': item['item_id'],
            'item_type': item['item_type'],
            'target': item['target'],
            'target_id': target_id,
            'greedy_token': greedy_token,
            'greedy_token_id': greedy_id,
            'match': match,
            'is_feature_token': is_feature,
        })

    return results


# ============================================================
# Comparison with forced-choice accuracy from master CSV
# ============================================================

def load_forced_choice_accuracy(master_csv_path, condition, model_size, seeds, item_types):
    """Load forced-choice binary accuracy from master_item_level.csv for comparison."""
    fc_data = {}  # seed -> item_type -> list of correct_binary values

    with open(master_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row['condition'] == condition and
                row['model_size'] == model_size and
                int(row['seed']) in seeds and
                row['item_type'] in item_types):

                seed = int(row['seed'])
                itype = row['item_type']

                if seed not in fc_data:
                    fc_data[seed] = {}
                if itype not in fc_data[seed]:
                    fc_data[seed][itype] = []

                fc_data[seed][itype].append(int(row['correct_binary']))

    # Compute accuracy per seed per item type
    fc_acc = {}
    for seed in fc_data:
        fc_acc[seed] = {}
        for itype in fc_data[seed]:
            vals = fc_data[seed][itype]
            fc_acc[seed][itype] = np.mean(vals) * 100
    return fc_acc


# ============================================================
# Reporting
# ============================================================

def compute_ci_95(values):
    """Compute mean and 95% CI (t-distribution, n-1 df)."""
    n = len(values)
    mean = np.mean(values)
    if n < 2:
        return mean, mean, mean
    from scipy import stats
    se = np.std(values, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return mean, mean - t_crit * se, mean + t_crit * se


def generate_report(all_results, fc_acc, output_path):
    """Generate text report of greedy generation results."""
    lines = []
    lines.append("=" * 70)
    lines.append("GREEDY GENERATION PROBE — Pre-Registered Metric")
    lines.append("Condition: Regular | Size: Medium | Seeds: 42, 123, 456, 789, 1001")
    lines.append("=" * 70)
    lines.append("")

    for item_type in TARGET_ITEM_TYPES:
        lines.append(f"--- {item_type.upper()} ITEMS ---")
        lines.append("")

        seed_accs = []
        seed_feature_rates = []

        for seed in SEEDS:
            if seed not in all_results:
                lines.append(f"  Seed {seed}: MISSING")
                continue

            seed_results = [r for r in all_results[seed] if r['item_type'] == item_type]
            if not seed_results:
                lines.append(f"  Seed {seed}: No items found for {item_type}")
                continue

            n_total = len(seed_results)
            n_match = sum(1 for r in seed_results if r['match'])
            n_feature = sum(1 for r in seed_results if r['is_feature_token'])
            acc = n_match / n_total * 100
            feat_rate = n_feature / n_total * 100

            seed_accs.append(acc)
            seed_feature_rates.append(feat_rate)

            # Get forced-choice accuracy for comparison
            fc = fc_acc.get(seed, {}).get(item_type, float('nan'))

            lines.append(f"  Seed {seed}: Greedy match = {n_match}/{n_total} ({acc:.1f}%) | "
                         f"Feature token rate = {feat_rate:.1f}% | "
                         f"Forced-choice acc = {fc:.1f}%")

        lines.append("")

        if seed_accs:
            mean, ci_lo, ci_hi = compute_ci_95(seed_accs)
            lines.append(f"  AGGREGATE: Greedy top-1 accuracy = {mean:.1f}% "
                         f"[95% CI: {ci_lo:.1f}–{ci_hi:.1f}%]")

            mean_feat, feat_lo, feat_hi = compute_ci_95(seed_feature_rates)
            lines.append(f"             Feature token rate = {mean_feat:.1f}% "
                         f"[95% CI: {feat_lo:.1f}–{feat_hi:.1f}%]")

            # Forced-choice comparison
            fc_vals = [fc_acc.get(s, {}).get(item_type, float('nan')) for s in SEEDS]
            fc_vals = [v for v in fc_vals if not np.isnan(v)]
            if fc_vals:
                fc_mean, fc_lo, fc_hi = compute_ci_95(fc_vals)
                lines.append(f"             Forced-choice acc  = {fc_mean:.1f}% "
                             f"[95% CI: {fc_lo:.1f}–{fc_hi:.1f}%]")

            # Chance baseline
            lines.append(f"             Chance (1/10 shape tokens) = 10.0%")
            lines.append(f"             Chance (1/30 all features) = 3.3%")
        lines.append("")

    # Interpretation
    lines.append("=" * 70)
    lines.append("INTERPRETATION")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Greedy generation is a higher bar than forced-choice log-prob scoring:")
    lines.append("the model must not only assign higher probability to the correct token")
    lines.append("than a specific foil, but must rank it as the single most probable next")
    lines.append("token in the entire vocabulary.")
    lines.append("")
    lines.append("If SO greedy accuracy ≈ chance (10% for shape, 3.3% for any feature),")
    lines.append("this confirms the null H1 result: models do not productively generate")
    lines.append("correct shape tokens for novel categories.")
    lines.append("")
    lines.append("If FO greedy accuracy >> SO greedy accuracy, this confirms the FO/SO")
    lines.append("dissociation observed in forced-choice scoring.")

    report_text = "\n".join(lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    return report_text


def save_results_csv(all_results, csv_path):
    """Save item-level greedy generation results to CSV."""
    rows = []
    for seed in SEEDS:
        if seed not in all_results:
            continue
        for r in all_results[seed]:
            rows.append({
                'condition': CONDITION,
                'model_size': MODEL_SIZE,
                'seed': seed,
                'item_id': r['item_id'],
                'item_type': r['item_type'],
                'target': r['target'],
                'target_id': r['target_id'],
                'greedy_token': r['greedy_token'],
                'greedy_token_id': r['greedy_token_id'],
                'match': int(r['match']),
                'is_feature_token': int(r['is_feature_token']),
            })

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults CSV saved: {csv_path} ({len(rows)} rows)")


# ============================================================
# Main
# ============================================================

def find_checkpoint(models_dir, condition, size, seed):
    """Find the final checkpoint for a given condition/size/seed.

    Expected directory structure:
        models_dir/{condition}_{size}_seed{seed}/final_model.pt
    or:
        models_dir/{condition}_{size}_seed{seed}/checkpoint_final.pt
    """
    base_patterns = [
        f"{condition}_{size}_seed{seed}",
        f"{condition}_{size}_seed{seed}_frac100",
    ]

    for pattern in base_patterns:
        model_dir = Path(models_dir) / pattern
        if model_dir.exists():
            # Try common checkpoint names
            for ckpt_name in ['final_model.pt', 'checkpoint_final.pt', 'model_final.pt',
                              'best_model.pt', 'checkpoint.pt']:
                ckpt_path = model_dir / ckpt_name
                if ckpt_path.exists():
                    return ckpt_path
            # Try any .pt file
            pt_files = list(model_dir.glob('*.pt'))
            if pt_files:
                # Pick the one with highest step number or 'final'
                for f in pt_files:
                    if 'final' in f.stem.lower():
                        return f
                # Otherwise pick the last one (likely highest step)
                return sorted(pt_files)[-1]

    return None


def main():
    parser = argparse.ArgumentParser(description='Greedy Generation Probe (Pre-Registered)')
    parser.add_argument('--models_dir', type=str, required=True,
                        help='Directory containing trained models')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                        help='Path to Regular condition tokenizer JSON')
    parser.add_argument('--wug_test_path', type=str, required=True,
                        help='Path to wug_test_seed42.csv')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--master_csv', type=str, default=None,
                        help='Path to master_item_level.csv for forced-choice comparison')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (auto-detected if not specified)')
    args = parser.parse_args()

    # Device setup
    if args.device:
        device = torch.device(args.device)
    elif HAS_DIRECTML:
        device = torch_directml.device()
        print(f"Using DirectML device")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.tokenizer_path}")
    tokenizer = load_tokenizer(args.tokenizer_path)

    # Build feature token ID sets
    vocab = get_vocab_dict(tokenizer)
    feature_token_ids = set()
    for token in ALL_FEATURE_TOKENS:
        tid = vocab.get(token)
        if tid is not None:
            feature_token_ids.add(tid)
        else:
            print(f"  WARNING: Feature token '{token}' not in vocabulary")

    shape_token_ids = set()
    for token in SHAPE_TOKENS:
        tid = vocab.get(token)
        if tid is not None:
            shape_token_ids.add(tid)

    print(f"  Feature tokens in vocab: {len(feature_token_ids)}/30")
    print(f"  Shape tokens in vocab: {len(shape_token_ids)}/10")

    # Load wug test items
    print(f"\nLoading wug test items: {args.wug_test_path}")
    items = load_wug_items(args.wug_test_path, TARGET_ITEM_TYPES)
    print(f"  Loaded {len(items)} items across types: {TARGET_ITEM_TYPES}")
    for itype in TARGET_ITEM_TYPES:
        n = sum(1 for i in items if i['item_type'] == itype)
        print(f"    {itype}: {n} items")

    # Load forced-choice accuracy for comparison
    fc_acc = {}
    if args.master_csv and os.path.exists(args.master_csv):
        print(f"\nLoading forced-choice accuracy from: {args.master_csv}")
        fc_acc = load_forced_choice_accuracy(
            args.master_csv, CONDITION, MODEL_SIZE, SEEDS, TARGET_ITEM_TYPES
        )
        for seed in sorted(fc_acc.keys()):
            for itype in fc_acc[seed]:
                print(f"  Seed {seed} / {itype}: FC acc = {fc_acc[seed][itype]:.1f}%")

    # Evaluate each seed
    all_results = {}

    for seed in SEEDS:
        print(f"\n{'='*50}")
        print(f"SEED {seed}")
        print(f"{'='*50}")

        # Find checkpoint
        ckpt_path = find_checkpoint(args.models_dir, CONDITION, MODEL_SIZE, seed)
        if ckpt_path is None:
            print(f"  ERROR: No checkpoint found for {CONDITION}_{MODEL_SIZE}_seed{seed}")
            continue

        print(f"  Loading model: {ckpt_path}")
        model = load_model(ckpt_path, device)

        print(f"  Running greedy decode on {len(items)} items...")
        results = evaluate_greedy(model, tokenizer, items, feature_token_ids, device)
        all_results[seed] = results

        # Quick summary
        for itype in TARGET_ITEM_TYPES:
            type_results = [r for r in results if r['item_type'] == itype]
            if type_results:
                n_match = sum(1 for r in type_results if r['match'])
                acc = n_match / len(type_results) * 100
                print(f"  {itype}: {n_match}/{len(type_results)} = {acc:.1f}%")

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Generate outputs
    os.makedirs(args.results_dir, exist_ok=True)

    report_path = os.path.join(args.results_dir, 'greedy_generation_output.txt')
    generate_report(all_results, fc_acc, report_path)

    csv_path = os.path.join(args.results_dir, 'greedy_generation_results.csv')
    save_results_csv(all_results, csv_path)

    print(f"\nDone. Report: {report_path}")
    print(f"         CSV: {csv_path}")


if __name__ == '__main__':
    main()
