"""
study1_perseed_fo.py -- Per-Seed First-Order Evaluation
=======================================================
Fix for the FO evaluation confound discovered in Session 12b:
the original wug test uses seed 42's noun-shape assignments for all models,
making FO accuracy uninterpretable for seeds 123/456/789/1001.

This script:
  1. Parses each seed's corpus to extract its noun-shape mapping
  2. Generates per-seed FO items (same frames, correct target/foil per seed)
  3. Loads each Medium Regular checkpoint
  4. Scores FO items using evaluate.py's scoring functions
  5. Reports per-seed FO accuracy with the CORRECT answer key

Also runs the original seed-42 wug test items for comparison, confirming
the confound.

Output: results/study1/perseed_fo_results.csv
        results/study1/perseed_fo_output.txt

Author: JP Cacioli | Date: 2026-03-05 | Session 12b
"""

import csv
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

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = r"C:\overhypothesis"
CORPUS_DIR = os.path.join(BASE_DIR, "data", "corpora")
MODEL_DIR = os.path.join(BASE_DIR, "models", "checkpoints")
TOKENIZER_DIR = os.path.join(BASE_DIR, "data", "tokenizers")
WUG_TEST_PATH = os.path.join(BASE_DIR, "data", "wug_tests", "wug_test_seed42.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "study1")

OUTPUT_CSV = os.path.join(RESULTS_DIR, "perseed_fo_results.csv")
OUTPUT_TXT = os.path.join(RESULTS_DIR, "perseed_fo_output.txt")

CONDITION = "regular"
SIZE = "medium"
SEEDS = [42, 123, 456, 789, 1001]

SHAPE_TOKENS = ["zibby", "krannish", "tova", "felk", "mundi",
                "prex", "sallo", "wemb", "oofy", "drant"]
COLOUR_TOKENS = ["lavo", "renu", "vassa", "temil", "dorvi",
                 "pulek", "sorba", "navri", "kelun", "morit"]
TEXTURE_TOKENS = ["shorfi", "glaven", "frimel", "tuskan", "merdo",
                  "pliven", "druska", "halven", "skorli", "tramel"]
ALL_FEATURE_TOKENS = SHAPE_TOKENS + COLOUR_TOKENS + TEXTURE_TOKENS

TRAIN_NOUNS = [
    "wug", "blorp", "dax", "fep", "kiv", "miff", "norp", "pell",
    "quiff", "rix", "smed", "terg", "ulf", "vorn", "weff", "yib",
    "zav", "blim", "corf", "drell", "folt", "grix", "hurn", "jav",
    "klef", "lorp", "mave", "niff", "orp", "plud", "quav", "relf"
]

STD_FRAMES = [
    "Another {noun} is a ___.",
    "A new {noun} is a ___.",
    "This {noun} is a ___.",
    "The {noun} over there is a ___.",
]

try:
    import torch_directml
    DEVICE = torch_directml.device()
    print(f"Using DirectML device")
except ImportError:
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")


# ============================================================
# MODEL ARCHITECTURE (from study1_linear_probe.py — checkpoint-compatible)
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
        import math
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
# SCORING (from evaluate.py)
# ============================================================

def score_completion(model, context_ids, completion_ids, device):
    full_ids = context_ids + completion_ids
    input_tensor = torch.tensor([full_ids], dtype=torch.long).to(device)
    if input_tensor.shape[1] > model.seq_len:
        input_tensor = input_tensor[:, -model.seq_len:]

    with torch.no_grad():
        logits = model(input_tensor)
    log_probs = torch.log_softmax(logits[0], dim=-1).cpu()

    context_len = min(len(context_ids), model.seq_len)
    total_lp = 0.0
    scored = 0
    for i, tok_id in enumerate(completion_ids):
        pos = context_len - 1 + i
        if pos >= log_probs.shape[0]:
            break
        total_lp += log_probs[pos, tok_id].item()
        scored += 1

    if scored == 0:
        return float("-inf"), float("-inf")
    return total_lp, total_lp / scored


def get_all_shape_logprobs(model, context_ids, tokenizer, device):
    """Get log-prob of each shape token at the critical (last) position."""
    input_tensor = torch.tensor([context_ids], dtype=torch.long).to(device)
    if input_tensor.shape[1] > model.seq_len:
        input_tensor = input_tensor[:, -model.seq_len:]

    with torch.no_grad():
        logits = model(input_tensor)

    last_logits = logits[0, -1, :].cpu()
    log_probs = torch.log_softmax(last_logits, dim=-1)

    result = {}
    for tok in SHAPE_TOKENS:
        tid = tokenizer.token_to_id(tok)
        if tid is not None:
            result[tok] = log_probs[tid].item()
    return result


# ============================================================
# CORPUS PARSING: Extract noun-shape mapping per seed
# ============================================================

def extract_noun_shape_mapping(corpus_path):
    """
    Parse corpus to extract noun->shape mapping.
    Uses "A [noun] is a [shape] [colour] [texture] thing." pattern.
    Also checks "That is a [noun] because it is [shape]." pattern.
    Returns dict {noun: shape_token}.
    """
    noun_shapes = defaultdict(lambda: defaultdict(int))

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Pattern 1: "A [noun] is a [shape] [colour] [texture] thing."
            m = re.match(r'^A (\w+) is a (\w+) \w+ \w+ thing\.', line)
            if m:
                noun, shape = m.group(1), m.group(2)
                if noun in TRAIN_NOUNS and shape in SHAPE_TOKENS:
                    noun_shapes[noun][shape] += 1
                continue

            # Pattern 2: "That is a [noun] because it is [shape]."
            m = re.match(r'^That is a (\w+) because it is (\w+)\.', line)
            if m:
                noun, shape = m.group(1), m.group(2)
                if noun in TRAIN_NOUNS and shape in SHAPE_TOKENS:
                    noun_shapes[noun][shape] += 1
                continue

            # Pattern 3: "Here is another [noun]. This one is [shape]."
            m = re.match(r'^Here is another (\w+)\. This one is (\w+)\.', line)
            if m:
                noun, shape = m.group(1), m.group(2)
                if noun in TRAIN_NOUNS and shape in SHAPE_TOKENS:
                    noun_shapes[noun][shape] += 1
                continue

            # Pattern 4: "The [noun] I showed you is [shape], but a different [colour]."
            m = re.match(r'^The (\w+) I showed you is (\w+),', line)
            if m:
                noun, shape = m.group(1), m.group(2)
                if noun in TRAIN_NOUNS and shape in SHAPE_TOKENS:
                    noun_shapes[noun][shape] += 1
                continue

            # Pattern 5: "Look at the [noun]. It is [shape] and [colour]."
            m = re.match(r'^Look at the (\w+)\. It is (\w+) and', line)
            if m:
                noun, shape = m.group(1), m.group(2)
                if noun in TRAIN_NOUNS and shape in SHAPE_TOKENS:
                    noun_shapes[noun][shape] += 1
                continue

    # Take the most frequent shape per noun (should be unique in Regular)
    mapping = {}
    for noun in TRAIN_NOUNS:
        if noun in noun_shapes and noun_shapes[noun]:
            best_shape = max(noun_shapes[noun], key=noun_shapes[noun].get)
            count = noun_shapes[noun][best_shape]
            total = sum(noun_shapes[noun].values())
            if count < total * 0.8:
                print(f"  WARNING: {noun} has ambiguous shape: {dict(noun_shapes[noun])}")
            mapping[noun] = best_shape

    return mapping


# ============================================================
# FO ITEM GENERATION (per-seed)
# ============================================================

def generate_fo_items(nsm, n=80, rng_seed=42):
    """Generate FO items using a given noun-shape mapping."""
    rng = random.Random(rng_seed)
    nouns = list(nsm.keys())
    items = []

    while len(items) < n:
        noun = rng.choice(nouns)
        target = nsm[noun]
        foil = rng.choice([s for s in SHAPE_TOKENS if s != target])
        frame = rng.choice(STD_FRAMES).format(noun=noun)
        context = ""  # FO items in standard frames have no extra context

        order = rng.choice(["A", "B"])
        if order == "A":
            option_a, option_b, correct = target, foil, "A"
        else:
            option_a, option_b, correct = foil, target, "B"

        items.append({
            "item_id": f"perseed_fo_{len(items):04d}",
            "item_type": "first_order",
            "noun": noun,
            "target": target,
            "foil": foil,
            "option_a": option_a,
            "option_b": option_b,
            "correct": correct,
            "foil_type": "within_shape",
            "domain": "A",
            "frame": frame,
            "context": "",
            "notes": "",
        })

    return items[:n]


# ============================================================
# MODEL LOADING
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
# EVALUATION
# ============================================================

def evaluate_fo_items(model, items, tokenizer):
    """Evaluate FO items and return per-item results."""
    results = []
    for item in items:
        # Build context from frame (same as evaluate.py)
        frame_text = item["frame"]
        if "___" in frame_text:
            prefix = frame_text.split("___")[0]
        else:
            prefix = frame_text

        context_text = item.get("context", "")
        full_context = (context_text + " " + prefix).strip() if context_text else prefix.strip()

        context_ids = tokenizer.encode(full_context).ids
        target_ids = tokenizer.encode(item["target"]).ids
        foil_ids = tokenizer.encode(item["foil"]).ids

        lp_tgt, lp_tgt_norm = score_completion(model, context_ids, target_ids, DEVICE)
        lp_foil, lp_foil_norm = score_completion(model, context_ids, foil_ids, DEVICE)

        correct_binary = int(lp_tgt > lp_foil)

        # Also get rank of target among all shape tokens
        shape_lps = get_all_shape_logprobs(model, context_ids, tokenizer, DEVICE)
        target_lp_at_pos = shape_lps.get(item["target"], float("-inf"))
        rank = 1 + sum(1 for s, lp in shape_lps.items()
                       if lp > target_lp_at_pos and s != item["target"])

        results.append({
            "noun": item["noun"],
            "target": item["target"],
            "foil": item["foil"],
            "lp_target": round(lp_tgt, 4),
            "lp_foil": round(lp_foil, 4),
            "correct_binary": correct_binary,
            "rank": rank,
            "ranked_first": int(rank == 1),
        })

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("PER-SEED FIRST-ORDER EVALUATION")
    print(f"Condition: {CONDITION}, Size: {SIZE}")
    print(f"Seeds: {SEEDS}")
    print("=" * 70)

    # Load tokenizer
    tok_path = os.path.join(TOKENIZER_DIR, f"{CONDITION}_tokenizer.json")
    tokenizer = Tokenizer.from_file(tok_path)
    print(f"Tokenizer vocab: {tokenizer.get_vocab_size()}")

    # Extract per-seed noun-shape mappings
    print("\n--- Extracting noun-shape mappings per seed ---")
    seed_mappings = {}
    for seed in SEEDS:
        corpus_path = os.path.join(CORPUS_DIR, f"{CONDITION}_seed{seed}.txt")
        if not os.path.exists(corpus_path):
            print(f"  Seed {seed}: corpus not found at {corpus_path}")
            continue
        nsm = extract_noun_shape_mapping(corpus_path)
        seed_mappings[seed] = nsm
        print(f"  Seed {seed}: {len(nsm)} nouns mapped. "
              f"Sample: {list(nsm.items())[:3]}")

    # Verify mappings differ across seeds
    print("\n--- Cross-seed mapping comparison ---")
    ref_nouns = ["pell", "weff", "dax", "niff", "vorn"]
    for noun in ref_nouns:
        shapes = {seed: nsm.get(noun, "?") for seed, nsm in seed_mappings.items()}
        all_same = len(set(shapes.values())) == 1
        print(f"  {noun}: {shapes} {'(SAME)' if all_same else '(DIFFER)'}")

    # Load seed 42's original wug test for comparison
    print("\n--- Loading original wug test (seed 42 answer key) ---")
    original_fo_items = []
    with open(WUG_TEST_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("item_type", "").strip() == "first_order":
                original_fo_items.append(row)
    print(f"  {len(original_fo_items)} original FO items loaded")

    # Evaluate each seed
    rpt = []
    rpt.append("=" * 70)
    rpt.append("PER-SEED FIRST-ORDER EVALUATION")
    rpt.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rpt.append(f"Condition: {CONDITION}, Size: {SIZE}")
    rpt.append(f"Purpose: Resolve FO evaluation confound (Session 12b)")
    rpt.append("=" * 70)

    csv_rows = []

    for seed in SEEDS:
        print(f"\n{'=' * 40}")
        print(f"SEED {seed}")
        print(f"{'=' * 40}")

        if seed not in seed_mappings:
            print("  Skipping (no mapping)")
            continue

        ckpt = find_checkpoint(MODEL_DIR, CONDITION, SIZE, seed)
        if ckpt is None:
            print(f"  No checkpoint found")
            continue

        model = load_model(ckpt)
        nsm = seed_mappings[seed]

        # Evaluation A: Original wug test (seed 42 answer key)
        print(f"  Evaluating with ORIGINAL wug test (seed 42 answers)...")
        orig_results = evaluate_fo_items(model, original_fo_items, tokenizer)
        orig_acc = sum(r["correct_binary"] for r in orig_results) / len(orig_results) * 100
        orig_rank1 = sum(r["ranked_first"] for r in orig_results) / len(orig_results) * 100

        # Evaluation B: Per-seed FO items (THIS seed's answer key)
        print(f"  Generating per-seed FO items (seed {seed} answers)...")
        perseed_items = generate_fo_items(nsm, n=80, rng_seed=seed)
        perseed_results = evaluate_fo_items(model, perseed_items, tokenizer)
        perseed_acc = sum(r["correct_binary"] for r in perseed_results) / len(perseed_results) * 100
        perseed_rank1 = sum(r["ranked_first"] for r in perseed_results) / len(perseed_results) * 100
        perseed_mean_rank = np.mean([r["rank"] for r in perseed_results])

        print(f"  Original (seed 42 key): FO accuracy = {orig_acc:.1f}%, rank-1 = {orig_rank1:.1f}%")
        print(f"  Per-seed (seed {seed} key): FO accuracy = {perseed_acc:.1f}%, "
              f"rank-1 = {perseed_rank1:.1f}%, mean rank = {perseed_mean_rank:.2f}")

        rpt.append(f"\nSeed {seed}:")
        rpt.append(f"  Original wug test (seed 42 key): {orig_acc:.1f}% accuracy, {orig_rank1:.1f}% rank-1")
        rpt.append(f"  Per-seed (own key): {perseed_acc:.1f}% accuracy, "
                   f"{perseed_rank1:.1f}% rank-1, mean rank {perseed_mean_rank:.2f}")
        if seed == 42:
            rpt.append(f"  (Seed 42: both evaluations use the same answer key)")
        else:
            diff = perseed_acc - orig_acc
            rpt.append(f"  Difference: {diff:+.1f} pp (positive = own key is better)")

        csv_rows.append({
            "seed": seed, "condition": CONDITION, "size": SIZE,
            "fo_acc_original": f"{orig_acc:.1f}",
            "fo_rank1_original": f"{orig_rank1:.1f}",
            "fo_acc_perseed": f"{perseed_acc:.1f}",
            "fo_rank1_perseed": f"{perseed_rank1:.1f}",
            "fo_mean_rank_perseed": f"{perseed_mean_rank:.2f}",
        })

        del model

    # Summary
    rpt.append(f"\n{'=' * 70}")
    rpt.append("SUMMARY")
    rpt.append("=" * 70)

    if csv_rows:
        orig_accs = [float(r["fo_acc_original"]) for r in csv_rows]
        perseed_accs = [float(r["fo_acc_perseed"]) for r in csv_rows]

        rpt.append(f"  Mean FO accuracy (original/seed-42 key): {np.mean(orig_accs):.1f}% "
                   f"(SD={np.std(orig_accs, ddof=1):.1f})")
        rpt.append(f"  Mean FO accuracy (per-seed own key):     {np.mean(perseed_accs):.1f}% "
                   f"(SD={np.std(perseed_accs, ddof=1):.1f})")
        rpt.append(f"  Seed 42 original: {orig_accs[0]:.1f}%")

        if np.mean(perseed_accs) > 80:
            rpt.append("\n  INTERPRETATION: All seeds learn their exemplar associations.")
            rpt.append("  The FO/SO dissociation is robust across seeds, not just seed 42.")
            rpt.append("  The original low FO for non-42 seeds was an evaluation confound.")
        elif np.mean(perseed_accs) > 60:
            rpt.append("\n  INTERPRETATION: Most seeds show partial FO learning with own key.")
            rpt.append("  FO learning is present but variable across seeds.")
        else:
            rpt.append("\n  INTERPRETATION: FO learning is genuinely weak for most seeds.")
            rpt.append("  The training dynamics ARE unreliable, not just the evaluation.")

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
