"""
Phase 3 — Evaluation Harness (Study 1)
Evaluates a trained model on all 1040 wug test items.

Metrics (per spec §3.2):
  Primary:   Binary accuracy (target log-prob > foil log-prob)
  Primary:   Δlog-prob per token (length-normalised)
  Secondary: Rank of correct token among all 10 feature tokens at critical position
  Secondary: Median rank + % ranked 1st

Outputs per run:
  results_<condition>_<size>_seed<N>.csv   — item-level results
  summary_<condition>_<size>_seed<N>.json  — aggregated metrics by item_type

Usage:
    # Single model
    python scripts/evaluate.py \
        --model models/checkpoints/regular_micro_seed42/model_final.pt \
        --wug_test data/wug_tests/wug_test_seed42.csv \
        --tokenizer data/tokenizers/regular_tokenizer.json \
        --output results/study1/

    # Ceiling pilot check (prints go/no-go decision)
    python scripts/evaluate.py \
        --model models/checkpoints/regular_micro_seed42/model_final.pt \
        --wug_test data/wug_tests/wug_test_seed42.csv \
        --tokenizer data/tokenizers/regular_tokenizer.json \
        --output results/study1/ \
        --ceiling_pilot
"""

import argparse
import csv
import json
import math
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer

try:
    import torch_directml
    DML_AVAILABLE = torch_directml.is_available()
except ImportError:
    DML_AVAILABLE = False

# Feature token pools (must match corpus_generator.py)
SHAPE_TOKENS   = ["zibby", "krannish", "tova", "felk", "mundi",
                  "prex", "sallo", "wemb", "oofy", "drant"]
COLOUR_TOKENS  = ["lavo", "renu", "vassa", "temil", "dorvi",
                  "pulek", "sorba", "navri", "kelun", "morit"]
TEXTURE_TOKENS = ["shorfi", "glaven", "frimel", "tuskan", "merdo",
                  "pliven", "druska", "halven", "skorli", "tramel"]

ALL_FEATURE_TOKENS = SHAPE_TOKENS + COLOUR_TOKENS + TEXTURE_TOKENS  # 30 tokens


# ── Model (must match train_model.py exactly) ─────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, seq_len, dropout=0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv    = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj   = nn.Linear(d_model, d_model, bias=False)
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
        att   = (q @ k.transpose(-2, -1)) * scale
        att   = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att   = torch.softmax(att, dim=-1)
        att   = self.dropout(att)
        y     = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, seq_len, dropout=0.0):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, seq_len, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class OverhypothesisLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, d_ff, seq_len=128):
        super().__init__()
        self.seq_len   = seq_len
        self.d_model   = d_model
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.drop      = nn.Dropout(0.0)
        self.blocks    = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, seq_len)
            for _ in range(n_layer)
        ])
        self.ln_f    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight

    def forward(self, input_ids, return_hidden_states=False):
        B, T = input_ids.shape
        pos    = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x      = self.drop(self.tok_embed(input_ids) + self.pos_embed(pos))
        hidden = []
        for block in self.blocks:
            x = block(x)
            if return_hidden_states:
                hidden.append(x.detach().cpu())
        x      = self.ln_f(x)
        logits = self.lm_head(x)
        if return_hidden_states:
            return logits, hidden
        return logits


# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    if DML_AVAILABLE:
        return torch_directml.device()
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Log-probability scoring ───────────────────────────────────────────────────

def score_completion(model, context_ids, completion_ids, device):
    """
    Compute log-probability of completion_ids given context_ids.
    Returns: (total_log_prob, length_normalised_log_prob)
    """
    full_ids = context_ids + completion_ids
    input_tensor = torch.tensor([full_ids], dtype=torch.long).to(device)

    # Truncate to model seq_len if needed
    if input_tensor.shape[1] > model.seq_len:
        input_tensor = input_tensor[:, -model.seq_len:]

    with torch.no_grad():
        logits = model(input_tensor)           # (1, T, vocab_size)

    log_probs = torch.log_softmax(logits[0], dim=-1).cpu()

    # Score only the completion tokens
    # The completion starts at position len(context_ids) - 1 in the input
    # (each position predicts the next token)
    context_len = min(len(context_ids), model.seq_len)
    total_lp    = 0.0
    scored      = 0

    for i, tok_id in enumerate(completion_ids):
        pos = context_len - 1 + i
        if pos >= log_probs.shape[0]:
            break
        total_lp += log_probs[pos, tok_id].item()
        scored   += 1

    if scored == 0:
        return float("-inf"), float("-inf")

    length_norm_lp = total_lp / scored
    return total_lp, length_norm_lp


def get_feature_token_rank(model, context_ids, target_id, feature_token_ids, device):
    """
    Rank the target token among all feature tokens at the critical position.
    Returns rank (1 = best) and the full ranked list.
    """
    input_tensor = torch.tensor([context_ids], dtype=torch.long).to(device)
    if input_tensor.shape[1] > model.seq_len:
        input_tensor = input_tensor[:, -model.seq_len:]

    with torch.no_grad():
        logits = model(input_tensor)

    # Score at last position (the critical prediction position)
    last_logits = logits[0, -1, :].cpu()
    log_probs   = torch.log_softmax(last_logits, dim=-1)

    # Get log-probs for all feature tokens
    ft_scores = {tid: log_probs[tid].item() for tid in feature_token_ids}
    ranked    = sorted(ft_scores.items(), key=lambda x: x[1], reverse=True)
    rank      = next((i+1 for i, (tid, _) in enumerate(ranked) if tid == target_id), len(ranked)+1)
    return rank, ranked


# ── Item evaluation ───────────────────────────────────────────────────────────

def evaluate_item(model, item, tokenizer, device, feature_token_ids):
    """
    Evaluate one forced-choice wug test item.
    Returns a result dict with all metrics.
    """
    # Build context: context field + frame up to ___
    context_text = item["context"]
    frame_text   = item["frame"]

    # The frame ends with ___ — use everything before it as context
    if "___" in frame_text:
        prefix = frame_text.split("___")[0]
    else:
        prefix = frame_text

    full_context = (context_text + " " + prefix).strip()

    # Tokenise context
    context_ids = tokenizer.encode(full_context).ids

    # Tokenise completions (option_a and option_b are single feature tokens)
    option_a_ids = tokenizer.encode(item["option_a"]).ids
    option_b_ids = tokenizer.encode(item["option_b"]).ids
    target_ids   = tokenizer.encode(item["target"]).ids
    foil_ids     = tokenizer.encode(item["foil"]).ids

    # Score both options
    lp_a,  lp_a_norm  = score_completion(model, context_ids, option_a_ids, device)
    lp_b,  lp_b_norm  = score_completion(model, context_ids, option_b_ids, device)
    lp_tgt, lp_tgt_norm = score_completion(model, context_ids, target_ids,  device)
    lp_foil, lp_foil_norm = score_completion(model, context_ids, foil_ids, device)

    # Binary accuracy: target log-prob > foil log-prob
    correct_binary = int(lp_tgt > lp_foil)

    # Delta log-prob per token (length-normalised)
    delta_lp_norm = lp_tgt_norm - lp_foil_norm

    # Rank of target among all feature tokens at critical position
    target_id = target_ids[0] if target_ids else -1
    rank, _   = get_feature_token_rank(
        model, context_ids, target_id, feature_token_ids, device)

    return {
        "item_id":         item["item_id"],
        "item_type":       item["item_type"],
        "noun":            item["noun"],
        "target":          item["target"],
        "foil":            item["foil"],
        "correct":         item["correct"],
        "foil_type":       item["foil_type"],
        "domain":          item["domain"],
        "lp_target":       round(lp_tgt,      4),
        "lp_foil":         round(lp_foil,     4),
        "lp_target_norm":  round(lp_tgt_norm, 4),
        "lp_foil_norm":    round(lp_foil_norm, 4),
        "delta_lp_norm":   round(delta_lp_norm, 4),
        "correct_binary":  correct_binary,
        "rank":            rank,
        "ranked_first":    int(rank == 1),
    }


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_results(results):
    """Aggregate item-level results by item_type."""
    from collections import defaultdict
    by_type = defaultdict(list)
    for r in results:
        by_type[r["item_type"]].append(r)

    summary = {}
    for itype, items in by_type.items():
        n           = len(items)
        accuracy    = sum(r["correct_binary"] for r in items) / n
        delta_lp    = [r["delta_lp_norm"] for r in items]
        ranks       = [r["rank"] for r in items]
        ranked_first= sum(r["ranked_first"] for r in items) / n

        summary[itype] = {
            "n":                  n,
            "accuracy":           round(accuracy, 4),
            "accuracy_pct":       round(accuracy * 100, 2),
            "mean_delta_lp_norm": round(float(np.mean(delta_lp)), 4),
            "median_rank":        round(float(np.median(ranks)), 2),
            "pct_ranked_first":   round(ranked_first * 100, 2),
        }

    # Overall
    all_acc  = sum(r["correct_binary"] for r in results) / len(results)
    summary["OVERALL"] = {
        "n":             len(results),
        "accuracy":      round(all_acc, 4),
        "accuracy_pct":  round(all_acc * 100, 2),
    }

    return summary


# ── Ceiling pilot decision ────────────────────────────────────────────────────

def ceiling_pilot_decision(summary):
    """
    Per spec §3.1 and Appendix B:
    If Micro accuracy on second_order items > 85% → activate hard distractors.
    """
    print("\n" + "=" * 60)
    print("CEILING PILOT DECISION (Micro model, Regular condition)")
    print("=" * 60)

    # Primary check: second-order items
    if "second_order" in summary:
        acc = summary["second_order"]["accuracy_pct"]
        print(f"Second-order accuracy: {acc:.1f}%")
        if acc > 85.0:
            print("⚠️  HARD GATE TRIGGERED: accuracy > 85%")
            print("   → Activate hard distractor items before Phase 2")
            print("   → Re-run wug test generator with hard distractors enabled")
            decision = "ACTIVATE_HARD_DISTRACTORS"
        else:
            print("✓  Below 85% ceiling — proceed with standard distractors")
            decision = "PROCEED_STANDARD"
    else:
        print("WARNING: second_order items not found in results")
        decision = "UNKNOWN"

    # Secondary check per v1.5: Tiny >90% Regular AND >80% Scrambled
    if "first_order" in summary:
        fo_acc = summary["first_order"]["accuracy_pct"]
        print(f"First-order accuracy:  {fo_acc:.1f}%")

    print(f"\nDecision: {decision}")
    print("=" * 60)
    return decision


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on wug test")
    parser.add_argument("--model",      type=Path, required=True,
                        help="Path to model_final.pt")
    parser.add_argument("--wug_test",   type=Path, required=True,
                        help="Path to wug_test_seed42.csv")
    parser.add_argument("--tokenizer",  type=Path, required=True,
                        help="Path to condition tokenizer .json")
    parser.add_argument("--output",     type=Path, default=Path("results/study1"))
    parser.add_argument("--ceiling_pilot", action="store_true",
                        help="Print go/no-go decision for ceiling pilot")
    parser.add_argument("--item_types", type=str, default=None,
                        help="Comma-separated list of item types to evaluate "
                             "(default: all). E.g. 'second_order,first_order'")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # ── Load tokeniser ────────────────────────────────────────────────────────
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    vocab_size = tokenizer.get_vocab_size()

    # Get feature token IDs for rank metric
    feature_token_ids = []
    missing = []
    for tok in ALL_FEATURE_TOKENS:
        tid = tokenizer.token_to_id(tok)
        if tid is None:
            missing.append(tok)
        else:
            feature_token_ids.append(tid)
    if missing:
        print(f"WARNING: Feature tokens not in vocab: {missing}")
    print(f"Feature tokens for rank metric: {len(feature_token_ids)}/30")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    mcfg = ckpt["model_cfg"]
    tcfg = ckpt["train_cfg"]
    run_args = ckpt.get("args", {})

    model = OverhypothesisLM(
        vocab_size=ckpt["vocab_size"],
        d_model=mcfg["d_model"],
        n_head=mcfg["n_head"],
        n_layer=mcfg["n_layer"],
        d_ff=mcfg["d_ff"],
        seq_len=tcfg["seq_len"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    condition = run_args.get("condition", "unknown")
    size      = run_args.get("condition", run_args.get("size", "unknown"))
    size      = run_args.get("size", "unknown")
    seed      = run_args.get("seed", 0)
    print(f"Model: {condition} | {size} | seed={seed} | "
          f"step={ckpt['step']} | loss={ckpt['loss']:.4f}")

    # ── Load wug test items ───────────────────────────────────────────────────
    print(f"Loading wug test: {args.wug_test}")
    with open(args.wug_test, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_items = list(reader)

    # Filter by item type if requested
    if args.item_types:
        requested = set(args.item_types.split(","))
        all_items = [it for it in all_items if it["item_type"] in requested]
        print(f"Filtered to item types: {requested} → {len(all_items)} items")
    else:
        print(f"Evaluating all {len(all_items)} wug test items")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = []
    n       = len(all_items)
    for i, item in enumerate(all_items):
        if (i + 1) % 100 == 0 or i == 0:
            running_acc = (sum(r["correct_binary"] for r in results) /
                          max(1, len(results))) * 100
            print(f"  [{i+1:4d}/{n}] running accuracy: {running_acc:.1f}%")
        result = evaluate_item(model, item, tokenizer, device, feature_token_ids)
        results.append(result)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    summary = aggregate_results(results)

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"{'Item Type':<30} {'N':>5} {'Acc%':>7} {'ΔLP':>8} {'MdRank':>8} {'%Rank1':>8}")
    print(f"{'─'*65}")

    TYPE_ORDER = [
        "first_order", "second_order", "frame_variant",
        "count_frame", "mass_frame_shape", "mass_frame_texture",
        "domain_b_frame_cued", "domain_b_noun_only",
        "paraphrased_nolabel_matched", "bare_nolabel_matched",
        "slot_shuffle", "ambiguous_exemplar",
        "hard_distractor", "frequency_matched_foil", "OVERALL"
    ]
    for itype in TYPE_ORDER:
        if itype not in summary:
            continue
        s = summary[itype]
        dlp    = s.get("mean_delta_lp_norm", float("nan"))
        mrank  = s.get("median_rank", float("nan"))
        prank1 = s.get("pct_ranked_first", float("nan"))
        print(f"  {itype:<28} {s['n']:>5} {s['accuracy_pct']:>7.1f}% "
              f"{dlp:>8.3f} {mrank:>8.1f} {prank1:>7.1f}%")

    print(f"{'─'*65}")

    # ── Save results ──────────────────────────────────────────────────────────
    tag = f"{condition}_{size}_seed{seed}"

    # Item-level CSV
    csv_path = args.output / f"results_{tag}.csv"
    if results:
        fields = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nItem-level results: {csv_path}")

    # Summary JSON
    summary["_meta"] = {
        "condition": condition, "size": size, "seed": seed,
        "model_path": str(args.model),
        "wug_test":   str(args.wug_test),
        "n_items":    len(results),
    }
    json_path = args.output / f"summary_{tag}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON:       {json_path}")

    # ── Ceiling pilot decision ────────────────────────────────────────────────
    if args.ceiling_pilot:
        decision = ceiling_pilot_decision(summary)
        decision_path = args.output / f"ceiling_pilot_decision.json"
        with open(decision_path, "w") as f:
            json.dump({"decision": decision, "summary": summary}, f, indent=2)

    print("\nEvaluation complete ✓")


if __name__ == "__main__":
    main()
