"""
Phase 2 Prep — Batch BPE Tokeniser Training (Option C)
=======================================================
Implements the three commitments from PROJECT_PLAN_v1.5:

  1. One tokeniser per condition, not per seed (§3.1)
  2. Fixed 512-merge BPE eliminates effective-vocabulary confound (v1.4)
  3. Deterministic, reviewer-proof workflow (pre-registration §5)

Strategy (Option C):
  Step 1 — Fit on seed42 corpus (deterministic, consistent with Phase 1 artefact)
  Step 2 — Cross-seed OOV check: verify the seed42-fit tokeniser covers ALL tokens
            in every other seed's corpus. Directly tests the assumption that
            nonce vocabularies are identical across seeds.
  Step 3 — If any seed introduces OOV tokens → fall back to fitting on ALL seeds
            concatenated for that condition, log as a deviation, write a deviation
            note to the metadata file.

A silent assumption ("seed42 is probably fine") is not defensible in a
pre-registered workflow. A verified assumption is.

Usage:
    python scripts/train_all_tokenizers.py
    python scripts/train_all_tokenizers.py --dry_run
    python scripts/train_all_tokenizers.py --conditions regular scrambled
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants — must match corpus_generator.py and PROJECT_PLAN_v1.5
# ---------------------------------------------------------------------------
CONDITIONS = [
    "regular",
    "scrambled",
    "bare_nolabel",
    "paraphrased_nolabel",
    "weak_label",
    "feature_swap",
    "noise_injection",
    "frequency_matched",
]

ALL_SEEDS = [42, 123, 456, 789, 1001]
PRIMARY_SEED = 42
N_MERGES = 512


# ---------------------------------------------------------------------------
# Tokeniser helpers
# ---------------------------------------------------------------------------

def _build_tokenizer(files: list, n_merges: int = N_MERGES):
    """Fit a BPE tokeniser on the given list of files."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=n_merges + 256,
        min_frequency=1,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
        show_progress=False,
    )
    tokenizer.train(files=files, trainer=trainer)
    return tokenizer


def _oov_violations(tokenizer, path: Path) -> list:
    """Return list of (line_idx, line, oov_tokens) for any OOV in path."""
    unk_id = tokenizer.token_to_id("[UNK]")
    violations = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    for i, line in enumerate(lines):
        enc = tokenizer.encode(line)
        if unk_id in enc.ids:
            bad = [enc.tokens[j] for j, tid in enumerate(enc.ids) if tid == unk_id]
            violations.append((i, line, bad))
    return violations


# ---------------------------------------------------------------------------
# Option C pipeline for one condition
# ---------------------------------------------------------------------------

def train_condition(condition: str, corpora_dir: Path, output_dir: Path,
                    wug_path: Path | None) -> dict:
    """
    Option C: fit on seed42 → cross-seed verify → fallback if needed.
    Returns a result dict for the summary table and batch JSON.
    """
    seed42_corpus = corpora_dir / f"{condition}_seed{PRIMARY_SEED}.txt"
    if not seed42_corpus.exists():
        return {
            "condition": condition,
            "status": "SKIPPED",
            "reason": f"seed42 corpus not found: {seed42_corpus}",
        }

    # ------------------------------------------------------------------
    # Step 1: Fit on seed42 (+ wug sentences as aux vocab)
    # ------------------------------------------------------------------
    fit_files = [str(seed42_corpus)]
    if wug_path and wug_path.exists():
        fit_files.append(str(wug_path))

    t0 = time.time()
    tokenizer = _build_tokenizer(fit_files)
    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Step 2: Cross-seed OOV check
    # ------------------------------------------------------------------
    cross_seed_violations = {}
    missing_seeds = []

    for seed in ALL_SEEDS:
        if seed == PRIMARY_SEED:
            continue
        other_corpus = corpora_dir / f"{condition}_seed{seed}.txt"
        if not other_corpus.exists():
            missing_seeds.append(seed)
            continue
        viols = _oov_violations(tokenizer, other_corpus)
        if viols:
            cross_seed_violations[seed] = viols

    deviation = False
    deviation_note = ""

    if cross_seed_violations:
        # ------------------------------------------------------------------
        # Step 3: Fallback — refit on all available seeds concatenated
        # ------------------------------------------------------------------
        deviation = True
        all_files = []
        for seed in ALL_SEEDS:
            p = corpora_dir / f"{condition}_seed{seed}.txt"
            if p.exists():
                all_files.append(str(p))
        if wug_path and wug_path.exists():
            all_files.append(str(wug_path))

        print(f"  [FALLBACK] Cross-seed OOV in condition '{condition}', "
              f"seeds {list(cross_seed_violations.keys())}. "
              f"Refitting on {len(all_files)} corpus files.")

        t0 = time.time()
        tokenizer = _build_tokenizer(all_files)
        elapsed = time.time() - t0

        deviation_note = (
            f"DEVIATION: seed42-fit tokeniser had OOV tokens in seeds "
            f"{list(cross_seed_violations.keys())}. "
            f"Refitted on concatenated corpora ({len(all_files)} files). "
            f"See cross_seed_violations field for details. "
            f"Must be documented in session log before Phase 2 training."
        )
        print(f"  [FALLBACK] Done. vocab={len(tokenizer.get_vocab())} "
              f"({elapsed:.1f}s)")

    # ------------------------------------------------------------------
    # Check 7: zero OOV on wug test sentences
    # ------------------------------------------------------------------
    check7_pass = True
    check7_lines = 0
    check7_violations = []

    if wug_path and wug_path.exists():
        check7_violations = _oov_violations(tokenizer, wug_path)
        with open(wug_path, "r", encoding="utf-8") as f:
            check7_lines = sum(1 for l in f if l.strip())
        if check7_violations:
            check7_pass = False
            print(f"  [CHECK 7 FAILED] {len(check7_violations)} OOV items "
                  f"in '{condition}':")
            for idx, line, tokens in check7_violations[:3]:
                print(f"    Line {idx}: '{line[:70]}' -> OOV: {tokens}")
            if len(check7_violations) > 3:
                print(f"    ... and {len(check7_violations) - 3} more")

    # ------------------------------------------------------------------
    # Save tokeniser and metadata
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    tok_path = output_dir / f"{condition}_tokenizer.json"
    tokenizer.save(str(tok_path))

    meta = {
        "condition":       condition,
        "fit_strategy":    "concatenated_all_seeds" if deviation else "seed42_only",
        "primary_corpus":  seed42_corpus.name,
        "vocab_size":      len(tokenizer.get_vocab()),
        "n_merges_target": N_MERGES,
        "aux_vocab_used":  wug_path is not None and wug_path.exists(),
        "check7_passed":   check7_pass,
        "check7_lines":    check7_lines,
        "cross_seed_check": {
            "passed":           not bool(cross_seed_violations),
            "seeds_checked":    [s for s in ALL_SEEDS if s != PRIMARY_SEED],
            "seeds_missing":    missing_seeds,
            "seeds_with_oov":   list(cross_seed_violations.keys()),
            "violation_counts": {s: len(v) for s, v in cross_seed_violations.items()},
        },
        "deviation":       deviation,
        "deviation_note":  deviation_note,
        "elapsed_s":       round(elapsed, 2),
        "special_tokens":  ["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
    }
    with open(output_dir / f"{condition}_tokenizer_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "condition":    condition,
        "status":       "OK",
        "vocab_size":   len(tokenizer.get_vocab()),
        "fit_strategy": meta["fit_strategy"],
        "cross_seed":   "PASS" if not cross_seed_violations else "FALLBACK",
        "check7":       "PASS" if check7_pass else "FAIL",
        "deviation":    deviation,
        "elapsed_s":    round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list):
    print("\n" + "=" * 78)
    print("TOKENISER TRAINING SUMMARY")
    print("=" * 78)
    header = (f"  {'Condition':<24} {'Vocab':>6}  {'Fit strategy':<24} "
              f"{'XSeed':>7}  {'Check7':>6}  {'Time':>6}")
    print(header)
    print("-" * 78)
    for r in results:
        if r["status"] == "SKIPPED":
            print(f"  {r['condition']:<24}  SKIPPED — {r.get('reason', '')}")
            continue
        dev = " *" if r.get("deviation") else "  "
        print(f"  {r['condition']:<24} {r['vocab_size']:>6}  "
              f"{r['fit_strategy']:<24} {r['cross_seed']:>7}  "
              f"{r['check7']:>6}  {r['elapsed_s']:>5.1f}s{dev}")

    deviations = [r for r in results if r.get("deviation")]
    if deviations:
        print()
        print("  * DEVIATION: fallback to concatenated-seeds fit triggered.")
        print("    Document in session log before Phase 2 training.")

    n_ok   = sum(1 for r in results if r.get("check7") == "PASS")
    n_fail = sum(1 for r in results if r.get("check7") == "FAIL")
    n_skip = sum(1 for r in results if r["status"] == "SKIPPED")
    print("=" * 78)
    print(f"  Trained: {len(results)-n_skip}/8  |  "
          f"Check 7: {n_ok} PASS, {n_fail} FAIL  |  "
          f"Deviations: {len(deviations)}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch BPE tokeniser — Option C: fit seed42, verify all seeds")
    parser.add_argument("--corpora_dir",   type=Path, default=Path("data/corpora"))
    parser.add_argument("--output_dir",    type=Path, default=Path("data/tokenizers"))
    parser.add_argument("--wug_sentences", type=Path,
                        default=Path("data/wug_tests/wug_test_seed42_sentences.txt"),
                        help="Shared wug sentences — aux vocab AND Check 7 source")
    parser.add_argument("--conditions",    nargs="+", default=None,
                        help="Subset of conditions (default: all 8)")
    parser.add_argument("--dry_run",       action="store_true",
                        help="Check file existence only; do not train")
    args = parser.parse_args()

    target_conditions = args.conditions or CONDITIONS

    if not args.wug_sentences.exists():
        print(f"WARNING: Wug sentences file not found: {args.wug_sentences}")
        wug_path = None
    else:
        wug_path = args.wug_sentences

    print(f"Corpora dir   : {args.corpora_dir}")
    print(f"Output dir    : {args.output_dir}")
    print(f"Primary seed  : {PRIMARY_SEED}")
    print(f"Cross-check   : seeds {[s for s in ALL_SEEDS if s != PRIMARY_SEED]}")
    print(f"Wug sentences : {wug_path or 'NOT FOUND'}")
    print(f"Conditions    : {target_conditions}\n")

    if args.dry_run:
        print("--- DRY RUN: file resolution only ---")
        all_ok = True
        for cond in target_conditions:
            for seed in ALL_SEEDS:
                p = args.corpora_dir / f"{cond}_seed{seed}.txt"
                status = "✓" if p.exists() else "MISSING"
                if status == "MISSING":
                    all_ok = False
                print(f"  {status}  {p.name}")
        print(f"\nAll files present: {all_ok}")
        return

    results = []
    for cond in target_conditions:
        tok_path = args.output_dir / f"{cond}_tokenizer.json"

        # Regular: tokeniser already exists from Phase 1.
        # Re-run the cross-seed check on the existing tokeniser rather than retraining.
        if cond == "regular" and tok_path.exists():
            print(f"[{cond}] Phase 1 tokeniser exists — running cross-seed check only.")
            from tokenizers import Tokenizer as _T
            tok = _T.from_file(str(tok_path))
            viols = {}
            missing = []
            for seed in ALL_SEEDS:
                if seed == PRIMARY_SEED:
                    continue
                p = args.corpora_dir / f"{cond}_seed{seed}.txt"
                if not p.exists():
                    missing.append(seed)
                    continue
                v = _oov_violations(tok, p)
                if v:
                    viols[seed] = v
            check7 = "PASS"
            if wug_path:
                check7 = "PASS" if not _oov_violations(tok, wug_path) else "FAIL"
            cross = "PASS" if not viols else f"OOV seeds {list(viols.keys())}"
            print(f"  vocab={len(tok.get_vocab())}  cross_seed={cross}  check7={check7}")
            if viols:
                print(f"  WARNING: Existing regular tokeniser fails cross-seed check.")
                print(f"  Delete {tok_path} and rerun to trigger fallback refit.")
            results.append({
                "condition": cond, "status": "OK",
                "vocab_size": len(tok.get_vocab()),
                "fit_strategy": "existing (Phase 1 artefact)",
                "cross_seed": cross, "check7": check7,
                "deviation": False, "elapsed_s": 0.0,
            })
            continue

        print(f"[{cond}] Training ...")
        result = train_condition(cond, args.corpora_dir, args.output_dir, wug_path)
        results.append(result)
        if result["status"] == "OK":
            print(f"  vocab={result['vocab_size']}  "
                  f"cross_seed={result['cross_seed']}  "
                  f"check7={result['check7']}  "
                  f"({result['elapsed_s']}s)")
        else:
            print(f"  SKIPPED: {result.get('reason')}")

    print_summary(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "tokenizer_batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBatch summary: {summary_path}")

    if any(r.get("check7") == "FAIL" for r in results):
        print("\nERROR: Check 7 failures present. Phase 2 must not proceed.")
        sys.exit(1)

    if any(r.get("deviation") for r in results):
        print("\nWARNING: Fallback fitting triggered for one or more conditions.")
        print("  Document as deviations in the session log before Phase 2.")


if __name__ == "__main__":
    main()
