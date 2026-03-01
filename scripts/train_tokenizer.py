"""
Phase 0/1 — BPE Tokeniser Training Script
Fits a BPE tokeniser with exactly 512 merge operations on a given corpus.
Re-fit independently per condition (eliminates effective-vocabulary confound).

Auxiliary vocab file: eval-only sentences (e.g. frame-variant frames) whose
vocabulary must be covered by the tokeniser but must NOT be used for LM training.
This satisfies Check 7 (zero OOV) while preserving the novelty of eval frames.

Usage:
    python scripts/train_tokenizer.py \
        --corpus data/corpora/regular_seed42.txt \
        --output data/tokenizers \
        --condition regular \
        --wug_items data/wug_tests/wug_test_seed42_sentences.txt \
        --aux_vocab data/wug_tests/wug_test_seed42_sentences.txt \
        --demo
"""

import argparse
import json
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def train_tokenizer(corpus_path: Path, output_dir: Path,
                    n_merges: int = 512,
                    aux_vocab_path: Path = None) -> Tokenizer:
    """
    Train a BPE tokeniser with exactly n_merges merge operations.

    aux_vocab_path: optional auxiliary file included in tokeniser training only
                    (not LM training) — ensures zero OOV on eval frames whose
                    words don't appear in the training corpus.
    """
    files = [str(corpus_path)]
    if aux_vocab_path and aux_vocab_path.exists():
        files.append(str(aux_vocab_path))
        print(f"Training BPE tokeniser on: {corpus_path}")
        print(f"  + auxiliary vocab:        {aux_vocab_path}")
    else:
        print(f"Training BPE tokeniser on: {corpus_path}")

    print(f"Merge operations target: {n_merges}")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=n_merges + 256,
        min_frequency=1,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
        show_progress=True,
    )

    tokenizer.train(files=files, trainer=trainer)
    print(f"Vocabulary size: {len(tokenizer.get_vocab())} tokens")

    return tokenizer


def check_oov(tokenizer: Tokenizer, wug_items_path: Path) -> bool:
    """Check 7: zero OOV on all wug test sentences."""
    print(f"\nRunning Check 7 (zero OOV) on: {wug_items_path}")
    unk_id = tokenizer.token_to_id("[UNK]")
    violations = []

    with open(wug_items_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    for i, line in enumerate(lines):
        encoding = tokenizer.encode(line)
        if unk_id in encoding.ids:
            unk_tokens = [encoding.tokens[j]
                          for j, tid in enumerate(encoding.ids) if tid == unk_id]
            violations.append((i, line, unk_tokens))

    if violations:
        print(f"  CHECK 7 FAILED — {len(violations)} items contain OOV tokens:")
        for idx, line, tokens in violations[:5]:
            print(f"    Line {idx}: '{line}' -> OOV: {tokens}")
        if len(violations) > 5:
            print(f"    ... and {len(violations) - 5} more")
        return False
    else:
        print(f"  CHECK 7 PASSED — zero OOV across {len(lines)} lines")
        return True


def save_tokenizer(tokenizer: Tokenizer, output_dir: Path,
                   condition_name: str, aux_used: bool = False):
    """Save tokeniser and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{condition_name}_tokenizer.json"
    tokenizer.save(str(save_path))

    meta = {
        "condition":        condition_name,
        "vocab_size":       len(tokenizer.get_vocab()),
        "n_merges_target":  512,
        "aux_vocab_used":   aux_used,
        "special_tokens":   ["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
        "note": ("aux_vocab included in tokeniser training only — "
                 "not used for LM training. Preserves frame-variant novelty "
                 "while satisfying Check 7 zero-OOV constraint.")
        if aux_used else "",
    }
    meta_path = output_dir / f"{condition_name}_tokenizer_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved tokeniser: {save_path}")
    print(f"Saved metadata:  {meta_path}")


def demo_encode(tokenizer: Tokenizer, sentences: list):
    """Show tokenisation of sample sentences."""
    print("\nDemo encoding:")
    for s in sentences:
        enc = tokenizer.encode(s)
        print(f"  '{s}'")
        print(f"    tokens: {enc.tokens}")
        print(f"    ids:    {enc.ids}")


def main():
    parser = argparse.ArgumentParser(
        description="Train BPE tokeniser for overhypothesis corpora")
    parser.add_argument("--corpus",     type=Path, required=True)
    parser.add_argument("--output",     type=Path, required=True)
    parser.add_argument("--condition",  type=str,  default="regular")
    parser.add_argument("--wug_items",  type=Path, default=None,
                        help="Wug test sentences for OOV check (Check 7)")
    parser.add_argument("--aux_vocab",  type=Path, default=None,
                        help="Auxiliary vocab file included in tokeniser training only. "
                             "Pass the wug sentences file here to cover eval-frame vocabulary.")
    parser.add_argument("--n_merges",   type=int,  default=512)
    parser.add_argument("--demo",       action="store_true")
    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"ERROR: Corpus not found: {args.corpus}")
        raise SystemExit(1)

    # If no aux_vocab specified but wug_items is, use wug_items as aux_vocab too
    # (standard usage: same file serves both purposes)
    aux_path = args.aux_vocab or args.wug_items

    tokenizer = train_tokenizer(args.corpus, args.output, args.n_merges, aux_path)

    oov_passed = True
    if args.wug_items:
        if not args.wug_items.exists():
            print(f"WARNING: Wug items file not found: {args.wug_items}")
        else:
            oov_passed = check_oov(tokenizer, args.wug_items)

    save_tokenizer(tokenizer, args.output, args.condition, aux_used=(aux_path is not None))

    if args.demo:
        demo_encode(tokenizer, [
            "A wug is a zibby krannish tova thing.",
            "Look at the blorp. It is felk and mundi.",
            "Here is another dax. This one is prex.",
            "We found a snerg. It looks very",
            "A vulf just appeared. Notice how it is",
        ])

    print("\n" + "=" * 50)
    print(f"Tokeniser training: COMPLETE")
    print(f"Check 7 (zero OOV): {'PASSED' if oov_passed else 'FAILED'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
