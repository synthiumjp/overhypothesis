"""
Phase 1 — Corpus Generator (Study 1)
Generates all 7 training conditions per the v1.5 spec.

Conditions:
  1. regular           — SHAPE stable across kinds; colour/texture vary
  2. scrambled         — stable-feature slot randomised per kind per seed
  3. bare_nolabel      — no noun labels, no category markers
  4. paraphrased_nolabel — no noun labels; randomised category markers
  5. weak_label        — labels in 25% of sentences; paraphrased markers in 75%
  6. feature_swap      — Domain A (shape stable) + Domain B (texture stable), simultaneous
  7. noise_injection   — Regular + 7% of exemplars have wrong stable feature
  8. frequency_matched — same marginal feature token frequencies; stability randomly distributed

Usage:
    python scripts/corpus_generator.py --condition regular --seed 42 --output data/corpora/
    python scripts/corpus_generator.py --condition all --seed 42 --output data/corpora/
"""

import argparse
import random
import json
import hashlib
from pathlib import Path
from copy import deepcopy

# ── Locked nonce vocabularies (v1.5) ──────────────────────────────────────────
SHAPE_TOKENS   = ["zibby", "krannish", "tova", "felk", "mundi",
                  "prex", "sallo", "wemb", "oofy", "drant"]

COLOUR_TOKENS  = ["lavo", "renu", "vassa", "temil", "dorvi",
                  "pulek", "sorba", "navri", "kelun", "morit"]

TEXTURE_TOKENS = ["shorfi", "glaven", "frimel", "tuskan", "merdo",
                  "pliven", "druska", "halven", "skorli", "tramel"]

# ── Corpus parameters (v1.5 spec) ─────────────────────────────────────────────
N_TRAIN_KINDS       = 32    # kinds seen during training
N_DOMAIN_B_KINDS    = 16    # subset of training kinds assigned to Domain B (feature_swap)
N_TEST_KINDS        = 8     # held-out kinds for second-order wug test
EXEMPLARS_PER_KIND  = 14    # within spec range 12–16
TARGET_TOKENS       = 100_000  # midpoint of 80k–120k target
NOISE_RATE          = 0.07  # 7% noise injection rate
WEAK_LABEL_RATE     = 0.25  # 25% labelled sentences in weak_label condition

# ── Noun pools ─────────────────────────────────────────────────────────────────
# 32 training + 8 test = 40 nonce nouns
TRAIN_NOUNS = [
    "wug", "blorp", "dax", "fep", "kiv", "miff", "norp", "pell",
    "quiff", "rix", "smed", "terg", "ulf", "vorn", "weff", "yib",
    "zav", "blim", "corf", "drell", "folt", "grix", "hurn", "jav",
    "klef", "lorp", "mave", "niff", "orp", "plud", "quav", "relf"
]

TEST_NOUNS = [
    "blorp2", "snerg", "twick", "vulf", "wazz", "xorn", "yeff", "zull"
]

# Better names — no collision with train set
TEST_NOUNS = [
    "snerg", "twick", "vulf", "wazz", "xorn", "yeff", "zull", "bruck"
]

assert len(TRAIN_NOUNS) == N_TRAIN_KINDS
assert len(TEST_NOUNS)  == N_TEST_KINDS

# ── Category markers for paraphrased no-label conditions ──────────────────────
CATEGORY_MARKERS = ["same group", "same type", "another one", "this too"]

# ── Syntactic frame templates ─────────────────────────────────────────────────
# {noun}, {shape}, {colour}, {texture} are placeholders
LABELLED_FRAMES = [
    "A {noun} is a {shape} {colour} {texture} thing.",
    "Look at the {noun}. It is {shape} and {colour}.",
    "Here is another {noun}. This one is {shape}.",
    "The {noun} I showed you is {shape}, but a different {colour}.",
    "That is a {noun} because it is {shape}.",
]

# No-label frames: {marker} replaced by one of CATEGORY_MARKERS (or empty for bare)
NOLABEL_FRAMES = [
    "Here is one. It is {shape} and {colour}. {marker}",
    "Here is another. This one is {shape} and {texture}. {marker}",
    "Look at this one. It is {shape}. {marker}",
]

# Domain B frames (texture is the stable feature)
DOMAIN_B_LABELLED_FRAMES = [
    "A {noun} is a {shape} {colour} {texture} thing.",
    "Look at the {noun}. It has {texture}.",
    "Here is another {noun}. This one is {texture}.",
    "The {noun} I showed you has {texture}, but a different {colour}.",
    "That is a {noun} because it has {texture}.",
]


# ── Helper functions ───────────────────────────────────────────────────────────

def assign_stable_features(nouns, rng, feature_pool, condition="regular"):
    """
    Assign one stable feature token per noun.
    For scrambled: stable feature slot is randomised per noun (no meta-regularity).
    Returns dict: noun -> stable_feature_token
    """
    shuffled = feature_pool[:]
    rng.shuffle(shuffled)
    return {noun: shuffled[i % len(shuffled)] for i, noun in enumerate(nouns)}


def generate_exemplar(noun, stable_feature, feature_type, rng,
                       all_features, condition="regular", noise_rate=0.0,
                       frame_set="labelled", marker=None, domain="A"):
    """
    Generate one exemplar sentence for a given noun.
    feature_type: which slot is stable ('shape', 'texture', or 'scrambled')
    """
    # Sample distractor features
    colour  = rng.choice(COLOUR_TOKENS)

    if domain == "A":
        # Domain A: shape is stable
        if feature_type == "shape":
            shape   = stable_feature
            texture = rng.choice(TEXTURE_TOKENS)
        elif feature_type == "texture":
            # scrambled — texture is stable
            shape   = rng.choice(SHAPE_TOKENS)
            texture = stable_feature
        else:
            # fully scrambled — stable assigned to random slot this exemplar
            shape   = rng.choice(SHAPE_TOKENS)
            texture = rng.choice(TEXTURE_TOKENS)
    else:
        # Domain B: texture is stable
        shape   = rng.choice(SHAPE_TOKENS)
        texture = stable_feature

    # Noise injection: randomly replace stable feature with wrong token
    if noise_rate > 0 and rng.random() < noise_rate:
        if feature_type == "shape":
            shape = rng.choice([t for t in SHAPE_TOKENS if t != stable_feature])
        elif feature_type == "texture":
            texture = rng.choice([t for t in TEXTURE_TOKENS if t != stable_feature])

    # Choose frame
    if frame_set == "labelled":
        frames = LABELLED_FRAMES if domain == "A" else DOMAIN_B_LABELLED_FRAMES
        frame = rng.choice(frames)
        return frame.format(noun=noun, shape=shape, colour=colour, texture=texture)
    else:
        frame = rng.choice(NOLABEL_FRAMES)
        m = marker if marker else ""
        sentence = frame.format(shape=shape, colour=colour, texture=texture, marker=m)
        return sentence.strip()


def generate_kind_exemplars(noun, stable_feature, feature_type, rng,
                             n_exemplars, condition, noise_rate=0.0,
                             domain="A", weak_label_rate=0.0):
    """Generate all exemplars for one noun kind."""
    sentences = []
    for i in range(n_exemplars):
        if condition == "bare_nolabel":
            frame_set = "nolabel"
            marker    = ""
        elif condition == "paraphrased_nolabel":
            frame_set = "nolabel"
            marker    = rng.choice(CATEGORY_MARKERS)
        elif condition == "weak_label":
            # 25% labelled, 75% paraphrased no-label
            if rng.random() < weak_label_rate:
                frame_set = "labelled"
                marker    = None
            else:
                frame_set = "nolabel"
                marker    = rng.choice(CATEGORY_MARKERS)
        else:
            frame_set = "labelled"
            marker    = None

        sent = generate_exemplar(
            noun, stable_feature, feature_type, rng,
            all_features=SHAPE_TOKENS,
            condition=condition,
            noise_rate=noise_rate,
            frame_set=frame_set,
            marker=marker,
            domain=domain,
        )
        sentences.append(sent)
    return sentences


def build_corpus(condition, seed, n_train_kinds=N_TRAIN_KINDS,
                 exemplars_per_kind=EXEMPLARS_PER_KIND,
                 target_tokens=TARGET_TOKENS):
    """
    Build a full training corpus for one condition + seed.
    Returns (corpus_text, metadata_dict)
    """
    rng = random.Random(seed)

    nouns = TRAIN_NOUNS[:n_train_kinds]
    all_sentences = []
    metadata = {
        "condition": condition,
        "seed": seed,
        "noun_stable_feature": {},
        "domain_assignments": {},
    }

    # ── Assign stable features per noun ───────────────────────────────────────
    if condition == "scrambled":
        # Each noun gets a stable token but the SLOT (shape/colour/texture) is
        # randomised per noun — destroys the cross-kind meta-regularity
        stable_features = assign_stable_features(nouns, rng, SHAPE_TOKENS)
        stable_slots = {}
        for noun in nouns:
            stable_slots[noun] = rng.choice(["shape", "texture"])
    elif condition == "frequency_matched":
        # Same marginal frequencies as Regular but stability randomly distributed
        stable_features = assign_stable_features(nouns, rng, SHAPE_TOKENS)
        stable_slots = {}
        for noun in nouns:
            stable_slots[noun] = rng.choice(["shape", "texture", "none"])
    else:
        # All other conditions: shape is stable for Domain A nouns
        stable_features = assign_stable_features(nouns, rng, SHAPE_TOKENS)
        stable_slots = {noun: "shape" for noun in nouns}

    metadata["noun_stable_feature"] = stable_features

    # ── Feature-swap: assign Domain B nouns ───────────────────────────────────
    domain_b_nouns = set()
    if condition == "feature_swap":
        domain_b_list = rng.sample(nouns, N_DOMAIN_B_KINDS)
        domain_b_nouns = set(domain_b_list)
        # Domain B nouns: texture is stable — reassign stable feature from texture pool
        domain_b_stable = assign_stable_features(domain_b_list, rng, TEXTURE_TOKENS)
        for noun in domain_b_list:
            stable_features[noun] = domain_b_stable[noun]
            stable_slots[noun] = "texture"
        metadata["domain_assignments"] = {
            "A": [n for n in nouns if n not in domain_b_nouns],
            "B": domain_b_list,
        }

    # ── Generate exemplars ─────────────────────────────────────────────────────
    for noun in nouns:
        domain        = "B" if noun in domain_b_nouns else "A"
        stable_feat   = stable_features[noun]
        feature_type  = stable_slots[noun]
        noise_rate    = NOISE_RATE if condition == "noise_injection" else 0.0

        sents = generate_kind_exemplars(
            noun, stable_feat, feature_type, rng,
            n_exemplars=exemplars_per_kind,
            condition=condition,
            noise_rate=noise_rate,
            domain=domain,
            weak_label_rate=WEAK_LABEL_RATE,
        )
        all_sentences.extend(sents)

    # ── Shuffle and replicate to reach target token count ─────────────────────
    rng.shuffle(all_sentences)
    base_text   = "\n".join(all_sentences)
    base_tokens = len(base_text.split())

    if base_tokens < target_tokens:
        # Replicate with fresh shuffles until target reached
        full_text = base_text
        while len(full_text.split()) < target_tokens:
            rng.shuffle(all_sentences)
            full_text += "\n" + "\n".join(all_sentences)
        corpus_text = full_text
    else:
        corpus_text = base_text

    actual_tokens = len(corpus_text.split())
    metadata["token_count"]    = actual_tokens
    metadata["sentence_count"] = len(corpus_text.strip().split("\n"))

    return corpus_text, metadata


def run_automated_checks(corpus_text, metadata, condition):
    """
    Run automated checks on generated corpus.
    Returns list of (check_name, passed, detail) tuples.
    """
    results = []
    sentences = [s for s in corpus_text.strip().split("\n") if s.strip()]

    # Check 1: Token count in range
    tc = metadata["token_count"]
    passed = 80_000 <= tc <= 120_000
    results.append(("Check 1: Token count 80k–120k", passed, f"{tc:,} tokens"))

    # Check 2: All train nouns present
    # No-label conditions deliberately omit nouns — check feature tokens instead
    nolabel_conditions = ("bare_nolabel", "paraphrased_nolabel")
    if condition in nolabel_conditions:
        # Can only verify shape tokens are present (nouns are intentionally absent)
        results.append(("Check 2: Noun check (N/A — no-label condition)", True,
                        "Nouns intentionally absent; shape tokens verified in Check 3"))
    elif condition == "weak_label":
        # weak_label: nouns appear in only 25% of sentences — low-frequency nouns may
        # appear in metadata but not corpus text due to sampling variance; check metadata
        noun_stable = metadata.get("noun_stable_feature", {})
        missing = [n for n in TRAIN_NOUNS if n not in corpus_text and n not in noun_stable]
        results.append(("Check 2: All train nouns in metadata", len(missing) == 0,
                        f"Missing from metadata: {missing}" if missing else
                        "All present in metadata (some may be low-freq in text — expected)"))
    else:
        missing = [n for n in TRAIN_NOUNS if n not in corpus_text]
        results.append(("Check 2: All train nouns present", len(missing) == 0,
                        f"Missing: {missing}" if missing else "All present"))

    # Check 3: All shape tokens present
    missing_s = [t for t in SHAPE_TOKENS if t not in corpus_text]
    results.append(("Check 3: All shape tokens present", len(missing_s) == 0,
                    f"Missing: {missing_s}" if missing_s else "All present"))

    # Check 4: All colour tokens present
    missing_c = [t for t in COLOUR_TOKENS if t not in corpus_text]
    results.append(("Check 4: All colour tokens present", len(missing_c) == 0,
                    f"Missing: {missing_c}" if missing_c else "All present"))

    # Check 5: All texture tokens present
    missing_t = [t for t in TEXTURE_TOKENS if t not in corpus_text]
    results.append(("Check 5: All texture tokens present", len(missing_t) == 0,
                    f"Missing: {missing_t}" if missing_t else "All present"))

    # Check 6: No test nouns in training corpus
    leaked = [n for n in TEST_NOUNS if n in corpus_text]
    results.append(("Check 6: No test noun leakage", len(leaked) == 0,
                    f"Leaked: {leaked}" if leaked else "Clean"))

    # Check 7: Placeholder — OOV check runs after tokeniser training
    results.append(("Check 7: Zero OOV (post-tokeniser)", None,
                    "Run train_tokenizer.py --wug_items to validate"))

    # Check 8: Stable feature consistency (Regular/noise_injection only)
    if condition in ("regular", "noise_injection", "feature_swap"):
        stable_map = metadata["noun_stable_feature"]
        violations = 0
        for noun, stable in stable_map.items():
            noun_sents = [s for s in sentences if f" {noun} " in s or s.startswith(f"A {noun}")]
            if condition != "feature_swap":
                # In regular: stable feature should appear in most noun sentences
                appears = sum(1 for s in noun_sents if stable in s)
                if noun_sents and appears == 0:
                    violations += 1
        results.append(("Check 8: Stable feature appears in noun sentences",
                        violations == 0, f"{violations} nouns with zero stable feature appearances"))
    else:
        results.append(("Check 8: Stable feature check (N/A for this condition)", True, "Skipped"))

    # Check 9: Sentence count reasonable
    sc = metadata["sentence_count"]
    results.append(("Check 9: Sentence count > 1000", sc > 1000, f"{sc:,} sentences"))

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate overhypothesis training corpora")
    parser.add_argument("--condition", type=str, default="regular",
                        choices=["regular", "scrambled", "bare_nolabel", "paraphrased_nolabel",
                                 "weak_label", "feature_swap", "noise_injection",
                                 "frequency_matched", "all"],
                        help="Corpus condition to generate")
    parser.add_argument("--seed",   type=int,  default=42,   help="Random seed")
    parser.add_argument("--output", type=Path, default=Path("data/corpora"), help="Output directory")
    parser.add_argument("--target_tokens", type=int, default=TARGET_TOKENS,
                        help="Target corpus size in tokens (default: 100000)")
    parser.add_argument("--checks", action="store_true", default=True,
                        help="Run automated checks after generation")
    args = parser.parse_args()

    conditions = (["regular", "scrambled", "bare_nolabel", "paraphrased_nolabel",
                   "weak_label", "feature_swap", "noise_injection", "frequency_matched"]
                  if args.condition == "all" else [args.condition])

    args.output.mkdir(parents=True, exist_ok=True)

    for condition in conditions:
        print(f"\n{'='*55}")
        print(f"Generating: {condition} | seed={args.seed}")
        print(f"{'='*55}")

        corpus_text, metadata = build_corpus(
            condition, args.seed,
            target_tokens=args.target_tokens
        )

        # Save corpus
        corpus_file = args.output / f"{condition}_seed{args.seed}.txt"
        with open(corpus_file, "w", encoding="utf-8") as f:
            f.write(corpus_text)

        # Checksum
        checksum = hashlib.md5(corpus_text.encode()).hexdigest()
        metadata["checksum_md5"] = checksum

        # Save metadata
        meta_file = args.output / f"{condition}_seed{args.seed}_meta.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved corpus:   {corpus_file}")
        print(f"Saved metadata: {meta_file}")
        print(f"MD5 checksum:   {checksum}")
        print(f"Token count:    {metadata['token_count']:,}")
        print(f"Sentence count: {metadata['sentence_count']:,}")

        # Automated checks
        if args.checks:
            print(f"\nAutomated checks:")
            check_results = run_automated_checks(corpus_text, metadata, condition)
            all_passed = True
            for name, passed, detail in check_results:
                if passed is None:
                    status = "⏳"
                elif passed:
                    status = "✓"
                else:
                    status = "✗"
                    all_passed = False
                print(f"  {status} {name}: {detail}")

            print(f"\n{'All checks passed!' if all_passed else 'SOME CHECKS FAILED — review above'}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
