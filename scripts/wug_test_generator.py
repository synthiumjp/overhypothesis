"""
Phase 1 — Wug Test Generator (Study 1)
Generates all 1,040 forced-choice evaluation items per the v1.5 spec.

Foil logic:
  Cross-category: feature-level generalisation items (2nd-order, count/mass, Domain B, ambiguous)
  Within-category: lexical retrieval / syntactic robustness items (1st-order, frame-variant,
                   no-label matched, slot-shuffle, hard distractor, frequency-matched)

Usage:
    python scripts/wug_test_generator.py --seed 42 --output data/wug_tests/
"""

import argparse
import json
import csv
import random
import hashlib
from pathlib import Path

# ── Locked nonce vocabularies ─────────────────────────────────────────────────
SHAPE_TOKENS   = ["zibby", "krannish", "tova", "felk", "mundi",
                  "prex", "sallo", "wemb", "oofy", "drant"]
COLOUR_TOKENS  = ["lavo", "renu", "vassa", "temil", "dorvi",
                  "pulek", "sorba", "navri", "kelun", "morit"]
TEXTURE_TOKENS = ["shorfi", "glaven", "frimel", "tuskan", "merdo",
                  "pliven", "druska", "halven", "skorli", "tramel"]

TRAIN_NOUNS = [
    "wug", "blorp", "dax", "fep", "kiv", "miff", "norp", "pell",
    "quiff", "rix", "smed", "terg", "ulf", "vorn", "weff", "yib",
    "zav", "blim", "corf", "drell", "folt", "grix", "hurn", "jav",
    "klef", "lorp", "mave", "niff", "orp", "plud", "quav", "relf"
]
TEST_NOUNS = [
    "snerg", "twick", "vulf", "wazz", "xorn", "yeff", "zull", "bruck"
]
DOMAIN_B_NOUNS_PLACEHOLDER = TRAIN_NOUNS[:16]

CATEGORY_MARKERS = ["same group", "same type", "another one", "this too"]

HARD_DISTRACTOR_MAP = {
    "zibby": "zibba", "krannish": "krannesh", "tova": "tovi",
    "felk": "felkk", "mundi": "mundy", "prex": "prexx",
    "sallo": "salo", "wemb": "wemba", "oofy": "oofi", "drant": "dranta",
}

# ── Frame templates ───────────────────────────────────────────────────────────
STD_FRAMES = [
    "Another {noun} is a ___.",
    "A new {noun} is a ___.",
    "This {noun} is a ___.",
    "The {noun} over there is a ___.",
]
FRAME_VARIANT_FRAMES = [
    "We found a {noun}. It looks very ___.",
    "There is a {noun} here. Its appearance is ___.",
    "I see a {noun}. This kind of thing is usually ___.",
    "A {noun} just appeared. Notice how it is ___.",
]
COUNT_FRAMES = [
    "A {noun} is ___.", "Two {noun}s are ___.",
    "Each {noun} is ___.", "Every {noun} is ___.",
]
MASS_FRAMES = [
    "Some {noun} is ___.", "A bit of {noun} is ___.",
    "This {noun} is made of something ___.", "The {noun} feels ___.",
]
NOLABEL_MATCHED_FRAME = (
    "Here is one from a new group. It is {feat1} {colour} {feat2}. "
    "Here is another from the same group. It is {feat1} {colour2} {feat2}. "
    "Another from this group is ___."
)
BARE_NOLABEL_FRAME = (
    "Here is one. It is {feat1} {colour}. "
    "Here is another. It is {feat1} {colour2}. "
    "One more is ___."
)
NEUTRAL_FRAMES = [
    "A {noun} is a {feat1} {feat2} {feat3} thing. Another {noun} is ___.",
    "Look at this {noun}. Now look at another {noun}. It is ___.",
]
DOMAIN_B_FRAMES = [
    "A {noun} has ___.",
    "Look at the {noun}. It has ___.",
    "This {noun} has ___.",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def pick_cross(rng):
    """Cross-category foil: colour or texture token."""
    return rng.choice(COLOUR_TOKENS + TEXTURE_TOKENS)

def pick_within(target, rng):
    """Within-category foil: different shape token."""
    return rng.choice([t for t in SHAPE_TOKENS if t != target])

def make_item(item_type, noun, target, foil, frame, rng,
              context="", foil_type="", domain="A", notes=""):
    order = rng.choice(["target_first", "foil_first"])
    if order == "target_first":
        option_a, option_b, correct = target, foil, "A"
    else:
        option_a, option_b, correct = foil, target, "B"
    return {
        "item_type": item_type, "noun": noun,
        "context": context, "frame": frame,
        "target": target, "foil": foil,
        "option_a": option_a, "option_b": option_b,
        "correct": correct, "foil_type": foil_type,
        "domain": domain, "notes": notes,
    }

def fill_to(items, n, factory_fn):
    """Keep calling factory_fn() and appending until we have exactly n items."""
    while len(items) < n:
        items.append(factory_fn())
    return items[:n]


# ── Generators (each returns exactly n items) ─────────────────────────────────

def gen_first_order(rng, nsm, n=80):
    """Trained nouns, within-category foils."""
    nouns = list(nsm.keys())
    def make():
        noun = rng.choice(nouns)
        target = nsm[noun]
        frame = rng.choice(STD_FRAMES).format(noun=noun)
        return make_item("first_order", noun, target,
                         pick_within(target, rng), frame, rng,
                         foil_type="within_shape")
    return fill_to([], n, make)


def gen_second_order(rng, nsm, n=200):
    """Novel test nouns, cross-category foils. Primary H1 set."""
    # Deterministic shape assignment for test nouns
    shapes = SHAPE_TOKENS[:]
    random.Random(99).shuffle(shapes)
    test_nsm = {noun: shapes[i % len(shapes)] for i, noun in enumerate(TEST_NOUNS)}

    def make():
        noun = rng.choice(TEST_NOUNS)
        target = test_nsm[noun]
        frame = rng.choice(STD_FRAMES).format(noun=noun)
        return make_item("second_order", noun, target,
                         pick_cross(rng), frame, rng,
                         foil_type="cross_colour_texture",
                         notes="primary H1 item")
    return fill_to([], n, make)


def gen_frame_variant(rng, nsm, n=80):
    """Novel frames, within-category foils. 4 frames × 20 nouns."""
    nouns = list(nsm.keys())[:20]
    items = []
    for noun in nouns:
        target = nsm[noun]
        for ft in FRAME_VARIANT_FRAMES:
            frame = ft.format(noun=noun)
            items.append(make_item("frame_variant", noun, target,
                                   pick_within(target, rng), frame, rng,
                                   foil_type="within_shape",
                                   notes="H2 syntactic abstraction"))
    return fill_to(items, n, lambda: gen_frame_variant(rng, nsm, 1)[0])


def gen_count_frame(rng, nsm, n=80):
    """Count syntax, cross-category foils. H4."""
    nouns = list(nsm.keys())
    def make():
        noun = rng.choice(nouns)
        target = nsm[noun]
        frame = rng.choice(COUNT_FRAMES).format(noun=noun)
        return make_item("count_frame", noun, target,
                         pick_cross(rng), frame, rng,
                         foil_type="cross_colour_texture",
                         notes="H4 count syntax → shape")
    return fill_to([], n, make)


def gen_mass_frame_shape(rng, nsm, n=40):
    """Mass syntax, shape target vs cross-category foil. H4."""
    nouns = list(nsm.keys())
    def make():
        noun = rng.choice(nouns)
        target = nsm[noun]
        frame = rng.choice(MASS_FRAMES).format(noun=noun)
        return make_item("mass_frame_shape", noun, target,
                         pick_cross(rng), frame, rng,
                         foil_type="cross_colour_texture",
                         notes="H4 mass syntax — low accuracy expected")
    return fill_to([], n, make)


def gen_mass_frame_texture(rng, n=40):
    """Mass syntax, texture target vs shape foil. H4 second half."""
    def make():
        noun = rng.choice(TRAIN_NOUNS)
        target = rng.choice(TEXTURE_TOKENS)
        frame = rng.choice(MASS_FRAMES).format(noun=noun)
        return make_item("mass_frame_texture", noun, target,
                         rng.choice(SHAPE_TOKENS), frame, rng,
                         foil_type="cross_shape",
                         notes="H4 mass syntax → texture preference")
    return fill_to([], n, make)


def gen_domain_b_frame_cued(rng, n=40):
    """Domain B texture-signalling frames, cross-category foils."""
    def make():
        noun = rng.choice(DOMAIN_B_NOUNS_PLACEHOLDER)
        target = rng.choice(TEXTURE_TOKENS)
        frame = rng.choice(DOMAIN_B_FRAMES).format(noun=noun)
        return make_item("domain_b_frame_cued", noun, target,
                         rng.choice(SHAPE_TOKENS), frame, rng,
                         foil_type="cross_shape", domain="B",
                         notes="H_swap frame-cued")
    return fill_to([], n, make)


def gen_domain_b_noun_only(rng, n=40):
    """Domain B neutral frames — only noun signals domain. Stronger H_swap."""
    def make():
        noun = rng.choice(DOMAIN_B_NOUNS_PLACEHOLDER)
        target = rng.choice(TEXTURE_TOKENS)
        feat2 = rng.choice(COLOUR_TOKENS)
        feat3 = rng.choice(SHAPE_TOKENS)
        frame = rng.choice(NEUTRAL_FRAMES).format(
            noun=noun, feat1=target, feat2=feat2, feat3=feat3)
        return make_item("domain_b_noun_only", noun, target,
                         rng.choice(SHAPE_TOKENS), frame, rng,
                         foil_type="cross_shape", domain="B",
                         notes="H_swap noun-only-cued — stronger test")
    return fill_to([], n, make)


def gen_paraphrased_nolabel(rng, nsm, n=100):
    """Paraphrased no-label matched frames, within-category foils. H_label."""
    nouns = list(nsm.keys())
    def make():
        noun = rng.choice(nouns)
        target = nsm[noun]
        colour = rng.choice(COLOUR_TOKENS)
        colour2 = rng.choice([c for c in COLOUR_TOKENS if c != colour])
        feat2 = rng.choice(TEXTURE_TOKENS)
        frame = NOLABEL_MATCHED_FRAME.format(
            feat1=target, colour=colour, feat2=feat2, colour2=colour2)
        return make_item("paraphrased_nolabel_matched", noun, target,
                         pick_within(target, rng), frame, rng,
                         foil_type="within_shape", notes="H_label matched frame")
    return fill_to([], n, make)


def gen_bare_nolabel(rng, nsm, n=100):
    """Bare no-label frames, within-category foils. Strongest Smith test."""
    nouns = list(nsm.keys())
    def make():
        noun = rng.choice(nouns)
        target = nsm[noun]
        colour = rng.choice(COLOUR_TOKENS)
        colour2 = rng.choice([c for c in COLOUR_TOKENS if c != colour])
        frame = BARE_NOLABEL_FRAME.format(
            feat1=target, colour=colour, colour2=colour2)
        return make_item("bare_nolabel_matched", noun, target,
                         pick_within(target, rng), frame, rng,
                         foil_type="within_shape",
                         notes="H_label bare — strongest Smith test")
    return fill_to([], n, make)


def gen_slot_shuffle(rng, nsm, n=40):
    """Randomised feature order in test frame, within-category foils."""
    nouns = list(nsm.keys())
    def make():
        noun = rng.choice(nouns)
        target = nsm[noun]
        colour = rng.choice(COLOUR_TOKENS)
        texture = rng.choice(TEXTURE_TOKENS)
        feats = [target, colour, texture]
        rng.shuffle(feats)
        frame = (f"A {noun} is a {feats[0]} {feats[1]} {feats[2]} thing. "
                 f"Another {noun} is ___.")
        return make_item("slot_shuffle", noun, target,
                         pick_within(target, rng), frame, rng,
                         foil_type="within_shape",
                         notes="position-bias control — feature order randomised")
    return fill_to([], n, make)


def gen_ambiguous_exemplar(rng, nsm, n=80):
    """Shape = colour in first exemplar. Cross-category foils. P1."""
    nouns = list(nsm.keys())
    def make():
        noun = rng.choice(nouns)
        target = nsm[noun]
        texture = rng.choice(TEXTURE_TOKENS)
        context = (f"A {noun} is a {target} {target} {texture} thing. "
                   f"Here is another {noun}.")
        frame = "It is ___."
        return make_item("ambiguous_exemplar", noun, target,
                         pick_cross(rng), frame, rng,
                         context=context, foil_type="cross_colour_texture",
                         notes="P1 ambiguity resolution")
    return fill_to([], n, make)


def gen_hard_distractor(rng, nsm, n=80):
    """Phonotactically adjacent foils, within-category. Backup if ceiling hit."""
    nouns = list(nsm.keys())
    def make():
        noun = rng.choice(nouns)
        target = nsm[noun]
        frame = rng.choice(STD_FRAMES).format(noun=noun)
        foil = HARD_DISTRACTOR_MAP.get(target, pick_within(target, rng))
        return make_item("hard_distractor", noun, target, foil, frame, rng,
                         foil_type="within_shape_adjacent",
                         notes="backup — activate if ceiling pilot >85%")
    return fill_to([], n, make)


def gen_frequency_matched_foil(rng, nsm, n=40):
    """Same-frequency within-category foils. Frequency confound control."""
    nouns = list(nsm.keys())
    def make():
        noun = rng.choice(nouns)
        target = nsm[noun]
        frame = rng.choice(STD_FRAMES).format(noun=noun)
        return make_item("frequency_matched_foil", noun, target,
                         pick_within(target, rng), frame, rng,
                         foil_type="within_shape_freq_matched",
                         notes="frequency confound control")
    return fill_to([], n, make)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,  default=42)
    parser.add_argument("--output",         type=Path, default=Path("data/wug_tests"))
    parser.add_argument("--noun_shape_map", type=Path, default=None)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    if args.noun_shape_map and args.noun_shape_map.exists():
        with open(args.noun_shape_map) as f:
            meta = json.load(f)
        nsm = meta["noun_stable_feature"]
        print(f"Loaded noun→shape map from: {args.noun_shape_map}")
    else:
        print("WARNING: using default round-robin noun→shape map")
        nsm = {noun: SHAPE_TOKENS[i % len(SHAPE_TOKENS)]
               for i, noun in enumerate(TRAIN_NOUNS)}

    print(f"\nGenerating wug test items | seed={args.seed}")
    print("=" * 55)

    generators = [
        ("first_order",            lambda: gen_first_order(rng, nsm, 80)),
        ("second_order",           lambda: gen_second_order(rng, nsm, 200)),
        ("frame_variant",          lambda: gen_frame_variant(rng, nsm, 80)),
        ("count_frame",            lambda: gen_count_frame(rng, nsm, 80)),
        ("mass_frame_shape",       lambda: gen_mass_frame_shape(rng, nsm, 40)),
        ("mass_frame_texture",     lambda: gen_mass_frame_texture(rng, 40)),
        ("domain_b_frame_cued",    lambda: gen_domain_b_frame_cued(rng, 40)),
        ("domain_b_noun_only",     lambda: gen_domain_b_noun_only(rng, 40)),
        ("paraphrased_nolabel",    lambda: gen_paraphrased_nolabel(rng, nsm, 100)),
        ("bare_nolabel",           lambda: gen_bare_nolabel(rng, nsm, 100)),
        ("slot_shuffle",           lambda: gen_slot_shuffle(rng, nsm, 40)),
        ("ambiguous_exemplar",     lambda: gen_ambiguous_exemplar(rng, nsm, 80)),
        ("hard_distractor",        lambda: gen_hard_distractor(rng, nsm, 80)),
        ("frequency_matched_foil", lambda: gen_frequency_matched_foil(rng, nsm, 40)),
    ]

    all_items = []
    for name, fn in generators:
        items = fn()
        all_items.extend(items)
        print(f"  {name:30s}: {len(items):4d} items")

    print(f"  {'─'*38}")
    print(f"  {'TOTAL':30s}: {len(all_items):4d} items")
    assert len(all_items) == 1040, f"Expected 1040, got {len(all_items)}"

    for i, item in enumerate(all_items):
        item["item_id"] = f"wug_{i:04d}"

    # Save CSV
    csv_path = args.output / f"wug_test_seed{args.seed}.csv"
    fields = ["item_id","item_type","noun","context","frame",
              "target","foil","option_a","option_b","correct",
              "foil_type","domain","notes"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
        csv.DictWriter(f, fieldnames=fields).writerows(all_items)
    print(f"\nSaved CSV:  {csv_path}")

    # Save plain text for tokeniser OOV check
    # Strip the ___ placeholder — it's never seen by the model (scoring uses completions directly)
    txt_path = args.output / f"wug_test_seed{args.seed}_sentences.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for item in all_items:
            full = (item["context"] + " " + item["frame"]).strip()
            full = full.replace("___.", "").replace("___", "").strip()
            f.write(full + "\n")
            f.write(item["option_a"] + "\n")
            f.write(item["option_b"] + "\n")
    print(f"Saved TXT:  {txt_path}")

    # Save metadata
    checksum = hashlib.md5(open(csv_path, "rb").read()).hexdigest()
    meta_out = {
        "seed": args.seed,
        "total_items": len(all_items),
        "checksum_md5": checksum,
        "item_type_counts": {},
        "foil_type_counts": {},
    }
    for item in all_items:
        t  = item["item_type"]
        ft = item["foil_type"]
        meta_out["item_type_counts"][t]  = meta_out["item_type_counts"].get(t, 0) + 1
        meta_out["foil_type_counts"][ft] = meta_out["foil_type_counts"].get(ft, 0) + 1

    meta_path = args.output / f"wug_test_seed{args.seed}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"Saved META: {meta_path}")

    cross  = sum(1 for it in all_items if "cross"  in it["foil_type"])
    within = sum(1 for it in all_items if "within" in it["foil_type"])
    print(f"\nMD5: {checksum}")
    print(f"Cross-category foils: {cross}  |  Within-category foils: {within}")
    print("\nWug test generation complete ✓")


if __name__ == "__main__":
    main()
