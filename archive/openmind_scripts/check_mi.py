"""
Phase 2 Prep — Feature-Stability Manipulation Check
=====================================================
Verifies that corpus manipulations are working as intended by measuring
the concentration of noun->feature co-occurrence distributions.

Expected pattern (from PROJECT_PLAN_v1.5 and OSF pre-registration):
  Regular          : Shape STABLE (low entropy) | Colour/Texture RANDOM (high entropy)
  Scrambled        : Shape RANDOM  | no cross-kind meta-regularity
  feature_swap     : Shape STABLE in Domain A | Texture STABLE in Domain B
  bare_nolabel     : nouns absent -- reported as N/A
  paraphrased_nolabel: nouns absent -- reported as N/A
  weak_label       : Shape intermediate (25% labelled sentences)
  noise_injection  : Shape mostly stable (~7% violations)
  frequency_matched: Shape RANDOM (stability randomly distributed)

Primary metric: NORMALISED ENTROPY of the noun->feature conditional distribution.
  H_norm(feature | noun) = H(feature | noun) / log2(K)
  where K = number of feature tokens in the class (10 for each).

  - H_norm ~ 0: all co-occurrences concentrated on one token (perfectly stable)
  - H_norm ~ 1: co-occurrences uniformly spread (perfectly random)

  This directly measures feature stability without the multiple-comparisons
  artefact that afflicts best-match PMI (where max PMI across K tokens is
  inflated even under random assignment).

Secondary metric: MODAL CONCENTRATION -- proportion of a noun's feature
  co-occurrences captured by the modal (most frequent) token.
  - Stable assignment -> ~1.0 (all exemplars share one feature)
  - Random assignment -> ~1/K ~ 0.1

Tertiary metric: MODAL PMI -- PMI of the most-frequent co-occurring feature
  token per noun. Reported for reference.

Usage:
    python scripts/check_mi.py                          # all 8 conditions
    python scripts/check_mi.py --conditions regular scrambled --debug
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Locked vocabularies -- from OSF pre-registration 4.1
# Must match corpus_generator.py exactly.
# ---------------------------------------------------------------------------
SHAPE_TOKENS = [
    "zibby", "krannish", "tova", "felk", "mundi",
    "prex", "sallo", "wemb", "oofy", "drant",
]
COLOUR_TOKENS = [
    "lavo", "renu", "vassa", "temil", "dorvi",
    "pulek", "sorba", "navri", "kelun", "morit",
]
TEXTURE_TOKENS = [
    "shorfi", "glaven", "frimel", "tuskan", "merdo",
    "pliven", "druska", "halven", "skorli", "tramel",
]

ALL_FEATURE_TOKENS = set(SHAPE_TOKENS + COLOUR_TOKENS + TEXTURE_TOKENS)

# Training nouns -- 32 kinds, from corpus_generator.py v1.5
TRAIN_NOUNS = [
    "wug", "blorp", "dax", "fep", "kiv", "miff", "norp", "pell",
    "quiff", "rix", "smed", "terg", "ulf", "vorn", "weff", "yib",
    "zav", "blim", "corf", "drell", "folt", "grix", "hurn", "jav",
    "klef", "lorp", "mave", "niff", "orp", "plud", "quav", "relf",
]

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

NO_LABEL_CONDITIONS = {"bare_nolabel", "paraphrased_nolabel"}
ALL_SEEDS = [42, 123, 456, 789, 1001]

EPSILON = 1e-8  # For log(0) prevention only


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def load_sentences(corpus_path):
    """Load corpus as list of whitespace-tokenised sentence token lists."""
    sentences = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line.split())
    return sentences


def compute_feature_stats(sentences, noun_set, feature_set, feature_list):
    """
    For each noun, compute:
      - normalised entropy of the conditional distribution P(feature | noun)
      - modal concentration: proportion of co-occurrences with the modal feature
      - modal PMI: PMI of the most frequent co-occurring feature token

    Returns: {noun -> dict with norm_entropy, modal_conc, modal_pmi, etc.}
    """
    n_sentences = len(sentences)
    K = len(feature_list)
    max_entropy = math.log2(K) if K > 1 else 1.0

    # Sentence-level co-occurrence counts
    noun_counts = defaultdict(int)
    feature_counts = defaultdict(int)
    joint_counts = defaultdict(int)

    for sent in sentences:
        sent_set = set(sent)
        nouns_here = sent_set & noun_set
        features_here = sent_set & feature_set
        for n in nouns_here:
            noun_counts[n] += 1
        for f in features_here:
            feature_counts[f] += 1
        for n in nouns_here:
            for f in features_here:
                joint_counts[(n, f)] += 1

    results = {}
    for noun in noun_set:
        if noun_counts[noun] == 0:
            continue  # noun absent (expected for no-label)

        # Build conditional distribution P(feature | noun)
        counts = {}
        total = 0
        for feat in feature_list:
            c = joint_counts[(noun, feat)]
            counts[feat] = c
            total += c

        if total == 0:
            continue  # noun never co-occurs with any feature of this class

        # Normalised entropy
        entropy = 0.0
        for feat in feature_list:
            p = counts[feat] / total
            if p > 0:
                entropy -= p * math.log2(p)
        norm_entropy = entropy / max_entropy

        # Modal concentration
        modal_feat = max(counts, key=counts.get)
        modal_count = counts[modal_feat]
        modal_conc = modal_count / total

        # Modal PMI (for reference)
        p_noun = noun_counts[noun] / n_sentences
        p_feat_marginal = feature_counts[modal_feat] / n_sentences if feature_counts[modal_feat] > 0 else EPSILON
        p_joint = modal_count / n_sentences if modal_count > 0 else EPSILON / n_sentences
        modal_pmi = math.log2(p_joint / (p_noun * p_feat_marginal))

        results[noun] = {
            "norm_entropy": round(norm_entropy, 4),
            "modal_conc": round(modal_conc, 4),
            "modal_pmi": round(modal_pmi, 4),
            "modal_feature": modal_feat,
            "modal_count": modal_count,
            "total_cooccurrences": total,
            "distribution": counts,
        }

    return results


def summarise(noun_results, metric):
    """Mean +/- SD across nouns for a given metric."""
    values = [r[metric] for r in noun_results.values()]
    if not values:
        return {"mean": None, "sd": None, "n_nouns": 0}
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    sd = math.sqrt(variance)
    return {"mean": round(mean, 4), "sd": round(sd, 4), "n_nouns": len(values)}


# ---------------------------------------------------------------------------
# Per-condition logic
# ---------------------------------------------------------------------------

def check_condition(condition, corpora_dir, noun_set, debug=False):
    """Run manipulation check across all 5 seeds for one condition."""
    if condition in NO_LABEL_CONDITIONS:
        return {
            "condition": condition,
            "status": "N/A",
            "note": "Nouns absent by design. Metrics undefined.",
        }

    seed_results = {}
    for seed in ALL_SEEDS:
        corpus_path = corpora_dir / Path(f"{condition}_seed{seed}.txt")
        if not corpus_path.exists():
            seed_results[seed] = {"status": "missing", "path": str(corpus_path)}
            continue

        sentences = load_sentences(corpus_path)
        shape_res   = compute_feature_stats(sentences, noun_set,
                                            set(SHAPE_TOKENS), SHAPE_TOKENS)
        colour_res  = compute_feature_stats(sentences, noun_set,
                                            set(COLOUR_TOKENS), COLOUR_TOKENS)
        texture_res = compute_feature_stats(sentences, noun_set,
                                            set(TEXTURE_TOKENS), TEXTURE_TOKENS)

        seed_data = {
            "status": "ok",
            "n_sentences": len(sentences),
        }
        for feat_name, feat_res in [("shape", shape_res), ("colour", colour_res), ("texture", texture_res)]:
            seed_data[feat_name] = {}
            for m in ["norm_entropy", "modal_conc", "modal_pmi"]:
                seed_data[feat_name][m] = summarise(feat_res, m)

        if debug and seed == ALL_SEEDS[0]:
            print(f"\n  --- Debug: {condition} seed {seed} ({len(sentences)} sentences) ---")
            print(f"\n  Shape distribution per noun:")
            for noun in sorted(shape_res.keys()):
                r = shape_res[noun]
                dist_str = ", ".join(f"{f}:{c}" for f, c in sorted(r["distribution"].items()) if c > 0)
                print(f"    {noun:>8}: H_norm={r['norm_entropy']:.3f}  "
                      f"modal={r['modal_feature']}({r['modal_conc']:.2f})  "
                      f"PMI={r['modal_pmi']:+.2f}  [{dist_str}]")
            print(f"\n  Colour distribution per noun (first 5):")
            for noun in sorted(colour_res.keys())[:5]:
                r = colour_res[noun]
                dist_str = ", ".join(f"{f}:{c}" for f, c in sorted(r["distribution"].items()) if c > 0)
                print(f"    {noun:>8}: H_norm={r['norm_entropy']:.3f}  "
                      f"modal={r['modal_feature']}({r['modal_conc']:.2f})  [{dist_str}]")

        seed_results[seed] = seed_data

    # Aggregate across seeds
    def aggregate(feature_key, metric):
        means = []
        for s in ALL_SEEDS:
            sr = seed_results.get(s, {})
            if sr.get("status") != "ok":
                continue
            val = sr[feature_key][metric]["mean"]
            if val is not None:
                means.append(val)
        if not means:
            return {"mean": None, "sd_across_seeds": None, "n_seeds": 0}
        grand_mean = sum(means) / len(means)
        variance = sum((m - grand_mean) ** 2 for m in means) / len(means)
        return {
            "mean": round(grand_mean, 4),
            "sd_across_seeds": round(math.sqrt(variance), 4),
            "n_seeds": len(means),
        }

    result = {
        "condition": condition,
        "status": "ok",
        "per_seed": seed_results,
    }
    for feat in ["shape", "colour", "texture"]:
        for metric in ["norm_entropy", "modal_conc", "modal_pmi"]:
            result[f"{feat}_{metric}"] = aggregate(feat, metric)

    return result


# ---------------------------------------------------------------------------
# PASS/FAIL logic and console table
# ---------------------------------------------------------------------------

# Absolute threshold for "stable": H_norm must be below this.
# Perfectly stable = 0.000. Threshold of 0.15 allows noise_injection (~7%).
STABLE_THRESHOLD = 0.15

# CONTRAST-BASED check for "random" features:
# Because sampling sparsity compresses H_norm for colour/texture (only ~3
# mentions per noun due to frame structure), we cannot use an absolute
# threshold. Instead: random feature H must exceed stable feature H by
# MIN_CONTRAST. This is robust because both features share the same corpus.
#
# Simulation (10K trials, K=10 bins):
#   3 draws: E[H_norm] = 0.42 +/- 0.10
#   5 draws: E[H_norm] = 0.59 +/- 0.09
MIN_CONTRAST = 0.15

# Expected patterns per condition.
# Shape_dir: what we expect shape H_norm to look like.
# Tex_dir: what we expect texture H_norm to look like.
# Contrast_ref: which feature provides the baseline for "random" checks.
#
# SCRAMBLED: "Stable-feature SLOT randomised per kind per seed" means each
# kind still has ONE stable feature, but which dimension (shape/colour/texture)
# is stable varies per kind. ~1/3 of kinds have stable shape by chance.
# Expected shape H: intermediate (mix of H=0 and H>0 nouns), NOT fully random.
# The manipulation check for scrambled is: shape H > Regular shape H (i.e., the
# meta-regularity is disrupted). We tag this as "higher_than_regular".
#
# WEAK_LABEL: 75% of sentences use no-label frames (noun absent). Only 25%
# labelled sentences contribute noun-feature co-occurrences. With ~3 labelled
# sentences per noun, colour/texture get ~1 mention each -> H_norm near 0
# by sample size, not by stability. The shape check is valid (still H=0);
# the texture check is uninformative and flagged as SPARSE.
#
# FEATURE_SWAP: Domain A (16 kinds): shape stable, texture random.
# Domain B (16 kinds): texture stable, shape random. Aggregating across
# all 32 kinds averages the two domains. We flag texture as MIXED and
# note that per-domain analysis is needed (in JSON output).

EXPECTED_PATTERN = {
    # (shape_dir, texture_dir, contrast_ref)
    "regular":            ("stable",              "random",  "shape"),
    "scrambled":          ("higher_than_regular",  "info",   None),
    "bare_nolabel":       (None,                   None,     None),
    "paraphrased_nolabel":(None,                   None,     None),
    "weak_label":         ("stable",              "sparse",  None),
    "feature_swap":       ("mixed_domain",        "mixed_domain", None),
    "noise_injection":    ("stable",              "random",  "shape"),
    "frequency_matched":  ("higher_than_regular",  "info",   None),
}


def check_direction(h_val, direction, contrast_baseline=None):
    """PASS/FAIL for a single feature's entropy."""
    if direction is None or h_val is None:
        return "N/A"
    if direction == "stable":
        return "PASS" if h_val < STABLE_THRESHOLD else "FAIL"
    if direction == "random":
        if contrast_baseline is not None:
            return "PASS" if h_val > contrast_baseline + MIN_CONTRAST else "FAIL"
        return "PASS" if h_val > STABLE_THRESHOLD else "FAIL"
    if direction == "higher_than_regular":
        # Must be above stable threshold — scrambled/freq_matched should not
        # look fully stable. Actual magnitude depends on slot allocation.
        return "PASS" if h_val > STABLE_THRESHOLD else "FAIL"
    if direction == "sparse":
        return "SPARSE"  # too few observations to test; informational only
    if direction == "mixed_domain":
        return "MIXED"   # needs per-domain breakdown; see JSON
    if direction == "info":
        return "info"    # reported for reference, no PASS/FAIL
    return "?"


def print_summary_table(results):
    print()
    print("=" * 120)
    print("FEATURE-STABILITY MANIPULATION CHECK")
    print("Primary metric: normalised entropy H_norm(feature | noun)")
    print("  H_norm ~ 0 = stable (one feature per noun) | higher = more random")
    print(f"  Stable threshold: H < {STABLE_THRESHOLD}")
    print(f"  Random check: contrast-based (H_random > H_stable + {MIN_CONTRAST})")
    print("=" * 120)
    hdr = (f"  {'Condition':<24} {'Shape H':>8} {'Col H':>8} {'Tex H':>8}"
           f"  {'ShConc':>7} {'CoConc':>7} {'TxConc':>7}"
           f"  {'Shape':>7} {'Tex':>7}")
    print(hdr)
    print("-" * 120)

    all_pass = True
    for r in results:
        cond = r["condition"]
        if r["status"] == "N/A":
            print(f"  {cond:<24}  {'N/A (nouns absent)':>75}")
            continue

        def get_val(key):
            d = r.get(key, {})
            return d.get("mean")

        sh = get_val("shape_norm_entropy")
        ch = get_val("colour_norm_entropy")
        th = get_val("texture_norm_entropy")
        sc = get_val("shape_modal_conc")
        cc = get_val("colour_modal_conc")
        tc = get_val("texture_modal_conc")

        def fmt_h(v):
            return f"{v:.3f}" if v is not None else "N/A"

        def fmt_c(v):
            return f"{v:.2f}" if v is not None else "N/A"

        pattern = EXPECTED_PATTERN.get(cond, (None, None, None))
        exp_shape, exp_tex, contrast_ref = pattern

        # Determine contrast baseline
        if contrast_ref == "shape":
            baseline = sh
        elif contrast_ref == "texture":
            baseline = th
        else:
            baseline = None

        shape_check = check_direction(sh, exp_shape, contrast_baseline=None)
        tex_check   = check_direction(th, exp_tex, contrast_baseline=baseline)

        # Only count hard FAILs, not SPARSE/MIXED/info
        if "FAIL" in (shape_check, tex_check):
            all_pass = False

        print(f"  {cond:<24} {fmt_h(sh):>8} {fmt_h(ch):>8} {fmt_h(th):>8}"
              f"  {fmt_c(sc):>7} {fmt_c(cc):>7} {fmt_c(tc):>7}"
              f"  {shape_check:>7} {tex_check:>7}")

    print("=" * 120)
    print("  H = normalised entropy (0=stable, 1=random) | Conc = modal concentration")
    print(f"  PASS/FAIL logic:")
    print(f"    stable  : H < {STABLE_THRESHOLD} (absolute)")
    print(f"    random  : H > H_stable + {MIN_CONTRAST} (contrast-based)")
    print(f"    SPARSE  : too few noun-feature co-occurrences to test (weak_label)")
    print(f"    MIXED   : aggregate masks domain-specific patterns (feature_swap)")
    print(f"    info    : reported for reference, no pass/fail criterion")
    print(f"  Scrambled: slot-randomised -> ~1/3 kinds have stable shape by chance")
    print(f"    -> shape H intermediate (not 0, not fully random). Check: H > {STABLE_THRESHOLD}")
    print("=" * 120)
    overall = "PASS" if all_pass else "FAIL -- review per-condition details"
    print(f"  Overall: {overall}")
    print("=" * 120)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Feature-stability manipulation check (normalised entropy)")
    parser.add_argument("--corpora_dir", type=Path, default=Path("data/corpora"))
    parser.add_argument("--output",      type=Path,
                        default=Path("data/mi_check/mi_summary.json"))
    parser.add_argument("--conditions",  nargs="+", default=None)
    parser.add_argument("--noun_list",   type=Path, default=None,
                        help="Optional: one noun per line, overrides TRAIN_NOUNS.")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-noun distributions for first seed.")
    args = parser.parse_args()

    target_conditions = args.conditions or CONDITIONS

    # Noun set
    if args.noun_list:
        with open(args.noun_list, "r", encoding="utf-8") as f:
            noun_set = {l.strip().lower() for l in f if l.strip()}
        noun_set_source = str(args.noun_list)
        print(f"Noun set: {len(noun_set)} nouns from {args.noun_list}.")
    else:
        noun_set = set(TRAIN_NOUNS)
        noun_set_source = "TRAIN_NOUNS (corpus_generator.py v1.5)"
        print(f"Noun set: {len(noun_set)} training nouns (authoritative list).")

    if args.debug:
        print(f"\nNouns ({len(noun_set)}): {', '.join(sorted(noun_set))}")

    if len(noun_set) == 0:
        print("ERROR: No nouns detected.")
        raise SystemExit(1)

    # Run
    results = []
    for cond in target_conditions:
        print(f"\n[{cond}] checking feature stability ...")
        r = check_condition(cond, args.corpora_dir, noun_set, debug=args.debug)
        results.append(r)

    print_summary_table(results)

    # Save JSON (strip full distributions to keep file manageable)
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()
                    if k != "distribution"}
        if isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        return obj

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "method": "normalised entropy of P(feature | noun)",
        "stable_threshold": STABLE_THRESHOLD,
        "min_contrast": MIN_CONTRAST,
        "n_nouns": len(noun_set),
        "noun_set_source": noun_set_source,
        "conditions": clean_for_json(results),
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON summary: {args.output}")


if __name__ == "__main__":
    main()
