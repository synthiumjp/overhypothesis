"""
study1_tost.py — TOST Equivalence Test for Second-Order Accuracy
=================================================================
Purpose: Formally establish absence of effect on SO accuracy using
Two One-Sided Tests (TOST) with equivalence bounds of +/-5 pp and +/-10 pp
around 50% chance.

Paper location: S4.1 (one sentence) + S5.6 Limitations (replaces hedge)
Output: results/study1/tost_results.txt

NOTE: This analysis was NOT pre-registered. It is reported as a
supplementary post-hoc analysis to strengthen the null interpretation.

Author: JP Cacioli
Date: 2026-03-05
Session: 11 (V9 revision)
"""

import os
import glob
import csv
import numpy as np
from scipy import stats
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
RESULTS_DIR = r"C:\overhypothesis\results\study1"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "tost_results.txt")

BOUNDS_PP = [5, 10]
CHANCE = 50.0
ALPHA = 0.05

PRIMARY_CONDITIONS = ["regular", "scrambled"]
ALL_CONDITIONS = [
    "regular", "scrambled", "bare_nolabel", "paraphrased_nolabel",
    "weak_label", "feature_swap", "noise_injection", "frequency_matched"
]
SIZES = ["tiny", "small", "medium"]
SEEDS = [42, 123, 456, 789, 1001]


# ============================================================
# DATA LOADING — from item-level CSVs
# ============================================================

def load_so_accuracy():
    """
    Load SO accuracy per condition/size/seed from item-level CSVs.

    CSV columns (from evaluate.py):
      item_id, item_type, noun, target, foil, correct, foil_type, domain,
      lp_target, lp_foil, lp_target_norm, lp_foil_norm, delta_lp_norm,
      correct_binary, rank, ranked_first

    correct_binary = 1 if target log-prob > foil log-prob, else 0
    correct = "A" or "B" (which completion is the right answer — NOT binary)

    Filename: results_{condition}_{size}_seed{seed}.csv
    Dose-response: results_{condition}_{size}_seed{seed}_frac{pct}.csv (skip)
    """
    data = {}  # (condition, size) -> { seed: accuracy_pct }

    csv_files = glob.glob(os.path.join(RESULTS_DIR, "results_*.csv"))
    print(f"Found {len(csv_files)} item-level CSV files.")

    parsed = 0
    skipped_frac = 0
    skipped_parse = 0

    for cf in csv_files:
        basename = os.path.splitext(os.path.basename(cf))[0]
        name_part = basename.replace("results_", "")

        # Skip dose-response
        if "_frac" in name_part:
            skipped_frac += 1
            continue

        # Parse {condition}_{size}_seed{seed}
        if "_seed" not in name_part:
            skipped_parse += 1
            continue
        cond_size, seed_str = name_part.rsplit("_seed", 1)
        try:
            seed = int(seed_str)
        except ValueError:
            skipped_parse += 1
            continue

        # Split condition from size
        condition = None
        size = None
        for sz in SIZES:
            if cond_size.endswith(f"_{sz}"):
                condition = cond_size[:-len(f"_{sz}")]
                size = sz
                break
        if condition is None:
            skipped_parse += 1
            continue

        # Read CSV
        so_correct = 0
        so_total = 0
        try:
            with open(cf, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("item_type", "").strip() != "second_order":
                        continue
                    so_total += 1

                    # Use correct_binary column (1 = target > foil)
                    cb = row.get("correct_binary", "").strip()
                    if cb in ["1", "1.0"]:
                        so_correct += 1
                    elif cb in ["0", "0.0"]:
                        pass  # incorrect, counted in total
                    else:
                        # Fallback: compute from log-probs
                        try:
                            lp_t = float(row.get("lp_target", 0))
                            lp_f = float(row.get("lp_foil", 0))
                            if lp_t > lp_f:
                                so_correct += 1
                        except (ValueError, TypeError):
                            pass
        except Exception as e:
            print(f"  Error reading {os.path.basename(cf)}: {e}")
            continue

        if so_total > 0:
            so_acc = (so_correct / so_total) * 100
            key = (condition, size)
            if key not in data:
                data[key] = {}
            data[key][seed] = so_acc
            parsed += 1

    print(f"  Parsed: {parsed}, Skipped dose-response: {skipped_frac}, "
          f"Skipped unparseable: {skipped_parse}")
    return data


# ============================================================
# TOST
# ============================================================

def tost_one_sample(values, mu0, delta, alpha=0.05):
    n = len(values)
    mean = np.mean(values)
    sd = np.std(values, ddof=1)
    se = sd / np.sqrt(n)

    if se < 1e-12:
        # All values identical
        inside = (mu0 - delta) < mean < (mu0 + delta)
        return {
            "mean": mean, "sd": sd, "se": se, "n": n,
            "t_lower": np.inf if inside else -np.inf,
            "p_lower": 0.0 if inside else 1.0,
            "t_upper": np.inf if inside else -np.inf,
            "p_upper": 0.0 if inside else 1.0,
            "p_tost": 0.0 if inside else 1.0,
            "equivalent": inside,
            "ci_90": (mean, mean), "ci_95": (mean, mean),
            "mu0": mu0, "delta": delta,
            "bounds": (mu0 - delta, mu0 + delta),
        }

    t_lower = (mean - (mu0 - delta)) / se
    p_lower = stats.t.sf(t_lower, df=n - 1)

    t_upper = ((mu0 + delta) - mean) / se
    p_upper = stats.t.sf(t_upper, df=n - 1)

    p_tost = max(p_lower, p_upper)

    t_crit_90 = stats.t.ppf(1 - alpha, df=n - 1)
    ci_90 = (mean - t_crit_90 * se, mean + t_crit_90 * se)

    t_crit_95 = stats.t.ppf(0.975, df=n - 1)
    ci_95 = (mean - t_crit_95 * se, mean + t_crit_95 * se)

    return {
        "mean": mean, "sd": sd, "se": se, "n": n,
        "t_lower": t_lower, "p_lower": p_lower,
        "t_upper": t_upper, "p_upper": p_upper,
        "p_tost": p_tost, "equivalent": p_tost < alpha,
        "ci_90": ci_90, "ci_95": ci_95,
        "mu0": mu0, "delta": delta,
        "bounds": (mu0 - delta, mu0 + delta),
    }


def format_tost_result(condition, size, result):
    lines = []
    lines.append(f"  {condition} / {size} (n={result['n']} seeds)")
    lines.append(f"    Mean SO accuracy: {result['mean']:.1f}% (SD={result['sd']:.1f})")
    lines.append(f"    95% CI: [{result['ci_95'][0]:.1f}, {result['ci_95'][1]:.1f}]")
    lines.append(f"    90% CI: [{result['ci_90'][0]:.1f}, {result['ci_90'][1]:.1f}]")
    lines.append(f"    Equivalence bounds: [{result['bounds'][0]:.0f}, {result['bounds'][1]:.0f}]")
    lines.append(f"    t_lower={result['t_lower']:.3f} (p={result['p_lower']:.4f}), "
                 f"t_upper={result['t_upper']:.3f} (p={result['p_upper']:.4f})")
    verdict = "EQUIVALENT (reject H0)" if result["equivalent"] else "NOT EQUIVALENT (fail to reject H0)"
    lines.append(f"    TOST p = {result['p_tost']:.4f} -> {verdict}")
    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("TOST Equivalence Test -- Second-Order Accuracy")
    print("=" * 60)

    data = load_so_accuracy()

    if not data:
        print("\nERROR: No SO accuracy data found.")
        print(f"Expected item-level CSVs in: {RESULTS_DIR}")
        return

    # Print summary
    print(f"\nLoaded SO accuracy for {len(data)} condition x size cells:")
    for (cond, sz), seed_dict in sorted(data.items()):
        vals = [f"{seed_dict[s]:.1f}" for s in sorted(seed_dict.keys())]
        print(f"  {cond}/{sz}: {len(seed_dict)} seeds -> {vals}")

    # Sanity check
    reg_med = data.get(("regular", "medium"), {})
    if reg_med:
        sample_mean = np.mean(list(reg_med.values()))
        print(f"\n  Sanity check -- Regular/Medium mean: {sample_mean:.1f}%")
        if sample_mean < 1.0:
            print("  ERROR: Values near 0%. Expected ~51% from paper Table 1.")
            print("  Check that correct_binary column is being read correctly.")
            return

    # Run TOST
    report = []
    report.append("=" * 70)
    report.append("TOST EQUIVALENCE TEST -- SECOND-ORDER ACCURACY")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Alpha: {ALPHA}")
    report.append(f"Null (chance): {CHANCE}%")
    report.append(f"Equivalence bounds tested: +/-{BOUNDS_PP} pp")
    report.append("NOTE: This analysis is NOT pre-registered.")
    report.append("=" * 70)

    all_results = {}

    for bound_pp in BOUNDS_PP:
        report.append(f"\n{'=' * 50}")
        report.append(f"EQUIVALENCE BOUND: +/-{bound_pp} pp  ->  "
                      f"[{CHANCE - bound_pp}, {CHANCE + bound_pp}]")
        report.append(f"{'=' * 50}")

        report.append("\n--- PRIMARY CONDITIONS (H1 comparison) ---")
        for condition in PRIMARY_CONDITIONS:
            for size in SIZES:
                key = (condition, size)
                if key in data and len(data[key]) >= 2:
                    vals = np.array([data[key][s] for s in sorted(data[key].keys())])
                    result = tost_one_sample(vals, mu0=CHANCE, delta=bound_pp, alpha=ALPHA)
                    all_results[(condition, size, bound_pp)] = result
                    report.append(format_tost_result(condition, size, result))
                else:
                    report.append(f"  {condition} / {size}: INSUFFICIENT DATA")

        report.append("\n--- ALL OTHER CONDITIONS ---")
        for condition in ALL_CONDITIONS:
            if condition in PRIMARY_CONDITIONS:
                continue
            for size in SIZES:
                key = (condition, size)
                if key in data and len(data[key]) >= 2:
                    vals = np.array([data[key][s] for s in sorted(data[key].keys())])
                    result = tost_one_sample(vals, mu0=CHANCE, delta=bound_pp, alpha=ALPHA)
                    all_results[(condition, size, bound_pp)] = result
                    report.append(format_tost_result(condition, size, result))

        n_tested = sum(1 for (c, s, b) in all_results if b == bound_pp)
        n_equiv = sum(1 for (c, s, b), r in all_results.items()
                      if b == bound_pp and r["equivalent"])
        report.append(f"\n  Summary (+/-{bound_pp} pp): {n_equiv}/{n_tested} cells "
                     f"show equivalence to chance at alpha={ALPHA}")

    # Pooled Regular
    report.append(f"\n{'=' * 50}")
    report.append("POOLED ANALYSIS (Regular, all seeds x all sizes)")
    report.append(f"{'=' * 50}")

    regular_all = []
    for size in SIZES:
        key = ("regular", size)
        if key in data:
            regular_all.extend(data[key].values())

    if regular_all:
        regular_all = np.array(regular_all)
        report.append(f"  N = {len(regular_all)} observations")
        report.append(f"  Mean = {np.mean(regular_all):.1f}%, "
                     f"SD = {np.std(regular_all, ddof=1):.1f}")
        for bound_pp in BOUNDS_PP:
            result = tost_one_sample(regular_all, mu0=CHANCE, delta=bound_pp, alpha=ALPHA)
            verdict = "EQUIVALENT" if result["equivalent"] else "NOT EQUIVALENT"
            report.append(f"  +/-{bound_pp} pp: TOST p = {result['p_tost']:.4f} -> {verdict}")
            report.append(f"    90% CI: [{result['ci_90'][0]:.1f}, {result['ci_90'][1]:.1f}]")

    # Interpretation
    report.append(f"\n{'=' * 70}")
    report.append("INTERPRETATION FOR PAPER")
    report.append(f"{'=' * 70}")

    key10 = all_results.get(("regular", "medium", 10))
    if key10:
        if key10["equivalent"]:
            report.append(
                f"\nS4.1 sentence: 'TOST equivalence testing confirms that Regular "
                f"Medium SO accuracy (M = {key10['mean']:.1f}%, 90% CI "
                f"[{key10['ci_90'][0]:.1f}, {key10['ci_90'][1]:.1f}]) falls within "
                f"+/-10 pp of chance (p = {key10['p_tost']:.3f}), formally "
                f"establishing the absence of a meaningful effect.'"
            )
        else:
            report.append(
                f"\nS4.1: TOST at +/-10 pp did NOT reach significance "
                f"(p = {key10['p_tost']:.3f}). 95% CI "
                f"[{key10['ci_95'][0]:.1f}, {key10['ci_95'][1]:.1f}] spans 50%."
            )

    key5 = all_results.get(("regular", "medium", 5))
    if key5 and not key5["equivalent"]:
        report.append(
            f"\nS5.6: 'TOST at +/-5 pp does not reach significance "
            f"(p = {key5['p_tost']:.3f}), reflecting limited power "
            f"with n={key5['n']} seeds.'"
        )

    report.append(f"\n{'=' * 70}")
    report.append("END OF TOST REPORT")
    report.append(f"{'=' * 70}")

    full_report = "\n".join(report)
    print("\n" + full_report)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(full_report)
    print(f"\nReport saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
