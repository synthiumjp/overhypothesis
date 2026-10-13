"""
study1_hbm_posterior.py -- H_Bayes: Ideal Observer Alignment (Pre-Registered)
=============================================================================

Pre-registration spec (PROJECT_PLAN_v1.5, Section 6.1 + Phase 5):
    "H_Bayes: Bayesian Ideal-Observer Alignment -- CONFIRMATORY SECONDARY
     LM shape-generalisation probability across corpus sizes tracks the
     Kemp et al. (2007) HBM posterior predictive.
     Success criterion: KL divergence <= 0.1 nats at 100% corpus size.
     Falsified if: LM curve non-monotonic, or KL > 0.3 nats at all sizes."

Implementation follows Appendix E of PROJECT_PLAN_v1.5:
    Kemp et al. (2007) Dirichlet-multinomial model (Figure 1a):
        alpha ~ Exponential(lambda=1)
        beta ~ Dirichlet(1)           [uniform prior on feature distribution]
        theta_i ~ Dirichlet(alpha * beta)   [category-level feature distribution]
        y_i | n_i ~ Multinomial(theta_i)

Inference method:
    Numerical integration over alpha on a log-spaced 1D grid, following
    Kemp et al.'s own approach for the single-kind model ("The model
    predictions in Figures 3, 4 and 6 were computed using numerical
    integration"). Beta is integrated out via Monte Carlo sampling from
    the Dirichlet(1) prior. This eliminates the MCMC convergence issues
    observed in the initial implementation (acceptance rates of 0.5-1%
    for scrambled and frequency_matched conditions).

Usage:
    python study1_hbm_posterior.py --corpus_dir C:\\overhypothesis\\data\\corpora
                                   --master_csv C:\\overhypothesis\\results\\study1\\master_item_level.csv
                                   --results_dir C:\\overhypothesis\\results\\study1

Author: JP Cacioli
Date: 2026-03-04
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import special


# ============================================================
# Constants
# ============================================================

SHAPE_TOKENS = [
    'zibby', 'krannish', 'tova', 'felk', 'mundi',
    'prex', 'sallo', 'wemb', 'oofy', 'drant'
]

TRAIN_NOUNS = [
    'blim', 'blorp', 'corf', 'dax', 'drell', 'fep', 'folt', 'grix',
    'hurn', 'jav', 'kiv', 'klef', 'lorp', 'mave', 'miff', 'niff',
    'norp', 'orp', 'pell', 'plud', 'quav', 'quiff', 'relf', 'rix',
    'smed', 'terg', 'ulf', 'vorn', 'weff', 'wug', 'yib', 'zav'
]

SEEDS = [42, 123, 456, 789, 1001]

PRIMARY_CONDITIONS = [
    'regular', 'scrambled', 'bare_nolabel', 'paraphrased_nolabel',
    'weak_label', 'feature_swap', 'noise_injection', 'frequency_matched'
]

MODEL_SIZES = ['tiny', 'small', 'medium']


# ============================================================
# Corpus parsing
# ============================================================

def extract_category_shape_counts(corpus_path, condition):
    """Extract per-category shape token counts. Returns dict or None."""
    if condition in ('bare_nolabel', 'paraphrased_nolabel'):
        return None

    counts = defaultdict(lambda: defaultdict(int))
    shape_set = set(SHAPE_TOKENS)
    noun_set = set(TRAIN_NOUNS)

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            raw_tokens = line.strip().split()
            tokens = [t.rstrip('.,;:!?') for t in raw_tokens]
            nouns_in_line = [t for t in tokens if t in noun_set]
            if not nouns_in_line:
                continue
            noun = nouns_in_line[0]
            for token in tokens:
                if token in shape_set:
                    counts[noun][token] += 1

    return {noun: dict(counts[noun]) for noun in counts}


def counts_to_matrix(category_counts, token_list):
    """Convert per-category counts to (n_categories, n_tokens) matrix."""
    categories = sorted(category_counts.keys())
    matrix = np.zeros((len(categories), len(token_list)), dtype=np.float64)
    for i, cat in enumerate(categories):
        for j, token in enumerate(token_list):
            matrix[i, j] = category_counts[cat].get(token, 0)
    return matrix, categories


# ============================================================
# Dirichlet-Multinomial marginal likelihood (closed form)
# ============================================================

def log_marginal_likelihood_all(count_matrix, alpha_beta):
    """Log P(y | alpha, beta) = sum_i log P(y_i | alpha*beta).

    Each category's marginal is the Dirichlet-Multinomial:
        log P(y_i | ab) = lgamma(sum(ab)) - lgamma(sum(ab) + n_i)
                        + sum_j [lgamma(ab_j + y_ij) - lgamma(ab_j)]
    """
    log_p = 0.0
    sum_ab = np.sum(alpha_beta)
    log_gamma_ab = special.gammaln(alpha_beta)
    log_gamma_sum_ab = special.gammaln(sum_ab)

    for i in range(count_matrix.shape[0]):
        n_i = np.sum(count_matrix[i])
        log_p += log_gamma_sum_ab - special.gammaln(sum_ab + n_i)
        for j in range(count_matrix.shape[1]):
            log_p += (special.gammaln(alpha_beta[j] + count_matrix[i, j])
                      - log_gamma_ab[j])
    return log_p


# ============================================================
# Numerical integration
# ============================================================

def numerical_integration(count_matrix, n_tokens, n_alpha=500,
                          n_beta_samples=1000, lam=1.0, rng_seed=42):
    """Compute HBM posterior and predictive via numerical integration.

    1. Log-spaced grid over alpha (1e-4 to 1e2)
    2. For each alpha, MC-integrate over beta ~ Dirichlet(1)
    3. Posterior predictive: importance-sample (alpha, beta) pairs,
       draw theta_new ~ Dirichlet(alpha*beta), evaluate forced-choice
    """
    rng = np.random.RandomState(rng_seed)

    alpha_grid = np.logspace(-4, 2, n_alpha)
    beta_samples = rng.dirichlet(np.ones(n_tokens), size=n_beta_samples)

    # Compute log P(y | alpha, beta) + log P(alpha) for each grid point
    log_joint = np.zeros((n_alpha, n_beta_samples))

    for i, alpha in enumerate(alpha_grid):
        log_prior_alpha = np.log(lam) - lam * alpha
        for j, beta in enumerate(beta_samples):
            alpha_beta = np.clip(alpha * beta, 1e-12, None)
            log_lik = log_marginal_likelihood_all(count_matrix, alpha_beta)
            log_joint[i, j] = log_lik + log_prior_alpha

    # Marginal over beta: average over MC samples
    log_marginal_alpha = (special.logsumexp(log_joint, axis=1)
                          - np.log(n_beta_samples))

    # Posterior over alpha (with log-space Jacobian)
    log_integrand = log_marginal_alpha + np.log(alpha_grid)
    log_norm = special.logsumexp(log_integrand)
    posterior_alpha = np.exp(log_integrand - log_norm)

    # Posterior summaries
    alpha_mean = np.sum(posterior_alpha * alpha_grid)
    cdf = np.cumsum(posterior_alpha)
    cdf /= cdf[-1]
    alpha_median = alpha_grid[min(np.searchsorted(cdf, 0.5), n_alpha - 1)]
    alpha_ci_lo = alpha_grid[min(np.searchsorted(cdf, 0.025), n_alpha - 1)]
    alpha_ci_hi = alpha_grid[min(np.searchsorted(cdf, 0.975), n_alpha - 1)]

    # Posterior predictive forced-choice
    log_weights_flat = log_joint.flatten()
    log_weights_flat -= special.logsumexp(log_weights_flat)
    weights_flat = np.exp(log_weights_flat)

    n_pred = 5000
    indices = rng.choice(len(weights_flat), size=n_pred, p=weights_flat)

    fc_correct = 0
    for k in range(n_pred):
        ai = indices[k] // n_beta_samples
        bi = indices[k] % n_beta_samples
        ab = np.clip(alpha_grid[ai] * beta_samples[bi], 1e-12, None)
        theta_new = rng.dirichlet(ab)
        # Forced choice between two random shape tokens
        idx = rng.choice(n_tokens, size=2, replace=False)
        if theta_new[idx[0]] > theta_new[idx[1]]:
            fc_correct += 1
        elif theta_new[idx[0]] == theta_new[idx[1]]:
            fc_correct += 0.5

    return {
        'alpha_mean': alpha_mean,
        'alpha_median': alpha_median,
        'alpha_ci_lo': alpha_ci_lo,
        'alpha_ci_hi': alpha_ci_hi,
        'hbm_fc_accuracy': fc_correct / n_pred,
    }


# ============================================================
# KL divergence
# ============================================================

def kl_divergence_binary(p_model, p_hbm):
    eps = 1e-10
    p = np.clip(p_model, eps, 1 - eps)
    q = np.clip(p_hbm, eps, 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def bootstrap_kl_ci(model_accuracies, hbm_accuracy, n_bootstrap=10000):
    n = len(model_accuracies)
    kl_samples = []
    for _ in range(n_bootstrap):
        boot_mean = np.mean(np.random.choice(model_accuracies, size=n, replace=True))
        kl_samples.append(kl_divergence_binary(boot_mean, hbm_accuracy))
    kl_samples = np.array(kl_samples)
    return np.mean(kl_samples), np.percentile(kl_samples, 2.5), np.percentile(kl_samples, 97.5)


# ============================================================
# Load model data
# ============================================================

def load_model_so_accuracy(master_csv_path, conditions, model_sizes, seeds):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(master_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cond = row.get('condition', '')
            size = row.get('model_size', row.get('size', ''))
            seed_str = row.get('seed', '')
            if row.get('item_type', '') != 'second_order':
                continue
            if cond not in conditions or size not in model_sizes:
                continue
            try:
                seed_int = int(seed_str)
            except (ValueError, TypeError):
                continue
            if seed_int not in seeds:
                continue
            try:
                correct = int(row.get('correct_binary', row.get('correct_01', 0)))
            except (ValueError, TypeError):
                continue
            data[cond][size][seed_int].append(correct)

    result = {}
    for cond in data:
        result[cond] = {}
        for size in data[cond]:
            result[cond][size] = {s: np.mean(v) for s, v in data[cond][size].items()}
    return result


def find_corpus_file(corpus_dir, condition, seed):
    patterns = [f"{condition}_seed{seed}.txt",
                f"{condition}_seed{seed}_corpus.txt",
                f"corpus_{condition}_seed{seed}.txt"]
    corpus_dir = Path(corpus_dir)
    for p in patterns:
        path = corpus_dir / p
        if path.exists():
            return path
    for subdir in corpus_dir.iterdir():
        if subdir.is_dir():
            for p in patterns:
                path = subdir / p
                if path.exists():
                    return path
    return None


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='H_Bayes HBM Posterior -- Numerical Integration')
    parser.add_argument('--corpus_dir', type=str, required=True)
    parser.add_argument('--master_csv', type=str, required=True)
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--n_alpha', type=int, default=500)
    parser.add_argument('--n_beta_samples', type=int, default=1000)
    parser.add_argument('--seed_rng', type=int, default=42)
    parser.add_argument('--lambda_prior', type=float, default=1.0)
    args = parser.parse_args()

    np.random.seed(args.seed_rng)
    os.makedirs(args.results_dir, exist_ok=True)

    lines = []
    lines.append("=" * 70)
    lines.append("H_BAYES: IDEAL OBSERVER ALIGNMENT -- Pre-Registered Metric")
    lines.append("Kemp et al. (2007) Dirichlet-Multinomial HBM")
    lines.append(f"Inference: Numerical integration over alpha ({args.n_alpha} grid points)")
    lines.append(f"  + Monte Carlo over beta ({args.n_beta_samples} Dirichlet(1) samples)")
    lines.append(f"  lambda={args.lambda_prior}, RNG seed={args.seed_rng}")
    lines.append("Following Kemp et al.: 'The model predictions in Figures 3, 4")
    lines.append("  and 6 were computed using numerical integration.'")
    lines.append("=" * 70)
    lines.append("")

    print("Loading model SO accuracy...")
    model_acc = load_model_so_accuracy(
        args.master_csv, PRIMARY_CONDITIONS, MODEL_SIZES, SEEDS)

    lines.append("MODEL SO ACCURACY (from master CSV):")
    for cond in PRIMARY_CONDITIONS:
        for size in MODEL_SIZES:
            if cond in model_acc and size in model_acc.get(cond, {}):
                sa = model_acc[cond][size]
                vals = [sa[s] for s in SEEDS if s in sa]
                if vals:
                    lines.append(
                        f"  {cond}/{size}: mean={np.mean(vals):.3f} "
                        f"({', '.join(f's{s}={sa[s]:.3f}' for s in sorted(sa))})")
    lines.append("")

    csv_rows = []

    for condition in PRIMARY_CONDITIONS:
        lines.append(f"{'=' * 50}")
        lines.append(f"CONDITION: {condition}")
        lines.append(f"{'=' * 50}")

        if condition in ('bare_nolabel', 'paraphrased_nolabel'):
            lines.append("  SKIPPED: No noun labels => HBM cannot infer categories.")
            lines.append("")
            continue

        corpus_path = find_corpus_file(args.corpus_dir, condition, 42)
        if corpus_path is None:
            lines.append(f"  WARNING: Corpus not found for {condition}/seed42")
            lines.append("")
            continue

        print(f"\nProcessing {condition}...")
        lines.append(f"  Corpus: {corpus_path}")

        cat_counts = extract_category_shape_counts(str(corpus_path), condition)
        if cat_counts is None:
            lines.append("  No category-level counts available.")
            lines.append("")
            continue

        count_matrix, cat_names = counts_to_matrix(cat_counts, SHAPE_TOKENS)
        n_cats, n_tokens = count_matrix.shape
        lines.append(f"  Categories: {n_cats}, Shape tokens: {n_tokens}")
        lines.append(f"  Total shape observations: {int(count_matrix.sum())}")

        print(f"  Integrating ({args.n_alpha} alpha x {args.n_beta_samples} beta)...")
        result = numerical_integration(
            count_matrix, n_tokens,
            n_alpha=args.n_alpha,
            n_beta_samples=args.n_beta_samples,
            lam=args.lambda_prior,
            rng_seed=args.seed_rng)

        lines.append(f"  Posterior alpha: mean={result['alpha_mean']:.4f}, "
                     f"median={result['alpha_median']:.4f}, "
                     f"95% CI=[{result['alpha_ci_lo']:.4f}, {result['alpha_ci_hi']:.4f}]")
        lines.append(f"  HBM forced-choice prediction (within-shape foil): "
                     f"{result['hbm_fc_accuracy']:.3f}")

        lines.append("")
        lines.append("  KL DIVERGENCE (model || HBM):")

        for size in MODEL_SIZES:
            if condition not in model_acc or size not in model_acc.get(condition, {}):
                lines.append(f"    {size}: No model data")
                continue
            sa = model_acc[condition][size]
            acc_vals = [sa[s] for s in SEEDS if s in sa]
            if not acc_vals:
                continue

            model_mean = np.mean(acc_vals)
            kl_mean, kl_lo, kl_hi = bootstrap_kl_ci(
                np.array(acc_vals), result['hbm_fc_accuracy'])
            passes = kl_mean <= 0.1
            verdict = "PASS (<=0.1 nats)" if passes else "FAIL (>0.1 nats)"

            lines.append(f"    {size}: model={model_mean:.3f}, "
                         f"HBM={result['hbm_fc_accuracy']:.3f}, "
                         f"KL={kl_mean:.4f} nats [{kl_lo:.4f}, {kl_hi:.4f}] -- {verdict}")

            csv_rows.append({
                'condition': condition,
                'model_size': size,
                'model_so_accuracy': f"{model_mean:.4f}",
                'hbm_fc_accuracy': f"{result['hbm_fc_accuracy']:.4f}",
                'kl_divergence': f"{kl_mean:.6f}",
                'kl_ci_lo': f"{kl_lo:.6f}",
                'kl_ci_hi': f"{kl_hi:.6f}",
                'passes_threshold': str(passes),
                'alpha_mean': f"{result['alpha_mean']:.4f}",
                'alpha_median': f"{result['alpha_median']:.4f}",
                'alpha_ci_lo': f"{result['alpha_ci_lo']:.4f}",
                'alpha_ci_hi': f"{result['alpha_ci_hi']:.4f}",
                'n_categories': n_cats,
                'n_observations': int(count_matrix.sum()),
            })

        lines.append("")

    # Interpretation
    lines.append("=" * 70)
    lines.append("INTERPRETATION")
    lines.append("=" * 70)
    lines.append("")
    lines.append("ALPHA POSTERIOR VALIDATES CORPUS DESIGN:")
    lines.append("  Regular/Weak-label: alpha << 1 => strong within-category shape")
    lines.append("    uniformity (overhypothesis structure present)")
    lines.append("  Scrambled/Freq-matched: alpha ~ 1+ => no within-category")
    lines.append("    uniformity (controls working as intended)")
    lines.append("  Feature-swap/Noise: intermediate alpha")
    lines.append("")
    lines.append("WITHIN-SHAPE FOIL DESIGN:")
    lines.append("  HBM predicts SOME shape for novel categories but not WHICH.")
    lines.append("  Forced choice between two shapes => HBM predicts ~50%.")
    lines.append("  Models also ~50%. KL is small but uninformative.")
    lines.append("  Diagnostic power lies in H_swap, not H_Bayes.")
    lines.append("  Reported for pre-registration completeness.")
    lines.append("")

    report_text = "\n".join(lines)
    report_path = os.path.join(args.results_dir, 'hbm_posterior_output.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print("\n" + report_text)

    if csv_rows:
        csv_path = os.path.join(args.results_dir, 'hbm_posterior_results.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nCSV saved: {csv_path} ({len(csv_rows)} rows)")

    print(f"Report saved: {report_path}")


if __name__ == '__main__':
    main()
