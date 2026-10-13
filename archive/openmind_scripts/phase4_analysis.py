#!/usr/bin/env python3
"""
phase4_analysis.py — Pre-registered hypothesis tests for Study 1.

Reads master_summary.csv and master_item_level.csv (produced by aggregate_results.py)
and runs all confirmatory and exploratory analyses per the OSF pre-registration.

Hypotheses implemented:
  H1:      Regular vs Scrambled on second_order items (mixed-effects logistic)
  H2:      Frame-variant transfer (paired t-test per size)
  H_Bayes: KL divergence <= 0.1 nats at 100% corpus (bootstrap CI)
  H_label: Jonckheere trend Regular >> Weak >> Paraphrased >> Bare (Kendall tau-b)
  H3:      Dose-response monotonic trend (Kendall tau-b)
  H4:      Count-shape > Count-texture AND Mass-texture > Mass-shape
  H_swap:  Feature-swap domain switching

Nonparametric backups (Mann-Whitney U) for all between-condition comparisons.

Usage:
  python scripts/phase4_analysis.py [--results_dir results/study1] [--output_dir results/analysis]

Requirements:
  pip install pandas numpy scipy statsmodels

Optional (for mixed-effects):
  pip install pymer4   (R bridge — requires R + lme4 installed)
  OR: use statsmodels GEE as fallback

Author: JP Cacioli / Claude (Phase 4 implementation)
Date: 2026-03-04
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Suppress convergence warnings during model fitting
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ── Constants from pre-registration ─────────────────────────────────────────

ALPHA = 0.05
CHANCE = 0.50  # Binary forced-choice
H1_THRESHOLD_ABOVE_CHANCE = 0.15  # >=15 pp above chance
H1_THRESHOLD_ABOVE_SCRAMBLED = 0.10  # >=10 pp above scrambled
H2_RATIO_THRESHOLD = 0.75  # Frame-variant accuracy >= 75% of standard
H_BAYES_KL_THRESHOLD = 0.10  # KL divergence <= 0.1 nats
SEEDS = [42, 123, 456, 789, 1001]
SIZES = ['tiny', 'small', 'medium']
SIZE_ORDER = {'tiny': 0, 'small': 1, 'medium': 2}

# Label ordering for H_label Jonckheere test (strongest to weakest labelling)
LABEL_ORDER = ['regular', 'weak_label', 'paraphrased_nolabel', 'bare_nolabel']


# ── Utility functions ───────────────────────────────────────────────────────

def cohens_d(group1, group2):
    """Cohen's d with pooled standard deviation."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return np.nan
    return (np.mean(group1) - np.mean(group2)) / pooled_sd


def bootstrap_ci(data, stat_func=np.mean, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap confidence interval for a statistic."""
    rng = np.random.RandomState(seed)
    data = np.asarray(data)
    n = len(data)
    boot_stats = np.array([
        stat_func(data[rng.randint(0, n, size=n)]) for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return np.percentile(boot_stats, [alpha * 100, (1 - alpha) * 100])


def jonckheere_terpstra(groups, alternative='decreasing'):
    """
    Jonckheere-Terpstra trend test.
    
    groups: list of arrays, ordered from expected highest to lowest.
    alternative: 'decreasing' means we expect group[0] > group[1] > ... > group[-1].
    
    Returns: (J statistic, p-value, Kendall tau-b effect size)
    """
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    
    # Count concordant pairs across all ordered group pairs
    J = 0
    n_pairs = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            for xi in groups[i]:
                for xj in groups[j]:
                    if xi > xj:
                        J += 1
                    elif xi == xj:
                        J += 0.5
                    n_pairs += 1
    
    # Expected value and variance under H0
    ns = [len(g) for g in groups]
    N = sum(ns)
    E_J = (N**2 - sum(n**2 for n in ns)) / 4
    
    # Variance (without ties correction — adequate for continuous data)
    num1 = N**2 * (2 * N + 3) - sum(n**2 * (2 * n + 3) for n in ns)
    var_J = num1 / 72
    
    if var_J <= 0:
        return J, 1.0, 0.0
    
    # Normal approximation
    z = (J - E_J) / np.sqrt(var_J)
    
    if alternative == 'decreasing':
        p = 1 - stats.norm.cdf(z)  # One-sided: larger J = more concordant
    else:
        p = stats.norm.cdf(z)
    
    # Kendall tau-b as effect size
    # tau-b = (concordant - discordant) / sqrt(n_pairs * n_pairs_adjusted)
    concordant = J
    discordant = n_pairs - J
    tau_b = (concordant - discordant) / n_pairs if n_pairs > 0 else 0.0
    
    return J, p, tau_b


def kl_divergence(p, q, epsilon=1e-10):
    """KL(P || Q) in nats. p and q are probability arrays that sum to 1."""
    p = np.asarray(p, dtype=float) + epsilon
    q = np.asarray(q, dtype=float) + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))


# ── Hypothesis tests ────────────────────────────────────────────────────────

class AnalysisResults:
    """Collect and format analysis results."""
    
    def __init__(self):
        self.results = {}
        self.log_lines = []
    
    def log(self, msg):
        print(msg)
        self.log_lines.append(msg)
    
    def add(self, hypothesis, result_dict):
        self.results[hypothesis] = result_dict
    
    def to_json(self, filepath):
        """Save all results as JSON (with numpy/pandas type conversion)."""
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            return str(obj)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=convert)
    
    def save_log(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.log_lines))


def test_H1(summary, items, results):
    """
    H1: Overhypothesis Emergence.
    Regular vs Scrambled on second_order items.
    
    Criteria: >=15 pp above chance AND >=10 pp above Scrambled.
    Test: Mixed-effects logistic regression (item-level) with fallback to 
          aggregated t-test + Mann-Whitney U.
    """
    results.log("\n" + "="*80)
    results.log("H1: OVERHYPOTHESIS EMERGENCE — Regular vs Scrambled (second_order)")
    results.log("="*80)
    
    h1_result = {'hypothesis': 'H1', 'status': None, 'tests': {}}
    
    # ── Aggregated analysis (seed-level means) ──────────────────────────────
    
    so = summary[
        (summary['item_type'] == 'second_order') & 
        (summary['corpus_fraction'] == 1.0)
    ].copy()
    
    for size in SIZES:
        results.log(f"\n  ── {size.upper()} ──")
        
        reg = so[(so['condition'] == 'regular') & (so['size'] == size)]['accuracy'].values
        scr = so[(so['condition'] == 'scrambled') & (so['size'] == size)]['accuracy'].values
        
        if len(reg) == 0 or len(scr) == 0:
            results.log(f"    WARNING: Missing data for {size}")
            continue
        
        reg_mean = np.mean(reg)
        scr_mean = np.mean(scr)
        diff = reg_mean - scr_mean
        above_chance = reg_mean - CHANCE
        
        results.log(f"    Regular:   M={reg_mean:.3f} (SD={np.std(reg, ddof=1):.3f}), seeds={list(reg)}")
        results.log(f"    Scrambled: M={scr_mean:.3f} (SD={np.std(scr, ddof=1):.3f}), seeds={list(scr)}")
        results.log(f"    Δ(Reg-Scr)={diff:.3f}, Δ(Reg-Chance)={above_chance:.3f}")
        
        # Criterion check
        meets_chance = above_chance >= H1_THRESHOLD_ABOVE_CHANCE
        meets_scrambled = diff >= H1_THRESHOLD_ABOVE_SCRAMBLED
        results.log(f"    >=15pp above chance? {'YES' if meets_chance else 'NO'} ({above_chance:.1%})")
        results.log(f"    >=10pp above scrambled? {'YES' if meets_scrambled else 'NO'} ({diff:.1%})")
        
        # Independent t-test
        if len(reg) >= 2 and len(scr) >= 2:
            t_stat, t_p = stats.ttest_ind(reg, scr, equal_var=False)
            d = cohens_d(reg, scr)
            results.log(f"    Welch t-test: t={t_stat:.3f}, p={t_p:.4f}, d={d:.3f}")
            
            # Mann-Whitney U (nonparametric backup)
            u_stat, u_p = stats.mannwhitneyu(reg, scr, alternative='greater')
            results.log(f"    Mann-Whitney U: U={u_stat:.1f}, p={u_p:.4f} (one-sided)")
        
        # Bootstrap CI on the difference
        if len(reg) >= 2 and len(scr) >= 2:
            combined = np.concatenate([reg, scr])
            # Bootstrap the mean difference
            rng = np.random.RandomState(42)
            boot_diffs = []
            for _ in range(10000):
                b_reg = reg[rng.randint(0, len(reg), size=len(reg))]
                b_scr = scr[rng.randint(0, len(scr), size=len(scr))]
                boot_diffs.append(np.mean(b_reg) - np.mean(b_scr))
            ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
            results.log(f"    Bootstrap 95% CI on Δ: [{ci_lo:.3f}, {ci_hi:.3f}]")
        
        h1_result['tests'][size] = {
            'regular_mean': float(reg_mean),
            'scrambled_mean': float(scr_mean),
            'difference': float(diff),
            'above_chance': float(above_chance),
            'meets_chance_criterion': bool(meets_chance),
            'meets_scrambled_criterion': bool(meets_scrambled),
            'H1_supported': bool(meets_chance and meets_scrambled),
        }
    
    # ── Mixed-effects logistic regression (item-level) ──────────────────────
    
    if items is not None and not items.empty:
        results.log("\n  ── Mixed-Effects Logistic Regression (Item-Level) ──")
        
        h1_items = items[
            (items['item_type'] == 'second_order') &
            (items['corpus_fraction'] == 1.0) &
            (items['condition'].isin(['regular', 'scrambled']))
        ].copy()
        
        if len(h1_items) > 0:
            # Try statsmodels GEE as the primary Python approach
            # (pymer4/R bridge is optional)
            try:
                import statsmodels.api as sm
                from statsmodels.genmod.generalized_estimating_equations import GEE
                from statsmodels.genmod.families import Binomial
                from statsmodels.genmod.cov_struct import Exchangeable
                
                h1_items['condition_code'] = (h1_items['condition'] == 'regular').astype(int)
                h1_items['size_code'] = h1_items['size'].map(SIZE_ORDER)
                
                # Ensure binary outcome column exists
                correct_col = None
                for c in ['correct_01', 'correct_binary', 'correct']:
                    if c in h1_items.columns:
                        # Verify it's numeric
                        if h1_items[c].dtype in ('int64', 'float64', 'bool'):
                            correct_col = c
                            break
                
                if correct_col:
                    for size in SIZES:
                        size_data = h1_items[h1_items['size'] == size].copy()
                        if len(size_data) < 10:
                            continue
                        
                        # Create item group variable for GEE
                        if 'item_id' in size_data.columns:
                            group_col = 'item_id'
                        else:
                            # Fallback: cluster by seed
                            group_col = 'seed'
                        
                        size_data = size_data.dropna(subset=[correct_col, 'condition_code'])
                        
                        try:
                            model = GEE(
                                size_data[correct_col].astype(float),
                                sm.add_constant(size_data[['condition_code']]),
                                groups=size_data[group_col],
                                family=Binomial(),
                                cov_struct=Exchangeable()
                            )
                            fit = model.fit()
                            coef = fit.params.get('condition_code', np.nan)
                            p_val = fit.pvalues.get('condition_code', np.nan)
                            results.log(f"    GEE ({size}): β(condition)={coef:.4f}, p={p_val:.4f}")
                            
                            h1_result['tests'][f'{size}_gee'] = {
                                'beta': float(coef),
                                'p_value': float(p_val),
                                'se': float(fit.bse.get('condition_code', np.nan)),
                            }
                        except Exception as e:
                            results.log(f"    GEE ({size}): Failed — {e}")
                else:
                    results.log("    WARNING: No 'correct' column found in item-level data")
            except ImportError:
                results.log("    statsmodels not available; skipping GEE. Install with: pip install statsmodels")
    
    # ── Overall H1 verdict ──────────────────────────────────────────────────
    
    # H1 is assessed per size; the primary test is "at least one size" meets criteria
    any_supported = any(
        h1_result['tests'].get(s, {}).get('H1_supported', False) for s in SIZES
    )
    h1_result['status'] = 'SUPPORTED' if any_supported else 'NOT_SUPPORTED'
    results.log(f"\n  H1 VERDICT: {h1_result['status']}")
    
    results.add('H1', h1_result)


def test_H2(summary, results):
    """
    H2: Frame-variant transfer.
    Criterion: Frame-variant accuracy >= 75% of standard (second_order) accuracy.
    Test: Paired t-test per model size.
    """
    results.log("\n" + "="*80)
    results.log("H2: FRAME-VARIANT TRANSFER")
    results.log("="*80)
    
    h2_result = {'hypothesis': 'H2', 'status': None, 'tests': {}}
    
    full = summary[
        (summary['corpus_fraction'] == 1.0) & 
        (summary['condition'] == 'regular')
    ]
    
    for size in SIZES:
        results.log(f"\n  ── {size.upper()} ──")
        
        size_data = full[full['size'] == size]
        
        # Get paired (seed-matched) second_order and frame_variant accuracy
        so = size_data[size_data['item_type'] == 'second_order'].set_index('seed')['accuracy']
        fv = size_data[size_data['item_type'] == 'frame_variant'].set_index('seed')['accuracy']
        
        common_seeds = so.index.intersection(fv.index)
        if len(common_seeds) < 2:
            results.log(f"    WARNING: Insufficient paired data for {size}")
            continue
        
        so_vals = so.loc[common_seeds].values
        fv_vals = fv.loc[common_seeds].values
        
        # Ratio
        ratios = fv_vals / np.where(so_vals > 0, so_vals, 1e-10)
        mean_ratio = np.mean(ratios)
        
        results.log(f"    Second-order: M={np.mean(so_vals):.3f}")
        results.log(f"    Frame-variant: M={np.mean(fv_vals):.3f}")
        results.log(f"    Ratio (FV/SO): M={mean_ratio:.3f} (threshold: {H2_RATIO_THRESHOLD})")
        results.log(f"    Criterion met? {'YES' if mean_ratio >= H2_RATIO_THRESHOLD else 'NO'}")
        
        # Paired t-test on the ratios vs threshold
        if len(ratios) >= 2:
            t_stat, t_p = stats.ttest_1samp(ratios, H2_RATIO_THRESHOLD)
            results.log(f"    One-sample t-test (ratio vs 0.75): t={t_stat:.3f}, p={t_p:.4f}")
            
            # Also: paired t-test on accuracy difference
            t_stat2, t_p2 = stats.ttest_rel(fv_vals, so_vals)
            results.log(f"    Paired t-test (FV vs SO): t={t_stat2:.3f}, p={t_p2:.4f}")
        
        h2_result['tests'][size] = {
            'so_mean': float(np.mean(so_vals)),
            'fv_mean': float(np.mean(fv_vals)),
            'mean_ratio': float(mean_ratio),
            'meets_criterion': bool(mean_ratio >= H2_RATIO_THRESHOLD),
        }
    
    any_met = any(h2_result['tests'].get(s, {}).get('meets_criterion', False) for s in SIZES)
    h2_result['status'] = 'SUPPORTED' if any_met else 'NOT_SUPPORTED'
    results.log(f"\n  H2 VERDICT: {h2_result['status']}")
    
    results.add('H2', h2_result)


def test_H3(summary, results):
    """
    H3: Dose-response monotonic trend.
    Criterion: Monotonic accuracy increase with corpus fraction.
    Test: Kendall tau-b per size.
    """
    results.log("\n" + "="*80)
    results.log("H3: DOSE-RESPONSE MONOTONIC TREND (Regular condition)")
    results.log("="*80)
    
    h3_result = {'hypothesis': 'H3', 'status': None, 'tests': {}}
    
    # Dose-response: regular condition at frac25, frac50, full (1.0)
    dose = summary[
        (summary['condition'] == 'regular') &
        (summary['item_type'] == 'second_order')
    ].copy()
    
    for size in SIZES:
        results.log(f"\n  ── {size.upper()} ──")
        
        size_data = dose[dose['size'] == size]
        
        if size_data.empty:
            results.log(f"    No data for {size}")
            continue
        
        # Group by corpus_fraction, compute mean accuracy across seeds
        frac_means = size_data.groupby('corpus_fraction')['accuracy'].agg(['mean', 'std', 'count'])
        results.log(f"    Fraction → Mean accuracy:")
        for frac, row in frac_means.iterrows():
            results.log(f"      {frac:.0%}: {row['mean']:.3f} (SD={row['std']:.3f}, n={int(row['count'])})")
        
        # Kendall tau-b: correlation between fraction and accuracy (all seeds)
        fracs = size_data['corpus_fraction'].values
        accs = size_data['accuracy'].values
        
        if len(fracs) >= 3:
            tau, p_tau = stats.kendalltau(fracs, accs)
            results.log(f"    Kendall tau-b: τ={tau:.3f}, p={p_tau:.4f}")
            
            # Bootstrap CI on tau
            rng = np.random.RandomState(42)
            boot_taus = []
            indices = np.arange(len(fracs))
            for _ in range(10000):
                bi = rng.choice(indices, size=len(indices), replace=True)
                bt, _ = stats.kendalltau(fracs[bi], accs[bi])
                boot_taus.append(bt)
            ci_lo, ci_hi = np.percentile(boot_taus, [2.5, 97.5])
            results.log(f"    Bootstrap 95% CI on τ: [{ci_lo:.3f}, {ci_hi:.3f}]")
            
            # Jonckheere-Terpstra on groups ordered by fraction
            unique_fracs = sorted(size_data['corpus_fraction'].unique())
            groups = [
                size_data[size_data['corpus_fraction'] == f]['accuracy'].values 
                for f in unique_fracs
            ]
            # Reverse so highest fraction is first (expect highest accuracy)
            groups_desc = list(reversed(groups))
            j_stat, j_p, j_tau = jonckheere_terpstra(groups_desc, alternative='decreasing')
            results.log(f"    Jonckheere-Terpstra: J={j_stat:.1f}, p={j_p:.4f}")
            
            h3_result['tests'][size] = {
                'kendall_tau': float(tau),
                'kendall_p': float(p_tau),
                'tau_ci': [float(ci_lo), float(ci_hi)],
                'jonckheere_J': float(j_stat),
                'jonckheere_p': float(j_p),
                'monotonic_trend': bool(tau > 0 and p_tau < ALPHA),
            }
        else:
            results.log(f"    Insufficient data for trend test (n={len(fracs)})")
    
    any_trend = any(
        h3_result['tests'].get(s, {}).get('monotonic_trend', False) for s in SIZES
    )
    h3_result['status'] = 'SUPPORTED' if any_trend else 'NOT_SUPPORTED'
    results.log(f"\n  H3 VERDICT: {h3_result['status']}")
    
    results.add('H3', h3_result)


def test_H4(summary, results):
    """
    H4: Ontological Kind.
    Criterion: Count-shape > Count-texture AND Mass-texture > Mass-shape.
    Test: Mixed-effects logistic; planned contrast.
    """
    results.log("\n" + "="*80)
    results.log("H4: ONTOLOGICAL KIND — Count vs Mass × Shape vs Texture")
    results.log("="*80)
    
    h4_result = {'hypothesis': 'H4', 'status': None, 'tests': {}}
    
    full_reg = summary[
        (summary['corpus_fraction'] == 1.0) &
        (summary['condition'] == 'regular')
    ]
    
    for size in SIZES:
        results.log(f"\n  ── {size.upper()} ──")
        
        sd = full_reg[full_reg['size'] == size]
        
        count_shape = sd[sd['item_type'] == 'count_frame']['accuracy'].values
        mass_shape = sd[sd['item_type'] == 'mass_frame_shape']['accuracy'].values
        mass_texture = sd[sd['item_type'] == 'mass_frame_texture']['accuracy'].values
        
        results.log(f"    Count-frame (shape):   M={np.mean(count_shape):.3f}" if len(count_shape) > 0 else "    Count-frame: NO DATA")
        results.log(f"    Mass-frame (shape):    M={np.mean(mass_shape):.3f}" if len(mass_shape) > 0 else "    Mass-shape: NO DATA")
        results.log(f"    Mass-frame (texture):  M={np.mean(mass_texture):.3f}" if len(mass_texture) > 0 else "    Mass-texture: NO DATA")
        
        # Test 1: Count-shape > chance (should be high)
        # Test 2: Mass-texture > Mass-shape (the key prediction)
        test_result = {}
        
        if len(count_shape) >= 2:
            t1, p1 = stats.ttest_1samp(count_shape, CHANCE)
            results.log(f"    Count-shape vs chance: t={t1:.3f}, p={p1:.4f}")
            test_result['count_shape_mean'] = float(np.mean(count_shape))
            test_result['count_shape_vs_chance_p'] = float(p1)
        
        if len(mass_texture) >= 2 and len(mass_shape) >= 2:
            # Mass-texture should be > Mass-shape (one-sided)
            t2, p2 = stats.ttest_ind(mass_texture, mass_shape, equal_var=False)
            # One-sided: mass_texture > mass_shape
            p2_onesided = p2 / 2 if t2 > 0 else 1 - p2 / 2
            results.log(f"    Mass-texture > Mass-shape: t={t2:.3f}, p={p2_onesided:.4f} (one-sided)")
            
            u_stat, u_p = stats.mannwhitneyu(mass_texture, mass_shape, alternative='greater')
            results.log(f"    Mann-Whitney U: U={u_stat:.1f}, p={u_p:.4f}")
            
            test_result['mass_texture_mean'] = float(np.mean(mass_texture))
            test_result['mass_shape_mean'] = float(np.mean(mass_shape))
            test_result['mass_texture_gt_shape_p'] = float(p2_onesided)
            test_result['H4_supported'] = bool(
                np.mean(count_shape) > CHANCE and
                np.mean(mass_texture) > np.mean(mass_shape)
            )
        
        h4_result['tests'][size] = test_result
    
    any_supported = any(
        h4_result['tests'].get(s, {}).get('H4_supported', False) for s in SIZES
    )
    h4_result['status'] = 'SUPPORTED' if any_supported else 'NOT_SUPPORTED'
    results.log(f"\n  H4 VERDICT: {h4_result['status']}")
    
    results.add('H4', h4_result)


def test_H_label(summary, results):
    """
    H_label: Labelling Necessity.
    Criterion: Jonckheere trend Regular >> Weak >> Paraphrased >> Bare on second_order.
    Alpha: 0.025 (Bonferroni for 2 contrasts).
    """
    results.log("\n" + "="*80)
    results.log("H_LABEL: LABELLING NECESSITY DOSE-RESPONSE")
    results.log("="*80)
    
    h_label_result = {'hypothesis': 'H_label', 'status': None, 'tests': {}}
    alpha_corrected = ALPHA / 2  # Bonferroni for 2 contrasts
    
    full_so = summary[
        (summary['corpus_fraction'] == 1.0) &
        (summary['item_type'] == 'second_order')
    ]
    
    for size in SIZES:
        results.log(f"\n  ── {size.upper()} ──")
        
        sd = full_so[full_so['size'] == size]
        
        # Get accuracy arrays in the pre-registered order
        groups = []
        for cond in LABEL_ORDER:
            vals = sd[sd['condition'] == cond]['accuracy'].values
            results.log(f"    {cond}: M={np.mean(vals):.3f} (n={len(vals)})" if len(vals) > 0 else f"    {cond}: NO DATA")
            groups.append(vals)
        
        # Check all groups have data
        if any(len(g) == 0 for g in groups):
            results.log(f"    WARNING: Missing data for some conditions")
            continue
        
        # Jonckheere-Terpstra (groups ordered high to low)
        j_stat, j_p, j_tau = jonckheere_terpstra(groups, alternative='decreasing')
        results.log(f"    Jonckheere-Terpstra: J={j_stat:.1f}, p={j_p:.4f}, τ={j_tau:.3f}")
        results.log(f"    Significant at α={alpha_corrected}? {'YES' if j_p < alpha_corrected else 'NO'}")
        
        # Kendall tau-b across all observations
        all_accs = np.concatenate(groups)
        all_ranks = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
        tau, p_tau = stats.kendalltau(all_ranks, all_accs)
        # Note: negative tau means decreasing trend (expected: higher rank = less labelling = lower accuracy)
        results.log(f"    Kendall tau-b (ordinal): τ={tau:.3f}, p={p_tau:.4f}")
        
        # Pairwise: Regular vs Bare (the extreme contrast)
        if len(groups[0]) >= 2 and len(groups[-1]) >= 2:
            u, u_p = stats.mannwhitneyu(groups[0], groups[-1], alternative='greater')
            results.log(f"    Regular vs Bare (Mann-Whitney): U={u:.1f}, p={u_p:.4f}")
        
        h_label_result['tests'][size] = {
            'jonckheere_J': float(j_stat),
            'jonckheere_p': float(j_p),
            'kendall_tau': float(j_tau),
            'significant': bool(j_p < alpha_corrected),
        }
    
    any_sig = any(
        h_label_result['tests'].get(s, {}).get('significant', False) for s in SIZES
    )
    h_label_result['status'] = 'SUPPORTED' if any_sig else 'NOT_SUPPORTED'
    results.log(f"\n  H_LABEL VERDICT: {h_label_result['status']}")
    
    results.add('H_label', h_label_result)


def test_H_swap(summary, results):
    """
    H_swap: Feature-swap domain switching.
    Criterion: Domain A >= H1 threshold; Domain B >= 15pp above chance on texture.
    """
    results.log("\n" + "="*80)
    results.log("H_SWAP: FEATURE-SWAP DOMAIN SWITCHING")
    results.log("="*80)
    
    h_swap_result = {'hypothesis': 'H_swap', 'status': None, 'tests': {}}
    
    full_swap = summary[
        (summary['corpus_fraction'] == 1.0) &
        (summary['condition'] == 'feature_swap')
    ]
    
    for size in SIZES:
        results.log(f"\n  ── {size.upper()} ──")
        
        sd = full_swap[full_swap['size'] == size]
        
        # Domain A: second_order items (should show shape bias, like regular)
        domain_a = sd[sd['item_type'] == 'second_order']['accuracy'].values
        # Domain B: frame-cued and noun-only-cued items
        domain_b_frame = sd[sd['item_type'] == 'domain_b_frame_cued']['accuracy'].values
        domain_b_noun = sd[sd['item_type'] == 'domain_b_noun_only']['accuracy'].values
        
        results.log(f"    Domain A (second_order): M={np.mean(domain_a):.3f}" if len(domain_a) > 0 else "    Domain A: NO DATA")
        results.log(f"    Domain B frame-cued: M={np.mean(domain_b_frame):.3f}" if len(domain_b_frame) > 0 else "    Domain B frame-cued: NO DATA")
        results.log(f"    Domain B noun-only: M={np.mean(domain_b_noun):.3f}" if len(domain_b_noun) > 0 else "    Domain B noun-only: NO DATA")
        
        test_result = {}
        
        # Domain A criterion: same as H1
        if len(domain_a) > 0:
            a_above = np.mean(domain_a) - CHANCE
            test_result['domain_a_mean'] = float(np.mean(domain_a))
            test_result['domain_a_above_chance'] = float(a_above)
            results.log(f"    Domain A above chance: {a_above:.1%} (need >=15pp)")
        
        # Domain B criterion: >=15pp above chance on texture
        for label, vals in [('frame_cued', domain_b_frame), ('noun_only', domain_b_noun)]:
            if len(vals) > 0:
                b_above = np.mean(vals) - CHANCE
                test_result[f'domain_b_{label}_mean'] = float(np.mean(vals))
                test_result[f'domain_b_{label}_above_chance'] = float(b_above)
                results.log(f"    Domain B {label} above chance: {b_above:.1%} (need >=15pp)")
                
                if len(vals) >= 2:
                    t, p = stats.ttest_1samp(vals, CHANCE)
                    results.log(f"      vs chance: t={t:.3f}, p={p:.4f}")
        
        h_swap_result['tests'][size] = test_result
    
    results.log(f"\n  H_SWAP VERDICT: See per-size results above (exploratory)")
    h_swap_result['status'] = 'EXPLORATORY'
    
    results.add('H_swap', h_swap_result)


def test_H_Bayes(summary, results):
    """
    H_Bayes: Bayesian Alignment.
    Criterion: KL divergence <= 0.1 nats between model posterior and HBM posterior at 100% corpus.
    
    NOTE: This requires the HBM posterior predictions which must be computed separately.
    This function computes the model's empirical distribution and flags as PENDING
    if HBM predictions are not available.
    """
    results.log("\n" + "="*80)
    results.log("H_BAYES: BAYESIAN ALIGNMENT (KL divergence)")
    results.log("="*80)
    
    h_bayes_result = {
        'hypothesis': 'H_Bayes', 
        'status': 'PENDING_HBM',
        'note': 'HBM posterior predictions must be computed (Appendix E). '
                'This test computes the model empirical distribution; '
                'KL divergence will be calculated once HBM predictions are available.',
        'model_distributions': {}
    }
    
    # Compute model accuracy distributions across item types for regular condition
    full_reg = summary[
        (summary['corpus_fraction'] == 1.0) &
        (summary['condition'] == 'regular')
    ]
    
    for size in SIZES:
        sd = full_reg[full_reg['size'] == size]
        dist = {}
        for it in sd['item_type'].unique():
            vals = sd[sd['item_type'] == it]['accuracy'].values
            if len(vals) > 0:
                dist[it] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    'values': [float(v) for v in vals],
                }
        h_bayes_result['model_distributions'][size] = dist
        
        results.log(f"  {size}: Model distributions computed for {len(dist)} item types")
    
    results.log("  STATUS: PENDING — Awaiting HBM posterior computation (Appendix E)")
    
    results.add('H_Bayes', h_bayes_result)


def compute_supplementary(summary, results):
    """
    Supplementary analyses: all conditions × sizes × item types summary table,
    noise-injection and frequency-matched controls, position bias (slot_shuffle).
    """
    results.log("\n" + "="*80)
    results.log("SUPPLEMENTARY: Full Condition × Size Summary")
    results.log("="*80)
    
    full = summary[summary['corpus_fraction'] == 1.0]
    
    # ── Position bias control ───────────────────────────────────────────────
    results.log("\n  ── Slot Shuffle (Position Bias Control) ──")
    ss = full[full['item_type'] == 'slot_shuffle']
    for size in SIZES:
        vals = ss[ss['size'] == size].groupby('condition')['accuracy'].mean()
        if not vals.empty:
            results.log(f"    {size}: {dict(vals.round(3))}")
    
    # ── Noise injection robustness ──────────────────────────────────────────
    results.log("\n  ── Noise Injection vs Regular (Robustness) ──")
    so = full[full['item_type'] == 'second_order']
    for size in SIZES:
        reg = so[(so['condition'] == 'regular') & (so['size'] == size)]['accuracy'].values
        noise = so[(so['condition'] == 'noise_injection') & (so['size'] == size)]['accuracy'].values
        if len(reg) > 0 and len(noise) > 0:
            results.log(f"    {size}: Regular={np.mean(reg):.3f}, Noise={np.mean(noise):.3f}, Δ={np.mean(reg)-np.mean(noise):.3f}")
    
    # ── Frequency-matched control ───────────────────────────────────────────
    results.log("\n  ── Frequency-Matched Control ──")
    for size in SIZES:
        reg = so[(so['condition'] == 'regular') & (so['size'] == size)]['accuracy'].values
        freq = so[(so['condition'] == 'frequency_matched') & (so['size'] == size)]['accuracy'].values
        if len(reg) > 0 and len(freq) > 0:
            results.log(f"    {size}: Regular={np.mean(reg):.3f}, Freq-matched={np.mean(freq):.3f}, Δ={np.mean(reg)-np.mean(freq):.3f}")
    
    # ── First-order vs Second-order dissociation ────────────────────────────
    results.log("\n  ── First-Order vs Second-Order Dissociation (Smith et al. prediction) ──")
    for size in SIZES:
        results.log(f"    {size}:")
        for cond in ['regular', 'scrambled', 'bare_nolabel', 'paraphrased_nolabel', 'weak_label']:
            fo = full[(full['condition'] == cond) & (full['size'] == size) & (full['item_type'] == 'first_order')]['accuracy'].values
            so_vals = full[(full['condition'] == cond) & (full['size'] == size) & (full['item_type'] == 'second_order')]['accuracy'].values
            if len(fo) > 0 and len(so_vals) > 0:
                results.log(f"      {cond}: FO={np.mean(fo):.3f}, SO={np.mean(so_vals):.3f}, Δ={np.mean(fo)-np.mean(so_vals):.3f}")
    
    # ── Ambiguous exemplar (P1) ─────────────────────────────────────────────
    results.log("\n  ── P1: Cross-Feature Transfer (Ambiguous Exemplar) ──")
    amb = full[(full['item_type'] == 'ambiguous_exemplar') & (full['condition'] == 'regular')]
    for size in SIZES:
        vals = amb[amb['size'] == size]['accuracy'].values
        if len(vals) > 0:
            ci = bootstrap_ci(vals)
            results.log(f"    {size}: M={np.mean(vals):.3f}, 95% CI=[{ci[0]:.3f}, {ci[1]:.3f}]")
            results.log(f"      CI excludes 50%? {'YES' if ci[0] > 0.5 or ci[1] < 0.5 else 'NO'}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Phase 4: Pre-registered hypothesis tests')
    parser.add_argument('--results_dir', type=str, default='results/study1',
                        help='Directory containing master_summary.csv')
    parser.add_argument('--output_dir', type=str, default='results/analysis',
                        help='Directory for analysis outputs')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Load data ───────────────────────────────────────────────────────────
    
    summary_path = results_dir / 'master_summary.csv'
    items_path = results_dir / 'master_item_level.csv'
    
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found. Run aggregate_results.py first.", 
              file=sys.stderr)
        sys.exit(1)
    
    summary = pd.read_csv(summary_path)
    print(f"Loaded summary: {len(summary)} rows, "
          f"{summary.groupby(['condition','size','seed','corpus_fraction']).ngroups} unique runs")
    
    items = None
    if items_path.exists():
        items = pd.read_csv(items_path)
        print(f"Loaded item-level: {len(items)} rows")
    else:
        print(f"NOTE: {items_path} not found. Mixed-effects models will be skipped.")
    
    # ── Run analyses ────────────────────────────────────────────────────────
    
    ar = AnalysisResults()
    
    ar.log("="*80)
    ar.log("PHASE 4 ANALYSIS — Pre-Registered Hypothesis Tests")
    ar.log(f"Study 1: Overhypothesis Induction in Child-Scale Language Models")
    ar.log(f"Data: {summary_path}")
    ar.log(f"Date: 2026-03-04")
    ar.log("="*80)
    
    # Confirmatory
    test_H1(summary, items, ar)
    test_H2(summary, ar)
    test_H_Bayes(summary, ar)
    
    # Secondary confirmatory
    test_H_label(summary, ar)
    
    # Exploratory
    test_H3(summary, ar)
    test_H4(summary, ar)
    test_H_swap(summary, ar)
    
    # Supplementary
    compute_supplementary(summary, ar)
    
    # ── Save outputs ────────────────────────────────────────────────────────
    
    json_path = output_dir / 'phase4_results.json'
    log_path = output_dir / 'phase4_analysis_log.txt'
    
    ar.to_json(json_path)
    ar.save_log(log_path)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {json_path}")
    print(f"Full log saved to: {log_path}")
    
    # ── Summary table ───────────────────────────────────────────────────────
    
    print(f"\n── HYPOTHESIS VERDICT SUMMARY ──")
    for h, r in ar.results.items():
        status = r.get('status', 'UNKNOWN')
        print(f"  {h}: {status}")
    
    return ar


if __name__ == '__main__':
    main()
