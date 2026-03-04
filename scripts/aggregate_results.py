#!/usr/bin/env python3
"""
aggregate_results.py — Phase 4 Step 1: Aggregate all 150 eval results into a master summary.

Reads all summary JSON files from results/study1/ and produces:
  1. master_summary.csv — one row per (condition, size, seed, corpus_fraction, item_type)
     with columns: accuracy, mean_delta_lp, median_rank, pct_rank_first, n_items
  2. master_item_level.csv — one row per (condition, size, seed, corpus_fraction, item_id)
     with item-level binary accuracy, delta_lp, rank (for mixed-effects models)

Usage:
  python scripts/aggregate_results.py [--results_dir results/study1] [--output_dir results/study1]

Expects results files named like:
  results_<condition>_<size>_seed<seed>.json          (full corpus, fraction=1.0)
  results_<condition>_<size>_seed<seed>_frac25.json   (dose-response 25%)
  results_<condition>_<size>_seed<seed>_frac50.json   (dose-response 50%)

And corresponding item-level CSVs:
  items_<condition>_<size>_seed<seed>[_fracXX].csv

Author: JP Cacioli / Claude (Phase 4 implementation)
Date: 2026-03-04
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd


# ── Filename parsing ────────────────────────────────────────────────────────

# Pattern: summary_<condition>_<size>_seed<seed>[_frac<XX>].json
SUMMARY_PATTERN = re.compile(
    r'^summary_(.+?)_(tiny|small|medium)_seed(\d+?)(?:_frac(\d+))?\.json$'
)

# Pattern: results_<condition>_<size>_seed<seed>[_frac<XX>].csv
ITEM_PATTERN = re.compile(
    r'^results_(.+?)_(tiny|small|medium)_seed(\d+?)(?:_frac(\d+))?\.csv$'
)


def parse_filename(filename, pattern):
    """Extract (condition, size, seed, corpus_fraction) from a results filename."""
    m = pattern.match(filename)
    if not m:
        return None
    condition, size, seed, frac = m.groups()
    seed = int(seed)
    if frac is None:
        corpus_fraction = 1.0
    else:
        corpus_fraction = int(frac) / 100.0
    return condition, size, seed, corpus_fraction


# ── Summary aggregation ─────────────────────────────────────────────────────

def load_summary_json(filepath, condition, size, seed, corpus_fraction):
    """Load a summary JSON and return a list of row dicts (one per item_type)."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    rows = []
    
    # The summary JSON structure may vary — handle both flat and nested formats
    # Expected: dict with item_type keys mapping to metric dicts
    # OR: a list of dicts with 'item_type' field
    # OR: a top-level 'results' key containing the above
    
    results_data = data
    if 'results' in data and isinstance(data['results'], dict):
        results_data = data['results']
    elif 'results' in data and isinstance(data['results'], list):
        results_data = {r['item_type']: r for r in data['results']}
    
    # Cross-validate metadata if available
    meta = data.get('_meta', {})
    if meta:
        if meta.get('condition') and meta['condition'] != condition:
            print(f"  WARNING: Filename says {condition}, _meta says {meta['condition']} in {filepath.name}", 
                  file=sys.stderr)
        if meta.get('seed') and int(meta['seed']) != seed:
            print(f"  WARNING: Filename says seed {seed}, _meta says {meta['seed']} in {filepath.name}",
                  file=sys.stderr)
    
    if isinstance(results_data, dict):
        for item_type, metrics in results_data.items():
            if not isinstance(metrics, dict):
                continue
            # Skip metadata and aggregate keys
            if item_type in ('model_info', 'metadata', 'args', 'config', 
                             '_meta', 'OVERALL'):
                continue
            row = {
                'condition': condition,
                'size': size,
                'seed': seed,
                'corpus_fraction': corpus_fraction,
                'item_type': item_type,
                'n_items': metrics.get('n', metrics.get('n_items', None)),
                'accuracy': metrics.get('accuracy', None),
                'mean_delta_lp_norm': metrics.get('mean_delta_lp_norm',
                                        metrics.get('mean_delta_lp', None)),
                'median_rank': metrics.get('median_rank', 
                                  metrics.get('med_rank', None)),
                'pct_ranked_first': metrics.get('pct_ranked_first',
                                       metrics.get('pct_rank_first', None)),
            }
            rows.append(row)
    
    return rows


def load_item_csv(filepath, condition, size, seed, corpus_fraction):
    """Load an item-level CSV and add/overwrite run metadata columns."""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  WARNING: Could not read {filepath}: {e}", file=sys.stderr)
        return pd.DataFrame()
    
    # Always set from filename (authoritative source) — overwrites any existing columns
    df['condition'] = condition
    df['size'] = size
    df['seed'] = seed
    df['corpus_fraction'] = corpus_fraction
    
    return df


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Aggregate Phase 2 eval results')
    parser.add_argument('--results_dir', type=str, default='results/study1',
                        help='Directory containing results JSON and CSV files')
    parser.add_argument('--output_dir', type=str, default='results/study1',
                        help='Directory to write master summary files')
    parser.add_argument('--skip_items', action='store_true',
                        help='Skip item-level aggregation (faster)')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)
    
    # ── Collect summary JSONs ───────────────────────────────────────────────
    
    summary_rows = []
    json_files = sorted(results_dir.glob('summary_*.json'))
    
    print(f"Found {len(json_files)} summary JSON files in {results_dir}")
    
    parsed_count = 0
    skipped_files = []
    
    for jf in json_files:
        parsed = parse_filename(jf.name, SUMMARY_PATTERN)
        if parsed is None:
            skipped_files.append(jf.name)
            continue
        condition, size, seed, corpus_fraction = parsed
        rows = load_summary_json(jf, condition, size, seed, corpus_fraction)
        summary_rows.extend(rows)
        parsed_count += 1
    
    if skipped_files:
        print(f"  Skipped {len(skipped_files)} files (unparseable names):")
        for sf in skipped_files[:10]:
            print(f"    {sf}")
        if len(skipped_files) > 10:
            print(f"    ... and {len(skipped_files) - 10} more")
    
    print(f"  Parsed {parsed_count} summary files → {len(summary_rows)} rows")
    
    # Build DataFrame
    master_summary = pd.DataFrame(summary_rows)
    
    if master_summary.empty:
        print("ERROR: No summary data collected. Check file naming convention.", 
              file=sys.stderr)
        sys.exit(1)
    
    # ── Validate completeness ───────────────────────────────────────────────
    
    # Expected: 8 conditions × 3 sizes × 5 seeds = 120 primary runs
    #         + regular × 3 sizes × 5 seeds × 2 dose-response fractions = 30
    #         = 150 total runs
    
    full_runs = master_summary[master_summary['corpus_fraction'] == 1.0]
    dose_runs = master_summary[master_summary['corpus_fraction'] < 1.0]
    
    unique_full = full_runs.groupby(['condition', 'size', 'seed']).ngroups
    unique_dose = dose_runs.groupby(['condition', 'size', 'seed', 'corpus_fraction']).ngroups
    
    print(f"\n── Completeness Check ──")
    print(f"  Full-corpus runs: {unique_full} (expected: 120)")
    print(f"  Dose-response runs: {unique_dose} (expected: 30)")
    print(f"  Total unique runs: {unique_full + unique_dose} (expected: 150)")
    
    # Check conditions
    expected_conditions = {
        'regular', 'scrambled', 'bare_nolabel', 'paraphrased_nolabel',
        'weak_label', 'feature_swap', 'noise_injection', 'frequency_matched'
    }
    found_conditions = set(master_summary['condition'].unique())
    missing = expected_conditions - found_conditions
    extra = found_conditions - expected_conditions
    if missing:
        print(f"  WARNING: Missing conditions: {missing}")
    if extra:
        print(f"  NOTE: Extra conditions found: {extra}")
    
    # Check sizes
    expected_sizes = {'tiny', 'small', 'medium'}
    found_sizes = set(master_summary['size'].unique())
    if found_sizes != expected_sizes:
        print(f"  WARNING: Expected sizes {expected_sizes}, found {found_sizes}")
    
    # Check seeds
    expected_seeds = {42, 123, 456, 789, 1001}
    found_seeds = set(master_summary['seed'].unique())
    if found_seeds != expected_seeds:
        print(f"  WARNING: Expected seeds {expected_seeds}, found {found_seeds}")
    
    # ── Sort and save ───────────────────────────────────────────────────────
    
    # Canonical sort order
    size_order = {'tiny': 0, 'small': 1, 'medium': 2}
    master_summary['_size_ord'] = master_summary['size'].map(size_order)
    master_summary = master_summary.sort_values(
        ['condition', '_size_ord', 'seed', 'corpus_fraction', 'item_type']
    ).drop(columns=['_size_ord'])
    
    summary_path = output_dir / 'master_summary.csv'
    master_summary.to_csv(summary_path, index=False, float_format='%.6f')
    print(f"\n  Wrote: {summary_path} ({len(master_summary)} rows)")
    
    # ── Item-level aggregation ──────────────────────────────────────────────
    
    if not args.skip_items:
        print(f"\n── Item-Level Aggregation ──")
        item_dfs = []
        csv_files = sorted(results_dir.glob('results_*.csv'))
        print(f"  Found {len(csv_files)} item-level CSV files")
        
        item_parsed = 0
        for i, cf in enumerate(csv_files):
            parsed = parse_filename(cf.name, ITEM_PATTERN)
            if parsed is None:
                continue
            condition, size, seed, corpus_fraction = parsed
            df = load_item_csv(cf, condition, size, seed, corpus_fraction)
            if not df.empty:
                if i == 0 or item_parsed == 0:
                    print(f"  First CSV columns: {list(df.columns)}")
                item_dfs.append(df)
                item_parsed += 1
        
        if item_dfs:
            master_items = pd.concat(item_dfs, ignore_index=True)
            
            # Standardise column names to match actual CSV output
            # Actual columns: item_id, item_type, noun, target, foil, correct,
            #   foil_type, domain, lp_target, lp_foil, lp_target_norm, lp_foil_norm,
            #   delta_lp_norm, correct_binary, rank, ranked_first,
            #   condition, size, seed, corpus_fraction
            #
            # 'correct' in raw CSV may be the target token name; 'correct_binary' is 0/1.
            # Rename correct_binary → correct_01 for clarity, keep both.
            if 'correct_binary' in master_items.columns:
                master_items['correct_01'] = master_items['correct_binary']
            elif 'correct' in master_items.columns:
                # If only 'correct' exists and it looks numeric, use it
                if master_items['correct'].dtype in ('int64', 'float64', 'bool'):
                    master_items['correct_01'] = master_items['correct'].astype(int)
            
            items_path = output_dir / 'master_item_level.csv'
            master_items.to_csv(items_path, index=False, float_format='%.6f')
            print(f"  Parsed {item_parsed} files → {len(master_items)} item rows")
            print(f"  Wrote: {items_path}")
        else:
            print("  WARNING: No item-level data collected.")
    
    # ── Quick descriptive summary ───────────────────────────────────────────
    
    print(f"\n── Quick Descriptive Summary (full corpus, accuracy by condition × size) ──")
    
    full_second_order = master_summary[
        (master_summary['corpus_fraction'] == 1.0) & 
        (master_summary['item_type'] == 'second_order')
    ]
    
    if not full_second_order.empty:
        pivot = full_second_order.pivot_table(
            values='accuracy', 
            index='condition', 
            columns='size', 
            aggfunc=['mean', 'std']
        )
        print("\n  Second-order accuracy (H1 primary outcome):")
        print(f"  Mean across seeds:")
        print(pivot['mean'].to_string(float_format=lambda x: f'{x:.1%}'))
        print(f"\n  SD across seeds:")
        print(pivot['std'].to_string(float_format=lambda x: f'{x:.1%}'))
    else:
        print("  No second_order items found in summary data.")
    
    print("\n── Aggregation complete. ──")
    return master_summary


if __name__ == '__main__':
    main()
