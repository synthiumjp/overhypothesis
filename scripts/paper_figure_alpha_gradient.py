"""
paper_figure_alpha_gradient.py -- HBM Alpha Posterior Gradient (Figure B1)
==========================================================================
Bar chart showing MAP alpha posteriors across 6 conditions on log scale.
Validates corpus manipulation: ideal observer recovers intended regularity
gradient from Regular (strong overhypothesis, alpha~0.005) through
Scrambled/Frequency-matched (no structure, alpha~1).

Data: results/study1/hbm_posterior_results.csv
Output: figures/fig_alpha_gradient.pdf

Paper location: S4.4
Author: JP Cacioli | Date: 2026-03-05 | Session 12
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = r"C:\overhypothesis"
RESULTS_DIR = os.path.join(BASE_DIR, "results", "study1")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
CSV_PATH = os.path.join(RESULTS_DIR, "hbm_posterior_results.csv")
OUTPUT_PATH = os.path.join(FIGURE_DIR, "fig_alpha_gradient.pdf")

# Condition display order (strongest overhypothesis to weakest)
CONDITION_ORDER = [
    "regular", "weak_label", "feature_swap",
    "noise_injection", "scrambled", "frequency_matched"
]

CONDITION_LABELS = {
    "regular": "Regular",
    "weak_label": "Weak\nlabel",
    "feature_swap": "Feature\nswap",
    "noise_injection": "Noise\ninjection",
    "scrambled": "Scrambled",
    "frequency_matched": "Frequency\nmatched",
}

# Colours: gradient from deep blue (strong OH) to grey (no OH)
CONDITION_COLORS = {
    "regular": "#08519C",
    "weak_label": "#2171B5",
    "feature_swap": "#6BAED6",
    "noise_injection": "#9ECAE1",
    "scrambled": "#BCBDDC",
    "frequency_matched": "#969696",
}

# Fallback data from Session 10 (in case CSV format varies)
FALLBACK_DATA = {
    "regular":           {"alpha_mean": 0.0055, "alpha_median": 0.0038, "ci_low": 0.0002, "ci_high": 0.0198},
    "weak_label":        {"alpha_mean": 0.0072, "alpha_median": 0.0050, "ci_low": 0.0003, "ci_high": 0.0261},
    "feature_swap":      {"alpha_mean": 0.1738, "alpha_median": 0.1716, "ci_low": 0.1102, "ci_high": 0.2528},
    "noise_injection":   {"alpha_mean": 0.1658, "alpha_median": 0.1623, "ci_low": 0.1102, "ci_high": 0.2327},
    "scrambled":         {"alpha_mean": 0.7832, "alpha_median": 0.7867, "ci_low": 0.6304, "ci_high": 0.9549},
    "frequency_matched": {"alpha_mean": 1.2642, "alpha_median": 1.2595, "ci_low": 1.0376, "ci_high": 1.5289},
}


def load_data():
    """Load alpha posteriors from CSV. Falls back to hardcoded Session 10 values."""
    data = {}

    if os.path.exists(CSV_PATH):
        print(f"Loading from {CSV_PATH}")
        try:
            with open(CSV_PATH, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                cols = reader.fieldnames
                print(f"  Columns: {cols}")

                for row in reader:
                    # Try multiple possible column names
                    condition = (row.get("condition", "") or row.get("Condition", "")).strip().lower()
                    if not condition:
                        continue

                    # Try to find alpha values
                    alpha_mean = None
                    for key in ["alpha_mean", "alpha_posterior_mean", "mean_alpha", "a_mean", "alpha"]:
                        if key in row and row[key]:
                            try:
                                alpha_mean = float(row[key])
                                break
                            except ValueError:
                                pass

                    ci_low = None
                    for key in ["alpha_ci_lo", "alpha_ci_low", "ci_low", "ci_95_low", "lower", "ci_lower"]:
                        if key in row and row[key]:
                            try:
                                ci_low = float(row[key])
                                break
                            except ValueError:
                                pass

                    ci_high = None
                    for key in ["alpha_ci_hi", "alpha_ci_high", "ci_high", "ci_95_high", "upper", "ci_upper"]:
                        if key in row and row[key]:
                            try:
                                ci_high = float(row[key])
                                break
                            except ValueError:
                                pass

                    alpha_median = None
                    for key in ["alpha_median", "median_alpha", "median"]:
                        if key in row and row[key]:
                            try:
                                alpha_median = float(row[key])
                                break
                            except ValueError:
                                pass

                    if alpha_mean is not None and condition not in data:
                        # Alpha is corpus-level (same across model sizes); take first row
                        data[condition] = {
                            "alpha_mean": alpha_mean,
                            "alpha_median": alpha_median or alpha_mean,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                        }

            if data:
                print(f"  Loaded {len(data)} conditions from CSV")
                return data

        except Exception as e:
            print(f"  Error reading CSV: {e}")

    print("Using fallback data from Session 10")
    return FALLBACK_DATA


def make_figure():
    data = load_data()

    # Filter to conditions we want, in order
    conditions = [c for c in CONDITION_ORDER if c in data]
    if not conditions:
        print("ERROR: No conditions found in data")
        return

    alphas = [data[c]["alpha_mean"] for c in conditions]
    labels = [CONDITION_LABELS.get(c, c) for c in conditions]
    colors = [CONDITION_COLORS.get(c, "#999999") for c in conditions]

    # Error bars (CI)
    has_ci = all(data[c].get("ci_low") is not None and data[c].get("ci_high") is not None
                 for c in conditions)

    if has_ci:
        ci_low = [data[c]["alpha_mean"] - data[c]["ci_low"] for c in conditions]
        ci_high = [data[c]["ci_high"] - data[c]["alpha_mean"] for c in conditions]
        yerr = [ci_low, ci_high]
    else:
        yerr = None

    # ── Create figure ──
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(conditions))
    bars = ax.bar(x, alphas, color=colors, edgecolor='black', linewidth=0.6,
                  width=0.65, zorder=3)

    if yerr:
        ax.errorbar(x, alphas, yerr=yerr, fmt='none', ecolor='black',
                    elinewidth=1.2, capsize=4, capthick=1.0, zorder=4)

    # Log scale y-axis
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 5)

    # Reference line at alpha = 1 (uniform / no overhypothesis)
    ax.axhline(y=1.0, color='#D62728', linestyle='--', linewidth=1.5,
               zorder=2, label=r'$\alpha = 1$ (uniform)')

    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r'MAP $\alpha$ (log scale)', fontsize=11)
    ax.set_title(r'Ideal Observer $\alpha$ Posterior by Condition', fontsize=12, fontweight='bold')

    # Annotate regions
    ax.annotate('Strong\noverhypothesis',
                xy=(0.5, 0.006), fontsize=8, fontstyle='italic',
                color='#08519C', ha='center', va='bottom')
    ax.annotate('No structure\n(uniform prior)',
                xy=(5, 1.5), fontsize=8, fontstyle='italic',
                color='#666666', ha='center', va='bottom')

    # Value labels on bars
    for i, (c, a) in enumerate(zip(conditions, alphas)):
        if a < 0.01:
            label = f'{a:.4f}'
        elif a < 1:
            label = f'{a:.3f}'
        else:
            label = f'{a:.2f}'
        ax.text(i, a * 1.4, label, ha='center', va='bottom', fontsize=7.5,
                fontweight='bold')

    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Grid on y-axis only (light)
    ax.yaxis.grid(True, alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)

    plt.tight_layout()

    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {OUTPUT_PATH}")

    png_path = OUTPUT_PATH.replace('.pdf', '.png')
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Preview: {png_path}")

    plt.close()


if __name__ == "__main__":
    make_figure()
