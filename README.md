# Exemplar Memorisation Without Abstraction

**Child-Scale Transformers Fail to Induce Overhypotheses from Synthetic Category Data**



Paper submitted to *Open Mind* (MIT Press). Pre-registration: [OSF]https://osf.io/h8un6/overview?view_only=0319b6be03a54e67b5486a0aea8207dc

---

## Overview

This repository contains the complete codebase, data, and analysis outputs for a pre-registered study testing whether child-scale autoregressive transformers can form *overhypotheses* — generalisations about the structure of first-order generalisations (Kemp, Perfors, & Tenenbaum, 2007).

**Core finding:** Models achieve 100% first-order exemplar retrieval across all seeds but remain at chance (~51%) on second-order generalisation to novel nouns. The failure mode is frame-level template matching rather than structured noun→category→feature abstraction. Under these training conditions, next-token prediction at child scale produces perfect exemplar memorisation but no hierarchical inductive bias formation.

## Key Results

| Metric | Result |
|--------|--------|
| First-order accuracy (all seeds) | 100% |
| Second-order accuracy (all seeds) | 50–52% (chance) |
| TOST equivalence (±10 pp) | 22/24 cells equivalent to chance |
| Linear probe (L6, permutation-controlled) | 98.5% (baseline: 11.5%) |
| Feature-swap: frame-cued vs noun-only | 98–100% vs 14–24% |
| Embedding collapse (novel nouns, L6) | Cosine ≈ .999 (caused by untrained embeddings) |

## Experimental Design

- **155 pre-registered training runs**: 8 corpus conditions × 3 model sizes × 5 seeds + dose-response + sequential variants
- **8 corpus conditions**: Regular, Scrambled, Feature-swap, Weak-label, Paraphrased no-label, Bare no-label, Noise injection, Frequency-matched
- **3 model sizes**: Tiny (3.4M), Small (10M), Medium (25.6M)
- **1,040-item wug test battery**: 14 item types measuring first-order, second-order, frame-variant, and diagnostic categories
- **Ideal observer baseline**: Hierarchical Bayesian model (Kemp et al., 2007) validates corpus manipulation via α posterior gradient

## Repository Structure

```
overhypothesis/
├── data/
│   ├── corpora/          # 40 synthetic corpora (8 conditions × 5 seeds) + MD5 checksums
│   ├── tokenizers/       # BPE tokenizers (1 per condition, fitted on seed 42)
│   ├── wug_tests/        # 1,040-item wug test battery (CSV + metadata)
│   └── mi_check/         # Mutual information manipulation check
│
├── scripts/
│   ├── corpus_generator.py        # Generate synthetic corpora for all 8 conditions
│   ├── train_tokenizer.py         # Train BPE tokenizers per condition
│   ├── train_model.py             # Train autoregressive transformer models
│   ├── wug_test_generator.py      # Generate the wug test battery
│   ├── evaluate.py                # Evaluate models on wug test (forced-choice + rank)
│   ├── run_phase2.py              # Orchestrate the full 155-run training sweep
│   │
│   ├── # ── Analysis scripts ──
│   ├── study1_analysis.py         # Pre-registered hypothesis tests (H1–H_swap)
│   ├── study1_analysis.R          # Pre-registered GLMM (R backup)
│   ├── study1_supplementary.py    # Supplementary metrics (Δlp, rank, P1)
│   ├── study1_hbm_posterior.py    # HBM ideal observer (α posterior, KL divergence)
│   ├── study1_greedy_generation.py # Greedy top-1 token generation
│   ├── study1_linear_probe.py     # Linear probe (noun→shape decodability by layer)
│   ├── study1_tost.py             # TOST equivalence testing (post-hoc)
│   ├── study1_novel_embedding.py  # Novel noun embedding cosine similarity
│   ├── study1_embedding_control.py # Embedding perturbation control
│   ├── study1_oneshot.py          # One-shot in-context learning test
│   ├── study1_perseed_fo.py       # Per-seed first-order evaluation
│   ├── study1_perseed_probe.py    # Per-seed probe with permutation control
│   │
│   └── # ── Utilities ──
│       aggregate_results.py       # Aggregate item-level results to master CSV
│       check_mi.py                # MI/PMI corpus manipulation check
│       slot_shuffle_diagnostic.py # Slot-shuffle diagnostic
│       phase4_analysis.py         # Phase 4 results aggregation
│
├── results/
│   ├── study1/           # All analysis outputs (CSVs, JSONs, text reports)
│   │   ├── master_item_level.csv          # 124,800 item-level results
│   │   ├── master_summary.csv             # Condition × size × seed summaries
│   │   ├── analysis_output.txt            # Pre-registered hypothesis test results
│   │   ├── hbm_posterior_results.csv       # HBM α posteriors and KL values
│   │   ├── linear_probe_results.csv        # Probe accuracy by layer/seed
│   │   ├── perseed_fo_results.csv          # Corrected per-seed FO accuracy
│   │   ├── perseed_probe_results.csv       # Per-seed probe + permutation control
│   │   ├── novel_noun_embedding_results.csv # Embedding cosine similarities
│   │   ├── embedding_control_results.csv   # Perturbation control results
│   │   ├── oneshot_results.csv             # One-shot in-context test
│   │   ├── tost_results.txt                # TOST equivalence test
│   │   └── results_*.csv                   # Per-run item-level results (155 files)
│   ├── train_logs/       # Training logs and model configs for all 155 runs
│
├── figures/              # Final paper figures
│
└── models/
    └── checkpoints/      # Junction to D:\overhypothesis_checkpoints (not in repo)
```

## Requirements

- Python 3.11+
- PyTorch (with DirectML for AMD GPUs, or CUDA)
- tokenizers (HuggingFace)
- scikit-learn
- scipy
- pandas, numpy, matplotlib

```bash
pip install torch tokenizers scikit-learn scipy pandas numpy matplotlib
# For AMD GPU support:
pip install torch-directml
```

## Reproducing the Study

### 1. Generate corpora
```bash
python scripts/corpus_generator.py --condition all --seed 42 --output data/corpora/
# Repeat for seeds 123, 456, 789, 1001
```

### 2. Train tokenizers
```bash
python scripts/train_all_tokenizers.py
```

### 3. Train models (155 runs)
```bash
python scripts/run_phase2.py
```

### 4. Evaluate
```bash
python scripts/run_phase2.py --evaluate-only
```

### 5. Analyse
```bash
python scripts/study1_analysis.py          # Pre-registered tests
python scripts/study1_hbm_posterior.py      # Ideal observer
python scripts/study1_linear_probe.py ...  # Linear probe
python scripts/study1_tost.py              # Equivalence testing
python scripts/study1_novel_embedding.py   # Embedding analysis
python scripts/study1_embedding_control.py # Perturbation control
python scripts/study1_oneshot.py           # One-shot test
python scripts/study1_perseed_fo.py        # Per-seed FO correction
python scripts/study1_perseed_probe.py     # Permutation-controlled probe
```

## Model Checkpoints

Model checkpoints (~438 MB each, 155 runs) are not included in the repository due to size. Available upon request from the corresponding author. Training logs and model configurations for all runs are included in `results/train_logs/`.

## Pre-registration

All hypotheses, corpus conditions, model sizes, evaluation metrics, and statistical tests were pre-registered on OSF before any model was trained. The pre-registration is publicly available at: https://osf.io/h8un6/overview?view_only=0319b6be03a54e67b5486a0aea8207dc

Exploratory analyses (TOST, embedding analysis, perturbation control, one-shot test, per-seed evaluation, permutation-controlled probe) are clearly labelled as post-hoc throughout the paper and this repository.

## Citation

```bibtex
@article{
  title={Exemplar Memorisation Without Abstraction: Child-Scale Transformers 
         Fail to Induce Overhypotheses from Synthetic Category Data},

}
```

## License

Code: MIT License. Data and analysis outputs: CC-BY 4.0.
