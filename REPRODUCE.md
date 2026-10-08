# Reproducing Studies 1b, 1c and 1d

Branch `study1b-1c-evaluation`. Pre-registration: osf.io/qj9hb (original), Amendment 1 osf.io/x6zhn (Studies 1b and 1c), Amendment 2 (texture co-primary for H5b), Amendment 3 osf.io/qcfjp (Study 1d).

Every number reported for Studies 1b to 1d can be regenerated from files in this repository, except where a row below says the input is archived elsewhere.

## Fixed inputs (verify before anything else)

| File | MD5 | Made by |
|---|---|---|
| `data/wug_tests/oneshot_items.csv` | `6d1b6064392d1395cb217af494d240c7` | `scripts/oneshot_eval.py make-items` (seed 2026) |
| `data/wug_tests/english_items.csv` | `687ff90580ea03a6d76aba55499e7835` | `scripts/english_oneshot.py make-items` (seed 2027) |
| `data/corpora/*.txt` | `data/corpora/checksums_md5.txt` | `scripts/corpus_generator.py` |
| `data/corpora_bursty/*.txt` | `data/corpora_bursty/bursty_checksums.json` | `scripts/make_bursty_corpus.py` |
| `results/study1d/models.csv` | `2d8debf6a3d3cd5ce52aabf2b24107e3` | registered model list |

The bursty corpora regenerate byte-identically: `python scripts/make_bursty_corpus.py` and compare with the checksum file.

## Study 1b and 1c (synthetic corpora)

| Step | Command | Output |
|---|---|---|
| Ideal observer, replication-corrected | `python scripts/oneshot_eval.py eval-hbm --items data/wug_tests/oneshot_items.csv --corpus data/corpora/<cond>_seed<s>.txt --condition <cond> --seed <s> --out results/study1b/hbm_corrected/hbm_<cond>_<s>.csv` | `results/study1b/hbm_corrected/` |
| Score trained models | `python scripts/oneshot_eval.py make-manifest ...` then `eval-lm` (needs the checkpoints, see below) | `results/study1b/raw/lm_results.csv.gz`, `random_results.csv.gz` (untrained floor), `results/study1c/raw/lm_results.csv.gz`; checksums in `results/raw_checksums_md5.txt` |
| Per-run summary | `python scripts/oneshot_eval.py summarise --inputs study1c_lm_results.csv study1b_lm_results.csv --out results/study1c/summary` (unzip first; reproduces the committed file exactly) | `results/study1c/summary_per_run.csv` (MD5 `4337444288f38c22b42ac02983d43fc4`) |
| Registered tests H4b, H5b | `python scripts/study1c_tests.py --runs results/study1c/summary_per_run.csv --hbm "results/study1c/hbm/hbm_*.csv"` | `results/study1c/tests_output.txt` |

## Study 1d (pretrained models, natural English)

| Step | Command | Output |
|---|---|---|
| Score models | `python scripts/english_oneshot.py eval --items data/wug_tests/english_items.csv --manifest results/study1d/models.csv --out scores.csv` (add `--backend mlx` for the 70B models) | item-level scores |
| Merged scores used in the paper | assembled as described under Deviations | `results/study1d/scores_merged.csv` (MD5 `38474c067938ffb01079635346e56a00`, 81 checkpoints) |
| Registered tests H6 to H8 | `python scripts/study1d_tests.py --scores results/study1d/scores_merged.csv --out results/study1d/tests_final` | `results/study1d/tests_final.json`, `tests_final_secondary.csv` |
| Summary | | `results/study1d/RESULTS.md` |

## Exploratory and robustness analyses (all studies)

`python scripts/exploratory_analyses.py` writes:

- `results/study1b/hbm_corrected_summary.csv`: corrected ideal observer by condition (promised in Amendment 1)
- `results/study1c/kind_specific_by_dim.csv`: same-kind minus other-kind repetition per dimension, Studies 1b and 1c
- `results/study1d/kind_specific_final.csv`, `kind_specific_trajectory.csv`: the same for Study 1d
- `results/study1d/robustness_*.csv`: leave one noun out, raw deltas, sentence order (promised in Amendment 3), word effects

## Deviations and their checks (Study 1d)

| Deviation | Check | Files |
|---|---|---|
| pythia-2.8b steps 0 to 16,000 loaded a stale copy of the final weights from the Hub; rescored from the per-step shards | identical scores across checkpoints before the fix; file hashes on the Hub | `results/study1d/raw/` (before), `results/study1d/fix28b/` (after) |
| pythia-2.8b step 64,000 identical to step 143,000 on the Hub; dropped | same SHA-256 on the Hub | |
| 70B models and checks run under Python 3.12 / PyTorch 2.13 (others under 3.14 / 2.14) | rescoring two checkpoints: mean abs. difference 0.0003 and 0.005 nats | `results/study1d/qwen/val_torch.csv` |
| Qwen2.5-72B and Llama-3.1-70B scored with MLX | MLX vs PyTorch: 0.003 nats (pythia-410m), 0.010 nats (Qwen2.5-0.5B) | `results/study1d/qwen/val_mlx.csv`, `val_torch.csv` |

## Not in this repository

- Trained checkpoints for Studies 1, 1b and 1c (about 8 GB). To be archived with a DOI before submission.
- Pretrained models for Study 1d are public on Hugging Face at the revisions in `results/study1d/models.csv`.
