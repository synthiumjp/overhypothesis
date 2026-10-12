# Study 1 with corrected model loading (12 Oct 2026)

## The error

Study 1 models were trained on DirectML. `train_model.py` ties the output matrix to the input embedding, but the tie broke when the model moved to the device, so every model was trained, and saved, with two separate matrices. `oneshot_eval.py` rebuilt each model with the tie and then loaded the weights, so the output matrix overwrote the input embedding. Every Study 1 score before commit 483ad6b therefore came from a model whose input embedding had been replaced. The fix unties before loading; `--legacy_tied_load` reproduces the old scores exactly (checked on three models: e.g. ColourStable grouped medium seed 42, -0.736 both ways).

All 315 final models, 225 trained-noun runs and 461 tracking checkpoints were rescored on 12 Oct with the same manifests and items (`docs/RUN_RESCORE.md`). Scores: `raw/*.csv.gz` (MD5s of the uncompressed files in `raw_md5.txt`). The Bayesian reference, untrained baseline, Study 2 and Study 3 are unaffected. The earlier outputs stay in `results/study1b`, `study1c`, `study1x`, `study1w` as a record and are superseded by this folder.

## Registered tests: before and after (OI_colour, k = 1, same noun)

| Test | Before: estimate, Holm p | Corrected: estimate, Holm p | Corrected verdict |
|---|---|---|---|
| H1b Regular random > 0 | 0.18, .12 | 0.47 (SD 0.90), .061 | Not supported |
| H2b Regular > Scrambled (random) | 0.18 vs 0.09, .59 | 0.47 vs -0.25, .032 | Supported |
| H3b same noun > different noun | 0.0007, .59 | 0.005, .46 (TOST ±0.2 p = .001) | Not supported; equivalent to zero |
| H4b grouped Regular > random Regular | 1.36 vs 0.18, < .001 | 6.49 vs 0.47, < .001 (15/15) | Supported |
| H5b colour: grouped Regular > grouped Scrambled | 1.36 vs 0.80, .031 | 6.49 vs 2.12, < .001 (15/15) | Supported |
| H5b texture | 1.10 vs -0.04, < .001 | 5.36 vs 0.46, < .001 | Supported |
| H11 grouped Regular > grouped NoStable | diff 1.45, < .001 | 6.49 vs -0.44, diff 6.93, < .001 | Supported |
| H12 grouped NoStable > grouped ColourStable | diff 1.17, < .001 | -0.44 vs -3.20, diff 2.76, < .001 | Supported |
| H13a index rises with log B | slope 0.29, < .001 | slope 1.63 (SE 0.10), < .001 | Supported |
| H13b B 3.6 > random | 0.24 vs 0.18, .23 | 1.43 vs 0.47, diff 0.96 [0.59, 1.32], d = 1.04, 15/15, < .001 | **Supported** (was not) |
| H17 window 512 > 128, random | 0.21 vs 0.18, .67 | 0.50 vs 0.47, .87 | Not supported (TOST p = .15, not equivalent) |
| H18 window 512 > 128, B 3.6 | 0.29 vs 0.24, .67 | 0.80 vs 1.43, diff -0.63, 1.00 | Not supported; the longer window gave less |
| H19 Regular > NoStable, window 512 | 0.21 vs -0.08, .048 | 0.50 vs 0.10, .045 | Supported |

## Index by condition (corrected; `study1x_exploratory_by_condition.csv`)

| Condition | B | tiny | small | medium | all | models > 0 |
|---|---|---|---|---|---|---|
| Regular random | 0.9 | 0.33 | 0.71 | 0.38 | 0.47 | 10 |
| Regular B 1.4 | 1.4 | 0.73 | 1.05 | 0.70 | 0.83 | 12 |
| Regular B 3.6 | 3.6 | 0.97 | 2.05 | 1.27 | 1.43 | 15 |
| Regular B 7 | 7.0 | 1.85 | 2.89 | 1.45 | 2.06 | 15 |
| Regular B 12 | 12.1 | 3.18 | 4.53 | 2.88 | 3.53 | 15 |
| Regular grouped | 23.1 | 6.41 | 6.83 | 6.23 | 6.49 | 15 |
| NoStable random | 0.9 | -0.10 | -0.33 | -0.12 | -0.18 | 5 |
| NoStable grouped | 23.1 | -0.41 | -0.28 | -0.63 | -0.44 | 4 |
| ColourStable random | 0.9 | -0.04 | -0.70 | -0.51 | -0.42 | 4 |
| ColourStable grouped | 23.1 | -2.11 | -3.72 | -3.77 | -3.20 | 0 |
| Scrambled random | | -0.11 | -0.45 | -0.21 | -0.25 | 4 |
| Scrambled grouped | | 2.24 | 2.35 | 1.76 | 2.12 | 14 |
| Regular random, window 512 | 0.9 | 0.30 | 0.51 | 0.70 | 0.50 | |
| Regular B 3.6, window 512 | 3.6 | 0.31 | 1.02 | 1.06 | 0.80 | |
| NoStable random, window 512 | 0.9 | 0.21 | 0.04 | 0.05 | 0.10 | |

## What changes

1. Child-level clustering now helps clearly: B 3.6 triples the random-order index (H13b supported, every model positive). The dose-response is steep and monotone (exploratory: slope 1.06 per log B without the grouped corpus).
2. The small-model deficit disappears: grouping helps all three sizes about equally (size × grouping interaction n.s.; previously the tiny models gained least).
3. Random-order Regular models exceed Scrambled (H2b), though the index alone is not reliably above zero after correction (H1b).
4. The Feature-swap result reported as exploratory (texture preference) does not hold (0.51 vs 0.60).
5. Grouped models reach most of their final index by step 2,500 (tiny models by step 500; `tracking_trajectory.csv`).
6. Trained nouns (S14): grouped Regular models now show a kind-specific index above zero (0.30 nats, p = .008, uncorrected, 1 of 15 conditions). Novel nouns: within ±0.12 of zero in every condition.
7. Window test: a longer window did not help random order and gave a smaller index at B 3.6 (0.80 vs 1.43). The registered S24 prediction (0.88, 0.94) was fitted to the uncorrected scores; refitted to the corrected scores (exploratory), the same model predicts 4.3 and 4.6, far above what was observed. Adjacency, not co-occurrence within the window, is what helps.

## Reproduce

```
R=results/rescore
python scripts/oneshot_eval.py summarise --inputs $R/raw/study1c_lm.csv.gz $R/raw/study1b_lm.csv.gz --out $R/study1bc
python scripts/oneshot_eval.py summarise --inputs "$R/raw/study1x_lm_*.csv.gz" --out $R/study1x
python scripts/oneshot_eval.py summarise --inputs "$R/raw/study1w_lm_*.csv.gz" --out $R/study1w
python scripts/oneshot_eval.py summarise --inputs $R/raw/trained.csv.gz --out $R/trained
python scripts/study1b_tests.py --runs $R/study1bc_per_run.csv --out $R/study1b_tests_output.txt
python scripts/study1c_tests.py --runs $R/study1bc_per_run.csv --hbm "results/study1c/hbm/hbm_*.csv" > $R/study1c_tests_output.txt
python scripts/study1x_tests.py --runs $R/study1bc_per_run.csv $R/study1x_per_run.csv --trained $R/trained_per_run.csv --out $R/study1x_tests
python scripts/study1x_tost.py --runs $R/study1bc_per_run.csv $R/study1x_per_run.csv --out $R/study1x_tost_H13b.json
python scripts/study1x_exploratory.py --runs $R/study1bc_per_run.csv $R/study1x_per_run.csv --hbm "results/study1x/hbm/hbm_*.csv" --out $R/study1x_exploratory
python scripts/study1w_tests.py --runs $R/study1bc_per_run.csv $R/study1x_per_run.csv $R/study1w_per_run.csv --trained $R/trained_per_run.csv --out $R/study1w_tests
python scripts/study1w_tost.py --runs $R/study1bc_per_run.csv $R/study1x_per_run.csv $R/study1w_per_run.csv --out $R/study1w_tost.json
python scripts/window_cooccurrence.py predict --cooc results/study1w/cooccurrence.csv --runs $R/study1bc_per_run.csv $R/study1x_per_run.csv --out $R/prediction_corrected_exploratory.json
python scripts/rescore_study1.py --dir $R
```
