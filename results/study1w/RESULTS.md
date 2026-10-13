> **Superseded scores.** The language-model scores and tests in this folder were computed before 12 Oct 2026 with a loader that let the output matrix overwrite the input embedding (the checkpoints hold untied matrices). They are kept as a record. The corrected scores, used in the paper, are in `results/rescore/` (and `results/study1o/`, `results/cl_revision/`). Outputs that do not use a trained model (corpus statistics, the hierarchical Bayesian reference model) are unaffected.

# Study 1 window test (Amendment 6, osf.io/34gca): results

45 new models (Regular random order, Regular at B = 3.6, NoStable random order; 3 sizes × 5 seeds), trained on the PC exactly as in Study 1 except `--seq_len 512 --keep_tokens_per_step` (batch 8 for tiny and small, 4 for medium; same tokens per step). Training 11 Oct 2026; scoring 12 Oct (`docs/RUN_AMENDMENT6.md`, W1 to W3). Locked analysis code matches `docs/amendment6_checksums.txt` (`scripts/study1w_tests.py` MD5 7a7d0e354877f32ffccc01814b5f12af). The prediction (`prediction.json`) and the window measure (`cooccurrence.csv`) are byte-identical to the locked files.

## Data

| File | Content |
|---|---|
| `raw/lm_<corpus>.csv.gz` | novel-noun items, 15 models × 3,600 rows per corpus; no missing values or duplicates |
| `raw/trained.csv.gz` | trained-noun items, 45 models × 960 rows |
| `raw_md5.txt` | MD5s of the uncompressed raw files |
| `summary_per_run.csv`, `trained_per_run.csv` | `oneshot_eval.py summarise` on the .gz files (identical to the PC's summary) |
| `tests.json` | registered tests H17 to H19 and secondary S24 to S28 |
| `tost.json` | equivalence test for H17 (registered in the Amendment 6 text, run with `scripts/study1w_tost.py`); the same for H18, exploratory |

## Within-window co-occurrence (`cooccurrence.csv`, measured before training)

Share of sentences in a training window that follow an earlier same-kind sentence in the same window: Regular random 0.150 at W = 128 and 0.474 at W = 512; B 3.6 0.211 and 0.498; B 12 at W = 128 0.403; grouped 0.670.

## Registered tests (OI_colour, k = 1, same noun; one-sided; Holm across three)

| Test | Means | Estimate (mixed model) | Holm p | Verdict |
|---|---|---|---|---|
| H17 Regular random, W 512 > W 128 | 0.21 vs 0.18 | 0.02 [-0.18, 0.23], d = 0.06, 8 of 15 | .67 | Not supported |
| H18 Regular B 3.6, W 512 > W 128 | 0.29 vs 0.24 | 0.04 [-0.16, 0.24], d = 0.11, 9 of 15 | .67 | Not supported |
| H19 Regular random W 512 > NoStable random W 512 | 0.21 vs -0.08 | 0.29 [0.03, 0.55], d = 0.69, 12 of 15 | .048 | Supported |

H17 equivalence (TOST, ±0.2 nats): registered model p = .044, 90% CI [-0.14, 0.19], equivalent narrowly; paired differences p = .071. H18 (exploratory): p = .062 and .067.

## S24: the co-occurrence prediction

| Condition | Predicted | Observed [95% CI] | Reading |
|---|---|---|---|
| Regular random, W 512 | 0.88 | 0.21 [-0.03, 0.45] | below |
| Regular B 3.6, W 512 | 0.94 | 0.29 [0.10, 0.47] | below |

As registered, "below" means that same-kind sentences must be close together, not only in the same training window. At W = 512, random order puts an earlier same-kind sentence in the window for 47% of sentences, more than the B 12 corpus at W = 128 (40%, index 0.66), yet the index stayed at the random-order level.

## Secondary (`tests.json`, no correction)

- S25 texture: H17 -0.07 (p = .71), H18 0.06 (p = .32), H19 0.31 (p = .009).
- S26 more exemplars: indices stay about the same with two or four exemplars (Regular W 512: 0.21, 0.19, 0.12).
- S27 size × window: no interaction (all p > .8).
- S28 trained nouns: kind-specific index not above zero in any new condition (means -0.21, -0.15, -0.10).
- By size (OI_colour): Regular W 512 tiny 0.20, small 0.32, medium 0.09; B 3.6 W 512 0.27, 0.38, 0.21; NoStable W 512 0.15, -0.16, -0.23.

## Reading

A four times longer training window, holding four times as many same-kind sentences, did not change what random order or child-level clustering taught. What helps in grouped and highly clustered order is that same-kind sentences come next to each other, not that they share the training window. H19 shows that the small preference random-order models do learn depends on the stable dimension.

## Notes

- The PC's summary file was named `summaryn_per_run.csv` (typo at the prompt); the summaries here were regenerated from the raw files and match it exactly.
- Parameter counts: the new models have the Study 1 counts plus the larger position table (512 rows).

## Reproduce

```
python scripts/oneshot_eval.py summarise --inputs "results/study1w/raw/lm_*.csv.gz" --out results/study1w/summary
python scripts/oneshot_eval.py summarise --inputs results/study1w/raw/trained.csv.gz --out results/study1w/trained
python scripts/study1w_tests.py --runs results/study1c/summary_per_run.csv results/study1x/summary_per_run.csv \
  results/study1w/summary_per_run.csv --trained results/study1w/trained_per_run.csv --out results/study1w/tests
python scripts/study1w_tost.py --runs results/study1c/summary_per_run.csv results/study1x/summary_per_run.csv \
  results/study1w/summary_per_run.csv --out results/study1w/tost.json
```
