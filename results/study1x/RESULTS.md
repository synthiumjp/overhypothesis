# Study 1 extension (Amendment 5, Part A): results

The eight new corpus conditions (`data/corpora_x/`, measured clustering in `data/corpora_x/manifest.json`) were trained on the PC: Windows, DirectML training, 3 sizes × 5 seeds each, 120 models. Scoring was on the CPU (`docs/RUN_AMENDMENT5.md`, A2 to A4). The locked analysis code matches `docs/amendment5_checksums.txt` (`scripts/study1x_tests.py` MD5 f918ddf0cde017dce98c825bbb399e7e, `scripts/oneshot_eval.py` e9e144254e84777680ed8369cbf65141). The registered tests were run on 11 Oct 2026.

## Data

| File | Content |
|---|---|
| `raw/lm_<corpus>.csv.gz` | novel-noun items, 15 runs × 3,600 rows per corpus, no missing values, no duplicates |
| `raw/trained.csv.gz` | trained-noun items (S14), 12 conditions × 15 runs × 960 rows |
| `raw_md5.txt` | MD5s of the uncompressed raw files |
| `summary_per_run.csv`, `trained_per_run.csv` | `oneshot_eval.py summarise` on the .gz files (reproduces byte for byte) |
| `hbm/` | ideal observer on all 40 new corpora |
| `tests.json` | registered tests and registered secondary analyses |
| `exploratory_*` | `scripts/study1x_exploratory.py` (not registered) |

## Registered tests (OI_colour, k = 1, same noun; one-sided; Holm across four)

| Test | Means | Estimate (mixed model) | Holm p | Verdict |
|---|---|---|---|---|
| H11 grouped Regular > grouped NoStable | 1.36 vs -0.09 | 1.45 [1.10, 1.81], d = 2.14, 14 of 15 runs | < .001 | Supported |
| H12 grouped NoStable > grouped ColourStable | -0.09 vs -1.26 | 1.17 [0.70, 1.64], d = 1.54, 12 of 15 runs | < .001 | Supported |
| H13a OI rises with log B over the six Regular orders | see below | slope 0.29 (SE 0.04), Spearman 0.51 over runs | < .001 | Supported |
| H13b Regular at B 3.6 > Regular in random order | 0.24 vs 0.18 | 0.06 [-0.10, 0.22], d = 0.14, 9 of 15 runs | .23 | Not supported |

The Welch, paired and seed-paired t tests agree with the mixed model in every case.

H13b follow-up (registered in the Amendment 5 text, not in the locked script; `scripts/study1x_tost.py`, `tost_H13b.json`): TOST at ±0.2 nats on the same mixed model gives p = .047 (90% CI [-0.07, 0.20]), so the difference is equivalent to zero within ±0.2 nats, narrowly. On the paired differences over the 15 matched models, p = .055.

## OI_colour by condition (k = 1, same noun; `exploratory_by_condition.csv`)

| Condition | measured B | tiny | small | medium | all | runs > 0 |
|---|---|---|---|---|---|---|
| Regular, random | 0.9 | 0.15 | 0.30 | 0.10 | 0.18 | 9 |
| Regular, B 1.4 | 1.4 | 0.24 | 0.50 | 0.23 | 0.32 | 11 |
| Regular, B 3.6 | 3.6 | 0.06 | 0.40 | 0.26 | 0.24 | 12 |
| Regular, B 7 | 7.0 | 0.15 | 0.61 | 0.52 | 0.43 | 14 |
| Regular, B 12 | 12.1 | 0.27 | 0.93 | 0.76 | 0.66 | 13 |
| Regular, grouped | 23.1 | 0.51 | 1.79 | 1.79 | 1.36 | 14 |
| NoStable, random | 0.9 | -0.22 | -0.29 | -0.23 | -0.25 | 6 |
| NoStable, grouped | 23.1 | -0.16 | 0.00 | -0.11 | -0.09 | 5 |
| ColourStable, random | 0.9 | 0.04 | -0.25 | -0.55 | -0.25 | 5 |
| ColourStable, grouped | 23.1 | -0.23 | -1.97 | -1.58 | -1.26 | 2 |

In grouped ColourStable, the colour component carries the effect: colour repetition +1.03, shape -0.23. In grouped Regular it is the shape component: shape +1.17, colour -0.19.

## What this shows

1. **Grouping alone does not make the index.** With no stable dimension, grouped order gives nothing (H11). The Study 1c grouping effect needs the regularity in the corpus. It is not a side effect of nouns recurring close together.
2. **The index follows whichever dimension is stable.** Make colour stable and grouped training gives a strong colour bias (H12). The models learn the corpus statistic. They have no built-in preference for shape.
3. **More clustering, larger index.** The rise is graded and roughly monotone in log B (H13a). Exploratory: refitting the registered model without the grouped corpus still gives a positive slope, 0.15 (SE 0.03, one-sided p < .001, `exploratory_doseresponse.json`). So the trend does not rest on one end point.
4. **Clustering at the child-directed level is not enough here.** At B 3.6 (the CHILDES level from Study 1e), the index is no larger than in random order (H13b). Clear gains appear from about B 7.
5. **The ideal observer ignores order.** It gives the same index for every order of a corpus: Regular 6.25, NoStable -0.11, ColourStable -6.17 (`exploratory_hbm.csv`). The statistics are in every Regular corpus. The transformers use them only when same-kind examples come close together.
6. **Model size.** As in Study 1c, the tiny models gain least from grouping. The grouping × tiny interaction is -1.33, p < .001 (`tests.json`, secondary).

## Registered secondary analyses (`tests.json`, no correction)

- OI_texture: H11 holds for texture too (1.40 [1.01, 1.79]). H12 and H13b are null for texture, as expected, since texture is not stable in either ColourStable or NoStable.
- Kind-specific index with novel nouns (same minus different noun): within ±0.005 in every condition. Novel nouns are never seen in training, so the models cannot tell them apart.
- S14, kind-specific index with trained nouns: no condition is reliably above zero. Regular random 0.16 (p = .08) and ColourStable grouped 0.15 (p = .04, uncorrected, one of 12 tests); the rest are near zero.

## Notes on processing

- The A2/A3 runs first failed because the output folders did not exist. No scores were written before the folders were made, so there were no partial or duplicate rows. `RUN_AMENDMENT5.md` now makes the folders first.
- The trained-noun summary (`trained_per_run.csv`) and `summary_per_run.csv` were made here from the PC's raw files. The PC's own `summary_per_run.csv` had identical values (Windows line endings only).

## Reproduce

```
python scripts/oneshot_eval.py summarise --inputs "results/study1x/raw/lm_*.csv.gz" --out results/study1x/summary
python scripts/oneshot_eval.py summarise --inputs results/study1x/raw/trained.csv.gz --out results/study1x/trained
python scripts/study1x_tests.py --runs results/study1c/summary_per_run.csv results/study1x/summary_per_run.csv \
  --trained results/study1x/trained_per_run.csv --out results/study1x/tests
python scripts/study1x_tost.py --runs results/study1c/summary_per_run.csv results/study1x/summary_per_run.csv \
  --out results/study1x/tost_H13b.json
python scripts/study1x_exploratory.py --runs results/study1c/summary_per_run.csv results/study1x/summary_per_run.csv \
  --hbm "results/study1x/hbm/hbm_*.csv" --out results/study1x/exploratory
```
