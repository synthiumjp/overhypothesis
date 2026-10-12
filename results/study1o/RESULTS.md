# Amendment 7: same-object versus different-object clustering (results)

75 models (5 orders × 3 sizes × 5 seeds) trained on the PC on 12 to 13 Oct 2026 on `data/corpora_o` (`docs/RUN_AMENDMENT7.md`) and scored on 13 Oct with the corrected loader. Locked analysis code matches `docs/amendment7_checksums.txt` (`scripts/study1o_tests.py` MD5 31580baf2d7fdc3e79b34c1d2c9c0efa). Complete: 5 × 15 × 3,600 novel-noun rows and 75 × 960 trained-noun rows, with no missing values or duplicates.

## Loss check of the corrected loader (`results/rescore/loss_check.csv`)

Eight randomly chosen Study 1 models; mean next-token loss on each model's own training corpus (200 windows):

| Model | End of training (last 5 logs) | Corrected loading | Old loading |
|---|---|---|---|
| feature_swap tiny 42 | 0.228 | 0.158 | 1.710 |
| bare_nolabel small 123 | 0.135 | 0.102 | 1.519 |
| regular_bursty small 789 | 0.078 | 0.073 | 1.125 |
| regular_bursty medium 123 | 0.091 | 0.081 | 0.948 |
| scrambled medium 123 | 0.101 | 0.082 | 1.380 |
| weak_label medium 456 | 0.112 | 0.090 | 0.582 |
| graded_b3p6 tiny 789 | 0.204 | 0.153 | 1.968 |
| graded_b3p6 small 1001 | 0.079 | 0.073 | 1.635 |

The corrected loader reproduces the trained models; it is slightly lower than the training log because dropout is off at evaluation. The old loading was 5 to 20 times worse.

## Registered tests (OI_colour, k = 1, same noun; one-sided; Holm across three)

| Test | Means | Estimate [95% CI] | d | Models in direction | Holm p | Verdict |
|---|---|---|---|---|---|---|
| H20 diff_grouped > same_grouped | 5.65 vs 2.99 | 2.66 [1.91, 3.42] | 2.39 | 15/15 | < .001 | Supported |
| H21 diff_b3p6 > same_b3p6 | 1.18 vs 0.45 | 0.73 [0.36, 1.11] | 1.14 | 13/15 | < .001 | Supported |
| H22 diff_b3p6 > random | 1.18 vs 0.22 | 0.96 [0.60, 1.31] | 1.70 | 13/15 | < .001 | Supported |

Welch, paired and seed-paired tests agree (seed-paired p = .001, .013, .005).

## Secondary

- **S29.** same_b3p6 vs random: 0.22 [-0.10, 0.55], two-sided p = .18. Not different, and not equivalent within ±0.2 (TOST p = .56).
- **S30.** diff_grouped vs random: 5.43 [4.82, 6.04]. same_grouped vs random: 2.76 [2.20, 3.32]. Both p < .001.
- **S31.** Components (nats):

| Order | shape | colour | texture | OI_colour |
|---|---|---|---|---|
| random | 0.58 | 0.36 | 0.18 | 0.22 |
| same_b3p6 | 1.34 | 0.90 | -0.06 | 0.45 |
| diff_b3p6 | 1.25 | 0.07 | 0.12 | 1.18 |
| same_grouped | 4.63 | 1.65 | 0.58 | 2.99 |
| diff_grouped | 5.37 | -0.28 | 0.31 | 5.65 |

  Same-object runs taught the models to repeat colour as well as shape. Different-object runs taught them to repeat shape and change colour. The positive index after same-object grouping comes from shape being repeated more than colour. Shape is named in all four frames and colour in two, so the shape repeats are partly a copying habit.
- **S32.** Texture: H20 1.01 (p = .002); H21 -0.27 (p = .90); H22 0.73 (p < .001).
- **S33.** Trained nouns, kind-specific index: same_grouped -1.06. After same-object training, the models expected more of the old object's features after the same noun, colour included, as if the same noun named the same object. This parallels Study 2's finding with "a" against "another". Other conditions: -0.17 to 0.14, none above zero.
- **S34.** The new random order (0.22) does not differ from Study 1 random order (0.47; p = .34).

## Reading

Nearby repetition teaches the regularity mainly when it brings different objects of a kind. Re-describing the same object teaches the model to expect every feature to repeat. At the level of clustering in speech to children, same-object runs gave no reliable gain over random order, while different-object runs gave a large one. Natural corpora should therefore be placed by their clustering of different objects. In speech to children this is B_eff ≈ 1.7 to 2.0 (`results/rescore/effective_b.json`), where the Study 1 curve gives about twice the random-order index.

## Reproduce

```
python scripts/oneshot_eval.py summarise --inputs "results/study1o/raw/lm_*.csv.gz" --out results/study1o/summary
python scripts/oneshot_eval.py summarise --inputs results/study1o/raw/trained.csv.gz --out results/study1o/trained
python scripts/study1o_tests.py --runs results/rescore/study1bc_per_run.csv results/study1o/summary_per_run.csv \
  --trained results/study1o/trained_per_run.csv --out results/study1o/tests
```
