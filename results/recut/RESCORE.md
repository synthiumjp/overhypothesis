# Exploratory rescoring for the ACL recut

Not registered. Fresh-text models, one exemplar (k = 1). Seed-level values: sizes averaged within seed, t-based 95% intervals over the seeds, two-sided tests.

Reproduction of results/amendment8/raw with original embeddings: 144000 rows, max absolute difference 2.18e-05 nats.

Scorings: emb = original (untrained novel-noun embeddings, as reported) or typical (mean training-noun embedding plus a distinct offset of typical size); measure = full (whole continuation, as reported) or word (changed word only).

## Index and noun-specific index by condition (mean of seed means)

| Condition | Index original/full | Index typical/full | Index original/word | Index typical/word | Noun-specific original/full | Noun-specific typical/full | Noun-specific original/word | Noun-specific typical/word |
|---|---|---|---|---|---|---|---|---|
| regular_random | -1.56 | -1.54 | -1.26 | -1.50 | -0.05 | -0.31 | 0.01 | -0.20 |
| regular_b3p6 | 1.28 | -1.41 | 1.45 | -1.25 | -0.03 | -0.15 | 0.03 | -0.04 |
| regular_grouped | 9.52 | 1.46 | 9.65 | 1.38 | 0.12 | 0.00 | 0.12 | 0.01 |
| nostable_grouped | -3.98 | -3.51 | -3.74 | -3.24 | -0.35 | -0.26 | -0.34 | -0.21 |
| objects_same | 5.87 | -1.85 | 6.06 | -1.87 | -0.07 | 0.14 | -0.06 | 0.06 |
| objects_diff | 7.60 | -1.25 | 7.59 | -1.33 | -0.01 | 0.19 | -0.01 | 0.11 |
| reassigned_random | 0.21 | 0.18 | 0.32 | 0.27 | 0.02 | -0.10 | 0.03 | -0.01 |
| reassigned_grouped | 9.80 | 3.87 | 9.94 | 3.98 | 0.37 | 0.70 | 0.37 | 0.74 |

## Seed-level contrasts

| Contrast | Scoring | Estimate [95% CI] | p | Seeds > 0 |
|---|---|---|---|---|
| Grouped minus random | original/full | 11.08 [9.57, 12.59] | 3.39e-05 | 5/5 |
| Grouped minus random | typical/full | 3.00 [1.79, 4.21] | 0.00235 | 5/5 |
| Grouped minus random | original/word | 10.91 [9.41, 12.41] | 3.58e-05 | 5/5 |
| Grouped minus random | typical/word | 2.88 [1.63, 4.14] | 0.0031 | 5/5 |
| B 3.5 minus random | original/full | 2.84 [1.41, 4.27] | 0.00527 | 5/5 |
| B 3.5 minus random | typical/full | 0.13 [-0.34, 0.59] | 0.485 | 3/5 |
| B 3.5 minus random | original/word | 2.71 [1.47, 3.95] | 0.00374 | 5/5 |
| B 3.5 minus random | typical/word | 0.25 [-0.10, 0.60] | 0.121 | 4/5 |
| Grouped Regular minus grouped No stable feature | original/full | 13.50 [11.48, 15.52] | 4.94e-05 | 5/5 |
| Grouped Regular minus grouped No stable feature | typical/full | 4.98 [3.42, 6.53] | 0.000886 | 5/5 |
| Grouped Regular minus grouped No stable feature | original/word | 13.39 [11.63, 15.16] | 2.98e-05 | 5/5 |
| Grouped Regular minus grouped No stable feature | typical/word | 4.62 [3.43, 5.81] | 0.000421 | 5/5 |
| Different-object minus same-object runs | original/full | 1.72 [0.54, 2.90] | 0.0154 | 5/5 |
| Different-object minus same-object runs | typical/full | 0.60 [0.26, 0.93] | 0.00762 | 5/5 |
| Different-object minus same-object runs | original/word | 1.54 [0.38, 2.69] | 0.0208 | 5/5 |
| Different-object minus same-object runs | typical/word | 0.54 [0.15, 0.93] | 0.0191 | 5/5 |
| Random-order index | original/full | -1.56 [-2.45, -0.67] | 0.00826 | 0/5 |
| Random-order index | typical/full | -1.54 [-2.40, -0.67] | 0.00797 | 0/5 |
| Random-order index | original/word | -1.26 [-1.88, -0.64] | 0.00479 | 0/5 |
| Random-order index | typical/word | -1.50 [-2.12, -0.88] | 0.00255 | 0/5 |
| No stable feature index | original/full | -3.98 [-5.27, -2.69] | 0.00102 | 0/5 |
| No stable feature index | typical/full | -3.51 [-5.49, -1.54] | 0.00784 | 0/5 |
| No stable feature index | original/word | -3.74 [-5.19, -2.30] | 0.00197 | 0/5 |
| No stable feature index | typical/word | -3.24 [-4.99, -1.49] | 0.00681 | 0/5 |
| Noun-specific index, Reassigned grouped | original/full | 0.37 [0.24, 0.49] | 0.00119 | 5/5 |
| Noun-specific index, Reassigned grouped | typical/full | 0.70 [0.31, 1.10] | 0.00784 | 5/5 |
| Noun-specific index, Reassigned grouped | original/word | 0.37 [0.24, 0.50] | 0.00145 | 5/5 |
| Noun-specific index, Reassigned grouped | typical/word | 0.74 [0.34, 1.14] | 0.00682 | 5/5 |
| Noun-specific: Reassigned minus Regular, grouped | original/full | 0.25 [0.10, 0.39] | 0.00978 | 5/5 |
| Noun-specific: Reassigned minus Regular, grouped | typical/full | 0.70 [0.23, 1.17] | 0.0149 | 5/5 |
| Noun-specific: Reassigned minus Regular, grouped | original/word | 0.24 [0.01, 0.47] | 0.042 | 5/5 |
| Noun-specific: Reassigned minus Regular, grouped | typical/word | 0.73 [0.20, 1.26] | 0.0184 | 5/5 |
