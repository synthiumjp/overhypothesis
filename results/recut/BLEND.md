# Noun-familiarity curve (exploratory)

Novel-noun input embeddings blended between their original untrained values (0) and the typical embedding (1: mean training-noun embedding plus a distinct offset of typical size). Fresh-text models, k = 1, whole-continuation Delta. Means of seed means; slopes are per-seed least-squares slopes against the blend, with t-based 95% intervals over five seeds.

## Index

| Condition | 0 | 0.25 | 0.5 | 0.75 | 1 | Slope [95% CI] | p |
|---|---|---|---|---|---|---|---|
| regular_random | -1.56 | -1.83 | -1.93 | -1.79 | -1.54 | 0.03 [-0.22, 0.28] | 0.72 |
| regular_grouped | 9.52 | 6.70 | 4.32 | 2.10 | 1.46 | -8.28 [-9.87, -6.70] | 0.000131 |
| objects_same | 5.87 | 3.71 | 1.08 | -1.41 | -1.85 | -8.23 [-10.05, -6.40] | 0.000233 |
| objects_diff | 7.60 | 4.53 | 1.87 | -0.82 | -1.25 | -9.22 [-9.97, -8.46] | 4.53e-06 |
| reassigned_grouped | 9.80 | 8.03 | 5.34 | 3.87 | 3.87 | -6.41 [-8.74, -4.08] | 0.00158 |

## Delta shape

| Condition | 0 | 0.25 | 0.5 | 0.75 | 1 | Slope [95% CI] | p |
|---|---|---|---|---|---|---|---|
| regular_random | -1.55 | -1.83 | -1.92 | -1.78 | -1.52 | 0.05 [-0.26, 0.35] | 0.686 |
| regular_grouped | 6.88 | 4.27 | 2.08 | -0.08 | -0.72 | -7.82 [-9.59, -6.04] | 0.000258 |
| objects_same | 6.87 | 4.59 | 1.90 | -0.65 | -1.17 | -8.52 [-10.22, -6.83] | 0.000154 |
| objects_diff | 8.05 | 4.85 | 2.08 | -0.69 | -1.14 | -9.57 [-10.14, -9.00] | 1.29e-06 |
| reassigned_grouped | 6.81 | 4.83 | 2.24 | 0.97 | 1.17 | -6.06 [-7.28, -4.84] | 0.000159 |

## Delta colour

| Condition | 0 | 0.25 | 0.5 | 0.75 | 1 | Slope [95% CI] | p |
|---|---|---|---|---|---|---|---|
| regular_random | 0.01 | -0.00 | 0.01 | 0.01 | 0.02 | 0.01 [-0.23, 0.25] | 0.89 |
| regular_grouped | -2.64 | -2.43 | -2.24 | -2.18 | -2.18 | 0.47 [0.20, 0.74] | 0.00873 |
| objects_same | 0.99 | 0.88 | 0.82 | 0.76 | 0.68 | -0.30 [-0.63, 0.03] | 0.0661 |
| objects_diff | 0.46 | 0.32 | 0.21 | 0.14 | 0.11 | -0.35 [-0.64, -0.07] | 0.0254 |
| reassigned_grouped | -2.99 | -3.19 | -3.10 | -2.91 | -2.70 | 0.35 [-0.90, 1.61] | 0.482 |

## Noun-specific index

| Condition | 0 | 0.25 | 0.5 | 0.75 | 1 | Slope [95% CI] | p |
|---|---|---|---|---|---|---|---|
| regular_random | -0.05 | -0.12 | -0.22 | -0.30 | -0.31 | -0.28 [-0.54, -0.01] | 0.0444 |
| regular_grouped | 0.12 | 0.05 | -0.03 | -0.02 | 0.00 | -0.12 [-0.32, 0.07] | 0.151 |
| objects_same | -0.07 | 0.10 | 0.25 | 0.24 | 0.14 | 0.22 [-0.19, 0.64] | 0.208 |
| objects_diff | -0.01 | 0.13 | 0.21 | 0.27 | 0.19 | 0.21 [-0.27, 0.70] | 0.283 |
| reassigned_grouped | 0.37 | 0.68 | 0.53 | 0.50 | 0.70 | 0.20 [-0.26, 0.66] | 0.299 |

## Grouped minus random (index) by blend

| Blend | Estimate [95% CI] | p |
|---|---|---|
| 0 | 11.08 [9.57, 12.59] | 3.39e-05 |
| 0.25 | 8.53 [6.90, 10.15] | 0.000129 |
| 0.5 | 6.24 [4.81, 7.68] | 0.000268 |
| 0.75 | 3.89 [2.47, 5.31] | 0.0016 |
| 1 | 3.00 [1.79, 4.21] | 0.00235 |

## Embedding norms (mean over models)

Trained nouns 0.546; novel nouns, original 0.359; typical 0.546.
