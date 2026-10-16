# Amendment 8 results (16 October 2026)

120 models trained on fresh-order corpora (`scripts/make_fresh_corpora.py`; all 80 corpus MD5s on the training PC matched the registered manifest). Scored with the corrected loader. Tests: `scripts/amendment8_tests.py` → `tests.json`. Seed-level estimates (five seed means, sizes averaged; 95% t intervals, 4 df).

## Confirmatory (Holm within Part A and within Part B)

| | Hypothesis | Means | Seed-level difference [95% CI] | p | Holm p | Seeds | Verdict |
|---|---|---|---|---|---|---|---|
| H23 | Index: grouped > random | 9.52 vs −1.56 | 11.08 [9.57, 12.59] | < .001 | < .001 | 5/5 | Supported |
| H24 | Index: B 3.6 > random | 1.28 vs −1.56 | 2.84 [1.41, 4.27] | .003 | .005 | 5/5 | Supported |
| H25 | Index: grouped Regular > grouped No stable feature | 9.52 vs −3.98 | 13.50 [11.48, 15.52] | < .001 | < .001 | 5/5 | Supported |
| H26 | Index: different-object > same-object | 7.60 vs 5.87 | 1.72 [0.54, 2.90] | .008 | .008 | 5/5 | Supported |
| H27 | Noun-specific index, novel nouns, reassigned_grouped > 0 | 0.37 | 0.37 [0.24, 0.49] | < .001 | .001 | 5/5 | Supported |
| H28 | The same, reassigned_grouped > regular_grouped | 0.37 vs 0.12 | 0.25 [0.10, 0.39] | .005 | .005 | 5/5 | Supported |
| H29 | Noun-specific index, trained nouns, reassigned_grouped > 0 | 11.24 | 11.24 [10.44, 12.03] | < .001 | < .001 | 5/5 | Supported |

The mixed models agree in every case.

## Memorisation (S35)

| Corpus | Fixed: own file | Fixed: held-out arrangement | Fixed gap | Fresh: own | Fresh: held-out | Fresh gap |
|---|---|---|---|---|---|---|
| Regular, random | 0.11 | 1.25 | 1.14 | 0.55 | 0.55 | 0.00 |
| Regular, B 3.6 | 0.10 | 1.26 | 1.16 | 0.54 | 0.54 | 0.00 |
| Regular, grouped | 0.09 | 0.85 | 0.77 | 0.35 | 0.35 | 0.00 |
| No stable feature, grouped | 0.09 | 1.00 | 0.91 | 0.39 | 0.39 | 0.00 |
| Objects, same-object grouped | 0.08 | 0.66 | 0.57 | 0.28 | 0.28 | 0.00 |
| Objects, different-object grouped | 0.09 | 0.76 | 0.67 | 0.33 | 0.33 | 0.00 |

Nats per token. The fixed-corpus models had memorised their files: on a new arrangement of the same sentences their loss rose by 0.57 to 1.16 nats, to well above the fresh models' loss. Fixed minus fresh gap, seed level: 0.57 to 1.16, every interval excluding 0.

## Fresh against fixed (S36)

| Corpus | Fresh | Fixed | Difference [95% CI] | p (two-sided) |
|---|---|---|---|---|
| Regular, random | −1.56 | 0.47 | −2.03 [−2.42, −1.65] | < .001 |
| Regular, B 3.6 | 1.28 | 1.43 | −0.15 [−1.67, 1.37] | .80 |
| Regular, grouped | 9.52 | 6.49 | 3.03 [2.02, 4.04] | .001 |
| No stable feature, grouped | −3.98 | −0.44 | −3.54 [−5.15, −1.92] | .004 |
| Objects, same-object grouped | 5.87 | 2.99 | 2.89 [1.38, 4.40] | .006 |
| Objects, different-object grouped | 7.60 | 5.65 | 1.95 [1.50, 2.39] | < .001 |

Effects, fresh vs fixed: grouped − random 11.08 vs 6.02 (difference 5.06 [3.88, 6.25]); B 3.6 − random 2.84 vs 0.96 (1.88 [0.14, 3.63]); Regular − No stable feature 13.50 vs 6.93 (6.57 [5.05, 8.09]); different − same object 1.72 vs 2.66 (−0.94 [−2.54, 0.66], p = .18).

Fresh random-order models expected shape to change (Δshape −1.55; index below zero in all 15 models). The fixed random-order models showed the same early in training (exploratory, from `results/rescore/tracking_trajectory.csv` and the Study 1 training logs). While their training loss was near the fresh models' level (about 0.55 to 0.6 nats per token; steps 500 to 1,500 for the small and 500 to 2,500 for the medium models), their index was −0.4 to −1.5. It rose above zero only as the loss fell below that level, towards the memorised 0.08 to 0.10. In random order a sentence about a different kind repeats the previous shape 7% of the time, below the 10% for colour (S40).

## Part B

- Trained nouns (H29): after shape reassignment, the noun-specific index was 11.24 nats, against 1.04 [0.76, 1.31] in fresh regular_grouped. The noun-aware reference from the corpus is 11.41. The models tied the copying of shape to the noun almost exactly as much as the corpus warrants.
- Novel nouns (H27, H28): 0.37 [0.24, 0.49], against 0.12 [0.04, 0.21] in regular_grouped. Small, and absent in the medium models (H28 by size: 0.27, 0.46, 0.02). Consistent with untrained embeddings limiting the novel-noun test.
- The noun-blind index was the same with and without reassignment (S39: 9.80 vs 9.52; 0.28 [−2.78, 3.35]).
- reassigned_random (S37): index 0.21 [−0.31, 0.72]; noun-specific 0.02 (novel) and −0.08 (trained). Recurrence of a kind elsewhere in the window, without adjacency, taught neither.

## Components (S38), k = 1, same noun

| Corpus | Δ shape | Δ colour | Δ texture | Index |
|---|---|---|---|---|
| Regular, random | −1.55 | 0.01 | −0.12 | −1.56 |
| Regular, B 3.6 | 1.18 | −0.10 | −0.03 | 1.28 |
| Regular, grouped | 6.88 | −2.64 | 0.32 | 9.52 |
| No stable feature, grouped | −3.88 | 0.10 | 0.44 | −3.98 |
| Objects, same-object | 6.87 | 0.99 | 0.13 | 5.87 |
| Objects, different-object | 8.05 | 0.46 | 0.14 | 7.60 |
| Reassigned, grouped | 6.81 | −2.99 | 1.19 | 9.80 |
| Reassigned, random | 0.22 | 0.02 | −0.16 | 0.21 |

Every Part A contrast had the same direction at all three sizes; H26 was positive in 13 of 15 seed-by-size pairs.
