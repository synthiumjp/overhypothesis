# Amendment 6: The window test (Study 1)

Linked to the project registration (osf.io/qj9hb) and Amendments 1 to 5 (osf.io/x6zhn, Amendment 2, osf.io/qcfjp, osf.io/ya68j, osf.io/eb28h). Written 11 October 2026.

## Status and disclosure

This amendment is registered before any model with a 512-token training window has been trained. The only exception is a 100-step smoke test of one model, used to check the code and then deleted; it was not scored on the test items except to confirm that the scorer runs.

When it was written, the author knew all results of Amendment 5. For Study 1, these were:
- H11, H12 and H13a supported.
- H13b not supported: at B = 3.6 the index was 0.24 nats, against 0.18 in random order.
- The registered equivalence test for H13b gave p = .047 at ±0.2 nats.

The window measure below (the share of sentences with an earlier same-kind sentence in the training window) was devised after those results were known. The prediction in S24 was fitted to them. Both are therefore declared here, with their numbers, before any new model is trained.

One file locked by Amendment 5 changes. `scripts/train_model.py` gains two optional arguments (`--seq_len`, `--keep_tokens_per_step`). Without them it behaves exactly as before. All Amendment 5 models were trained before the change. Its Amendment 5 MD5 (b95dd5dffef2d457a36a81be4c758100) therefore no longer matches the current file; the earlier version is at commit c2bccb4.

Locked materials are in github.com/synthiumjp/overhypothesis, branch study1b-1c-evaluation, at the commit that adds this file. MD5 checksums are in `docs/amendment6_checksums.txt`. Run instructions are in `docs/RUN_AMENDMENT6.md`.

## Question

Study 1 models are trained on windows of 128 tokens (about 12 sentences), taken at random offsets from the running text. In random order, few windows hold two sentences about the same kind. Grouping puts them together.

Is co-occurrence within the training window what matters? If so, a longer window should let random order, and the clustering level of speech to children, teach the expectation. Or does it matter that same-kind sentences are close together, beyond sharing a window?

## Measure of within-window co-occurrence

`scripts/window_cooccurrence.py measure` tokenises each corpus as training does and takes a window at every token offset. Among the sentences that begin in a window, it counts the share that have an earlier sentence about the same kind in that window. Means over the five corpus seeds:

| Corpus | W = 128 | W = 512 |
|---|---|---|
| Regular, random | 0.150 | 0.474 |
| Regular, B 1.4 | 0.160 | 0.479 |
| Regular, B 3.6 | 0.211 | 0.498 |
| Regular, B 7 | 0.285 | 0.528 |
| Regular, B 12 | 0.403 | 0.585 |
| Regular, grouped | 0.670 | 0.746 |
| NoStable, random | 0.150 | 0.474 |

With a 512-token window, random order reaches a higher share than the B 12 corpus has at 128 tokens.

## New models

Three corpora, already in the repository, each trained in 3 sizes × 5 seeds (45 new models):
- Regular, random order (`data/corpora`)
- Regular, B 3.6 (`data/corpora_x/graded_b3p6`)
- NoStable, random order (`data/corpora_x/nostable_random`)

Training is identical to Study 1 except for two settings: `--seq_len 512 --keep_tokens_per_step`. The window is 512 tokens. The batch is cut to a quarter (tiny and small 8, medium 4), so every step and every model sees the same number of tokens as before. Steps, learning rates, warm-up, schedule, tokenizer and seeds are unchanged. The only other change to the model is the learned position table, which has 512 rows instead of 128. The models are scored on `oneshot_items.csv` exactly as in Study 1, and on the trained-noun items as in Amendment 5.

## Primary measure

OI_colour at k = 1, same noun, per model, as in Study 1.

## Confirmatory hypotheses (one-sided, Holm across the three)

- **H17.** Regular random order, W = 512 > Regular random order, W = 128 (the Study 1 models).
- **H18.** Regular at B 3.6, W = 512 > Regular at B 3.6, W = 128 (the Amendment 5 models).
- **H19.** Regular random order, W = 512 > NoStable random order, W = 512.

**Test.** As in Amendment 5: a linear mixed model with condition and size as fixed effects and a random intercept for corpus seed; one-sided z on the condition coefficient. Welch t over the 15 models, paired t over the 15 matched models, paired t over the five seed means and the number of matched pairs in the predicted direction are reported alongside. A hypothesis is supported if its Holm-corrected p < .05. If H17 is not supported, equivalence to zero is tested with TOST at ±0.2 nats on the same model.

## Secondary analyses (no correction)

- **S24. Co-occurrence prediction.** On the 90 Regular models trained with W = 128 (six orders), OI_colour ~ share + size with a random intercept for seed gives a slope of 2.19 nats per unit share (SE 0.21). Fitted before training (`results/study1w/prediction.json`), it predicts mean indices of **0.88** for Regular random at W = 512 and **0.94** for Regular B 3.6 at W = 512, against observed means of 0.18 and 0.24 at W = 128. For each condition we report the observed mean with its 95% t interval over the 15 models. If the interval contains the prediction, within-window co-occurrence accounts for the effect of order. If it lies wholly below, sentences about the same kind must be close together, not only in the same window. If it lies wholly above, the longer window helps in some further way.
- **S25.** H17 to H19 on OI_texture.
- **S26.** Indices after two and four exemplars, new conditions.
- **S27.** Size × window interaction for H17.
- **S28.** Kind-specific index with trained nouns, new conditions (one-sample t > 0).

## How results will be read

- **H17 and H18 supported, S24 consistent.** The order effect in Study 1 works through what a training window holds. Clustering at the level of speech to children then helps once the window is long enough. The manuscript will say that the H13b result depends on the window.
- **H17 not supported, or S24 below.** Same-kind sentences must be close, not merely in the same window.
- **H19.** Shows whether a longer window helps without a stable dimension.

These are claims about these models and corpora. A 512-token window holds about 50 of these short sentences, so the result bears on how much text a learner can relate at once, not on any particular human memory span.

## Deviations

Any departure from this amendment will be reported in the manuscript with its reason and, where possible, the registered analysis alongside.
