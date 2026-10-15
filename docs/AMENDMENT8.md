# Amendment 8: Fresh order and shape reassignment (Study 1)

Linked to the project registration (osf.io/qj9hb) and Amendments 1 to 7. Written 15 October 2026.

## Status and disclosure

This amendment is registered before any model is trained on the new corpora. When it was written, the author knew all results of Amendments 5 to 7, the corrected Study 1 scores, and two rounds of reviews of the manuscript. Two concerns from those reviews motivate it.

1. **Memorisation.** Each Study 1 model was trained on 128-token windows drawn from one fixed file of 11,648 sentences (about 120,000 tokens). Each position was seen 160 to 290 times, and the last training losses were 0.07 to 0.16 nats per token. The models may have learned the particular file, not only its statistics.
2. **The noun test.** With held-out nouns, whose embeddings were never trained, the noun-specific index was about zero. With trained nouns it was small. In the Study 1 corpora a kind's shape was fixed, so the noun alone predicted the shape. A model could then predict shape from the noun's weights and had no need to tie the copying of shape to the noun.

No model has been trained on the new corpora. The corpora were built once to fix their checksums, and only their clustering (below) was measured. The analysis code was run on simulated data to test it.

Locked materials: github.com/synthiumjp/overhypothesis, branch study1b-1c-evaluation, at the commit that adds this file. MD5 checksums are in `docs/amendment8_checksums.txt`, the corpus manifest (with an MD5 for every corpus file) in `docs/amendment8_corpus_manifest.json`, and run instructions in `docs/RUN_AMENDMENT8.md`. The corpora are too large for the repository (5.3 GB) and are rebuilt from the code; the build must reproduce every MD5 in the manifest.

## Questions

- **Part A.** Do the main arrangement effects hold when no training window recurs? The same few hundred sentences still recur many times, as in Study 1; what does not recur is any stretch of text in a particular arrangement.
- **Part B.** If the noun no longer predicts the shape across training, but the shape still stays fixed within a stretch of text, do models tie the copying of shape to the noun?

## Corpora (`scripts/make_fresh_corpora.py`)

Every corpus uses the Study 1 exemplars of its seed (or the Amendment 7 objects), the Study 1 frames and the Regular tokenizer. Each corpus is a long sequence of passes. Every pass uses each exemplar once, in an order drawn afresh for that pass, by the arrangement rule of the matching Study 1 corpus.

- 448-exemplar corpora: 7,500 passes (4,636 to 4,693 tokens per pass, depending on the seed; 34.8 to 35.2 million tokens).
- Objects corpora: 2,100 passes (1,664 sentences and 16,224 tokens per pass; 34.1 million tokens).

**Part A (fresh order):**
- **regular_random**: Regular exemplars, each pass fully shuffled.
- **regular_b3p6**: Regular, graded order, runs of four kept with f = 0.12 (B about 3.6).
- **regular_grouped**: Regular, runs of four by kind (as Study 1c).
- **nostable_grouped**: No stable feature exemplars (as Amendment 5), grouped.
- **objects_same**: Objects, same-object runs, grouped (as Amendment 7).
- **objects_diff**: Objects, different-object runs, grouped (as Amendment 7).

**Part B (shape reassignment):**
- **reassigned_grouped**: Regular exemplars. In each pass every kind gets a new shape word, drawn uniformly from the ten, and it is used in all of that kind's sentences in that pass. Grouped as regular_grouped.
- **reassigned_random**: the same, fully shuffled. Sentences about the same kind still meet within a training window, at a distance, with the same shape in a pass, so the noun is still a cue here. S37 therefore tests the effect of adjacency, not the absence of a noun cue.

In reassigned_grouped, the next sentence about the same kind repeats the shape almost always, and a sentence about another kind repeats it about one time in ten, as in regular_grouped. The difference is that the noun's shape changes from pass to pass, so it cannot be stored in the weights.

**Held-out files.** For each condition and seed, 100 further passes (27 for Objects) from an independent random generator, never used in training.

**Measured clustering B** (first 11,648 sentences of each training corpus; mean over seeds, with range):

| Corpus | B |
|---|---|
| regular_random | 0.94 (0.92 to 0.97) |
| regular_b3p6 | 3.51 (3.30 to 3.69) |
| regular_grouped | 23.4 (22.7 to 24.0) |
| nostable_grouped | 23.0 (22.5 to 23.6) |
| objects_same | 24.1 (23.8 to 24.6) |
| objects_diff | 24.5 (23.9 to 25.0) |
| reassigned_grouped | 23.1 (22.7 to 23.3) |
| reassigned_random | 0.89 (0.86 to 0.93) |

## Models and training

As in Study 1: three sizes (tiny, small, medium) × five seeds (42, 123, 456, 789, 1001) per corpus, 120 new models, 128-token windows, the same steps, batch sizes and learning rates, on the same PC. The only change is `train_model.py --fresh_windows`: the corpus is cut into non-overlapping 128-token windows, which are drawn in random order without replacement. The run stops if the corpus is too short for every step to use a new window. No window is therefore trained on twice. The fewest windows available are 266,174 (Objects), against 256,000 needed by the small models and 160,000 by the tiny and medium models.

This changes two things besides the absence of repetition. Windows lie on a fixed grid, so each token is trained at one position in one window, while the Study 1 models saw each stretch of text at many positions. And the tiny and medium models use about 60% of the windows, the small models 94 to 96%. Comparisons of fresh with fixed models (S35, S36) concern this whole training regime, not repetition alone.

Intermediate checkpoints (every 500 steps) are deleted after each run, to save disk space; only the final model is kept.

A run that fails to finish is rerun with the same settings. No completed run is excluded.

## Scoring

All models are scored with the corrected loader on `data/wug_tests/oneshot_items.csv` (novel nouns) and on the trained-noun items (`scripts/study1_trained_nouns.py`, family regular). The trained-noun items never use the stable shape of either noun in Regular, which matters only for the Part A models.

## Measures

- **Index:** OI_colour (Δshape − Δcolour) at k = 1, after the same noun.
- **Noun-specific index:** OI_colour at k = 1 after the same noun minus after a different noun, per model.

## Primary test

Seed level. The three sizes are averaged within each corpus seed. The five seed means of two conditions are compared with a one-sided paired t test; a single condition is compared with 0 by a one-sided one-sample t test (4 degrees of freedom). Holm correction is applied within each of the two registered families, Part A (H23 to H26) and Part B (H27 to H29), as in earlier amendments, where each amendment's family was corrected separately. A hypothesis is supported if its Holm p is below .05.

Why seed level. Amendments 5 to 7 used the mixed model as the primary test. The three sizes trained on one seed share a corpus, so the five seeds are the independent units, and the paper reports seed-level estimates. The cost is power: with 4 degrees of freedom, the first Holm step in Part B needs a one-sided p below .017 (t above about 3.0). Small effects will therefore not be supported.

Reported alongside, not used for the decision: a mixed model with size as a fixed effect and a random intercept for seed (as in Studies 1 and 1x), and the number of seeds in the predicted direction.

## Confirmatory hypotheses (one-sided; Holm within Part A and within Part B)

- **H23.** Index: regular_grouped > regular_random.
- **H24.** Index: regular_b3p6 > regular_random.
- **H25.** Index: regular_grouped > nostable_grouped.
- **H26.** Index: objects_diff > objects_same.
- **H27.** Noun-specific index, novel nouns: reassigned_grouped > 0.
- **H28.** Noun-specific index, novel nouns: reassigned_grouped > regular_grouped.
- **H29.** Noun-specific index, trained nouns: reassigned_grouped > 0.

## Secondary analyses (registered, no correction)

- **S35. Memorisation.** Loss (nats per token) on windows from the first 44,800 lines of the model's own training file and of the held-out file of the same condition and seed (`scripts/heldout_loss.py`), for the 90 new Part A models and the 90 matching fixed-corpus models (Study 1 random, B 3.6 and grouped; Amendment 5 No stable feature grouped; Amendment 7 same-object and different-object grouped). Reported: the losses and the gap (held-out minus own) per condition, and the seed-level difference in gap, fixed minus fresh. For fresh models the "own" windows include windows trained on once and windows never used, so their gap is expected to be near zero by design; the informative quantity is the gap of the fixed models.
- **S36. Fresh against fixed.** Each Part A condition against its fixed-corpus counterpart (index, two-sided), and the effects of H23 to H26 in the fixed-corpus models, with the difference in effect.
- **S37. reassigned_random.** Index and both noun-specific indices, and reassigned_grouped against reassigned_random on each (two-sided).
- **S38.** Components (Δshape, Δcolour, Δtexture) per condition; H23 to H29 by size; noun-specific indices of every new condition.
- **S39.** Index: reassigned_grouped against regular_grouped (two-sided).
- **S40.** What the held-out corpora say about the next sentence (as `scripts/corpus_conditionals.py`).

Analysis code: `scripts/amendment8_tests.py`, written and tested on simulated data before training.

## How results will be read

- **H23 to H26 supported:** the direction of the arrangement effects does not depend on memorising a fixed file. Whether their size changes is read from S36. If some are not supported, the paper says which Study 1 findings depend on the fixed file.
- **H27, H28 and H29 supported:** when the noun alone does not predict the shape, small transformers tie the copying of shape to the noun, and the Study 1 noun result reflected corpora in which the noun predicted the shape.
- **H27 and H29 supported but not H28:** models trained with reassignment tie copying to the noun, but not reliably more than models trained on regular_grouped; the Study 1 noun result is then not explained by the noun predicting the shape.
- **H29 supported but not H27:** the models tie copying to trained nouns, and the novel-noun test is limited by untrained embeddings.
- **H27 supported but not H29:** the models tie copying to novel nouns but not reliably to trained nouns. We would report this as unexpected and not interpret it further without a new test.
- **H27 to H29 not supported:** even when the corpus makes the noun the only cue to which shape will repeat, these models did not tie the repetition to the noun.
- Whatever the outcome, the paper reports the fresh-order results alongside the original ones and describes any difference.

## Deviations

Any departure from this amendment will be reported in the manuscript with its reason and, where possible, the registered analysis alongside.
