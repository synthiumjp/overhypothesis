# Amendment 7: Same-object versus different-object clustering (Study 1)

Linked to the project registration (osf.io/qj9hb) and Amendments 1 to 6. Written 12 October 2026.

## Status and disclosure

This amendment is registered before any model is trained on the new corpora. When it was written, the author knew all results of Amendments 5 and 6. The author also knew of a scoring error found on 12 October 2026 and of the corrected results.

**The scoring error.** The Study 1 models were trained on DirectML, where the tie between the input embedding and the output layer was lost, so each model has two separate matrices. The scoring code loaded each model into a tied architecture, so the output matrix overwrote the input embedding. All Study 1 models (Studies 1b, 1c, Amendment 5 Part A and Amendment 6) were rescored with corrected loading, using the same items. The corrected results (`results/rescore/RESULTS.md`) include:
- B = 3.6: 1.43 nats against 0.47 in random order (H13b supported after correction).
- Graded orders: 0.47, 0.83, 1.43, 2.06, 3.53, 6.49.
- 512-token window: 0.50 and 0.80.

All models of this amendment are scored with the corrected loader.

**Analyses already done.** Two analyses were done before this amendment, on data already reported, and are declared here as exploratory:
1. **Effective B** (`scripts/effective_b.py`, `results/rescore/effective_b.json`). The hand coding of Study 3 shows that many nearby repeated labels in natural speech refer to the same object. Scaling the excess clustering by the share of coded pairs that refer to different objects gives B_eff = 1 + (B − 1) × d:
   - speech to children: 1.95 (1.70 with a stricter d);
   - adult conversation: 1.66 (1.33);
   - written text: 1.22 (1.06).
2. **A loss check** (`scripts/loss_check.py`) of the corrected loader, run on a subset of models. It compares each model's loss on its own training corpus under corrected and old loading with its last training losses.

Locked materials: github.com/synthiumjp/overhypothesis, branch study1b-1c-evaluation, at the commit that adds this file. MD5 checksums are in `docs/amendment7_checksums.txt` and run instructions in `docs/RUN_AMENDMENT7.md`.

## Question

In Study 1, every nearby repetition of a kind brought a different exemplar. In natural speech, about half of nearby repeated object labels return to the same object (Study 3). Does nearby repetition teach the regularity when it re-describes the same object, or only when it brings different objects of the kind?

## Corpora (`scripts/make_study1o_corpora.py`; `data/corpora_o/`, `manifest.json`)

**One set of sentences per seed.**
- 32 kinds, the Study 1 nouns, each with its Study 1 stable shape.
- 13 objects per kind. Each object has that shape and its own colour and texture.
- Each object is described in four Study 1 frames: "A {noun} is a {shape} {colour} {texture} thing."; "Look at the {noun}. It is {shape} and {colour}."; "Here is another {noun}. This one is {shape}."; "That is a {noun} because it is {shape}."
- 1,664 sentences per pass and 7 passes: 11,648 sentences and 96,096 words, about as in Study 1.

**Five orders of the same sentences, each sentence equally often:**
- **random**: each pass shuffled.
- **same_grouped**: runs of four, the four descriptions of one object.
- **diff_grouped**: runs of four describing four different objects of one kind, one per frame. A Latin square over objects and frames uses every sentence once per pass.
- **same_b3p6** and **diff_b3p6**: the same runs, each kept with probability f = 0.11 and otherwise split into single sentences. f was chosen from the corpora alone, before any model was trained.

**Measured, means over seeds:**

| Order | B | Same-object share of neighbouring same-kind pairs |
|---|---|---|
| random | 0.96 | 0.06 |
| same_grouped | 24.3 | 0.99 |
| diff_grouped | 24.3 | 0.001 |
| same_b3p6 | 3.58 | 0.76 |
| diff_b3p6 | 3.63 | 0.01 |

## Models and scoring

Identical to Study 1: three sizes × five seeds per corpus, 75 new models, 128-token windows, the Regular tokenizer. They are scored on `oneshot_items.csv` with the corrected loader, and on the trained-noun items.

## Primary measure

OI_colour at k = 1, same noun.

## Confirmatory hypotheses (one-sided, Holm across the three)

- **H20.** diff_grouped > same_grouped.
- **H21.** diff_b3p6 > same_b3p6.
- **H22.** diff_b3p6 > random. This replicates H13b in the new corpora.

**Test.** As in Amendments 5 and 6: a mixed model OI ~ condition + size with a random intercept for corpus seed, and a one-sided z on the condition coefficient. Welch t, paired t over the 15 matched models, paired t over the five seed means, and the number of matched pairs in the predicted direction are reported alongside. A hypothesis is supported if its Holm p < .05.

## Secondary analyses (no correction)

- **S29.** same_b3p6 against random, two-sided, with TOST at ±0.2 nats.
- **S30.** diff_grouped and same_grouped each against random, two-sided.
- **S31.** Components (shape, colour, texture) and index by size, per condition.
- **S32.** H20 to H22 on OI_texture.
- **S33.** Kind-specific index with trained nouns, per condition.
- **S34.** The new random order against Study 1 random order, two-sided.

## How results will be read

- **H20 and H21 supported:** only nearby repetition that brings different objects of a kind teaches the regularity. The paper will then place natural corpora by their clustering of different objects (B_eff), not by B.
- **S29 equivalent to zero:** re-describing the same object adds nothing.
- **S29 positive:** same-object repetition also helps, and B_eff understates natural input.
- **H22** checks that the B 3.6 effect holds in corpora built differently.

## Deviations

Any departure from this amendment will be reported in the manuscript with its reason and, where possible, the registered analysis alongside.
