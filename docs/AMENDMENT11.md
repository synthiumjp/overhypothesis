# Amendment 11: confirmatory replication, a larger model, and single-history corpora

Project: osf.io/qj9hb. Written 25 October 2026, after all Amendment 10 results and their exploratory reanalyses, after
the Amendment 11 corpora were built and checksummed, and before any Amendment 11 model is trained. This amendment
turns the exploratory findings that the paper rests on into registered tests on new models.

## 1. What is known when this is written
All results of Amendments 1 to 10, including the exploratory analyses of the Amendment 10 Part A models
(results/amendment10/REANALYSIS.md, CROSSED.md, scores_formation_a.csv): in stable-shape (Regular) grouped corpora,
changing-shape (rotating) nouns in bursts (runs) copied 2.86 nats [1.99, 3.74] more than fixed-shape bursts at 1,000
exposures; fixed- and changing-shape spaced (single) nouns had equal copy scores (-0.02 [-0.16, 0.11]); fixed-shape spaced
nouns preferred their own shape to the context's by 4.5 to 7.7 nats; the copy score of changing-shape bursts was not
specific to the noun (at most 0.07 nats); at 1,000 exposures changing-shape bursts copied 3.19 nats [2.46, 3.91] more than
their row geometry predicts; the queried noun's history mattered more than the context noun's (1.35 against 0.39 nats).
These were exploratory. Effects were largest in the 3.4M models. No model has been trained on the Amendment 11 corpora.

## 2. Questions
1. Do the exploratory findings of Amendment 10 replicate in new models with new seeds?
2. Does the main finding (changing-shape bursts copy more than fixed-shape bursts) hold in a model three times larger
   than the largest so far?
3. Does it hold when every new noun in a corpus has the same history, so that other new nouns cannot shape it?

## 3. Materials (all fixed before training)
- Corpora: scripts/make_familiarity_corpora_a11.py, which calls the Amendment 10 Part A builder unchanged. Base text:
  stable-shape kinds in runs of four, frame 7 in one sentence in fifteen. Three noun designs:
  - mixed: the Amendment 10 Part A design (fixed, no-shape and changing histories x spaced and bursts x 100 and 1,000
    exposures, three nouns per cell, four unseen nouns);
  - pure_changing: all 36 familiar nouns changing-shape (spaced and bursts x 100 and 1,000, nine per cell), four unseen;
  - pure_fixed: likewise, all fixed-shape.
  In all designs 18 familiar nouns have 100 exposures and 18 have 1,000, so new nouns fill the same share of the text.
- Seeds: Part A and Part B 2029, 3037, 4049, 5051, 6067, 7079, 8081, 9091, 10103, 11117 (new); Part C the first five.
- Assignment: docs/amendment11_assignment.json. Corpus checksums: docs/amendment11_corpus_manifest.json. A checker re-reads
  every corpus (exposures, units, frames, shapes within runs, window placement, between-run placement, frame-7 share).
- Models: scripts/train_model.py, as in Amendment 10 (untied embeddings, fixed reading order, weights every 1,000 steps),
  sizes tiny (3.4M), small (11.0M), medium (25.6M), and a new size large: 12 layers, 12 heads, width 768, feed-forward
  3,072 (85.4M parameters), trained on the same 160,000 windows as medium (batch 16, 10,000 steps, peak learning rate
  1e-4, 1,000 warm-up steps). The large size is added to train_model.py before any model is trained; the device may be
  DirectML, CUDA, Apple MPS or CPU.
- Scoring: scripts/familiarity_eval.py (copy score, knowledge, noun-specific copy, L(s) - L(own) as -compete),
  scripts/a10_reanalysis.py (row geometry with rebuilt initial rows), scripts/a10_crossed.py (crossed histories), run on
  final models. Tests: scripts/a11_tests.py. A dry run on the relabelled Amendment 10 data reproduces the Amendment 10
  exploratory values.

## 4. Part A: confirmatory replication (30 models: mixed design, three sizes, ten seeds)
Seed is the unit: cell value per model = mean over its nouns (pooled exposure levels weighted equally), sizes averaged
within seed, t tests over ten seeds. Holm correction over H50, H51, H52 (difference), H53, H55, H56. A hypothesis is
supported when its Holm threshold is met and its direction holds at two of the three sizes. Equivalence tests (H52, H54):
two one-sided tests against +-0.3 nats at alpha .05, not Holm-adjusted.
- M3 (manipulation check). Unseen nouns copy more than 1 nat. If M3 fails, Part A is not interpreted.
- H50 (primary). At 1,000 exposures, changing-shape bursts copy more than fixed-shape bursts (one-sided).
- H51. Changing-shape bursts copy more than changing-shape spaced nouns, 100 and 1,000 pooled (one-sided; replicates H42).
- H52. Fixed-shape spaced nouns and changing-shape spaced nouns, pooled: two-sided difference test and equivalence within
  +-0.3. Reading: "equivalent" if TOST p < .05; "different" if the difference is supported; otherwise "inconclusive".
- H53. For spaced nouns, pooled, L(s) - L(own) is lower for fixed-shape than changing-shape nouns (one-sided): a stored
  shape outweighs the copied one in prediction.
- H54. At 1,000 exposures, the noun-specific copy score of changing-shape bursts (same-noun context minus other-noun
  context of the same history) is equivalent to zero within +-0.3.
- H55. At 1,000 exposures, changing-shape bursts copy more than their row geometry predicts: in a noun-level regression of
  the copy score of familiar nouns on cosine with the initial row, distance from the trained kinds' mean row (relative),
  row norm, size and seed, with a changing-burst indicator, the indicator's coefficient is above zero (one-sided, standard
  errors clustered by seed; direction by size from per-size regressions).
- H56. At 1,000 exposures, the queried noun's history matters more than the context noun's: (queried changing bursts minus
  queried fixed bursts) minus (context changing bursts minus context fixed bursts), averaged over familiar histories of
  the other noun, is above zero (one-sided).

| Result | Reading |
|---|---|
| H50 supported | The main finding replicates: copying lasts where a noun's own earlier sentences were the only source of its shape. |
| H50 not supported | The Amendment 10 result does not replicate; the paper's central exploratory claim is withdrawn. |
| H52 equivalent, H53 supported | A stored shape does not reduce copying but outweighs it in prediction. |
| H54 equivalent | The copying is not tied to the noun. |
| H55 supported | Changing-shape bursts give the row a copying tendency beyond its distance from initialisation. |
| H56 supported | Copying is set mainly by the queried noun's history. |

## 5. Part B: a larger model (10 models: mixed design, size large, ten seeds)
- H57. At 1,000 exposures, changing-shape bursts copy more than fixed-shape bursts in large models (one-sided t over ten
  seeds, alpha .05). If not supported, the paper reports that the main finding was not shown at 85M parameters. All other
  measures are descriptive. If fewer than ten large models can be trained, the test uses those that were (at least five).

## 6. Part C: single-history corpora (30 models: pure_changing and pure_fixed, three sizes, five seeds)
- H58. At 1,000 exposures, changing-shape bursts in pure_changing corpora copy more than fixed-shape bursts in pure_fixed
  corpora with the same seed and size (one-sided t over five seeds, sizes averaged, alpha .05; direction at two of three
  sizes).
- Exploratory: the same history's copy score in pure against mixed corpora (seeds 2029 to 6067), as a measure of how much
  other new nouns shape a noun's copying.

## 7. Power
From the Amendment 10 seed-level standard deviations: H50 (SD about 1.2 nats) has power 1.0 for the observed 2.9 nats
and 0.75 for half of it at the strictest Holm threshold (alpha .0083); H56 (SD about 0.6) has power 0.997 for the observed
0.97 and 0.66 for half; the equivalence tests have power above 0.95 if the true difference is zero. For H57 the standard
deviation is unknown; with ten seeds a 1-nat effect with SD 1 nat has power about 0.9.

## 8. Reporting
Every registered test is reported whatever its outcome. Part A results replace the corresponding exploratory Amendment 10
results as the paper's evidence; the Amendment 10 values remain reported as the discovery sample. Deviations are recorded
in docs/AMENDMENT11_DEVIATIONS.md.
