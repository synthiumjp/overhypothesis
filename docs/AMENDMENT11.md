# Amendment 11: confirmatory replication, a larger model, and single-history corpora

Project: osf.io/qj9hb. Written 25 October 2026, after all Amendment 10 results and their exploratory reanalyses, after
the Amendment 11 corpora were built and checksummed, and before any Amendment 11 model is trained. Short test runs
(100 steps, 3.4M and 85.6M sizes, seed 2029) were made only to check that the pipeline runs; they are not results and
were deleted. This amendment turns the exploratory findings that the paper rests on into registered tests on new
models.

## 1. What is known when this is written
All results of Amendments 1 to 10, including the exploratory analyses of the Amendment 10 Part A models
(results/amendment10/REANALYSIS.md, CROSSED.md, scores_formation_a.csv): in stable-shape (Regular) grouped corpora,
changing-shape (rotating) nouns in bursts (runs) copied 2.86 nats [1.99, 3.74] more than fixed-shape bursts at 1,000
exposures; fixed- and changing-shape spaced (single) nouns had equal copy scores (-0.02 [-0.16, 0.11]); fixed-shape spaced
nouns preferred their own shape to the context's by 4.5 to 7.7 nats; the copy score of changing-shape bursts was barely
specific to the noun (at most 0.07 nats); at 1,000 exposures changing-shape bursts copied about 3 nats more than their
row geometry predicts; the queried noun's history mattered more than the context noun's (1.35 against 0.39 nats).
These were exploratory. Effects were largest in the 3.4M models. No model has been trained on the Amendment 11 corpora
beyond the pipeline checks above.

A dry run of scripts/a11_tests.py on the Amendment 10 Part A data, relabelled with the Amendment 11 seed names, gives
the values that an exact replication would give: H50 2.86, H51 3.39, H52 -0.02 (equivalent), H53 6.09, H54 0.05,
H55 2.84 (seed-level), H56 0.97. These are the Amendment 10 values computed by the registered code, not new data.

## 2. Questions
1. Do the exploratory findings of Amendment 10 replicate in new models with new seeds?
2. Does the main finding (changing-shape bursts copy more than fixed-shape bursts) hold in a model more than three
   times larger than the largest so far?
3. Does the main finding hold when every new noun in a corpus has the same history, so that new nouns with other
   histories are absent? And does a noun copy differently when such nouns are absent?

## 3. Materials and rules (all fixed before training)
- Corpora: scripts/make_familiarity_corpora_a11.py, which calls the Amendment 10 Part A builder unchanged. Base text:
  stable-shape kinds in runs of four, frame 7 in one sentence in fifteen. Three noun designs:
  - mixed: the Amendment 10 Part A design (fixed, no-shape and changing histories x spaced and bursts x 100 and 1,000
    exposures, three nouns per cell, four unseen nouns);
  - pure_changing: all 36 familiar nouns changing-shape (spaced and bursts x 100 and 1,000, nine per cell), four unseen;
  - pure_fixed: likewise, all fixed-shape.
  In all designs 18 familiar nouns have 100 exposures and 18 have 1,000, so new nouns fill the same share of the text.
- Seeds: Part A and Part B 2029, 3037, 4049, 5051, 6067, 7079, 8081, 9091, 10103, 11117 (new); Part C the first five.
  The seeds are new, but the assignment of noun forms to cells follows the Amendment 10 rule by seed position, so the
  k-th Amendment 11 seed uses the same noun forms in the same cells as the k-th Amendment 10 seed. Model weights,
  window order and the placement of nouns in the text are new.
- Assignment: docs/amendment11_assignment.json. Corpus checksums: docs/amendment11_corpus_manifest.json. A checker re-reads
  every corpus (exposures, units, frames, shapes within runs, window placement, between-run placement, frame-7 share).
  Script checksums: docs/amendment11_checksums.txt.
- Models: scripts/train_model.py, as in Amendment 10 (untied embeddings, fixed reading order, weights every 1,000 steps),
  sizes tiny (3.4M), small (10.9M; earlier documents round this to 11.0M), medium (25.6M), and a new size large:
  12 layers, 12 heads, width 768, feed-forward 3,072 (85.6M parameters with untied embeddings), trained on the same
  160,000 windows as medium (batch 16, 10,000 steps, peak learning rate 1e-4). Large uses 1,000 warm-up steps; medium
  uses 500. The device may be DirectML, CUDA, Apple MPS or CPU; the PyTorch version is recorded with the results.
- Failed models. A run that reaches a non-finite loss stops and counts as failed (the trainer writes
  FAILED_nonfinite_loss.txt, which is kept). A finished model fails if its final training loss (the loss of the last
  training batch, as logged) is more than 0.5 nats above the median final loss of the models with the same design and
  size, the model itself included (scripts/a11_model_check.py), or if any registered score is not finite (copy score,
  compete and noun-specific copy score of the new nouns; copy score, row norm and relative distance in the row
  analysis; copy score in the crossed analysis). A failed model is retrained once with the same seed and command; GPU
  training is not bit-for-bit deterministic, so a transient numerical failure need not recur. If it fails again it is
  excluded. Every retrain and exclusion is recorded in docs/AMENDMENT11_DEVIATIONS.md. Nothing is imputed.
- Complete seeds. Seed is the unit, so a seed enters a part only when all of its models for that part pass. A model
  missing from any scoring output that the part uses counts as missing (Part A uses the main scores, the row analysis
  with all 18 familiar nouns at 1,000 exposures, and the crossed analysis with all 36 pairs of familiar histories at
  1,000 exposures). Part A needs at least 8 complete seeds (all three sizes, mixed design); Part B at least 5 large
  models; Part C at least 4 seeds with both pure designs at all three sizes. With fewer, that part is reported as not
  interpreted. All tests below run over the complete seeds (ten when nothing fails).
- Initial rows. H55 needs each model's initial input rows, rebuilt from its seed by scripts/a10_reanalysis.py (checked
  on the unseen nouns, whose rows never change). If they are not reproduced for a model, that is a fault of the
  rebuild, not of the model: its seed is left out of H55 only, and H55 is not interpreted if fewer than 8 seeds remain.
- Large models are trained in the seed order listed above. If training must stop early, the test uses every large
  model completed, decided before any large model is scored.
- Scoring: scripts/familiarity_eval.py (copy score, knowledge, noun-specific copy, and compete = L(s*) - L(s), where L is
  the log probability of a shape after the test sentence, s* the noun's own shape and s the shape in the context),
  scripts/a10_reanalysis.py (row geometry with rebuilt initial rows), scripts/a10_crossed.py (crossed histories), run on
  final models. Tests: scripts/a11_tests.py.

## 4. Part A: confirmatory replication (30 models: mixed design, three sizes, ten seeds)
Seed is the unit: cell value per model = mean over its nouns (pooled exposure levels weighted equally), sizes averaged
within seed, t tests over seeds. Holm correction over H50, H51, H52 (difference), H53, H55, H56. A hypothesis is
supported when its Holm threshold is met and its direction holds at two of the three sizes (for H52, the direction of
the pooled estimate); otherwise it is not supported. A hypothesis that meets its Holm threshold but fails the size rule
is reported as such. Equivalence tests (H52, H54): two one-sided tests against +-0.3 nats at alpha .05, not
Holm-adjusted; the 90% interval is reported with each. "Not interpreted" means the tests are computed and reported but no conclusion is drawn from them.
- M3 (manipulation check). Unseen nouns, with their own (trained, original) input rows, copy more than 1 nat (one-sided, alpha .05). If M3 fails, Part A is not
  interpreted.
- H50 (primary). At 1,000 exposures, changing-shape bursts copy more than fixed-shape bursts (one-sided).
- H51. Changing-shape bursts copy more than changing-shape spaced nouns, 100 and 1,000 pooled (one-sided; replicates H42).
- H52. Fixed-shape spaced nouns and changing-shape spaced nouns, pooled: two-sided difference test and equivalence within
  +-0.3. Reading: "different but within +-0.3" if both hold; "equivalent" if only the equivalence test holds;
  "different (fixed above)" or "different (fixed below)" if only the difference is supported; otherwise "inconclusive".
- H53. For fixed-shape spaced nouns, pooled, compete = L(s*) - L(s) is above zero (one-sided): the noun's own shape is
  predicted over the shape in the context.
- H54. At 1,000 exposures, the noun-specific copy score of changing-shape bursts is within +-0.3 (equivalence). The
  noun-specific copy score is the copy score when the test sentence names the same noun as the context, minus the copy
  score when the context is unchanged and the test sentence names the next noun of the same cell (same history and
  exposure level); items whose context shape is the other noun's own shape are dropped from both sides. Reading: if the equivalence test holds, any
  noun-specific part is below 0.3 nats, whether or not it differs from zero (both are reported).
- H55. At 1,000 exposures, changing-shape bursts copy more than their row geometry predicts. Per model, an ordinary
  least-squares regression over the 18 familiar nouns of the copy score on cosine with the initial row, distance from
  the trained kinds' mean row (relative), row norm and a changing-burst indicator; the indicator's coefficient is
  averaged over sizes within seed and tested one-sided over the complete seeds. Direction by size: mean coefficient per
  size. Reported alongside, not tested: the pooled noun-level regression with size and seed fixed effects and
  seed-clustered errors (one-sided, t reference with G - 1 degrees of freedom, G = number of seeds), and the same
  seed-level test without no-shape nouns.
- H56. At 1,000 exposures, the queried noun's history matters more than the context noun's: (queried changing bursts minus
  queried fixed bursts) minus (context changing bursts minus context fixed bursts), averaged over familiar histories of
  the other noun, is above zero (one-sided).

| Result | Reading |
|---|---|
| H50 supported | The main finding replicates: copying lasts where a noun's own earlier sentences were the only source of its shape. |
| H50 meets Holm but holds at only one size | Not supported. The paper says at which size it holds and does not claim the effect across sizes. |
| H50 does not meet Holm | The Amendment 10 result does not replicate; the paper's central exploratory claim is withdrawn. |
| H51 supported | Bursts, not changing shapes alone, produce the extra copying. |
| H51 not supported | Burst presentation is not shown to add copying for changing-shape nouns; the claim is withdrawn. |
| H52 equivalent or different but within +-0.3, H53 supported | A stored shape does not reduce copying by more than 0.3 nats but outweighs it in prediction. |
| H52 different (not within +-0.3) | A stored shape changes how much a spaced noun copies; the paper reports the direction and drops the claim that it does not. |
| H52 inconclusive | The paper says the data cannot tell whether a stored shape changes copying. |
| H53 not supported | The claim that a stored shape wins in prediction is withdrawn. |
| H54 equivalent | Any part of the copying tied to the noun is below 0.3 nats. |
| H54 not equivalent | A noun-specific part of 0.3 nats or more cannot be ruled out; the paper says so. |
| H55 supported | Changing-shape bursts give the row a copying tendency beyond its distance from initialisation and its norm. |
| H55 not supported, H50 supported | The extra copying can be accounted for by the measured row geometry. |
| H55 and H50 not supported | There is no extra copying for the row geometry to explain; H55 is not read further. |
| H56 supported | Copying is set mainly by the queried noun's history. |
| H56 not supported | The queried and context nouns' histories are not shown to differ in effect. |
| M3 fails | Part A is not interpreted: the models do not copy from context at all, so no comparison of copying is meaningful. |

## 5. Part B: a larger model (10 models: mixed design, size large, ten seeds)
- M3B. Unseen nouns in large models, with their own input rows, copy more than 1 nat (one-sided, alpha .05). If M3B fails, Part B is not
  interpreted.
- H57. At 1,000 exposures, changing-shape bursts copy more than fixed-shape bursts in large models (one-sided t over
  seeds, alpha .05). If H57 is supported, the main finding holds at 85.6M parameters. If not, the paper reports that
  the main finding was not shown at 85.6M parameters and states the estimate and interval. All other large-model measures are descriptive. Large models
  are not part of the Part A Holm family.

## 6. Part C: single-history corpora (30 models: pure_changing and pure_fixed, three sizes, five seeds)
- M3C. In each pure design, unseen nouns, with their own input rows, copy more than 1 nat (one-sided, alpha .05). If either fails, Part C is not
  interpreted.
- H58. At 1,000 exposures, changing-shape bursts in pure_changing corpora copy more than fixed-shape bursts in
  pure_fixed corpora with the same seed and size (one-sided t over seeds, sizes averaged, alpha .05; direction at two of
  three sizes). H58 compares two corpora that differ only in the history of their new nouns. It shows whether the
  main difference appears without nouns of other histories in the same corpus. It does not by itself say whether
  those nouns change copying; H59 tests that.
- H59 (secondary). For changing-shape bursts and, separately, fixed-shape bursts at 1,000 exposures: copy score in the
  pure corpus minus copy score in the mixed corpus with the same seed and size (seeds 2029 to 6067, Part A models),
  sizes averaged; at least 4 seeds. Two-sided t and equivalence within +-0.3, alpha .05 each, without a size rule:
  "different but within +-0.3" if both hold, "equivalent" if only the equivalence test holds, "different (pure above
  mixed)" or "different (pure below mixed)" if only the difference test holds, otherwise "inconclusive". A difference
  would mean that new nouns with other histories change how a noun copies.

| Result | Reading |
|---|---|
| H58 supported | The main difference does not need nouns of other histories in the same corpus. |
| H58 not supported, H50 supported | The main difference is shown only when histories are mixed; the paper says so and treats competition between new nouns as a possible cause. |
| H58 and H50 not supported | The main difference is not shown in either design. |
| H59 equivalent (or within +-0.3) for both | Nouns of other histories do not change copying by more than 0.3 nats. |
| H59 different (not within +-0.3) for either | Other new nouns change that history's copying; the paper reports the direction and size. |
| H59 inconclusive, or split | The paper reports each history's result separately and draws no conclusion for an inconclusive one. |
| M3C fails | Part C is not interpreted. |

## 7. Power
From the Amendment 10 seed-level standard deviations (dry run): H50 (SD 1.22 nats) has power above .99 for the
observed 2.86 nats and .76 for half of it at the strictest Holm threshold (alpha .0083; .96 at .05). H55 (SD 1.21)
has similar power (.76 for half at .0083). H56 (SD 0.56) has power .98 for the observed 0.97 nats but only .46 for
half of it at .0083 (.80 at .05); a smaller H56 effect may be missed. The equivalence tests of H52 and H54 have power
above .95 if the true difference is zero. H59, with five seeds, has much less: about .8 if the seed-level SD is 0.18
nats, .3 if it is 0.3; an inconclusive H59 is likely and is reported as such. H58 with five seeds has power .99 for the H50 effect and .69 for half of it if the SD is
as in H50; the SD of a between-corpus contrast may be larger. For H57 the SD is unknown; with ten seeds a 1-nat
effect with SD 1 nat has power about .90 (about .58 with five). These figures ignore the rule that the direction
must hold at two of three sizes, so they overstate power somewhat.

## 8. Reporting
Every registered test is reported whatever its outcome, with the number of models and seeds used and any exclusions.
A part with no models is reported as not run. When a part is not interpreted, its test values are still reported but
no outcome is read from them.
Part A results replace the corresponding exploratory Amendment 10 results as the paper's evidence; the Amendment 10
values remain reported as the discovery sample. Deviations are recorded in docs/AMENDMENT11_DEVIATIONS.md.

## 9. Materials (committed with checksums before any model is trained)
scripts/make_familiarity_corpora_a11.py and the builders it calls (scripts/make_familiarity_corpora_a10.py,
scripts/make_familiarity_corpora.py, scripts/corpus_generator.py), docs/amendment11_assignment.json,
docs/amendment11_corpus_manifest.json, scripts/train_model.py, scripts/amendment9_manifest.py,
scripts/familiarity_items.py, scripts/familiarity_eval.py, scripts/oneshot_eval.py, scripts/a10_reanalysis.py,
scripts/a10_crossed.py, scripts/a11_model_check.py, scripts/a11_tests.py, the Amendment 9 tokenizer
(data/tokenizers/amendment9_tokenizer.json), this document and docs/RUN_AMENDMENT11.md. Their MD5 checksums are in
docs/amendment11_checksums.txt.
