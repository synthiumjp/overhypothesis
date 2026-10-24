# Amendment 10: recurrence, matched frames, embedding rows, and pretrained models

Project: osf.io/qj9hb. Written 24 October 2026, after the Amendment 9 results (osf.io/zde87) and before any Amendment 10
corpus is used for training, any model is trained or any pretrained model is run on these items. Revised after an
independent methods review. This is the last expansion of the study.

## 1. What is known when this is written
All results of Amendments 1 to 9, including the Amendment 9 confirmatory, formation and geometry results
(results/amendment9/). In grouped Regular models, carry of a shape given in the context fell from 4.53 nats for unseen nouns
to under 1 nat by 100 exposures for every noun history; fixed-shape nouns carried 0.3 to 0.8 nats less than shape-absent
nouns (exploratory); familiar shape-absent nouns carried 0.33 nats more than unseen nouns given a typical embedding,
equivalent within 1 nat (H31). In Reassigned grouped models familiar nouns kept 0.2 to 1.9 nats. A blind review of the
resulting paper raised two confounds: (1) every familiarity noun appeared at most once per window, so familiarity was
confounded with never recurring in context, while the trained kinds recur in runs; (2) shape-absent nouns never appeared in
the test frame. A literature check for a pretrained test is in claude/lit_review_amendment10.md. Code has been run only on
random or 100-step smoke models.

## 2. Questions
1. Does a familiar noun keep its carry if it recurs in context during training? Does it matter whether its recurrences
   make the context useful (a shape that stays the same within a run but changes between runs)?
2. With frames matched, does a stored shape lower carry relative to a noun that never appeared with a shape?
3. Which property of a noun's input row matters for carry in the Amendment 9 models?
4. In pretrained models, does giving a familiar noun an atypical input embedding increase how much colour it takes from the
   context, and do colour-typical nouns take less than colour-variable ones?

## 3. Part A: controlled corpora (60 models)
Built by scripts/make_familiarity_corpora_a10.py. Everything not stated is as in Amendment 9 and its addendum (exact windows,
fixed reading order, exposures evenly spaced from 5% to 95% of read positions, at most one familiarity unit per window,
units only between runs, random in-window position, untied embeddings, the Amendment 9 tokenizer and nouns, weights saved
every 1,000 steps, one machine).
- Cells, three nouns each, rotating over the ten Amendment 9 seeds: history x presentation x exposures {100, 1,000}, plus four
  unseen nouns (40 nouns). Histories:
  - fixed: frame 1 ("A {noun} is a {shape} {colour} {texture} thing."), always the noun's own shape s*;
  - absent: frame 7 ("A {noun} is a {colour} {texture} thing."), which names no shape;
  - rotating: frame 1 with a shape drawn from the corpus's shape distribution, a new one for every sentence when presented
    singly (Amendment 9's uninformative history) and one per run, changing between runs, when presented in runs (like the
    Reassigned kinds).
- Presentation: single = one sentence per unit; run = four consecutive sentences about the noun inside one window. Exposures
  count sentences (25 or 250 runs), so runs and singles are matched in sentences, not in units or spacing.
- Trained kinds use frame 7 in one sentence in fifteen (replacing Amendment 9's frame 6), so frame 7 is not new. Frame 7 is
  one token shorter than frame 1; we report this rather than add a filler word, which the tokenizer would split.
- Base corpora: Regular grouped (confirmatory) and Reassigned grouped (exploratory). Sizes 3.4M, 11.0M, 25.6M; ten seeds.
- Measures: the Amendment 9 measures and items (carry, stored preference, knowledge; 36 context draws per noun; unseen nouns
  also scored with the typical embedding), plus the expectation of a shape rather than a colour at the query position
  (log of the summed probability of the ten shapes minus that of the ten colours; exploratory).
- Corpus checksums are recorded before training (docs/amendment10_corpus_manifest.json); a checker re-reads every corpus and
  verifies exposures, units, frames, shapes within runs, window placement, between-run placement and the frame-7 share.

## 4. Part B: pretrained models (inference only)
- Nouns. 221 candidate nouns were coded before any model was run, by two independent passes (separate AI agents, the same
  fixed prompt): colour typicality (data/amendment10/coding_pass_A.csv, _B.csv; exact agreement 94%, weighted kappa 0.96)
  and kind, count/mass and polysemy (kind_pass_A.csv, _B.csv; agreement 99.5%, 100% and 96%). Colour-typical: both rate 1 and
  name the same colour; colour-variable: both rate 3. Excluded: nouns that are also colour terms, nouns both passes flag as
  polysemous, and nouns that are not one token in the Pythia tokenizer. scripts/pythia_prep.py counts each noun in the Pile
  (infini-gram, lower-case singular, the token ids used are logged and checked) and pairs each colour-typical noun with the
  nearest unused colour-variable noun of the same count/mass type within 0.3 log10 units (at most 40 pairs). This gave
  22 pairs (44 nouns) and 11 colour words, fixed on 24 October 2026 before any model was run (data/amendment10/nouns.csv). Kind
  (natural/artifact) is a covariate: colour-typical nouns are mostly natural kinds. The lists (data/amendment10/nouns.csv,
  colours.csv) are fixed, with checksums, before the pilot, and nothing about the items may change after the pilot.
- Models: Pythia deduplicated, final checkpoint (step 143000): 1B, 1.4B, 2.8B, 6.9B, 12B (primary); 410M is the pilot; 70M and
  160M are descriptive. Exploratory: Pythia-1.4B deduplicated at steps 1000 to 143000 (doubling). Float32 on CPU, one machine.
  If a primary model cannot be run, the rules below use the primary models that were run, provided at least four were.
- Items: three neutral frames (scripts/pythia_familiarity.py): "The {noun} was {colour}. The {noun} was", "This {noun} is
  {colour}. This {noun} is", "I saw the {colour} {noun}. The {noun} was"; the basic colour terms that are single tokens. Carry
  = mean over frames and ordered colour pairs of log p(c1 | context c1) - log p(c1 | context c2), with colour probabilities
  renormalised over the colour set.
- Embedding conditions (input row only): original; atypical_init (a random direction with the norm of a row at
  initialisation, initializer_range x sqrt(hidden size): the counterpart of the untrained rows of the unseen nouns in
  Amendments 8 and 9); atypical (the same direction with the noun's own norm; exploratory); typical (the mean row of the listed
  nouns plus a distinct orthonormal offset of typical length).
- Pilot (Pythia-410M, never in a test) checks that the items run and fixes the smallest effect of interest for H48 as
  max(0.3 nats, 0.2 x the pilot's mean increase from original to atypical_init) (scripts/a10_tests.py pilotB).

## 5. Part C: embedding rows in the Amendment 9 models (inference only)
scripts/a9_rowswap.py, on the 60 final grouped models of Amendment 9 (the 'own' condition must reproduce the Amendment 9
scores). Carry depends on the noun only through its input row, so Part C asks which property of the row matters, not whether
the row matters. Unseen nouns: own row; own direction scaled to the mean norm of the trained kinds' rows (ownnorm); a random
direction at that norm (randnorm); the trained kinds' mean row. Shape-absent nouns at 10,000 exposures: own row; own
direction scaled to the unseen nouns' mean norm.

## 6. Hypotheses
Part A (grouped Regular models; 100 and 1,000 exposures pooled; seed-level as in Amendment 9):
- M2 (manipulation check). Unseen nouns carry more than 1 nat. If M2 fails, Part A is not interpreted.
- H40. Absent nouns presented in runs differ from absent nouns presented singly (two-sided; equivalence +-0.3 nats).
- H41. Fixed nouns presented in runs differ from fixed nouns presented singly (two-sided; equivalence +-0.3).
- H42. Rotating nouns presented in runs carry more than rotating nouns presented singly (one-sided).
- H43. Fixed nouns presented singly carry less than absent nouns presented singly (one-sided).
- H44. Absent nouns presented singly differ from unseen nouns given the typical embedding (two-sided; equivalence +-0.3).
- K1 (check for H41). Knowledge of fixed nouns is equivalent within +-1 nat between runs and singles. If not, H41 is not read
  as evidence about recurrence, because runs may change how strongly s* is stored.
Readings (registered). Each two-sided contrast is read as "above", "below" (Holm threshold met and direction at two of three
sizes), "equivalent" (TOST p < .05; equivalence tests are not Holm-adjusted) or "inconclusive".

| Result | Reading |
|---|---|
| H40 or H41 above | Recurrence in training preserves carry: isolation contributed to the Amendment 9 loss of carry. |
| H40 and H41 equivalent | Recurrence alone does not preserve carry. |
| H40 below | Recurring with a varying slot after "is a" teaches the noun not to copy that slot. |
| H42 supported | A familiar noun keeps carry when its own recurrences make the context useful; familiarity alone does not switch carry off. |
| H42 not supported | Familiarity reduces carry even when recurrence makes the context useful. |
| H43 supported | A stored shape lowers carry with frames matched. |
| H44 equivalent | With frames matched, a familiar absent noun is equivalent to an unseen noun with a typical embedding. |

Part B (each primary model; nouns or matched pairs as units):
- H45. Carry is higher with atypical_init than with the original embedding (paired over nouns, one-sided).
- H46. Carry is higher with atypical_init than with the typical embedding (paired over nouns, one-sided).
- H47. With original embeddings, colour-typical nouns carry less than their matched colour-variable nouns (paired over pairs,
  one-sided).
- H48. The increase in H45 differs between colour-typical and colour-variable nouns (paired over pairs, two-sided), with an
  equivalence test against the pilot's smallest effect of interest.
Part C (grouped Regular models):
- H49. Unseen nouns given their own direction at the trained kinds' norm carry less than with their own rows (one-sided).
  If supported, the size of the row alone lowers carry; if not, where the row points matters (the other conditions,
  exploratory, show where).

## 7. Analysis and decision rules
- Part A: seed is the unit; cell means per model, sizes averaged within seed; t tests over ten seeds; Holm across H40 to H44.
  A hypothesis is supported when its Holm threshold is met and its direction holds at two of the three sizes. Smallest effect
  of interest 0.3 nats (the smallest history effect in Amendment 9). Power: Amendment 9 seed-level standard deviations of
  comparable contrasts at 100 and 1,000 exposures were 0.26 to 0.38 nats (single cells at 100 exposures 0.46 to 0.52; no
  estimate exists for runs). With SD 0.27 a 0.5-nat difference has power 0.98 at alpha .01 and equivalence within +-0.3
  has power 0.89 if the true difference is 0 (0.69 if it is 0.1); with SD 0.375 these are 0.80 and 0.52. Inconclusive
  outcomes are therefore possible and are reported as such.
- Part B: Holm across H45 to H48 within each primary model. H45 to H47 are supported when Holm is met in all but at most one
  of the primary models run (at least four). H48 is read as "typical larger", "variable larger", "equivalent" or
  "inconclusive" by the same count, with the direction required to agree. A mixed model (carry ~ condition x group + model +
  kind, random intercepts for noun and pair, random slopes for condition by noun) is reported alongside. The five models
  share training data and nouns, so agreement across them shows robustness across size, not across items.
- Part C: one test at alpha .05, supported if it also holds at two of the three sizes. randnorm minus ownnorm (expected near
  zero) and the other conditions are exploratory.
- Every confirmatory test is reported whatever its outcome. Changes after registration are reported as deviations.
- Exploratory: Reassigned Part A results; slot-type expectation; knowledge and stored preference; formation across saved
  weights; Pythia checkpoints, 70M and 160M; the own-norm atypical condition; kind and colour knowledge (prior entropy) as
  covariates; all Part C conditions other than H49.

## 8. Materials (committed with checksums before any corpus is used or model run)
scripts/make_familiarity_corpora_a10.py, docs/amendment10_assignment.json, scripts/familiarity_items.py and
scripts/familiarity_eval.py (with the Amendment 10 design option and the slot measure), scripts/amendment9_manifest.py,
scripts/pythia_prep.py, scripts/pythia_familiarity.py, scripts/a9_rowswap.py, scripts/a10_tests.py, data/amendment10/
(coding passes, candidate list; nouns.csv, colours.csv and prep_log.json after the preparation step),
docs/RUN_AMENDMENT10.md, docs/amendment10_corpus_manifest.json and docs/amendment10_checksums.txt.
