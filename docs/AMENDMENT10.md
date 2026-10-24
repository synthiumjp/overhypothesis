# Amendment 10: familiarity without isolation, matched frames, embedding rows, and pretrained models

Project: osf.io/qj9hb. Written 24 October 2026, after the Amendment 9 results (osf.io/zde87) and before any Amendment 10
corpus is built, model trained or pretrained model run on these items. This is the last expansion of the study.

## 1. What is known when this is written
All results of Amendments 1 to 9, including the Amendment 9 confirmatory, formation and geometry results
(results/amendment9/). In grouped Regular models carry of a shape given in the context fell from 4.53 nats for unseen nouns
to under 1 nat by 100 exposures for every noun history; fixed-shape nouns carried 0.3 to 0.8 nats less than shape-absent
nouns (exploratory); familiar shape-absent nouns carried 0.33 nats more than unseen nouns given a typical embedding,
equivalent within 1 nat (H31). A blind review of the resulting paper raised two confounds that this amendment addresses:
(1) in Amendment 9 every familiarity noun appeared at most once per window, so familiarity was confounded with never recurring
in context, whereas trained kinds recur in runs; (2) shape-absent nouns never appeared in the test frame. The literature
check for a pretrained test is in claude/lit_review_amendment10.md. No Amendment 10 model or pretrained model has been run on
these items. Code was tested only on random or 100-step smoke models.

## 2. Questions
1. Does a familiar noun keep its carry if it recurs in context during training (runs), as the trained kinds do?
2. With frames matched, does a stored shape lower carry relative to a shape-absent noun?
3. Which property of a noun's input row matters for carry in the Amendment 9 models?
4. In pretrained models, does giving a familiar noun an atypical input embedding increase how much colour it takes from the
   context, and do colour-typical nouns take less than colour-variable ones?

## 3. Part A: controlled corpora (60 models)
Built by scripts/make_familiarity_corpora_a10.py; everything not stated is as in Amendment 9 and its addendum (exact windows,
fixed reading order, exposures evenly spaced from 5% to 95% of read positions, at most one familiarity unit per window,
units only between runs, random in-window position, untied embeddings, the Amendment 9 tokenizer and nouns, weights saved
every 1,000 steps, one machine).
- Cells, three nouns each, rotating over the ten Amendment 9 seeds: history {fixed shape, shape-absent} x presentation
  {single, run} x exposures {100, 1,000}; plus four unseen nouns (28 nouns).
- Presentation: single = one sentence per unit; run = four consecutive sentences about the noun in one window (25 or 250
  runs). Exposures count sentences.
- Frames matched: fixed-shape nouns appear only in frame 1, "A {noun} is a {shape} {colour} {texture} thing."; shape-absent
  nouns only in frame 7, "A {noun} is a {colour} {texture} thing." Trained kinds use frame 7 in one sentence in fifteen
  (replacing the Amendment 9 frame 6), so frame 7 is not new.
- Base corpora: Regular grouped (confirmatory) and Reassigned grouped (exploratory). Sizes 3.4M, 11.0M, 25.6M; ten seeds;
  60 models.
- Measures: the Amendment 9 measures and items (carry, stored preference, knowledge; 36 context draws per noun; unseen
  nouns also scored with the typical embedding).

## 4. Part B: pretrained models (inference only)
- Nouns: 221 candidate English nouns were coded before any model was run by two independent passes (separate AI agents, the
  same fixed prompt; data/amendment10/coding_pass_A.csv, _B.csv; exact agreement 94%, weighted kappa 0.96). Colour-typical:
  both passes rate 1 (one strongly typical colour) and name the same colour; colour-variable: both rate 3. Nouns that are also
  colour terms are excluded. scripts/pythia_prep.py keeps nouns that are one token in the Pythia tokenizer, counts them in the
  Pile (infini-gram, proportional estimates) and pairs each colour-typical noun with the nearest unused colour-variable noun
  within 0.3 log10 units of count (at most 40 pairs). The resulting lists (data/amendment10/nouns.csv, colours.csv) are fixed
  before any model output is seen and committed with checksums.
- Models: Pythia deduplicated, final checkpoint (step 143000): 1B, 1.4B, 2.8B, 6.9B, 12B (primary). 410M is the pilot. 70M and
  160M are descriptive. Exploratory: Pythia-1.4B deduplicated at steps 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000.
  Float32 on CPU, one machine.
- Items: three frames (scripts/pythia_familiarity.py), each giving the noun a colour and then mentioning the noun again; the
  basic colour terms that are single tokens. Carry = mean over frames and ordered colour pairs of log p(c1 | context c1) -
  log p(c1 | context c2), with the colour probabilities renormalised over the colour set.
- Embedding conditions (input row only): original; atypical (a random direction with the noun's own norm); typical (mean row
  of the listed nouns plus a distinct orthonormal offset of typical length).
- Pilot (Pythia-410M, never in a test): checks that the items run and fixes the smallest effect of interest for H47 as
  max(0.3 nats, 0.2 x the pilot's mean carry with original embeddings), computed by scripts/a10_tests.py pilotB.

## 5. Part C: embedding rows in the Amendment 9 models (inference only)
scripts/a9_rowswap.py, on the 60 final grouped models of Amendment 9. For unseen nouns: own row; a random direction scaled to
the mean norm of the trained kinds' rows with no shift toward their mean (randnorm); own direction at that norm; the trained
kinds' mean row. For shape-absent nouns at 10,000 exposures: own row; own direction scaled to the unseen nouns' mean norm.

## 6. Hypotheses
Part A (grouped Regular models; 100 and 1,000 exposures pooled; seed-level as in Amendment 9):
- M2 (manipulation check). Unseen nouns carry more than 1 nat. If M2 fails, Part A is not interpreted.
- H40. Shape-absent nouns presented in runs differ in carry from shape-absent nouns presented singly (two-sided), with an
  equivalence test against +-0.3 nats. A difference with runs above singles supports the account that isolation, not
  familiarity, removed carry in Amendment 9; equivalence supports familiarity.
- H41. The same contrast for fixed-shape nouns (two-sided, equivalence +-0.3).
- H42. With frames matched, fixed-shape nouns presented singly carry less than shape-absent nouns presented singly (one-sided).
- H43. Shape-absent nouns presented singly differ from unseen nouns given the typical embedding (two-sided, equivalence +-0.3),
  the frame-matched counterpart of H31.
Part B (each primary model; nouns or matched pairs as units):
- H44. Carry is higher with the atypical than with the original embedding (paired over nouns, one-sided).
- H45. Carry is higher with the atypical than with the typical embedding (paired over nouns, one-sided).
- H46. With original embeddings, colour-typical nouns carry less than their matched colour-variable nouns (paired over pairs,
  one-sided).
- H47. The increase in H44 differs between colour-typical and colour-variable nouns (paired over pairs, two-sided), with an
  equivalence test against the pilot's smallest effect of interest.
Part C (grouped Regular models):
- H48. Unseen nouns given a random direction at the trained kinds' norm carry less than with their own rows (one-sided).

## 7. Analysis and decision rules
- Part A: seed is the unit; cell means per model, sizes averaged within seed; t tests over ten seeds; Holm across H40 to
  H43. A hypothesis is supported when its Holm threshold is met and its direction holds at two of the three sizes; equivalence
  when the TOST p is below .05. The smallest effect of interest is 0.3 nats, the smallest history effect in Amendment 9.
  Power: Amendment 9 seed-level standard deviations of comparable contrasts in grouped Regular models were 0.26 to 0.28 nats,
  which gives power of about 0.95 for a 0.5-nat difference at the strictest Holm level and about 0.8 for equivalence at
  +-0.3. The Reassigned contrasts are exploratory.
- Part B: Holm across H44 to H47 within each primary model; a hypothesis is supported when it holds in at least four of the
  five primary models. A mixed model (carry ~ condition x group + model, random intercepts for noun) is reported alongside.
- Part C: one test at alpha .05, supported if it also holds at two of the three sizes; the other conditions are exploratory.
- Every confirmatory test is reported whatever its outcome. Changes after registration are reported as deviations.
- Exploratory: Reassigned Part A results; formation across saved weights; knowledge and stored preference; Pythia checkpoints;
  Pythia 70M and 160M; colour knowledge (prior entropy) as a covariate; all Part C conditions other than H48.

## 8. Materials (committed with checksums before any corpus is built or model run)
scripts/make_familiarity_corpora_a10.py, docs/amendment10_assignment.json, scripts/familiarity_items.py and
scripts/familiarity_eval.py (with the Amendment 10 design option), scripts/amendment9_manifest.py, scripts/pythia_prep.py,
scripts/pythia_familiarity.py, scripts/a9_rowswap.py, scripts/a10_tests.py, data/amendment10/ (coding passes, candidate list;
nouns.csv and colours.csv after the preparation step), docs/RUN_AMENDMENT10.md, and the recorded corpus checksums
(docs/amendment10_corpus_manifest.json).
