# Amendment 9: noun familiarity, stored knowledge, and the circuit that carries a kind's feature

Project: osf.io/qj9hb. Written 23 October 2026, before any Amendment 9 corpus is built or model trained.
DRAFT for review; not yet registered.

## 1. What is known when this is written

All results of Amendments 1 to 8, the corrected scores, and the following exploratory analyses of the 120
fresh-text models of Amendment 8 (results/recut/):

- With unseen test nouns (untrained input embeddings), grouped training made models carry the context's shape
  to the next mention (grouped Regular: Delta shape 6.88, index 9.52); random order made them expect it to change
  (index -1.56).
- Giving the unseen nouns a typical embedding (mean trained-noun embedding plus a distinct offset) cut the shape
  carry in grouped Regular models to Delta shape -0.72 (index 1.46). Trained nouns, never offered their own kind's
  shape, gave Delta shape 0.06.
- Three contrasts kept their direction in all five seeds for unseen, typical and trained nouns: grouped above random,
  Regular above No stable feature (grouped), different-object above same-object runs. The B = 3.5 effect held only
  for unseen nouns. The negative indices held for typical nouns.
- In Reassigned grouped models, trained nouns carried shape forward after the same noun (Delta shape 10.01) and not
  after a different noun (0.08).
- A literature review (claude/lit_review_amendment9.md) found the population-level trade-off between in-context and
  in-weights learning established (Chan et al. 2022, 2025; Anand et al. 2025; Kim et al. 2026), frequency-gated heads
  shown only in pretrained models (Yu et al. 2023; Jin et al. 2024), and an opposite ordering of initialisation effects
  in a different task (Kim, Linzen & Smolensky 2022).

No model with familiarity nouns has been trained. No circuit analysis has been run on any model.

## 2. Questions

1. Is the shape a model carries from context displaced by how familiar the test noun is, or by whether the noun has
   a stored shape?
2. Which attention heads carry a kind's feature from one sentence to the next mention, does grouped training build
   them, and does the noun's familiarity or stored shape switch them off?

## 3. Design

### 3.1 Familiarity nouns
Twenty new nonce nouns (familiarity nouns), each a single token, are added to three base corpora. Within every model
they fill ten cells, two nouns per cell:

- Exposure: 0, 1, 10, 100 or 1,000 sentences in the text the model trains on.
- Consistency: consistent (the noun always has one shape, drawn once per noun and seed) or random (a new shape in every
  sentence). Colour and texture are drawn at random in every sentence for both.

Nouns are assigned to cells by a Latin square that rotates across the five seeds, so no noun is tied to one cell.
Exposure-0 nouns are in the vocabulary but never in the training text.

Each exposure sentence uses one of the five frames, with frame proportions matched to the base corpus. A familiarity
noun's sentences are never adjacent to one another, so a model never sees two sentences about the same familiarity
noun in a row; what it learns about carrying features forward comes only from the base corpus. Exposure sentences are
inserted between base sentences (in grouped order, only between runs, so runs stay intact) and spread evenly through
the text.

### 3.2 Base corpora and exact exposure
Three base corpora, built as in Amendment 8 with every pass arranged afresh:
- Regular, grouped order (builds the carry);
- Regular, random order (comparison);
- Reassigned, grouped order (positive control: a kind's shape changes every pass).

Each corpus is built to exactly the number of 128-token windows a model reads (steps x batch: 160,000 windows for the
3.4M and 25.6M models, 256,000 for the 11.0M models), so every window is read exactly once, in shuffled order. Every
exposure sentence lies wholly inside one window. Exposure counts are therefore exact and are verified from the
tokenised corpus before training.

### 3.3 Models
Three sizes (3.4M, 11.0M, 25.6M), five seeds (42, 123, 456, 789, 1001), three base corpora: 45 models, trained with the
Amendment 8 recipe and settings. A new tokeniser, built as before, includes the 20 familiarity nouns and the 8 original
test nouns as single tokens; this is checked before training. Input and output embeddings remain untied, as in all
earlier models, so results are comparable; an exposure-0 noun's input embedding is never trained, while its output
row receives the usual gradient through the softmax. Model weights are saved every 1,000 steps (weights only) for the
analysis of formation.

### 3.4 Test items
For every familiarity noun, with 25 draws of features (k = 1):
- (a) Carry items: as in earlier amendments; a frame-1 context sentence, then two continuations with the same noun (and,
  separately, a different familiarity noun from the same cell) that differ only in the probed feature. For consistent
  nouns the context and continuations never use the noun's own shape.
- (b) Stored-shape items (consistent nouns, all exposures): the context gives shape s (not the noun's own); the two
  continuations offer s and the noun's own shape s*. Delta_stored = log P(s*) - log P(s).
- (c) Knowledge check (all nouns): no context; two sentences "A {noun} is a {shape} {colour} {texture} thing." that
  differ only in the shape: the noun's own shape against another (consistent nouns) or two random shapes (random nouns).
The 8 original test nouns and the 32 base kind nouns are scored on the Amendment 8 items as reference points.

### 3.5 Measures
- Carry: Delta shape on (a), same noun; also the index (Delta shape - Delta colour) and the noun-specific index.
- Stored preference: Delta_stored on (b).
- Knowledge: preference for the own shape on (c).
- Exposure enters analyses as log10(exposure + 1) (0, 0.30, 1.04, 2.00, 3.00).

## 4. Hypotheses

### Manipulation check
M1. In every base corpus, the knowledge preference (c) for consistent nouns rises with exposure (seed-level slope
above zero). If M1 fails, H30 to H32 are not interpreted.

### Familiarity and stored knowledge (grouped Regular models)
H30. For consistent nouns, carry (Delta shape, same noun) falls as exposure rises (slope below zero).
H31. At exposures 100 and 1,000 (pooled), carry is larger for random nouns than for consistent nouns. (Prediction:
stored knowledge, not familiarity, displaces the context.)
H32. For consistent nouns, the stored preference (b) rises with exposure (slope above zero).

### Arrangement
H33. At exposures 0 and 1 (pooled, both consistency types), carry is larger in grouped Regular models than in
random Regular models.

### Positive control (Reassigned grouped models)
H34. For random nouns, the noun-specific index rises with exposure (slope above zero): a model ties the carried shape
to a noun it can recognise.

### Circuit (grouped Regular models)
Head identification uses half of the feature draws (identification items); tests use the other half (test items).
For each model, a head is a candidate copy head if, on identification items with exposure-0 nouns, (i) its attention
from the position predicting the continuation's shape word to the context's shape token and (ii) its direct logit
attribution to the context's shape word are both in the top 10% of heads. The candidate set is fixed before any test
item is scored.
H35. On test items with exposure-0 nouns, mean-ablating the candidate heads (replacing their output by its mean over
test items) reduces carry by at least half, and by more than mean-ablating as many randomly chosen heads from the same
layers (seed-level difference above zero).
H36. In the candidate heads, attention from the shape-predicting position to the context's shape token falls as
exposure rises for consistent nouns (slope below zero).

## 5. Analysis

Unit of replication: the seed. Each model gives one value per cell (mean over items); sizes are averaged within seed.
Slopes are least-squares slopes against log10(exposure + 1), computed per seed; pooled comparisons are differences of
seed means. Tests are one-sided t tests over the five seeds (4 degrees of freedom), with Holm correction across
H30 to H36 (seven tests). M1 is a manipulation check outside the family. The smallest effect of interest remains
0.2 nats. A mixed model (value ~ predictor + size, random intercept for seed) is reported alongside.

Exploratory (labelled as such):
- carry and stored preference by model size; the same analyses in random Regular models;
- the original test nouns with typical and norm-matched random embeddings;
- geometry of each noun's input and output rows (norm, cosine to the base-noun centroid, unconditional logit) as
  mediators of carry;
- activation patching (noising and denoising) and path patching to locate the gate: whether familiarity reduces the
  copy heads' attention (query-key) or a later component overrides their output; patching a familiar noun's embedding
  or residual stream into an unfamiliar noun's position;
- formation of the candidate heads and of carry across the saved checkpoints, by base corpus;
- heads with negative logit attribution to the context's feature (copy suppression) as an account of the negative
  indices;
- items with the context shape attached to a different familiarity noun (context-constrained use).

## 6. Decision rules and reporting

- Every confirmatory test is reported whatever its outcome. If H35 fails, no head is described as "the" copy circuit.
- If a corpus check fails (exposure counts, adjacency, single-token nouns), the corpus is rebuilt before training;
  no model is trained on a corpus that fails a check.
- Any change after registration is reported as a deviation with its reason.

## 7. Materials

Scripts, item files, corpus checksums and run instructions will be committed and their checksums listed before
training: scripts/make_familiarity_corpora.py, scripts/familiarity_items.py, scripts/familiarity_eval.py,
scripts/circuit_analysis.py, scripts/amendment9_tests.py, docs/RUN_AMENDMENT9.md.
