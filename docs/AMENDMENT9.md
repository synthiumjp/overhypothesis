# Amendment 9: noun familiarity, stored knowledge, and the heads that carry a kind's feature

Project: osf.io/qj9hb. Written 23 October 2026, before any Amendment 9 corpus is built or model trained.
DRAFT 2 (after an independent methods review); not yet registered.

## 1. What is known when this is written

All results of Amendments 1 to 8, the corrected scores, and these exploratory analyses of the 120 fresh-text models
of Amendment 8 (results/recut/):

- With unseen test nouns (input embeddings never trained), grouped training made models carry the context's shape
  to the next mention (grouped Regular: Delta shape 6.88, index 9.52); random order made them expect it to change
  (index -1.56).
- Giving the unseen nouns a typical embedding (mean trained-noun embedding plus a distinct offset of typical length)
  cut the carry in grouped Regular models to Delta shape -0.72. Trained nouns, never offered their own kind's shape,
  gave 0.06. So most of the carry seen with unseen nouns depends on the noun's embedding being atypical.
- Three contrasts kept their direction in all five seeds for unseen, typical and trained nouns: grouped above random,
  Regular above No stable feature (grouped), different-object above same-object runs. The B = 3.5 effect held only for
  unseen nouns. The negative indices held for typical nouns.
- In Reassigned grouped models, trained nouns carried shape forward after the same noun (Delta shape 10.01) and not after
  a different noun (0.08).
- Literature (claude/lit_review_amendment9.md): the population-level trade-off between in-context and in-weights
  learning is established (Chan et al. 2022, 2025; Anand et al. 2025; Kim et al. 2026); heads gated by frequency are
  shown only in pretrained models (Yu et al. 2023; Jin et al. 2024); Kim, Linzen & Smolensky (2022) found the opposite
  ordering of initialisation effects in another task.
- The model code ties input and output embeddings; the tie was broken by the training hardware (DirectML) in every
  model so far.

No Amendment 9 model has been trained. No circuit analysis has been run on any model.

## 2. Questions

1. When a model meets a noun, is the shape it carries from the context displaced by how familiar the noun is
   (an in-distribution embedding), or only by the noun having a stored shape?
2. Which attention heads carry a kind's feature from one sentence to the next mention, and are the same heads used
   when a trained noun's shape can come only from the context?

## 3. Design

### 3.1 Familiarity nouns
New nonce nouns are added to three base corpora. Within every model they fill these cells:

| Knowledge type | Exposures (sentences in the training text) |
|---|---|
| Consistent: the noun always has one shape (its own shape s*) | 10, 100, 1,000, 10,000 |
| Uninformative: a new shape in every sentence, drawn from the base corpus's shape distribution | 10, 100, 1,000, 10,000 |
| Shape-absent: the noun appears only in a frame that names colour and texture but not shape | 10, 100, 1,000, 10,000 |
| Unseen: never in the training text (in the vocabulary only) | 0 |

Two or three nouns per exposed cell (fixed from the pilot) and four unseen nouns: 28 or 40 familiarity nouns. With
three per cell, 90,000 of the 160,000 windows hold one familiarity sentence among about ten base sentences. Every familiarity noun, including unseen
ones, is assigned an own shape s* (for unseen and non-consistent nouns s* is a placebo, used only in items). Nouns
are assigned to cells by a rotation across the ten seeds; the cells each noun visits are listed in
the materials. Colour and texture are drawn at random in every sentence. For comparison, a base kind noun occurs in
about 60,000 sentences per model.

A sixth frame is added, which names colour and texture but not shape: "The {noun} is {colour} and {texture}." It is
used by base kinds in 1 of every 15 of their sentences (so it is not novel) and is the only frame used by
shape-absent nouns. Consistent and uninformative nouns use frames 1 to 5 in the base proportions, balanced
deterministically at 10 exposures.

### 3.2 Base corpora, reading order and exact exposure
Three base corpora, built as in Amendment 8 with every pass arranged afresh:
- Regular, grouped order (builds the carry);
- Regular, random order (comparison);
- Reassigned, grouped order (a kind's shape changes every pass; trained nouns must take it from the context).

Each corpus is built at token level to exactly N windows of 128 tokens plus one, where N = steps x batch (160,000
windows for the 3.4M and 25.6M models, which share a corpus; 256,000 for the 11.0M models). The reading order of the
windows is a seeded permutation written with the corpus and used by the trainer, so every window is read exactly once
and the step at which each window is read is known in advance. Familiarity sentences are inserted between base
sentences (in grouped order only between runs), wholly inside a window's 128 input positions, at most one familiarity
sentence per window. Exposures of each noun are placed in windows read at evenly spaced steps between the end of
warm-up and 95% of training, so the last exposure falls at the same point for every cell. Exposure counts, frame
counts and placement are verified from the tokenised corpus before training; no model is trained on a corpus that
fails a check.

### 3.3 Models
Three sizes (3.4M, 11.0M, 25.6M parameters), ten seeds (42, 123, 456, 789, 1001 and five new seeds listed in the
materials), three base corpora: 90 models, trained with the Amendment 8 settings. All Amendment 9 models, including
the pilot, are trained on one machine (the DirectML GPU used for Amendment 8), so hardware cannot differ between cells. Input and output embeddings are untied explicitly in code, and training stops
if they share storage. An unseen noun's input row is therefore never updated by its own occurrence; weight decay still
shrinks it (about 8% over training in earlier models). A new tokeniser, built as before, adds every nonce noun as a
single token; this is checked before training. Weights are saved every 1,000 steps (weights only) for the analysis of
formation; this replaces the earlier 500-step full checkpoints.

### 3.4 Pilot (before registration)
One pilot seed (2027), not among the ten, is trained in all three base corpora at the 3.4M size only, from corpora built
by the final scripts. The pilot is used only to check the corpora and exposure counts, that every measure runs, that
consistent nouns reach the M1 floor, and to estimate seed and noun variance for the new measures. From it, before
registration, the number of nouns per cell (two or three) is fixed so that power for a 1-nat effect at the strictest
Holm level is at least 0.8, and the head-selection k is checked to select heads with clear prefix-matching scores.
Pilot models are reported but never enter a confirmatory test. No confirmatory model is trained before registration.

### 3.5 Measures
Every artificial word is one token, so all measures are taken at the shape position of a frame-1 continuation,
"A {noun} is a [shape] ...", from the full log-softmax over the ten shape words. Let L(x) be the log probability of
shape x at that position, and Lbar the mean of L over the shapes that are neither the context shape s nor the noun's
own shape s*.
- Carry: L(s) - Lbar, after a frame-1 context sentence about the same noun with shape s (s is never s*).
- Stored preference: L(s*) - Lbar, in the same items (primary); L(s*) - L(s) is reported as competition (secondary).
- Knowledge: L(s*) - mean of L over the other nine shapes, after a neutral prefix (one base sentence about another
  kind) and no context about the noun.
- Agreement (exploratory): carry when the context shape equals s*.
Each measure is averaged over 30 context draws per noun, with every shape used equally often as s. The same items are
given with a different familiarity noun of the same cell in the continuation (noun-specific carry = same minus
different). Unseen nouns are also scored with a typical embedding (the Amendment 8 procedure) as the in-distribution
baseline.

## 4. Hypotheses
Exposure enters as log10(exposure) over the exposed levels (1, 2, 3, 4). Slopes are per seed.

### Manipulation check
M1. In every base corpus, knowledge for consistent nouns is at least 0.2 nats at 1,000 exposures and rises with
exposure. If M1 fails, H30 and H31 are not interpreted.

### Familiarity against stored knowledge (grouped Regular models)
H30. The fall in carry with exposure is steeper for consistent nouns than for shape-absent nouns (interaction:
slope(consistent) - slope(absent) < 0).
H31. At 1,000 and 10,000 exposures (pooled), carry for shape-absent nouns differs from carry for typical-embedding
unseen nouns (two-sided), with an equivalence test against +-1 nat (about one seventh of the 7.6-nat fall from unseen
to typical embeddings in Amendment 8). Equivalence supports "familiarity alone does not
displace the carry beyond what an in-distribution embedding does"; a difference supports an effect of familiarity
itself.
H32. For consistent nouns, the stored preference rises with exposure (slope > 0).

### Heads (grouped Regular and Reassigned grouped models)
Candidate heads are chosen in each model before any familiarity item is scored, by a task-independent signature:
the copying and prefix-matching scores of Olsson et al. (2022) on repeated random sequences of the language's words.
The k heads with the highest product of the two scores are candidates (k = 2, 4 and 6 for the 4-, 6- and 8-layer
models). Ablation replaces a head's output by its mean over base-corpus windows at the shape position of frame-1
sentences.
H33. In grouped Regular models, ablating the candidate heads reduces carry for unseen nouns (a) by more than half
(one-sided t of the proportional reduction against 0.5) and (b) by more than ablating k heads drawn at random from the
same layers (mean over 200 draws excluding candidates; difference > 0).
H34. In Reassigned grouped models, ablating the candidate heads reduces the same-noun carry of trained base nouns
(difference > 0). This asks whether the heads carry an in-distribution noun's feature from context, not only an
unseen token's.
H35. Given H33, the candidate heads' attention from the shape position to the context shape token falls more steeply
with exposure for consistent than for shape-absent nouns (interaction < 0; smallest effect of interest 0.02 of
attention probability per decade). A null does not rule out a gate acting after these heads.

### Replication check (outside the family)
R1. With unseen nouns, carry is larger in grouped than in random Regular models (Amendment 8 replication).

## 5. Analysis
Unit of replication: the seed. Each model gives one value per cell; sizes are averaged within seed (the 3.4M and
25.6M models share a corpus; the 11.0M models read a longer corpus built by the same rules with the same noun
assignment and exposure schedule scaled to their training). One-sided t tests over ten seeds unless stated, Holm
correction across H30 to H35 (H35 only if H33(a) holds). Smallest effects of interest: 1 nat for differences in carry, stored
preference and knowledge (0.2 nats for the M1 floor); 0.3 nats per decade for slopes (about 1 nat from 10 to 10,000
exposures); 0.02 attention probability per decade. A supported
hypothesis must also hold in direction at two of the three sizes. A mixed model (value ~ predictors + size, random
intercepts for seed and noun) is reported alongside. Power: in the Amendment 8 rescoring, seed-level standard deviations of
size-averaged values were 0.5 to 1.5 nats (typical embeddings, grouped Regular: 0.81 for Delta shape; grouped minus
random index: 0.98). With SD 1 nat, a one-sided t at the strictest Holm level (0.05/6) has power 0.19 for a 1-nat effect
and 0.68 for 2 nats with five seeds, and 0.60 and 1.00 with ten. Equivalence within +-1 nat has power 0.29 with five
seeds and 0.80 with ten (simulation, true difference 0). Ten seeds are therefore used. These figures come from a
different measure with eight test nouns; the pilot (section 3.4) re-estimates them, and the number of nouns per cell is
fixed from the pilot before registration.

Exploratory (labelled as such): all analyses in random Regular models; by size; geometry of each noun's rows (norm,
cosine to the base-noun centroid, unconditional logit) as mediators; noun-specific carry for familiarity nouns, which
never repeat in context (generalisation of noun-tied carry, two-sided); activation patching (noising and denoising),
path patching and attention knockout to locate the gate, including patching a familiar noun's embedding or residual
stream into an unseen noun's position; resample ablation and the direct-logit prediction against the observed effect
of ablation (self-repair); the effect of ablation on colour; heads with negative logit attribution to the context's
feature (copy suppression) as an account of the negative indices; formation and transience of the heads and of carry
across saved weights; the agreement items.

## 6. Decision rules and reporting
- Every confirmatory test is reported whatever its outcome. If H33 fails, no set of heads is called the copy circuit.
- Any change after registration is reported as a deviation with its reason.

## 7. Materials
Committed with checksums before training: scripts/make_familiarity_corpora.py (with corpus checks),
scripts/familiarity_items.py, scripts/familiarity_eval.py, scripts/circuit_analysis.py, scripts/amendment9_tests.py,
the trainer changes (explicit untie, fixed reading order, weight saving), the noun-to-cell assignment, and
docs/RUN_AMENDMENT9.md.
