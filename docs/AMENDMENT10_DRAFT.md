# Amendment 10 (DRAFT, not registered): familiarity and colour carry in pretrained models

Project: osf.io/qj9hb. Drafted 24 October 2026, after the Amendment 9 results and before any pretrained model has been
run on these items. For JP's review; nothing is registered, built or run.

## 1. Why
Amendment 9 (90 small transformers, artificial language) found that a noun stops taking its shape from the context once
it is familiar, whether or not it has a stored shape; a familiar noun that never appeared with a shape behaves like an
unseen noun given a typical embedding. Where the context stays useful for familiar nouns (Reassigned training), they keep
some carry, and less when their history gives a stored shape. Reviewers will ask whether this holds in real models. The
literature (claude/lit_review_amendment10.md) shows that pretrained models rely on memory more for frequent entities
(Mallen et al. 2023; Kandpal et al. 2023; Du et al. 2024; Kim et al. 2026), but frequency is always confounded with
knowing the answer, and nobody has swapped a familiar word's representation while holding its knowledge fixed.

## 2. Questions
1. Does giving a familiar noun an atypical input embedding make a pretrained model take its colour from the context more
   (the real-model counterpart of the unseen tokens of Amendments 8 and 9)?
2. Does a typical (mean) embedding behave like a familiar one?
3. Among familiar nouns of matched frequency, do nouns with a typical colour take less colour from the context than nouns
   whose colour varies (the counterpart of fixed-shape against uninformative nouns where context is useful)?

## 3. Design
### 3.1 Models
Pythia deduplicated models, final checkpoints: 1B, 1.4B, 2.8B, 6.9B and 12B (primary; in-context use is weak below about
1B, Du et al. 2024); 70M, 160M and 410M reported descriptively. Inference only, float32, on one machine (the Mac Studio,
CPU), so results are deterministic. Exploratory: Pythia-1.4B deduplicated at steps 1k, 2k, 4k, 8k, 16k, 32k, 64k and
143k, against estimated exposure (Pile count x step / 143k).

### 3.2 Nouns
Three groups of concrete, single-token (with leading space, GPT-NeoX tokenizer) English nouns, 40 each, frequency-matched
on Pile counts (infini-gram, v4_piletrain_llama; treated as proportional estimates):
- Colour-typical: nouns with one typical colour (banana, snow, grass, ...).
- Colour-variable: artefacts whose colour varies (car, shirt, cup, ...).
- Colour-rare (exploratory): concrete nouns rarely described by colour.
Group membership is coded before any model is run, by two independent coding passes (two separate AI agents with a fixed
prompt and scale), with disagreements dropped; the coding prompt, model version and lists are registered. Each noun's
stored colour knowledge is also measured from the model itself (entropy of its colour distribution after a neutral
frame) and used as a covariate.

### 3.3 Items and measure
Context and query frames, for example "The {noun} was {c1}. Later the {noun} was" (several frames, registered). Colours:
8 to 10 single-token colour words. At the query position the probabilities of the colour words are renormalised over the
colour set. Carry for a noun = mean over ordered colour pairs (c1, c2), c1 != c2, of
log p(c1 | context c1) - log p(c1 | context c2). The noun's own colour prior cancels in this contrast.

### 3.4 Embedding conditions (input embedding only; output embedding unchanged)
- Original.
- Atypical: a random vector with the norm of the noun's own row (fixed seed per noun).
- Typical: the mean input embedding of all nouns in the set, plus a distinct offset of typical length (as Amendments 8-9).

## 4. Hypotheses (draft)
H40. Carry is higher with the atypical than with the original embedding (all nouns pooled).
H41. Carry is higher with the atypical than with the typical embedding (a typical embedding behaves familiar).
H42. With original embeddings, colour-typical nouns carry less than frequency-matched colour-variable nouns.
H43 (two-sided, with equivalence at the smallest effect of interest). The increase from H40 does not differ between
colour-typical and colour-variable nouns.

## 5. Analysis (draft)
Unit: noun. For each primary model, a paired (H40, H41) or Welch (H42) test over nouns; Holm across H40 to H43 within
each model; a hypothesis is supported if it holds in at least four of the five primary models. A mixed model (carry ~
condition x group + model, random intercepts for noun and frame) is reported alongside. Smallest effect of interest and
power from a pilot on Pythia-410M (not a primary model). Exploratory: checkpoint trajectory; colour-rare nouns; nonce and
rare multi-token words; Land & Bartolo (2024) under-trained tokens as a naturally atypical condition.

## 6. Open decisions for JP
- Whether to run this at all (it breaks the stopping rule set before Amendment 9).
- Machine: the Mac Studio (shared with the Foresight Engine; about a day of CPU time) or the PC if memory allows 12B.
- Whether the noun coding by two AI agent passes is acceptable, or whether a human coder is needed.
