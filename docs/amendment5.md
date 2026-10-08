# Amendment 5: Extensions to Studies 1 to 3

Linked to the project registration (osf.io/qj9hb) and Amendments 1 to 4 (osf.io/x6zhn, Amendment 2, osf.io/qcfjp, osf.io/ya68j). Written 9 October 2026.

## Status and disclosure

This amendment is registered before any model is trained on the new corpora, before any model is scored on the new items, and before any new transcript is read.

When it was written, the author knew all results of the registered studies so far ("Studies 1b, 1c, 1d, 1e"; called Studies 1, 2 and 3 in the manuscript), and had four reviews of the manuscript. These checks on existing data had also been done and are reported as exploratory:

1. In Study 2, the kind-specific index split into its parts. In Pythia 1.4B and larger, shape repeated more after the same noun (+0.08 to +0.19 nats) and colour repeated less (−0.11 to −0.24).
2. The Study 2 kind-specific index rose with Pythia size (Spearman ρ = .72, p = .03).
3. In Study 3, the raw share of same-kind pairs was higher in the Pile (0.60) than in CHILDES (0.53). Per 1,000 words, CHILDES had 10.1 same-kind pairs and the Pile 0.87.
4. In Study 3, a bootstrap over CHILDES corpora and then transcripts gave a 95% interval for B of [3.14, 4.28]; all 33 corpora with enough pairs to compute B had B > 1. (The 2,116 transcripts come from 35 corpora after the 400-word rule, not 36 as first reported.)
5. Reading the Study 2 items showed that the same-noun continuation began "Here is another snerg." and the different-noun continuation "Here is a zull." The kind-specific index therefore confounded the noun with the word "another".

Locked materials are in github.com/synthiumjp/overhypothesis, branch study1b-1c-evaluation, commit b3db968. MD5 checksums of every script, item file, corpus and model list are in `docs/amendment5_checksums.txt`. Run instructions are in `docs/RUN_AMENDMENT5.md`.

## Part A. Study 1 extension: what does grouping teach?

### Questions

A1. Does grouping help because every kind keeps its shape, or would grouping raise the shape-over-colour index without that regularity, through the sentence frames or a habit of repeating recent words?
A2. Does grouping strengthen whichever dimension is stable within kinds?
A3. How much clustering is needed? In particular, does clustering at the level measured in speech to children (B about 3.6) already help?
A4. Do the models tie the preference to the noun when they can tell nouns apart (nouns seen in training)?

### Corpora

All new corpora are built from the five Study 1 Regular corpora (`scripts/make_study1x_corpora.py`). They keep the 32 kinds, 14 exemplars per kind, five frames, 26 passes and token count.

- **NoStable**: every exemplar gets its own randomly drawn shape word, so no dimension is stable within a kind. The substitution is made once per exemplar, so all 26 passes repeat the same 448 exemplars.
- **ColourStable**: shape varies as in NoStable, and each kind keeps one colour word wherever a frame mentions colour.
- Both in **random order** (the exact sentence order of the Regular source) and in **grouped order** (the Study 1c procedure, same seed).
- **Graded clustering** (Regular content, order only): within each pass, each kind's exemplars are cut into runs of four, each run is kept with probability f, and otherwise split into single sentences; all units are shuffled. f = 0.02, 0.12, 0.27 and 0.50 give measured B (W = 100, `cds_burstiness.py`) of 1.37, 3.61, 6.99 and 12.14 (means over seeds). Regular random order has B = 0.91 and Regular grouped order B = 23.1. The values of f were chosen from the B of the corpora alone, before any model was trained.

### Models and scoring

Identical to Study 1: three sizes (tiny, small, medium) × five seeds per corpus, same training settings, Regular tokenizer; 8 corpora × 15 = 120 new models. They are scored on the Study 1 item file (`oneshot_items.csv`, novel nouns) exactly as in Study 1. The ideal observer is fitted to every new corpus.

Trained-noun items (`study1_trained_nouns.py`): one sentence about a trained noun A, then the same frame about A or about another trained noun B. No stable word of A or B in that corpus appears anywhere in the item. 960 items per corpus seed. They are scored on all Study 1 Regular and Scrambled models (random and grouped order) and all 120 new models.

### Primary measure

OI_colour (Δshape − Δcolour) at k = 1, same noun, per model, as in Study 1.

### Confirmatory hypotheses (one-sided, Holm across the four)

- **H11.** Grouped Regular > grouped NoStable.
- **H12.** Grouped NoStable > grouped ColourStable (that is, grouping with a stable colour moves the index towards colour).
- **H13a.** Across the six Regular orders (random, B 1.4, 3.6, 7, 12, grouped), OI_colour rises with log B.
- **H13b.** Regular at B 3.6 > Regular in random order.

**Test.** A linear mixed model with condition and model size as fixed effects and a random intercept for corpus seed (the three sizes trained on one seed share a corpus); one-sided z on the condition coefficient (H11, H12, H13b) or on the slope of log B (H13a, with B measured per corpus). Also reported for every contrast: Welch t over the 15 models, paired t over the 15 matched models, paired t over the five seed means, and the number of matched pairs in the predicted direction. A hypothesis is supported if its Holm-corrected p < .05. If H13b is not supported, equivalence to zero is tested with TOST at ±0.2 nats.

### Secondary analyses (no correction)

- The same contrasts on OI_texture.
- The kind-specific index (same minus different noun) for every condition.
- Size by grouping interaction (Regular random vs grouped).
- **S14.** The kind-specific index with trained nouns, per condition, one-sample t over models (> 0).

### How results will be read

If H11 is supported, grouping helps because of the regularity, beyond frames and recency. If not, the grouping effect in Study 1 is mostly a general habit of repeating the previous shape, and the manuscript will say so. If H12 is supported, grouping strengthens whichever dimension is stable. H13a and H13b show whether the effect grows with clustering and whether the level found in speech to children is enough to measure. If S14 is positive in grouped Regular models, the zero kind-specific index with novel nouns reflected untrained noun embeddings; if it is zero, the models learned a preference that ignores the noun.

## Part B. Study 2 extension: matched wording, more nouns, more seeds, denser checkpoints

### Items (`study2_extension.py`, `english_items_v2.csv`)

All items have k = 1, 16 novel nouns (the 8 of Study 2 plus brell, sprock, fendle, plim, skeb, tulver, grisp, vosh; all unattested in wordfreq), 12 draws, three probed dimensions, and same and different continuation nouns. Same-noun and different-noun continuations always use the same wording. Four sets of 1,152 items:

- **M (primary):** frame A, word list 1; continuation "Here is a snerg." or "Here is a brell."
- **M2:** as M, with "Here is another …" for both continuations.
- **F:** frame B, word list 1: "I found a snerg. The snerg is oval. The snerg is amber. The snerg is silky."
- **W:** as M, with a second word list of eight words per dimension (shape: oblong, hexagonal, conical, tapered, elliptical, tubular, cubic, rounded; colour: magenta, lilac, teal, beige, indigo, lavender, ivory, coral; texture: velvety, waxy, prickly, bumpy, gritty, coarse, slippery, rough; mean Zipf 3.29, 3.39, 3.31).

### Models

- Final checkpoints of all 13 Study 2 models (all four sets; the two 70B models on set M only).
- Dense Pythia checkpoints (set M): 410M, 1B, 1.4B and 2.8B at steps 2,000, 3,000, 5,000 to 15,000 every 1,000, 18,000, 20,000, 24,000 and 32,000; 6.9B and 12B at 4,000, 8,000, 12,000 and 32,000.
- PolyPythias (van der Wal et al.; set M): nine further seeds of 14M, 31M, 70M, 160M and 410M, plus the original 31M, each at steps 4,000, 16,000, 64,000 and 143,000.

Scoring is unchanged (`english_oneshot.py eval`): full-sequence log probability, half precision above 1B parameters, MLX for the 70B models.

### Measures

Kind-specific index = (Δshape same − Δshape different) − (Δcolour same − Δcolour different), per noun and draw. Raw index = Δshape same − Δcolour same.

### Confirmatory hypotheses (one-sided, Holm across the three)

- **H14a.** On set M, the kind-specific index at the final checkpoint rises with size across the nine original Pythia models (Spearman ρ; permutation p, 10,000 permutations).
- **H14b.** On set M, the kind-specific index pooled over Pythia 410M to 12B at the final checkpoint is above zero. Test: linear mixed model on item-level values with crossed random intercepts for noun, item (noun × draw), repeated shape word and repeated colour word (REML, Powell optimiser). If that model does not converge, a crossed (pigeonhole) bootstrap over nouns, shape words and colour words decides instead. Both are reported.
- **H15.** On set M at step 143,000, the kind-specific index is higher at 410M than at 160M across the ten seeds of each (Welch t).

### Secondary analyses (no correction)

- **S15. Onset.** On set M, for each Pythia size, the first checkpoint from which the 95% interval (bootstrap over nouns) of the kind-specific index stays above zero at every later scored checkpoint; and the same for the raw index.
- **S16. Robustness.** Sets M2, F and W: sign and size of the kind-specific index per model; agreement with set M.
- **S17. Components.** The four components (shape same, shape different, colour same, colour different) and the texture index, per model and set.
- **S18.** The crossed random-effects model applied to the original Study 2 items (exploratory, data already seen).
- Tokens per feature word in each tokenizer; whether results hold when only words of one token are used.
- BabyLM and 70B models: descriptive.

### How results will be read

If the set M kind-specific index is near zero where the original index was positive, the original result reflected the word "another", and the manuscript will say so. H15 shows whether the change of sign between 160M and 410M holds across training seeds.

## Part C. Study 3 extension: adult conversation, rates and what repeated labels refer to

### Data

Adult conversation from TalkBank CABank, English: Santa Barbara Corpus of Spoken American English, CallHome English, CallFriend English (North and South). All speakers; one unit per utterance; the same 70 target kinds, window (100 words) and 400-word rule as Study 3. Download addresses and MD5s will be recorded. If a listed corpus cannot be obtained, the others are used and the deviation reported.

### Confirmatory hypothesis

- **H16.** B in CHILDES differs from B in adult conversation (two-sided; percentile bootstrap over documents; α = .05). No direction is predicted.

### Secondary analyses (no correction)

- **S19. Rates.** Per corpus: target-noun pairs per 1,000 words, same-kind pairs per 1,000 words, the excess over the shuffled baseline per 1,000 words, B weighted by pairs, by transcript and by corpus, and a two-stage bootstrap (corpora, then documents within corpus).
- **S20. Local order only.** B with shuffling only inside blocks of about 500 words, and B on documents cut into 1,000-word segments.
- **S21. All nouns.** B for all common nouns (spaCy en_core_web_sm 3.8.0 lemmas), every corpus alike.
- **S22. Automatic cues** for every same-kind pair: "another" or "other", numbers, a plural form of the noun, and property words (colour, shape, size, material) between the two mentions.
- **S23. Hand coding.** 500 randomly drawn same-kind pairs (300 CHILDES, 100 CABank, 100 Pile), shown without their corpus, coded by the author for referent (same object, different object, generic, unclear) and stated properties (`docs/exemplar_coding_rubric.md`). A second coder codes a random 25% blind; Cohen's kappa is reported. Reported per corpus: the share of pairs coded "different" and the share with a stated property, with Wilson 95% intervals.

### How results will be read

If adult conversation is as clustered as speech to children or more, the clustering is a property of conversation, not of speech to children in particular, and the manuscript will say so. S22 and S23 show how often repeated labels in speech to children come with distinct exemplars or described properties, which is what the Study 1 mechanism needs.

## Part D. Exploratory analyses of existing data

Labelled exploratory in the manuscript: Study 1 contrasts at the level of corpus seeds and the size by condition interaction; the Study 2 components, the kind-specific size correlation and the crossed random-effects re-analysis; the Study 3 rates and weightings above, applied to the existing counts; a table of every pythia-2.8B checkpoint affected by the stale-weights defect, with scores before and after the fix.

## Deviations

Any departure from this amendment will be reported in the manuscript with its reason and, where possible, the registered analysis alongside.
