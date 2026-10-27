# Cold-read review (ACL)

## 1. Summary

In small transformers trained on an artificial object language, a noun keeps copying a shape from the previous sentence only if its training sentences came in bursts whose shape changed between bursts, so that its own recent context was the only way to predict the shape. That tendency lives in the noun's input embedding along a direction shared across nouns; it is not bound to the noun and it spreads to other nouns, while never-seen nouns copy mainly because their embeddings are untrained.

It fits only by dropping half the abstract, which also calls its own headline "expected": the paper has not decided what its contribution is.

## 2. Cohesion

Mostly one story (three accounts, histories built to separate them), weakened because the main result is conceded as expected (l. 98–100) and the "beyond the expected" findings are a grab-bag, two of them following from the design (W2, W3).

The point where a reader gets lost is **Section 3, "Three experiments" (l. 210–242)**. Before any result, the reader must hold three corpus types, four histories, two presentations, five exposure levels, Experiment 1's two flaws, the fix, the new word-class confound the fix created, and Experiment 3's three parts. "Stable/shifting" (corpus) and "fixed/changing" (noun) are near-synonyms at two levels. By Section 4.4 the reader is flipping back to Table 1.

## 3. ARR review

**Strengths**
- Careful design: frequency, storability and arrangement are manipulated per word within one model, and fixed-shape bursts control for how often the embedding is updated.
- Discovery is separated from confirmation: Experiment 3 (new seeds, 85.6M, single-history corpora) replicates the primary effect (3.68 nats at all sizes; 4.18 at 85.6M).
- Failures (H30, H33, H59, Pythia) and confounds are reported openly.
- The unseen-embedding result (4.1) matters for anyone probing with nonce words.

**Weaknesses, ranked**
1. **The rejection of the memory account does not follow** (serious; fixable in text). Table 2 says memory also predicts changing > fixed bursts, so H50 cannot separate memory from useful context. The only test against memory (spaced fixed vs changing) went memory's way (−0.25, reliably below zero), with both at the floor (0.46 vs 0.57). The data show memory acting *in training*: once a shape is storable, copying stops being rewarded (fixed bursts 1.03 vs changing 4.71).
2. **"Added up rather than competed" is overclaimed** (serious; text). The copy score leaves out s*, so a stored shape's logit cannot affect it by construction. "Additive" can only mean "no suppression of the context logit", and that was tested only where copying was at most about 1.6 nats.
3. **Two "unexpected" findings follow from the design** (moderate; text). Noun-blindness: "nothing required the model to check" (l. 440). Spread: in fixed-only corpora unseen nouns also copy less (4.87 vs 6.88), so part of it is corpus-wide; steering shows a shared direction, not transfer during training.
4. **The learned copying is narrow** (moderate; needs experiments). It vanishes after one intervening sentence, keeps 35–40% in another frame and is absent for colour; bursts train exactly the test configuration. This undercuts the 128-word WikiText argument.
5. **The mechanism is thin and partly tautological** (moderate; text). The row is the only noun-specific input, so the row swap and pre-block patching are guaranteed to move copying. Steering is mostly a 3.4M effect (subtracting the direction gives 0.02 at 25.6M; MECHANISM.md). ρ = 0.73 pools 360 non-independent points, unseen nouns included.
6. **Scale** (moderate). 6.36 → 2.44 → 2.24 nats, then 4.18 at 85.6M (other hardware, longer warm-up). Size-averaging lets 3.4M dominate; report per size.
7. **A Pythia null stated as a finding** ("did not transfer"): TOST failed in most models (Table I13), and real nouns' rows were swapped rather than under-trained tokens probed.
8. **The appendices belong to another paper** (serious for reviewability; text). Appendices A–F cover an earlier study (index measure, fixed-file text, a loader bug that flipped verdicts, BabyLM, child speech, 70B models in MLX), and Appendix A says they "do not bear on the claims". H–J use registration labels throughout. E.2 reveals that untied embeddings, which drive Section 4.1, began as a hardware bug; the main text should say so.

None of these is fatal.

**Questions for the authors**
- Was the useful-context account written down before Experiment 1?
- H49 was supported (rescaling the norm lowered copying), yet the text says norm does not matter. Which is the claim?
- In shifting corpora, fixed-shape bursts at 100 exposures copy strongly and have a stored shape. What is knowledge there? That is the real test of additivity.
- On what sequences was prefix matching computed? 0.028–0.037 against 0.027 for uniform is surprising.

**Missing references (confident)**
- Singh, Moskovitz, Hill, Chan, Saxe 2024, "What needs to go right for an induction head?", ICML.
- Chan, Dasgupta, Kim, Kumaran, Lampinen, Hill 2022, "Transformers generalize differently from information stored in context vs in weights", arXiv:2210.05675.
- Xie, Raghunathan, Liang, Ma 2022, "An explanation of in-context learning as implicit Bayesian inference", ICLR.
- Prystawski, Li, Goodman 2023, "Why think step by step? Reasoning emerges from the locality of experience", NeurIPS.
- Madsen, Kauchak, Elkan 2005, "Modeling word burstiness using the Dirichlet distribution", ICML.
- Zhang & Nanda 2024, "Towards best practices of activation patching", ICLR.

**Scores**: Soundness 3; Excitement 2.5; Overall 2.5 (3 after the text fixes in Section 6); Confidence 4.

## 4. Writing

The five hardest sentences, with plain rewrites:
1. "That a model copies where copying pays was expected, and nouns met in bursts whose shape changed between bursts did keep copying." → *As expected, nouns kept copying when copying had helped in training: when they came in bursts and their shape changed between bursts.*
2. "It crossed the three histories with spaced and burst presentation (…) at 100 and 1,000 exposures, where copying had fallen below 1 nat and to near zero in Experiment 1, in stable-shape and shifting-shape corpora." → *Each history was presented spaced or in bursts of four. Each noun had 100 or 1,000 sentences. At these levels, copying had nearly vanished in Experiment 1.*
3. "Only the useful-context account predicts that changing-shape bursts copy more than changing-shape spaced nouns, unless bursts leave rows less trained, in which case the representation account predicts it too." → *Bursts might help only because they update the embedding less often. Fixed-shape bursts test this: they get the same number of updates, but their shape can be stored.*
4. "The difference was reliably above zero, so a stored shape did lower copying, and we could not show it was under 0.3 nats; Experiment 1 gave a similar 0.30 …, and Experiment 2 none (…TOST…)." → *A stored shape lowered copying by 0.25 nats: small but real. Experiment 1 found 0.30; Experiment 2 found none.*
5. "Patching the query noun's internal state … moved copying fully if done before the first block and not at all after the second, while patching the prediction position carried all of it from the second block on." → *By the end of the second block, the noun's information has moved to the position where the shape is predicted.*

**Jargon and bookkeeping to remove**
- Terms: "row" (say embedding); a measure named "knowledge"; "placebo shape"; "new nouns" that include "familiar" ones; "trained kinds"; stable/shifting vs fixed/changing.
- Bookkeeping: ten-plus "pre-registered in Experiment 3; Experiment 2: x" asides (move to Table 3); "manipulation check"; "could not be interpreted"; TOST; pointers to Tables J3/J4, Figure I5.
- Appendices: "carry", "rotating", "Amendment 9–11", H40–H59, M1/K1/R1, "ownnorm", "pure_fixed", script paths.

## 5. Specific checks

**Claims stronger than the evidence**: "memory and context added up rather than competed" (W2); "The memory account found little support" (W1); "this tendency spread to other new nouns" (W3); "did not transfer to Pythia" (W7); "What suppresses copying is … not its size" (the registered norm test, H49, was supported); "carried by the noun's input embedding" (abstract; rests on exploratory steering and patching).

**Are conditions motivated before they are tested?** The three accounts are. The shifting-shape and random-order corpora, the no-shape history (later dropped as confounded), the 10,000-exposure level, the tied-embedding check and Pythia are not.

**Grounds for suspecting p-hacking or post-hoc storytelling**: Experiment 3's replication largely defuses this for the primary effect. The risks are elsewhere: eleven amendments in about three weeks; equivalence bounds set from earlier effect sizes; Experiment 2 designed in response to an AI review of Experiment 1; H49's supported effect re-described as negligible, and H43 supported then disowned; mechanism, WikiText and tied-embedding analyses added after the results yet carrying the abstract; and verdicts flipped by a loader bug (Appendix A). Each is disclosed, but together they read as a story built after the results. One main-text paragraph separating confirmatory from exploratory claims would fix this.

## 6. Three text-only changes that would most raise Overall

1. **Reframe the theory**: memory acts in training by removing the reward for copying, which the data support. Drop "added up" and "memory found little support"; lead with the per-word, within-model test of Chan et al. (2022), not "expected".
2. **Cut the appendices to this paper**: remove A–F, keep one registration table in main-text terms, and disclose the untied-embedding accident.
3. **Simplify Section 3 and calibrate**: rename the conditions so corpus and noun terms do not collide, move the bookkeeping into Table 3, report the primary test per size, and label the mechanism (exploratory) and Pythia (null) in the abstract.
