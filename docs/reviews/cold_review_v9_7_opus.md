# Cold read (ACL AC)

## 1. Summary
In small transformers trained on an artificial language, a noun keeps copying its shape from the previous sentence only if its training sentences came in bursts whose shape changed between bursts, so that only the context could supply the shape. Elsewhere copying fades with exposure, unseen nouns copy because their embeddings are untrained, and a stored shape is said to suppress copying during training rather than at test.

## 2. Narrative
There are two papers, and the seam shows. **Paper A (Exp. 2→3):** changing-shape bursts keep copying alive; the effect lives in the embedding, is not tied to the noun, and spreads to other nouns. It is cohesive, replicated and well controlled. **Paper B (Exp. 1, embedding swaps, Pythia):** nonce-word probes measure untrained embeddings. Most of its registered tests failed (H30, H33, H45–H48). The three-accounts frame looks built to hold both. The Intro says the results "join two of them" (representation and useful context), while the Discussion says "the useful-context and memory accounts are complementary", so in the end all three accounts are partly right. The abstract shows it: the title is about familiar nouns, but the last two sentences are about unseen nouns and Pythia.

Specific seams:
- **Fig. 2 (Exp. 1)** is full-width but only shows that spaced nouns lose copying and that random order removes it. The paper itself says Exp. 1 can't be compared with Exps. 2–3 (60% vs 8% text share, different frames).
- **Setup** includes the "hardware fault" history. Limitations and the AI statement add more.
- **4.4** mentions "mixed or fixed-only corpora" a paragraph before explaining them. "It is carried by the embedding" crams three analyses into one paragraph whose evidence sits only in the appendix.
- **4.5** (Pythia, shifting kinds) and the natural-text paragraph are bolted on; neither tests the main claim.
- **The appendices use another vocabulary**: carry, row, Regular/Reassigned, singles/runs, rotating/shape-absent, atypical_init, Amendments 9–11. Table C1 is needed to translate.

**Cut or move:** the hardware sentence, the H49 norm details, Pythia (keep one sentence) and natural text. Shrink Fig. 2. **Add:** the tied-embedding pattern (W2), a validity check on the typical embedding (W3), and a statement that H50 was exploratory in Exp. 2.

## 3. ARR review
**Strengths:**
- A clean per-word causal design: frequency, whether a shape can be stored, and arrangement are varied separately.
- Pre-registered, with failures reported.
- Replicated on new seeds and checked at 85.6M.
- Strong controls: fixed-shape bursts, the geometry regression, single-history corpora.
- Steering and patching locate the effect.
- Transparent.

**Weaknesses.** None is fatal.

*Fixable in text:*
1. **The memory test is blind to competition.** The copy score is a difference of log-probabilities that leaves out s\*. If logits add, raising s\* changes only the normaliser, which cancels out of that difference. So the finding "at test it barely competes" (0.25 nats) is largely built into the measure. The 5.84-nat preference for s\* *is* competition.
2. **Tied embeddings undercut the headline, and the main text hides it.** In tied 3.4M models at 1,000 exposures, fixed-shape bursts copy 2.39 nats, no-shape bursts 2.86 and fixed-shape spaced nouns 1.49 (scores_tied.csv). "Familiar nouns stop copying" fails here; only Appendix F hints at it.
3. **The typical-embedding control is questionable.** In shifting-kind corpora, trained kinds copy 9.98 nats but the typical embedding copies 1.57 (Table D2), and the plain mean 1.05 (Table E6). An embedding near the centroid suppresses copying even where real familiar nouns copy. So the control cannot show that untrained embeddings cause copying, and the recommended check needs a caveat.
4. **Scale is non-monotonic and confounded.** The key gap is 6.36, 2.44 and 2.24 across the three small sizes, then 4.18 at 85.6M, which used other hardware and a longer warm-up. "(4.2 nats at 85.6M)" in the abstract implies a robustness the data don't show.
5. **"Memory acts in training" is near-circular.** It restates the useful-context account, and training-time memory was never manipulated.

*Needs new experiments:*
- Tied embeddings at more than one size.
- A control built from real familiar nouns' embeddings (scaled-up row swaps).
- A graded share of changing-shape nouns, to test whether the effect spreads.

**Questions:**
- Is L a log-softmax over the vocabulary or over the ten shapes? Does the copy score change when s\*'s logit is ablated?
- If unseen nouns copy only because their embeddings are untrained, why does their copying drop from 7.35 to 4.87 in fixed-only corpora?
- Why do the 10.9M models read 256k windows?

**Scores:** Soundness 3; Excitement 2.5 (the core result is close to "gradient descent keeps copying where copying pays"; what's new is that it is per-token and lives in the embedding); Overall 3 (Findings); Confidence 4.

## 4. Theory
The core inference (changing vs fixed bursts, with geometry regressed out) is sound. These claims go beyond the evidence:
- "The effect of a storable shape appears to arise in training": nothing was manipulated during training, and the test measure is blind (W1).
- "Never-seen nouns copied mainly because their embeddings were untrained": tied unseen nouns still copy 3.96 nats, unseen copying depends on the corpus mix and on order, and the control is suspect (W3).
- "Familiar nouns lose copying unless their own earlier sentences were the only source": true only with untied embeddings (W2).
- "In Pythia the embedding effect did not appear, so this caution applies most to models trained on small or synthetic data": this generalises from a null in one model family.
- "tied or untied": rests on one tied size.

## 5. Writing
The three hardest sentences:
1. *"Where object kinds keep their shapes, nouns lost their copying of a shape from the preceding sentence with exposure, unless their own earlier sentences had predicted a shape that memory could not."*
   → "When each kind keeps its shape, a new noun's habit of copying a shape from the previous sentence fades with training. It survives only for nouns whose own earlier sentences were the only clue to their current shape."
2. *"Only useful context predicts that changing-shape bursts copy more than changing-shape spaced nouns, unless bursts help simply by updating the embedding less often."*
   → "Useful context predicts that changing-shape nouns copy more in bursts. Bursts might also help simply because the embedding is updated in fewer windows; fixed-shape bursts rule this out."
3. *"…most changes were not small enough to count as none."*
   → "…in most models we could not show the change was negligible (within ±0.3 nats)." The same fix applies to 4.3's "not shown to be within ±0.3 nats".

Leftover jargon and bookkeeping:
- "stored preference", "Above geometry", "fixed stand-in".
- The hardware fault, "deleted 100-step pipeline checks", the restarted training loop.
- Script and results paths, a9/a10/a11 filenames, and hypothesis numbers starting at H30.

## 6. Three text changes that would most raise Overall
1. **Make Exp. 2→3 the paper.** Present Exp. 1 and the embedding swaps as motivation and a boundary condition. Move Pythia and natural text to the appendix. End the abstract on bursts.
2. **Be honest about the measures.** Say that the copy score cannot register softmax competition, and retitle 4.3. Report the full tied-embedding pattern and limit the headline to untied models. Flag the typical-embedding anomaly.
3. **Use one vocabulary.** Rewrite the appendices in main-text terms. Move amendment numbers, paths and project history out of the main text into one disclosure sentence.
