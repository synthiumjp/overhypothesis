# Cold-read review

## 1. Summary
In small transformers trained on an artificial language where kinds keep their shapes, familiar nouns stop copying a shape from the preceding sentence. The exception is nouns whose previous sentence, in training, was the only predictor of their current shape (changing-shape bursts). That retained copying is a general "copy the latest shape" gate carried by the noun's embedding, not a noun–shape binding, and it spreads to other nouns. A stored shape barely suppresses it at test.

## 2. Cohesion
The core holds together: unseen nouns copy because of their embeddings, familiarity removes copying, and useful bursts keep it. Results has eleven subsections, though, with several spokes off that core. Readers will get lost in four places:
- **The Setup paragraph on the three experiments.** Experiment 1's two flaws, the colour-expectation side effect and the window shares all come before any result.
- **Table 2.** Its footnotes carry the logic.
- **§4.5's crossed-history contrasts** (1.72 / 0.47 / 1.25), which are never spelled out.
- **§4.7.** "3.67 nats more than geometry predicts" sits next to "carried by the noun's embedding". "Geometry" means three crude untrainedness measures, but the text never says so.

## 3. ARR review
**Strengths**
- The design is clean: frequency, storability and arrangement are crossed per item within one model, and fixed-shape bursts control for update count.
- Registrations are staged honestly, the primary result replicates on new seeds and at 85.6M, and nulls are reported (H30, H33, Pythia).
- The same-noun versus other-noun control (0.08) is sharp: history gates *whether* the model copies, not *what* it copies.
- The nonce-probe caveat is robust: typical embedding 4.53 → −0.33, and with tied embeddings 3.96 → 0.02.

**Weaknesses: fatal**
- None, provided the anonymity issue below is fixed.

**Weaknesses: fixable in text**
1. **Anonymity.** The `results/amendment11/manifest_large.csv`, `manifest_tied.csv` and `scores_*.csv` files contain `/Users/chrismarmo/` paths, and the repository has a CITATION.cff. As "anonymised" supplementary material, that risks a desk reject.
2. **The memory argument overreaches** (see §4).
3. **Model count.** The abstract says 240 models. Table 3 sums to 220, plus 10 tied models makes 230.
4. **Novelty.** The paper doesn't separate itself sharply from Chan et al. 2022, Singh et al. 2023 (transience) and Anand et al. 2025.
5. **Scale.** The size average is dominated by 3.4M (6.36 versus 2.44 and 2.24). The 85.6M "rebound" is confounded with warm-up and hardware.
6. **Typical-embedding control.** In shifting corpora trained kinds copy about 10 nats, yet their mean embedding plus noise copies only 0.4–1.6 (Tables M2, N2). The "typical" embedding is therefore not behaviourally typical, which weakens the representation reading.

**Weaknesses: need new experiments**
- Memory competition is tested only on spaced nouns at floor (copy score under 2 nats).
- Nothing manipulates memory during training.
- One language and one recipe; copying fails with even a one-sentence gap.

**Questions**
1. In the steering analysis (Table O4), does stored preference stay high while copying rises by 1.51? If so, that is your best evidence against test-time competition.
2. What is $L(s^*)-L(s)$ for fixed-shape bursts?
3. At 1,000 exposures, no-shape bursts (1.05) and fixed-shape bursts (1.03) copy equally. Why isn't this offered as evidence that memory isn't what removes copying?

**Scores**
- Soundness: 3.5
- Excitement: 2.5
- Overall: 3 (Findings)
- Confidence: 4

## 4. Theoretical argument
- **The three accounts sit at different levels**, so they are not rivals. Memory is about test-time computation, representation is about where something is stored, and useful context is about the training signal.
- **Useful context is defined with memory inside it** ("memory could not"). That makes "the useful-context and memory accounts are thus complementary" partly true by definition.
- **Representation is a strawman** as stated ("whatever the word was seen with"). The steering result shows the embedding does carry the effect.
- **The copy score is blind to additive memory boosts.** It is a log-ratio among non-$s^*$ shapes, so a boost to $s^*$ cancels out. H52 can only detect memory suppressing the context pathway, so its "small" effect is partly built into the measure.
- **Burst versus spaced is sound for H50 and H51 taken jointly.** The weak link is extrapolating test-time suppression from spaced nouns at floor (0.10 out of 0.57) to burst nouns.

**Claims stronger than the evidence**
- "Memory competition is unlikely to explain this."
- "What decided copying was whether the context had been needed in training."
- "The effect of a storable shape appears to arise in training" (abstract). Training-time memory is never manipulated.
- "Bursts thus first preserve copying … and changing-shape bursts then give the embedding a copying tendency of its own." This is a temporal claim drawn from different nouns at two exposure levels.
- "a word's tendency to copy depends on the company it keeps". It rests on fixed bursts alone, and H59 was inconclusive for changing bursts.
- "The decline did not continue."

## 5. Writing
**Hardest sentences, with plain rewrites**
1. *"Where object kinds keep their shapes, nouns lost their copying of a shape from the preceding sentence with exposure, unless their own earlier sentences had predicted a shape that memory could not."*
   → "When kinds keep their shapes, nouns stop copying the previous sentence's shape as they become familiar, unless in training that sentence was the only way to predict their shape."
2. *"Memory mattered, but its effect appears to arise in training: where a shape could be stored, the context was not needed, and copying was lost."*
   → "A storable shape mattered mainly by making the context unnecessary during training; at test it barely reduced copying."
3. *"…the context about the queried noun added 0.08 nats [0.02, 0.14] over the control, within ±0.3 (pre-registered), and no other condition … exceeded 0.14."*
   → "Copying was nearly identical whether the context named the queried noun or another noun (0.08 nats, inside the registered ±0.3 bound)."

**Leftover jargon and bookkeeping**
- The hardware-fault story in Setup.
- "registered bound", "not shown to be within ±0.3", "this registered comparison is not informative".
- "(−1.02, pre-registered, and −1.00)", where the two experiments are unlabelled.
- "Above geometry".
- "most changes were not small enough to count as none".
- "Earlier experiments … not reported here".
- "(Experiment 3: 0.20)" asides.
- Old labels in the appendices (carry, row, rotating, runs/singles, Regular/Reassigned, atypical_init) and script paths.

## 6. Three text-only changes that would most raise Overall
1. **Recast the theory.** State one training-signal hypothesis, tested against two nulls: untrainedness and test-time suppression. Say outright that the copy score cannot see additive memory boosts. Soften the "memory" claims quoted in §4, and add the no-shape ≈ fixed bursts comparison and the tied-embedding H52 null (−0.01) as converging evidence.
2. **Restructure Results into four parts:**
   - unseen nouns;
   - familiarity and bursts;
   - what is learned (a gate rather than a binding, spillover, narrowness);
   - generalisation (size, tied embeddings, Pythia).

   Report only Experiment 3 numbers in the prose, move the registration bookkeeping to the appendix, and fix the model count.
3. **Lead with the genuinely new results**: per-word gating without binding, and spillover across nouns that depends on corpus company. Position them against transience (Singh et al.) and embedding resets (Anand et al.). Turn the nonce-probe caveat and the Pythia null into advice for evaluation practice.
