# Cold review, v9.9 (ARR / ACL 2027), area-chair read

Paper: `paper_acl/main.pdf`, 24 pages, built 29 Oct 2026. Title: "Whether Small Transformers Copy a Feature from Context Depends on the Noun's Training History".
Read as a reviewer first (PDF text and rendered pages), then checked against `results/` and `docs/AMENDMENT*.md`.
Page numbers are PDF pages. "l." is the ACL review line number.

---

## 1. Summary

The paper trains small decoder-only transformers (3.4M to 85.6M parameters) on an artificial language and inserts 40 new nouns whose training history is controlled: how many sentences, whether the noun's shape is fixed, changing or absent, and whether its sentences come singly or in bursts of four. It measures how strongly the model copies a shape word from the previous sentence into the next mention of the noun (the "copy score"). The claim is that copying is high for unseen nouns because of their untrained embedding, is lost with exposure, and is retained only for nouns whose earlier sentences were the only clue to their current shape (changing-shape bursts: 3.68 nats more than fixed-shape bursts at 1,000 exposures in a pre-registered replication). Exploratory analyses say the retained copying is a generic "copy the last shape" tendency gated by the queried noun's input embedding, partly along a direction shared across nouns, read out in the first two blocks.

---

## 2. Narrative

**Does it read as one paper?** Mostly, up to the end of Section 4.3. The core is clean: one manipulation, one measure, one registered replication. After that it reads as three papers stapled together.

**Seams.**

1. **Three experiments, one of which is the paper.** Experiment 3 carries every headline number. Experiment 1 gives the baseline and has no main-text figure or table (only "Appendix D, Figure D1"). Experiment 2 appears as a shadow column in Table 4 and as Figure 2c. The reader has to hold "Exp. 2 value / Exp. 3 value / pooled / 1,000 alone" for each number. The sentence "Where Experiment 3 tested a finding of Experiment 2, we report the Experiment 3 value; Table 4 gives both" is printed twice (p4, l.258–260 and l.263–265), which shows the join.

2. **The three "accounts" are set up in the Introduction and then dropped.** Table 3 (p4) hedges two of the three accounts with footnotes ("Or >0 if an embedding left less trained copies more"; "No larger than for spaced nouns"), so they do not make clean, different predictions. The Discussion never says which account lost. Worse, the paper's own answer ("whether the model copies ... depends on the noun's embedding", p6 l.451–453) is a representation account. The accounts sit at different levels: "useful context" is about what training rewards, "representation" is about where the result is stored. They are not rivals.

3. **Section 4.4 changes genre.** It moves from registered behavioural tests to exploratory mechanism (steering, patching, attention), all in one dense paragraph with eight numbers (p5–6, l.404–433). The paragraph headed "It is carried by the embedding" opens with "A noun enters the model only through its embedding", which makes the heading true by construction (see weakness A1).

4. **Section 4.5 (shifting-kind corpora) is seven lines** (p6 l.441–449), exploratory, from Experiment 2 only, and contradicts the section title of 4.2 ("Familiar nouns stop copying unless the context was needed"): there, fixed-shape bursts keep 3.57 nats. It is an important boundary condition presented as an afterthought.

5. **The Discussion spends a third of its length on things the paper does not show.** Pythia (a registered null), WikiText-103 counts for 44 nouns, and infants. None of these supports a claim. The Pythia test in particular asks a different question (does replacing a real noun's embedding change colour copying) from the paper's question (does training history decide copying).

6. **The appendix tells a second story.** Appendix B and C (p11–12) disclose an earlier study whose models were "meant to be tied [but] had been trained untied because the tie was lost on the graphics hardware", "a loading error [that] had changed their scores", and registrations made "in stages, some after related results were known". Honest, but a cold reader meets a history of eleven amendments and pipeline faults before any reassurance.

**Cut.** The Pythia test (move to one sentence plus appendix, or drop). The WikiText paragraph. The children paragraph (keep one sentence in Related Work). The head-ablation material from Experiment 1 (H33–H35, Table D4, Figure D2); it found nothing and the paper says so. Table 3 in its current form.

**Move.** Bring a compact Experiment 1 panel into Figure 2 (it is the baseline the title depends on). Promote shifting-kind corpora and tied embeddings into a short "Boundary conditions" subsection with their own numbers by size; both limit the headline claim and both are currently buried. Put the per-size numbers (Table F2) in the main text; the pooled mean hides a threefold size effect.

**Add.** A plain statement of what the model learned to do: "after a noun with this kind of embedding, repeat the shape word of the immediately preceding sentence in the same frame". A sanity table of training loss / held-out loss by size. One paragraph saying which account is rejected and why.

---

## 3. ARR review

### Strengths

1. Clear, causal design. The four histories (Figure 1b) separate frequency, storability and arrangement within one model. This is the right way to ask the question.
2. A real confirmatory replication. Experiment 3 was registered after Experiment 2 and before training; the headline tests replicate on new seeds, a larger model and single-history corpora. I checked H50–H59 against `results/amendment11/RESULTS.md`; every value matches.
3. Honest reporting. Failed predictions (H30, H33, H52 not equivalent, H59 inconclusive, all Pythia tests) are reported. The Limitations section is unusually frank.
4. A useful measurement point: the copy score excludes the noun's own shape, and the paper separately reports the output preference L(s*) − L(s) (5.84 nats), so "suppression" and "being outvoted" are not confused.
5. The embedding-control result is a practical warning: unseen nouns score 6.88 with their own embedding and 0.79 with a typical one (Exp. 3). Nonce-word probes in from-scratch models can measure the initialisation.
6. Numbers are reproducible. Every main-text number I recomputed from the CSVs matched (Section 4 below lists the few exceptions).

### Weaknesses: fixable in text

**A1. "It is carried by the embedding" is true by construction (p5 l.404–405; abstract l.026).** All other weights are shared across nouns. A per-noun behavioural difference at a fixed test item can only live in the noun's input embedding. The embedding swap in Table E4 ("moved the copying with the embedding (2.88 nats ...)") therefore just re-measures the condition difference (3.85 − 0.99 = 2.86). Say so, and present only the non-trivial part: the direction is shared across nouns and transfers.

**A2. The steering result is oversold.** Abstract: "carried by the noun's input embedding, along a direction shared across nouns". The direction moves fixed-shape spaced nouns by 1.51 nats, against a 3.68–4.14 nat condition difference. By size it is 2.41, 1.12, 1.00. Subtracting it from changing-shape bursts gives −0.82 [−1.37, −0.28] against −0.30 [−0.59, −0.01] for a random vector, and at 25.6M the effect is +0.02 (`results/amendment11/MECHANISM.md`). So a shared direction carries roughly a third of the effect, mainly in the smallest model. The abstract states this exploratory result without the word "exploratory".

**A3. "Faded", "survived", "kept", "lost" describe a time course that was not measured.** Abstract l.011–013: "a familiar noun's copying ... faded with training. It survived for nouns ...". The evidence is a comparison between different nouns with 100 and 1,000 exposures at the end of training. The only time-course data (Figure E5, `scores_formation_a.csv`) show copying for changing-shape bursts being built, not kept: at 25.6M it rises from 0.79 to 3.29, at 10.9M from 1.13 to 3.32. The lines joining 100 and 1,000 in Figure 2 reinforce the wrong reading. Use "nouns with more exposures copied less" and "acquired".

**A4. The headline number is a mean over very different sizes.** 3.68 = mean of 6.36, 2.44, 2.24. "Familiar nouns stop copying" is also mainly a 3.4M result: at 25.6M the other histories keep 1.3–1.7 nats at 1,000 exposures (Table F2: no-shape spaced 1.67, no-shape bursts 1.54, fixed bursts 1.32) against 3.56 for changing bursts. Report by size in the main text and drop "near zero" language for the larger models.

**A5. "Size" is not a clean factor.** Sizes differ in steps, batch, learning rate, warm-up and amount of text (160,000 vs 256,000 windows; Table B1, Appendix A). The 85.6M models also differ in hardware and PyTorch build. The paper says "a difference present at every size up to 85.6M parameters" (abstract); it should not invite a scaling reading. State plainly that the non-monotonic pattern (6.36, 2.44, 2.24, 4.18) cannot be interpreted.

**A6. Tied embeddings undercut the title claim and are buried.** With tied embeddings (3.4M), every history keeps copying at 1,000 exposures: 1.4 to 2.9 nats (fixed bursts 2.39), and the key contrast falls from 6.36 to 1.90. So "familiar nouns stop copying unless the context was needed" does not hold with tied embeddings. This is in Limitations and one clause on p4–5. It belongs in the abstract or the section title should be weakened.

**A7. "Spreads to other nouns" rests on one between-corpus contrast and selective reporting (p5 l.389–403).** Evidence: fixed-shape bursts 1.03 (mixed) vs 0.03 (fixed-only), five seeds; unseen nouns 7.35 vs 4.87. Not mentioned: unseen nouns in changing-only corpora also copy less than in mixed (6.16 vs 7.35), which does not fit "spreading from changing-shape nouns". The changing-shape arm of H59 is inconclusive ([−1.44, 1.92]). And a shared attention circuit strengthened by some nouns is "shared weights", not "spread between nouns". Reword and report all three unseen values.

**A8. "Geometry accounted for burst nouns' copying" at 100 exposures depends on the analysis (p5–6 l.409–411).** Per-model regression: 0.45 [−0.25, 1.15] (matches). The pooled noun-level regression that the paper reports for 1,000 exposures gives 1.24 [0.83, 1.65] at 100 (my run of `geometry_test(rows, e=100)` on `results/amendment11/reanalysis.csv`). A non-significant estimate is not evidence that geometry "accounted for" it. Also the indicator is for changing-shape bursts, not "burst nouns".

**A9. H55 ("beyond geometry") adds almost nothing.** 3.67 beyond geometry against a raw difference of 3.68. Three scalars (cosine to init, distance to mean, norm) fitted on 18 nouns per model with 3 nouns carrying the indicator cannot capture a direction. Say that the test rules out these three summaries, nothing more.

**A10. The copying is very narrow; the title and abstract do not say so.** One intervening sentence removes it (≤0.44 nats). Another frame keeps 35–40%. It is not tied to the noun the context described (0.08 nats). So the behaviour is "repeat the shape word of the adjacent same-frame sentence when the query noun's embedding is of the copy type". "Copy a feature from context" in the title promises more. Also the test item is in-distribution for burst nouns and out-of-distribution for spaced nouns, which never recur within a window in training; say this where spaced and burst nouns are compared (H51).

**A11. No basic model-quality numbers.** No training or held-out loss, no accuracy on the trained kinds, by size. A reader cannot tell whether the 3.4M model has learned the language to the same degree as the 25.6M model.

**A12. Novelty relative to Chan et al. (2022a) and Singh et al. (2023) is thin as stated.** "We find the same rule acting on single words within one model" (p7 l.474). A reviewer will say: gradient descent keeps a behaviour where it lowers the loss and not otherwise; that this holds per token follows from per-token embeddings. The surprising parts are (i) the test-time suppression is tiny (0.25) while the training-time effect is large, (ii) the tendency is not noun–shape binding, (iii) the typical-embedding control. Lead with those.

**A13. Pre-registration history needs a cleaner statement.** Eleven amendments; "Earlier experiments in this language, not reported here" (p7 l.553–555); a loading error; an untied-by-accident study. State in one place: how many earlier studies, that none of their data enters this paper, and that the three experiments here are the complete set run with this design.

### Weaknesses: need new analysis or experiments

**B1. The missing control: bursts in which the shape changes within the burst.** Fixed-shape bursts control for recurrence but add a stored shape and make every test item a conflict item (context names s ≠ s*, never seen in training). No-shape bursts store nothing but expect a colour ("confounded with word class", the paper's own words). So no condition has recurrence + a shape slot + nothing storable + an uninformative context. A burst of four sentences each with a freshly drawn shape is that condition. Without it, "useful context" is not separated from "recurrence without a competing stored shape".

**B2. No dose–response.** "Useful" is binary. Varying how often the burst repeats its shape (e.g. 25/50/75/100%) would turn a two-cell contrast into a law and is cheap in this setup.

**B3. No evidence outside the toy language.** The only pretrained test is a null on a different question. The natural experiment is easy to state: add new tokens to a pretrained model (Pythia-160M to 1B), fine-tune on natural-ish sentences under the four histories, and measure the same copy score. Without something like this the Excitement score stays low at ACL.

**B4. Mechanism stops short of a causal test.** Attention is correlational (ρ = 0.80 across 12 cells). Patching shows only that information leaves the noun position by block 2, which is what any attention-based readout would show. The registered head search in Experiment 1 found none. Needed: ablate or patch first-block attention from the prediction position to the context shape word, per head, and show the changing-vs-fixed difference goes away. Table D4 also has an unexplained oddity: at 10.9M, ablating the "candidate" heads raises unseen copying from 2.98 to 5.80.

**B5. Tied embeddings at one size only.** Given A6, tied models at 10.9M and 25.6M are needed before the claim can be stated generally. Many deployed models tie.

**B6. Noun and item variance are ignored.** Three nouns per cell, 36 items each, analysed as seed means. A mixed model with noun and item effects would show whether the effect generalises over nouns, and would use the data better than n = 10 seed means. Single-history corpora rest on five seeds; H59 (changing) has a CI of ±1.7 nats against a ±0.3 bound.

**B7. Congruent items for fixed-shape nouns.** The copy score for fixed-shape nouns is only defined on conflict items. The `agree` column exists in `scores.csv` but is not reported. Does a fixed-shape burst noun's L(s*) rise when the context names s*? That would show whether the copying route is absent or merely outvoted for those nouns.

**B8. Exposure is confounded with training time and density** (acknowledged). A noun with 1,000 exposures concentrated in the last 10% of training, or checkpoints matched on exposures, would separate them.

### Questions for the authors

1. In changing-shape bursts, what does the model do at the first sentence of a burst after a context about another noun? Is the copy gated by the noun, or by the frame repeating?
2. Why does the effect rise again at 85.6M (4.18) after falling to 2.24 at 25.6M? Is it the warm-up, the hardware, or size?
3. With tied embeddings, why do all histories keep 1.4–2.9 nats? Does the output role of the embedding hold it away from the trained kinds' region?
4. Unseen nouns' copying itself grows over training (Figure E5, stable kinds, from about 1.5 to 6). So is unseen-noun copying "because their embeddings are untrained", or because the model builds a copy circuit that untrained embeddings fail to switch off?
5. In shifting-kind corpora the typical embedding still gives 0.43–1.57 while trained kinds copy at 9.98. What is "typical" about it?
6. How were the 0.3-nat equivalence bounds justified for Experiment 3, given H52 came out at −0.25 and "not equivalent"?
7. Were any Experiment 3 analyses run and not reported? The registration lists the tests; please confirm the list in Table C2 is complete.
8. The 40 new nouns outnumber the 32 trained kinds, and in Experiment 1 they fill 60% of windows. How does that shape what "typical" and "trained" mean?

### Scores

- **Soundness: 3.5.** The registered behavioural claims are well supported and reproducible; the mechanism claims, "spreads", and the time-course wording go beyond the evidence, and one key control (B1) is missing.
- **Excitement: 2.5.** Careful work, but a toy language, small models, a narrow behaviour, a null in pretrained models, and a main result many will find expected.
- **Overall: 3 (Findings), leaning 2.5.** Solid enough for Findings if the over-claims are trimmed; not main conference without B1 and some evidence beyond the toy setting.
- **Reviewer confidence: 4.** I read the whole paper and appendices and recomputed the main numbers from the released CSVs; I did not re-run training or verify every citation.

---

## 4. Errors

### Number check against results files (what was verified)

Matched exactly (paper vs file): abstract 3.7 / 3.68 [2.99, 4.37]; by size 6.36, 2.44, 2.24; H57 4.18 [2.95, 5.41]; H51 4.14 [3.60, 4.69]; H52 −0.25 [−0.35, −0.15], TOST p .14; H53 5.84 [5.57, 6.11]; H54 0.08 [0.02, 0.14]; H55 3.67 [3.03, 4.32]; H56 1.25 [0.94, 1.56] (1.72 vs 0.47); H58 5.51 [4.24, 6.78]; H59 0.24 [−1.44, 1.92] and −1.00 [−1.20, −0.80]; 3.58 [2.94, 4.22]; 0.33 [0.18, 0.48]; all of Table 4; Table F2; steering 1.51 [1.20, 1.82], 2.41, 1.00, 1.99; patching 5–58% and 47–96%; attention 0.80 [0.76, 0.85], 0.91/0.79/0.72, 0.069 [0.048, 0.089]; all tied-embedding numbers (3.96 [2.43, 5.49], 0.02, 1.90 [1.20, 2.60], 2.36, −0.01, 4.61, 4.29, 1.4–2.9); Exp. 1 4.53 [3.54, 5.52], −0.33, Table D1–D2; Exp. 2 Table E1, E2, E4; Pythia Table E7; WikiText counts. Sources: `results/amendment11/{RESULTS.md, SUMMARY.md, MECHANISM.md, scores.csv, scores_tied.csv, scores_large.csv, reanalysis.csv}`, `results/amendment10/{RESULTS_A.md, REANALYSIS.md, CROSSED.md, RESULTS_B.md, RESULTS_C.md, scores_formation_a.csv}`, `results/amendment9/{RESULTS.md, scores.csv}`, `results/natural_regime.csv`.

### Numbered problems

**Claim / number mismatches**

1. p5 l.333–334: "Fixed-shape bursts likewise added only about 0.5 nats at 1,000." Experiment 3 value is 0.57 [0.39, 0.74] (`results/amendment11/SUMMARY.md`, "Fixed: bursts - spaced ... 1,000 0.57"). 0.47 is the Experiment 2 value. The paragraph otherwise reports Experiment 3. Write "about 0.6".
2. p18, Table E4 caption: "(2.88 nats [2.01, 3.76] in each direction)". The two directions are 2.88 [2.01, 3.76] and 2.88 [1.99, 3.76] (`results/amendment10/REANALYSIS.md`). Trivial, but "each" is not exact.
3. p14, Table D1, H35: CI "[0.000, 0.005]". File: "[-0.000, 0.005]" (`results/amendment9/RESULTS.md`). Sign of the lower bound lost in rounding; the verdict depends on it crossing zero.
4. p7 l.463–465: "At test a stored shape lowered copying by 0 to 0.4 nats across three experiments". In Experiment 1 fixed-shape nouns scored higher than changing-shape nouns at 10 and 100 exposures (3.70 vs 3.57; 0.30 vs 0.16; Table D2), so the range runs from about −0.14 to 0.40.
5. p4 l.289–290: "near zero from 1,000 on, whatever their history". Fixed-shape nouns are at −0.30 and −0.37 (Table D2); that is the largest stored-shape effect in the paper and is called zero here.
6. p22, Table F1 footnote: "By size: 6.36, 2.47, 2.19." These are H55 by size (`RESULTS.json`), but they sit after three H55 variants with no label, and H50 by size is "6.36, 2.44 and 2.24" (p4). Readers will see a mismatch. Label it "H55 by size".

**Claims not supported by the cited evidence**

7. Abstract l.011–013: "a familiar noun's copying of a shape from the previous sentence faded with training." Evidence is across nouns with different exposure counts at the end of training, not over training. Figure E5 shows copying rising over training for changing-shape bursts and for unseen nouns.
8. p5 l.404 heading "It is carried by the embedding." and p5 l.404–405 "A noun enters the model only through its embedding." The second sentence makes the first a tautology (see A1).
9. p5–6 l.409–411: "At 100 exposures this geometry accounted for burst nouns' copying (0.45 [−0.25, 1.15] beyond it)". Non-significance read as support; pooled noun-level estimate at 100 is 1.24 [0.83, 1.65]; and the test concerns changing-shape bursts only, not "burst nouns".
10. p6 l.426–430: "after the second block the noun's own position no longer mattered and the effect had moved to the position where the shape is predicted." True in the patching table, but it does not show the effect is "read early" in any specific sense; it is what any attention readout does. After the first block the two fractions sum to more than 1 in one case (0.32 + 0.96; `MECHANISM.md`, R_into_F, medium), so "carried x% of the difference" is not additive.
11. Abstract l.027–028 and p5 l.389: "spread to other nouns trained alongside" / "It spreads between nouns." Rests on fixed-only vs mixed (five seeds); the changing-only unseen value (6.16 vs 7.35 mixed, Table F2) is not mentioned and does not fit.
12. p4 l.296–298: "In random-order corpora no noun copied, so copying is learned from the runs of sentences about a kind." In stable-kind corpora the trained kinds themselves do not copy (Table D2: "Trained −0.02"), so "learned from the runs" needs an explanation of what in the runs is learned.
13. p7 l.490–494: "such a probe can measure the untrained embedding more than anything learned about context, so effects should be checked against a typical embedding." The next sentence reports a null in Pythia. The recommendation is not supported beyond the toy models, and the paper's own limitation says the typical embedding "may be atypical in its own way".
14. p6 l.447–449: "When the corpus rewards copying, recurrence alone sustains it." In shifting-kind corpora no-shape bursts fall to 2.09 and fixed bursts to 3.57 from 100 to 1,000 exposures (Table E2: 3.79 → 2.09; 5.35 → 3.57), while changing bursts rise. "Sustains" overstates; usefulness still matters.
15. p4 l.310–313: "This copying did not fade over training: in Experiment 2 it rose from 0.8–2.6 nats at the first saved weights to 3.85 at the end". Numbers match, but the endpoints mix sizes (0.8 is 25.6M, 2.6 is 3.4M; 3.85 is the size-averaged final value) and steps differ by size.

**Undefined or late-defined terms**

16. p1–2 l.094–095: "What useful bursts keep". "Useful bursts" is never defined.
17. p5 l.377: "no other condition in mixed or fixed-only corpora". "Mixed" corpora are not defined in the main text; "fixed-only" and "changing-only" appear only in the Table 2 caption as "all familiar nouns have changing shapes, or all fixed shapes".
18. p5 Table 4: "Beyond geometry" is used as a row label before the geometry analysis is introduced (p5 l.404).
19. p3 l.222–223: "(for nouns without a shape, one fixed shape is left out instead)". Which shape, and why, is not stated.
20. p13 Table C2 and p14 Table D1: "candidate heads" is defined only in the Table D4 caption (p15).
21. p19 Table E6 and text: "ownnorm", "randnorm", "shrunk" are script names used as column heads.
22. p4 Table 3, first row "Unseen: typical − untrained ... –": the "–" under Exp. is not explained (caption explains only "Blank"). It also shows that the typical-embedding contrast, a headline result, was never a registered test.

**Inconsistent terminology**

23. "History" is used three ways: Table 1 (p3) lists four histories without presentation; Figure 1b (p2) is titled "Four training histories of one noun" and includes presentation; p6 l.444 "every burst history"; Figure 2 caption separates "history (colour) and presentation".
24. "Trained kinds" vs "trained nouns": p4 l.283 "trained nouns' embeddings", p6 l.455 "the trained nouns'", elsewhere "trained kinds". Appendix A also uses "base kinds" (Table D2 caption).
25. "Embedding analysis" (main text, Table C1) vs "Row analysis" (Table C2 header "Row analysis (Amendment 10, Part C)" and caption "E6 (row analysis)") vs "Reanalysis" (p17).
26. "Random, untrained scale" (Figure E4 legend) vs "random (untrained norm)" (Table E7, E8) vs "Random embedding of untrained scale" (Table C1).
27. "block" vs "layer": p6 l.428 "after the second block", l.431 "first-layer attention"; Table D4 "layer 0".
28. "Exposures" vs "sentences": abstract "after 1,000 sentences"; elsewhere "1,000 exposures". "queried noun" (p5) vs "query noun" (p22).
29. Stored knowledge is called "stored shape", "stored preference", "memory", "knowledge" (Table C1) and "own-shape preference" (Table D3); two of these are different measures and the text does not keep them apart (p3 l.230–233 vs p14 l.919–921).
30. p16 Figure D2 axis labels use registration names: "Regular unseen nouns", "Reassigned unseen nouns", "Reassigned trained kinds", "Grouped models". Main-text terms are stable-kind / shifting-kind.
31. Section 4.2 title "Familiar nouns stop copying unless the context was needed" vs Section 4.5 and the tied-embedding result, where they do not stop.

**Cross-references, figures, tables**

32. Appendix figure numbering is wrong: Figures D1, D2, then E3, E4, E5 (p16, p20). The figure counter is not reset in Appendix E (`appendix_a10.tex` resets only the table counter). Tables restart correctly (E1...). Main-text references "Figure E3", "Figure E4", "Figure E5" resolve, but the numbers should be E1–E3.
33. p4 l.258–260 and l.263–265: the same sentence twice ("Where Experiment 3 tested ... Table 4 gives ...").
34. p2 Figure 1: the notes under panel (a) and (b) ("zibby: shape; lavo: colour; shorfi: texture." etc.) are about 5 pt at print size and overlap the box edge; the figure uses a different font (DejaVu Serif) from the text.
35. p6 Figure 2: lines join 100 and 1,000 exposures, which are different nouns; this reads as a time course. Panel (b) shows two "unseen" lines with no bands and no typical-embedding line, unlike (a) and (c). In (c) the "unseen" label and band sit on top of the 100-exposure points.
36. p20 Figure E5: the "Unseen" line zig-zags by 1–2 nats because sizes save weights at different fractions; the caption calls these "small zig-zags". Plot by size or interpolate.
37. p5 Table 4: no intervals, although the text gives them; the "r" marks are on different columns in different rows without a rule the reader can infer (e.g. "0.80r" pooled in Exp. 2, "0.08r" at 1,000 in Exp. 3).
38. p3 Table 2: Exp. 3 "Models 70" omits the ten tied-embedding models that the Results section reports; Exp. 3 row says "as Exp. 2" but Exp. 3 has no shifting-kind corpora (stated only in the caption).
39. p15 Table D4: stable-kind 10.9M, unseen copy score "Intact 2.98, Cand. 5.80". Ablating candidates nearly doubles copying; not discussed.
40. p5 l.383: "Whose history matters?" is a run-in question inside a paragraph with a different bold lead ("A tendency, mostly not a link."); it looks like a lost paragraph heading.

**Anonymity**

41. p1 l.072–073 and p8 l.600–603: "(Kim et al., 2022; Cacioli, 2026)" / "J.-P. Cacioli. 2026. Repetition without exclusivity: Scale sensitivity of referential mechanisms in child-scale language models. In Proceedings of the 2nd BabyLM Workshop. To appear." A single-author, not-yet-published workshop paper, cited for a side point, in a submission that also says "an earlier study in this language". Third-person form is allowed, but this is the easiest de-anonymisation path. Cut it or replace it with a published source.
42. p11 l.875–876 and p21 l.989–990: "one AMD graphics card (DirectML)", "an Apple M3 Ultra", "(MPS, PyTorch 2.14.1)". Not names, but an unusual hardware pair; fine to keep, noted for completeness.
43. Supplementary material (the paper says "anonymised copies are provided"). The repo's result files contain a user name in the checkpoint path column: `results/amendment11/scores_tied.csv`, `scores_large.csv`, `scores_all.csv`, `manifest_large.csv`, `manifest_tied.csv`, `model_check_large.csv`, e.g. "/Users/chrismarmo/ckpt_a11_tied/regular_grouped/tiny_seed2029/model_final.pt". Other CSVs contain "E:\AI Checkpoints\overhypothesis_a11\...". `docs/AMENDMENT11_DEVIATIONS.md` and `docs/AMENDMENT11.md` contain live OSF IDs ("osf.io/8ws5c, within project osf.io/qj9hb"). `README.md`, `CITATION.cff`, `LICENSE`, `REPRODUCE.md`, `docs/HF_MODEL_CARD.md`, `scripts/aggregate_results.py`, `paper_latex/` and `paper_acl/refs.bib` contain the author name. `results/amendment11/tied_check.txt` says "run on the Mac 29 Oct 2026". The appendix points reviewers to "results/amendment11/MECHANISM.md" and "results/natural_regime.csv", so they will open this tree. The `ckpt` column must be stripped and the docs scrubbed before upload.
44. PDF metadata is clean (no Author/Title fields). No URL or local path leaks in the PDF text itself.

**Citations**

45. Reference list mixes two styles: initials ("S. Biderman, H. Schoelkopf, ...", "A. Bietti, V. Cabannes, ...") and full first names ("Ekin Akyürek, Bailin Wang, ..."). The same person appears as "S. C. Y. Chan" (2022a) and "Stephanie C. Y. Chan" (2022b).
46. p9: "C. Olsson ... et al. 2022. In-context learning and induction heads. ArXiv:2209.11895." Truncated author list with "et al." in the bibliography; "ArXiv:" capitalised here and in Raventós et al. and Xie et al., but "arXiv preprint arXiv:" in Chan et al. 2022b and Kim et al. 2022. Raventós et al. and Xie et al. give both a venue and an arXiv number.
47. p9: "Minsung Kim ... 2026. ... In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 23242–23257." I could not verify this entry; check title, venue and pages against the Anthology.
48. Several entries lack pages or publisher detail while others have DOIs (Kandpal et al. 2023; Yu et al. 2023; Jin et al. 2024; Mallen et al. 2023; Anand et al. 2025; Chan et al. 2025; Reddy 2024). Vlach et al. 2008 lacks an issue number.
49. p2 l.108–110: "favours rare classes and can fade as stored solutions take over (Chan et al., 2025; Singh et al., 2023)". Two claims, two citations, not matched one to one; say which paper shows which.
50. p7 l.503–505: WikiText-103 is cited as "(Merity et al., 2017)", the same key used on p2 for pointer models. Correct paper, but the reader will not know the dataset comes from it; say "introduced in".

---

## 5. The three changes that would most raise the score

1. **Run the missing control and a dose–response (B1, B2).** Add bursts whose shape is redrawn within the burst, and bursts whose shape repeats with probability 0.25/0.5/0.75. Same corpora builder, same sizes, ten seeds, registered. This turns "changing bursts vs fixed bursts" into a direct test of usefulness with recurrence, slot and storability held fixed, and it answers the strongest objection a reviewer can raise. It lifts Soundness to 4.

2. **Rewrite the claims to what was shown, and reorganise around one experiment.** Make Experiment 3 the paper; give Experiments 1 and 2 one paragraph each as the path to it. Report by size in the main text. Replace "faded / survived / kept" with cross-sectional wording. Drop "carried by the embedding" as a finding; say "a shared direction carries about a third of the effect, mostly at 3.4M (exploratory)". State the tied-embedding and shifting-kind limits in the abstract. Reword "spreads". Cut Pythia, WikiText and the infant analogy to a few sentences. Fix the figure numbering, the duplicated sentence and the terminology. This costs no compute and removes most of the reasons to reject.

3. **Show it once outside the toy language (B3), or failing that, do the causal mechanism test and tied embeddings at all sizes (B4, B5).** New tokens fine-tuned into a small pretrained model under the four histories would move Excitement from 2.5 to 3.5. If that is out of reach before the deadline, a per-head attention-patching result that removes the changing-vs-fixed difference, plus tied models at 10.9M and 25.6M, would at least make the mechanism and scope claims solid.
