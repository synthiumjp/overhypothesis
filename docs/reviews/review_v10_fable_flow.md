# Review v10: narrative flow, logic and internal consistency

Paper: `paper_acl/main.pdf` (26 pages), "Whether Small Transformers Copy a Feature from Context Depends on the Noun's Training History".
Read cold from the PDF. Source used only to locate text. Numbers cross-checked against the appendix tables and `results/amendment10`, `results/amendment11`, `results/control_redrawn` summaries.
Page numbers are PDF pages. Line numbers are the ACL review-line numbers.

---

## A. Verdict on flow and logic

The paper has a clear spine and it is far from the rejected manuscript. Three accounts are stated in the Introduction, Table 3 turns them into predictions, and Sections 4.1 to 4.3 answer the questions in order, each ending on the question the next one takes up. Almost every number I checked matches its table. The weak points are of a different kind. (1) The prose is compressed to the point of being cryptic in the places that carry the argument: the abstract, the last Introduction paragraph, Section 4.4 and the Discussion pack one result per clause and use phrases the reader cannot decode on first contact ("outweighed it at the output", "what training kept", "a link between noun and shape"). (2) The headline inference "memory acts through training, not at prediction" is hedged in Section 4.3 ("this is an inference") but stated flat in the abstract, Introduction and Discussion, and its direct test sits on spaced nouns that copy almost nothing. (3) The alternative that the control kills, "recurrence", is never one of the three accounts and has no column in Table 3, so the control answers a question the Introduction did not pose. (4) Several results first appear in the Discussion or Limitations (the Pythia test, the WikiText counts, the head search, weaker storage in bursts). (5) There is a cluster of real consistency errors: "control" means two things, Figure 2's caption does not describe panel (b), one Experiment 3 value is attributed to Experiment 2, two main-text numbers have no source in the appendix, and Section 4.5's conclusion contradicts Section 4.3. (6) The implications paragraph is thin and its strongest claim (nonce-word probes) does not transfer to how such probes are run in pretrained models. None of this needs new experiments. It needs about a page of rewriting and a dozen corrections.

---

## B. The CogSys checklist

### 1. Not written like a scientific manuscript; lab notebook; cryptic or over-succinct — **Partly**

The structure is a paper. The sentence-level style still slips into notebook compression.

Evidence:
- Abstract, p1: "copied 3.7 nats more after 1,000 sentences than nouns in fixed-shape bursts (6.4, 2.4 and 2.2 at 3.4M, 10.9M and 25.6M parameters; 4.2 at 85.6M; 1.9 with tied embeddings)". Seven numbers in one sentence, before the reader knows what a nat of copy score is.
- Abstract: "A stored shape barely suppressed the copied one but outweighed it at the output; its larger effect was on what training kept." Three undefined ideas in one line.
- Section 3, p3, l.212–227: the "Three experiments" paragraph reads as a project log: "Experiment 2 removed both differences and added bursts, at 100 and 1,000 exposures, where Experiment 1's copying had nearly vanished. Its no-shape nouns, now in the test frame without a shape, learned to expect a colour after is a (by 1 to 5 nats), so comparisons with them are confounded with word class."
- Section 4.4, p7, l.490–514: five analyses in 25 lines, one clause each.
- Discussion, p8, l.566: "per-word embeddings make this possible but do not require it." No reader can unpack this.
- Appendix C, p13: "so we chose untied embeddings for this paper and check every model and that a loading error had changed their scores; both were corrected". Not a sentence.
- Appendix D, p15: the addendum list ("36 context draws per noun; uniform frame proportions; ... the gate for H35; and that heads without prefix matching are described as copying heads") is a notebook entry.

Fix: cut the size breakdown from the abstract (keep 3.7 and the range). Give each of the three "what this adds" results one full sentence of plain explanation where it first appears. Move the Experiment 2 motivation out of Setup into a two-sentence bridge at the start of 4.2. Rewrite Appendix C's sentence and D's addendum as prose.

### 2. Research questions / hypotheses in the Introduction; conditions motivated — **Passes, with one gap**

The question is stated (p1, l.048–051: "what in a word's own training history decides whether a model copies its features from the context?"). Three accounts follow with predictions. Table 3 maps accounts to contrasts. Conditions are motivated in "Predictions" (p4).

The gap: recurrence. The control exists to separate "usefulness from recurrence", and the Discussion says "the control rules out the alternative that recurrence alone keeps copying". But recurrence is not one of the three accounts. It first appears, undefined, at l.087 ("A control then separated recurrence from usefulness"). In Setup it is described only as "reasons unrelated to usefulness: the embedding is updated in fewer windows, and a test item ... resembles what burst nouns met in training". Table 3 has no column for it. Half-kept bursts have no prediction anywhere. No-shape nouns are in every experiment but appear in no row of Table 3, so a reader cannot see why they were run.

Fix: add a fourth bullet to the Introduction, "Recurrence. A noun that recurs within a window keeps copying because its test item looks like its training, whatever the earlier sentence said." Add a "Recur." column to Table 3 (bursts − spaced: >0; changing − fixed bursts: 0; changing − redrawn: 0). Add a row for no-shape bursts − spaced, and one line for half-kept ("between redrawn and changing under useful context").

### 3. Cryptic labels or notation in the main text — **Passes in the main text; Partly in tables and appendices**

No H-numbers or amendment names in the main text. Good. Remaining cryptic items:
- Table 2: "70 + 10t" and a row labelled "C".
- Table 3: superscripts b and c carry the logic of two cells; a blank and "–" mean different things.
- Table 4: "Beyond geometry", "Same noun − other noun", superscripts r, a, c.
- Figure 2: "typical emb."
- Appendices: H30–H59, M1–M3C, K1, R1, "Amendment 9/10/11, Part A/B/C", "the familiarity study", "grouped models", "pure_changing", "atypical_init", "frame 7", "DLA", "gate". Appendix E uses "H48" in running text (p18) with no gloss; Table E7's caption is written in H-labels only.

Fix: in appendices, lead each table row with the plain prediction and put the label in a narrow last column, or drop labels from running text. Rename "Beyond geometry" to "Changing bursts: observed − predicted from embedding geometry". Replace "C" with "Control".

### 4. Many backward/forward cross-references; not readable in sequence — **Partly**

The main text reads in order. But many results cannot be checked without jumping to an appendix that belongs to a different experiment:
- 4.1 (Experiment 1) cites "Experiment 3: 6.88 and 0.79", tied embeddings (Appendix F) and "Figure E3" (Experiment 2). The embedding manipulation numbers (4.00, 4.09, −0.44) are in Table E6, but the text points only to "Appendix D, Figure D1".
- 4.2 cites Figure E3 for per-size values the figure does not show.
- Forward references: Introduction → Section 5; Figure 1 → Section 4.2; Related work → Section 5; Table 4 caption → Section 4.4; Discussion → "(Limitations)".
- 4.4 gives numbers with no pointer at all ("at most 0.44 nats", "35 to 40%", "0.33 nats [0.18, 0.48]").

Fix: one pointer per number group, to the table that holds it. Put Experiment 1's embedding table in Appendix D. Drop "to which we return in Section 5" from Related work.

### 5. Engagement with prior literature — **Partly**

Coverage is broad (about 45 references across in-context/in-weights, caches, knowledge conflicts, new tokens, word learning). The problem is depth. Related work is a list with one clause per paper, and the paper never says how its design differs from its nearest neighbours.

Evidence: "Resetting embeddings during training keeps in-context use alive for rare tokens (Anand et al., 2025), and in synthetic biographies frequent entities shift from context to stored knowledge (Kim et al., 2026). We manipulate frequency, stored knowledge and arrangement separately for single words within one model." One sentence of positioning for the two closest papers.

Fixes:
- Spend three sentences each on Chan et al. (2022b), Singh et al. (2023), Anand et al. (2025) and Kim et al. (2026): what they varied, at what grain, and what this paper adds (per-word histories inside one model; redundancy with memory separated from recurrence).
- Say what "copying" here is relative to in-context learning. The paper's own result is that copying is gone after one intervening sentence and no induction-like heads were found. That makes it closer to a cache/adjacent-repeat mechanism than to few-shot ICL. Related work should say so.
- Missing work I am confident exists: Geva et al. (2023), "Dissecting recall of factual associations in auto-regressive language models" (EMNLP), directly relevant to the patching result that information moves from the noun's position to the last position; Neeman et al. (2023), "DisentQA" (ACL), and Li et al. (2023), "Large language models with controllable working memory" (Findings of ACL), on training models to follow or ignore context; Gao et al. (2019), "Representation degeneration problem in training natural language generation models" (ICLR), and Gong et al. (2018), "FRAGE: Frequency-agnostic word representation" (NeurIPS), on how rare-token embeddings differ, relevant to tied versus untied results; Singh et al. (2024), "What needs to go right for an induction head?" (ICML); Lazaridou, Marelli and Baroni (2017), "Multimodal word meaning induction from minimal exposure to natural text" (Cognitive Science).
- To verify before citing: Wang, Jiang, Linzen and Lake (2025), "Rapid word learning through meta in-context learning"; Teehan et al. (2024), "CoLLEGe: Concept embedding generation for large language models"; Nguyen and Reddy (2024/25), "Differential learning kinetics govern the transition from memorization to generalization during in-context learning"; Park et al. (2024/25), "Competition dynamics shape algorithmic phases of in-context learning"; Zucchet et al. (2025), "How do language models learn facts? Dynamics, curricula and hallucinations"; Monea et al. (2024), Fakepedia grounding paper; Xu et al. (2024), "Knowledge conflicts for LLMs: A survey".
- The knowledge-conflict link is asserted in the first paragraph and never used again. Either connect the "stored shape wins by 5.84 nats" result to it in the Discussion or drop the framing.

### 6. Effects that could be artefacts of the setup; baselines and controls — **Partly**

The paper is better controlled than most: fixed-shape bursts match frame, presentation and update count; redrawn bursts match recurrence; single-history corpora remove cross-condition leakage; a typical-embedding baseline is used throughout. Open artefact routes that a reviewer will raise:

a. **Unseen-noun copying is an artefact of a never-updated input embedding.** The paper says this itself and treats it as a finding. But the implication for "probes with invented words" does not follow for pretrained models, where nonce words are split into trained subword tokens. "Every word is one token" (p3) is this paper's analogue of the tokenisation problem. The Pythia test, which replaced real nouns' embeddings, found nothing. Say plainly that the unseen-noun result applies to newly added tokens with untrained embeddings.

b. **Test items conflict with memory for fixed nouns only.** The context "never names" s*, so every fixed-shape test item is a knowledge-conflict item the noun never met in training, while changing-shape items are in-distribution. The paper's defence is the spaced contrast (0.10 nats), but spaced nouns sit at the floor (0.46 against 0.57). The better defences are already in the paper and should lead: no-shape bursts (1.05) and redrawn bursts (0.49) store nothing, face no conflict, and still lose copying; and the steering direction raises copying in fixed-shape spaced nouns.

c. **Exposure count is confounded with exposure rate.** Nouns with 1,000 exposures are met ten times as densely as nouns with 100. Figure E3 hints that rate matters: a 1,000-exposure fixed-shape burst noun early in training (about 170 exposures) scores 0.2 to 1.3, while a 100-exposure one at the end scores 4.4 to 4.8. "Nouns met more often copied less" may be "nouns met more densely". Limitations mentions exposure against training time, not rate.

d. **Optimiser treatment of sparse embedding rows.** With AdamW, a row updated in few steps gets different effective step sizes. The fixed-burst and redrawn controls cover this for the main contrast. It is not covered for the exposure effect. One sentence would do.

e. **One test frame, adjacent sentence only.** Copying vanishes with one intervening sentence and keeps 35 to 40% in another frame. So "copy a feature from context" is, on the evidence, "repeat the shape of the immediately preceding sentence in the trained frame". The title and abstract should carry that scope.

f. **Baseline level moves between corpora.** Unseen nouns score 4.53 (Exp. 1), 6.35 (Exp. 2), 6.88 (Exp. 3), 4.92 (control, same seeds and sizes as Exp. 3), 7.35 (Exp. 3, five seeds), 4.87 and 6.16 (single-history). Changing-shape bursts at 1,000 score 4.71 in Experiment 3 and 3.82 in the control with the same seeds. The paper says "we compare conditions only within an experiment" but then compares the control's redrawn bursts with Experiment 2's no-shape bursts (p5, l.394) and plots panels on a shared axis. State the control's lower level in the text and give the likely reason (other nouns in the corpus).

g. **Three nouns per condition, 36 items per noun.** Fine, since nouns rotate over conditions across seeds, but say so in the main text; it is only in Appendix A.

### 7. Patterns described without mechanism — **Partly**

The paper tries: embedding swap, a shared direction (steering), patching, attention correlation. It is honest that the heads were not found. Gaps:

- The attention result is not explained. First-block attention from the last position ("a") to the context's shape word cannot depend on the query noun through the query or the key, since neither contains the noun. It can depend on it only through the softmax, if the noun's own key takes more or less of the attention. The paper says "There, attention in the first block to the context's shape word tracked copying" (p7, l.510) and stops. Say which route it is. If a familiar noun's embedding draws attention to itself and so starves the shape word, that is the mechanism, and it is a one-line test.
- "turns it off, noun by noun, through the noun's embedding" (Discussion) against steering that adds a direction which turns copying on. Decide whether the learned thing is an off-switch or an on-direction. The data (random direction at trained norm keeps copying; trained-kind mean removes it; changing-minus-fixed direction adds it) fit "familiar embeddings share a component that suppresses copying, and changing-shape bursts lack it or cancel it".
- Why does a noun's copying depend on other nouns (fixed bursts 1.03 mixed, 0.03 fixed-only)? No account is offered. "Why changing-only corpora also lowered unseen nouns' copying we cannot say." A shared mechanism for burst-shaped input, tuned by whichever burst nouns are present, would explain both and fits the "tendency, not a link" result. Say it as a hypothesis.
- The dynamics are not explained: copying for changing-shape bursts grows over training (2.6 → 4.9) while the cross-section falls from 100 to 1,000 exposures (6.11 → 4.71). One sentence should reconcile these.

### 8. Implications for language models — **Partly**

There is an "Implications for evaluation" paragraph. It makes two claims. The first (nonce probes may measure the untrained embedding) is then withdrawn for pretrained models in the next sentence. The second ("tests of in-context use should report how the probe words occurred in training") is useful but one line. Nothing is said about the three motivations in the first paragraph of the Introduction: knowledge conflicts, learning new tokens, interpreting nonce probes. The abstract ends on a finding, not an implication.

Fix: add three plain sentences. (i) For added tokens: a new token with an untrained embedding will be read from context by default; that reliance fades with exposure unless the token's context stays informative. (ii) For knowledge conflicts: in these models the conflict is settled less by competition at prediction than by whether training left any context-reading in place for that word, so per-entity training history (burstiness, stability) is a candidate predictor of context-following; mark it as a hypothesis for larger models. (iii) For evaluation: report frequency and burstiness of probe words. End the abstract with one of these.

---

## C. Flow, section by section

### Abstract (p1)
- Sentence 2 ("Across populations of items, bursty items whose meaning changes favour copying; we test this word by word within one model") assumes the reader knows Chan et al. Rewrite: "Earlier work shows that models learn to use context for classes of items that come in bursts and change meaning. We ask whether the same holds word by word inside one model."
- "except nouns whose own earlier sentences were the only clue to their current shape" cannot be parsed before the design is known. Rewrite: "except nouns whose shape changed from one burst of sentences to the next, so that only the previous sentence could give it."
- The size list interrupts the result. Rewrite: "In a pre-registered replication these nouns copied 3.7 nats more than nouns in fixed-shape bursts (2.2 to 6.4 across four model sizes)." Move tied embeddings out: it is exploratory and sits inside a sentence that begins "In a pre-registered replication".
- The control's status is missing. Add "(a pre-specified but unregistered control)".
- "not a link between noun and shape" misleads; changing shapes have no stable link by design. Rewrite: "What these nouns kept was a tendency to repeat the latest shape, whichever noun the previous sentence was about."
- No implication. Add one closing sentence (see B8).

### 1 Introduction (p1–2)
- Paragraphs 1 to 3 work: example, gap, three accounts, why natural data cannot separate them, design.
- Break 1: l.083–087 lists four studies in four sentences with no reason for each. "A first experiment set the baseline" (baseline of what?). Rewrite: "Experiment 1 asked whether familiarity alone ends copying, with nouns met one sentence at a time. Experiment 2 added bursts, the case where the useful-context account predicts copying should last. Experiment 3 repeated Experiment 2 on new models, with tests registered in advance. A final control, pre-specified but not registered, asked whether bursts keep copying because the context is useful or merely because the noun recurs."
- Break 2: l.088–110 is the abstract again, longer, with terms not yet defined ("redrawn", "at the output", "what training keeps", "the other nouns in the corpus"). It gives nine findings with equal weight. Cut to three numbered contributions, each tied to an account: (1) familiarity ends copying unless the context stays the only source (against representation alone); (2) a stored shape hardly lowers copying at prediction (against memory competition at test); (3) usefulness, not recurrence, keeps copying, and what is kept is a general tendency carried by the embedding.
- "Recurrence" appears here for the first time, undefined (see B2).
- Last sentence promises "parallels with children's word learning in Section 5"; fine, but it is the only forward reference the Introduction needs.

### 2 Related work (p2–3)
- Reads as a citation list. No paragraph ends by saying what is still unknown, except l.131–133.
- "arrangement" (l.132) is used once; the paper says "presentation" everywhere else.
- The third paragraph jumps from tied embeddings to "Lexical exposure can inflate apparent generalisation" to children's word learning without a link. Split children's word learning into its own two-sentence paragraph or move it to the Discussion, where it is used.
- Missing: a statement of what "copying" is relative to ICL and induction (see B5).

### 3 Setup (p3–4)
- Order is good: language, models, histories, experiments, measures, predictions, analysis.
- Break: "Three experiments" (l.212–239) contains results (Experiment 2's no-shape nouns "learned to expect a colour ... by 1 to 5 nats"; "where Experiment 1's copying had nearly vanished"). A reader meets findings before Results. Move the two result clauses to 4.1/4.2 and keep here only what was run.
- Break: "Predictions" comes after "Measures", but the experiments paragraph before it already assumes the reader knows why bursts and no-shape nouns exist. Move "Predictions" (with Table 3) directly after "Controlling a noun's history".
- The control is introduced with its status first and its purpose second (l.227–235). Reverse: purpose, design, then status.
- Half-kept bursts are defined in running text only; Table 1 and Figure 1 omit them.
- "Because log probabilities share one normaliser, a stored shape that only raises its own probability leaves the copy score unchanged" is a key idea, stated once, tersely. Add: "So a low copy score for a fixed-shape noun is not a by-product of the model being confident in the stored shape."
- The measure "L(s*) − L(s)" has no name; "stored preference" does. Name both.
- "added after review" (l.228): whose review? In an anonymous first submission this is confusing.

### 4.1 Baseline (p4–5)
- Answers: do unseen nouns copy, and does familiarity end it? Ends on a good question ("What, then, keeps it on for a familiar noun?").
- Break: the first paragraph mixes Experiment 1, Experiment 3 and tied models. The reader has just been told Experiment 1 is the baseline. Put "(Experiment 3: 6.88 and 0.79)" and the tied sentence at the end of the paragraph as "The same held in ...".
- "What matters most is what an embedding shares with the trained kinds' embeddings." Unclear. Rewrite: "So length is not what matters. An embedding stops copying when it points where trained nouns' embeddings point."
- "Over training, unseen nouns' copying grew (Figure E3)": Figure E3 is Experiment 2. Say so.
- The title says spaced nouns "stop". True in Experiment 1 at 1,000 and above. In Experiments 2 and 3 spaced nouns keep 1.5 to 3.2 nats at 100 and 0.5 to 1.0 at 1,000. Retitle: "The baseline: unseen nouns copy, and spaced nouns lose it with exposure".

### 4.2 Copying stays high only where the context was needed (p5)
- Three run-in heads give a clean chain: bursts with changing shapes copy; not because of less-trained embeddings; not because of recurrence. This is the best-built section.
- Break: after the tied-embedding sentence, "Fixed-shape bursts copied only 0.57 nats [0.39, 0.74] more than fixed-shape spaced nouns at 1,000." It is an untied Experiment 3 value, it reads as a tied result, and its purpose is not stated. Move it before the tied sentence and say why: "Presentation in bursts by itself added little: fixed-shape bursts copied only 0.57 nats more than fixed-shape spaced nouns at 1,000."
- "Recurrence helped only early: at 100 exposures ..." "Early" is a time word; 100-exposure nouns are different nouns at the end of training. Rewrite: "Recurrence helped only for nouns met less often: at 100 exposures ...".
- The parenthesis "(pre-registered there, both levels pooled; 0.20 [−0.10, 0.49] at 1,000)" gives an Experiment 3 number under an Experiment 2 label (see E2).
- The 85.6M sentence carries three caveats inside the result. Split.
- The title uses "needed"; the control paragraph uses "right"; the Discussion uses "useful". For fixed-shape bursts the previous sentence is always right and copying is still lost. "Needed" (right, and not available from memory) is the accurate word. Use it throughout, or define "useful" once as "right and not redundant with memory".

### 4.3 A stored shape wins at the output but barely suppresses copying (p5–6)
- Follows from 4.2: if fixed-shape bursts lose copying, is that memory beating context at test?
- "At test" paragraph: the logic (copy score insensitive to a boost of s*) was given two pages earlier. Repeat it in one clause.
- "although we could not show the effect was below 0.3 nats" is a registered equivalence result written as an aside. Say "the registered equivalence test (±0.3) was inconclusive".
- "Fixed- and changing-shape spaced nouns copied little in any case (under 2 nats)" is the main weakness of the test and is dropped in mid-paragraph. Lead the "In training" paragraph with it: "Spaced nouns copy little, so this test has little to suppress. Three further comparisons do not depend on it."
- The last two sentences are the paper's central inference. They are correct to hedge. The abstract and Discussion should carry the same hedge.

### 4.4 What is kept, and where (p6–7)
- Five run-in paragraphs, each a different analysis. The section title promises "what" and "where"; the order is what (tendency), who (query noun), context (other nouns), where (embedding), scope (narrow). "It depends on the other nouns" interrupts the mechanism thread. Move it to the end of 4.2 (it is a robustness result for the main contrast) or to 4.5 (corpus effects).
- "For changing-shape bursts at 1,000 exposures it did so by only 0.08 nats": "it" has no referent. Rewrite: "They did, but by only 0.08 nats".
- "as training never required it to check": good, keep, but it is the only mechanism sentence in the paragraph; expand by one clause ("a burst never contains a sentence about another noun").
- "In changing-only corpora, however, ... 0.33 nats [0.18, 0.48]." A new result with no explanation and no source table. Either explain or cut.
- "What in the embedding carries it": results arrive before the reader knows why they matter. Open with the question: "If the difference enters through the embedding, is it a simple property such as how far the embedding has moved?"
- "There, attention in the first block ...": "There" is ambiguous (see B7).
- "It is narrow": three results with no source and no experiment named. This paragraph limits the whole paper; it should not be the last five lines of a mechanism section. Give it its own subsection or put it in 4.2.

### 4.5 When the whole corpus rewards copying (p7)
- Twelve lines, Experiment 2 only, exploratory. It answers a fair question (does this depend on the trained kinds being stable?) but the question is not asked first. Open with: "All results so far come from corpora whose trained kinds keep their shapes. What if they do not?"
- The conclusion is wrong as written (see D8, E16).

### 5 Discussion (p7–8)
- "What decides copying" is a good summary but restates the Introduction's last paragraph for the third time (abstract, Introduction, Discussion).
- "What this adds": "per-word embeddings make this possible but do not require it" must be rewritten or cut. Possible meaning: "The model can do this because each word has its own embedding, but nothing forces it to."
- New results appear here: the Pythia test, the WikiText-103 counts, and "spaced nouns first stored their shape better". Each needs at least one sentence in Results, or a short Section 4.6 "Checks beyond the artificial language" holding Pythia and WikiText.
- "Implications for evaluation" is the only implications paragraph and half of it is a null (see B8).
- "Analogies with children" is fine in length. "as analogies only" is the right hedge.

### Limitations (p8–9)
- A list of 17 items with no order. Group: (i) language and scale; (ii) design confounds (exposure, word class, floor for the memory test); (iii) status of analyses (control, tied, mechanism); (iv) pretrained and natural-text checks.
- Two items are results, not limits: "a registered search for induction-like heads in Experiment 1 found none" and "bursts weakened stored shapes at 100 exposures". Move to Results.
- Add: exposure count against exposure rate; fixed-shape test items are conflict items; one-token words.

---

## D. Logic: claims, support, verdict

1. **Title: "Whether small transformers copy a feature from context depends on the noun's training history."**
Support: 4.14 [3.60, 4.69] (bursts − spaced, changing), 3.68 [2.99, 4.37] (changing − fixed bursts), replicated in the control and at 85.6M. **Supported.** Scope note: "a feature" is shape only; colour was not copied by any noun. "From context" is the adjacent sentence only. Suggest "copy a shape from the previous sentence" in the abstract's first use.

2. **"Nouns met more often copied ... less" (abstract; "Familiar nouns copy less the more often they have occurred", Introduction).**
Support: Experiment 1, spaced, 10 to 10,000 exposures; Experiments 2 and 3, 100 against 1,000. **Supported for the cross-section; overstated as a process.** These are different nouns at the end of training, and count is confounded with rate (B6c). Over training, copying grew for changing-shape bursts. The paper says both things but uses "lose", "stop", "kept", "early", "first" throughout. Fix: state once that "lose" and "keep" compare nouns of different exposure, and avoid time words.

3. **"Unseen nouns copy because nothing in their untrained embeddings turns this off."**
Support: typical embedding removes copying (−0.33; 0.79; 0.02 tied); trained-kind mean removes it (−0.44); random direction at trained norm keeps it (4.09). **Supported**, with one caveat: by size the typical embedding leaves −0.19, 1.20 and 1.37 nats (Experiment 3), so "removed" holds fully only at 3.4M. Limitations already notes the typical embedding "may be atypical in its own way". Say "removed most of it".

4. **"Copying stays high only where the context was needed" (4.2 title).**
Support: changing-shape bursts 4.71 against at most 1.1 for other histories; fixed-shape bursts, no-shape bursts and redrawn bursts all low. **Supported in stable-kind corpora with untied embeddings.** With tied embeddings other histories keep 1.4 to 2.9 nats. In shifting-kind corpora every burst history keeps 2.1 to 6.8. The title is true only for stable-kind, untied models; the text says so later. Add "in stable-kind corpora" to the title or the first sentence.

5. **"Recurrence alone did not keep copying" (abstract, Introduction, Discussion).**
Support: redrawn − spaced 0.07 [−0.01, 0.15] at 1,000; changing − redrawn 3.33 [2.74, 3.92], all seeds and sizes. **Supported at 1,000 exposures; correctly qualified at 100 (0.96 above spaced).** Not registered; the abstract and Introduction do not say so. Add the status.

6. **"in graded form: bursts kept more copying the more often their previous sentence was right" (Discussion).**
Support: three repeat rates, 0.49 / 1.20 / 3.82. **Overstated in two ways.** Three points give an order, not a function, and the step is uneven (0.71 then 2.63). And fixed-shape bursts have a repeat rate of 1.00 and score 0.82, so "right" is not the variable; "right and not available from memory" is. Rewrite: "Among bursts with nothing to store, copying rose in order with the share of sentences repeating the previous shape (three levels)."

7. **"A stored shape barely suppressed the copied one" (abstract, 4.3 title, Discussion).**
Support: fixed − changing spaced, −0.25 [−0.35, −0.15]; −0.40 at 100, −0.10 at 1,000; equivalence test inconclusive (TOST p = 0.14). **Overstated as worded.** The contrast was significantly different from zero and not shown equivalent. It was measured where there is almost nothing to suppress (under 2 nats; 0.5 at 1,000). In proportion it is about a fifth, as the paper itself computes. The claim the data support is: "Suppression at prediction is too small to explain the 3.7-nat burst difference." Use that wording in the abstract and title of 4.3.

8. **"its larger effect was on what training kept" / "memory's effect fell on training, not on prediction" (abstract, Discussion).**
Support: indirect. Fixed-shape bursts lose copying though test suppression is small; no-shape and redrawn bursts also lose it. **Understated hedge in the abstract and Discussion, properly hedged in 4.3** ("Memory was not manipulated during training, so this is an inference"). Also, the evidence shows that copying is lost by default and kept only when needed; it does not show that memory does anything during training. No-shape and redrawn bursts lose copying with no memory at all. The accurate claim: "copying is kept only where training needs it; a storable shape removes that need." Use this in all three places.

9. **"What bursts kept was mostly a tendency to copy the latest shape, not a link between noun and shape."**
Support: same − other noun 0.08 [0.02, 0.14], within ±0.3; query noun's history matters more than the context noun's (1.72 against 0.47). **Supported**, but (a) the effect is different from zero, and 0.33 [0.18, 0.48] in changing-only corpora, so "mostly" is needed and present; (b) "link between noun and shape" is the wrong description (see C, Abstract).

10. **"how much of it a noun keeps depends on the other nouns in the corpus."**
Support: fixed-shape bursts 0.03 (fixed-only) against 1.03 (mixed), difference 1.00 [0.80, 1.20]. **Supported for fixed-shape bursts only.** For changing-shape bursts the registered comparison was inconclusive (0.24 [−1.44, 1.92]). For unseen nouns, "copied less in both single-history designs" is supported for fixed-only (4.87 [3.84, 5.90] against 7.35) but not shown for changing-only (6.16 [4.25, 8.08]); five seeds. Reword to say which nouns.

11. **"So whether a stored shape ends copying depends on the corpus: where the trained kinds themselves reward copying, it does not." (4.5)**
**Unsupported as worded, and it contradicts 4.3.** Section 4.3 argues a stored shape does not end copying. And in shifting-kind corpora the gap between changing- and fixed-shape bursts at 1,000 is 3.25 nats (6.82 − 3.57), larger than Experiment 2's stable-kind gap (2.86). What changes is the floor: every burst history keeps copying. Rewrite: "So how much copying a familiar noun keeps depends on the corpus. Where the trained kinds themselves reward copying, every history in bursts keeps some, and the gap between changing and fixed shapes remains (3.25 nats)."

12. **"The pattern is clearest in the smallest models" (Introduction, 4.2, Discussion).**
Per-size effects: 6.36 (3.4M), 2.44 (10.9M), 2.24 (25.6M), 4.18 (85.6M). **Overstated.** It is largest at 3.4M; 85.6M is second. And Setup says sizes "are not a scaling trend". Say "largest in the 3.4M models" and nothing about order.

13. **"history mattered beyond three simple measures of the embedding's geometry" / "the difference lives in the embedding".**
Support: 3.67 [3.03, 4.32]; embedding swap moves the copying; the paper notes this "rules out these three summaries, not geometry in general". **Supported and well hedged.**

14. **"by the end of the second block the difference between histories had passed from the query noun's position to the position where the shape is predicted, as a readout through attention would require."**
Support: patching, Appendix F. **Supported as description.** The link to the attention correlation is not argued (B7).

15. **"In small models trained from scratch, such a probe can measure the untrained embedding more than anything learned about context."**
Support: unseen against typical embedding. **Supported for single-token words with never-trained embeddings; does not transfer to subword-tokenised nonce words.** The Pythia null is reported honestly. Narrow the claim (B6a).

16. **"Natural text resembles changing-shape bursts in one respect."**
Support: WikiText-103 counts for 44 nouns, colour only, names inflate agreement. **Supported as a rough observation**, and hedged ("exploratory", "rough"). It appears only in the Discussion; give it a line in Results.

17. **"a registered search for induction-like heads in Experiment 1 found none" (Limitations).**
Table C2: the ablation test for unseen nouns was not supported; the one for trained kinds in shifting-kind corpora was "Supported; random heads cut it nearly as much". Table D4: at 25.6M shifting-kind, ablating the candidates took unseen copying from 6.88 to 0.26 (random heads: 3.26). **Too strong.** Say "did not identify heads whose ablation reliably removed copying; prefix-matching scores were at chance".

---

## E. Internal consistency errors

Numbers that I checked and found consistent across abstract, text, tables and results files: 3.68/3.7; 6.36, 2.44, 2.24; 4.18; 1.90; 4.14; 3.33 with 5.82, 2.04, 2.14; 0.07; 1.20; 0.96; 3.01; 0.25/0.40/0.10; 5.84; 3.58; 1.05 against 1.03; 0.49 against 0.82; 5.51; 4.27; 1.00; 1.72 against 0.47 and 1.25; 3.67; cosines 0.85 and 0.78; 1.51, 1.99, 2.41, 1.12, 1.00; 0.80 [0.76, 0.85]; all of 4.5; all Table 4 cells except the two listed below; model and corpus counts.

Errors and conflicts:

1. **p2, Figure 1(a):** "Control: the query names another new noun (A zelp is a ___)." Everywhere else "control" is the redrawn/half-kept study, including the same figure's caption ("Redrawn-shape bursts are the control"). Setup calls these "other-noun items". Fix: "Other-noun item: the query names another new noun."

2. **p5, l.394–396:** "as no-shape bursts did in Experiment 2 (pre-registered there, both levels pooled; 0.20 [−0.10, 0.49] at 1,000)". The value 0.20 [−0.10, 0.49] is Experiment 3's (Table 4, Exp. 3 column; `amendment11/SUMMARY.md`). Experiment 2's values are 0.80 pooled and 0.05 at 1,000. Fix: "as no-shape bursts did in Experiments 2 and 3 (0.80 and 0.69 pooled, the former pre-registered; 0.05 and 0.20 at 1,000)".

3. **p6, Table 4:** "Beyond geometry ... 2.84" for Experiment 2. Table E4's caption (p19) gives "3.19 nats [2.46, 3.91] more than geometry predicts at 1,000". 2.84 appears nowhere in the appendix or in the results summaries I searched. Fix: report one value with its method, or give both and say which is seed-level and which noun-level (as Table F1's note does for 3.67 and 3.82).

4. **p6, l.448–450:** "a noun-specific part of 0.33 nats [0.18, 0.48]" in changing-only corpora. No appendix table carries it. Table 4's Experiment 2 "Same noun − other noun 0.05, 0.05" also has no appendix source. Fix: add a row to Table F1 or F2, or a sentence in Appendix F.

5. **p6, Figure 2 caption:** "presentation (solid: bursts; dashed: spaced)". In panel (b) every series is bursts; dashed with hollow markers means 100 exposures and solid means 1,000; the dotted lines are references. Fix: add "In (b), dashed: 100 exposures; solid: 1,000 exposures; dotted: reference levels."

6. **p6, Figure 2 caption (b):** "copy score of bursts with changing shape against how often a burst sentence repeats the previous one's shape (redrawn, half-kept, changing)". "Changing-shape bursts" is one of the three conditions; here "bursts with changing shape" names all three. Fix: "copy score of bursts with no fixed shape".

7. **p6, Table 4:** the rows "Bursts: changing − redrawn" and "Redrawn − spaced" sit under the "Exp. 3" header. They come from the control's models, where changing-shape bursts score 3.82, not Experiment 3's 4.71. A footnote says so, but the header says otherwise. Fix: give the control its own column pair, and add its "Bursts: changing − fixed" (0.90 pooled is not shown; 3.01 at 1,000).

8. **p5, l.378–379:** "Fixed-shape bursts copied only 0.57 nats [0.39, 0.74] more than fixed-shape spaced nouns at 1,000." It follows the tied-embedding sentence and reads as a tied result. It is the untied Experiment 3 value (1.03 − 0.46). Fix: move before the tied sentence and label it.

9. **Time words for cross-sectional contrasts.** p5, l.392: "Recurrence helped only early: at 100 exposures". p8, l.611: "spaced nouns first stored their shape better". Against p5, l.353: "The comparison between 100 and 1,000 exposures is between different nouns at the end of training" and Figure 2's "lines join conditions, not time points". Also "lose", "stop", "kept", "ends copying" throughout. Fix: "only for nouns met less often"; "at 100 exposures spaced nouns had stored their shape better".

10. **Half-kept bursts missing from the design displays.** p3, Table 1 lists "Redrawn shape" but not half-kept. p2, Figure 1 says "No-shape nouns (not shown)" and omits half-kept without saying so. Table 3 has no half-kept prediction. Table C1 (p13) has neither redrawn nor half-kept. Fix: add a Table 1 row; "No-shape and half-kept nouns (not shown)".

11. **p2, Figure 1(b) caption:** "with the same number of sentences, bursts occupy a quarter as many windows as spaced sentences". The drawing shows three spaced sentences in three windows and eight burst sentences in two. Fix: draw one burst in three windows, or say "not to scale: sentence counts are matched in the experiments".

12. **Two "own-shape" measures.** p4, l.256–257: "which shape the model prefers, L(s*) − L(s)" (no name). p15, Appendix D: "own-shape preference = L(s*) − L̄", a different quantity, tabled as "Own shape" in Table D3. Table C2 has both under near-identical wording. Fix: name the first "own-over-context preference" and keep "own-shape preference" for the second; define both in Setup.

13. **p7, l.515–521 ("It is narrow"):** "at most 0.44 nats" is Experiment 3's value, which is in no appendix table (Table E4 shows Experiment 2's 0.37). "kept 35 to 40%": the tables give 40% (Experiment 2: 1.53 of 3.85) and the results file gives 38% (Experiment 3: 1.77 of 4.71); nothing gives 35. No experiment is named. Fix: name the experiment, give "38 to 40%", and add the Experiment 3 values to Appendix F.

14. **Model size.** p2, l.104–106: "clearest in the smallest models"; p5, l.350: "sharpest in the smallest models"; p7, l.545: "most clearly in the smallest models"; p8: "The effect shrank from 3.4M to 25.6M". Against p3, l.194: "differences between sizes are not a scaling trend"; p5, l.371: "we do not read the pattern across them as a trend"; and 85.6M (4.18) above 10.9M (2.44) and 25.6M (2.24). Fix: "largest in the 3.4M models" in all four places.

15. **p7, l.532–534, Section 4.5:** "So whether a stored shape ends copying depends on the corpus: where the trained kinds themselves reward copying, it does not." Conflicts with Section 4.3's title and argument (a stored shape does not end copying at all) and with the data (the changing − fixed gap is 3.25 nats in shifting-kind corpora). Fix: see D11.

16. **Results that first appear after Results.** p8, l.611 and l.628: weaker storage in bursts at 100 exposures (Table E3: 3.02 against 5.37; registered check K1) is used in the Discussion and Limitations and never reported in Section 4. p8, l.589: the Pythia test. p8, l.597: WikiText-103. p8, l.648: the head search. Also p4, l.290: "These are all the experiments run with this design" and Table 2 list three experiments and a control; the pretrained test (five primary Pythia models, registered) is a fourth registered study and is in neither. Fix: one short Section 4.6 and a Table 2 footnote.

17. **Status of the control.** p3, l.227–229: "not pre-registered, but with its design and predictions fixed before training". p4, l.299–300: "exploratory, including the control". p25: "Primary contrast, fixed in advance". Table 3: "–: none; exploratory". Abstract and Introduction: no status. A study with a pre-fixed primary contrast is not exploratory in the usual sense. Fix: one label everywhere, "pre-specified, not registered", including the abstract and l.086.

18. **"After review".** p3, l.228 and p8: "added after review"; p25: "Added in answer to a review". No review is identified. Appendix E mentions "a blind review ... by an AI assistant" for Experiment 2. In an anonymous first submission the reader cannot tell what this means. Fix: say what kind of review (for example "an internal review of a draft").

19. **Number of frames.** p3: "one of five sentence frames"; p8: "five sentence frames"; p12: "Shape is named in all five frames". Against p12: no-shape nouns use "a sixth frame, The [noun] is [colour] and [texture], which the trained kinds also used in one sentence in fifteen", and p18 calls another "frame 7". So each corpus has six frames, one without a shape. Fix: "five frames that name the shape, plus a sixth without a shape used in one sentence in fifteen".

20. **Differences between Experiments 1 and 2.** p3, l.215–218: Experiment 1's nouns "differed from the trained kinds in two ways" and "Experiment 2 removed both differences and added bursts". Appendix A (p12) shows more changed: fixed- and changing-shape nouns went from frames 1 to 5 to frame 1 only, and new nouns' share of windows fell from about 60% to 8%. Fix: list all changes in Setup.

21. **p5, l.337:** "Over training, unseen nouns' copying grew (Figure E3)" inside the Experiment 1 paragraph; Figure E3 is Experiment 2. **p5, l.355–360:** per-size values "2.6, 1.1 and 0.8 ... 4.9, 3.3 and 3.3" and "between 0.2 and 1.3" cite Figure E3, which shows sizes averaged. Fix: name the experiment; add the per-size values to Appendix E or cite the supplementary file.

22. **Section 4.1 title:** "familiar spaced nouns stop". In Experiments 2 and 3 spaced nouns score 1.5 to 3.2 nats at 100 exposures and 0.5 to 1.0 at 1,000 (Tables E2, F2). p3, l.219: "where Experiment 1's copying had nearly vanished" is true of Experiment 1 only. Fix: retitle (see C) and add "in Experiment 1".

23. **p4, l.304 and l.309–315:** the embedding manipulation (4.00, 4.09, −0.44) is reported as Experiment 1 with pointer "Appendix D, Figure D1"; the data are in Table E6 (Appendix E) and the test was registered with Experiment 2. Also "from 4.53 to 4.00" (a fall of 0.53) against Table E6's "−0.52". Fix: point to Table E6; note the rounding or give three decimals once.

24. **Same-noun minus other-noun against the crossed table.** p6: other-noun items scored 0.08 lower (Table F1). Table F3 (p24): changing-burst context with a different changing-burst query noun scores 4.73, above the same-noun 4.71 in Table F2. These are different item sets (36 against 24 items), but the reader sees a sign conflict. Fix: one clause in Table F3's caption.

25. **Terms: two words for one thing.**
 - "arrangement" (p2, l.132) against "presentation" everywhere else.
 - "condition" (Setup, Figure 1), "cell" (Appendices D, E, F), "history" as a column head that holds history and presentation (Tables F2, G1), "noun type" (Table D2), "Noun" (Tables E2 to E4).
 - "useful" / "needed" / "right" / "no guide" / "paid off" / "only clue" / "only source" for the same property.
 - "link between noun and shape" (abstract, Introduction) / "tied to the noun" (Setup) / "noun-specific part" (4.4) / "Same noun − other noun" (Table 4) / "noun-specific copy score" (Table F1).
 - "runs" (trained kinds) against "bursts" (new nouns) for the same structure; Table C1 says the registrations call bursts "runs".
 - "first block" (4.4, Appendix F) against "layer 0" / "layer-1 head" (Table D4).
 - "trained kinds" against "trained nouns" (Table E2 caption).
 - "typical embedding" (mean plus offset) and "trained kinds' mean embedding" are different things and both "remove" copying; say so once.
 Fix: pick one of each; add redrawn and half-kept to Table C1.

26. **p13, Appendix C:** "so we chose untied embeddings for this paper and check every model and that a loading error had changed their scores; both were corrected and disclosed". Broken sentence; "both" has one referent. Fix: "... so we use untied embeddings in this paper and check every model. We also found that a loading error had changed the earlier scores. Both problems were corrected and disclosed in the registrations."

27. **p7, l.469–475:** "Unseen nouns copied less in both single-history designs than in mixed corpora (fixed-only 4.87, changing-only 6.16, mixed 7.35)". The changing-only interval is [4.25, 8.08] (Table F2), which covers 7.35. Not reported at all: unseen nouns in the control score 4.92 [4.59, 5.25] (p26) against 6.88 in Experiment 3 with the same seeds and sizes, and changing-shape bursts 3.82 against 4.71. Fix: "lower in fixed-only corpora, and not reliably different in changing-only"; add the control's lower level to 4.2.

28. **Abstract:** "In a pre-registered replication, ... (...; 1.9 with tied embeddings)". The tied models are exploratory and were added after the results (Table 2 note; Appendix F). Fix: move tied out of the parenthesis.

29. **p8, Limitations:** "trained with one recipe" against p3: "Sizes also differ in training steps, batch size, learning rate and amount of text". Fix: "one optimiser and schedule family".

30. **p8, l.648–650:** "a registered search for induction-like heads in Experiment 1 found none" against Table C2 (one of the two ablation tests "Supported") and Table D4. Fix: see D17.

31. **p4, l.289–290:** "Each experiment was registered before any of its models was trained." Table 2 is titled "Experiments" and includes the control, which was not registered; Appendix F notes "deleted 100-step pipeline checks". Fix: "Each of the three experiments ... (the control was not registered; see below)".

---

## Is the control integrated or bolted on?

Mostly integrated. It is in Figure 1, Tables 1 to 4, Figure 2(b), Sections 4.2 and 4.3, the Discussion and Limitations, and it has its own appendix. The seams that still show:
- The alternative it tests is not one of the Introduction's accounts and has no column in Table 3.
- The word "control" already labels the other-noun items in Figure 1(a).
- Half-kept bursts are missing from Table 1, Figure 1, Table 3 and Table C1.
- Its rows sit under "Exp. 3" in Table 4.
- Figure 2's caption does not describe its panel.
- Its status is worded four ways and is absent from the abstract and Introduction.
- Its lower baseline (unseen 4.92; changing bursts 3.82) is not mentioned.
- Sections 4.4 and 4.5 and Appendices C to F do not know it exists.

---

## What a reader remembers after one read

"New words copy because their embeddings are untrained. Familiar words stop, unless they come in bursts whose shape changes between bursts. A redrawn-burst control shows this is about the context being informative, not about recurrence."

That is most of the intended message. Two intended points do not land. The first is "memory acts through training, not at prediction": it is stated in abstract phrases, hedged in one place and not in others, and its direct test is at the floor. The second is "why this matters for language models": the one implication with teeth is withdrawn for pretrained models. A reader will also remember the density: many numbers, many caveats, and four studies with different baselines.

---

## F. The ten most important changes, in order

1. **Make recurrence a named account.** Add it as a fourth bullet in the Introduction and a column in Table 3, with rows for no-shape and half-kept bursts. Then the control answers a question the paper asked.

2. **Fix the "control" collision and the control's displays.** Rename Figure 1(a)'s "Control" to "Other-noun item"; correct Figure 2's caption for panel (b); give the control its own columns in Table 4; add half-kept to Table 1, Figure 1 and Table C1; use one status label ("pre-specified, not registered") everywhere, including the abstract and Introduction.

3. **Restate the memory claim so the evidence supports it.** Replace "a stored shape barely suppressed the copied one" and "its larger effect was on what training kept" with: "suppression at prediction is too small to explain the burst difference; copying is kept only where training needs it, and a storable shape removes that need." Carry Section 4.3's hedge into the abstract and Discussion. Lead with the no-shape and redrawn bursts, not the floor-level spaced contrast.

4. **Correct Section 4.5's conclusion.** It contradicts 4.3 and the data (the 3.25-nat gap remains in shifting-kind corpora). Use the rewrite in D11.

5. **Fix the wrong or unsourced numbers.** The Experiment 3 value attributed to Experiment 2 (0.20 [−0.10, 0.49]); Table 4's 2.84 against Table E4's 3.19; the unsourced 0.33 [0.18, 0.48], 0.44 and "35 to 40%"; the misplaced 0.57 sentence.

6. **Rewrite the abstract and the last Introduction paragraph.** Cut the size list. Replace the nine-finding paragraph with three numbered contributions tied to the accounts. Replace "link between noun and shape", "at the output" and "only clue" with plain descriptions. End the abstract on an implication.

7. **Move results out of the Discussion and Limitations.** Add a short Section 4.6 for the Pythia test and the WikiText counts; report weaker storage in bursts and the head search in Section 4. Move Experiment 2's findings out of Setup. Add the pretrained test to Table 2.

8. **Scope the claims to what was measured.** Shape only, adjacent sentence only, single-token words, stable-kind corpora, untied embeddings, largest at 3.4M (not "smallest models"). Remove time words ("early", "first", "lose", "stop") for contrasts between different nouns, or define them once. Add exposure rate and the conflict status of fixed-shape test items to Limitations.

9. **Give the implications real content and position the work.** Three plain sentences on added tokens, knowledge conflicts and evaluation (B8), with the nonce-probe claim limited to untrained single-token embeddings. In Related work, spend three sentences each on Chan et al. (2022b), Singh et al. (2023), Anand et al. (2025) and Kim et al. (2026), say how "copying" here relates to in-context learning, and add Geva et al. (2023) and the context-following training papers.

10. **Unify terms and tighten the mechanism paragraph.** One word each for presentation, condition, needed/useful, noun-specific copying, block/layer; name the L(s*) − L(s) measure. In 4.4, open each analysis with its question, explain how first-block attention can depend on the query noun, reconcile "turns it off" with a direction that turns it on, and move "It depends on the other nouns" and "It is narrow" out of the mechanism thread. Repair Appendix C's broken sentence and the frame count.
