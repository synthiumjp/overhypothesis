## Part 1. The 11 desk-rejection complaints

1. **Lab-notebook prose: mostly fixed in the main text, still present in the appendices.** Results are organised by claim. The introduction and Setup still tell the design history: "A second removed two differences ... that the first left open"; "Experiment 2 (60 models) removed both differences." Appendices: "We found this on 12 October 2026, fixed the loader"; "their training loop was started again by mistake."
2. **Tokenisation and confounds: tokenisation is fine; some confounds are disclosed but still live.** Every word is one token. Still open: no-shape nouns differ from the others in the word class they expect next ("by 1 to 5 nats"); bursts weakened stored shapes at 100 exposures (check K1 failed); exposure is confounded with training time. The unseen-noun result "depends on untied embeddings" (by design here, but Appendix E says the earlier studies' tie "was lost on the training hardware").
3. **Narrow domain: still present.** One feature, ten values, one-sentence lag; the effect "almost vanished" with one sentence in between. "For colour ... no noun copied." The Pythia probe uses colour only, with 22 noun pairs.
4. **Weak baselines: largely fixed.** Good internal controls (typical embedding, other-noun control, window-matched fixed-shape bursts, random order). Missing: tied-embedding models; the trained kinds' own copy score as reference.
5. **No mechanism: still present.** "The mechanism remains open." The head search failed; the row swap is circular by the paper's own wording ("as it must, since each noun is a single token").
6. **No implications: partly fixed.** A three-sentence Implications paragraph; its main claim (nonce-word probes) is undercut by the paper's own Pythia null.
7. **Missing domain literature: mostly fixed.** Gaps listed below.
8. **Hypotheses not stated in the introduction: fixed.** Three accounts with predictions appear in the introduction and in Table 2.
9. **Conditions unmotivated: mostly fixed.** Fixed-shape bursts are clearly motivated as a control for row training. Unmotivated: 100/1,000 exposures, runs of four, the 85.6M warm-up.
10. **Cryptic labels: fixed in the main text, still present in the appendices.** The appendices use "carry," "atypical\_init," "pure\_changing," "Amendment 10, Part A" and "H40–H59," decoded in a translation table (Appendix G).
11. **Bookkeeping: still present.** The 8 main pages carry about 36 "(pre-registered)"/"(exploratory)" tags, for example "(pre-registered in Experiment 3; Experiment 2: 3.39; Figure 3a; Table 3)". The appendices keep leftovers from earlier studies: "rows 26 to 32," "two models of about 70 billion parameters," "hand coding of Study 2," and an uncited Appendix F using a different measure ("index").

**Anonymity:** Appendix J prints `osf.io/qj9hb`, while Appendix A says "link withheld for review." This could cause a desk rejection.

---

## Part 2. ARR review

**Summary.** 220 small transformers are trained on an artificial object language, controlling each new noun's frequency, shape history and presentation (spaced or bursts), and measuring copying of a shape from the previous sentence. Unseen nouns copy because their input embeddings are untrained. Familiar nouns lose copying, and a stored shape lowers it only slightly. Nouns met in bursts whose shape changes between bursts keep copying (+3.7 nats over fixed-shape bursts, replicated in a pre-registered study, +4.2 at 85.6M). This copying is not bound to the noun and partly spreads to other nouns. The embedding manipulation does not transfer to Pythia.

**Strengths**
- Careful control of single words' histories inside one model; fixed- vs changing-shape bursts (same windows) cleanly separate useful context from representation.
- Pre-registered before training; Experiment 3 confirms Experiment 2; failed predictions reported honestly.
- The finding that memory and context add up rather than compete ("a stored shape barely suppressed copying ... but outweighed it in prediction") is useful for the knowledge-conflict literature.

**Weaknesses (most serious first)**
1. **The headline result is what the training loss directly rewards.** Changing-shape bursts train exactly the test string (*A N is a s … A N is a s*). The paper concedes this: "That a model copies where copying lowers its loss is expected." The non-obvious findings (memory and context add up; copying is noun-blind; it spreads) are framed as secondary. *Fix (no new experiments):* treat useful context as the expected baseline, and lead the abstract and introduction with the three non-obvious findings.
2. **The untrained-embedding claim may not generalise.** It depends on untied embeddings, and it fails in Pythia. Even so, Implications advises checking "with rows that look familiar." *Fix:* limit the claim explicitly to untied models with untrained rows. *New experiments:* one tied-embedding replication at 3.4M, and Pythia's known under-trained tokens (Land & Bartolo, 2024) instead of random rows.
3. **There is no mechanism, and the row analysis is weak.** "Beyond what geometry predicts" only shows that three scalar summaries (cosine to initial row, distance to the mean, norm) do not capture the row. *Fix with the existing checkpoints:* take the mean difference between changing-burst and fixed-burst rows and add it to other nouns' rows as a steering vector. Also run activation patching at the query position, and attention-to-shape by layer.
4. **The "spread" claim goes beyond the evidence.** "plausibly changing-shape bursts" is not tested: fixed-only corpora also lack no-shape nouns and changing-shape spaced nouns. *Fix:* say "nouns of other histories," or test it with the crossed table (Table J3).
5. **The scope is narrow and the effect is fragile.** Effects are largest at 3.4M (6.36 vs 2.24 nats at 25.6M); 85.6M differs in warm-up. In shifting-shape corpora the conclusions reverse ("recurrence alone sustains it"), and the paper never says which regime natural text resembles. *Fix:* discuss this, and estimate per-noun burstiness and attribute stability in the Pile with infini-gram, which the authors already use.

**Questions for authors**
1. With tied embeddings, do unseen nouns still copy?
2. What is the trained kinds' copy score in stable-shape corpora?
3. In Table J3, unseen queried nouns copy between 3.47 and 7.09 nats depending on the context noun's history. Why does the context noun matter so much here?
4. Why does the Pythia probe compare contexts rather than use the same copy score?
5. New nouns fill "about 60% of training windows" in Experiment 1. Does this change the trained kinds' behaviour?

**Missing references**
- Press & Wolf, 2017, EACL; Inan, Khosravi & Socher, 2017, ICLR (embedding tying).
- Herbelot & Baroni, 2017, EMNLP; Khodak et al., 2018, ACL (learning embeddings for new words from little data).
- Minixhofer, Paischer & Rekabsaz, 2022, NAACL (WECHSEL); Dobler & de Melo, 2023, EMNLP (FOCUS); both on initialising new tokens, relevant to the Implications.
- Akyürek et al., 2024, ICML ("In-context language learning"); Edelman et al., 2024, NeurIPS (statistical induction heads).
- Geva et al., 2023, EMNLP (factual recall mechanism); Elhage et al., 2021, Transformer Circuits Thread.
- Xie et al., 2022, ICLR (in-context learning as implicit Bayesian inference; relevant to the overhypothesis framing).

**Presentation issues**
- Abstract: "a stored shape lowered copying slightly but outweighed the copied shape." → "A stored shape barely reduced copying. But the model still preferred the stored shape over the copied one."
- §3: "The useful-context account alone predicts that changing-shape bursts copy more than changing-shape spaced nouns, unless bursts also keep rows less trained, which the representation account would then predict too." → "Only the useful-context account predicts that changing-shape bursts copy more than spaced nouns. The representation account predicts this too if bursts leave rows less trained. Fixed-shape bursts test that."
- §3: "leaving out the noun's own shape s\* (a fixed placebo shape for nouns that have none), which s never is." → "The average excludes the noun's own shape s\*. The context never names s\*. Nouns without a shape get a fixed placebo shape."
- §4.4: "The difference was reliable but not shown to be smaller than 0.3 nats, so a stored shape did lower copying." The conclusion follows from the significance test, not from the equivalence test. → "The difference was reliably above zero, so a stored shape did lower copying. We could not show it was under 0.3 nats."
- §4.5: the sentence beginning "At 1,000 exposures, a queried noun..." packs five estimates into one sentence. Split it into three: the query effect, the context effect, the difference.
- Replace non-standard "row" with "embedding."

**Scores**
- Soundness: 3.5
- Excitement: 2.5
- Overall: 3 (Findings)
- Confidence: 4

**Five changes that would most raise the Overall score (ranked)**
1. Reframe around the non-obvious findings (memory and context add up; copying is not noun-bound; it spreads) and present useful-context copying as the expected baseline.
2. Add a mechanism analysis on the existing checkpoints: steering with the row-difference vector, plus activation patching.
3. Settle the tied-embedding question with one small replication. Also test Pythia's under-trained tokens, or narrow the claim and the Implications.
4. Strip the bookkeeping. Move registration status into a table column. Delete appendix material from earlier studies (Table A4, Appendix F, the 70B and Study 2 notes) and the dated process narrative. Fix the OSF anonymity leak.
5. Link the result to natural text: say which regime (stable or shifting) real nouns resemble, using Pile burstiness statistics, and soften the "spread" attribution.
