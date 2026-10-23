# Amendment 9: registration record and addendum

## Registration
- Registered on OSF on 23 October 2026 (Australia/Sydney) as osf.io/zde87, a registration in project osf.io/qj9hb.
- Registered text: docs/AMENDMENT9.md. Two copies of draft 2 existed on the day, identical except that one adds the
  sentence "All Amendment 9 models, including the pilot, are trained on one machine (the DirectML GPU used for
  Amendment 8), so hardware cannot differ between cells." (MD5 without that sentence 1ef2b4116e3f37199a4e76048f85bc3b;
  with it 84afcd6027a9109707840510ce062b47). The repository keeps the copy that was registered.
- At registration no Amendment 9 corpus had been built, no Amendment 9 model trained, and no circuit analysis run on
  any model.

## Clarifications and fixed details
Written on the day of registration, before any Amendment 9 corpus was built, while the scripts were being written.
None changes a hypothesis, a test or a decision rule.

1. The registered file's header still reads "DRAFT 2 ... not yet registered". The text was registered as written; the
   header is a leftover from drafting.
2. Section 3.4 says the pilot is run and the number of nouns per cell fixed "before registration". Because the
   registration came first, this is read as "before any confirmatory model is trained". The rule is fixed in advance:
   two nouns per cell unless the pilot's variance estimates give power below 0.8 for a 1-nat effect at the strictest
   Holm level, in which case three (if three also falls short, three are used and the shortfall is reported). The pilot
   result and the choice are recorded below, dated, before any confirmatory model trains.
3. All Amendment 9 models, including the pilot, are trained on one machine (the DirectML GPU used for Amendment 8).
4. Context draws. Section 3.5 asks for 30 draws per noun with every shape used equally often as the context shape s,
   and s never s*. Nine shapes cannot be used equally often in 30 draws, so 36 draws are used (each of the nine shapes
   other than s* four times). Knowledge items also use 36 neutral prefixes.
5. Frames. "Frames 1 to 5 in the base proportions" means the proportions of the generator (corpus_generator.py draws the
   five frames with equal probability). Consistent and uninformative nouns therefore use each of frames 1 to 5 for
   exactly one fifth of their exposures at every exposure level. Base kinds use frame 6 for each sentence with
   probability 1/15, independently, in every pass.
6. Exposure schedule. Warm-up ends at 4% of training for the 3.4M models and 5% for the others, and the 3.4M and 25.6M
   models share a corpus and a reading order. Exposures are therefore placed at evenly spaced read positions from 5% to
   95% of the windows, which is after warm-up for every size. If two exposures fall in one window, the later is moved to
   the nearest free window inside that band (never outside it); every move is listed in the corpus manifest. Moves are at
   most a few dozen windows (under 1.5 training steps).
7. Window boundaries. A familiarity sentence lies wholly inside one window as both inputs and targets: its first token
   is at least the window's second position, and its last token at most the window's last input position. Within the
   window it goes at the first permitted boundary at or after a random offset (so its position in the window has the
   same spread in grouped and random corpora), or earlier if no later boundary would fit.
8. Base exemplars. The five new seeds (2718, 3141, 1618, 1414, 1732) and the pilot seed (2027) take their 448 base
   exemplars from corpus_generator.py exactly as the original seeds did; the builder checks that this reproduces the
   exemplars of the five original seeds.
9. Untying. The output matrix starts as a copy of the input matrix and is then trained separately, which is what
   happened on DirectML in every earlier model.
10. Ablation position. Mean ablation acts at the position that predicts the shape (the last token of an item). The mean
    is the head's output at positions that predict a shape in frame-1 sentences of base-corpus windows (2,000 positions
    from the model's own corpus).
11. Random heads. Each random draw replaces every candidate head by a non-candidate head from the same layer (without
    replacement); if a layer has too few, the remainder come from other layers.
12. Familiarity nouns (40, used in this order to fill cells): klerd, pind, klirm, lilb, zelp, virk, masp, grirk,
    staft, blark, fresp, plask, trort, plert, grosk, snilp, pilf, klulk, falp, zurb, parb, zusp, klesk, fuft, trirb,
    morb, maft, plorb, klarb, stind, sturm, snamp, derk, grert, snolb, dalb, polp, fronk, pisp, plond. None is a
    substring of, or within one letter of, another word of the language. With two nouns per cell the first 28 are used.
    The noun-to-cell rotation and each noun's s* are in docs/amendment9_assignment.json. s* is drawn afresh for each
    seed (distinct within a cell), so the shapes of different cells do not line up in the same way in every seed.
13. Pilot rule, computed by scripts/amendment9_tests.py pilot. From the pilot's grouped Regular model, the standard
    deviation of carry between nouns of the same cell (sigma) is combined with the Amendment 8 seed-level standard
    deviation (0.81 nats) to give the expected seed-level standard deviation of the H31 contrast with m nouns per cell,
    sqrt(0.81^2 + sigma^2/(2m) + sigma^2/4). Two nouns per cell are used if that gives power of at least 0.8 for a
    1-nat difference (two-sided alpha 0.05/6, ten seeds), otherwise three.
14. Test details, fixed in scripts/amendment9_tests.py before any data: slopes are least-squares slopes over the cell
    means of one model and are then averaged over sizes within seed; M1 passes in a base corpus when the mean knowledge
    at 1,000 exposures is at least 0.2 nats and its slope is above zero (one-sided t, p < .05); H31 compares the shape-
    absent cells at 1,000 and 10,000 exposures (pooled) with the unseen nouns under the typical embedding, both in grouped
    Regular models; H33 is supported only if both parts are, so its p value in the Holm family is the larger of the two;
    the mixed models use crossed random intercepts for seed and noun. Only final models enter a test (saved
    intermediate weights are exploratory). A hypothesis is supported when its Holm threshold is met and its direction
    holds at two of the three sizes (for H33, both parts). If M1 fails, H30 and H31 are reported but not interpreted.
    H35 enters the Holm family only if H33(a) has an unadjusted p below .05 ("given H33").
15. Item and circuit details: scripts/familiarity_items.py, scripts/familiarity_eval.py and
    scripts/circuit_analysis.py (candidate heads from 100 sequences of 25 distinct base-language words repeated once;
    200 random draws for unseen nouns, 20 for trained kinds).
16. Head selection after the pilot. The registered rule (the k heads with the highest product of prefix-matching and
    copying scores) is kept whatever the pilot shows. If the candidates' prefix-matching scores are not clearly above
    the level of uniform attention (about 1/37 for these sequences), the candidates are reported as copying heads, not
    induction heads, and no claim about induction is made.

## Pilot result (23 October 2026, before any confirmatory corpus was built or model trained)
- The three pilot corpora built on the training PC matched the checksums recorded beforehand
  (docs/amendment9_corpus_manifest_pilot.json) and passed every corpus check. The three 3.4M pilot models trained in
  4.4 minutes each. All scores, head analyses and logs are in results/amendment9/ (commit d9823de).
- Nouns per cell (item 13). The standard deviation of carry between nouns of a cell in the grouped Regular pilot model
  was 0.634 nats. The expected seed-level standard deviation of the H31 contrast is 0.926 nats with two nouns per cell
  and 0.907 with three, giving power 0.55 and 0.57 for a 1-nat difference. Neither reaches 0.8, so three nouns per cell
  are used and the shortfall is reported. Almost all of the expected standard deviation comes from the assumed
  seed-level term (0.81 nats from Amendment 8); the registered ten seeds are kept.
- Manipulation check. Knowledge for consistent nouns in the grouped Regular pilot model was 0.41, 9.88, 10.33 and 11.43
  nats at 10, 100, 1,000 and 10,000 exposures.
- Heads (item 16). In all three pilot models the highest prefix-matching score was at most 0.028, the level of uniform
  attention, and the candidates were layer-0 heads. The candidates are therefore described as copying heads.
- The pilot results were seen before the confirmatory models were trained. No hypothesis, test, threshold or rule was
  changed because of them; the only decision taken from the pilot is the number of nouns per cell, by the rule fixed in
  item 13. Confirmatory corpus checksums were recorded before training (docs/amendment9_corpus_manifest.json).
