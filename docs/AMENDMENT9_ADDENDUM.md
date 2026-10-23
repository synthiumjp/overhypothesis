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
   the nearest free window; every move is listed in the corpus manifest.
7. Window boundaries. A familiarity sentence lies wholly inside one window as both inputs and targets: its first token
   is at least the window's second position, and its last token at most the window's last input position.
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
    The noun-to-cell rotation and each noun's s* are in docs/amendment9_assignment.json.

## Pilot result
(To be completed after the pilot and before any confirmatory model is trained.)
