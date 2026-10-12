# Local input order shapes what language models learn about object kinds

Jon-Paul Cacioli, Independent researcher

Oct 9, 2026 · @JP

## Abstract

Children come to expect that a new object with a new name will share its shape with other members of its kind, an overhypothesis about kinds in general. We asked whether next-word prediction can learn a text-based version of this expectation, and what the input must look like. After one described exemplar of a novel noun, does a language model expect the next exemplar to keep its shape more than its colour, more so for the same noun, and more than any other property?

In Study 1, small transformers trained on synthetic corpora learned a weak preference from sentences in random order (0.47 nats) and a strong one from the same sentences grouped into runs about one kind (6.49 nats). Grouping helped only when one dimension stayed fixed within kinds, and the models then preferred that dimension. The gain rose steadily with clustering: at the level found in speech to children, the preference was three times that in random order. A longer training window did not help; adjacency did. The preference was not tied to the noun. In Study 2, Pythia models (14 million to 12 billion parameters) preferred shape over colour more with size, but with matched wording none reliably tied the preference to the kind, and none preferred shape over texture. In Study 3, object labels recurred nearby 3.6 times as often as expected in speech to children under three, against 2.3 in adult conversation and 1.4 in written text. All tests were pre-registered.

## 1. Introduction

A two-year-old who hears "this is a dax" about a new object will apply "dax" to other objects of the same shape, even when they differ in size or texture (Landau, Smith, & Jones, 1988). This shape bias is not present from the start. It grows as children learn count nouns, and teaching toddlers a few shape-based categories makes it appear earlier and speeds their vocabulary growth (Smith, Jones, Landau, Gershkoff-Stowe, & Samuelson, 2002). The bias therefore looks learned. What is learned is not a fact about any one kind. It is a fact about kinds in general: things with the same name tend to share a shape. Kemp, Perfors and Tenenbaum (2007) called this an overhypothesis, borrowing Goodman's (1955) term, and showed that a hierarchical Bayesian learner acquires it from a handful of examples.

Language models learn from text alone, by predicting the next word. They use far more text than any child hears, roughly four to five orders of magnitude more (Frank, 2023). Two questions follow. Can next-word prediction learn a text-based version of the shape bias at all? And what does the input need to look like? The second question is about data. If structure in the input can make up for some of the amount, then how pretraining data is arranged matters, not only how much of it there is.

A shape bias makes a prediction about text. If a learner reads one description of a new kind of object, it should expect the next object of that kind to share the first one's shape more than its colour. We measure this expectation from a model's probabilities, at three levels:

1. **Shape over colour.** After one exemplar, the model expects a new exemplar to keep its shape more than its colour.
2. **Tied to the kind.** This expectation is stronger when the next exemplar has the same name than when it has a different name.
3. **Shape over everything else.** Shape is preferred over every other property, as children prefer it for solid objects.

Only the second and third levels describe an overhypothesis about named kinds. The first can arise from a simpler habit, such as expecting any next object to resemble the last one.

Three studies address these questions. Study 1 asks, with small transformers trained on controlled synthetic corpora, whether a model learns the regularity when every kind keeps its shape, and whether the order of the text matters when its content is fixed. Study 2 asks whether models trained on ordinary English show the expectation, and how it depends on model size, amount of training and training data closer to a child's. Study 3 measures how much of the local order that helped in Study 1 natural corpora contain: speech to young children, conversation between adults, and the written text language models learn from. Each study was registered before its models were scored or its transcripts read, and registered extensions test the main alternative explanations.

In small transformers, local order decided how much a fixed corpus taught. The effect needed a regularity in the corpus, followed whichever dimension was stable, and grew steadily with how clustered the input was. Speech to children was more clustered than adult conversation or written text, and at its level of clustering our models learned three times as much as from random order. A training window four times longer did not help: what mattered was that exemplars of a kind came next to each other, not that they shared a window. Models trained on large-scale English reached the first level with scale, and none reliably reached the second once the wording of the test was matched. Chan et al. (2022) showed that bursty input makes in-context learning emerge. Our results concern a different outcome: an expectation about which properties members of a kind share, stored in the weights and measured after a single exemplar. They also place natural input on the same scale as the manipulation.

Our claims are about transformers. The shape bias supplies the test, as it did for image classifiers in Ritter et al. (2017), and speech to children is one natural corpus among others, not the thing being explained. Section 7.4 sets out what the results predict for children, as hypotheses to test.

## 2. Background

### 2.1 The shape bias as a learned overhypothesis

Landau et al. (1988) showed 2- and 3-year-olds and adults a named novel object, then asked which other objects had the same name. All groups weighted shape more than size or texture. The bias grew from age two to three, and in children it was stronger when the object was named than when it was not. It also depends on what kind of thing is named. Two-year-olds extend the name of a solid object by shape, but the name of a non-solid substance by its material (Soja, Carey, & Spelke, 1991). The bias tracks the statistics of the words children learn. Most early count nouns name solid things in categories organised by shape, and both connectionist models and toddlers trained on such vocabularies come to attend to shape (Samuelson, 2002). Colunga and Smith (2005) showed that an associative network trained on the structure of early noun vocabularies acquires these kind-specific expectations.

The training study of Smith et al. (2002) is the closest analogue to our Study 1. Seventeen-month-olds were taught four novel categories, each organised by shape, in weekly play sessions. They then extended new names by shape, and their vocabulary of object names grew faster than that of control children. Children taught the same objects in categories organised by colour or texture, or the same shape categories without names, did not show these gains.

Kemp et al. (2007) made the idea precise. In their hierarchical Bayesian model, each kind has its own distribution over features, and a higher level describes how variable any kind tends to be on each dimension. After several kinds that are uniform in shape but varied in colour, the model expects a brand-new kind to be uniform in shape too, after a single exemplar. We use their model as a reference point for how strongly each of our corpora supports the regularity.

### 2.2 The order of children's input

Children do not hear sentences in random order. Speech to children often comes in variation sets, runs of nearby utterances that repeat and vary talk about the same thing (Küntay & Slobin, 1996). Adults learning an artificial language did better when the input contained variation sets than when the same sentences were scrambled (Onnis, Waterfall, & Edelman, 2008). Parents' talk about objects during play is mostly clustered in time, and clustered naming of a novel object predicted whether toddlers learned its name (Slone, Abney, Smith, & Yu, 2023). Word recurrence is also clustered in written text: a word that has appeared once in a document is far more likely to appear again than its overall frequency predicts (Church, 2000). Haga, Fukatsu, Oba, Bisazza and Oseki (2024) added generated variation sets to the child-directed speech used to train small language models. The benefit varied across benchmarks, and keeping each set together was often no better than shuffling. Feng, Goodman and Frank (2024) trained language models on child-directed speech and found that its local properties affected what the models learned, while its global developmental order did not.

The order of training data also matters for transformers. Chan et al. (2022) trained transformers on image sequences and found that in-context learning, the ability to use examples within the current input, emerged only when items of a class appeared in bursts. Placing related documents in the same context window during pretraining also improves the use of information within the context (Shi et al., 2024).

### 2.3 Neural networks and the shape bias

Neural networks can show a shape bias. Ritter, Barrett, Santoro and Botvinick (2017) found one in image classifiers, with strength that varied across training runs. Feinman and Lake (2018) trained simple networks on artificial objects and found that a shape bias emerged after only a few examples from each of a few categories. Both studies used perceptual input. We ask whether the regularity can be learned from descriptions in text.

Language models learn words in an order driven mainly by frequency, more so than children do (Chang & Bergen, 2022). The BabyLM Challenge asks what models can learn from about as much text as a child hears, 10 to 100 million words (Warstadt et al., 2023). The Pythia suite released models of many sizes with checkpoints saved throughout training, on the same data in the same order (Biderman et al., 2023), and PolyPythias added nine further training seeds for the five smallest sizes (van der Wal et al., 2025). Together they let us ask when in training an ability appears, how it depends on size, and how much it varies between training runs.

## 3. Measuring the expectation

Studies 1 and 2 use the same test. Each item gives the model a short context describing one exemplar of a novel noun, with a shape, a colour and a texture. It then compares two continuations describing another exemplar. They differ on one dimension only. In one, that dimension repeats the value from the context. In the other, it takes a new value. Here is an item from Study 2, probing shape:

> Context: *Here is a snerg. It is cylindrical. It is purple. It is smooth.* Repeat: *Here is another snerg. It is **cylindrical**. It is turquoise. It is fluffy.* New: *Here is another snerg. It is **circular**. It is turquoise. It is fluffy.*

The model scores each continuation by the sum of the log probabilities of its tokens given the context. The difference is its preference for repeating that dimension, in nats:

```latex
\Delta_{d} = \log P(\text{repeat on } d \mid \text{context}) - \log P(\text{new on } d \mid \text{context})
```

The two continuations differ in a single word, so the difference also includes how readily the model produces that word. Every model tends to repeat words it has just seen. The quantity of interest is how much more it repeats shape than another dimension:

```latex
\mathrm{OI}_{\text{colour}} = \Delta_{\text{shape}} - \Delta_{\text{colour}}
```

A positive shape-over-colour index means the model expects shape to stay the same more than colour. General copying, and anything else that affects all three dimensions alike, cancels out. The shape-over-texture index is defined in the same way. With one exemplar, nothing in the context shows which dimension is stable, so a shape preference must come from training. Contexts with two or four exemplars, in which one dimension is held constant, test whether the model also uses evidence within the context.

**Tied to the kind.** Every item is also scored with a different novel noun in the continuation (*Here is a zull…*). The kind-specific index is the index after the same noun minus the index after a different noun. Because the same feature words appear under both nouns, it removes effects of the particular words. A model that expects any next object to share the last one's shape, whatever it is called, scores zero.

**Reference points.** An untrained model, with the same architecture and random weights, shows what the index gives without learning. In Study 1, the hierarchical Bayesian model of Kemp et al. (2007), fitted to each training corpus and scored on the same items, shows how strongly the corpus supports the regularity. It sees the corpus as a set of exemplars, not as a sequence, so it is a reference point rather than a limit a sequence model could reach. Its index after a different noun is zero, because it treats a different name as a different kind.

**Statistics.** In Study 1, each trained model gives one value. The three model sizes trained on one corpus seed share that corpus, so every contrast is also estimated with a mixed model with a random intercept for corpus seed, and at the level of the five seeds. In Studies 2 and 3, confidence intervals come from resampling whole nouns or whole documents. Each study's main tests were registered in advance, one-sided unless stated, with Holm (1979) correction across the tests of that study. The smallest effect we registered as meaningful is 0.2 nats, and equivalence to zero was tested against that bound (Lakens, 2017).

## 4. Study 1: Small models trained on synthetic corpora

Study 1 asks whether a transformer learns the regularity when its input contains nothing else, and whether the order of that input matters. We used corpora in which we control every regularity, and models small enough to train many times.

### 4.1 Method

**Corpora.** Each corpus describes 32 kinds of object, each named by a novel noun (*wug*, *dax*, *blorp* and so on), with 14 exemplars per kind. Each exemplar has a shape, a colour and a texture, each a novel word from a vocabulary of ten per dimension. Sentences use five frames:

1. *A wug is a \[shape\] \[colour\] \[texture\] thing.*
2. *Look at the wug. It is \[shape\] and \[colour\].*
3. *Here is another wug. This one is \[shape\].*
4. *The wug I showed you is \[shape\], but a different \[colour\].*
5. *That is a wug because it is \[shape\].*

Shape appears in all five frames, colour in three and texture in one. Every comparison below is therefore between corpora that share the frames.

In the **Regular** corpus, every kind keeps one shape across its 14 exemplars, while colour and texture vary. In the **Scrambled** corpus, each kind has one stable word drawn from the shape vocabulary, but for about half the kinds it sits in the texture position. Colour never stays fixed. The 448 sentences of a corpus are repeated in 26 freshly shuffled passes, about 102,000 words in all. Five corpora were generated per condition, one per seed. The random-order corpora and models were made for an earlier registered study (Cacioli, 2026), whose model scores were affected by the loading error described in Section 8.

**Grouped order.** For each Regular and Scrambled corpus we built a version with the same sentences in a different order. Within each pass, each kind's exemplars were shuffled, cut into runs of four, and the runs were shuffled across kinds. Every sentence occurs exactly as often as before. In random order, about 330 of the 11,648 pairs of neighbouring sentences describe the same kind. In grouped order, about 8,400 do, and a 128-token training window holds about three runs (Figure 1).

&#91;embedded content: Figure 1. The order manipulation of Study 1. Kind names and order are illustrative; the proportions are from the training corpora.\]

**Models and training.** We trained decoder-only transformers of three sizes: 3.4 million parameters (4 layers, width 256), 11.0 million (6 layers, width 384) and 25.6 million (8 layers, width 512), for 5,000, 8,000 or 10,000 steps, with 128-token windows, AdamW and a cosine learning-rate schedule. Each size was trained once on each of the five corpus seeds of a condition, giving 15 models per condition. The tokenizer is a byte-pair encoding with 351 entries, built on whitespace-separated words. Checkpoints were saved every 500 steps. The model code ties the input embedding to the output layer, but the tie was lost when the models were moved to the graphics card, so all models were trained with separate input and output embeddings; the parameter counts include both. Every model is scored as it was trained (Section 8).

**Test items.** Eight novel nouns that never occur in training (*snerg*, *twick* and so on) appear in the first frame above, for example *A snerg is a krannish pulek skorli thing*. With 25 draws of features per noun, one, two or four exemplars, three probed dimensions and two continuation nouns, there are 3,600 items. In the two- and four-exemplar contexts, shape is the dimension held constant.

**Registered tests.** For random order we predicted that Regular models would show a shape-over-colour index above zero, that it would exceed that of Scrambled models, and that it would be tied to the named kind. For grouped order we predicted that grouping would raise the index for Regular models, and that grouped Regular models would exceed grouped Scrambled models on both the shape-over-colour and the shape-over-texture index. Tests were one-sided t tests across the 15 models per condition (Welch tests between conditions), with Holm correction within each family, and every effect was refitted as a mixed model with a random intercept for corpus seed.

**Registered extension.** Grouping could raise the index without the regularity, through the frames or a habit of repeating the previous shape. Amendment 5 therefore added two corpora built from Regular. In **NoStable**, every exemplar gets its own shape word, so no dimension is stable within a kind. In **ColourStable**, shape varies in the same way and each kind keeps one colour. Each is trained in random and grouped order. Four further Regular corpora vary only how clustered the order is: B = 1.4, 3.6, 7 and 12 on the measure of Study 3, against 0.9 for random order and 23 for grouped order. The levels 1.4 and 3.6 match written text and speech to children (Section 6). All 120 new models are trained as above. All Study 1 models are also scored on items whose nouns occurred in training, to test whether the models tie the preference to the noun when they can tell nouns apart.

### 4.2 Results

| Corpus | Order | Shape over colour | Shape over texture | Kind-specific | Bayesian reference, shape over colour |
| --- | --- | --- | --- | --- | --- |
| Regular | random | 0.47 (0.90) | 0.51 (0.74) | 0.01 | 6.25 |
| Scrambled | random | −0.25 (0.72) | −0.54 (0.90) | −0.05 | 1.15 |
| Regular | grouped | 6.49 (1.49) | 5.36 (1.28) | −0.05 | 6.25 |
| Scrambled | grouped | 2.12 (1.07) | 0.46 (1.14) | 0.05 | 1.15 |
| Untrained |  | 0.02 (0.07) | 0.01 (0.07) |  |  |

Table 1. Indices in nats after one exemplar, mean (SD) across 15 models. Kind-specific is the shape-over-colour index after the same noun minus after a different noun; no condition exceeds 0.05 in absolute value. The Bayesian reference model's shape-over-texture index is 5.85 for Regular and 0.54 for Scrambled.

**Random order.** Regular models showed a shape-over-colour index of 0.47 nats, with a wide spread across models (SD 0.90). It was not reliably above zero after correction (p = .031 uncorrected, Holm p = .061), nor small enough to be called equivalent to zero. It was above the index of Scrambled models (−0.25; Holm p = .032). The index was the same after the same noun and after a different noun (difference 0.005 nats, equivalent to zero, p = .001). Models trained on the Feature-swap control, in which half the kinds keep their texture instead of their shape, did not differ from Regular models in their preference for texture (shape over texture 0.60 against 0.51; p = .73).

**Grouped order.** The same sentences in grouped order raised the Regular shape-over-colour index to 6.49 nats (t = 13.4, Holm p < .001, d = 4.9; mixed model 6.02, SE 0.27). The rise held in all 15 matched models and all five corpus seeds, and it was about the same at every model size (interaction p > .6). Grouped Regular models also exceeded grouped Scrambled models on shape over colour (2.12; Holm p < .001, d = 3.4; mixed model 4.37, SE 0.43) and on shape over texture (5.36 against 0.46; Holm p < .001, d = 4.1), again in all five seeds. Grouped Scrambled models matched the pattern their corpus supports. Colour is never stable there, so shape over colour is correct, but shape and texture are equally often stable, and these models showed no reliable preference between them (0.46, p = .14).

Grouping also raised the index in Scrambled models, by 2.37 nats. The part of the grouped Regular index that depends on every kind keeping its shape is at least the difference between grouped Regular and grouped Scrambled, 4.37 nats. The registered extension tests this directly with the NoStable control (Section 4.3).

The preference was still not tied to the noun. After grouped training, the index was the same whichever noun the continuation used.

**More exemplars.** With two and four exemplars of the same shape in the context, the index of grouped Regular models was lower than with one (5.24 and 4.30 nats), and that of random-order Regular models was near or below zero (−0.64 and −0.10). More exemplars in the context did not add to what the weights had learned.

**Over training.** Figure 2 shows the index at every saved checkpoint. Grouped models reached 80% of their final level within 1,000 steps (tiny and small models) or 3,000 steps (medium) and held it. Random-order models first moved towards colour, to about −1.5 nats, and reached 0.3 to 0.7 by the end of training, after many passes through the same sentences. The kind-specific index stayed within 0.2 nats of zero at all 461 checkpoints.

&#91;embedded content: Figure 2. Study 1, Regular corpora in random and grouped order; mean of five models per size at each of 461 saved checkpoints.\]

### 4.3 Extension: stable dimensions and graded clustering

**Registered tests.** Grouping did nothing without a stable dimension. Grouped NoStable models had a shape-over-colour index of −0.44 nats, against 6.49 for grouped Regular models (H11; mixed model 6.93 \[6.30, 7.56\], d = 6.2, Holm p < .001; higher in all 15 matched models). When colour was the stable dimension, grouping produced a preference for colour. Grouped ColourStable models had an index of −3.20, well below grouped NoStable (H12; 2.76 \[2.20, 3.32\], d = 3.0, Holm p < .001). Across the six Regular orders, the index rose with the logarithm of B (H13a; slope 1.63 nats, SE 0.10, Holm p < .001). At B = 3.6, the level of speech to children in Study 3, the index was three times that in random order (H13b; 1.43 against 0.47 nats, difference 0.96 \[0.59, 1.32\], d = 1.04, Holm p < .001; higher in all 15 matched models). Welch, paired and seed-level tests agreed with the mixed models in every case.

| Corpus | Order | B | Shape over colour | Models above zero | Bayesian reference |
| --- | --- | --- | --- | --- | --- |
| Regular | random | 0.9 | 0.47 (0.90) | 10 | 6.25 |
| Regular | graded | 1.4 | 0.83 (0.72) | 12 | 6.25 |
| Regular | graded | 3.6 | 1.43 (0.93) | 15 | 6.25 |
| Regular | graded | 7.0 | 2.06 (0.95) | 15 | 6.25 |
| Regular | graded | 12.1 | 3.53 (1.12) | 15 | 6.25 |
| Regular | grouped | 23.1 | 6.49 (1.49) | 15 | 6.25 |
| NoStable | random | 0.9 | −0.18 (0.39) | 5 | −0.11 |
| NoStable | grouped | 23.1 | −0.44 (0.54) | 4 | −0.11 |
| ColourStable | random | 0.9 | −0.42 (0.70) | 4 | −6.17 |
| ColourStable | grouped | 23.1 | −3.20 (1.18) | 0 | −6.17 |

Table 2. Study 1 extension. Shape-over-colour index in nats after one exemplar of a novel noun, mean (SD) across 15 models. B is the measured clustering of each corpus (Section 6). The Bayesian reference model gives the same value for every order of a corpus.

**What grouping teaches.** The components of the index show where it comes from. In grouped Regular models, a repeated shape gained 5.36 nats and a repeated colour lost 1.13. In grouped ColourStable models the pattern reversed: a repeated colour gained 2.39 nats and a repeated shape lost 0.81. In grouped NoStable models, neither shape nor colour was favoured (−0.64 and −0.20). Grouping therefore did not teach a general habit of copying features from the context. It taught the models to keep the dimension that stayed fixed within a kind in training and to let the others vary, and they applied this to nouns they had never seen. The Bayesian reference model reached the same answer for every order of a corpus. The information was in every Regular corpus, but the transformers used it far more when examples of a kind came close together.

**How much clustering.** The rise was steep and steady (Table 2, Figure 5): 0.47, 0.83, 1.43, 2.06, 3.53 and 6.49 nats from random to grouped order. In an exploratory refit without the grouped corpus, the slope was still 1.06 nats per unit of log B (SE 0.08, p < .001), so the trend does not rest on the grouped end point. Grouping helped models of every size about equally.

**Trained nouns.** To ask whether the models tie the preference to the noun when they can tell nouns apart, we also scored every Study 1 model on items whose nouns occurred in training. No condition gave a kind-specific index reliably above zero after correction for the 15 conditions tested. The largest was 0.30 nats, in grouped Regular models (p = .008 uncorrected); the next were 0.18 at B = 7 (p = .15) and 0.12 at B = 3.6 (p = .22). With novel nouns, the kind-specific index was within 0.12 nats of zero in every condition.

### 4.4 Window test

**Method.** Study 1 models are trained on 128-token windows taken at random offsets from the running text, about 12 sentences each. If grouping helps only by putting exemplars of a kind into the same window, a longer window should let random order teach the expectation too. Amendment 6, registered after the results above, trained 45 further models exactly as before but with 512-token windows and a quarter of the batch, so that each step saw the same number of tokens: Regular in random order, Regular at B = 3.6, and NoStable in random order. Before training, we measured for every corpus the share of sentences in a window that follow an earlier sentence about the same kind. With 512 tokens, random order reaches a share of 0.47, more than the B = 12 corpus with 128 tokens (0.40). We registered a prediction from this share, fitted to the scores of the 90 Regular models of Sections 4.2 and 4.3. Those scores were later found to be affected by the loading error described in Section 8, so we report the prediction refitted to the corrected scores, as exploratory: 4.3 nats for random order and 4.6 at B = 3.6.

**Results.** The longer window did not help. Random-order models gave 0.50 nats, against 0.47 with 128 tokens (H17; difference 0.03 \[−0.30, 0.36\], Holm p = .87), though the difference was not small enough to be called equivalent to zero (p = .15). At B = 3.6 the longer window gave less: 0.80 against 1.43 (H18; −0.63 \[−1.00, −0.26\], Holm p = 1.00). Both fell far below the refitted prediction (Figure 5). The small preference that random-order models learned depended on the stable dimension: with 512 tokens, Regular models exceeded NoStable models (0.50 against 0.10; H19, 0.40 \[0.04, 0.77\], Holm p = .045). With trained nouns, the kind-specific index was again not above zero.

The gain from grouping therefore did not come from how many exemplars of a kind shared a training window. It came from exemplars of a kind arriving next to each other.

**Summary.** Grouping descriptions of a kind into runs taught small models a strong shape-over-colour expectation after one exemplar, which they applied to unseen nouns. With the order randomised, the same sentences taught much less. Grouping helped only when one dimension stayed fixed within each kind, and the models then preferred whichever dimension that was. The gain grew steadily with how clustered the order was, and clustering at the level measured in speech to children already tripled it. A longer training window did not help: what mattered was that exemplars of a kind came next to each other. The expectation was not tied to the noun, so these models reached the first level only.

## 5. Study 2: Models trained on ordinary English

Study 2 asks whether models trained on large-scale English text show the expectation, at which level, and how it depends on model size, the amount of training, and training data closer in size to a child's.

### 5.1 Method

**Items.** The items follow the format in Section 3. Each feature gets its own sentence, so English adjective order cannot favour one dimension, and the order of the three sentences is counterbalanced. Each dimension uses eight English words with similar word frequencies (mean Zipf frequency 3.75 for shape, 3.82 for colour and 3.70 for texture):

| Dimension | Words |
| --- | --- |
| Shape | pointed, oval, circular, curved, rectangular, triangular, spherical, cylindrical |
| Colour | grey, purple, amber, violet, scarlet, crimson, maroon, turquoise |
| Texture | smooth, shiny, sticky, fuzzy, fluffy, furry, glossy, silky |

The nouns are eight novel words (*snerg*, *twick*, *vulf*, *wazz*, *glorp*, *yeff*, *zull*, *kleb*), spelled with ordinary subword pieces the models have learned. With one, two or four exemplars, 12 draws per noun, three probed dimensions and two continuation nouns, there are 2,880 items. In these registered items the same-noun continuation began "Here is another snerg." and the different-noun continuation "Here is a zull." The kind-specific index therefore compares the noun and the word "another" together. The registered extension removes this difference (below).

**Models.** The Pythia models (Biderman et al., 2023) range from 14 million to 12 billion parameters and were all trained on the same 300 billion tokens of the Pile (Gao et al., 2020) in the same order. We scored seven sizes at ten checkpoints each, from initialisation (step 0) to the end of training (step 143,000; one step is about 2.1 million tokens), and the 6.9- and 12-billion-parameter models at four checkpoints. We also scored the two GPT-2 baseline models released by the BabyLM Challenge organisers, with 124 million parameters, trained on its 10- and 100-million-word corpora, much of which is transcribed speech, including speech to children (Warstadt et al., 2023). Two large models trained on far more text, Qwen2.5 with 72 billion parameters (Qwen Team, 2024) and Llama 3.1 with 70 billion (Grattafiori et al., 2024), were registered as descriptive. Models above one billion parameters were scored at half precision.

**Registered tests.** We predicted that the shape-over-colour index at the end of training would rise with Pythia model size (Spearman correlation across nine sizes), that it would rise over training within each size (mean within-size correlation, tested by permuting checkpoints), and that each BabyLM baseline would show an index above zero (bootstrap over nouns), with Holm correction across the four tests. The texture index, the kind-specific index, the use of evidence with more exemplars and the onset over training were registered as secondary analyses.

**Registered extension.** Amendment 5 adds 16 nouns and four new item sets in which the same-noun and different-noun continuations use the same wording: "Here is a snerg." against "Here is a brell." (primary); both with "another"; a second frame ("I found a snerg. The snerg is oval. …"); and a second, frequency-matched word list. All Study 2 models are scored at their final checkpoint. Pythia 410M to 2.8B are scored at every 1,000 steps between 2,000 and 15,000 and at five later checkpoints, and nine further training seeds of each of the five smallest sizes (van der Wal et al., 2025) are scored at four checkpoints. The registered tests are that the kind-specific index rises with Pythia size, that it is above zero when pooled over Pythia 410M to 12B (a mixed model with crossed random effects for nouns, items and feature words), and that it is higher at 410M than at 160M across ten training seeds.

### 5.2 Results

**Shape over colour.** The shape-over-colour index rose with model size (Spearman ρ = .92, Holm p = .003), from −2.44 nats at 14 million parameters to 1.17 at 12 billion, and the result held when any one noun was left out (ρ = .92 to .95). It did not rise steadily over training (mean within-size ρ = −.34). At every size, training first pushed the index towards colour, to about −2 nats between steps 1,000 and 4,000, and only the larger models then recovered. The BabyLM baselines did not show a reliable index (10 million words, 0.47 \[−1.05, 1.93\]; 100 million words, 0.37 \[−0.04, 0.73\]; Holm p = .52 and .12). Their intervals do not distinguish a small positive effect from zero. These baselines have 124 million parameters, and Pythia models below 410 million parameters did not show the expectation even after 300 billion tokens, so the BabyLM result says little about the amount of training data.

**Words matter.** How readily a model repeats a feature word depends on the word. In the largest models, repeating *smooth* was preferred by 4.3 nats on average but *silky* by only 0.6, and among colours the range was 0.5 (*maroon*) to 3.6 (*purple*). Part of each raw comparison is therefore about the words, not the dimensions. Untrained Pythia models give a raw shape-over-colour index of about 0.5 nats at every size, because the test words are split into different numbers of tokens. Word effects in trained models are learned, and may differ between sizes. The size result above should therefore also be checked on a measure free of them.

**Tied to the kind.** The kind-specific index compares the same words after the same and a different noun, so word effects cancel; it is within 0.03 nats of zero in untrained models. On the original items it rose with size (exploratory: ρ = .72, permutation p = .02). Every Pythia model from 410 million parameters up gave values above zero (0.27 to 0.51 nats; Figure 3), and in the larger models shape repeated more and colour less after the same noun, as a learner that represents kinds as stable in shape should show. Over training, the index turned negative by step 4,000 and was positive by step 16,000 in models of 1.4 billion parameters and more. These items, however, put "another" only in the same-noun continuation. Section 5.3 shows that the values depend on that word.

&#91;embedded content: Figure 3. Study 2, kind-specific index at the final checkpoint, on the original items (only the same-noun continuation contained "another") and with matched wording (Section 5.3); 95% bootstrap intervals over nouns (8 original, 16 matched).\]

**Onset.** The registered onset analysis used the raw index: the first checkpoint from which its interval stays above zero. On that definition the shape-over-colour index appeared only at the final checkpoint (about 300 billion tokens), in models of 1 billion parameters and more.

**Texture.** Shape did not win over texture. On the raw index every trained model preferred to keep texture over shape. On the kind-specific index, shape and texture were tied to the kind about equally in most models, and texture more than shape in the two largest Pythia models (6.9 billion, −0.27; 12 billion, −0.36; both intervals below zero). No model reached the third level.

**More exemplars.** When two or four exemplars in the context shared a shape, or shared a colour, every trained model followed the context: the index was higher when shape was the shared dimension. The difference was large in Pythia models from 410 million parameters (7.7 to 9.6 nats with two exemplars, 12.6 to 15.2 with four), smaller in the three smallest Pythia models and the BabyLM baselines (2.0 to 3.3 and 4.4 to 8.0), and under 1 nat in untrained models. What developed slowly with size and training was the expectation after one exemplar, not the use of evidence in the context.

### 5.3 Extension: matched wording, training seeds and dense checkpoints

**Matched wording.** In the items of Section 5.2, the same-noun continuation began "Here is another snerg." and the different-noun continuation "Here is a zull." The extension's primary items use the same wording for both ("Here is a snerg." against "Here is a brell."), with 16 nouns. Three further sets use "another" in both continuations, a second frame ("I found a snerg. The snerg is oval."), and a second, frequency-matched word list. All models were scored at their final checkpoint, Pythia 410M to 2.8B at every 1,000 steps from 2,000 to 15,000 and at later checkpoints to 32,000, the 6.9B and 12B models at four checkpoints, and nine further training seeds of the five smallest Pythia sizes at four checkpoints.

**Registered tests.** With matched wording, the kind-specific index did not rise with Pythia size (ρ = −.42, Holm p = 1.00), and it was not above zero when pooled over Pythia 410M to 12B (−0.15 nats, SE 0.12, Holm p = 1.00; crossed bootstrap −0.12 \[−0.50, 0.25\]). It was higher at 410M than at 160M across ten training seeds of each (0.15 against −0.15 nats; t = 3.04, Holm p = .012), but only 7 of the 10 seeds at 410M were positive (range −0.20 to 0.61). As the registration stated in advance, the positive kind-specific index on the original items therefore depended on the word "another".

**What the index depends on (exploratory).** At the final checkpoint, with matched wording (Figure 3), the index was −0.01 nats in Pythia 410M, 0.16 \[−0.10, 0.41\] in 1B, −0.31 in 1.4B, −0.28 in 2.8B, −0.11 in 6.9B and −0.19 in 12B. Qwen gave 0.56 \[0.28, 0.84\] and Llama −0.16 \[−0.36, 0.06\]. When both continuations began "Here is another", the index was positive in Pythia 410M (0.28), 1B (0.77), 2.8B (0.30) and 6.9B (0.17), and near zero in 1.4B and 12B. The components show why. After "another", a model that expects a new member of the same kind should keep the shape and change the colour, and the larger models did: colour repeated less after the same noun than after a different one. After "a", the same noun may name the same object again, and then every feature should repeat. Colour then repeated more after the same noun, and the index fell. On the second frame and the second word list, the index was near zero or mixed in every model.

**Over training.** On the matched items, Pythia 1B showed a positive kind-specific index from step 6,000 to step 32,000 (0.22 to 0.74 nats, with intervals above zero at 14 of 18 checkpoints), which fell to 0.16, with an interval including zero, by the end of training. The 1.4B and 2.8B models showed shorter positive stretches (Figure 4). No model met the registered onset criterion, an interval above zero at every later checkpoint, on either index.

&#91;embedded content: Figure 4. Study 2, kind-specific index with matched wording across Pythia training (Section 5.3); 410M to 2.8B at 18 checkpoints, 6.9B and 12B at five.\]

**Shape over colour.** On the matched items the raw shape-over-colour index still rose with size, from −3.13 nats at 14 million parameters to 0.46 at 6.9 billion and 0.27 at 12 billion, but it was smaller than on the original items.

**Summary.** Models trained on large-scale English preferred shape over colour more with size, the first level. On the original items, models from 410 million parameters also preferred it more after the same noun, but those items confounded the noun with the word "another". With matched wording the kind-specific index was not above zero and did not rise with size. It was positive when both continuations said "another", and in the middle of training in some models. No model preferred shape over texture. Ordinary training at this scale gave the first level, and no stable second level.

## 6. Study 3: Local recurrence in speech to children

Study 1 showed that grouping descriptions of a kind into runs lets a small model learn the regularity. Study 3 measures how much of this local recurrence natural corpora contain, on the scale of Study 1. Speech to young children is a natural place to look, because clustered talk about objects is well documented there (Section 2.2). Written text is what language models are trained on.

### 6.1 Method

**Speech to children.** We used the North American English collection of CHILDES (MacWhinney, 2000), downloaded in full from TalkBank, and kept every transcript whose target child was under 36 months (3,134 transcripts). From each we took every utterance not spoken by the target child. Transcripts of fewer than 400 words were excluded, as registered, leaving 2,116 transcripts from 35 corpora.

**Written text.** For comparison we used the first 10,000 documents of the Pile (Gao et al., 2020), the training data of the Pythia models, split into sentences. After the same 400-word rule, 2,474 documents remained.

**Target kinds.** We counted mentions of 70 nouns from the MacArthur-Bates Communicative Development Inventory (Fenson et al., 2007), the checklist of words parents report their children know. We chose solid, inanimate objects named by count nouns, such as *ball*, *cup*, *shoe* and *truck*, the kinds for which children show the shape bias. Singular and plural forms count as the same kind. We left out words often used as verbs or adjectives in speech to children, such as *brush* and *orange*.

**Measure.** We took every pair of consecutive mentions of target nouns that fell in different utterances or sentences and no more than 100 words apart; 100 words is about the length of the 128-token training window in Study 1. The measure is the share of these pairs that repeat the same label. A document about one topic has a high share for that reason alone, so the share is compared with the share in the same document after shuffling its utterances or sentences, which keeps the words and the topic but removes local order. Burstiness, B, is the observed share divided by the shuffled share. B = 1 means a label recurs no more locally than the document's content predicts. Confidence intervals come from resampling whole documents. The measure counts repeated labels. It does not show whether the two mentions refer to two objects or one.

**Registered tests.** We predicted that B in speech to children would exceed 1 and would exceed B in the Pile, one-sided with Holm correction. To put the values on the scale of Study 1, we applied the same measure to the Study 1 corpora.

**Registered extension.** Amendment 5 adds adult conversation from TalkBank's CABank (the Santa Barbara Corpus of Spoken American English, CallHome English and CallFriend English), processed in the same way, with a two-sided test against speech to children. It also adds the rate of same-label pairs per 1,000 words, alternative weightings, shuffling only within 500-word blocks, all common nouns, automatic cues of distinct objects and described properties, and hand coding of 500 random same-label pairs for whether they refer to the same object or to different objects of the kind.

### 6.2 Results

The transcripts held 74,488 mentions of the target nouns and the Pile documents 32,808. Both predictions were supported. In speech to children, consecutive mentions repeated the same label 53% of the time, against 15% after shuffling, so B = 3.57 \[3.46, 3.68\] (Holm p < .001). In the Pile, B = 1.43 \[1.28, 1.63\]. The difference, 2.14 \[1.92, 2.34\], was reliable (Holm p < .001). Table 3 sets these values beside the Study 1 corpora.

| Input | B \[95% CI\] |
| --- | --- |
| Study 1 corpus, random order | 0.92 \[0.89, 0.95\] |
| Written text (Pile) | 1.43 \[1.28, 1.63\] |
| Speech to children (CHILDES) | 3.57 \[3.46, 3.68\] |
| Speech to children, parents only | 3.47 \[3.36, 3.58\] |
| Study 1 corpus, grouped order | 23.3 \[23.0, 23.7\] |

Table 3. Local recurrence of object labels, within 100 words. B = 1 is the level expected from the content of each document without local order. The random-order Study 1 corpus falls slightly below 1 because each pass contains every exemplar once, so a kind is slightly less likely to follow itself than under a full shuffle.

**Rates.** B is a ratio, and the two corpora differ in how often target nouns occur at all. The raw share of same-label pairs was in fact higher in the Pile (60%) than in speech to children (53%), because a Pile document that mentions these nouns usually mentions few of them. Per 1,000 words, however, speech to children contained 10.1 same-label pairs and the Pile 1.0. Of these, 7.2 and 0.45 per 1,000 words were in excess of the shuffled baseline (exploratory). Per word of input, a child hears about ten times as many nearby repeated object labels as a language model reads in the Pile, and about sixteen times as many beyond what the content of the input predicts.

**Robustness.** B in speech to children was 4.20 within 25 words and 3.39 within 200, while the Pile stayed between 1.41 and 1.43. It was 3.23 for children under 18 months, 3.89 at 18 to 24 months and 3.78 at 24 to 36 months, and similar for mothers and fathers alone. All 64 nouns with at least 100 pairs had B above 1, from 2.38 for *toy* to 11.65 for *coat* (median 4.81). Weighting each transcript equally gave B = 2.91 (Pile 1.31), and each CHILDES corpus equally 3.56. Resampling corpora and then transcripts gave a 95% interval of \[3.13, 4.28\], and all 33 corpora with enough pairs had B above 1 (exploratory).

**Register.** Part of the gap may come from conversation in general. Technical text in the Pile was the least clustered (Wikipedia 1.30, GitHub 1.10). Fiction and film dialogue were more clustered (Books3, 2.29 from 9 documents; film subtitles, 6.8 from only 30 pairs). These samples are too small to compare, which is why the extension adds adult conversation.

### 6.3 Extension: adult conversation, density and repeated objects

**Adult conversation.** Amendment 5 added 277 transcripts of adult conversation from CABank: the Santa Barbara Corpus of Spoken American English (Du Bois & Englebretson, 2004), CallHome English (Canavan et al., 1997) and CallFriend English, northern and southern (Canavan & Zipperlen, 1996a, 1996b). The files were downloaded from TalkBank on 9 October 2026; their checksums are in the repository. After the 400-word rule, 245 transcripts remained, with 720,895 words and 1,040 target pairs. Adult conversation was more clustered than written text, B = 2.33 \[2.01, 2.79\], but less clustered than speech to children. The registered two-sided test found a difference of 1.23 \[0.77, 1.57\] in favour of speech to children (p < .001). Part of the gap between children's input and the Pile is therefore a gap between speech and writing. Speech to children is still clearly more clustered than speech between adults (Table 4).

| Measure | Adult conversation (CABank) | Speech to children (CHILDES) | Written text (Pile) |
| --- | --- | --- | --- |
| B \[95% CI\] | 2.33 \[1.99, 2.94\] | 3.57 \[3.13, 4.28\] | 1.43 \[1.22, 1.85\] |
| B, each transcript weighted equally | 2.27 | 2.91 | 1.31 |
| Sources with B above 1 | 4 of 4 | 33 of 33 | 18 of 18 |
| Same-label pairs per 1,000 words | 0.94 | 10.1 | 1.0 |
| Beyond the shuffled baseline, per 1,000 words | 0.67 | 7.2 | 0.45 |
| B, shuffling only within 500-word blocks | 1.28 | 2.08 | 1.16 |
| B, 1,000-word segments | 1.55 | 2.69 | 1.22 |
| B, all common nouns | 5.73 | 9.65 | 2.07 |
| Pairs with a cue of another object | 66% | 39% | 66% |
| Pairs with a property word | 21% | 15% | 24% |

Table 4. Study 3 extension (Amendment 5). B is local recurrence within 100 words; B = 1 means no clustering beyond the content of each document. The 95% interval resamples sources (corpora, or subsets of the Pile) and then documents within them. A cue of another object is "another" or "other", a number, or a plural form of the noun between the two mentions. Property words name a colour, shape, size or material.

**Density.** Adult conversation had about as many same-label pairs per 1,000 words as written text (0.94 against 1.0) and about a tenth as many as speech to children (10.1). Beyond the shuffled baseline the rates were 0.67, 0.45 and 7.2. Per word, a child hears about ten times as many nearby repeated object labels as adults hear from each other.

**Local order only.** Shuffling a whole document also removes changes of topic across it, so part of B could reflect topic rather than local runs. Shuffling only within blocks of about 500 words keeps the topic at that scale. B fell in every corpus, to 2.08 in speech to children, 1.28 in adult conversation and 1.16 in written text, so part of B does come from changes of topic. The order of the three stayed the same. Cutting every document into 1,000-word segments, which removes differences in document length, gave 2.69, 1.55 and 1.22.

**All nouns.** Counting every common noun instead of our 70 kinds (lemmas from spaCy; Honnibal et al., 2020) gave B = 9.65 in speech to children, 5.73 in adult conversation and 2.07 in written text. The pattern does not depend on the noun list.

**Cues of a second object.** B counts repeated labels. It cannot tell two balls from one ball named twice. As a first, automatic check we looked between the two mentions of each same-label pair for "another" or "other", a number, a plural form of the noun, and property words. In speech to children 39% of pairs had a cue of another object, against 66% in adult conversation and in written text. Property words appeared in 15%, 21% and 24% of pairs. Repeated labels in speech to children thus less often come with a sign of a second object or a described property. This fits talk that returns to the same object. For a next-word learner the difference matters: recurrence helps only if it brings different exemplars of a kind, because one object named twice shows no variation within the kind (Section 7.1). The hand coding below tests it directly.

**Transcription marks (exploratory).** The registered cleaning step splits a word at an overlap bracket or a lengthening colon inside it (do:g becomes do and g). Such marks are common in conversation transcripts: 2.4% of words in the Santa Barbara corpus. Removing them first changed B in adult conversation from 2.33 to 2.34 and left speech to children at 3.57.

**Reproduction.** Run again with newer software, the counts for speech to children and the Pile were identical to those behind Section 6.2.

**Hand coding.** The author coded 500 random same-label pairs (300 from speech to children, 100 from adult conversation, 100 from written text) without knowing their source: whether the two mentions refer to the same object, to different objects of the kind, or to the kind in general, and whether a property of the object is stated between them. A label not used for a physical object (Apple the company, the periodic table) was coded unclear. A second coder coded a random quarter of the pairs, also blind to source and to the first coder's codes. The second coder was an independent instance of Claude (Anthropic) given only the coding rules and the pairs; its codes were stored unopened until the first coder had finished. Agreement on 124 pairs was 84% for the referent (Cohen's κ = 0.78) and 95% for properties. One pair, discussed during coding, was left out of this check.

| Code | Speech to children | Adult conversation | Written text |
| --- | --- | --- | --- |
| Different objects | 27% \[23, 33\] | 25% \[18, 34\] | 13% \[8, 21\] |
| Same object | 46% \[41, 52\] | 25% \[18, 34\] | 12% \[7, 20\] |
| The kind in general | 17% \[14, 22\] | 28% \[20, 38\] | 32% \[24, 42\] |
| Unclear | 9% | 22% | 43% |
| Not used for an object (within unclear) | 3% | 17% | 43% |
| A property stated | 10% \[7, 14\] | 9% \[5, 16\] | 2% \[1, 7\] |

Table 5. What repeated object labels refer to: hand coding of 300, 100 and 100 random same-label pairs. Wilson 95% intervals in brackets.

In speech to children, nearly half of repeated labels referred again to the same object, and about a quarter brought a different object of the kind, as often as in adult conversation. The automatic cues therefore overstated the difference between the two: the share of repetitions that bring a second object is similar. In written text, most repeated labels did not refer to a physical object or referred to the kind in general. What sets speech to children apart is density. Combining these shares with the rates in Table 4, speech to children contains about 2.8 nearby repetitions per 1,000 words that bring a different object of the kind, against about 0.2 in adult conversation and 0.1 in written text (exploratory). Stated properties were rare in every corpus (2% to 10%).

**Summary.** Relative to what the content predicts, object labels recur locally in speech to young children about two and a half times as much as in the written text language models learn from. Per word, the excess is about sixteen times larger. This is far less clustered than the grouped corpus of Study 1. Whether the recurrences bring distinct exemplars, and whether the clustering is special to speech to children, are tested in the extension.

## 7. General discussion

We asked whether next-word prediction can learn a text-based version of the shape bias, and what the input must look like. Small models trained on controlled corpora learned to prefer shape over colour after one exemplar, strongly when descriptions of a kind came close together and every kind kept its shape. They applied this to whatever object came next, whatever it was called. Models trained on large-scale English learned to prefer shape over colour with scale, but tied the preference to the named kind only when the wording marked a new object, and none preferred shape over texture. Object labels recur locally in speech to young children more than in adult conversation or written text, and that level of recurrence was enough to triple what our small models learned. What mattered was that exemplars of a kind came next to each other.

### 7.1 Why local order matters

The random and grouped corpora of Study 1 contain the same sentences, each as often. The Bayesian reference model, which treats a corpus as a set of exemplars, extracts the same regularity from both. A language model is different. It learns by predicting each word from the words before it, and a kind's stable shape can help that prediction only if an earlier exemplar of the same kind is in view. Our first account was that grouping works by putting such exemplars into the training window. In random order, only about 15% of the sentences in a 128-token window follow an earlier sentence about the same kind; in grouped order, about two thirds do. The window test rules this account out (Section 4.4). With a 512-token window, random order put an earlier exemplar of the same kind in view for nearly half the sentences, more than the B = 12 corpus did with 128 tokens, yet the index did not rise. What mattered was that exemplars of a kind came next to each other.

The extension shows that grouping does not work on its own. Grouping a corpus in which no dimension is stable within kinds gave no preference. Grouping a corpus in which colour is stable gave a preference for colour (Section 4.3). So the models did not learn a general habit of copying the previous object. They learned which dimension stays fixed within a kind, from whatever their corpus offered, and used it for nouns they had never seen. The benefit grew steadily with clustering, and at the level we measured in speech to children the index was already three times that in random order. A training window four times longer did not add to this (Section 4.4).

&#91;embedded content: Figure 5. Study 1 index by the clustering of the training corpus (Sections 4.2 to 4.4): 128-token windows (filled) and 512-token windows (hollow), and the clustering of natural corpora on the same measure (Section 6; shaded bands and bars: 95% intervals resampling sources, then documents).\]

Figure 5 puts Studies 1 and 3 on one scale. Written text and adult conversation fall where the gain was small; speech to children falls where it was already clear. A window four times longer did not raise the index, and at B = 3.6 it lowered it.

This fits the finding that transformers acquire in-context learning only when items of a class arrive in bursts (Chan et al., 2022). It qualifies the finding that pretraining on related documents in one window improves the use of context (Shi et al., 2024): in our models, sharing a window added nothing to what was learned. One possibility is that small models of this kind rely mostly on the last sentence or two when they predict, so they learn a regularity mainly when the previous sentence carries it. With one exemplar in the context, nothing in the context says that shape is the stable dimension, so the grouped models' preference comes from their weights. Evidence that was only ever available within a context became a prior.

Within the amount of training tested here, more passes over randomly ordered input did not produce what grouped input taught within 1,000 to 3,000 steps. We did not vary the number of kinds or exemplars, which is what amount means for the Bayesian learner, and in large-scale English the amount of training clearly mattered. The claim is therefore that local order changes what a fixed amount of input can teach, not that amount is unimportant. Tokenisation cannot explain the order effect, because the random and grouped versions of each corpus use the same tokenizer, sentences and test items.

### 7.2 A preference for the next object, or for the named kind

The synthetic models learned that the next object tends to keep the last one's shape, whatever it is called. Two things may explain this. In grouped corpora the next object usually is of the same kind, so "keep the last shape" predicts almost as well as "keep the shape of the last object with this name". And the test nouns never occurred in training, so their input embeddings were never trained, and the models have no basis for treating two of them as different kinds. The trained-noun items of the extension separate these explanations. With nouns that occurred in training, the kind-specific index was not reliably above zero in any condition (Section 4.3), although grouped Regular models came closest (0.30 nats, p = .008 uncorrected). These models learned a preference that largely ignores the noun.

The nouns of Study 2 are spelled with familiar subword pieces, so the models can tell them apart, yet the kind-tied preference was fragile. With matched wording it was not reliably above zero at the end of training. It appeared when both continuations said "another", which tells the model that a second object of the same kind follows. With "a", the model may take the second mention to describe the same object again, and then colour should repeat too, which is what the components showed. It also appeared in the middle of training in some models and faded by the end. A model trained on ordinary text can represent that members of a kind share shape but not colour, but training did not make this a stable default. Neither the small models of Study 1, even with nouns they knew, nor the English-trained models reached a stable second level.

### 7.3 Shape and texture

Children extend names of solid objects by shape and names of substances by material (Soja et al., 1991). The English-trained models treated shape and texture as about equally stable within kinds, and the two largest Pythia models tied texture to the kind more than shape. Text may describe texture mainly when it is characteristic of a kind, as with the fur of an animal or the material of a fabric, and our items do not say whether the object is solid. What these models learned is closer to an expectation that kinds share physical properties but not colour. A shape bias in the strict sense may need perceptual input, or input that separates solid objects from substances.

### 7.4 Implications

Language models receive four to five orders of magnitude more language than children do (Frank, 2023). In Pythia models a reliable preference for shape over colour appeared only at the end of training, after about 300 billion tokens, and a kind-tied preference never became stable. That is far beyond the 100 million words of the largest BabyLM training set, which was chosen as a developmentally plausible amount (Warstadt et al., 2023), although tokens and words are not the same unit.

For language model training, Study 1 shows that the same content, presented in runs, was learned quickly, provided the content held a regularity. The benefit grew steadily with how clustered the input was, and the level found in speech to children already tripled it. This fits the finding that the local properties of child-directed speech, not its global order, affect what models learn from it (Feng et al., 2024). The local arrangement of examples in pretraining data deserves attention alongside their number. Natural corpora differ in how much of it they offer: object labels recur locally far more in speech to young children than in written training text (Study 3), and although about half of them return to the same object, per word they bring a different object of the kind about ten times as often as adult conversation and twenty times as often as written text (Section 6.3).

**Predictions for development.** Transformers learn differently from children, and from different input, so these results are not an account of how children acquire the shape bias. They do yield predictions that can be tested with children. If a sequence learner benefits from nearby exemplars that differ within a kind, then in a training study like that of Smith et al. (2002), naming several different members of a category close together should build the shape bias faster than naming one object repeatedly, or naming the same members spread out in time. Slone et al. (2023) found that clustered naming of one novel object predicted whether toddlers learned its name. Whether clustering across different objects of a kind predicts generalisation by shape is open. In our models the benefit grew steadily with clustering, so such a study could compare several levels of clustering, not only clustered against spread-out naming.

### 7.5 Limitations

Our models learn from text. Children learn object names while seeing and handling the objects, and their shape bias rests on perception as well as on words. Our tests ask only whether a regularity in descriptions can be learned from descriptions.

The synthetic world of Study 1 is simple. It has 32 kinds, three dimensions and five frames, and the frames mention shape more often than colour or texture, which is why we compare only corpora that share them. In Scrambled, the stable word of half the kinds is a shape word in the texture position, so Scrambled is not free of regularity; the NoStable control of the extension is, and grouping it taught nothing.

The graded corpora vary clustering in one way, by keeping or splitting runs of four sentences. Natural clustering takes other forms. The corpora hold about 100,000 words and the models see 128 tokens at a time. How much clustering at natural levels helps may therefore depend on the amount of input or on the learner. We did not vary the amount of input, the number of training steps, or model size beyond 26 million parameters.

In Study 2 the kind-specific index depended on wording, and only the five smallest Pythia sizes have several training seeds. The models' sensitivity to "another" suggests that items stating explicitly whether a second object is present would measure the second level more cleanly.

In Study 3, CHILDES transcripts come mostly from recordings of play and daily routines chosen by researchers, which may involve more talk about objects than a typical day. The 35 corpora differ greatly in size. The Pile is written text: the right comparison for language models, but not for adult conversation. And repeated labels may often refer to one object, which provides no evidence about how members of a kind vary.

## 8. Pre-registration, deviations and availability

**Pre-registration.** The project was registered on OSF before any model was trained, and all registrations are in one OSF project (osf.io/qj9hb). Each later study was added as an amendment, registered before its models were scored or its transcripts read. Each amendment was written after the results of earlier studies were known, and says so; Amendment 5 was also written after reviews of a draft of this paper and after the exploratory checks it lists. In the registrations the studies carry working names and the hypotheses short labels. Table 6 links them to this paper. Amendment 6 was written after the results of Amendment 5 and says so. The prediction it tests was fitted to those results before any new model was trained. For it, the training script gained an option for a longer window, which leaves earlier runs unchanged.

| In this paper | Registered tests | Registration |
| --- | --- | --- |
| Study 1, random order: index above zero, above Scrambled, tied to the kind | H1b, H2b, H3b | Amendment 1, "Study 1b" |
| Study 1, grouped order: grouping raises the index; grouped Regular above grouped Scrambled on colour and on texture | H4b, H5b | Amendments 1 and 2 (texture added in Amendment 2, before any grouped model was trained), "Study 1c" |
| Study 1, checkpoint tracking (Figure 2) | registered exploratory | Amendment 1 |
| Study 1 extension: NoStable and ColourStable controls, graded clustering | H11, H12, H13a, H13b; S14 (trained nouns) | Amendment 5 |
| Study 1 window test: 512-token training windows (Section 4.4) | H17, H18, H19; S24 to S28 | Amendment 6 |
| Study 2: model size, training, BabyLM baselines | H6, H7, H8 | Amendment 3, "Study 1d" |
| Study 2: texture, kind-specific index, more exemplars, onset | S5, S6, S7, S8 | Amendment 3 |
| Study 2 extension: matched wording, seeds, dense checkpoints | H14a, H14b, H15; S15 to S18 | Amendment 5 |
| Study 3: recurrence above 1, above written text | H9, H10 | Amendment 4, "Study 1e" |
| Study 3: Study 1 scale, windows, age, parents only, Pile components | S9 to S13 | Amendment 4 |
| Study 3 extension: adult conversation, rates, coding | H16; S19 to S23 | Amendment 5 |

Table 6. Registered tests and where they appear. All other analyses are exploratory and labelled so in the text.

**Deviations.** In Study 1, the Bayesian reference model as first computed counted each of the 26 repeated passes through a corpus as new exemplars; as promised in Amendment 1, we corrected it to count each exemplar once. In Study 2, four things departed from the registration. For the 2.8-billion-parameter Pythia model at steps 0 to 16,000, the public repository holds a stale copy of the final weights beside the correct ones, and our loader first read the stale copy; we found this from identical scores across checkpoints and rescored from the correct files (every affected checkpoint, before and after, is listed in the repository). The step 64,000 checkpoint of the same model is identical to the final one in the public repository and was dropped. The two 70-billion-parameter models were scored with Apple's MLX library rather than PyTorch, because PyTorch crashed the machine; on smaller models the two libraries differed by 0.003 to 0.010 nats on average. These models and some checks ran under a different software environment from the other models, which changed rescored values by 0.005 nats or less on average. In Study 3, the registered download address for CHILDES had moved, and the registered collection was downloaded from its current address. In the Amendment 5 and 6 extensions, six things departed from the registration. Scoring the 70-billion-parameter models needed changes to how their weights were loaded (read before conversion, a fixed memory limit, batches of four with a short pause), made in the scoring script after registration and only in its MLX code; batches of one and four gave scores within 0.001 nats. The library that downloads model weights was held at an earlier version, because the current one did not work with the model library. CABank now requires an account, so the files were downloaded from its corpus pages after signing in. In the hand coding, the rule for labels not used for a physical object was added to the guidance during coding and applied by both coders, and one pair of the reliability subset was discussed with an AI assistant during coding and left out of the agreement check. Finally, the equivalence tests for H13b (Amendment 5) and H17 (Amendment 6) were stated in the registration text but not in the locked analysis scripts, and were run with separate scripts.

**Correction.** The model code ties the input embedding to the output layer. On the graphics card we used (through DirectML), the tie was lost when each model was moved to the device, so all Study 1 models were trained with separate input and output embeddings. Our scoring code rebuilt each model with the tie in place before loading its weights, so the output weights overwrote the input embedding, and every Study 1 score we first computed came from a model with the wrong input embedding. We found this before submission, fixed the loader, and rescored all 315 trained models, the 461 saved checkpoints and the trained-noun items, with the same items and otherwise the same code. All Study 1 numbers in this paper come from the corrected scores; the repository keeps both. The correction made every Study 1 effect larger and changed the outcome of two registered tests: Regular above Scrambled in random order (H2b) and B = 3.6 above random order (H13b) became supported. A Feature-swap difference first reported as exploratory disappeared. The prediction registered in Amendment 6 had been fitted to the uncorrected scores, so we report it refitted to the corrected scores, as exploratory. The earlier preprint that used these models (Cacioli, 2026) was affected in the same way. Study 2, Study 3 and the Bayesian reference model were not affected.

**Availability.** All code, test items, synthetic corpora, model scores, transcript-level counts and analysis outputs are at github.com/synthiumjp/overhypothesis, where REPRODUCE.md lists the command that regenerates each number, with checksums for every input. The trained models of Study 1 will be archived with a DOI. The pretrained models of Study 2 are public, at the revisions listed in the repository. CHILDES and CABank transcripts are available from TalkBank under its terms and are not redistributed.

**Use of AI tools.** Claude (Anthropic) was used to write and check analysis code, run analyses on the released data, search for and check references, and draft and edit text. A separate instance of Claude, given only the coding rules and the pairs, was the blind second coder for the reliability check in Section 6.3. Apart from that coder, Claude had access to the data and results throughout. The author designed the studies, reviewed all code, analyses and text, and is responsible for them.

**Funding and competing interests.** None.

## References

Biderman, S., Schoelkopf, H., Anthony, Q., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., Skowron, A., Sutawika, L., & van der Wal, O. (2023). Pythia: A suite for analyzing large language models across training and scaling. In *Proceedings of the 40th International Conference on Machine Learning* (PMLR Vol. 202, pp. 2397–2430).

Cacioli, J.-P. (2026). *Exemplar retrieval without overhypothesis induction: Limits of distributional sequence learning in early word learning* (arXiv:2604.05243). arXiv.

Canavan, A., Graff, D., & Zipperlen, G. (1997). *CALLHOME American English speech* (LDC97S42) \[Data set\]. Linguistic Data Consortium.

Canavan, A., & Zipperlen, G. (1996a). *CALLFRIEND American English-Non-Southern Dialect* (LDC96S46) \[Data set\]. Linguistic Data Consortium.

Canavan, A., & Zipperlen, G. (1996b). *CALLFRIEND American English-Southern Dialect* (LDC96S47) \[Data set\]. Linguistic Data Consortium.

Chan, S. C. Y., Santoro, A., Lampinen, A. K., Wang, J. X., Singh, A. K., Richemond, P. H., McClelland, J. L., & Hill, F. (2022). Data distributional properties drive emergent in-context learning in transformers. In *Advances in Neural Information Processing Systems* (Vol. 35).

Chang, T. A., & Bergen, B. K. (2022). Word acquisition in neural language models. *Transactions of the Association for Computational Linguistics, 10*, 1–16. https://doi.org/10.1162/tacl\_a\_00444

Church, K. W. (2000). Empirical estimates of adaptation: The chance of two Noriegas is closer to p/2 than p². In *COLING 2000 Volume 1: The 18th International Conference on Computational Linguistics*. https://aclanthology.org/C00-1027

Colunga, E., & Smith, L. B. (2005). From the lexicon to expectations about kinds: A role for associative learning. *Psychological Review, 112*(2), 347–382.

Du Bois, J. W., & Englebretson, R. (2004). *The Santa Barbara Corpus of Spoken American English* \[Data set\]. TalkBank. https://doi.org/10.21415/T5VG6X

Feinman, R., & Lake, B. M. (2018). Learning inductive biases with simple neural networks. In *Proceedings of the 40th Annual Conference of the Cognitive Science Society*.

Feng, S. Y., Goodman, N. D., & Frank, M. C. (2024). Is child-directed speech effective training data for language models? In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (pp. 22055–22071). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.emnlp-main.1231

Fenson, L., Marchman, V. A., Thal, D. J., Dale, P. S., Reznick, J. S., & Bates, E. (2007). *MacArthur-Bates Communicative Development Inventories: User's guide and technical manual* (2nd ed.). Paul H. Brookes.

Frank, M. C. (2023). Bridging the data gap between children and large language models. *Trends in Cognitive Sciences, 27*(11), 990–992. https://doi.org/10.1016/j.tics.2023.08.007

Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N., Presser, S., & Leahy, C. (2020). *The Pile: An 800GB dataset of diverse text for language modeling* (arXiv:2101.00027). arXiv.

Goodman, N. (1955). *Fact, fiction, and forecast*. Harvard University Press.

Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., et al. (2024). *The Llama 3 herd of models* (arXiv:2407.21783). arXiv.

Haga, A., Fukatsu, A., Oba, M., Bisazza, A., & Oseki, Y. (2024). BabyLM Challenge: Exploring the effect of variation sets on language model training efficiency. In *The 2nd BabyLM Challenge at the 28th Conference on Computational Natural Language Learning* (pp. 252–261). Association for Computational Linguistics.

Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70.

Honnibal, M., Montani, I., Van Landeghem, S., & Boyd, A. (2020). *spaCy: Industrial-strength natural language processing in Python* \[Computer software\]. Zenodo. https://doi.org/10.5281/zenodo.1212303

Kemp, C., Perfors, A., & Tenenbaum, J. B. (2007). Learning overhypotheses with hierarchical Bayesian models. *Developmental Science, 10*(3), 307–321. https://doi.org/10.1111/j.1467-7687.2007.00585.x

Küntay, A., & Slobin, D. I. (1996). Listening to a Turkish mother: Some puzzles for acquisition. In D. I. Slobin, J. Gerhardt, A. Kyratzis, & J. Guo (Eds.), *Social interaction, social context, and language: Essays in honor of Susan Ervin-Tripp* (pp. 265–286). Lawrence Erlbaum.

Lakens, D. (2017). Equivalence tests: A practical primer for t tests, correlations, and meta-analyses. *Social Psychological and Personality Science, 8*(4), 355–362. https://doi.org/10.1177/1948550617697177

Landau, B., Smith, L. B., & Jones, S. S. (1988). The importance of shape in early lexical learning. *Cognitive Development, 3*, 299–321.

MacWhinney, B. (2000). *The CHILDES project: Tools for analyzing talk* (3rd ed.). Lawrence Erlbaum.

Onnis, L., Waterfall, H. R., & Edelman, S. (2008). Learn locally, act globally: Learning language from variation set cues. *Cognition, 109*(3), 423–430.

Qwen Team. (2024). *Qwen2.5 technical report* (arXiv:2412.15115). arXiv.

Ritter, S., Barrett, D. G. T., Santoro, A., & Botvinick, M. M. (2017). Cognitive psychology for deep neural networks: A shape bias case study. In *Proceedings of the 34th International Conference on Machine Learning* (PMLR Vol. 70, pp. 2940–2949).

Samuelson, L. K. (2002). Statistical regularities in vocabulary guide language acquisition in connectionist models and 15–20-month-olds. *Developmental Psychology, 38*(6), 1016–1037.

Shi, W., Min, S., Lomeli, M., Zhou, C., Li, M., Szilvasy, G., James, R., Lin, X. V., Smith, N. A., Zettlemoyer, L., Yih, S., & Lewis, M. (2024). In-context pretraining: Language modeling beyond document boundaries. In *The Twelfth International Conference on Learning Representations*.

Slone, L. K., Abney, D. H., Smith, L. B., & Yu, C. (2023). The temporal structure of parent talk to toddlers about objects. *Cognition, 230*, Article 105266. https://doi.org/10.1016/j.cognition.2022.105266

Smith, L. B., Jones, S. S., Landau, B., Gershkoff-Stowe, L., & Samuelson, L. (2002). Object name learning provides on-the-job training for attention. *Psychological Science, 13*(1), 13–19.

Soja, N. N., Carey, S., & Spelke, E. S. (1991). Ontological categories guide young children's inductions of word meaning: Object terms and substance terms. *Cognition, 38*, 179–211.

van der Wal, O., Lesci, P., Müller-Eberstein, M., Saphra, N., Schoelkopf, H., Zuidema, W., & Biderman, S. (2025). PolyPythias: Stability and outliers across fifty language model pre-training runs. In *The Thirteenth International Conference on Learning Representations*.

Warstadt, A., Mueller, A., Choshen, L., Wilcox, E., Zhuang, C., Ciro, J., Mosquera, R., Paranjabe, B., Williams, A., Linzen, T., & Cotterell, R. (2023). Findings of the BabyLM Challenge: Sample-efficient pretraining on developmentally plausible corpora. In *Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning* (pp. 1–34). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.conll-babylm.1
