# Study 3 extension (Amendment 5, Part C): results

Run 9 Oct 2026 on the Mac (separate Python environment: spaCy 3.8.16, en_core_web_sm 3.8.0, numpy 2.5.3, pandas 3.0.6), with the locked code (commit c2bccb4).

## Data

CABank English, downloaded 9 Oct 2026 from TalkBank after signing in (TalkBank now requires a login for CABank; the CHILDES download did not). MD5s in `cabank_md5.txt`.

| Corpus | Page | Transcripts |
|---|---|---|
| SBCSAE | talkbank.org/ca/access/SBCSAE.html | 60 |
| CallHome English | talkbank.org/ca/access/CallHome/eng.html | 176 |
| CallFriend English (North) | talkbank.org/ca/access/CallFriend/eng-n.html | 31 |
| CallFriend English (South) | talkbank.org/ca/access/CallFriend/eng-s.html | 10 |

277 transcripts; 245 kept after the 400-word rule (720,895 words, 1,040 target pairs).

Reproducibility check: the CHILDES and Pile rows of `counts.csv` are identical to `results/study1e/counts.csv`, observed and shuffled counts alike. The synthetic shuffled counts differ because the new CABank input sits before them in the shared random stream; their observed counts are identical.

## Registered test H16 (two-sided)

| | B | 95% CI |
|---|---|---|
| CHILDES child input | 3.57 | |
| CABank adult conversation | 2.33 | [2.01, 2.79] |
| Difference | 1.23 | [0.77, 1.57] |

p < .001 (permutation floor 1/10,001). Child input is burstier than adult conversation.

## Rates and weightings (`rates.csv`, W = 100 words)

| | CABank | CHILDES | Pile |
|---|---|---|---|
| B, pair weighted | 2.33 | 3.57 | 1.43 |
| B, transcript weighted | 2.27 | 2.91 | 1.31 |
| B, mean over corpora | 2.53 | 3.56 | 2.08 |
| Two-stage CI (corpora, then documents) | [1.99, 2.94] | [3.13, 4.28] | [1.22, 1.85] |
| Corpora with B > 1 | 4/4 | 33/33 | 18/18 |
| Same-kind pairs per 1,000 words | 0.94 | 10.10 | 1.01 |
| Excess over shuffle per 1,000 words | 0.67 | 7.23 | 0.45 |

## Block shuffles (`blocks.csv`)

| Baseline | CABank | CHILDES | Pile |
|---|---|---|---|
| Shuffle within 500-word blocks | 1.28 | 2.08 | 1.16 |
| 1,000-word segments, each shuffled | 1.55 | 2.69 | 1.22 |

Part of the full-shuffle B comes from topic shifts across a transcript. Local clustering is still clearly larger in child input.

## All common nouns (`allnouns.csv`, spaCy lemmas)

B = 9.65 (CHILDES), 5.73 (CABank), 2.07 (Pile). The ordering holds beyond the 70 target kinds.

## Cues in same-kind pairs (`cues.csv`, share of pairs)

| Cue | CABank | CHILDES | Pile |
|---|---|---|---|
| "another" | .070 | .062 | .142 |
| numeral | .300 | .098 | .355 |
| plural | .521 | .312 | .439 |
| any property word | .211 | .146 | .236 |
| any cue of a distinct exemplar | .659 | .385 | .664 |

Pairs: CHILDES 32,218; Pile 7,025; CABank 681. Same-kind recurrences in child input carry cues of a different exemplar less often, which fits repeated talk about the same object. Hand coding of 500 pairs (`coding_sheet.csv`, kept out of git) will measure this directly.

## Exploratory: in-word transcription marks (`ca_sensitivity/`, not registered)

The registered cleaner splits words at in-word overlap brackets and lengthening colons (do:g, wal⌈k). Removing those marks first:

| | Registered | Marks removed |
|---|---|---|
| CABank B | 2.33 | 2.34 |
| CABank same-kind pairs per 1,000 words | 0.94 | 1.07 |
| CHILDES B | 3.57 | 3.57 |

No conclusion changes.

## Hand coding (S23; `coding_summary.csv`, `scripts/study3_coding_summary.py`)

First coder: JP, all 500 pairs, blind to source. Second coder: an independent Claude instance given only the rubric and 125 random pairs (seed 2031), sealed until the first coder finished. Agreement on 124 pairs (pair 0 excluded: discussed during coding): referent 84%, kappa 0.78; property 95%.

Share of pairs (Wilson 95% interval), first coder:

| Code | CHILDES (300) | CABank (100) | Pile (100) |
|---|---|---|---|
| Different objects | .27 [.23, .33] | .25 [.18, .34] | .13 [.08, .21] |
| Same object | .46 [.41, .52] | .25 [.18, .34] | .12 [.07, .20] |
| Kind in general | .17 [.14, .22] | .28 [.20, .38] | .32 [.24, .42] |
| Unclear | .09 | .22 | .43 |
| of which not an object | .03 | .17 | .43 |
| Stated property | .10 [.07, .14] | .09 [.05, .16] | .02 [.01, .07] |

The automatic cues overstated the gap: the share of repetitions that bring a different object is similar in CHILDES and CABank. Combined with the rates above (exploratory): different-object repetitions per 1,000 words, CHILDES 2.8, CABank 0.24, Pile 0.13.
