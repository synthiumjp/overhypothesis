# Study 1e results: burstiness of child-directed speech

Registered as Amendment 4 (osf.io/ya68j) before any transcript or adult text was read.
Code: `scripts/cds_burstiness.py` (MD5 `c4885f97af62e9d6cca59c523841065f`).
CHILDES source: `0-Eng-NA-MOR.zip` from talkbank.org/childes/access/Eng-NA/ (91 MB, MD5 `789a8815b2c222917b68be07b7246425`), downloaded 8 Oct 2026. Adult text: NeelNanda/pile-10k.

## Data
- CHILDES Eng-NA: 3,134 transcripts with a target child under 36 months (3.3M words of child input); 2,116 kept after the 400-word rule, from 36 corpora; 74,488 target-noun mentions.
- Pile sample: 9,363 documents prepared; 2,474 kept; 32,808 mentions.

## Registered tests (W = 100 words; one-sided; Holm across 2)

| Test | Result | 95% CI | Holm p | Verdict |
|---|---|---|---|---|
| H9 CHILDES B > 1 | B = 3.57 | [3.46, 3.68] | .0002 | Supported |
| H10 CHILDES B > Pile B | 3.57 vs 1.43, difference 2.14 | [1.92, 2.34] | .0002 | Supported |

## On the Study 1c scale (S9)

| Corpus | B |
|---|---|
| Synthetic, standard order (Study 1 Regular) | 0.92 [0.89, 0.95] |
| Pile sample (adult text) | 1.43 [1.28, 1.63] |
| CHILDES, child input | 3.57 [3.46, 3.68] |
| CHILDES, parents only (S12) | 3.47 [3.36, 3.58] |
| Synthetic, bursty order (Study 1c) | 23.3 [23.0, 23.7] |

Child input is about 2.5 times as bursty as adult text, but far below the corpus that taught the Study 1c models.

## Secondary
- S10 windows: CHILDES B = 4.20, 3.86, 3.57, 3.39 at W = 25, 50, 100, 200; Pile 1.41 to 1.43.
- S11 age: 3.23 (<18 months), 3.89 (18 to 24), 3.78 (24 to 36).
- S13 Pile components: technical text sits near 1.1 to 1.4 (GitHub 1.10, StackExchange 1.08, Pile-CC 1.27, Wikipedia 1.30). Narrative and conversational text is burstier (Books3 2.29 from 9 documents; OpenSubtitles 6.8 from 30 pairs). Part of the child-adult gap is register, not only speech to children.
- Per noun: all 64 CHILDES nouns with at least 100 pairs have B > 1 (median 4.81, range 2.38 for toy to 11.65 for coat). Pile median 1.59 (45 nouns with at least 30 pairs).
- R (repeat rate): CHILDES 0.53 vs shuffled 0.27 (BR 1.96); Pile 0.25 vs 0.14 (BR 1.78).

## Deviations
None. The download address in the run instructions was wrong (404); the registered source, the whole Eng-NA collection, was then downloaded from its current address.
