# Study 1d results: natural-English test on pretrained models

Registered as Amendment 3 (osf.io/qcfjp) before any model was scored.
Data: `scores_merged.csv` (81 checkpoints). Registered tests: `scripts/study1d_tests.py`, output `tests_final.json`, `tests_final_secondary.csv`.

## Confirmatory tests (OI_colour, k = 1, matched noun; one-sided; Holm across 4)

| Test | Result | Statistic | Holm p | Verdict |
|---|---|---|---|---|
| H6 scale (Pythia final, 9 sizes) | rises with size | rho = .92 | .003 | Supported |
| H7 training (Pythia, within size) | dips then recovers | mean rho = -.34 | 1.00 | Not supported |
| H8a BabyLM 10M | 0.47 [-1.05, 1.93] | | .52 | Not supported |
| H8b BabyLM 100M | 0.37 [-0.04, 0.73] | | .12 | Not supported |

Neither BabyLM CI lies inside the 0.2-nat equivalence bounds, so H8 is inconclusive, not evidence of absence.

## Final checkpoints (95% cluster-bootstrap CI over nouns)

| Model | OI_colour | OI_texture | Noun binding |
|---|---|---|---|
| Pythia 14M | -2.44 [-2.94, -1.93] | -1.21 | -0.13 [-0.20, -0.06] |
| Pythia 70M | -1.17 [-1.44, -0.88] | -1.50 | -0.25 [-0.42, -0.06] |
| Pythia 160M | -1.22 [-1.68, -0.73] | -1.80 | -0.40 [-0.55, -0.22] |
| Pythia 410M | -0.13 [-0.48, 0.21] | -1.27 | 0.30 [0.11, 0.44] |
| Pythia 1B | 0.53 [0.15, 0.97] | -0.97 | 0.51 [0.30, 0.71] |
| Pythia 1.4B | 0.44 [0.21, 0.64] | -1.05 | 0.27 [0.18, 0.36] |
| Pythia 2.8B | 0.43 [0.25, 0.66] | -0.76 | 0.35 [0.24, 0.47] |
| Pythia 6.9B | 0.84 [0.50, 1.14] | -0.86 | 0.32 [0.16, 0.45] |
| Pythia 12B | 1.17 [0.79, 1.52] | -0.51 | 0.39 [0.17, 0.60] |
| Qwen2.5-72B | -0.39 [-0.90, 0.13] | -1.29 | 0.29 [0.02, 0.56] |
| Llama-3.1-70B | 1.22 [0.79, 1.64] | -0.23 | 0.27 [0.00, 0.55] |

## Key observations (secondary and descriptive)
- Untrained floor: step 0 gives OI_colour about 0.5 at every Pythia size (an item-level artefact shared by all models). Pythia 12B and Llama clearly exceed it; the 6.9B interval [0.50, 1.14] includes it.
- Training trajectory: every size dips towards colour at steps 1,000 to 4,000 (about -2 nats), then larger models recover.
- Texture: every Pythia, Qwen and Llama model prefers texture over shape (OI_texture < 0); Llama is closest to zero. The exception is the BabyLM baseline trained on 10M words, which prefers shape over texture (OI_texture 1.17 [0.24, 2.34]; `tests_final.json`, S5_texture H8a). (Corrected 13 Oct 2026; the earlier wording said every trained model.)
- Noun binding on the original items appears from 410M parameters; the three smallest Pythia models show reversed binding. (These items confound the noun with "another"; see `results/study2x/` for the matched-wording sets.)
- Large models disagree: Llama-3.1-70B matches Pythia 12B on shape over colour; Qwen2.5-72B does not. Scale within one training corpus does not predict other families.
- Onset (S8): only at step 143,000 (about 300B tokens), far beyond a child's input.

## Deviations from Amendment 3
1. pythia-2.8b, steps 0 to 16,000: the Hugging Face branches hold a stale copy of the final weights next to the correct shards, and the loader picked the stale file. Detected from identical scores; confirmed from file hashes; rescored from the shards.
2. pythia-2.8b step 64,000: byte-identical to step 143,000 on the Hub with no shards. Not recoverable; dropped.
3. Software environment: Pythia and BabyLM ran under Python 3.14 / PyTorch 2.14; the 70B models and checks under Python 3.12 / PyTorch 2.13. Re-scoring two checkpoints gave mean |difference| 0.0003 and 0.005 nats.
4. Qwen2.5-72B and Llama-3.1-70B were scored with Apple MLX, not PyTorch, after PyTorch MPS crashed the machine. Same tokenizer, padding and summation. Validation against PyTorch: mean |difference| 0.003 nats (Pythia 410M) and 0.010 nats (Qwen2.5-0.5B, same architecture as the 72B).
5. Precision: as registered (float16 for models over 1B). No invalid values in any model.


Note (13 Oct 2026): the stale test outputs (`tests.json`, `tests_v2*`, `tests_v3*`, `tests_secondary.csv`, `tests_stdout.txt`) were moved to `archive/`; `tests_final*` are the reported results.
