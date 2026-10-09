# Study 2 extension (Amendment 5, Part B): results

Scored 9 Oct 2026 on the Mac (Python 3.12, PyTorch 2.13.0, transformers 5.14.1, huggingface-hub 1.33.0; 70B models with MLX 0.32.3). Locked analysis code: `scripts/study2_extension.py` (MD5 6920386db826f7927080d3e75c60f0cc, as in `docs/amendment5_checksums.txt`).

## Data

| File | Models | Items | Lines |
|---|---|---|---|
| `scores_final.csv.gz` | 9 Pythia + 2 BabyLM, final | all four sets (4,608) | 50,689 |
| `scores_final_large.csv.gz` | Qwen2.5-72B, Llama-3.1-70B | set M (1,152) | 2,305 |
| `scores_dense.csv.gz` | Pythia 410M to 12B, 76 checkpoints | set M | 87,553 |
| `scores_poly.csv.gz` | PolyPythia seeds 1 to 9 (14M to 410M) and seed 0 of 31M, 4 checkpoints | set M | 211,969 |

MD5s of the uncompressed files: `scores_md5.txt`. No model failed. Running `analyse` on the .gz files gives byte-identical outputs.

## Registered tests (set M: "Here is a snerg." vs "Here is a brell."; Holm across three)

| Test | Result | Holm p | Verdict |
|---|---|---|---|
| H14a kind-specific index rises with Pythia size (9 sizes, final) | rho = -0.42 | 1.00 | Not supported |
| H14b pooled Pythia 410M to 12B, final, above zero (crossed mixed model, converged) | -0.15 (SE 0.12); pigeonhole bootstrap -0.12 [-0.50, 0.25] | 1.00 | Not supported |
| H15 410M > 160M across 10 seeds, step 143,000 | 0.15 vs -0.15, t = 3.04; 7 of 10 seeds at 410M positive | .012 | Supported |

The registration said: "If the set M kind-specific index is near zero where the original index was positive, the original result reflected the word 'another', and the manuscript will say so." That is what happened.

## Secondary (`secondary_*.csv`)

- Final checkpoints, set M (kind-specific index, 95% CI over nouns): Pythia 410M -0.01, 1B 0.16 [-0.10, 0.41], 1.4B -0.31, 2.8B -0.28, 6.9B -0.11, 12B -0.19; Qwen2.5-72B 0.56 [0.28, 0.84]; Llama-3.1-70B -0.16 [-0.36, 0.06]; BabyLM 0.06 and 0.05.
- Set M2 (both continuations with "another"): positive in Pythia 410M (0.28), 1B (0.77), 2.8B (0.30), 6.9B (0.17), near zero in 1.4B and 12B. With "another" the same-noun continuation repeats shape and changes colour (colour component negative); with "a" both shape and colour repeat more after the same noun (colour component positive), as if the same object were described again.
- Sets F and W: near zero or mixed in every model.
- Over training (S15, set M): Pythia 1B was positive from step 6,000 to 32,000 (0.22 to 0.74, intervals above zero at 14 of 18 checkpoints) and fell to 0.16 (n.s.) at the end; 2.8B and 1.4B showed shorter positive stretches. No size has an onset by the registered definition (interval above zero at every later checkpoint), for the kind-specific or the raw index.
- PolyPythia (set M, final): 410M mean 0.17 (range -0.20 to 0.61); 160M -0.14; 14M to 70M near zero.
- Raw shape-over-colour index on set M still rises with size (14M -3.13, 410M -0.75, 1B -0.11, 6.9B 0.46, 12B 0.27; Llama 0.53, Qwen -0.21), but is smaller than on the original items.

## Deviations (scoring only)
huggingface-hub pinned to 1.x; 70B models: weights read before conversion, wired limit 170 GB, batch 4 with a 0.05 s pause (see the session 20 log). None changes the computation; batch 1 vs batch 4 agreed to 0.001 nats.

## Reproduce
```
python scripts/study2_extension.py analyse --scores results/study2x/scores_final.csv.gz \
  results/study2x/scores_final_large.csv.gz results/study2x/scores_dense.csv.gz \
  results/study2x/scores_poly.csv.gz --out results/study2x/tests
python scripts/study2x_secondary.py --summary results/study2x/tests_summary.csv --out results/study2x/secondary
```
