# Experiment 3 mechanism analyses (exploratory; added after the results)

## Steering (change in copy score, nats)

- Fixed, spaced (+v1000): baseline 0.46; direction 1.51 [1.20, 1.82] (tiny 2.41, small 1.12, medium 1.00); random -0.02 [-0.11, 0.08]
- Changing, spaced (+v1000): baseline 0.57; direction 1.06 [0.76, 1.36] (tiny 1.81, small 0.84, medium 0.53); random -0.01 [-0.06, 0.04]
- No shape, spaced (+v1000): baseline 0.86; direction 2.04 [1.58, 2.49] (tiny 3.12, small 1.59, medium 1.40); random -0.01 [-0.16, 0.13]
- No shape, bursts (+v1000): baseline 1.05; direction 3.06 [2.45, 3.67] (tiny 4.60, small 2.18, medium 2.39); random 0.02 [-0.16, 0.19]
- Unseen, typical embedding (+v1000): baseline 0.79; direction 1.99 [1.55, 2.44] (tiny 3.13, small 1.37, medium 1.48); random -0.02 [-0.24, 0.19]
- Fixed, bursts (+v100): baseline 1.03; direction 0.99 [0.53, 1.44] (tiny 2.35, small 0.35, medium 0.26); random 0.07 [-0.04, 0.19]
- Changing, bursts (-v100): baseline 4.71; direction -0.82 [-1.37, -0.28] (tiny -2.27, small -0.22, medium 0.02); random -0.30 [-0.59, -0.01]

## Activation patching (copy score of the target; fraction of the source-minus-clean difference)

- R_into_F, tiny: clean 0.69, source 4.49; after layer -1: noun 1.00, last 0.00; after layer 0: noun 0.15, last 0.83; after layer 1: noun -0.01, last 1.03; after layer 2: noun -0.01, last 1.02; after layer 3: noun 0.00, last 1.00
- R_into_F, small: clean 1.09, source 2.31; after layer -1: noun 1.00, last 0.00; after layer 0: noun 0.58, last 0.48; after layer 1: noun 0.02, last 1.03; after layer 2: noun 0.01, last 1.01; after layer 3: noun 0.00, last 1.00; after layer 4: noun -0.00, last 1.01; after layer 5: noun 0.00, last 1.00
- R_into_F, medium: clean 1.32, source 2.41; after layer -1: noun 1.00, last 0.00; after layer 0: noun 0.32, last 0.96; after layer 1: noun -0.00, last 1.02; after layer 2: noun -0.00, last 0.97; after layer 3: noun -0.00, last 0.99; after layer 4: noun 0.00, last 0.99; after layer 5: noun 0.00, last 1.00; after layer 6: noun 0.00, last 1.00; after layer 7: noun 0.00, last 1.00
- F_into_R, tiny: clean 7.05, source 0.81; after layer -1: noun 1.00, last 0.00; after layer 0: noun 0.05, last 0.89; after layer 1: noun -0.02, last 1.01; after layer 2: noun -0.01, last 1.00; after layer 3: noun 0.00, last 1.00
- F_into_R, small: clean 3.53, source 1.38; after layer -1: noun 1.00, last -0.00; after layer 0: noun 0.38, last 0.47; after layer 1: noun -0.02, last 1.00; after layer 2: noun -0.00, last 1.01; after layer 3: noun -0.01, last 1.00; after layer 4: noun -0.00, last 1.00; after layer 5: noun -0.00, last 1.00
- F_into_R, medium: clean 3.56, source 1.72; after layer -1: noun 1.00, last -0.00; after layer 0: noun 0.06, last 0.54; after layer 1: noun -0.03, last 1.03; after layer 2: noun 0.00, last 1.03; after layer 3: noun -0.01, last 1.03; after layer 4: noun -0.00, last 1.01; after layer 5: noun -0.00, last 1.00; after layer 6: noun -0.00, last 1.00; after layer 7: noun -0.00, last 1.00

## Attention from the last position to the context's shape word

- Layer 0, changing minus fixed bursts at 1,000: 0.069 [0.048, 0.089]
- Layer 0: Spearman with copy score over cells x models 0.73 (n = 360)
- Layer 1: Spearman with copy score over cells x models 0.13 (n = 360)
- Layer 2: Spearman with copy score over cells x models 0.23 (n = 360)
- Layer 3: Spearman with copy score over cells x models 0.08 (n = 360)
- Layer 4: Spearman with copy score over cells x models 0.10 (n = 240)
- Layer 5: Spearman with copy score over cells x models -0.20 (n = 240)
- Layer 6: Spearman with copy score over cells x models -0.04 (n = 120)
- Layer 7: Spearman with copy score over cells x models -0.36 (n = 120)
