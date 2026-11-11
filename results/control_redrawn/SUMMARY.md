# Redrawn-burst control (exploratory; not pre-registered)

Models: 30 (tiny 10, small 10, medium 10); 10 seeds with every size. Seed-level: sizes averaged within seed; t-based 95% intervals.

## Copy score by history (nats)

- Changing, bursts: 100: 4.09 [3.68, 4.49] (tiny 7.33, small 2.32, medium 2.61); 1000: 3.82 [3.21, 4.44] (tiny 6.09, small 2.59, medium 2.78)
- Half-kept, bursts: 100: 2.99 [2.57, 3.40] (tiny 4.78, small 1.91, medium 2.27); 1000: 1.20 [0.92, 1.47] (tiny 1.33, small 1.13, medium 1.13)
- Redrawn, bursts: 100: 2.28 [1.94, 2.62] (tiny 2.99, small 1.81, medium 2.03); 1000: 0.49 [0.41, 0.58] (tiny 0.28, small 0.55, medium 0.65)
- Fixed, bursts: 100: 3.18 [2.75, 3.62] (tiny 5.15, small 1.84, medium 2.56); 1000: 0.82 [0.68, 0.95] (tiny 0.48, small 0.98, medium 0.99)
- Changing, spaced: 100: 1.32 [1.11, 1.54] (tiny 1.44, small 1.08, medium 1.44); 1000: 0.42 [0.38, 0.47] (tiny 0.17, small 0.46, medium 0.64)
- Fixed, spaced: 100: 1.22 [1.07, 1.37] (tiny 0.66, small 1.18, medium 1.82); 1000: 0.39 [0.32, 0.46] (tiny 0.12, small 0.56, medium 0.48)
- Unseen: 4.92 [4.59, 5.25]

## Contrasts (paired within model, nats)

- Changing, bursts minus Redrawn, bursts: 100: 1.81 [1.33, 2.30] (tiny 4.34, small 0.52, medium 0.58); 1000: 3.33 [2.74, 3.92] (tiny 5.82, small 2.04, medium 2.14)
- Changing, bursts minus Half-kept, bursts: 100: 1.10 [0.70, 1.50] (tiny 2.55, small 0.42, medium 0.34); 1000: 2.63 [2.13, 3.12] (tiny 4.76, small 1.46, medium 1.65)
- Half-kept, bursts minus Redrawn, bursts: 100: 0.71 [0.36, 1.05] (tiny 1.79, small 0.10, medium 0.24); 1000: 0.71 [0.46, 0.95] (tiny 1.05, small 0.58, medium 0.48)
- Redrawn, bursts minus Changing, spaced: 100: 0.96 [0.76, 1.15] (tiny 1.55, small 0.73, medium 0.59); 1000: 0.07 [-0.01, 0.15] (tiny 0.11, small 0.09, medium 0.01)
- Redrawn, bursts minus Fixed, bursts: 100: -0.91 [-1.38, -0.43] (tiny -2.16, small -0.04, medium -0.52); 1000: -0.32 [-0.47, -0.18] (tiny -0.20, small -0.43, medium -0.35)
- Changing, bursts minus Fixed, bursts: 100: 0.90 [0.58, 1.22] (tiny 2.18, small 0.48, medium 0.06); 1000: 3.01 [2.42, 3.59] (tiny 5.62, small 1.62, medium 1.79)

## Fall from 100 to 1,000 exposures (nats)

- Changing, bursts: 0.26 [-0.06, 0.59]
- Half-kept, bursts: 1.79 [1.45, 2.13]
- Redrawn, bursts: 1.79 [1.45, 2.12]
- Fixed, bursts: 2.37 [1.93, 2.81]
- Changing, spaced: 0.90 [0.68, 1.12]
- Fixed, spaced: 0.83 [0.71, 0.95]
