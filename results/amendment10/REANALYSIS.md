# Experiment 2 reanalysis (exploratory; added after the results)

## stable-shape corpora

### 100 exposures
| Noun | own | typical | gap | frame2 | colour | cos_init | att_l0 |
|---|---|---|---|---|---|---|---|
| Unseen | 6.35 |  | 0.16 | 4.38 | -0.30 | 1.000 | 0.186 |
| Fixed, spaced | 1.52 | 0.59 | 0.04 | 0.58 | -0.14 | 0.820 | 0.125 |
| Fixed, bursts | 4.43 | 0.53 | 0.21 | 2.21 | -0.20 | 0.942 | 0.163 |
| No shape, spaced | 3.09 | 0.66 | -0.08 | 2.18 | -0.59 | 0.885 | 0.139 |
| No shape, bursts | 4.65 | 0.66 | 0.22 | 3.14 | -0.47 | 0.951 | 0.148 |
| Changing, spaced | 1.59 | 0.57 | 0.08 | 0.60 | -0.14 | 0.868 | 0.142 |
| Changing, bursts | 5.02 | 0.71 | 0.18 | 2.74 | -0.21 | 0.960 | 0.183 |

- swap: changing bursts own minus with fixed-burst rows: 0.59 [0.24, 0.93]
- swap: fixed bursts with changing-burst rows minus own: 0.60 [0.26, 0.95]
- gap: changing bursts 0.18 [-0.01, 0.37]; minus fixed bursts -0.03 [-0.19, 0.13]; minus changing spaced 0.10 [-0.07, 0.26]
- frame2: changing bursts 2.74 [2.19, 3.28]; minus fixed bursts 0.53 [0.24, 0.82]; minus changing spaced 2.14 [1.63, 2.65]
- colour: changing bursts -0.21 [-0.28, -0.14]; minus fixed bursts -0.01 [-0.06, 0.03]; minus changing spaced -0.07 [-0.12, -0.02]
- cos_init, changing bursts minus fixed bursts: 0.018 [0.014, 0.023]
- layer-0 attention to the context shape, changing bursts minus fixed bursts: 0.020 [0.013, 0.027]
- geometry regression: changing-bursts coefficient -0.09 [-0.51, 0.33] (p = 0.68); observed 5.02, predicted from the other histories' geometry 4.83; geometry-only R2 0.40

### 1000 exposures
| Noun | own | typical | gap | frame2 | colour | cos_init | att_l0 |
|---|---|---|---|---|---|---|---|
| Unseen | 6.35 |  | 0.16 | 4.38 | -0.30 | 1.000 | 0.186 |
| Fixed, spaced | 0.52 | 0.67 | 0.05 | 0.12 | -0.10 | 0.732 | 0.089 |
| Fixed, bursts | 0.99 | 0.61 | 0.13 | 0.32 | -0.12 | 0.778 | 0.099 |
| No shape, spaced | 0.98 | 0.63 | -0.04 | 0.71 | -0.22 | 0.793 | 0.102 |
| No shape, bursts | 1.03 | 0.68 | -0.07 | 0.66 | -0.34 | 0.806 | 0.097 |
| Changing, spaced | 0.50 | 0.61 | 0.10 | 0.20 | -0.10 | 0.794 | 0.108 |
| Changing, bursts | 3.85 | 0.66 | 0.37 | 1.53 | -0.12 | 0.849 | 0.168 |

- swap: changing bursts own minus with fixed-burst rows: 2.88 [2.01, 3.76]
- swap: fixed bursts with changing-burst rows minus own: 2.88 [1.99, 3.76]
- gap: changing bursts 0.37 [0.22, 0.52]; minus fixed bursts 0.24 [0.12, 0.37]; minus changing spaced 0.28 [0.14, 0.41]
- frame2: changing bursts 1.53 [1.11, 1.95]; minus fixed bursts 1.21 [0.82, 1.60]; minus changing spaced 1.33 [0.91, 1.75]
- colour: changing bursts -0.12 [-0.14, -0.09]; minus fixed bursts 0.00 [-0.01, 0.02]; minus changing spaced -0.02 [-0.04, -0.00]
- cos_init, changing bursts minus fixed bursts: 0.071 [0.061, 0.082]
- layer-0 attention to the context shape, changing bursts minus fixed bursts: 0.069 [0.049, 0.090]
- geometry regression: changing-bursts coefficient 3.19 [2.46, 3.91] (p = 8.7e-18); observed 3.85, predicted from the other histories' geometry 0.74; geometry-only R2 0.36

- noun-level Spearman correlation of layer-0 attention with copy score: 0.81

## shifting-shape corpora

### 100 exposures
| Noun | own | typical | gap | frame2 | colour | cos_init | att_l0 |
|---|---|---|---|---|---|---|---|
| Unseen | 5.02 |  | -0.00 | 4.72 | -0.29 | 1.000 | 0.213 |
| Fixed, spaced | 2.23 | 0.52 | -0.20 | 1.99 | -0.12 | 0.845 | 0.134 |
| Fixed, bursts | 5.35 | 0.45 | -0.15 | 4.99 | -0.19 | 0.945 | 0.204 |
| No shape, spaced | 1.83 | 0.48 | -0.04 | 1.90 | -0.22 | 0.868 | 0.167 |
| No shape, bursts | 3.79 | 0.44 | -0.17 | 3.51 | -0.29 | 0.946 | 0.200 |
| Changing, spaced | 2.37 | 0.41 | -0.16 | 2.38 | -0.13 | 0.867 | 0.177 |
| Changing, bursts | 5.40 | 0.45 | -0.17 | 5.41 | -0.20 | 0.945 | 0.203 |

- swap: changing bursts own minus with fixed-burst rows: 0.08 [-0.02, 0.18]
- swap: fixed bursts with changing-burst rows minus own: 0.07 [-0.05, 0.19]
- gap: changing bursts -0.17 [-0.26, -0.09]; minus fixed bursts -0.02 [-0.13, 0.09]; minus changing spaced -0.02 [-0.18, 0.15]
- frame2: changing bursts 5.41 [4.90, 5.91]; minus fixed bursts 0.41 [0.06, 0.77]; minus changing spaced 3.03 [2.44, 3.61]
- colour: changing bursts -0.20 [-0.27, -0.13]; minus fixed bursts -0.01 [-0.03, 0.02]; minus changing spaced -0.07 [-0.10, -0.04]
- cos_init, changing bursts minus fixed bursts: 0.000 [-0.004, 0.004]
- layer-0 attention to the context shape, changing bursts minus fixed bursts: -0.000 [-0.002, 0.001]
- geometry regression: changing-bursts coefficient 1.31 [1.09, 1.52] (p = 1.1e-33); observed 5.40, predicted from the other histories' geometry 4.17; geometry-only R2 0.35

### 1000 exposures
| Noun | own | typical | gap | frame2 | colour | cos_init | att_l0 |
|---|---|---|---|---|---|---|---|
| Unseen | 5.02 |  | -0.00 | 4.72 | -0.29 | 1.000 | 0.213 |
| Fixed, spaced | 0.44 | 0.49 | -0.07 | 0.45 | -0.09 | 0.747 | 0.058 |
| Fixed, bursts | 3.57 | 0.52 | -0.19 | 2.91 | -0.12 | 0.803 | 0.090 |
| No shape, spaced | 0.63 | 0.43 | -0.05 | 0.75 | -0.24 | 0.783 | 0.103 |
| No shape, bursts | 2.09 | 0.55 | -0.19 | 2.21 | -0.23 | 0.821 | 0.126 |
| Changing, spaced | 0.51 | 0.41 | -0.06 | 0.60 | -0.09 | 0.783 | 0.106 |
| Changing, bursts | 6.82 | 0.47 | -0.16 | 7.02 | -0.14 | 0.813 | 0.187 |

- swap: changing bursts own minus with fixed-burst rows: 3.35 [2.74, 3.96]
- swap: fixed bursts with changing-burst rows minus own: 3.26 [2.63, 3.89]
- gap: changing bursts -0.16 [-0.26, -0.07]; minus fixed bursts 0.03 [-0.08, 0.14]; minus changing spaced -0.10 [-0.17, -0.03]
- frame2: changing bursts 7.02 [6.29, 7.76]; minus fixed bursts 4.12 [3.18, 5.05]; minus changing spaced 6.42 [5.66, 7.19]
- colour: changing bursts -0.14 [-0.19, -0.10]; minus fixed bursts -0.02 [-0.04, -0.01]; minus changing spaced -0.05 [-0.08, -0.02]
- cos_init, changing bursts minus fixed bursts: 0.010 [-0.000, 0.020]
- layer-0 attention to the context shape, changing bursts minus fixed bursts: 0.097 [0.078, 0.117]
- geometry regression: changing-bursts coefficient 4.74 [4.32, 5.16] (p = 8.2e-107); observed 6.82, predicted from the other histories' geometry 2.07; geometry-only R2 0.21

- noun-level Spearman correlation of layer-0 attention with copy score: 0.25

