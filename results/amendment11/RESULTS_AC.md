# Amendment 11 results

Models: {"regular_grouped/tiny": 10, "regular_grouped/small": 10, "regular_grouped/medium": 10, "regular_grouped/large": 0, "regular_pure_changing/tiny": 5, "regular_pure_changing/small": 5, "regular_pure_changing/medium": 5, "regular_pure_changing/large": 0, "regular_pure_fixed/tiny": 5, "regular_pure_fixed/small": 5, "regular_pure_fixed/medium": 5, "regular_pure_fixed/large": 0, "model_check_files": 1, "excluded_failed_loss": [], "excluded_nonfinite_score": [], "mixed_models_missing_rows_or_crossed": [], "partA_complete_seeds": [2029, 3037, 4049, 5051, 6067, 7079, 8081, 9091, 10103, 11117], "partB_complete_seeds": [], "partC_complete_seeds": [2029, 3037, 4049, 5051, 6067], "H59_seeds": [2029, 3037, 4049, 5051, 6067]}

PartA: interpreted = True
PartB: interpreted = False (not run (no models))
PartC: interpreted = True

- M3: 5.880 [5.308, 6.452], p = 1.2e-09; pass = True; n = 10
- H50: 3.683 [2.993, 4.373], p = 3.63e-07; n = 10; sizes_in_direction = 3; holm_threshold = 0.0167; holm_reject = True; supported = True
- H51: 4.144 [3.599, 4.690], p = 1.72e-08; n = 10; sizes_in_direction = 3; holm_threshold = 0.01; holm_reject = True; supported = True
- H52: -0.250 [-0.349, -0.150], p = 0.000298, 90% [-0.330, -0.169]; n = 10; sizes_in_direction = 3; holm_threshold = 0.05; holm_reject = True; tost_p = 0.141; outcome = different (fixed below)
- H53: 5.841 [5.569, 6.113], p = 1.68e-12; n = 10; sizes_in_direction = 3; holm_threshold = 0.00833; holm_reject = True; supported = True
- H54: 0.078 [0.019, 0.136], p = 0.0145, 90% [0.031, 0.125]; n = 10; tost_p = 6.15e-06; outcome = different from zero but within +-0.3
- H55: 3.675 [3.029, 4.320], p = 2.11e-07; n = 10; n_models = 30; sizes_in_direction = 3; holm_threshold = 0.0125; holm_reject = True; supported = True
  - H55 pooled noun-level, seed-clustered (reported): 3.822 [3.184, 4.461], p = 1.36e-07
  - H55 without no-shape nouns (sensitivity): 3.663 [3.049, 4.276], p = 1.4e-07; n = 10
- H56: 1.249 [0.935, 1.564], p = 4.3e-06; n = 10; sizes_in_direction = 3; holm_threshold = 0.025; holm_reject = True; supported = True
- M3C_changing: 5.163 [3.246, 7.081], p = 0.000856; pass = True; n = 5
- M3C_fixed: 3.872 [2.842, 4.901], p = 0.000238; pass = True; n = 5
- H58: 5.510 [4.239, 6.781], p = 0.000137; n = 5; sizes_in_direction = 3; supported = True
- H59_changing_bursts: 0.240 [-1.440, 1.920], p = 0.712, 90% [-1.050, 1.530]; n = 5; tost_p = 0.463; outcome = inconclusive
- H59_fixed_bursts: -0.997 [-1.196, -0.799], p = 0.000154, 90% [-1.150, -0.845]; n = 5; tost_p = 1; outcome = different (pure below mixed)
