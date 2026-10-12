# Rescoring all Study 1 models with the corrected loader (12 Oct 2026)

## Why

`train_model.py` ties the output matrix (`lm_head.weight`) to the input embedding (`tok_embed.weight`). On DirectML the tie broke when the model moved to the device, so every Study 1 model was trained with two separate matrices, and both are saved. `oneshot_eval.py` rebuilt each model with the tie in place and then loaded the saved weights, so `lm_head.weight` overwrote `tok_embed.weight`: every score before commit 483ad6b used the output matrix as the input embedding. The fix unties before loading. `--legacy_tied_load` reproduces the old scores exactly (checked on three models).

This rescoring uses the same manifests, items and models as before. Only the loader changes. Outputs go to `results/rescore/`. The Bayesian reference, the untrained baseline (random weights, tied as at initialisation), Study 2 and Study 3 are not affected.

## Run (PC, C:\overhypothesis, venv active, after `git pull`)

Three windows in parallel. Each command resumes if stopped.

**Window A** (Study 1 final models, 120):

```powershell
New-Item -ItemType Directory -Force results\rescore | Out-Null
python scripts/oneshot_eval.py eval-lm --items data/wug_tests/oneshot_items.csv --manifest results/study1b/raw/manifest.csv --out results/rescore/study1b_lm.csv
```

**Window B** (grouped, extension and window-test models, 195):

```powershell
New-Item -ItemType Directory -Force results\rescore | Out-Null
python scripts/oneshot_eval.py eval-lm --items data/wug_tests/oneshot_items.csv --manifest results/study1c/raw/manifest.csv --out results/rescore/study1c_lm.csv
foreach ($c in "nostable_random","nostable_grouped","colourstable_random","colourstable_grouped","graded_b1p4","graded_b3p6","graded_b7","graded_b12") {
  python scripts/oneshot_eval.py eval-lm --items data/wug_tests/oneshot_items.csv --manifest results/study1x/manifest_$c.csv --out results/rescore/study1x_lm_$c.csv
}
foreach ($c in "regular","graded_b3p6","nostable_random") {
  python scripts/oneshot_eval.py eval-lm --items data/wug_tests/oneshot_items.csv --manifest results/study1w/manifest_$c.csv --out results/rescore/study1w_lm_$c.csv
}
```

**Window C** (trained nouns, then checkpoint tracking):

```powershell
New-Item -ItemType Directory -Force results\rescore\manifests | Out-Null
$T = "--tmp_dir", "results/rescore/manifests", "--out", "results/rescore/trained.csv"
python scripts/study1_trained_nouns.py eval --ckpt_dir "E:\AI Checkpoints\overhypothesis_checkpoints" --family regular --label regular @T
python scripts/study1_trained_nouns.py eval --ckpt_dir "E:\AI Checkpoints\overhypothesis_checkpoints" --family scrambled --label scrambled @T
python scripts/study1_trained_nouns.py eval --ckpt_dir "E:\AI Checkpoints\overhypothesis_bursty" --family regular --label regular_bursty @T
python scripts/study1_trained_nouns.py eval --ckpt_dir "E:\AI Checkpoints\overhypothesis_bursty" --family scrambled --label scrambled_bursty @T
foreach ($c in "nostable_random","nostable_grouped","colourstable_random","colourstable_grouped","graded_b1p4","graded_b3p6","graded_b7","graded_b12") {
  python scripts/study1_trained_nouns.py eval --ckpt_dir "E:\AI Checkpoints\overhypothesis_x\$c" --family regular --run_prefix regular --tokenizer regular --label "regular_$c" @T
}
$labels = @{ "regular" = "_w512"; "graded_b3p6" = "_graded_b3p6_w512"; "nostable_random" = "_nostable_random_w512" }
foreach ($c in "regular","graded_b3p6","nostable_random") {
  python scripts/study1_trained_nouns.py eval --ckpt_dir "E:\AI Checkpoints\overhypothesis_w512\$c" --family regular --run_prefix regular --tokenizer regular --label ("regular" + $labels[$c]) @T
}
python scripts/oneshot_eval.py eval-lm --items data/wug_tests/oneshot_items.csv --manifest results/study1c/raw/track_manifest_regular.csv --ks 1 --out results/rescore/track_regular.csv
python scripts/oneshot_eval.py eval-lm --items data/wug_tests/oneshot_items.csv --manifest results/study1c/raw/track_manifest_bursty.csv --ks 1 --out results/rescore/track_bursty.csv
```

When all three windows are back at the prompt:

```powershell
Compress-Archive -Path results\rescore -DestinationPath $env:USERPROFILE\Downloads\rescore.zip -Force
```

## Expected rows

| File | Rows |
|---|---|
| study1b_lm.csv | 432,000 (120 models × 3,600) |
| study1c_lm.csv | 108,000 |
| study1x_lm_*.csv | 54,000 each |
| study1w_lm_*.csv | 54,000 each |
| trained.csv | 216,000 (225 models × 960) |
| track_regular.csv + track_bursty.csv | 553,200 (461 checkpoints × 1,200) |
