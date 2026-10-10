# Running the Amendment 6 window test

Run nothing below until Amendment 6 is registered on OSF. In C:\overhypothesis, with the virtual environment active (`.\.venv\Scripts\Activate.ps1`), pull and check the locked files:

```powershell
git pull
python -c "import hashlib,sys;[print(hashlib.md5(open(l.split()[1],'rb').read()).hexdigest()==l.split()[0], l.split()[1]) for l in open('docs/amendment6_checksums.txt')]"
```

Every line must print `True`.

## W1. Train 45 models (3 corpora x 3 sizes x 5 seeds)

The loop skips any model that already has `model_final.pt`, so it can be stopped and restarted. To run two windows in parallel, give each window a different `$corpora` list (for example, first and second corpus in window 1, the third in window 2).

```powershell
$dirs = @{ "regular" = "data/corpora"; "graded_b3p6" = "data/corpora_x/graded_b3p6"; "nostable_random" = "data/corpora_x/nostable_random" }
$corpora = "regular","graded_b3p6","nostable_random"
foreach ($c in $corpora) { foreach ($z in "tiny","small","medium") { foreach ($s in 42,123,456,789,1001) {
  $out = "E:\AI Checkpoints\overhypothesis_w512\$c"
  if (Test-Path "$out\regular_${z}_seed$s\model_final.pt") { continue }
  python scripts/train_model.py --condition regular --size $z --seed $s --corpus_dir $dirs[$c] `
    --output_dir $out --seq_len 512 --keep_tokens_per_step
}}}
```

Each run should print `batch: 8 | seq_len: 512` (tiny, small) or `batch: 4 | seq_len: 512` (medium).

## W2. Score on the Study 1 items (novel nouns)

```powershell
New-Item -ItemType Directory -Force results\study1w\raw, results\study1w\manifests | Out-Null
$labels = @{ "regular" = "_w512"; "graded_b3p6" = "_graded_b3p6_w512"; "nostable_random" = "_nostable_random_w512" }
foreach ($c in "regular","graded_b3p6","nostable_random") {
  python scripts/oneshot_eval.py make-manifest --base_dir . --ckpt_dir "E:\AI Checkpoints\overhypothesis_w512\$c" `
    --conditions regular --label_suffix $labels[$c] --out results/study1w/manifest_$c.csv
  python scripts/oneshot_eval.py eval-lm --items data/wug_tests/oneshot_items.csv `
    --manifest results/study1w/manifest_$c.csv --out results/study1w/raw/lm_$c.csv
}
python scripts/oneshot_eval.py summarise --inputs "results/study1w/raw/lm_*.csv" --out results/study1w/summary
```

## W3. Trained-noun items (S28)

```powershell
foreach ($c in "regular","graded_b3p6","nostable_random") {
  python scripts/study1_trained_nouns.py eval --ckpt_dir "E:\AI Checkpoints\overhypothesis_w512\$c" --family regular `
    --run_prefix regular --tokenizer regular --label ("regular" + $labels[$c]) `
    --tmp_dir results/study1w/manifests --out results/study1w/raw/trained.csv
}
python scripts/oneshot_eval.py summarise --inputs results/study1w/raw/trained.csv --out results/study1w/trained
```

## W4. Registered tests

```powershell
python scripts/study1w_tests.py --runs results/study1c/summary_per_run.csv results/study1x/summary_per_run.csv `
  results/study1w/summary_per_run.csv --trained results/study1w/trained_per_run.csv --out results/study1w/tests
```

Then zip `results\study1w` and send it:

```powershell
Compress-Archive -Path results\study1w -DestinationPath $env:USERPROFILE\Downloads\study1w.zip -Force
```

## Already done before registration (in the repo)

```
python scripts/window_cooccurrence.py measure --out results/study1w/cooccurrence.csv
python scripts/window_cooccurrence.py predict --cooc results/study1w/cooccurrence.csv \
  --runs results/study1c/summary_per_run.csv results/study1x/summary_per_run.csv --out results/study1w/prediction.json
```
