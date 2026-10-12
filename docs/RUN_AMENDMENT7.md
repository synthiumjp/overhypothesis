# Running Amendment 7

PC, C:\overhypothesis, venv active (`.\.venv\Scripts\Activate.ps1`). Register Amendment 7 on OSF first. Then:

```powershell
git pull
python -c "import hashlib,sys;[print(hashlib.md5(open(l.split()[1],'rb').read()).hexdigest()==l.split()[0], l.split()[1]) for l in open('docs/amendment7_checksums.txt')]"
python scripts/make_study1o_corpora.py check
```

Every line must print `True`, and the check must report 25 corpora.

## L. Loss check (one window, about 15 minutes; can run alongside O1)

```powershell
New-Item -ItemType Directory -Force results\rescore | Out-Null
python scripts/loss_check.py --manifest results/study1b/raw/manifest.csv results/study1c/raw/manifest.csv results/study1x/manifest_graded_b3p6.csv --n_models 8 --out results/rescore/loss_check.csv
```

## O1. Train 75 models (two windows: medium in one, tiny and small in the other)

Window 1:
```powershell
foreach ($c in "random","same_grouped","diff_grouped","same_b3p6","diff_b3p6") { foreach ($z in "medium") { foreach ($s in 42,123,456,789,1001) {
  $out = "E:\AI Checkpoints\overhypothesis_o\$c"
  if (Test-Path "$out\regular_${z}_seed$s\model_final.pt") { continue }
  python scripts/train_model.py --condition regular --size $z --seed $s --corpus_dir data/corpora_o/$c --output_dir $out
}}}
```

Window 2: the same block with `foreach ($z in "tiny","small")`.

## O2. Score (two windows)

Window 1, novel-noun items:
```powershell
New-Item -ItemType Directory -Force results\study1o\raw, results\study1o\manifests | Out-Null
foreach ($c in "random","same_grouped","diff_grouped","same_b3p6","diff_b3p6") {
  python scripts/oneshot_eval.py make-manifest --base_dir . --ckpt_dir "E:\AI Checkpoints\overhypothesis_o\$c" --conditions regular --label_suffix "_o_$c" --out results/study1o/manifest_$c.csv
  python scripts/oneshot_eval.py eval-lm --items data/wug_tests/oneshot_items.csv --manifest results/study1o/manifest_$c.csv --out results/study1o/raw/lm_$c.csv
}
```

Window 2, trained-noun items:
```powershell
New-Item -ItemType Directory -Force results\study1o\raw, results\study1o\manifests | Out-Null
foreach ($c in "random","same_grouped","diff_grouped","same_b3p6","diff_b3p6") {
  python scripts/study1_trained_nouns.py eval --ckpt_dir "E:\AI Checkpoints\overhypothesis_o\$c" --family regular --run_prefix regular --tokenizer regular --label "regular_o_$c" --tmp_dir results/study1o/manifests --out results/study1o/raw/trained.csv
}
```

Then:
```powershell
Compress-Archive -Path results\study1o, results\rescore\loss_check.csv -DestinationPath $env:USERPROFILE\Downloads\study1o.zip -Force
```

## Analysis (in the repository)

```
python scripts/oneshot_eval.py summarise --inputs "results/study1o/raw/lm_*.csv.gz" --out results/study1o/summary
python scripts/oneshot_eval.py summarise --inputs results/study1o/raw/trained.csv.gz --out results/study1o/trained
python scripts/study1o_tests.py --runs results/rescore/study1bc_per_run.csv results/study1o/summary_per_run.csv \
  --trained results/study1o/trained_per_run.csv --out results/study1o/tests
```
