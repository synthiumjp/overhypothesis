# Running Amendment 8

PC, C:\overhypothesis, venv active (`.\.venv\Scripts\Activate.ps1`). **Register Amendment 8 on OSF first** (`docs/AMENDMENT8.md`). Then:

```powershell
git pull
python -c "import hashlib,sys;[print(hashlib.md5(open(l.split()[1],'rb').read()).hexdigest()==l.split()[0], l.split()[1]) for l in open('docs/amendment8_checksums.txt')]"
```

Every line must print `True`.

## C. Build the corpora (about 30 minutes, 5.3 GB on E:)

```powershell
python scripts/make_fresh_corpora.py build --out_dir E:\overhypothesis_fresh
python scripts/make_fresh_corpora.py check --out_dir E:\overhypothesis_fresh --against docs/amendment8_corpus_manifest.json
```

The second line of the check is the real test: it must report `80 of 80 registered corpora reproduced exactly`. If not, stop and send the output.

## S. Smoke test (about 2 minutes)

```powershell
python scripts/train_model.py --condition regular --size tiny --seed 42 --corpus_dir E:\overhypothesis_fresh\reassigned_grouped --output_dir E:\AI_smoke_fresh --fresh_windows --smoke_test
```

It should print `Fresh windows: 271,... available, 3,200 needed` and `Smoke test PASSED`. Then delete the folder:

```powershell
Remove-Item -Recurse -Force E:\AI_smoke_fresh
```

## T. Train 120 models (two windows, about 15 hours)

Each run first reads and tokenises its corpus (about 30 to 60 seconds), then trains as in Study 1. Each loop deletes the run's intermediate checkpoints once `model_final.pt` exists, so the 120 runs need only about 15 GB; a run in progress needs up to 6 GB more. Check the free space first:

```powershell
Get-PSDrive E
```

Window 1:
```powershell
foreach ($c in "regular_random","regular_b3p6","regular_grouped","nostable_grouped") { foreach ($z in "tiny","small","medium") { foreach ($s in 42,123,456,789,1001) {
  $out = "E:\AI Checkpoints\overhypothesis_fresh\$c"
  if (Test-Path "$out\regular_${z}_seed$s\model_final.pt") { continue }
  python scripts/train_model.py --condition regular --size $z --seed $s --corpus_dir E:\overhypothesis_fresh\$c --output_dir $out --fresh_windows
  if (Test-Path "$out\regular_${z}_seed$s\model_final.pt") { Remove-Item "$out\regular_${z}_seed$s\ckpt_step*.pt" -ErrorAction SilentlyContinue }
}}}
```

Window 2: the same block with `foreach ($c in "reassigned_grouped","reassigned_random","objects_same","objects_diff")`.

If the PC restarts, run the same blocks again: finished runs are skipped.

When both windows are done, count the models (must print 120):
```powershell
@(Get-ChildItem -Recurse -Filter model_final.pt "E:\AI Checkpoints\overhypothesis_fresh").Count
```

## E. Score (two windows, about 1 hour)

Window 1, novel-noun items:
```powershell
New-Item -ItemType Directory -Force results\amendment8\raw, results\amendment8\manifests | Out-Null
foreach ($c in "regular_random","regular_b3p6","regular_grouped","nostable_grouped","objects_same","objects_diff","reassigned_grouped","reassigned_random") {
  python scripts/oneshot_eval.py make-manifest --base_dir . --ckpt_dir "E:\AI Checkpoints\overhypothesis_fresh\$c" --conditions regular --label_suffix "_f_$c" --out results/amendment8/manifest_$c.csv
  python scripts/oneshot_eval.py eval-lm --items data/wug_tests/oneshot_items.csv --manifest results/amendment8/manifest_$c.csv --out results/amendment8/raw/lm_$c.csv
}
```

Window 2, trained-noun items:
```powershell
New-Item -ItemType Directory -Force results\amendment8\raw, results\amendment8\manifests | Out-Null
foreach ($c in "regular_random","regular_b3p6","regular_grouped","nostable_grouped","objects_same","objects_diff","reassigned_grouped","reassigned_random") {
  python scripts/study1_trained_nouns.py eval --ckpt_dir "E:\AI Checkpoints\overhypothesis_fresh\$c" --family regular --run_prefix regular --tokenizer regular --label "regular_f_$c" --tmp_dir results/amendment8/manifests --out results/amendment8/raw/trained.csv
}
```

## L. Memorisation check (one window, after Window 1 of E has written the manifests; about 2 hours)

New models:
```powershell
foreach ($c in "regular_random","regular_b3p6","regular_grouped","nostable_grouped","objects_same","objects_diff") {
  python scripts/heldout_loss.py --manifest results/amendment8/manifest_$c.csv --heldout_dir E:\overhypothesis_fresh\heldout\$c --out results/amendment8/heldout_loss.csv
}
```

The matching fixed-corpus models:
```powershell
python scripts/heldout_loss.py --manifest results/study1b/raw/manifest.csv --conditions regular --heldout_dir E:\overhypothesis_fresh\heldout\regular_random --out results/amendment8/heldout_loss.csv
python scripts/heldout_loss.py --manifest results/study1x/manifest_graded_b3p6.csv --heldout_dir E:\overhypothesis_fresh\heldout\regular_b3p6 --out results/amendment8/heldout_loss.csv
python scripts/heldout_loss.py --manifest results/study1c/raw/manifest.csv --conditions regular_bursty --heldout_dir E:\overhypothesis_fresh\heldout\regular_grouped --out results/amendment8/heldout_loss.csv
python scripts/heldout_loss.py --manifest results/study1x/manifest_nostable_grouped.csv --heldout_dir E:\overhypothesis_fresh\heldout\nostable_grouped --out results/amendment8/heldout_loss.csv
python scripts/heldout_loss.py --manifest results/study1o/manifest_same_grouped.csv --heldout_dir E:\overhypothesis_fresh\heldout\objects_same --out results/amendment8/heldout_loss.csv
python scripts/heldout_loss.py --manifest results/study1o/manifest_diff_grouped.csv --heldout_dir E:\overhypothesis_fresh\heldout\objects_diff --out results/amendment8/heldout_loss.csv
```

The file should end with 180 rows.

## Send back

```powershell
Copy-Item E:\overhypothesis_fresh\manifest.json results\amendment8\corpus_manifest.json
Compress-Archive -Path E:\overhypothesis_fresh\heldout -DestinationPath $env:USERPROFILE\Downloads\amendment8_heldout.zip -Force
Get-ChildItem "E:\AI Checkpoints\overhypothesis_fresh" -Recurse -Filter train_log.json | ForEach-Object { $d = "results\amendment8\train_logs\" + $_.Directory.Parent.Name; New-Item -ItemType Directory -Force $d | Out-Null; Copy-Item $_.FullName "$d\$($_.Directory.Name).json" }
Compress-Archive -Path results\amendment8 -DestinationPath $env:USERPROFILE\Downloads\amendment8.zip -Force
```

## Analysis (in the repository)

```
python scripts/oneshot_eval.py summarise --inputs "results/amendment8/raw/lm_*.csv*" --out results/amendment8/summary
python scripts/oneshot_eval.py summarise --inputs "results/amendment8/raw/trained.csv*" --out results/amendment8/trained
python scripts/amendment8_tests.py --runs results/amendment8/summary_per_run.csv results/rescore/study1bc_per_run.csv \
  results/rescore/study1x_per_run.csv results/study1o/summary_per_run.csv \
  --trained results/amendment8/trained_per_run.csv --loss results/amendment8/heldout_loss.csv \
  --heldout_dir <unzipped amendment8_heldout.zip, or a rebuild with make_fresh_corpora.py build --parts heldout> --out results/amendment8/tests
```
