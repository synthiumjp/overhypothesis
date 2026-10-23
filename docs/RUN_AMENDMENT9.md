# Running Amendment 9

PC, C:\overhypothesis, venv active (`.\.venv\Scripts\Activate.ps1`). Amendment 9 is registered (osf.io/zde87);
details fixed before any data are in docs/AMENDMENT9_ADDENDUM.md. Every model, including the pilot, is trained on
this PC.

```powershell
git pull
python -c "import hashlib;[print(hashlib.md5(open(l.split()[1],'rb').read()).hexdigest()==l.split()[0], l.split()[1]) for l in open('docs/amendment9_checksums.txt')]"
```

Every line must print `True`.

# Part 1: pilot (seed 2027, 3.4M models only; about 30 minutes)

The pilot only checks the materials and fixes the number of nouns per cell. Its models never enter a confirmatory test.

## P1. Build and check the pilot corpora (about 2 minutes, 0.3 GB on E:)

```powershell
python scripts/make_familiarity_corpora.py build --out_dir E:\overhypothesis_a9 --seeds 2027 --windows 160000 --per_cell 3
python scripts/make_familiarity_corpora.py check --out_dir E:\overhypothesis_a9 --seeds 2027 --windows 160000 --against docs/amendment9_corpus_manifest_pilot.json
```

The check must end with `3 of 3 corpora passed every check`. It re-tokenises each corpus and verifies the token count,
every exposure (as input and as target), the window and frame rules, the reading order, and that the files are
byte-identical to the ones recorded before training. If not, stop and send the output.

## P2. Smoke test (about 1 minute)

```powershell
$c = "E:\overhypothesis_a9\regular_grouped\w160k"
python scripts/train_model.py --condition regular --size tiny --seed 2027 --fresh_windows --corpus_file "$c\seed2027.txt" --window_order "$c\order_seed2027.npy" --tokenizer data/tokenizers/amendment9_tokenizer.json --untie_embeddings --save_every 50 --output_dir E:\AI_smoke_a9 --run_name smoke --smoke_test
```

It should print `Window order: order_seed2027.npy (160,000 windows, read in order, ...)`,
`Embeddings: untied (output layer starts as a copy of the input embedding)`, `device=directml` and
`Smoke test PASSED`. Then:

```powershell
Remove-Item -Recurse -Force E:\AI_smoke_a9
```

## P3. Train the three pilot models (about 20 minutes)

```powershell
foreach ($b in "regular_grouped","regular_random","reassigned_grouped") {
  $c = "E:\overhypothesis_a9\$b\w160k"
  $out = "E:\AI Checkpoints\overhypothesis_a9\$b"
  if (Test-Path "$out\tiny_seed2027\model_final.pt") { continue }
  python scripts/train_model.py --condition regular --size tiny --seed 2027 --fresh_windows --corpus_file "$c\seed2027.txt" --window_order "$c\order_seed2027.npy" --tokenizer data/tokenizers/amendment9_tokenizer.json --untie_embeddings --save_every 1000 --output_dir $out --run_name tiny_seed2027
}
```

## P4. Score, analyse heads, and write the pilot report (about 10 minutes, CPU)

```powershell
python scripts/amendment9_manifest.py --ckpt_dir "E:\AI Checkpoints\overhypothesis_a9" --corpus_dir E:\overhypothesis_a9 --per_cell 3 --seeds 2027 --out results/amendment9/manifest_pilot.csv
python scripts/familiarity_eval.py --manifest results/amendment9/manifest_pilot.csv --out results/amendment9/scores_pilot.csv
python scripts/circuit_analysis.py --manifest results/amendment9/manifest_pilot.csv --out_dir results/amendment9/circuits_pilot
python scripts/amendment9_tests.py pilot --scores results/amendment9/scores_pilot.csv --out results/amendment9/PILOT.md
foreach ($b in "regular_grouped","regular_random","reassigned_grouped") { Copy-Item "E:\AI Checkpoints\overhypothesis_a9\$b\tiny_seed2027\train_log.json" "results/amendment9/train_log_${b}_tiny_seed2027.json" }
git add results/amendment9
git commit -m "Amendment 9 pilot results"
git push origin HEAD:study1b-1c-evaluation
git push origin HEAD:master
```

Stop here. The pilot result and the number of nouns per cell go into the addendum (dated) and onto OSF before any
confirmatory corpus is built.

# Part 2: confirmatory models (90 models)

Three nouns per cell (pilot result, addendum). Upload the updated docs/AMENDMENT9_ADDENDUM.md to OSF first. Then
`git pull` and rerun the checksum line at the top; every line must print `True`.

Space needed on E: about 7 GB of corpora and about 50 GB of saved weights (weights every 1,000 steps). Check first:

```powershell
Get-PSDrive E
```

## C1. Build and check the 60 corpora (about 15 minutes, then about 30 minutes for the check)

```powershell
python scripts/make_familiarity_corpora.py build --out_dir E:\overhypothesis_a9 --per_cell 3 --windows 160000 256000
python scripts/make_familiarity_corpora.py check --out_dir E:\overhypothesis_a9 --per_cell 3 --windows 160000 256000 --against docs/amendment9_corpus_manifest.json
```

The check must end with `60 of 60 corpora passed every check`. If not, stop and send the output.

## C2. Train 90 models (two windows at once, about 15 to 18 hours)

Window 1 (seeds 42 to 1001):
```powershell
foreach ($s in 42,123,456,789,1001) { foreach ($b in "regular_grouped","reassigned_grouped","regular_random") { foreach ($z in "tiny","small","medium") {
  $w = if ($z -eq "small") { "w256k" } else { "w160k" }
  $c = "E:\overhypothesis_a9\$b\$w"
  $out = "E:\AI Checkpoints\overhypothesis_a9\$b"
  if (Test-Path "$out\${z}_seed$s\model_final.pt") { continue }
  python scripts/train_model.py --condition regular --size $z --seed $s --fresh_windows --corpus_file "$c\seed$s.txt" --window_order "$c\order_seed$s.npy" --tokenizer data/tokenizers/amendment9_tokenizer.json --untie_embeddings --save_every 1000 --output_dir $out --run_name "${z}_seed$s"
}}}
```

Window 2: the same block with `foreach ($s in 2718,3141,1618,1414,1732)`.

If the PC restarts, run the same blocks again: finished runs are skipped. When both are done, count the models (must
print 90):

```powershell
@(Get-ChildItem -Recurse -Filter model_final.pt "E:\AI Checkpoints\overhypothesis_a9" | Where-Object { $_.FullName -notmatch "seed2027" }).Count
```

## C3. Score, analyse heads and test (one window, CPU, several hours)

```powershell
python scripts/amendment9_manifest.py --ckpt_dir "E:\AI Checkpoints\overhypothesis_a9" --corpus_dir E:\overhypothesis_a9 --per_cell 3 --seeds 42 123 456 789 1001 2718 3141 1618 1414 1732 --out results/amendment9/manifest.csv
python scripts/familiarity_eval.py --manifest results/amendment9/manifest.csv --out results/amendment9/scores.csv
python scripts/circuit_analysis.py --manifest results/amendment9/manifest.csv --out_dir results/amendment9/circuits
python scripts/amendment9_tests.py tests --scores results/amendment9/scores.csv --circuits results/amendment9/circuits --out results/amendment9/RESULTS.md
New-Item -ItemType Directory -Force results\amendment9\train_logs | Out-Null
foreach ($b in "regular_grouped","reassigned_grouped","regular_random") { foreach ($z in "tiny","small","medium") { foreach ($s in 42,123,456,789,1001,2718,3141,1618,1414,1732) {
  Copy-Item "E:\AI Checkpoints\overhypothesis_a9\$b\${z}_seed$s\train_log.json" "results/amendment9/train_logs/${b}_${z}_seed$s.json"
}}}
git add results/amendment9
git commit -m "Amendment 9 confirmatory results"
git pull --rebase --autostash origin study1b-1c-evaluation
git push origin HEAD:study1b-1c-evaluation
git push origin HEAD:master
```

The scoring and head scripts can be stopped and restarted; finished models are skipped.

## C4. Exploratory: formation across saved weights (CPU, several hours; after C3)

```powershell
python scripts/amendment9_manifest.py --ckpt_dir "E:\AI Checkpoints\overhypothesis_a9" --corpus_dir E:\overhypothesis_a9 --per_cell 3 --seeds 42 123 456 789 1001 2718 3141 1618 1414 1732 --weights --out results/amendment9/manifest_formation.csv
python scripts/familiarity_eval.py --manifest results/amendment9/manifest_formation.csv --out results/amendment9/scores_formation.csv
python scripts/amendment9_figures.py --scores results/amendment9/scores.csv --circuits results/amendment9/circuits --formation results/amendment9/scores_formation.csv --out_dir paper_acl/figures
git add results/amendment9 paper_acl/figures
git commit -m "Amendment 9 formation scores (exploratory) and figures"
git pull --rebase --autostash origin study1b-1c-evaluation
git push origin HEAD:study1b-1c-evaluation
git push origin HEAD:master
```
