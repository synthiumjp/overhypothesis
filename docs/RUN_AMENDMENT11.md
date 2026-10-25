# Running Amendment 11

Registration: docs/AMENDMENT11.md. Every command runs from the repository folder. Nothing below runs before
registration, except the checksum check.

## 0. On every machine
```powershell
git pull --rebase --autostash origin study1b-1c-evaluation
python -c "import hashlib;[print(hashlib.md5(open(l.split()[1],'rb').read()).hexdigest()==l.split()[0], l.split()[1]) for l in open('docs/amendment11_checksums.txt')]"
```
Every line must print `True`.

## 1. Build and check the corpora (PC, about 15 minutes, then about 20 for the check; about 4.4 GB)
```powershell
python scripts/make_familiarity_corpora_a11.py build --out_dir E:\overhypothesis_a11
python scripts/make_familiarity_corpora_a11.py check --out_dir E:\overhypothesis_a11 --against docs/amendment11_corpus_manifest.json
```
The check must end with `40 of 40 corpora passed every check`.

## 2. Part A and Part C: train 60 models (PC, two windows, about 12 hours)
Window 1 (Part A, seeds 2029 to 6067, and all of Part C):
```powershell
foreach ($s in 2029,3037,4049,5051,6067) { foreach ($b in "regular_grouped","regular_pure_changing","regular_pure_fixed") { foreach ($z in "tiny","small","medium") {
  $w = if ($z -eq "small") { "w256k" } else { "w160k" }
  $c = "E:\overhypothesis_a11\$b\$w"
  $out = "E:\AI Checkpoints\overhypothesis_a11\$b"
  if (Test-Path "$out\${z}_seed$s\model_final.pt") { continue }
  if (Test-Path "$out\${z}_seed${s}_failed2") { continue }
  python scripts/train_model.py --condition regular --size $z --seed $s --fresh_windows --corpus_file "$c\seed$s.txt" --window_order "$c\order_seed$s.npy" --tokenizer data/tokenizers/amendment9_tokenizer.json --untie_embeddings --save_every 1000 --output_dir $out --run_name "${z}_seed$s"
}}}
```
Window 2 (Part A, seeds 7079 to 11117):
```powershell
foreach ($s in 7079,8081,9091,10103,11117) { foreach ($b in "regular_grouped") { foreach ($z in "tiny","small","medium") {
  $w = if ($z -eq "small") { "w256k" } else { "w160k" }
  $c = "E:\overhypothesis_a11\$b\$w"
  $out = "E:\AI Checkpoints\overhypothesis_a11\$b"
  if (Test-Path "$out\${z}_seed$s\model_final.pt") { continue }
  if (Test-Path "$out\${z}_seed${s}_failed2") { continue }
  python scripts/train_model.py --condition regular --size $z --seed $s --fresh_windows --corpus_file "$c\seed$s.txt" --window_order "$c\order_seed$s.npy" --tokenizer data/tokenizers/amendment9_tokenizer.json --untie_embeddings --save_every 1000 --output_dir $out --run_name "${z}_seed$s"
}}}
```
Count (must print 60): `@(Get-ChildItem -Recurse -Filter model_final.pt "E:\AI Checkpoints\overhypothesis_a11" | Where-Object { $_.FullName -notmatch "_failed" }).Count`

Failed runs (registered rule: one retrain only). If a run stops with `Non-finite loss`, or the model check in step 4
lists a model as FAIL, or RESULTS.md lists a model under `excluded_nonfinite_score`: move its folder to
`<name>_failed1` (keep it, with FAILED_nonfinite_loss.txt if there is one), note it in
docs/AMENDMENT11_DEVIATIONS.md, and train that one model once more with the same command; then repeat step 4.
If the retrained model fails again, rename its folder to `<name>_failed2` (the loops skip it), leave it excluded and
note that too. Never retrain a model a second time.

## 3. Part B: train 10 large models (Mac, MPS)
Each large model saves eleven weight files of about 340 MB, so the ten models need about 38 GB of free disk.
On the Mac, in ~/jpwork/overhypothesis, build only the corpora Part B needs, check them, then train:
```bash
git pull --rebase --autostash origin study1b-1c-evaluation
python scripts/make_familiarity_corpora_a11.py build --out_dir ~/overhypothesis_a11 --designs mixed --windows 160000
python scripts/make_familiarity_corpora_a11.py check --out_dir ~/overhypothesis_a11 --designs mixed --windows 160000 --against docs/amendment11_corpus_manifest.json
c=~/overhypothesis_a11/regular_grouped/w160k
time python scripts/train_model.py --condition regular --size large --seed 2029 --fresh_windows --corpus_file $c/seed2029.txt --window_order $c/order_seed2029.npy --tokenizer data/tokenizers/amendment9_tokenizer.json --untie_embeddings --save_every 1000 --output_dir ~/ckpt_a11/regular_grouped --run_name large_seed2029
```
The check must end with `10 of 10 corpora passed every check`. Training prints the device; it should say `mps`.
Send the time this first model took before starting the rest (in this order; the seed order is registered):
```bash
for s in 3037 4049 5051 6067 7079 8081 9091 10103 11117; do
  python scripts/train_model.py --condition regular --size large --seed $s --fresh_windows --corpus_file $c/seed$s.txt --window_order $c/order_seed$s.npy --tokenizer data/tokenizers/amendment9_tokenizer.json --untie_embeddings --save_every 1000 --output_dir ~/ckpt_a11/regular_grouped --run_name large_seed$s
done
```
If a run stops with `Non-finite loss`, follow the failed-run rule in step 2.

## 4. Score and test
On the machine with the models (the large ones can be scored on the Mac with the same commands and their own manifest):
```powershell
python scripts/amendment9_manifest.py --design a11 --ckpt_dir "E:\AI Checkpoints\overhypothesis_a11" --corpus_dir E:\overhypothesis_a11 --per_cell 3 --seeds 2029 3037 4049 5051 6067 7079 8081 9091 10103 11117 --out results/amendment11/manifest.csv
python scripts/a11_model_check.py --manifest results/amendment11/manifest.csv --out results/amendment11/model_check.csv
python scripts/familiarity_eval.py --manifest results/amendment11/manifest.csv --out results/amendment11/scores.csv
python scripts/a10_reanalysis.py --manifest results/amendment11/manifest.csv --out results/amendment11/reanalysis.csv --check_scores results/amendment11/scores.csv
python scripts/a10_crossed.py --manifest results/amendment11/manifest.csv --out results/amendment11/crossed.csv
```
Mac (large models):
```bash
python scripts/amendment9_manifest.py --design a11 --ckpt_dir ~/ckpt_a11 --corpus_dir ~/overhypothesis_a11 --per_cell 3 --seeds 2029 3037 4049 5051 6067 7079 8081 9091 10103 11117 --out results/amendment11/manifest_large.csv
python scripts/a11_model_check.py --manifest results/amendment11/manifest_large.csv --out results/amendment11/model_check_large.csv
python scripts/familiarity_eval.py --manifest results/amendment11/manifest_large.csv --out results/amendment11/scores_large.csv
python -c "import torch, platform; print(torch.__version__, platform.platform())" > results/amendment11/env_mac.txt
```
Then (either machine, after both are pushed):
```powershell
python -c "import pandas as pd; pd.concat([pd.read_csv('results/amendment11/scores.csv'), pd.read_csv('results/amendment11/scores_large.csv')]).to_csv('results/amendment11/scores_all.csv', index=False)"
python -c "import torch, platform; print(torch.__version__, platform.platform())" > results/amendment11/env_pc.txt
python scripts/a11_tests.py --scores results/amendment11/scores_all.csv --rows results/amendment11/reanalysis.csv --crossed results/amendment11/crossed.csv --model_check results/amendment11/model_check.csv results/amendment11/model_check_large.csv --out results/amendment11/RESULTS.md
git add results/amendment11
git commit -m "Amendment 11 results"
git pull --rebase --autostash origin study1b-1c-evaluation
git push origin HEAD:study1b-1c-evaluation
git push origin HEAD:master
```
