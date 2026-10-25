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
  python scripts/train_model.py --condition regular --size $z --seed $s --fresh_windows --corpus_file "$c\seed$s.txt" --window_order "$c\order_seed$s.npy" --tokenizer data/tokenizers/amendment9_tokenizer.json --untie_embeddings --save_every 1000 --output_dir $out --run_name "${z}_seed$s"
}}}
```
Count (must print 60): `@(Get-ChildItem -Recurse -Filter model_final.pt "E:\AI Checkpoints\overhypothesis_a11").Count`

## 3. Part B: train 10 large models (Mac, MPS)
On the Mac, in ~/jpwork/overhypothesis, build only the corpora Part B needs, check them, then train:
```bash
git pull --rebase --autostash origin study1b-1c-evaluation
python scripts/make_familiarity_corpora_a11.py build --out_dir ~/overhypothesis_a11 --designs mixed --windows 160000
python scripts/make_familiarity_corpora_a11.py check --out_dir ~/overhypothesis_a11 --designs mixed --windows 160000 --against docs/amendment11_corpus_manifest.json
for s in 2029 3037 4049 5051 6067 7079 8081 9091 10103 11117; do
  c=~/overhypothesis_a11/regular_grouped/w160k
  python scripts/train_model.py --condition regular --size large --seed $s --fresh_windows --corpus_file $c/seed$s.txt --window_order $c/order_seed$s.npy --tokenizer data/tokenizers/amendment9_tokenizer.json --untie_embeddings --save_every 1000 --output_dir ~/ckpt_a11/regular_grouped --run_name large_seed$s
done
```
The check must end with `10 of 10 corpora passed every check`. Training prints the device; it should say `mps`. Time the
first model and send the time before starting the rest.

## 4. Score and test
On the machine with the models (the large ones can be scored on the Mac with the same commands and their own manifest):
```powershell
python scripts/amendment9_manifest.py --design a11 --ckpt_dir "E:\AI Checkpoints\overhypothesis_a11" --corpus_dir E:\overhypothesis_a11 --per_cell 3 --seeds 2029 3037 4049 5051 6067 7079 8081 9091 10103 11117 --out results/amendment11/manifest.csv
python scripts/familiarity_eval.py --manifest results/amendment11/manifest.csv --out results/amendment11/scores.csv
python scripts/a10_reanalysis.py --manifest results/amendment11/manifest.csv --out results/amendment11/reanalysis.csv --check_scores results/amendment11/scores.csv
python scripts/a10_crossed.py --manifest results/amendment11/manifest.csv --out results/amendment11/crossed.csv
```
Mac (large models):
```bash
python scripts/amendment9_manifest.py --design a11 --ckpt_dir ~/ckpt_a11 --corpus_dir ~/overhypothesis_a11 --per_cell 3 --seeds 2029 3037 4049 5051 6067 7079 8081 9091 10103 11117 --out results/amendment11/manifest_large.csv
python scripts/familiarity_eval.py --manifest results/amendment11/manifest_large.csv --out results/amendment11/scores_large.csv
```
Then (either machine, after both are pushed):
```powershell
python -c "import pandas as pd; pd.concat([pd.read_csv('results/amendment11/scores.csv'), pd.read_csv('results/amendment11/scores_large.csv')]).to_csv('results/amendment11/scores_all.csv', index=False)"
python scripts/a11_tests.py --scores results/amendment11/scores_all.csv --rows results/amendment11/reanalysis.csv --crossed results/amendment11/crossed.csv --out results/amendment11/RESULTS.md
git add results/amendment11
git commit -m "Amendment 11 results"
git pull --rebase --autostash origin study1b-1c-evaluation
git push origin HEAD:study1b-1c-evaluation
git push origin HEAD:master
```
