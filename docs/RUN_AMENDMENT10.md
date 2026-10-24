# Running Amendment 10

Three parts: A (training, the PC), B (pretrained Pythia models, the Mac), C (inference on the Amendment 9 models, the PC).
Registration: docs/AMENDMENT10.md. Every command runs from the repository folder.

## 0. Before registration: noun lists (PC, about 10 minutes)

```powershell
cd C:\overhypothesis
.\.venv\Scripts\Activate.ps1
git pull --rebase --autostash origin study1b-1c-evaluation
pip install transformers requests
python scripts/pythia_prep.py --out_dir data/amendment10
git add data/amendment10
git commit -m "Amendment 10 noun lists (before any model run)"
git push origin HEAD:study1b-1c-evaluation
git push origin HEAD:master
```

It prints the number of matched pairs and the colour words. Stop here: the lists are checked and their checksums recorded,
then Amendment 10 is registered. Nothing below runs before registration.

After registration, on every machine:
```powershell
git pull --rebase --autostash origin study1b-1c-evaluation
python -c "import hashlib;[print(hashlib.md5(open(l.split()[1],'rb').read()).hexdigest()==l.split()[0], l.split()[1]) for l in open('docs/amendment10_checksums.txt')]"
```
Every line must print `True`.

## Part A (PC)

### A1. Build and check the 40 corpora (about 10 minutes, then about 20 for the check; about 4.4 GB)
```powershell
python scripts/make_familiarity_corpora_a10.py build --out_dir E:\overhypothesis_a10
python scripts/make_familiarity_corpora_a10.py check --out_dir E:\overhypothesis_a10 --against docs/amendment10_corpus_manifest.json
```
The check must end with `40 of 40 corpora passed every check`.

### A2. Train 60 models (two windows, about 12 hours)
Window 1:
```powershell
foreach ($s in 42,123,456,789,1001) { foreach ($b in "regular_grouped","reassigned_grouped") { foreach ($z in "tiny","small","medium") {
  $w = if ($z -eq "small") { "w256k" } else { "w160k" }
  $c = "E:\overhypothesis_a10\$b\$w"
  $out = "E:\AI Checkpoints\overhypothesis_a10\$b"
  if (Test-Path "$out\${z}_seed$s\model_final.pt") { continue }
  python scripts/train_model.py --condition regular --size $z --seed $s --fresh_windows --corpus_file "$c\seed$s.txt" --window_order "$c\order_seed$s.npy" --tokenizer data/tokenizers/amendment9_tokenizer.json --untie_embeddings --save_every 1000 --output_dir $out --run_name "${z}_seed$s"
}}}
```
Window 2: the same block with `foreach ($s in 2718,3141,1618,1414,1732)`. Count (must print 60):
```powershell
@(Get-ChildItem -Recurse -Filter model_final.pt "E:\AI Checkpoints\overhypothesis_a10").Count
```

### A3. Score and test (CPU)
```powershell
python scripts/amendment9_manifest.py --design a10 --ckpt_dir "E:\AI Checkpoints\overhypothesis_a10" --corpus_dir E:\overhypothesis_a10 --per_cell 3 --seeds 42 123 456 789 1001 2718 3141 1618 1414 1732 --out results/amendment10/manifest_a.csv
python scripts/familiarity_eval.py --manifest results/amendment10/manifest_a.csv --out results/amendment10/scores_a.csv
python scripts/a10_tests.py partA --scores results/amendment10/scores_a.csv --out results/amendment10/RESULTS_A.md
git add results/amendment10
git commit -m "Amendment 10 Part A results"
git pull --rebase --autostash origin study1b-1c-evaluation
git push origin HEAD:study1b-1c-evaluation
git push origin HEAD:master
```

## Part C (PC, CPU, about an hour; can run while Part A trains)
```powershell
python scripts/a9_rowswap.py --manifest results/amendment9/manifest.csv --out results/amendment10/rowswap.csv
python scripts/a10_tests.py partC --rowswap results/amendment10/rowswap.csv --out results/amendment10/RESULTS_C.md
git add results/amendment10
git commit -m "Amendment 10 Part C results"
git pull --rebase --autostash origin study1b-1c-evaluation
git push origin HEAD:study1b-1c-evaluation
git push origin HEAD:master
```
The script stops if the 'own' rows do not reproduce the Amendment 9 scores.

## Part B (Mac, CPU)

### B0. Set up (once)
```bash
git clone https://github.com/synthiumjp/overhypothesis.git
cd overhypothesis
git checkout study1b-1c-evaluation
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers pandas scipy statsmodels accelerate
```
Then the checksum line above (with `python`).

### B1. Pilot (Pythia-410M; never in a test)
```bash
python scripts/pythia_familiarity.py --model EleutherAI/pythia-410m-deduped --out results/amendment10/pythia_pilot.csv
python scripts/a10_tests.py pilotB --scores results/amendment10/pythia_pilot.csv
git add results/amendment10 && git commit -m "Amendment 10 Part B pilot" && git push origin HEAD:study1b-1c-evaluation && git push origin HEAD:master
```
Send the printed line. The smallest effect of interest it gives is recorded before the primary models run.

### B2. Primary and descriptive models (about a day)
```bash
for m in 1b 1.4b 2.8b 6.9b 12b 70m 160m; do
  python scripts/pythia_familiarity.py --model EleutherAI/pythia-$m-deduped --out results/amendment10/pythia_scores.csv
done
for st in 1000 2000 4000 8000 16000 32000 64000 143000; do
  python scripts/pythia_familiarity.py --model EleutherAI/pythia-1.4b-deduped --revision step$st --out results/amendment10/pythia_checkpoints.csv
done
python scripts/a10_tests.py partB --scores results/amendment10/pythia_scores.csv --pilot results/amendment10/pythia_pilot.csv --out results/amendment10/RESULTS_B.md
git add results/amendment10 && git commit -m "Amendment 10 Part B results" && git push origin HEAD:study1b-1c-evaluation && git push origin HEAD:master
```
If a model cannot load (memory), skip it and say so; the decision rules use the primary models that ran (at least four).
