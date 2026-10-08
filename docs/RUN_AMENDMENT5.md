# Running the Amendment 5 extensions

Run nothing below until Amendment 5 is registered on OSF. Pull the branch first (`git pull`), then check that the locked files match `docs/amendment5_checksums.txt`:

```
python -c "import hashlib,sys;[print(hashlib.md5(open(l.split()[1],'rb').read()).hexdigest()==l.split()[0], l.split()[1]) for l in open('docs/amendment5_checksums.txt')]"
```

Every line must print `True`. The repo now has a `.gitattributes` that stops Git from changing line endings. If an older checkout prints `False` for a file you have not edited, refresh it with `git rm --cached -r .` then `git reset --hard` and check again.

## Part A. Training PC (Windows, from C:\overhypothesis)

The new corpora are in the repo (`data/corpora_x/`). Each folder holds `regular_seed<s>.txt`, so `train_model.py` runs unchanged with the Regular tokenizer.

### A1. Train 120 models (8 corpora x 3 sizes x 5 seeds)

```powershell
$corpora = "nostable_random","nostable_grouped","colourstable_random","colourstable_grouped","graded_b1p4","graded_b3p6","graded_b7","graded_b12"
foreach ($c in $corpora) { foreach ($z in "tiny","small","medium") { foreach ($s in 42,123,456,789,1001) {
  python scripts/train_model.py --condition regular --size $z --seed $s `
    --corpus_dir data/corpora_x/$c --output_dir models/checkpoints_x/$c
}}}
```

About 3.5 hours per corpus on the Study 1c set-up, so about 28 hours in all. If time runs short, finish the controls first (registered order: the four control corpora, then b3p6, b1p4, b7, b12).

### A2. Score the new models on the Study 1 items (novel nouns)

```powershell
foreach ($c in $corpora) {
  python scripts/oneshot_eval.py make-manifest --base_dir . --ckpt_dir models/checkpoints_x/$c `
    --conditions regular --label_suffix "_$c" --out results/study1x/manifest_$c.csv
  python scripts/oneshot_eval.py eval-lm --items data/wug_tests/oneshot_items.csv `
    --manifest results/study1x/manifest_$c.csv --out results/study1x/raw/lm_$c.csv
}
python scripts/oneshot_eval.py summarise --inputs "results/study1x/raw/lm_*.csv" --out results/study1x/summary
```

### A3. Trained-noun items (S14), all Study 1 models and the new ones

```powershell
python scripts/study1_trained_nouns.py eval --ckpt_dir models/checkpoints --family regular --label regular --out results/study1x/raw/trained.csv
python scripts/study1_trained_nouns.py eval --ckpt_dir models/checkpoints --family scrambled --label scrambled --out results/study1x/raw/trained.csv
python scripts/study1_trained_nouns.py eval --ckpt_dir <Study 1c checkpoint folder> --family regular --label regular_bursty --out results/study1x/raw/trained.csv
python scripts/study1_trained_nouns.py eval --ckpt_dir <Study 1c checkpoint folder> --family scrambled --label scrambled_bursty --out results/study1x/raw/trained.csv
foreach ($c in $corpora) {
  python scripts/study1_trained_nouns.py eval --ckpt_dir models/checkpoints_x/$c --family regular `
    --run_prefix regular --tokenizer regular --label "regular_$c" --out results/study1x/raw/trained.csv
}
python scripts/oneshot_eval.py summarise --inputs results/study1x/raw/trained.csv --out results/study1x/trained
```

### A4. Ideal observer on the new corpora

```powershell
foreach ($c in $corpora) { foreach ($s in 42,123,456,789,1001) {
  python scripts/oneshot_eval.py eval-hbm --items data/wug_tests/oneshot_items.csv `
    --corpus data/corpora_x/$c/regular_seed$s.txt --condition $c --seed $s --out results/study1x/hbm/hbm_${c}_$s.csv
}}
```

### A5. Registered tests

```powershell
python scripts/study1x_tests.py --runs results/study1c/summary_per_run.csv results/study1x/summary_per_run.csv `
  --trained results/study1x/trained_per_run.csv --out results/study1x/tests
```

Zip `results/study1x` and send it.

## Part B. Mac (repo root, `.venv312` active)

```
python scripts/study2_extension.py make-manifests --out_dir results/study2x
python scripts/english_oneshot.py eval --items data/wug_tests/english_items_v2.csv \
  --manifest results/study2x/models_final_small.csv --out results/study2x/scores_final.csv
python scripts/english_oneshot.py eval --items data/wug_tests/english_items_v2_M.csv --backend mlx \
  --mlx_memory_gb 200 --manifest results/study2x/models_final_large.csv --out results/study2x/scores_final_large.csv
python scripts/english_oneshot.py eval --items data/wug_tests/english_items_v2_M.csv \
  --manifest results/study2x/models_dense.csv --out results/study2x/scores_dense.csv --delete_cache
python scripts/english_oneshot.py eval --items data/wug_tests/english_items_v2_M.csv \
  --manifest results/study2x/models_poly.csv --out results/study2x/scores_poly.csv --delete_cache
python scripts/study2_extension.py word-tokens --out results/study2x/word_tokens.csv
python scripts/study2_extension.py analyse --scores results/study2x/scores_final.csv \
  results/study2x/scores_final_large.csv results/study2x/scores_dense.csv results/study2x/scores_poly.csv \
  --out results/study2x/tests
```

The 70B models are scored on set M only (1,152 items), the others on all four sets. Use the memory guard and the same MLX settings as in Study 2. Zip `results/study2x` and send it.

## Part C. PC that holds `data/cds` (WSL or Windows)

1. Download from TalkBank CABank, English: SBCSAE, CallHome English (eng), CallFriend English (eng-n, eng-s). Note each zip's URL and MD5.
2. `pip install spacy` and `python -m spacy download en_core_web_sm` (version 3.8.0).
3. Run the commands in the header of `scripts/study3_extension.py`, in order.
4. Hand coding: open `results/study3x/coding_sheet.csv` and follow `docs/exemplar_coding_rubric.md`. Do not open `coding_sheet_key.csv` until all rows are coded.

Zip `results/study3x` (without transcripts) and send it.
