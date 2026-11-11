# Redrawn-burst control

Not pre-registered. Design, analysis and predictions are fixed here and committed before any model is trained. Added
in answer to a review (docs/reviews/cold_review_v9_9_fable.md, weakness 1).

## Question

Changing-shape bursts keep copying; fixed-shape bursts lose it. Two readings:

- **Useful context.** A burst helps only because the previous sentence predicts the shape.
- **Recurrence.** A burst helps because the noun recurs, whatever the previous sentence says.

Fixed-shape bursts cannot separate these, because they also give memory a shape to store. The control needs bursts in
which the noun recurs, nothing can be stored, and the previous sentence does not predict the shape.

## Design

Script: `scripts/control_redrawn.py`. Everything is as in the Experiment 3 mixed design except the noun histories:
base text, frames, placement, reading order, tokenizer, the 40 nouns, the ten seeds (2029 to 11117), the three sizes
(3.4M, 10.9M, 25.6M), training and test items. The corpus builder of Experiments 2 and 3 is used unchanged.

Cells, three nouns each, at 100 and 1,000 exposures, plus four unseen nouns (18 nouns at 100, 18 at 1,000, as in
Experiment 3):

| Cell | Shape in each sentence | Repeats previous sentence in burst |
|---|---|---|
| Changing, bursts | one shape per burst, changing between bursts | always |
| Half-kept, bursts | first drawn anew; each later one keeps the previous shape with probability 0.5 | about 0.55 |
| Redrawn, bursts | drawn anew for every sentence (the sentences of "changing, spaced", grouped in fours) | about 0.10 (chance) |
| Fixed, bursts | the noun's own shape | always |
| Changing, spaced | drawn anew for every sentence | (no bursts) |
| Fixed, spaced | the noun's own shape | (no bursts) |

All 20 corpora were built and checked before training (`docs/redrawn_corpus_manifest.json`; repeat rates within
bursts: changing 1.00, half-kept 0.535 to 0.565, redrawn 0.091 to 0.113, chance 0.102).

## Analysis

Copy score as in Experiment 3 (same-noun items, trained embeddings). Unit: seed; sizes averaged within seed; t-based
95% intervals over seeds; contrasts paired within model. Models failing the Experiment 3 model check are excluded
(no retraining).

**Primary contrast:** changing bursts minus redrawn bursts at 1,000 exposures.

Secondary, descriptive: redrawn bursts minus changing spaced; half-kept bursts against the other two burst types;
every contrast at 100 exposures and by size.

## Predictions

- **Useful context:** changing bursts copy clearly more than redrawn bursts at 1,000 (interval above zero), redrawn
  bursts copy about as little as changing spaced nouns, and half-kept bursts fall in between.
- **Recurrence:** redrawn bursts copy about as much as changing bursts (interval includes zero), and clearly more than
  changing spaced nouns.

A mixed result (redrawn bursts between changing bursts and changing spaced) would mean both contribute.

## Run (PC)

From `C:\overhypothesis`, with the virtual environment active.

```powershell
git pull --rebase --autostash origin study1b-1c-evaluation
python scripts/control_redrawn.py build --out_dir E:\overhypothesis_ctl
python scripts/control_redrawn.py check --out_dir E:\overhypothesis_ctl --against docs/redrawn_corpus_manifest.json
```
The check must end with `20 of 20 corpora passed every check`. Then train 30 models (about six hours):
```powershell
foreach ($s in 2029,3037,4049,5051,6067,7079,8081,9091,10103,11117) { foreach ($z in "tiny","small","medium") {
  $w = if ($z -eq "small") { "w256k" } else { "w160k" }
  $c = "E:\overhypothesis_ctl\regular_redrawn\$w"
  $out = "E:\AI Checkpoints\overhypothesis_ctl\regular_redrawn"
  if (Test-Path "$out\${z}_seed$s\model_final.pt") { continue }
  python scripts/train_model.py --condition regular --size $z --seed $s --fresh_windows --corpus_file "$c\seed$s.txt" --window_order "$c\order_seed$s.npy" --tokenizer data/tokenizers/amendment9_tokenizer.json --untie_embeddings --save_every 1000 --output_dir $out --run_name "${z}_seed$s"
}}
```
Score and push:
```powershell
python scripts/control_redrawn.py manifest --ckpt_dir "E:\AI Checkpoints\overhypothesis_ctl" --corpus_dir E:\overhypothesis_ctl --out results/control_redrawn/manifest.csv
python scripts/a11_model_check.py --manifest results/control_redrawn/manifest.csv --out results/control_redrawn/model_check.csv
python scripts/control_redrawn.py score --manifest results/control_redrawn/manifest.csv --out results/control_redrawn/scores.csv
python scripts/control_redrawn.py summary --scores results/control_redrawn/scores.csv --model_check results/control_redrawn/model_check.csv --out results/control_redrawn/SUMMARY.md --tex paper_acl/figures/tab_control.tex
git add results/control_redrawn paper_acl/figures/tab_control.tex
git commit -m "Redrawn-burst control results"
git pull --rebase --autostash origin study1b-1c-evaluation
git push origin HEAD:study1b-1c-evaluation
git push origin HEAD:master
```
