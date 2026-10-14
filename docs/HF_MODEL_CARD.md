---
license: cc-by-4.0
language:
- en
tags:
- language-acquisition
- overhypothesis
- shape-bias
- training-data-order
- small-transformers
---

# Trained models for "Different exemplars, close together"

Final weights, model settings and training logs for the 390 small transformers of Study 1 in J.-P. Cacioli, "Different Exemplars, Close Together: How the Arrangement of Training Text Teaches Language Models Which Features Persist".

- Code, corpora, tokenisers, test items and all scores: https://github.com/synthiumjp/overhypothesis
- Registrations: https://osf.io/qj9hb

## What the models are

Decoder-only transformers of three sizes (3.4M, 11.0M and 25.6M parameters), each trained from scratch on an artificial language of 32 kinds of object. In most corpora every kind has one fixed feature. The studies change only the arrangement of the training sentences (random order, graded clustering, grouped runs, runs of the same object or of different objects), and test whether a model that sees one sentence about a new object expects its shape, rather than its colour, to repeat. Five seeds per size and condition, 15 models per condition. Training settings are in Appendix C of the paper and in each `model_config.json`.

These are research artefacts for studying what models learn from the arrangement of their training data. They are not general language models.

## Files

73 archives, named after the repository's working folders and numbered. Each archive holds whole runs and can be unpacked on its own (`tar -xf name.tar`). The matching `.list.txt` lists its contents.

| Archives | Conditions | Models |
|---|---|---|
| `overhypothesis_checkpoints_part01`–`22` | random-order Regular and Mixed-slot; the six further content controls (Appendix D) | 120 |
| `overhypothesis_bursty_part01`–`06` | grouped Regular and Mixed-slot | 30 |
| `overhypothesis_x_part01`–`22` | graded orders; No stable feature and Colour stable corpora | 120 |
| `overhypothesis_w512_part01`–`09` | 512-token window | 45 |
| `overhypothesis_o_part01`–`14` | same-object and different-object runs | 75 |

Each run folder (for example `overhypothesis_x/graded_b3p6/regular_small_seed42/`) holds:

- `model_final.pt`: PyTorch checkpoint with `model_state`, `model_cfg`, `train_cfg`, `vocab_size` and the training arguments
- `model_config.json`: model and training settings
- `train_log.json`: loss and learning rate through training

`checksums_md5.txt` gives the MD5 of every file inside the archives.

## Use

Load and score with `scripts/oneshot_eval.py` in the GitHub repository, which also holds the tokenisers (`data/tokenizers/`). `REPRODUCE.md` there gives the commands. Scripts that need the models take the unpacked folder as `--ckpt_dir`.

Scores in the paper use the corrected loader of 12 October 2026 (see `results/rescore/RESULTS.md` in the repository).

## Not included

Intermediate checkpoints were not kept, except the 461 used for the learning curves, which are available from the author. Thirty runs trained on part of the corpus and one pilot, from the original registration, are not used in the paper and are not included.

## Citation

Cacioli, J.-P. (2026). Different exemplars, close together: How the arrangement of training text teaches language models which features persist. Manuscript submitted for publication.
