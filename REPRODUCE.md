# Reproducing the paper

Paper: "Different exemplars, close together: how the arrangement of training text teaches language models which features persist" (Computational Linguistics, short paper, submitted). All registrations: osf.io/qj9hb.

The repository uses working names from the registrations. **Paper Study 1** (small models trained on artificial languages) = repo Studies 1b, 1c, 1x (Amendment 5 Part A), 1w (Amendment 6) and 1o (Amendment 7); its scores are in `results/rescore/` and `results/study1o/`. **Paper Study 2** (clustering of natural input) = repo Study 1e and `study3x` (Amendment 5 Part C). **Paper Study 3** (pretrained models) = repo Study 1d and `study2x` (Amendment 5 Part B). Amendment 5 itself calls the pretrained-model study "Study 2" and the corpus study "Study 3".

**Corrected scores.** Study 1 scores were corrected on 12 Oct 2026 (model-loading error; `results/rescore/RESULTS.md`). Every Study 1 number in the paper comes from `results/rescore/`, `results/study1o/` or `results/cl_revision/`. Folders with older scores carry a `SUPERSEDED.md` note.

## The paper itself

The LaTeX source, bibliography and compiled PDF are in `paper_latex/` (build with `latexmk -pdf paper.tex` there; see `paper_latex/README.md`).

## Paper tables and figures

| Paper | Command (from the repo root) | Inputs | Output |
|---|---|---|---|
| Table 1 (corpora, measured B) | `python scripts/corpus_generator.py`, `scripts/make_bursty_corpus.py`, `scripts/make_study1x_corpora.py`, `scripts/make_study1o_corpora.py build --f_b36 0.11` (corpora are committed; regenerate to check) | | `data/corpora*/`; B in `data/corpora_x/manifest.json`, `data/corpora_o/manifest.json` |
| Table 2, Table D1, Table E2, Table F1 | `python scripts/paper_tables.py` | `results/rescore/*_per_run.csv`, `results/study1o/summary_per_run.csv`, `results/*/trained_per_run.csv` | `results/paper_tables/` |
| Table 3 (natural corpora) | `python scripts/cl_revision_analyses.py` (B, intervals, effective clustering); counts and coded shares from the Study 1e and Study 3 extension steps below | `results/study1e/counts.csv`, `results/study3x/` | `results/cl_revision/summary.json`; `results/study3x/rates.csv` |
| Table 4, Table A4, Table E1 (registered tests, seed-level tests) | registered tests: the locked test scripts in the steps below; for Study 1 they are rerun on the corrected scores with the commands in `results/rescore/RESULTS.md`; seed-level tests and noun-specific equivalence: `python scripts/cl_revision_analyses.py` | per-run files above | `results/rescore/*tests*.json`, `results/study1o/tests.json`, `results/study1d/tests_final.json`, `results/study2x/`, `results/study3x/tests.json`; `results/cl_revision/seed_level_tests.csv`, `noun_specific_by_condition.csv` |
| Table A3 (before and after the correction) | old scores: `results/study1b`, `study1c`, `study1x`, `study1w`; corrected: `results/rescore/` | | `results/rescore/RESULTS.md` |
| Table C1 (training settings) | `scripts/train_model.py` (`MODEL_CONFIGS`, `TRAIN_CONFIG`); training driver `scripts/run_phase2.py`; tokenisers `scripts/train_all_tokenizers.py` | | `results/train_logs/` (Study 1b models) |
| Table E3 (what the corpora reward) | `python scripts/corpus_conditionals.py` | `data/corpora*/` | `results/cl_revision/corpus_conditionals.csv` |
| Noun copy check (Section 3.5, Appendix E) | `python scripts/noun_copy_check.py ...` (command in `docs/RUN_NOUN_COPY.md`; needs the checkpoints) | checkpoints | `results/cl_revision/noun_copy_check.csv` |
| Figure 1 | schematic with illustrative sentences; no data | | |
| Figures 2, 3, 4 | `python scripts/make_paper_figures.py` | listed in the script header | `results/paper_figures/` (CSV of every plotted value, and PNGs) |

Python packages: `requirements.txt`. Zipf frequencies of the Study 3 word lists (Appendix G): `pip install wordfreq`, then the mean of `wordfreq.zipf_frequency(word, 'en')` over each list in `scripts/english_oneshot.py` (3.75, 3.82, 3.70). Files under `docs/RUN_*.md` are the run logs written for the training PC and Mac; their paths (for example `E:\AI Checkpoints`) are machine-specific.

**Changes to locked scripts after registration.** The MD5 checks in `docs/amendment5_checksums.txt` and `docs/amendment6_checksums.txt` fail for three scripts, each changed after locking for a documented reason:
- `scripts/english_oneshot.py`: commits 3c71203 and 0aeb583 (9 Oct 2026), MLX loading options for the 70B models only; scoring is unchanged.
- `scripts/train_model.py`: commit 3c4ce45 (11 Oct 2026), the window-length option locked for Amendment 6; 128-token training is unchanged.
- `scripts/oneshot_eval.py`: commit 483ad6b (12 Oct 2026), the loader correction; `--legacy_tied_load` reproduces the old scoring.

## Fixed inputs (verify before anything else)

| File | MD5 | Made by |
|---|---|---|
| `data/wug_tests/oneshot_items.csv` | `6d1b6064392d1395cb217af494d240c7` | `scripts/oneshot_eval.py make-items` (seed 2026) |
| `data/wug_tests/english_items.csv` | `687ff90580ea03a6d76aba55499e7835` | `scripts/english_oneshot.py make-items` (seed 2027) |
| `data/corpora/*.txt` | `data/corpora/checksums_md5.txt` | `scripts/corpus_generator.py` |
| `data/corpora_bursty/*.txt` | `data/corpora_bursty/bursty_checksums.json` | `scripts/make_bursty_corpus.py` |
| `results/study1d/models.csv` | `2d8debf6a3d3cd5ce52aabf2b24107e3` | registered model list |

The bursty corpora regenerate byte-identically: `python scripts/make_bursty_corpus.py` and compare with the checksum file.

## Study 1b and 1c (synthetic corpora)

| Step | Command | Output |
|---|---|---|
| Ideal observer, replication-corrected | `python scripts/oneshot_eval.py eval-hbm --items data/wug_tests/oneshot_items.csv --corpus data/corpora/<cond>_seed<s>.txt --condition <cond> --seed <s> --out results/study1b/hbm_corrected/hbm_<cond>_<s>.csv` | `results/study1b/hbm_corrected/` |
| Score trained models | `python scripts/oneshot_eval.py make-manifest ...` then `eval-lm` (needs the checkpoints, see below) | `results/study1b/raw/lm_results.csv.gz`, `random_results.csv.gz` (untrained floor), `results/study1c/raw/lm_results.csv.gz`; checksums in `results/raw_checksums_md5.txt` |
| Per-run summary | `python scripts/oneshot_eval.py summarise --inputs study1c_lm_results.csv study1b_lm_results.csv --out results/study1c/summary` (unzip first; reproduces the committed file exactly) | `results/study1c/summary_per_run.csv` (MD5 `4337444288f38c22b42ac02983d43fc4`) |
| Checkpoint tracking (registered exploratory) | `make-manifest --ckpt_name "ckpt_step*.pt"` then `eval-lm --ks 1`; analysed by `scripts/exploratory_analyses.py` | `results/study1c/raw/track_results.csv.gz`, `results/study1c/tracking_trajectory.csv` |
| Registered tests H1b to H3b (Study 1b, random order) | `python scripts/study1b_tests.py --runs results/study1c/summary_per_run.csv --out results/study1b/tests_output.txt` | `results/study1b/tests_output.txt` |
| Registered tests H4b, H5b | `python scripts/study1c_tests.py --runs results/study1c/summary_per_run.csv --hbm "results/study1c/hbm/hbm_*.csv"` | `results/study1c/tests_output.txt` |

## Study 1d (pretrained models, natural English)

| Step | Command | Output |
|---|---|---|
| Score models | `python scripts/english_oneshot.py eval --items data/wug_tests/english_items.csv --manifest results/study1d/models.csv --out scores.csv` (add `--backend mlx` for the 70B models) | item-level scores |
| Merged scores used in the paper | assembled as described under Deviations | `results/study1d/scores_merged.csv` (MD5 `38474c067938ffb01079635346e56a00`, 81 checkpoints) |
| Registered tests H6 to H8 | `python scripts/study1d_tests.py --scores results/study1d/scores_merged.csv --out results/study1d/tests_final` | `results/study1d/tests_final.json`, `tests_final_secondary.csv` |
| Summary | | `results/study1d/RESULTS.md` |

## Study 1 extension (Amendment 5, Part A)

| Step | Command | Output |
|---|---|---|
| Corpora | `python scripts/make_study1x_corpora.py` (already in the repo) | `data/corpora_x/`, `manifest.json` (measured B) |
| Train, score, ideal observer | `docs/RUN_AMENDMENT5.md`, A1 to A4 (needs the checkpoints) | `results/study1x/raw/*.csv.gz` (MD5s of the uncompressed files in `raw_md5.txt`), `hbm/` |
| Per-run summaries | `python scripts/oneshot_eval.py summarise --inputs "results/study1x/raw/lm_*.csv.gz" --out results/study1x/summary`; the same with `trained.csv.gz` and `--out results/study1x/trained` | `summary_per_run.csv`, `trained_per_run.csv` |
| Registered tests H11, H12, H13a, H13b | `python scripts/study1x_tests.py --runs results/study1c/summary_per_run.csv results/study1x/summary_per_run.csv --trained results/study1x/trained_per_run.csv --out results/study1x/tests` | `tests.json` |
| H13b equivalence (registered in the text of Amendment 5) | `python scripts/study1x_tost.py --runs results/study1c/summary_per_run.csv results/study1x/summary_per_run.csv --out results/study1x/tost_H13b.json` | `tost_H13b.json` |
| Exploratory | `python scripts/study1x_exploratory.py --runs results/study1c/summary_per_run.csv results/study1x/summary_per_run.csv --hbm "results/study1x/hbm/hbm_*.csv" --out results/study1x/exploratory` | `exploratory_*` |
| Summary | | `results/study1x/RESULTS.md` |

## Study 1 window test (Amendment 6)

| Step | Command | Output |
|---|---|---|
| Window measure and prediction (before training) | `python scripts/window_cooccurrence.py measure ...` and `predict ...` (usage in its header) | `results/study1w/cooccurrence.csv`, `prediction.json` |
| Train and score | `docs/RUN_AMENDMENT6.md`, W1 to W3 (needs the checkpoints) | `results/study1w/raw/*.csv.gz` (MD5s in `raw_md5.txt`) |
| Registered tests H17 to H19, S24 to S28 | see `results/study1w/RESULTS.md` | `tests.json`, `tost.json` |
| Summary | | `results/study1w/RESULTS.md` |

## Study 1, same-object vs different-object clustering (Amendment 7)

| Step | Command | Output |
|---|---|---|
| Corpora | `python scripts/make_study1o_corpora.py build --f_b36 0.11` (already in the repo; `check` verifies) | `data/corpora_o/`, `manifest.json` |
| Train and score | `docs/RUN_AMENDMENT7.md` | `results/study1o/raw/*.csv.gz` (MD5s in `raw_md5.txt`) |
| Registered tests H20 to H22, S29 to S34 | see `results/study1o/RESULTS.md` | `tests.json` |
| Loss check of the corrected loader | `python scripts/loss_check.py ...` | `results/rescore/loss_check.csv` |
| Effective B (exploratory) | `python scripts/effective_b.py --out results/rescore/effective_b.json` | `effective_b.json` |

## Study 2 extension (Amendment 5, Part B)

| Step | Command | Output |
|---|---|---|
| Items | `python scripts/study2_extension.py make-items --out data/wug_tests/english_items_v2.csv`; set M subset `english_items_v2_M.csv` | checksums in `docs/amendment5_checksums.txt` |
| Manifests | `python scripts/study2_extension.py make-manifests --out_dir results/study2x` | `models_*.csv` |
| Score | `python scripts/english_oneshot.py eval --items ... --manifest results/study2x/models_<x>.csv --out ... --delete_cache` (70B: `--backend mlx --mlx_wired_gb 170 --max_batch 4 --mlx_pause 0.05`, alone on the GPU) | `results/study2x/scores_*.csv.gz` (MD5s of the uncompressed files in `scores_md5.txt`) |
| Registered tests H14a, H14b, H15 | `python scripts/study2_extension.py analyse --scores results/study2x/scores_final.csv.gz results/study2x/scores_final_large.csv.gz results/study2x/scores_dense.csv.gz results/study2x/scores_poly.csv.gz --out results/study2x/tests` | `tests.json`, `tests_summary.csv`, `tests_kind_specific_items.csv.gz` |
| Secondary S15 to S17 | `python scripts/study2x_secondary.py --summary results/study2x/tests_summary.csv --out results/study2x/secondary` | `secondary_*.csv` |
| Summary | | `results/study2x/RESULTS.md` |

## Study 1e (child-directed speech)

| Step | Command | Output |
|---|---|---|
| CHILDES | download `0-Eng-NA-MOR.zip` from talkbank.org/childes/access/Eng-NA/ (MD5 `789a8815b2c222917b68be07b7246425`), then `python scripts/cds_burstiness.py prepare-childes --zip Eng-NA.zip --out data/cds/childes.jsonl` | 3,134 transcripts |
| Adult text | `python scripts/cds_burstiness.py prepare-adult --out data/cds/pile10k.jsonl` | 9,363 documents |
| Synthetic reference | `python scripts/cds_burstiness.py prepare-synth --out data/cds/synth.jsonl` | 10 corpora |
| Counts | `python scripts/cds_burstiness.py measure --inputs data/cds/childes.jsonl data/cds/pile10k.jsonl data/cds/synth.jsonl --out results/study1e/counts.csv` | `counts.csv`, `counts_by_noun.csv` |
| Registered tests H9, H10 | `python scripts/cds_burstiness.py test --counts results/study1e/counts.csv --out results/study1e/tests` | `tests.json`, `tests_secondary.csv` |
| Summary | | `results/study1e/RESULTS.md` |

CHILDES transcripts are not redistributed here (TalkBank terms); the per-document counts in `counts.csv` are.

## Study 3 extension (Amendment 5, Part C)

| Step | Command | Output |
|---|---|---|
| CABank | sign in at talkbank.org, download transcripts from the SBCSAE, CallHome/eng, CallFriend/eng-n and CallFriend/eng-s pages, save as `SBCSAE.zip`, `CallHome-eng.zip`, `CallFriend-eng-n.zip`, `CallFriend-eng-s.zip` (MD5s in `results/study3x/cabank_md5.txt`); then `python scripts/study3_extension.py prepare-cabank --zips ... --out data/cds/cabank.jsonl` | 277 transcripts |
| All steps | the commands in the header of `scripts/study3_extension.py`, in order (spaCy 3.8, en_core_web_sm 3.8.0) | `results/study3x/` |
| Transcription-mark check (exploratory) | `python scripts/study3_ca_sensitivity.py ...` (usage in its header) | `results/study3x/ca_sensitivity/` |
| Hand coding | code `coding_sheet.csv` with `tools/coding_tool.html`; second-coder subset `python scripts/study3_reliability.py subset`; agreement `... agree` | `reliability.json` |
| Summary | | `results/study3x/RESULTS.md` |

The coding sheet holds transcript excerpts and is not in git. The blind second coding sheet (MD5 `24729a7ec2b29ed02c7eb7175e6b66f3`) was sealed until the first coder had finished; the resulting codes are in `results/study3x/codes_second.csv` and agreement in `reliability.json`. The sheet itself holds transcript excerpts and is not public.

## Exploratory and robustness analyses (all studies)

`python scripts/exploratory_analyses.py` writes:

- `results/study1b/hbm_corrected_summary.csv`: corrected ideal observer by condition (promised in Amendment 1)
- `results/study1c/kind_specific_by_dim.csv`: same-kind minus other-kind repetition per dimension, Studies 1b and 1c
- `results/study1d/kind_specific_final.csv`, `kind_specific_trajectory.csv`: the same for Study 1d
- `results/study1d/robustness_*.csv`: leave one noun out, raw deltas, sentence order (promised in Amendment 3), word effects

## Deviations and their checks (Study 1d)

| Deviation | Check | Files |
|---|---|---|
| pythia-2.8b steps 0 to 16,000 loaded a stale copy of the final weights from the Hub; rescored from the per-step shards | identical scores across checkpoints before the fix; file hashes on the Hub | `results/study1d/raw/` (before), `results/study1d/fix28b/` (after) |
| pythia-2.8b step 64,000 identical to step 143,000 on the Hub; dropped | same SHA-256 on the Hub | |
| 70B models and checks run under Python 3.12 / PyTorch 2.13 (others under 3.14 / 2.14) | rescoring two checkpoints: mean abs. difference 0.0003 and 0.005 nats | `results/study1d/qwen/val_torch.csv` |
| Qwen2.5-72B and Llama-3.1-70B scored with MLX | MLX vs PyTorch: 0.003 nats (pythia-410m), 0.010 nats (Qwen2.5-0.5B) | `results/study1d/qwen/val_mlx.csv`, `val_torch.csv` |

## Not in this repository

- Trained Study 1 models: final weights (`model_final.pt`), `model_config.json` and `train_log.json` for all 390 models are on Zenodo, https://doi.org/10.5281/zenodo.23338406 (about 30 GB in five archives named after the checkpoint folders: `overhypothesis_checkpoints`, `_bursty`, `_x`, `_w512`, `_o`; MD5 of every file in `checksums_md5.txt` there). Scripts that need the checkpoints take the folder as `--ckpt_dir`.
- Intermediate checkpoints (`ckpt_step*.pt`) were not archived. The 461 used for the learning curves (listed in `results/study1c/raw/track_manifest_regular.csv` and `track_manifest_bursty.csv`) are kept by the author and available on request. Their scores are in `results/rescore/`.
- Thirty runs trained on 25% or 50% of the corpus, and one pilot, from the original registration, are not used in the paper and not archived.
- Pretrained models for Study 1d are public on Hugging Face at the revisions in `results/study1d/models.csv`.
