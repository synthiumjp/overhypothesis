# Different exemplars, close together

Code, data and analysis outputs for the paper "Different exemplars, close together: how the arrangement of training text teaches language models which features persist" (Jon-Paul Cacioli; manuscript submitted for publication).

All registrations are in one OSF project: [osf.io/qj9hb](https://osf.io/qj9hb).

## What the paper does

- **Study 1.** 390 small transformers (3.4M, 11.0M and 25.6M parameters) trained on artificial languages in which every kind has one fixed feature. The content is held fixed and only the arrangement of sentences changes (random order, graded clustering, grouped runs, runs of the same object or of different objects). After one sentence about a new object, the models are tested on whether they expect its shape, rather than its colour, to repeat.
- **Study 2.** The same clustering statistic, B, measured in speech recorded around young children (CHILDES), adult conversation (CABank) and written text (the Pile), with hand coding of what nearby repeated labels refer to.
- **Study 3.** The same test on pretrained models: Pythia (14M to 12B, with checkpoints and extra training seeds), the BabyLM GPT-2 baselines, Qwen2.5 72B and Llama 3.1 70B.

## Reproducing the paper

The paper's LaTeX source and PDF are in `paper_latex/`. `REPRODUCE.md` maps every table and figure of the paper to the command and inputs that produce it. The quickest checks:

```
pip install -r requirements.txt
python scripts/paper_tables.py         # Tables 2, D1, E2, F1 -> results/paper_tables/
python scripts/make_paper_figures.py   # Figures 2-4 -> results/paper_figures/
python scripts/corpus_conditionals.py  # Table E3
python scripts/cl_revision_analyses.py # seed-level tests, corpus intervals, effective clustering
cd data/corpora && md5sum -c checksums_md5.txt
```

Scoring needs the trained models. Their final weights and training logs are on Zenodo: [doi.org/10.5281/zenodo.23338406](https://doi.org/10.5281/zenodo.23338406).

## Names used in the repository

The repository keeps the working names used in the registrations:

| Paper | Repository | Registration |
|---|---|---|
| Study 1, random and grouped orders | Studies 1b and 1c | Amendments 1 and 2 |
| Study 1, graded orders and controls | `study1x` | Amendment 5, Part A |
| Study 1, 512-token window | `study1w` | Amendment 6 |
| Study 1, same-object and different-object runs | `study1o` | Amendment 7 |
| Study 2 (natural input) | Study 1e, `study3x` | Amendment 4; Amendment 5, Part C |
| Study 3 (pretrained models) | Study 1d, `study2x` | Amendment 3; Amendment 5, Part B |

Study 1 scores were corrected on 12 Oct 2026 after a model-loading error was found (`results/rescore/RESULTS.md`). The paper uses only corrected scores. Folders that hold older scores carry a `SUPERSEDED.md` note and are kept as a record.

## Layout

```
paper_latex/       LaTeX source, bibliography and PDF of the paper
data/
  corpora/, corpora_bursty/, corpora_x/, corpora_o/   Study 1 corpora, with manifests and checksums
  tokenizers/                                          tokenisers (every artificial word is one token)
  wug_tests/                                           test items (oneshot_items.csv for Study 1; english_items*.csv for Study 3)
scripts/                                               all code used by the paper (see REPRODUCE.md)
results/
  rescore/        corrected Study 1 scores and registered tests
  study1o/        Study 1 same-object and different-object runs (corrected loader)
  cl_revision/    seed-level tests, corpus statistics, noun checks
  paper_tables/, paper_figures/   the paper's tables and figure data, rebuilt by the scripts above
  study1d/, study2x/              Study 3 (pretrained models)
  study1e/, study3x/              Study 2 (natural corpora: counts, codes, reliability)
  study1/, study1b/, study1c/, study1x/, study1w/, review/, analysis/, figures/   superseded or earlier results (see SUPERSEDED.md in each)
  train_logs/     training logs of the 120 Study 1b models
docs/             registration texts of Amendments 5 to 7, locked checksums, run logs, coding rubric
tools/            browser tool used for hand coding
archive/          scripts of the original registration, kept as a record
```

## Data and terms

CHILDES and CABank transcripts are not redistributed (TalkBank ground rules). The repository holds per-document counts and hand codes without transcript text. Code is under the MIT licence (`LICENSE`); the project's own data under CC BY 4.0 (`LICENSE-DATA.md`).

## Citation

See `CITATION.cff`.
