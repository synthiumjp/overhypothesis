# Exploratory rescoring for the ACL recut (not registered)

Two checks on the 120 fresh-text models, scoring only (no training):
1. novel test nouns given typical embeddings instead of untrained random ones;
2. Delta measured on the changed word only.

PC, C:\overhypothesis, venv active (`.\.venv\Scripts\Activate.ps1`). The fresh models must be in
`E:\AI Checkpoints\overhypothesis_fresh\` (the paths in results/amendment8/manifest_*.csv).

```powershell
git pull
python scripts/recut_rescore.py --manifests "results/amendment8/manifest_*.csv" --limit 1 --out results/recut/rescore_smoke.csv
Remove-Item results\recut\rescore_smoke.csv
python scripts/recut_rescore.py --manifests "results/amendment8/manifest_*.csv" --out results/recut/rescore_raw.csv
python scripts/recut_rescore_analysis.py --raw results/recut/rescore_raw.csv --out results/recut/RESCORE.md
```

The first command scores one model as a check (well under a minute). The full run scores 120 models
under two embedding conditions (roughly an hour on CPU) and can be stopped and restarted; it resumes.
The analysis prints a table and writes results/recut/RESCORE.md. Its first line must report that the
original-embedding scores reproduce results/amendment8/raw (max difference about 1e-4 or less).

Send back: results/recut/RESCORE.md and results/recut/RESCORE_per_seed.csv (or zip results/recut).
