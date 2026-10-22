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

## Noun-familiarity curve (blended embeddings)

Scores 75 models (five conditions) at three blends between the original (0) and typical (1)
novel-noun embeddings; blends 0 and 1 come from the run above. About 40 minutes on CPU; resumable.

```powershell
git pull
python scripts/recut_rescore.py --manifests results/amendment8/manifest_regular_random.csv results/amendment8/manifest_regular_grouped.csv results/amendment8/manifest_objects_same.csv results/amendment8/manifest_objects_diff.csv results/amendment8/manifest_reassigned_grouped.csv --alphas 0.25,0.5,0.75 --out results/recut/rescore_blend.csv --norms_out results/recut/norms.csv
python scripts/recut_blend_analysis.py --out results/recut/BLEND.md
git add results/recut/BLEND.md results/recut/BLEND_per_seed.csv results/recut/norms.csv
git commit -m "Recut noun-familiarity curve (exploratory)"
git push origin HEAD:study1b-1c-evaluation
git push origin HEAD:master
```
