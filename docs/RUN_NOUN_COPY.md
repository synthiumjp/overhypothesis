# Noun copy check (exploratory, CL revision)

Can the Study 1 models tell the novel test nouns apart? PC, C:\overhypothesis, venv active.

```powershell
git pull
python scripts/noun_copy_check.py --manifest results/study1b/raw/manifest.csv results/study1c/raw/manifest.csv results/study1o/manifest_same_grouped.csv results/study1o/manifest_diff_grouped.csv --conditions regular regular_bursty regular_o_same_grouped regular_o_diff_grouped --out results/cl_revision/noun_copy_check.csv
```

60 models; a few minutes. Send `results/cl_revision/noun_copy_check.csv`.
