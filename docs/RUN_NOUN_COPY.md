# Noun copy check (exploratory, CL revision)

Can the Study 1 models tell the novel test nouns apart? PC, C:\overhypothesis, venv active.

```powershell
git pull
python scripts/noun_copy_check.py --manifest results/study1b/raw/manifest.csv results/study1c/raw/manifest.csv results/study1o/manifest_same_grouped.csv results/study1o/manifest_diff_grouped.csv --conditions regular regular_bursty regular_o_same_grouped regular_o_diff_grouped --out results/cl_revision/noun_copy_check.csv
```

60 models; a few minutes. Send `results/cl_revision/noun_copy_check.csv`.

## Result (13 Oct 2026)

Run on the PC; output committed as `results/cl_revision/noun_copy_check.csv` (60 rows).

- Copy preference was within ±0.033 nats in every model (condition means −0.002 to 0.000), and positive on 42% to 60% of items. No model predicts a novel noun again after reading it.
- Caveat: the novel nouns are never training targets, and the checkpoints hold separate input and output word matrices, so this check also depends on untrained output weights. It is weaker evidence than the next point.
- The decisive contrast is already in the trained-noun results. After grouped same-object training, the noun-specific index is −1.06 on trained nouns (0 of 15 models above zero; `results/study1o/trained_per_run.csv`) but +0.04 (90% interval −0.09 to 0.18) on novel nouns (`results/cl_revision/noun_specific_by_condition.csv`). The same models track a trained noun's recurrence and not a novel noun's.
- Conclusion for the paper: the registered novel-noun test is uninformative about whether the expectation is tied to the noun. The noun claim rests on trained nouns: in every condition with a trained-noun index above 0.3 nats, most of the index remains after a different noun (grouped Regular 1.03 same noun, 0.73 different noun; noun-specific 0.30, two-sided p = .015).
- A hidden-state probe for "same noun or not" was tried and dropped: an untrained model already decodes it at about 92% accuracy, so it cannot show what training added.
