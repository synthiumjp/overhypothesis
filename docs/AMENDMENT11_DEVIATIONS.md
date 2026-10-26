# Amendment 11 deviations

Registered 25 October 2026 (about 20:40 AEDT) as osf.io/8ws5c, within project osf.io/qj9hb, from commit 24bbbcb.
No Amendment 11 model had been trained when it was registered.

Record here, with date, every retrained or excluded model and any other departure from docs/AMENDMENT11.md.

- 26 October 2026, 19:23 AEDT (Part B, Mac): after all ten large models had finished, the training loop was started
  again by mistake. It began retraining seed 3037 and was stopped at step 1. Only that run's model_config.json was
  rewritten, by the same command, so its content is unchanged; model_final.pt, train_log.json and all weight files
  of seed 3037 kept their original times (09:42 and earlier). No model was retrained, and no score was affected.
