# Submitted predictions

The 12 prediction files submitted to Kaggle, in the official `PassengerId,Transported` format. They are predictions, not hidden test labels. Descriptions, SHA256 hashes and positive rates are in [submission-manifest.json](../docs/evidence/submission-manifest.json); public scores and submission IDs are in [kaggle-submissions.csv](../docs/evidence/kaggle-submissions.csv).

| File | Public | Note |
|---|---|---|
| `baseline_catboost.csv` | 0.80780 | 1st submission, CatBoost (optimistic stopping) |
| `hm10_innercv_logloss.csv` | 0.81131 | 2nd, CatBoost with honest stopping (CV-promoted) |
| `ha12_tabpfn.csv` | 0.81786 | 3rd, frozen TabPFN v3.5 (CV-promoted) |
| `ha27_stack3.csv` | 0.82020 | 4th, nested stack of 3 TabPFN variants (CV-promoted, official champion) |
| `sub_stack3_plus_a1ne2.csv` | 0.82604 | best of the final CV-ranked batch of 8 |
| `sub_a1mix_stack.csv` | 0.82487 | batch |
| `sub_avg4.csv` | 0.82440 | batch |
| `sub_combo7.csv` | 0.82417 | batch (highest CV) |
| `sub_combo5.csv` | 0.82417 | batch |
| `sub_stack_a1ctx.csv` | 0.82253 | batch |
| `sub_combo7_bag3.csv` | 0.82137 | batch |
| `sub_a1mix_single.csv` | 0.82066 | batch |
