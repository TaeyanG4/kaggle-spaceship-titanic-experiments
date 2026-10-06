# Kaggle dataset: saved TabPFN member predictions

Published as `taeyangg4/spaceship-titanic-tabpfn-member-predictions` so the Kaggle notebook can rebuild the submitted stacks without a TabPFN token or GPU.

| File | Rows | Columns |
|---|---|---|
| `member_oof_seed42.csv` | 8,693 train rows | `PassengerId`, `fold` (1-5), `frozen`, `ft`, `ftrefit`, `ft_long_ne2` — out-of-fold P(Transported) |
| `member_test_seed42.csv` | 4,277 test rows | `PassengerId`, same four members — mean of the five fold models' P(Transported) |

* Source runs: `ha12_tabpfn`, `ha23_ft`, `ha24_ftrefit`, `ha39_a1_ne2` (seed-42 folds; metrics in `reports/`).
* **No labels** are included. Train labels are read from the competition data inside the notebook.
* Replay check: a logistic regression (C=1) on the member logits, fitted on all OOF rows, reproduces `submissions/ha27_stack3.csv` (3 members, public 0.82020) and `submissions/sub_stack3_plus_a1ne2.csv` (4 members, public 0.82604) with **0 differing rows**.
* Probabilities are written with 17 significant digits so the replay is exact.
