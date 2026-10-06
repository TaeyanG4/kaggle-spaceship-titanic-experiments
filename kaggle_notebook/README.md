# Kaggle companion notebook

Self-contained notebook that rebuilds the submitted stacks from the official competition files.

- Kaggle ID: `taeyangg4/leak-free-2026-tabpfn-stack-top-public-0-82604` (to be published)
- Title: `Leak-free 2026 TabPFN Stack | Top Public 0.82604`
- Sources: competition `spaceship-titanic`, dataset `taeyangg4/spaceship-titanic-tabpfn-member-predictions` (files in [`kaggle_dataset/`](../kaggle_dataset/README.md))

## Two modes

| Mode | When | Needs | What it does |
|---|---|---|---|
| **replay** (default) | no `TABPFN_TOKEN` secret | CPU, no internet | loads the four members' saved OOF/test probabilities (no labels), checks their folds equal the rebuilt folds, recomputes nested accuracy from `train.csv`, refits the stacker, writes the submissions |
| **train** | `TABPFN_TOKEN` Kaggle Secret present | GPU + Internet, Prior Labs token | installs `tabpfn==9.1.0` and trains the four TabPFN v3.5 members from scratch (~1.5–2.5 h on a T4; `NB_FAST=1` for a smoke test) |

Outputs: `submission.csv` (4-member stack, public 0.82604) and `submission_stack3.csv` (3-member CV-promoted champion, public 0.82020). The last cell compares both files with the hashes of the files actually submitted. The notebook never submits by itself.

## Verified locally

- **Replay mode:** every member and stack accuracy equals the recorded value (e.g. nested stack3 0.829058, stack4 0.831128), and both output files are **hash-identical** to the submitted files.
- **Feature and fold parity:** `tests/test_notebook_parity.py` executes the notebook's setup, feature and fold cells and checks they equal `src/spaceship_titanic/` (identical frames, folds and early-stopping slices).
- **Train mode (full retrain, RTX 4070 Ti SUPER, about 40 min):** the frozen member is identical (0.826182); the fine-tuned members land within ±0.0012 of the recorded values because GPU fine-tuning is not bit-for-bit deterministic.

  | | retrain | recorded |
  |---|---|---|
  | frozen | 0.826182 | 0.826182 |
  | ft | 0.826297 | 0.827332 |
  | ftrefit | 0.828138 | 0.826987 |
  | ft_long_ne2 | 0.830208 | 0.829748 |
  | stack3 (nested) | 0.829403 | 0.829058 |
  | stack4 (nested) | 0.830898 | 0.831128 |

  The retrained `submission.csv` differs from the submitted 0.82604 file in 43 of 4,277 rows, and `submission_stack3.csv` from the 0.82020 file in 28 rows. Replay mode is the exact reproduction; train mode shows the recipe reproduces at the same level.

## Build and publish

```bash
uv run python kaggle_notebook/build_notebook.py
kaggle datasets create -p kaggle_dataset        # once
kaggle kernels push -p kaggle_notebook
```
