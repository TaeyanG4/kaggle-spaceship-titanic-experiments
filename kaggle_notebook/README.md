# Kaggle companion notebook

Self-contained notebook that rebuilds the submitted stacks from the official competition files.

- Kaggle ID: `taeyangg4/spaceship-titanic-leak-free-tabpfn-fine-tuning-stack` (to be published)
- Title: `Spaceship Titanic Leak-free TabPFN Fine-tuning Stack`
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
- **Train mode:** smoke-tested end to end (`NB_FAST=1`). TabPFN fine-tuning is not bit-for-bit deterministic across GPUs, so a full retrain can change a few rows.

## Build and publish

```bash
uv run python kaggle_notebook/build_notebook.py
kaggle datasets create -p kaggle_dataset        # once
kaggle kernels push -p kaggle_notebook
```
