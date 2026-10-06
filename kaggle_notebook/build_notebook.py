"""Build the Kaggle companion notebook (`spaceship_titanic_tabpfn_stack.ipynb`).

The notebook is self-contained: it re-implements the project's baseline features, frozen SGKF
folds and the four TabPFN v3.5 members, then fits the nested logistic-regression stacker. It runs
on Kaggle with GPU + internet and a `TABPFN_TOKEN` Kaggle secret, or locally with the official
CSVs in `data/raw/` and `TABPFN_TOKEN` in the environment. `NB_FAST=1` runs a short smoke test.
Run: `uv run python kaggle_notebook/build_notebook.py`.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
REPO = "https://github.com/TaeyanG4/kaggle-spaceship-titanic-experiments"
RAW = "https://raw.githubusercontent.com/TaeyanG4/kaggle-spaceship-titanic-experiments/main"

cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md(f"""
# Spaceship Titanic · Leak-free TabPFN v3.5 Fine-tuning Stack

**Public LB 0.82604** (best of one CV-ranked batch) · **0.82020** (official, CV-promoted champion)

**Only 12 submissions in total** — the CV-promoted models reached 0.81786 on the **3rd** and 0.82020 on the **4th** submission; the remaining 8 were one CV-ranked batch at the end.

This notebook rebuilds, from the official competition files only, the final model of a
70+ experiment research log: a **nested logistic-regression stack of four TabPFN v3.5 variants**
(frozen, fine-tuned, fine-tune-then-refit, and a longer fine-tune). Full write-up, every failed
idea and all evidence: **[GitHub · kaggle-spaceship-titanic-experiments]({REPO})**.

![score progression]({RAW}/docs/assets/score-progression.png)

| | Honest OOF accuracy (SGKF seeds 42 / 123 / 2026) | Public LB |
|---|---|---|
| CatBoost, honest early stopping | 0.8159 / 0.8178 / 0.8185 | 0.81131 |
| Frozen TabPFN v3.5 | 0.8262 / 0.8250 / 0.8243 | 0.81786 |
| **Stack of 3 TabPFN variants** (official champion) | 0.8291 / 0.8288 / 0.8269 | **0.82020** |
| **+ 100-epoch fine-tune member** (this notebook's `submission.csv`) | 0.8311 / 0.8295 / 0.8273 | **0.82604** |
""")

md("""
## How the score was earned — and what was *not* used

Every number above comes from a pipeline that only ever sees **`train.csv` labels inside the
training part of a fold**. Concretely:

| Risk | What this project did |
|---|---|
| Leaked / recovered test labels | **Never used.** No external answer files, no public submission CSVs, no "best public" override bits (several high-scoring public notebooks do this; they were audited and excluded). |
| Leaderboard probing | **None.** No submission was used to infer labels or to choose features, hyperparameters or rows. Only **12 submissions** were made: the first baseline plus three CV-promoted champions (0.82020 was the **4th** submission), then 8 CV-ranked candidates submitted **once, in a single batch**. |
| Validation leakage | Folds are **StratifiedGroupKFold by travel group** (`PassengerId` prefix); no group is split, and train/test share **0** groups. Encoders, early-stopping slices and epoch choices live **inside** the training fold. The scored fold is predicted once. |
| Optimistic CV | Early stopping on the scored fold was found to inflate accuracy by ~0.004 and was **removed**. The stacker for fold *k* is fitted on the **other four** folds' OOF only. |
| Overfitting to one split | Every idea had to beat a matched control on **3 split seeds** (mean ≥ +0.002, ≥ 2/3 positive) and then on **2 fresh seeds** (7, 99) never used for selection. ~70 ideas were rejected this way. |
| External data | **None**, apart from the pretrained TabPFN weights (trained on synthetic data, no Spaceship Titanic labels). |

One transductive step is used and stated openly: `GroupSize` and `SurnameSize` count passengers
over the **combined train + test feature rows** (no labels). Adversarial validation shows train and
test are indistinguishable (AUC 0.483–0.495 vs 0.506–0.520 for permuted labels).

![validation boundary]({RAW}/docs/assets/validation-boundary.png)
""".replace("{RAW}", RAW))

md("""
## Setup — two modes

| Mode | When | Needs | Time |
|---|---|---|---|
| **replay** (default) | no `TABPFN_TOKEN` secret | CPU only, the attached dataset [spaceship-titanic-tabpfn-member-predictions](https://www.kaggle.com/datasets/taeyangg4/spaceship-titanic-tabpfn-member-predictions) | seconds |
| **train** | `TABPFN_TOKEN` secret present | GPU (T4/P100), Internet on, a free Prior Labs token | ~1.5–2.5 h on a T4 |

**Replay** loads the saved out-of-fold and test probabilities of the four TabPFN members (no labels),
re-checks that their folds match the folds rebuilt below, recomputes the honest nested accuracy from
`train.csv`, refits the stacker and writes `submission.csv`. It reproduces the submitted files
**exactly** (checked by hash at the end). **Train** rebuilds the four members from scratch; TabPFN v3.5
weights need a Prior Labs token added as a Kaggle Secret named `TABPFN_TOKEN` (read into the
environment, never printed). Set `NB_FAST=1` for a 5-minute smoke test of the train mode.
""")

code("""
import copy
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")
os.environ.setdefault("TABPFN_NO_BROWSER", "1")
os.environ.setdefault("TQDM_DISABLE", "1")  # silence per-epoch progress bars
if "TABPFN_TOKEN" not in os.environ:
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")
    except Exception:  # noqa: BLE001, S110 - not on Kaggle or no secret: replay mode
        pass
MODE = "train" if os.environ.get("TABPFN_TOKEN") and os.environ.get("REPLAY") != "1" else "replay"
print("mode:", MODE)

FAST = os.environ.get("NB_FAST") == "1"   # smoke test: 1 fold, few epochs
SEED = 42                                 # outer split seed used for the submission
N_FOLDS = 5
FOLDS_TO_RUN = [1] if FAST and MODE == "train" else list(range(1, N_FOLDS + 1))
FT_BASE = {"epochs": 2 if FAST else 30, "learning_rate": 1e-5, "early_stopping_patience": 8,
           "time_limit": 600, "eval_metric": "log_loss"}
FT_LONG = {**FT_BASE, "epochs": 3 if FAST else 100, "early_stopping_patience": 20,
           "time_limit": 900}

for root in [Path("/kaggle/input/spaceship-titanic"), Path("data/raw"), Path("../data/raw")]:
    if (root / "train.csv").exists():
        DATA = root
        break
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
sample = pd.read_csv(DATA / "sample_submission.csv")
y = train.Transported.astype(int).to_numpy()
groups = train.PassengerId.str.split("_").str[0]
print(train.shape, test.shape, "| train/test shared groups:",
      len(set(groups) & set(test.PassengerId.str.split("_").str[0])))
""")

md("""
## Features (23 columns, target-free)

The same baseline feature set used by every experiment in the project. Nothing here reads
`Transported`. Feature-set changes (spend semantics, CryoSleep rules, peer statistics, surname or
cabin identity, location proxies) were all tested and **did not help** TabPFN, so the simple set stays.
""")

code("""
SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CATEGORICAL = ["HomePlanet", "CryoSleep", "Destination", "VIP", "CabinDeck", "CabinSide"]


def surname(name):
    return (name.fillna("__MISSING__").astype(str).str.strip().str.split().str[-1]
            .replace("", "__MISSING__"))


def frame_features(df):
    out = df.copy()
    pid = out.PassengerId.str.split("_", n=1, expand=True)
    out["PassengerGroup"] = pid[0]
    out["GroupMember"] = pd.to_numeric(pid[1], errors="coerce")
    cabin = out.Cabin.fillna("__MISSING__/__MISSING__/__MISSING__").str.split("/", expand=True)
    out["CabinDeck"], out["CabinNum"], out["CabinSide"] = (
        cabin[0], pd.to_numeric(cabin[1], errors="coerce"), cabin[2])
    out["Surname"] = surname(out.Name)
    out["TotalSpend"] = out[SPEND].fillna(0).sum(axis=1)
    out["NoSpend"] = (out.TotalSpend == 0).astype(int)
    out["SpendMissingCount"] = out[SPEND].isna().sum(axis=1)
    out["IsChild"] = (out.Age < 13).fillna(False).astype(int)
    out["IsTeen"] = ((out.Age >= 13) & (out.Age < 18)).fillna(False).astype(int)
    out["IsAdult"] = (out.Age >= 18).fillna(False).astype(int)
    return out


def build_features(train, test):
    a, b = frame_features(train.drop(columns="Transported")), frame_features(test)
    both = pd.concat([a, b], ignore_index=True)          # feature rows only, no labels
    group_n, surname_n = both.PassengerGroup.value_counts(), both.Surname.value_counts()
    for f in (a, b):
        f["GroupSize"] = f.PassengerGroup.map(group_n).astype(int)
        f["SurnameSize"] = f.Surname.map(surname_n).astype(int)
        f["IsAlone"] = (f.GroupSize == 1).astype(int)
    drop = ["Name", "Cabin", "PassengerGroup", "Surname", "PassengerId"]
    a, b = a.drop(columns=drop), b.drop(columns=drop)
    for f in (a, b):
        for c in CATEGORICAL:
            f[c] = f[c].astype("string").fillna("__MISSING__").astype(str)
    return a, b


X, X_test = build_features(train, test)
CAT_IDX = [X.columns.get_loc(c) for c in CATEGORICAL]
print(X.shape, list(X.columns))
""")

md("""
## Folds and the fold-local encoder

`StratifiedGroupKFold(5, shuffle=True, random_state=42)` on travel groups — identical to the
frozen folds of the project. Each fold fits its own ordinal encoder on its training rows.
""")

code("""
folds = np.zeros(len(train), dtype=int)
for k, (_, valid) in enumerate(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
                               .split(X, y, groups), 1):
    folds[valid] = k


def fold_parts(fold):
    fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(X.iloc[fit_idx][CATEGORICAL])

    def enc(frame):
        frame = frame.copy()
        frame[CATEGORICAL] = encoder.transform(frame[CATEGORICAL])
        return frame.to_numpy(dtype=np.float32)

    # early-stopping slice: 1/8 of the training fold, again split by travel group
    a, b = next(StratifiedGroupKFold(8, shuffle=True, random_state=SEED + fold)
                .split(fit_idx, y[fit_idx], groups.iloc[fit_idx]))
    return fit_idx, valid_idx, fit_idx[a], fit_idx[b], enc


print(pd.Series(folds).value_counts().sort_index().to_dict())
""")

md("""
## The four TabPFN v3.5 members

| Member | What it does |
|---|---|
| `frozen` | Pretrained TabPFN v3.5, in-context only (the training fold is the context). |
| `ft` | Fine-tuned inside the fold: ≤30 epochs, lr 1e-5, early stopping on the group-split slice. |
| `ftrefit` | Epoch count chosen on the slice, then a fresh fine-tune on the **whole** training fold. |
| `ft_long_ne2` | ≤100 epochs, patience 20; predicted with **2 estimators**, the same count used while fine-tuning. |

Each member writes out-of-fold probabilities for `train` and a fold-averaged probability for `test`.
""")

code("""
MEMBERS = ["frozen", "ft", "ftrefit", "ft_long_ne2"]
oof = {m: np.zeros(len(X)) for m in MEMBERS}
test_p = {m: np.zeros(len(X_test)) for m in MEMBERS}


def train_members():
    import subprocess
    import sys
    try:
        import tabpfn  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn==9.1.0"], check=True)
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    from tabpfn.finetuning import FinetunedTabPFNClassifier


    class EpochTracker:  # records the slice log loss after every fine-tuning epoch
        def __init__(self):
            self.val_loss = []
        def setup(self, config): pass
        def log_step(self, metrics, step): pass
        def log_epoch(self, metrics, step):
            if "val/log_loss" in metrics:
                self.val_loss.append((int(step), float(metrics["val/log_loss"])))
        def finish(self): pass


    def finetuner(seed, **kwargs):
        return FinetunedTabPFNClassifier(device="cuda", random_state=seed,
                                         model_version=ModelVersion.V3_5,
                                         extra_classifier_kwargs={"categorical_features_indices": CAT_IDX},
                                         **kwargs)


    started = time.time()
    for fold in FOLDS_TO_RUN:
        fit_idx, valid_idx, tr_idx, es_idx, enc = fold_parts(fold)
        fs = SEED + fold
        Xf, Xv, Xt = enc(X.iloc[fit_idx]), enc(X.iloc[valid_idx]), enc(X_test)
        Xtr, Xes = enc(X.iloc[tr_idx]), enc(X.iloc[es_idx])

        frozen = TabPFNClassifier.create_default_for_version(
            ModelVersion.V3_5, categorical_features_indices=CAT_IDX, device="cuda", random_state=fs)
        frozen.fit(Xf, y[fit_idx])
        oof["frozen"][valid_idx] = frozen.predict_proba(Xv)[:, 1]
        test_p["frozen"] += frozen.predict_proba(Xt)[:, 1] / len(FOLDS_TO_RUN)

        ft = finetuner(fs, **FT_BASE)
        ft.fit(Xtr, y[tr_idx], X_val=Xes, y_val=y[es_idx])
        oof["ft"][valid_idx] = ft.predict_proba(Xv)[:, 1]
        test_p["ft"] += ft.predict_proba(Xt)[:, 1] / len(FOLDS_TO_RUN)
        del ft

        tracker = EpochTracker()
        stage1 = finetuner(fs, experiment_logger=tracker, **FT_BASE)
        stage1.fit(Xtr, y[tr_idx], X_val=Xes, y_val=y[es_idx])
        best_epochs = min(tracker.val_loss, key=lambda t: t[1])[0] if tracker.val_loss else 1
        del stage1
        refit = finetuner(fs, epochs=best_epochs, learning_rate=FT_BASE["learning_rate"],
                          validation_split_ratio=None, early_stopping=False,
                          time_limit=FT_BASE["time_limit"])
        refit.fit(Xf, y[fit_idx])
        oof["ftrefit"][valid_idx] = refit.predict_proba(Xv)[:, 1]
        test_p["ftrefit"] += refit.predict_proba(Xt)[:, 1] / len(FOLDS_TO_RUN)
        del refit

        long = finetuner(fs, **FT_LONG)
        long.fit(Xtr, y[tr_idx], X_val=Xes, y_val=y[es_idx])
        ne2 = copy.deepcopy(long.finetuned_inference_classifier_)  # same fine-tuned weights
        ne2.n_estimators = 2
        ne2.fit(Xtr, y[tr_idx])
        oof["ft_long_ne2"][valid_idx] = ne2.predict_proba(Xv)[:, 1]
        test_p["ft_long_ne2"] += ne2.predict_proba(Xt)[:, 1] / len(FOLDS_TO_RUN)
        del long, ne2

        print(f"fold {fold}: " + "  ".join(
            f"{m} {accuracy_score(y[valid_idx], oof[m][valid_idx] >= 0.5):.4f}" for m in oof)
            + f"  ({(time.time() - started) / 60:.1f} min)")


if MODE == "train":
    train_members()
""")

code("""
if MODE == "replay":
    for root in [Path("/kaggle/input/spaceship-titanic-tabpfn-member-predictions"),
                 Path("kaggle_dataset"), Path("../kaggle_dataset")]:
        if (root / "member_oof_seed42.csv").exists():
            break
    saved_oof = pd.read_csv(root / "member_oof_seed42.csv")
    saved_test = pd.read_csv(root / "member_test_seed42.csv")
    assert saved_oof.PassengerId.equals(train.PassengerId)
    assert saved_test.PassengerId.equals(test.PassengerId)
    assert np.array_equal(saved_oof.fold.to_numpy(), folds), "saved folds differ from the rebuilt folds"
    for m in MEMBERS:
        oof[m], test_p[m] = saved_oof[m].to_numpy(), saved_test[m].to_numpy()
    print("replayed saved member predictions; folds identical to the rebuilt SGKF folds")
""")

md("""
## Nested stacker

A logistic regression on the members' **logits**. For the honest score, the stacker that predicts
fold *k* is fitted on the other folds' OOF rows only; for the submission, one stacker is fitted on
all OOF rows and applied to the fold-averaged test probabilities.
""")

code("""
def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


rows = np.flatnonzero(np.isin(folds, FOLDS_TO_RUN))
report = {m: accuracy_score(y[rows], oof[m][rows] >= 0.5) for m in oof}


def nested_stack(members):
    Z = np.column_stack([logit(oof[m]) for m in members])
    pred = np.zeros(len(y))
    for k in FOLDS_TO_RUN:
        fit = np.isin(folds, FOLDS_TO_RUN) & (folds != k)
        hold = folds == k
        if fit.sum() == 0:  # FAST mode has a single fold: no honest stack score
            return float("nan")
        pred[hold] = LogisticRegression(C=1.0, max_iter=1000).fit(Z[fit], y[fit]).predict_proba(Z[hold])[:, 1]
    return accuracy_score(y[rows], pred[rows] >= 0.5)


STACK3 = ["frozen", "ft", "ftrefit"]
STACK4 = STACK3 + ["ft_long_ne2"]
report["stack3 (nested)"] = nested_stack(STACK3)
report["stack4 (nested)"] = nested_stack(STACK4)
recorded = {"frozen": 0.826182, "ft": 0.827332, "ftrefit": 0.826987, "ft_long_ne2": 0.829748,
            "stack3 (nested)": 0.829058, "stack4 (nested)": 0.831128}
pd.DataFrame({"this run (OOF acc.)": report, "recorded in the project (seed 42)": recorded}).round(6)
""")

md("""
Small differences from the recorded numbers are expected: GPU kernels and library builds make
TabPFN fine-tuning non-deterministic across machines. The recorded values come from an RTX 4070 Ti
SUPER with `tabpfn 9.1.0`, `torch 2.14.1+cu126`.
""")

code("""
def fit_submission(members, path):
    Z = np.column_stack([logit(oof[m][rows]) for m in members])
    stacker = LogisticRegression(C=1.0, max_iter=1000).fit(Z, y[rows])
    T = np.column_stack([logit(test_p[m]) for m in members])
    prob = stacker.predict_proba(T)[:, 1]
    sub = sample.copy()
    sub["Transported"] = prob >= 0.5
    sub.to_csv(path, index=False)
    print(path, "weights", np.round(stacker.coef_[0], 3), "positive rate", round(sub.Transported.mean(), 4))
    return sub


sub4 = fit_submission(STACK4, "submission.csv")          # best public LB variant (0.82604)
sub3 = fit_submission(STACK3, "submission_stack3.csv")   # official CV-promoted champion (0.82020)
print("rows that differ:", int((sub3.Transported != sub4.Transported).sum()))
""")

md("""
## Is this the file that was actually submitted?

The submitted CSVs are kept in the GitHub repository with their hashes. Line endings are normalised
before hashing so the check works on any OS. In replay mode both hashes must match; in train mode
small GPU-level differences can change a few rows.
""")

code("""
import hashlib

SUBMITTED = {  # sha256 of the submitted files (LF line endings)
    "submission.csv": ("sub_stack3_plus_a1ne2.csv, public 0.82604",
                       "cba1b98a2a8ad88ed7a5d00113963bb64dfcf9ae9c37bd45bf700f2817953431"),
    "submission_stack3.csv": ("ha27_stack3.csv, public 0.82020",
                              "3cdcd62847747557fe4a921ed23f4249afc6bee09e73a20d09da7f61f3880f68"),
}
for path, (label, expected) in SUBMITTED.items():
    digest = hashlib.sha256(Path(path).read_bytes().replace(b"\\r\\n", b"\\n")).hexdigest()
    print(f"{path}: {'identical to' if digest == expected else 'differs from'} {label}")
""")

md(f"""
## What did not work (so you can skip it)

Every idea below was run against a matched control on three split seeds; bars left of the red
line never reached the promotion bar. Feature engineering, other GBDTs, deep tabular nets, other
foundation models (TabPFN v2.5, TabICL, TabDPT, Causilo, TabSTAR), pseudo-labelling,
group post-processing, non-linear meta-models and threshold tuning all failed to beat the plain
TabPFN stack. The remaining errors are **persistent and confident** across seeds and models,
concentrated in Earth passengers (1.4× the average error rate) and Cabin deck G (1.6×).

![experiment landscape]({RAW}/docs/assets/experiment-landscape.png)

**Takeaways**

1. Honest validation first: removing early stopping on the scored fold cost 0.004 CV — and the
   public LB rose, because the new CV was the one that transferred.
2. On this small table, a tabular foundation model beat every tuned GBDT by ~0.008.
3. The only lever after that was **fine-tuning** the foundation model, and the stacker's gain comes
   entirely from fine-tuned members (frozen variants and other model families added nothing).

Full experiment journey, integrity notes and reproduction steps: **[{REPO}]({REPO})**
""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"},
               "accelerator": "GPU"}
out = HERE / "spaceship_titanic_tabpfn_stack.ipynb"
nbf.write(nb, out)
print("wrote", out.name, len(cells), "cells")
