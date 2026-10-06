"""Build the Kaggle companion notebook (`spaceship_titanic_tabpfn_stack.ipynb`).

Flow: overview and integrity -> setup -> exploratory data analysis -> preprocessing and feature
engineering (with the validation design) -> modeling (four TabPFN v3.5 members + nested stacker)
-> evaluation -> submission -> research notes.

The notebook is self-contained. By default it runs in *replay* mode (CPU, no internet, no token):
it loads the four members' saved OOF/test probabilities from the attached dataset and reproduces
the submitted files exactly. With a `TABPFN_TOKEN` Kaggle secret (GPU + internet) it retrains the
members from scratch. Locally it reads `data/raw/` and `kaggle_dataset/`; `REPLAY=1` forces replay
and `NB_FAST=1` makes the train mode a short smoke test.
Run: `uv run python kaggle_notebook/build_notebook.py`.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
REPO = "https://github.com/TaeyanG4/kaggle-spaceship-titanic-experiments"
RAW = "https://raw.githubusercontent.com/TaeyanG4/kaggle-spaceship-titanic-experiments/main"
DATASET = "https://www.kaggle.com/datasets/taeyangg4/spaceship-titanic-tabpfn-member-predictions"

cells: list = []
LEDGER = HERE.parent / "docs/evidence/experiment-ledger.csv"


def cheat_sheet() -> str:
    """Markdown table of every logged idea, built from the committed experiment ledger."""
    import csv

    stage_names = {"1-catboost": "CatBoost era", "2-tabpfn": "TabPFN era", "3-stack": "stack era"}
    verdicts = {"promoted": "**promoted**", "rejected": "no", "inconclusive": "consistent, below bar",
                "post-hoc": "post hoc, unconfirmed"}
    lines = ["| Stage | Idea | Compared with | Mean Δ accuracy | Seeds better | Verdict |",
             "|---|---|---|---:|---|---|"]
    with open(LEDGER, encoding="utf-8") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda r: (r["stage"], -float(r["mean_delta"])))
    for r in rows:
        lines.append(f"| {stage_names[r['stage']]} | {r['label_en']} | {r['control']} | "
                     f"{float(r['mean_delta']) * 100:+.2f} pp | {r['positive_seeds']} | "
                     f"{verdicts[r['outcome']]} |")
    return "\n".join(lines)


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n").replace("{RAW}", RAW)
                                          .replace("{REPO}", REPO).replace("{DATASET}", DATASET)
                                          .replace("{CHEAT}", cheat_sheet())))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------- 0. overview and integrity
md("""
# Leak-free 2026 TabPFN Stack | Top Public 0.82604

### 2026 tabular foundation model × integrity-first, honest validation — rank 8 with only 12 submissions

> **TL;DR**
> * **0.82604** on the leaderboard (computed on all of the test data) — higher than every score
>   shown in the score-sorted public notebook list when it was surveyed (2026-10-05: top entry
>   0.82137, which overwrites predictions with embedded bits; best clean-looking one 0.81833).
> * **No leaks, no probing.** Group-aware CV, fold-local preprocessing, nested stacking, and a
>   two-stage promotion rule. The CV-promoted champion scored 0.82020 on only the **4th**
>   submission; **12 submissions in total**.
> * **2026 recipe.** A tabular foundation model (TabPFN v3.5), **fine-tuned inside each fold**,
>   stacked by a nested logistic regression.
> * **Reproducible to the byte.** Runs in seconds on CPU (no token, no internet) and writes a
>   `submission.csv` identical to the submitted file; one switch retrains everything on GPU.
> * **44 ideas tested, 3 promoted.** The cheat sheet in section 7 tells you what *not* to try.

![leaderboard, 2026-10-06]({RAW}/docs/assets/kaggle-leaderboard.png)

*The public leaderboard on 2026-10-06 (captured logged-out). The same eight entries against the
number of submissions:*

![leaderboard position]({RAW}/docs/assets/leaderboard-position.png)

| | Honest OOF accuracy (SGKF seeds 42 / 123 / 2026) | Leaderboard |
|---|---|---|
| CatBoost, honest early stopping | 0.8159 / 0.8178 / 0.8185 | 0.81131 |
| Frozen TabPFN v3.5 | 0.8262 / 0.8250 / 0.8243 | 0.81786 |
| **Stack of 3 TabPFN variants** (official, CV-promoted champion) | 0.8291 / 0.8288 / 0.8269 | **0.82020** |
| **+ 100-epoch fine-tune member** (this notebook's `submission.csv`) | 0.8311 / 0.8295 / 0.8273 | **0.82604** |

**Why this notebook is different**

| Usual top notebooks | This notebook |
|---|---|
| GBDT + hand-made rules, tuned on one split | Tabular foundation model, fine-tuned per fold; every idea tested on 3 + 2 split seeds |
| CV scores that are often optimistic (early stopping or target statistics on the scored fold) | Every number is out-of-fold and nested; the optimism was measured (+0.004) and removed |
| Many submissions | 12 submissions; leaderboard used once per promoted model |
| One person's run | Built by a team of AI coding agents (Codex, Claude Code, ChatGPT) under a shared research protocol, 78 recorded findings, cross-checked across platforms |

The full research log, every failed idea and all evidence: **[GitHub · kaggle-spaceship-titanic-experiments]({REPO})**.

**Contents**

0. Integrity: how the score was earned
1. Setup and data loading
2. Exploratory data analysis
3. Preprocessing and feature engineering (incl. the validation design)
4. Modeling: four TabPFN v3.5 members and a nested stacker
5. Evaluation
6. Submission and reproducibility check
7. Research notes: the cheat sheet of what did not work
""")

md("""
## 0. Integrity: how the score was earned — and what was *not* used

Every number in this notebook comes from a pipeline that only ever sees **`train.csv` labels
inside the training part of a fold**.

| Risk | What this project did |
|---|---|
| Leaked / recovered test labels | **Never used.** No external answer files, no public submission CSVs, no "best public" override bits (several high-scoring public notebooks do this; they were audited and excluded). |
| Leaderboard probing | **None.** Only **12 submissions**: the first baseline plus three CV-promoted champions (0.82020 was the **4th** submission), then 8 CV-ranked candidates submitted **once, in a single batch**. No submission was used to infer labels or to choose features, hyperparameters or rows. |
| Validation leakage | **StratifiedGroupKFold by travel group**; no group is split, train/test share **0** groups. Encoders, early-stopping slices and epoch choices live **inside** the training fold. |
| Optimistic CV | Early stopping on the scored fold inflated accuracy by ~0.004; it was found and **removed**. The stacker for fold *k* is fitted on the **other four** folds' OOF only. |
| Overfitting to one split | Every idea had to beat a matched control on **3 split seeds** (mean ≥ +0.002, ≥ 2/3 positive) and then on **2 fresh seeds** (7, 99). Only 3 of 44 logged ideas were promoted. |
| External data | **None**, apart from the pretrained TabPFN weights (synthetic-data pretraining, no Spaceship Titanic labels). |

One transductive step is used and documented: `GroupSize` and `SurnameSize` count passengers
over the **combined train + test feature rows** (no labels). Adversarial validation shows train
and test are indistinguishable (AUC 0.483–0.495 vs 0.506–0.520 for permuted labels).

![promotion rule]({RAW}/docs/assets/promotion-gate.png)
""")

# ---------------------------------------------------------------- 1. setup
md("""
## 1. Setup and data loading

| Mode | When | Needs | Time |
|---|---|---|---|
| **replay** (default) | no `TABPFN_TOKEN` secret | CPU only, the attached dataset [spaceship-titanic-tabpfn-member-predictions]({DATASET}) | seconds |
| **train** | `TABPFN_TOKEN` secret present | GPU (T4/P100), Internet on, a free Prior Labs token | ~1.5–2.5 h on a T4 |

**Replay** loads the saved out-of-fold and test probabilities of the four TabPFN members (no
labels), checks that their folds equal the folds rebuilt in section 3, recomputes the honest
nested accuracy from `train.csv`, refits the stacker and writes `submission.csv`. It reproduces
the submitted files **exactly** (checked by hash in section 6). **Train** rebuilds the four members
from scratch; TabPFN v3.5 weights need a Prior Labs token added as a Kaggle Secret named
`TABPFN_TOKEN` (read into the environment, never printed).
""")

code("""
import copy
import os
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")
NAVY, TEAL, SAND, RED, GREY = "#16324a", "#2f8f83", "#d2b587", "#c4573f", "#9aa7b0"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110,
                     "axes.titleweight": "bold", "axes.titlesize": 11})
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
SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
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

# ---------------------------------------------------------------- 2. EDA
md("""
## 2. Exploratory data analysis

### 2.1 Columns, missing values and the target
""")

code("""
both = pd.concat([train.drop(columns="Transported"), test], keys=["train", "test"])
overview = pd.DataFrame({"dtype": train.drop(columns="Transported").dtypes.astype(str),
                         "missing % (train)": train.drop(columns="Transported").isna().mean() * 100,
                         "missing % (test)": test.isna().mean() * 100,
                         "unique": both.nunique()}).round(2)

fig, axes = plt.subplots(1, 2, figsize=(15, 3.6), gridspec_kw={"width_ratios": [3, 1]})
miss = overview[["missing % (train)", "missing % (test)"]].drop(index=["PassengerId"])
x = np.arange(len(miss))
axes[0].bar(x - 0.2, miss.iloc[:, 0], 0.4, color=TEAL, label="train")
axes[0].bar(x + 0.2, miss.iloc[:, 1], 0.4, color=SAND, edgecolor=NAVY, label="test")
axes[0].set_xticks(x, miss.index, rotation=30)
axes[0].set_ylabel("% missing")
axes[0].set_title("Missing values per column (about 2% each, similar in train and test)")
axes[0].legend(frameon=False)
counts = train.Transported.value_counts()
axes[1].bar(counts.index.astype(str), counts.values, color=[TEAL, SAND], edgecolor=NAVY)
axes[1].set_title("Target balance (train)")
for i, v in enumerate(counts.values):
    axes[1].text(i, v / 2, f"{v}\\n{v / counts.sum():.1%}", ha="center", color=NAVY)
plt.tight_layout()
plt.show()
overview
""")

md("""
### 2.2 Who was transported?
""")

code("""
fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
target = pd.Series(y, index=train.index)
for ax, col in zip(axes[:3], ["HomePlanet", "CryoSleep", "Cabin deck"]):
    key = train.Cabin.str[0].fillna("?") if col == "Cabin deck" else train[col].astype(str)
    rate = target.groupby(key).mean().sort_values()
    ax.barh(rate.index, rate.values, color=TEAL)
    ax.axvline(y.mean(), color=RED, ls="--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_title(f"P(Transported) by {col}")
sizes = groups.value_counts().value_counts().sort_index()
axes[3].bar(sizes.index, sizes.values, color=SAND, edgecolor=NAVY)
axes[3].set_title("Travel groups by size (train)")
axes[3].set_xlabel("passengers in the group")
shared = len(set(groups) & set(test.PassengerId.str[:4]))
axes[3].text(0.98, 0.95, f"groups shared with test: {shared}", transform=axes[3].transAxes,
             ha="right", va="top", color=RED, fontweight="bold")
plt.tight_layout()
plt.show()

total = train[SPEND].sum(axis=1, min_count=1)
fig, axes = plt.subplots(1, 3, figsize=(17, 3.6))
bins = np.arange(0, 81, 4)
for flag, color in [(True, TEAL), (False, SAND)]:
    axes[0].hist(train.Age[train.Transported == flag], bins=bins, alpha=0.6, color=color,
                 label=f"Transported={flag}")
    axes[1].hist(np.log1p(total[train.Transported == flag].dropna()), bins=30, alpha=0.6,
                 color=color, label=f"Transported={flag}")
axes[0].set_title("Age")
axes[1].set_title("log(1 + total spend)")
for ax in axes[:2]:
    ax.legend(frameon=False)
spend_state = np.where(total.isna(), "unknown", np.where(total > 0, "spent", "spent nothing"))
cryo = train.CryoSleep.map({True: "CryoSleep", False: "awake"}).fillna("CryoSleep unknown")
table = target.groupby([cryo, spend_state]).mean().unstack()
table.plot.barh(ax=axes[2], color=[TEAL, SAND, GREY], edgecolor=NAVY)
axes[2].set_title("P(Transported) by CryoSleep and spending")
axes[2].set_xlim(0, 1)
axes[2].legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.show()
print("CryoSleep passengers with positive known spending:",
      int((train.CryoSleep.eq(True) & (total > 0)).sum()))
""")

md("""
**What the data says**

* The target is balanced (50.4% transported), so accuracy is a sensible metric.
* Every column is about 2% missing, in train and test alike — no column needs to be dropped.
* `CryoSleep` is the strongest single signal (about 82% transported), and sleepers never spend.
  Passengers who spent nothing are far more likely to be transported.
* Europa passengers and decks B/C are transported more often; Earth passengers and deck E/T less.
* Children are transported more often than adults.
* `PassengerId` encodes **travel groups**; most groups are single travellers, and **no group is
  shared between train and test**. That fact decides the validation design in section 3.2.

These patterns were all tried as hand-made rules or extra features during the research
(CryoSleep fills, spend semantics, group and cabin location features). Under group-aware
validation none of them beat the simple feature set below, because a strong model already
learns them from the raw columns.
""")

# ---------------------------------------------------------------- 3. preprocessing
md("""
## 3. Preprocessing and feature engineering

### 3.1 Features (23 columns, target-free)

| Step | Output |
|---|---|
| Split `PassengerId` (`gggg_pp`) | `GroupMember`; the group id itself is **not** a feature |
| Split `Cabin` (`deck/num/side`) | `CabinDeck`, `CabinNum`, `CabinSide` |
| Spending | `TotalSpend`, `NoSpend`, `SpendMissingCount` (raw five columns kept) |
| Age | `IsChild` (<13), `IsTeen` (13–17), `IsAdult` (≥18) |
| Group and family size | `GroupSize`, `SurnameSize`, `IsAlone` — counted over train + test **feature** rows |
| Missing values | categoricals → `"__MISSING__"`; numerics stay `NaN` (TabPFN handles them natively) |
| Encoding | ordinal codes, fitted **inside each training fold** (section 3.2) |

Nothing here reads `Transported`. Richer feature sets were tested and did not help TabPFN.
""")

code("""
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
pd.DataFrame({"kind": ["categorical" if c in CATEGORICAL else "numeric" for c in X.columns],
              "missing %": (X.isna().mean() * 100).round(2).values,
              "example": X.iloc[0].astype(str).values}, index=X.columns)
""")

md("""
### 3.2 Validation design: group-aware folds and fold-local preprocessing

`StratifiedGroupKFold(5, shuffle=True, random_state=42)` on travel groups — identical to the
frozen folds used throughout the project. Inside each training fold, a second group-split
(1/8 of the fold) is held out for early stopping and epoch choice, and the ordinal encoder is
fitted on the training fold only. The scored fold is only ever predicted.

![validation boundary]({RAW}/docs/assets/validation-boundary.png)
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


assert (groups.groupby(folds).nunique().sum() == groups.nunique()), "a group crosses folds"
pd.DataFrame({"rows": pd.Series(folds).value_counts().sort_index(),
              "groups": groups.groupby(folds).nunique(),
              "P(Transported)": pd.Series(y).groupby(folds).mean().round(4)}).rename_axis("fold")
""")

# ---------------------------------------------------------------- 4. modeling
md("""
## 4. Modeling

### 4.1 Why TabPFN?

TabPFN v3.5 is a *tabular foundation model*: a transformer pretrained on millions of synthetic
tables that predicts a new table **in context** — the training fold is passed as context, no
gradient boosting rounds to tune. On this small table (8.7k rows) it beat every tuned GBDT in the
project by about +0.008 accuracy on every split seed. Fine-tuning its weights inside each fold
then added a small but consistent gain.

### 4.2 The four members

| Member | What it does |
|---|---|
| `frozen` | Pretrained TabPFN v3.5, in-context only (the training fold is the context). |
| `ft` | Fine-tuned inside the fold: ≤30 epochs, lr 1e-5, early stopping on the group-split slice. |
| `ftrefit` | Epoch count chosen on the slice, then a fresh fine-tune on the **whole** training fold. |
| `ft_long_ne2` | ≤100 epochs, patience 20; predicted with **2 estimators**, the same count used while fine-tuning. |

Each member writes out-of-fold probabilities for `train` and a fold-averaged probability for `test`.

![final pipeline]({RAW}/docs/assets/final-pipeline.png)
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

        def setup(self, config):
            pass

        def log_step(self, metrics, step):
            pass

        def log_epoch(self, metrics, step):
            if "val/log_loss" in metrics:
                self.val_loss.append((int(step), float(metrics["val/log_loss"])))

        def finish(self):
            pass

    def finetuner(seed, **kwargs):
        return FinetunedTabPFNClassifier(
            device="cuda", random_state=seed, model_version=ModelVersion.V3_5,
            extra_classifier_kwargs={"categorical_features_indices": CAT_IDX}, **kwargs)

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
### 4.3 Nested stacker

A logistic regression on the members' **logits**. For the honest score, the stacker that predicts
fold *k* is fitted on the other folds' OOF rows only (nested). For the submission, one stacker is
fitted on all OOF rows and applied to the fold-averaged test probabilities.
""")

code("""
def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


rows = np.flatnonzero(np.isin(folds, FOLDS_TO_RUN))
STACK3 = ["frozen", "ft", "ftrefit"]
STACK4 = STACK3 + ["ft_long_ne2"]


def nested_pred(members):
    # OOF stack probabilities: fold k is predicted by a stacker fitted on the other folds
    Z = np.column_stack([logit(oof[m]) for m in members])
    pred = np.full(len(y), np.nan)
    for k in FOLDS_TO_RUN:
        fit = np.isin(folds, FOLDS_TO_RUN) & (folds != k)
        hold = folds == k
        if fit.sum() == 0:  # FAST mode has a single fold: no honest stack score
            return None
        stacker = LogisticRegression(C=1.0, max_iter=1000).fit(Z[fit], y[fit])
        pred[hold] = stacker.predict_proba(Z[hold])[:, 1]
    return pred


def nested_stack(members):
    pred = nested_pred(members)
    return float("nan") if pred is None else accuracy_score(y[rows], pred[rows] >= 0.5)


stack_scores = {"stack3 (nested)": nested_stack(STACK3), "stack4 (nested)": nested_stack(STACK4)}
stack_scores
""")

# ---------------------------------------------------------------- 5. evaluation
md("""
## 5. Evaluation

### 5.1 Honest OOF accuracy vs the values recorded in the project

In replay mode these must match exactly. In train mode, small differences are expected: GPU
kernels make TabPFN fine-tuning non-deterministic across machines (recorded values: RTX 4070 Ti
SUPER, `tabpfn 9.1.0`, `torch 2.14.1+cu126`).
""")

code("""
report = {m: accuracy_score(y[rows], oof[m][rows] >= 0.5) for m in MEMBERS} | stack_scores
recorded = {"frozen": 0.826182, "ft": 0.827332, "ftrefit": 0.826987, "ft_long_ne2": 0.829748,
            "stack3 (nested)": 0.829058, "stack4 (nested)": 0.831128}
pd.DataFrame({"this run (OOF acc.)": report, "recorded in the project (seed 42)": recorded}).round(6)
""")

md("""
### 5.2 Member diagnostics

Fine-tuning moves accuracy only a little but clearly improves log loss, and the four members are
highly correlated: they make most of their mistakes on the same passengers. This is why other
model families never helped the stack — the useful signal is the *direction* from frozen to
fine-tuned.
""")

code("""
diag = pd.DataFrame({"OOF accuracy": [accuracy_score(y[rows], oof[m][rows] >= 0.5) for m in MEMBERS],
                     "OOF log loss": [log_loss(y[rows], oof[m][rows]) for m in MEMBERS]},
                    index=MEMBERS)
corr = np.corrcoef([logit(oof[m][rows]) for m in MEMBERS])
fig, axes = plt.subplots(1, 3, figsize=(17, 3.8))
axes[0].bar(MEMBERS, diag["OOF accuracy"], color=TEAL)
axes[0].set_ylim(diag["OOF accuracy"].min() - 0.004, diag["OOF accuracy"].max() + 0.002)
axes[0].set_title("OOF accuracy (higher is better)")
axes[1].bar(MEMBERS, diag["OOF log loss"], color=SAND, edgecolor=NAVY)
axes[1].set_ylim(diag["OOF log loss"].min() - 0.005, diag["OOF log loss"].max() + 0.003)
axes[1].set_title("OOF log loss (lower is better)")
axes[2].imshow(corr, cmap="Blues", vmin=0.9, vmax=1)
axes[2].set_xticks(range(len(MEMBERS)), MEMBERS, rotation=20)
axes[2].set_yticks(range(len(MEMBERS)), MEMBERS)
for i in range(len(MEMBERS)):
    for j in range(len(MEMBERS)):
        axes[2].text(j, i, f"{corr[i, j]:.3f}", ha="center", va="center", fontsize=9,
                     color="white" if corr[i, j] > 0.97 else NAVY)
axes[2].set_title("Correlation of member logits")
for ax in axes[:2]:
    ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.show()
diag.round(6)
""")

md("""
### 5.3 Where the remaining errors are

Left: the stacker's weights — negative on the frozen and short fine-tune logits, positive on the
longer fine-tunes, i.e. it extrapolates along the fine-tuning direction. Middle: the honest
(nested) confusion matrix. Right: how far from 0.5 the predictions are. Most wrong predictions sit
near 0.5, but there is a long tail of **confident** errors; across seeds and models these are the
same rows, which is why more features or more models stopped helping. Below: error rate by
segment relative to the average — Earth passengers and Cabin deck G stay hard on every seed.
""")

code("""
p4 = nested_pred(STACK4)
if p4 is not None:
    stacker4 = LogisticRegression(C=1.0, max_iter=1000).fit(
        np.column_stack([logit(oof[m][rows]) for m in STACK4]), y[rows])
    wrong = (p4 >= 0.5) != y
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    w = stacker4.coef_[0]
    axes[0].bar(STACK4, w, color=[RED if v < 0 else TEAL for v in w])
    axes[0].axhline(0, color=NAVY, lw=0.8)
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_title("Stacker weights on member logits")
    cm = confusion_matrix(y, p4 >= 0.5)
    axes[1].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f"{cm[i, j]}\\n{cm[i, j] / cm.sum():.1%}", ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else NAVY, fontsize=11)
    axes[1].set_xticks([0, 1], ["pred False", "pred True"])
    axes[1].set_yticks([0, 1], ["actual False", "actual True"])
    axes[1].set_title(f"Nested OOF confusion matrix (acc. {1 - wrong.mean():.4f})")
    margin = np.abs(p4 - 0.5)
    bins = np.linspace(0, 0.5, 21)
    axes[2].hist(margin[~wrong], bins=bins, color=TEAL, alpha=0.7, label="correct", density=True)
    axes[2].hist(margin[wrong], bins=bins, color=RED, alpha=0.7, label="wrong", density=True)
    axes[2].set_xlabel("|P(Transported) - 0.5|  (confidence)")
    axes[2].set_title("Confidence of correct vs wrong predictions")
    axes[2].legend(frameon=False)
    plt.tight_layout()
    plt.show()
    print(f"wrong predictions: {wrong.sum()} | with confidence > 0.3: {(margin[wrong] > 0.3).mean():.0%}"
          f" | correct predictions with confidence > 0.3: {(margin[~wrong] > 0.3).mean():.0%}")

    segments = {"Earth": train.HomePlanet.eq("Earth"), "Europa": train.HomePlanet.eq("Europa"),
                "Mars": train.HomePlanet.eq("Mars"), "Cabin deck G": train.Cabin.str[0].eq("G"),
                "CryoSleep": train.CryoSleep.eq(True), "Travelling alone": X.IsAlone.eq(1)}
    ratio = pd.Series({k: wrong[v.to_numpy()].mean() / wrong.mean() for k, v in segments.items()})
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.barh(ratio.index, ratio.values, color=[RED if r > 1 else TEAL for r in ratio])
    ax.axvline(1, color=NAVY, ls="--", lw=1)
    ax.set_xlabel("error rate / average error rate")
    ax.set_title("Error rate by segment (stack of four, nested OOF)")
    plt.tight_layout()
    plt.show()
""")

# ---------------------------------------------------------------- 6. submission
md("""
## 6. Submission and reproducibility check

One stacker per variant is fitted on all OOF rows and applied to the fold-averaged test
probabilities.
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
**Is this the file that was actually submitted?** The submitted CSVs are kept in the GitHub
repository with their hashes. Line endings are normalised before hashing so the check works on any
OS. In replay mode both hashes must match; in train mode small GPU-level differences can change a
few rows.
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

md("""
**Getting the score**

| File | Stack | Public score when it was submitted |
|---|---|---|
| `submission.csv` | frozen + ft + ftrefit + ft_long_ne2 | **0.82604** |
| `submission_stack3.csv` | frozen + ft + ftrefit (CV-promoted champion) | **0.82020** |

Use **Submit to Competition** on the notebook's output (`submission.csv`) to attach the score to
this notebook. The leaderboard of this competition is computed on all of the test data, so the
score is the full test accuracy.
""")

# ---------------------------------------------------------------- 7. research notes
md("""
## 7. Research notes: the cheat sheet of what did not work

The research ran as a scored hypothesis queue shared by several AI coding agents (Codex, Claude
Code, ChatGPT) under the [research-orchestrator-skill](https://github.com/TaeyanG4/research-orchestrator-skill)
protocol: every experiment produced one recorded finding (failures included), and findings were
cross-checked by a different agent platform. Only models that passed the promotion rule were
submitted.

![score progression]({RAW}/docs/assets/score-progression.png)

![submission history]({RAW}/docs/assets/submission-history.png)

Every idea below was run against a matched control on three split seeds; bars left of the red
line never reached the promotion bar. Feature engineering, other GBDTs, deep tabular nets, other
foundation models (TabPFN v2.5, TabICL, TabDPT, Causilo, TabSTAR), pseudo-labelling, group
post-processing, non-linear meta-models and threshold tuning all failed to beat the plain TabPFN
stack.

![experiment landscape]({RAW}/docs/assets/experiment-landscape.png)

**Cheat sheet — every logged idea** (mean paired accuracy change vs the matched control over three
split seeds; the promotion bar is +0.20 pp with at least 2 of 3 seeds better, then 2 fresh seeds):

{CHEAT}

**Takeaways**

1. **Fix the validation first.** Removing early stopping on the scored fold cost 0.004 CV — and the
   leaderboard rose, because the new CV was the one that transferred.
2. **Small table, foundation model.** TabPFN v3.5 beat every tuned GBDT by about 0.008.
3. **Same model, different versions.** After that, the only lever was fine-tuning; the stacker's
   gain comes entirely from fine-tuned members, not from model diversity.
4. **Leaderboard last, and once.** Deciding with CV and submitting rarely is what made rank 8 with
   12 submissions possible.

Full experiment journey (Korean and English), integrity notes and reproduction steps:
**[{REPO}]({REPO})**

*If this saved you from a leaky trick or a few dozen experiments, an upvote helps others find a
leak-free baseline. Questions and challenges to any claim here are welcome in the comments.*
""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
out = HERE / "spaceship_titanic_tabpfn_stack.ipynb"
nbf.write(nb, out)
print("wrote", out.name, len(cells), "cells")
