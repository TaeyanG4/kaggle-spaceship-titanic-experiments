"""H-A-17: Viktor's public notebook model (page Best Score 0.81833), re-implemented honestly.

Source read statically from `.kaggle-research/spaceship-titanic/kernel-archives/viktortaran/
source_code.txt` (never executed). Target-free feature steps reproduce cells 24-48 on train+test
(Expenses, CryoSleep fill from zero Expenses, first-observed group fill of Cabin/VIP/HomePlanet/
Destination, mean/mode imputers, one-hot). The only label-dependent steps are moved inside each
outer fit fold: SMOTE(sampling_strategy=1) and the XGBoost fit (fixed params, 725 trees).
Arms: `viktor_faithful` (cell-64 drop list + fold-local SMOTE), `viktor_plain` (neither).
The drop list was chosen by the author on the full train set; it is kept as a fixed,
predeclared input, so the faithful arm carries that small selection optimism.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARAMS = {"lambda": 3.0610042624477543, "alpha": 4.581902571574289,
          "colsample_bytree": 0.9241969052729379, "subsample": 0.9527591724824661,
          "learning_rate": 0.06672065863100594, "n_estimators": 725, "max_depth": 5,
          "min_child_weight": 1, "num_parallel_tree": 1}
DROP = ["ShoppingMall", "Age", "CryoSleep_True", "HomePlanet_Earth", "HomePlanet_Europa",
        "VIP_True", "HomePlanet_Mars", "Destination_PSO J318.5-22", "VIP_False",
        "Destination_55 Cancri e", "FoodCourt", "Destination_TRAPPIST-1e"]
SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUM = ["ShoppingMall", "FoodCourt", "RoomService", "Spa", "VRDeck", "Expenses", "Age"]
CAT = ["CryoSleep", "Cabin_1", "Cabin_3", "VIP", "HomePlanet", "Destination"]


def viktor_features(train, test):
    """Return (x_train, x_test) as float frames, following the notebook's target-free steps."""
    import pandas as pd

    both = pd.concat([train.drop(columns="Transported"), test], ignore_index=True)
    both["Expenses"] = both[SPEND].sum(axis=1)
    both["CryoSleep"] = both["CryoSleep"].astype(object)
    both.loc[both.Expenses.eq(0) & both.CryoSleep.isna(), "CryoSleep"] = True
    room = both.PassengerId.str[:4]
    for col in ("Cabin", "VIP", "HomePlanet", "Destination"):
        first = both[col].groupby(room).transform(lambda s: s.dropna().iloc[0] if s.notna().any() else None)
        both[col] = both[col].where(both[col].notna(), first)
    parts = both.Cabin.str.split("/", expand=True)
    both["Cabin_1"], both["Cabin_3"] = parts[0], parts[2]
    frame = both[NUM + CAT].copy()
    frame[NUM] = frame[NUM].astype(float).fillna(frame[NUM].astype(float).mean())
    for col in CAT:
        values = frame[col].astype(object)
        frame[col] = values.fillna(values.mode().iloc[0]).astype(str)
    onehot = pd.get_dummies(frame[CAT], prefix=CAT, dtype=float)
    out = pd.concat([frame[NUM], onehot], axis=1)
    n = len(train)
    return out.iloc[:n].reset_index(drop=True), out.iloc[n:].reset_index(drop=True)


def run_viktor(name: str, seed: int, arm: str, threads: int) -> dict:
    import numpy as np
    from imblearn.over_sampling import SMOTE
    from sklearn.metrics import accuracy_score, log_loss
    from xgboost import XGBClassifier

    from spaceship_titanic.experiments import (
        diagnostics,
        file_hash,
        frozen_folds,
        load_data,
        save_predictions,
        timestamp,
        write_json,
    )

    if arm not in ("viktor_faithful", "viktor_plain"):
        raise ValueError(f"Unknown arm: {arm}")
    config = {"experiment_id": name, "arm": arm, "random_state": seed, "n_splits": 5,
              "params": PARAMS, "drop": DROP if arm == "viktor_faithful" else [],
              "smote": arm == "viktor_faithful"}
    code = {p: file_hash(ROOT / p) for p in ["scripts/viktor_runner.py",
                                             "src/spaceship_titanic/experiments.py"]}
    raw = {n: file_hash(ROOT / "data/raw" / n) for n in
           ["train.csv", "test.csv", "sample_submission.csv"]}
    fingerprint = hashlib.sha256(json.dumps({"config": config, "raw": raw, "code": code},
                                            sort_keys=True).encode()).hexdigest()
    metrics_path = ROOT / f"reports/{name}_metrics.json"
    if metrics_path.exists():
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        if previous["fingerprint"] == fingerprint:
            print(f"REUSE {name}: {previous['accuracy']:.6f}", flush=True)
            return previous
        raise ValueError(f"Existing experiment differs; use a new ID: {name}")

    started = time.monotonic()
    train, test, sample = load_data()
    folds = frozen_folds(train, seed, 5)
    x, tx = viktor_features(train, test)
    if config["drop"]:
        missing = set(DROP) - set(x.columns)
        if missing:
            raise ValueError(f"Drop-list columns not produced: {sorted(missing)}")
        x, tx = x.drop(columns=DROP), tx.drop(columns=DROP)
    y = train.Transported.astype(int).to_numpy()
    oof, test_probability, fold_results = np.zeros(len(x)), np.zeros(len(tx)), []
    for fold in range(1, 6):
        fold_start = time.monotonic()
        fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        fit_seed = seed + fold
        fx, fy = x.iloc[fit_idx], y[fit_idx]
        if config["smote"]:
            fx, fy = SMOTE(sampling_strategy=1, random_state=fit_seed).fit_resample(fx, fy)
        model = XGBClassifier(**PARAMS, random_state=fit_seed, n_jobs=threads)
        model.fit(fx, fy, verbose=False)
        probability = model.predict_proba(x.iloc[valid_idx])[:, 1]
        oof[valid_idx] = probability
        test_probability += model.predict_proba(tx)[:, 1] / 5
        fold_results.append({"fold": fold, "fit_rows": len(fy),
                             "accuracy": float(accuracy_score(y[valid_idx], probability >= 0.5)),
                             "seconds": time.monotonic() - fold_start})
    artifacts = save_predictions(name, train, test, sample, folds, oof, test_probability)
    result = {
        "experiment_id": name, "config": config, "fingerprint": fingerprint,
        "completed_at": timestamp(), "accuracy": float(accuracy_score(y, oof >= 0.5)),
        "log_loss": float(log_loss(y, oof)),
        "fold_std": float(np.std([f["accuracy"] for f in fold_results])),
        "folds": fold_results, "feature_columns": x.columns.tolist(), "threads": threads,
        "runtime_seconds": time.monotonic() - started, "raw_hashes": raw, "code_hashes": code,
        "artifacts": artifacts, "segments": diagnostics(train, oof),
    }
    write_json(metrics_path, result)
    print(f"DONE {name}: accuracy={result['accuracy']:.6f}, logloss={result['log_loss']:.6f}",
          flush=True)
    return result
