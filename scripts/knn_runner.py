"""H-Main-07: CatBoost champion + supervised KNN distance features built strictly inside folds.

Feature (gokinjo-style, k=1): for each class c, the standardized-space distance to the nearest
fit-fold row of class c. Fit-fold rows get it from a 5-fold inner StratifiedGroupKFold (a row and
its groupmates never see their own labels); valid/test rows get it from all fit-fold rows. The
scaler/imputer are fitted on the fit fold only. Iterations: champion rule `innercv_logloss`.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "CabinNum",
           "GroupSize"]
ONEHOT = ["HomePlanet", "CryoSleep", "Destination", "CabinDeck", "CabinSide"]
PARAMS = {"iterations": 800, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5.0,
          "random_strength": 0.5, "loss_function": "Logloss", "eval_metric": "Accuracy"}


def knn_space(frame, fit_rows):
    import numpy as np
    import pandas as pd

    numeric = frame[NUMERIC].astype(float).copy()
    for col in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
        numeric[col] = np.log1p(numeric[col])
    median = numeric.iloc[fit_rows].median()
    numeric = numeric.fillna(median)
    mean, std = numeric.iloc[fit_rows].mean(), numeric.iloc[fit_rows].std().replace(0, 1)
    numeric = (numeric - mean) / std
    dummies = pd.get_dummies(frame[ONEHOT].astype(str), dtype=float)  # target-free, train+test
    return pd.concat([numeric, dummies], axis=1).to_numpy()


def class_distances(space_fit, y_fit, space_query):
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    out = np.zeros((len(space_query), 2))
    for c in (0, 1):
        nn = NearestNeighbors(n_neighbors=1).fit(space_fit[y_fit == c])
        out[:, c] = nn.kneighbors(space_query)[0][:, 0]
    return out


def run_catboost_knn(name: str, seed: int, threads: int) -> dict:
    import numpy as np
    import pandas as pd
    from catboost import CatBoostClassifier
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.model_selection import StratifiedGroupKFold

    from spaceship_titanic.experiments import (
        diagnostics,
        file_hash,
        frozen_folds,
        load_data,
        save_predictions,
        timestamp,
        write_json,
    )
    from spaceship_titanic.versioned_features import build_versioned_features

    config = {"experiment_id": name, "feature_variant": "baseline+knn_k1_infold",
              "random_state": seed, "n_splits": 5, "stopping": "innercv_logloss",
              "model_params": PARAMS}
    code = {p: file_hash(ROOT / p) for p in [
        "scripts/knn_runner.py", "src/spaceship_titanic/experiments.py",
        "src/spaceship_titanic/versioned_features.py", "src/spaceship_titanic/features.py"]}
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
    x, tx, categorical = build_versioned_features(train, test, "baseline")
    y = train.Transported.astype(int).to_numpy()
    groups = train.PassengerId.str.split("_").str[0]
    both = pd.concat([x, tx], ignore_index=True)
    n = len(x)
    oof, test_probability, fold_results = np.zeros(n), np.zeros(len(tx)), []

    def model(**overrides):
        return CatBoostClassifier(**(PARAMS | overrides), thread_count=threads, verbose=False,
                                  allow_writing_files=False)

    for fold in range(1, 6):
        fold_start = time.monotonic()
        fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        fit_seed = seed + fold
        space = knn_space(both, fit_idx)
        knn = np.zeros((len(both), 2))
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=fit_seed)
        for a, b in inner.split(fit_idx, y[fit_idx], groups.iloc[fit_idx]):
            assert not set(groups.iloc[fit_idx[a]]) & set(groups.iloc[fit_idx[b]])
            knn[fit_idx[b]] = class_distances(space[fit_idx[a]], y[fit_idx[a]],
                                              space[fit_idx[b]])
        query = np.r_[valid_idx, np.arange(n, len(both))]
        knn[query] = class_distances(space[fit_idx], y[fit_idx], space[query])
        aug = both.copy()
        aug["KNN_dist_c0"], aug["KNN_dist_c1"] = knn[:, 0], knn[:, 1]
        aug["KNN_diff"] = knn[:, 0] - knn[:, 1]
        xf, tf = aug.iloc[:n], aug.iloc[n:]
        bests = []
        probe_split = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=fit_seed)
        for a, b in probe_split.split(fit_idx, y[fit_idx], groups.iloc[fit_idx]):
            probe = model(random_seed=fit_seed, eval_metric="Logloss")
            probe.fit(xf.iloc[fit_idx[a]], y[fit_idx[a]], cat_features=categorical,
                      eval_set=(xf.iloc[fit_idx[b]], y[fit_idx[b]]),
                      early_stopping_rounds=100, verbose=False)
            bests.append(int(probe.get_best_iteration()) + 1)
        best = int(np.median(bests))
        final = model(iterations=best, random_seed=fit_seed)
        final.fit(xf.iloc[fit_idx], y[fit_idx], cat_features=categorical)
        probability = final.predict_proba(xf.iloc[valid_idx], thread_count=threads)[:, 1]
        oof[valid_idx] = probability
        test_probability += final.predict_proba(tf, thread_count=threads)[:, 1] / 5
        fold_results.append({"fold": fold, "best_iteration": best,
                             "accuracy": float(accuracy_score(y[valid_idx], probability >= 0.5)),
                             "seconds": time.monotonic() - fold_start})
    artifacts = save_predictions(name, train, test, sample, folds, oof, test_probability)
    result = {
        "experiment_id": name, "config": config, "fingerprint": fingerprint,
        "completed_at": timestamp(), "accuracy": float(accuracy_score(y, oof >= 0.5)),
        "log_loss": float(log_loss(y, oof)),
        "fold_std": float(np.std([f["accuracy"] for f in fold_results])),
        "folds": fold_results, "threads": threads, "runtime_seconds": time.monotonic() - started,
        "raw_hashes": raw, "code_hashes": code, "artifacts": artifacts,
        "segments": diagnostics(train, oof),
    }
    write_json(metrics_path, result)
    print(f"DONE {name}: accuracy={result['accuracy']:.6f}, logloss={result['log_loss']:.6f}",
          flush=True)
    return result
