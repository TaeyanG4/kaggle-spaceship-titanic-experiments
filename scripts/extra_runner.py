"""CatBoost champion recipe (`innercv_logloss`) on baseline features plus target-free extras.

Extras come from `extra_features.py` (agent A items H-A-02/03/04). Cached by fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARAMS = {"iterations": 800, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5.0,
          "random_strength": 0.5, "loss_function": "Logloss", "eval_metric": "Accuracy"}


def run_catboost_extra(name: str, seed: int, extras: list[str], threads: int) -> dict:
    import extra_features
    import numpy as np
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

    config = {"experiment_id": name, "feature_variant": "baseline", "extras": list(extras),
              "random_state": seed, "n_splits": 5, "stopping": "innercv_logloss",
              "model_params": PARAMS}
    code = {p: file_hash(ROOT / p) for p in [
        "scripts/extra_runner.py", "scripts/extra_features.py",
        "src/spaceship_titanic/experiments.py", "src/spaceship_titanic/versioned_features.py",
        "src/spaceship_titanic/features.py"]}
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
    x, tx, categorical = extra_features.build(train, test, list(extras))
    for col in categorical:
        x[col], tx[col] = x[col].astype(str), tx[col].astype(str)
    y = train.Transported.astype(int).to_numpy()
    groups = train.PassengerId.str.split("_").str[0]
    oof, test_probability, fold_results = np.zeros(len(x)), np.zeros(len(tx)), []

    def model(**overrides):
        return CatBoostClassifier(**(PARAMS | overrides), thread_count=threads, verbose=False,
                                  allow_writing_files=False)

    for fold in range(1, 6):
        fold_start = time.monotonic()
        fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        fit_seed = seed + fold
        bests = []
        splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=fit_seed)
        for a, b in splitter.split(fit_idx, y[fit_idx], groups.iloc[fit_idx]):
            probe = model(random_seed=fit_seed, eval_metric="Logloss")
            probe.fit(x.iloc[fit_idx[a]], y[fit_idx[a]], cat_features=categorical,
                      eval_set=(x.iloc[fit_idx[b]], y[fit_idx[b]]), early_stopping_rounds=100,
                      verbose=False)
            bests.append(int(probe.get_best_iteration()) + 1)
        best = int(np.median(bests))
        final = model(iterations=best, random_seed=fit_seed)
        final.fit(x.iloc[fit_idx], y[fit_idx], cat_features=categorical)
        probability = final.predict_proba(x.iloc[valid_idx], thread_count=threads)[:, 1]
        oof[valid_idx] = probability
        test_probability += final.predict_proba(tx, thread_count=threads)[:, 1] / 5
        fold_results.append({"fold": fold, "best_iteration": best,
                             "accuracy": float(accuracy_score(y[valid_idx], probability >= 0.5)),
                             "seconds": time.monotonic() - fold_start})
    artifacts = save_predictions(name, train, test, sample, folds, oof, test_probability)
    result = {
        "experiment_id": name, "config": config, "fingerprint": fingerprint,
        "completed_at": timestamp(), "accuracy": float(accuracy_score(y, oof >= 0.5)),
        "log_loss": float(log_loss(y, oof)),
        "fold_std": float(np.std([f["accuracy"] for f in fold_results])),
        "folds": fold_results, "feature_columns": x.columns.tolist(), "categorical": categorical,
        "threads": threads, "runtime_seconds": time.monotonic() - started, "raw_hashes": raw,
        "code_hashes": code, "artifacts": artifacts, "segments": diagnostics(train, oof),
    }
    write_json(metrics_path, result)
    print(f"DONE {name}: accuracy={result['accuracy']:.6f}, logloss={result['log_loss']:.6f}",
          flush=True)
    return result
