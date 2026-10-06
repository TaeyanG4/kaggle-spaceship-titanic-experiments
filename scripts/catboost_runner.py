"""General CatBoost fold runner for feature-subset experiments (H-Main-03 onward).

Stopping modes: `valid` (stop on the scored fold; legacy, optimistic per D-Main-026), `inner` /
`inner_refit` (Accuracy stopping on a 1/8 stratified group holdout of the fit fold), `fixed<N>`
(N iterations, no stopping), `inner_logloss_refit` (Logloss stopping on the 1/8 holdout, refit on
the whole fit fold at that iteration), `innercv_logloss` (4-fold inner SGKF on the fit fold with
Logloss stopping; refit on the whole fit fold at the median best iteration).
Runs are cached by fingerprint (config + raw data + this file + feature code).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_PARAMS = {
    "iterations": 800, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5.0,
    "random_strength": 0.5, "loss_function": "Logloss", "eval_metric": "Accuracy",
}
EARLY_STOPPING_ROUNDS = 100


def run_catboost(name: str, seed: int, stopping: str, drop: list[str], threads: int,
                 variant: str = "baseline", params: dict | None = None) -> dict:
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
    from spaceship_titanic.versioned_features import build_versioned_features

    fixed = stopping.startswith("fixed") and stopping[5:].isdigit()
    if not fixed and stopping not in ("valid", "inner", "inner_refit", "inner_logloss_refit",
                                      "innercv_logloss"):
        raise ValueError(f"Unknown stopping mode: {stopping}")
    params = BASE_PARAMS | (params or {})
    config = {"experiment_id": name, "feature_variant": variant, "random_state": seed,
              "n_splits": 5, "stopping": stopping, "drop": sorted(drop),
              "early_stopping_rounds": EARLY_STOPPING_ROUNDS, "model_params": params}
    code = {p: file_hash(ROOT / p) for p in [
        "scripts/catboost_runner.py", "src/spaceship_titanic/experiments.py",
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
    x, tx, categorical = build_versioned_features(train, test, variant)
    missing = set(drop) - set(x.columns)
    if missing:
        raise ValueError(f"Unknown columns to drop: {sorted(missing)}")
    x, tx = x.drop(columns=drop), tx.drop(columns=drop)
    categorical = [c for c in categorical if c not in drop]
    y = train.Transported.astype(int)
    groups = train.PassengerId.str.split("_").str[0]
    oof, test_probability, fold_results = np.zeros(len(train)), np.zeros(len(test)), []

    def model(**overrides):
        return CatBoostClassifier(**(params | overrides), thread_count=threads,
                                  verbose=False, allow_writing_files=False)

    for fold in range(1, 6):
        fold_start = time.monotonic()
        fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        fit_seed = seed + fold
        if fixed:
            best = int(stopping[5:])
        elif stopping == "innercv_logloss":
            splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=fit_seed)
            bests = []
            for a, b in splitter.split(x.iloc[fit_idx], y.iloc[fit_idx], groups.iloc[fit_idx]):
                probe = model(random_seed=fit_seed, eval_metric="Logloss")
                probe.fit(x.iloc[fit_idx[a]], y.iloc[fit_idx[a]], cat_features=categorical,
                          eval_set=(x.iloc[fit_idx[b]], y.iloc[fit_idx[b]]),
                          early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
                bests.append(int(probe.get_best_iteration()) + 1)
            best = int(np.median(bests))
        else:
            if stopping == "valid":
                stop_fit, stop_eval = fit_idx, valid_idx
            else:
                splitter = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=fit_seed)
                a, b = next(splitter.split(x.iloc[fit_idx], y.iloc[fit_idx], groups.iloc[fit_idx]))
                stop_fit, stop_eval = fit_idx[a], fit_idx[b]
                assert not set(groups.iloc[stop_fit]) & set(groups.iloc[stop_eval])
                assert not set(stop_eval) & set(valid_idx)
            metric = {"eval_metric": "Logloss"} if stopping == "inner_logloss_refit" else {}
            final = model(random_seed=fit_seed, **metric)
            final.fit(x.iloc[stop_fit], y.iloc[stop_fit], cat_features=categorical,
                      eval_set=(x.iloc[stop_eval], y.iloc[stop_eval]),
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
            best = int(final.get_best_iteration()) + 1
        if fixed or stopping in ("inner_refit", "inner_logloss_refit", "innercv_logloss"):
            final = model(iterations=best, random_seed=fit_seed)
            final.fit(x.iloc[fit_idx], y.iloc[fit_idx], cat_features=categorical)
        probability = final.predict_proba(x.iloc[valid_idx], thread_count=threads)[:, 1]
        oof[valid_idx] = probability
        test_probability += final.predict_proba(tx, thread_count=threads)[:, 1] / 5
        score = float(accuracy_score(y.iloc[valid_idx], probability >= 0.5))
        fold_results.append({"fold": fold, "accuracy": score, "best_iteration": best,
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
