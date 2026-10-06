"""XGBoost fold runner with honest iteration selection (H-Main-05 onward).

Stopping: `innercv_logloss` (4-fold inner SGKF on the fit fold, Logloss early stopping, refit on
the whole fit fold at the median best iteration) or `fixed<N>`. Categoricals are ordinal-encoded
with an encoder fitted on the fit fold only. Cached by fingerprint like `catboost_runner`.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_PARAMS = {
    "n_estimators": 1500, "learning_rate": 0.04, "max_depth": 4, "min_child_weight": 5,
    "reg_lambda": 5.0, "subsample": 0.85, "colsample_bytree": 0.9, "tree_method": "hist",
    "eval_metric": "logloss",
}
EARLY_STOPPING_ROUNDS = 100


def run_xgboost(name: str, seed: int, stopping: str, drop: list[str], threads: int,
                variant: str = "baseline", params: dict | None = None) -> dict:
    import numpy as np
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import OrdinalEncoder
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
    from spaceship_titanic.versioned_features import build_versioned_features

    fixed = stopping.startswith("fixed") and stopping[5:].isdigit()
    if not fixed and stopping != "innercv_logloss":
        raise ValueError(f"Unknown stopping mode: {stopping}")
    params = BASE_PARAMS | (params or {})
    config = {"experiment_id": name, "model": "xgboost", "feature_variant": variant,
              "random_state": seed, "n_splits": 5, "stopping": stopping, "drop": sorted(drop),
              "early_stopping_rounds": EARLY_STOPPING_ROUNDS, "model_params": params}
    code = {p: file_hash(ROOT / p) for p in [
        "scripts/xgb_runner.py", "src/spaceship_titanic/experiments.py",
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
    x, tx = x.drop(columns=drop), tx.drop(columns=drop)
    categorical = [c for c in categorical if c not in drop]
    y = train.Transported.astype(int).to_numpy()
    groups = train.PassengerId.str.split("_").str[0]
    oof, test_probability, fold_results = np.zeros(len(train)), np.zeros(len(test)), []

    def encode(fit_frame, *frames):
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(fit_frame[categorical])
        out = []
        for frame in (fit_frame, *frames):
            frame = frame.copy()
            frame[categorical] = encoder.transform(frame[categorical]).astype(float)
            frame[categorical] = frame[categorical].replace(-1, np.nan)
            out.append(frame)
        return out

    def model(fit_seed, **overrides):
        return XGBClassifier(**(params | overrides), random_state=fit_seed, n_jobs=threads)

    for fold in range(1, 6):
        fold_start = time.monotonic()
        fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        fit_seed = seed + fold
        if fixed:
            best = int(stopping[5:])
        else:
            bests = []
            splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=fit_seed)
            for a, b in splitter.split(x.iloc[fit_idx], y[fit_idx], groups.iloc[fit_idx]):
                inner_fit, inner_eval = encode(x.iloc[fit_idx[a]], x.iloc[fit_idx[b]])
                probe = model(fit_seed, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
                probe.fit(inner_fit, y[fit_idx[a]], eval_set=[(inner_eval, y[fit_idx[b]])],
                          verbose=False)
                bests.append(int(probe.best_iteration) + 1)
            best = int(np.median(bests))
        fit_frame, valid_frame, test_frame = encode(x.iloc[fit_idx], x.iloc[valid_idx], tx)
        final = model(fit_seed, n_estimators=best)
        final.fit(fit_frame, y[fit_idx], verbose=False)
        probability = final.predict_proba(valid_frame)[:, 1]
        oof[valid_idx] = probability
        test_probability += final.predict_proba(test_frame)[:, 1] / 5
        score = float(accuracy_score(y[valid_idx], probability >= 0.5))
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
