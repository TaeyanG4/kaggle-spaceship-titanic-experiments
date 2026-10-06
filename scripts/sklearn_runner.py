"""sklearn diversity members for H-A-05: ExtraTrees and HistGradientBoosting (no new dependency).

Categoricals are ordinal-encoded on the fit fold only; numeric NaN is median-filled on the fit fold
for ExtraTrees (HistGB handles NaN). HistGB picks max_iter by 4-fold inner SGKF log loss
(median of the per-split argmin over a staged path); ExtraTrees has no iteration choice.
Cached by fingerprint like the other runners.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARAMS = {
    "extratrees": {"n_estimators": 600, "min_samples_leaf": 3, "max_features": 0.5,
                   "n_jobs": 1},
    "histgb": {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 20,
               "l2_regularization": 1.0, "max_iter": 1000, "early_stopping": False},
}


def run_sklearn(name: str, seed: int, kind: str, threads: int, variant: str = "baseline") -> dict:
    import numpy as np
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import OrdinalEncoder

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

    if kind not in PARAMS:
        raise ValueError(f"Unknown sklearn model: {kind}")
    params = PARAMS[kind] | ({"n_jobs": threads} if kind == "extratrees" else {})
    config = {"experiment_id": name, "model": kind, "feature_variant": variant,
              "random_state": seed, "n_splits": 5, "params": PARAMS[kind],
              "stopping": "innercv_logloss" if kind == "histgb" else "none"}
    code = {p: file_hash(ROOT / p) for p in [
        "scripts/sklearn_runner.py", "src/spaceship_titanic/experiments.py",
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
    y = train.Transported.astype(int).to_numpy()
    groups = train.PassengerId.str.split("_").str[0]
    oof, test_probability, fold_results = np.zeros(len(x)), np.zeros(len(tx)), []
    cat_mask = [c in categorical for c in x.columns]

    def encode(fit_frame, *frames):
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(fit_frame[categorical])
        median = fit_frame.drop(columns=categorical).median()
        out = []
        for frame in (fit_frame, *frames):
            frame = frame.copy()
            frame[categorical] = encoder.transform(frame[categorical])
            if kind == "extratrees":
                frame = frame.fillna(median)
            out.append(frame.to_numpy(dtype=float))
        return out

    def model(fit_seed, **overrides):
        if kind == "extratrees":
            return ExtraTreesClassifier(**(params | overrides), random_state=fit_seed)
        return HistGradientBoostingClassifier(**(params | overrides), random_state=fit_seed,
                                              categorical_features=cat_mask)

    for fold in range(1, 6):
        fold_start = time.monotonic()
        fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        fit_seed = seed + fold
        best = None
        overrides = {}
        if kind == "histgb":
            bests = []
            splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=fit_seed)
            for a, b in splitter.split(fit_idx, y[fit_idx], groups.iloc[fit_idx]):
                fa, fb = encode(x.iloc[fit_idx[a]], x.iloc[fit_idx[b]])
                probe = model(fit_seed).fit(fa, y[fit_idx[a]])
                losses = [log_loss(y[fit_idx[b]], p[:, 1], labels=[0, 1])
                          for p in probe.staged_predict_proba(fb)]
                bests.append(int(np.argmin(losses)) + 1)
            best = int(np.median(bests))
            overrides = {"max_iter": best}
        fa, fv, ft = encode(x.iloc[fit_idx], x.iloc[valid_idx], tx)
        final = model(fit_seed, **overrides).fit(fa, y[fit_idx])
        probability = final.predict_proba(fv)[:, 1]
        oof[valid_idx] = probability
        test_probability += final.predict_proba(ft)[:, 1] / 5
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
