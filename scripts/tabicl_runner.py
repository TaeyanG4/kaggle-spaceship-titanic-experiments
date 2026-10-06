"""H-A-18: TabICL (open-weight in-context tabular foundation model, BSD-3) as a fold-local member.

Baseline features passed as a DataFrame (categoricals as strings); TabICL's own preprocessing is
fitted on the fit fold only. No iteration selection. Cached by fingerprint like the other runners.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_tabicl(name: str, seed: int, threads: int) -> dict:
    import numpy as np
    import tabicl
    from sklearn.metrics import accuracy_score, log_loss
    from tabicl import TabICLClassifier

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

    config = {"experiment_id": name, "model": "tabicl",
              "tabicl_version": getattr(tabicl, "__version__", "unknown"),
              "feature_variant": "baseline", "random_state": seed, "n_splits": 5}
    code = {p: file_hash(ROOT / p) for p in [
        "scripts/tabicl_runner.py", "src/spaceship_titanic/experiments.py",
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
    for col in categorical:
        x[col], tx[col] = x[col].astype(str), tx[col].astype(str)
    y = train.Transported.astype(int).to_numpy()
    oof, test_probability, fold_results = np.zeros(len(x)), np.zeros(len(tx)), []
    for fold in range(1, 6):
        fold_start = time.monotonic()
        fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        model = TabICLClassifier(device="cuda", random_state=seed + fold)
        model.fit(x.iloc[fit_idx], y[fit_idx])
        probability = model.predict_proba(x.iloc[valid_idx])[:, 1]
        oof[valid_idx] = probability
        test_probability += model.predict_proba(tx)[:, 1] / 5
        fold_results.append({"fold": fold,
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
