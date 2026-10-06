"""H-A-12: TabPFN v3.5 (pretrained tabular foundation model) as a fold-local member.

Baseline features; categoricals ordinal-encoded with an encoder fitted on the fit fold and passed
as `categorical_features_indices`. No iteration selection (in-context learner). Requires the user
to have accepted the Prior Labs license and set `TABPFN_TOKEN` in the environment; this script
never reads, prints, or stores the token itself.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_tabpfn(name: str, seed: int, threads: int, version: str = "v3.5") -> dict:
    import os

    import numpy as np
    import tabpfn
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.preprocessing import OrdinalEncoder
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

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

    os.environ.setdefault("TABPFN_NO_BROWSER", "1")  # fail fast instead of waiting for a browser
    model_version = ModelVersion(version)
    config = {"experiment_id": name, "model": "tabpfn", "tabpfn_version": tabpfn.__version__,
              "model_version": version, "feature_variant": "baseline", "random_state": seed,
              "n_splits": 5}
    code = {p: file_hash(ROOT / p) for p in [
        "scripts/tabpfn_runner.py", "src/spaceship_titanic/experiments.py",
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
    cat_idx = [x.columns.get_loc(c) for c in categorical]
    oof, test_probability, fold_results = np.zeros(len(x)), np.zeros(len(tx)), []
    for fold in range(1, 6):
        fold_start = time.monotonic()
        fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(x.iloc[fit_idx][categorical])
        frames = []
        for frame in (x.iloc[fit_idx], x.iloc[valid_idx], tx):
            frame = frame.copy()
            frame[categorical] = encoder.transform(frame[categorical])
            frames.append(frame.to_numpy(dtype=np.float32))
        fa, fv, ft = frames
        model = TabPFNClassifier.create_default_for_version(
            model_version, categorical_features_indices=cat_idx, device="cuda",
            random_state=seed + fold)
        model.fit(fa, y[fit_idx])
        probability = model.predict_proba(fv)[:, 1]
        oof[valid_idx] = probability
        test_probability += model.predict_proba(ft)[:, 1] / 5
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
