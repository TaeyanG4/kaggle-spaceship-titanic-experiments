"""H-A-20: TabPFN v3.5 on other feature sets (separate module so `ha12_tabpfn` caches stay valid).

Same fold-local recipe as `tabpfn_runner.py` (ordinal encoder fitted on the fit fold, categorical
indices passed, v3.5 pinned); only the feature set changes: a `versioned_features` variant plus an
optional list of columns to drop.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_MINIMAL_DROP = ["TotalSpend", "NoSpend", "SpendMissingCount", "IsChild", "IsTeen", "IsAdult",
                    "SurnameSize", "IsAlone"]


def run_tabpfn_variant(name: str, seed: int, threads: int, variant: str,
                       drop: list[str] | None = None) -> dict:
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

    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    drop = sorted(drop or [])
    config = {"experiment_id": name, "model": "tabpfn", "tabpfn_version": tabpfn.__version__,
              "model_version": "v3.5", "feature_variant": variant, "drop": drop,
              "random_state": seed, "n_splits": 5}
    code = {p: file_hash(ROOT / p) for p in [
        "scripts/tabpfn_variant_runner.py", "src/spaceship_titanic/experiments.py",
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
            ModelVersion.V3_5, categorical_features_indices=cat_idx, device="cuda",
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
        "folds": fold_results, "feature_columns": x.columns.tolist(), "threads": threads,
        "runtime_seconds": time.monotonic() - started, "raw_hashes": raw, "code_hashes": code,
        "artifacts": artifacts, "segments": diagnostics(train, oof),
    }
    write_json(metrics_path, result)
    print(f"DONE {name}: accuracy={result['accuracy']:.6f}, logloss={result['log_loss']:.6f}",
          flush=True)
    return result
