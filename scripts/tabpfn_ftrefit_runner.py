"""H-A-24: TabPFN v3.5 fine-tuning, epoch count chosen fold-locally, then refit on the whole fold.

Per outer fit fold F:
- stage 1: fine-tune on F minus an SGKF(8) slice, early stopping on that slice's log loss
  (same settings as H-A-23); an epoch tracker records `val/log_loss` after every epoch and the
  best epoch count k is the argmin;
- stage 2: fine-tune a fresh model on all of F for exactly k epochs with validation disabled, so
  the inference context is the whole fold.
The outer validation fold and the test set are only predicted.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FT = {"epochs": 30, "learning_rate": 1e-5, "early_stopping_patience": 8, "time_limit": 600,
      "eval_metric": "log_loss"}


class EpochTracker:
    def __init__(self):
        self.val_loss: list[tuple[int, float]] = []

    def setup(self, config):
        pass

    def log_step(self, metrics, step):
        pass

    def log_epoch(self, metrics, step):
        if "val/log_loss" in metrics:
            self.val_loss.append((int(step), float(metrics["val/log_loss"])))

    def finish(self):
        pass


def run_tabpfn_ftrefit(name: str, seed: int, threads: int) -> dict:
    import warnings

    import numpy as np
    import tabpfn
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import OrdinalEncoder
    from tabpfn.constants import ModelVersion
    from tabpfn.finetuning import FinetunedTabPFNClassifier

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
    warnings.filterwarnings("ignore", message=".*output_dir.*")
    config = {"experiment_id": name, "model": "tabpfn_finetuned_refit", "model_version": "v3.5",
              "tabpfn_version": tabpfn.__version__, "finetune": FT,
              "selection": "SGKF(8) slice, argmin val/log_loss epoch", "refit": "whole fold",
              "feature_variant": "baseline", "random_state": seed, "n_splits": 5}
    code = {p: file_hash(ROOT / p) for p in [
        "scripts/tabpfn_ftrefit_runner.py", "src/spaceship_titanic/experiments.py",
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
    cat_idx = [x.columns.get_loc(c) for c in categorical]
    oof, test_probability, fold_results = np.zeros(len(x)), np.zeros(len(tx)), []
    for fold in range(1, 6):
        fold_start = time.monotonic()
        fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        fit_seed = seed + fold
        a, b = next(StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=fit_seed)
                    .split(fit_idx, y[fit_idx], groups.iloc[fit_idx]))
        tr_idx, es_idx = fit_idx[a], fit_idx[b]
        assert not set(groups.iloc[tr_idx]) & set(groups.iloc[es_idx])
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(x.iloc[fit_idx][categorical])

        def encode(rows=None, frame=None, _encoder=encoder):
            frame = (x.iloc[rows] if frame is None else frame).copy()
            frame[categorical] = _encoder.transform(frame[categorical])
            return frame.to_numpy(dtype=np.float32)

        tracker = EpochTracker()
        stage1 = FinetunedTabPFNClassifier(
            device="cuda", random_state=fit_seed, model_version=ModelVersion.V3_5,
            extra_classifier_kwargs={"categorical_features_indices": cat_idx},
            experiment_logger=tracker, **FT)
        stage1.fit(encode(tr_idx), y[tr_idx], X_val=encode(es_idx), y_val=y[es_idx])
        best_epochs = min(tracker.val_loss, key=lambda t: t[1])[0] if tracker.val_loss else 1
        del stage1
        refit = FinetunedTabPFNClassifier(
            device="cuda", random_state=fit_seed, model_version=ModelVersion.V3_5,
            extra_classifier_kwargs={"categorical_features_indices": cat_idx},
            epochs=best_epochs, learning_rate=FT["learning_rate"], validation_split_ratio=None,
            early_stopping=False, time_limit=FT["time_limit"])
        refit.fit(encode(fit_idx), y[fit_idx])
        probability = refit.predict_proba(encode(valid_idx))[:, 1]
        oof[valid_idx] = probability
        test_probability += refit.predict_proba(encode(frame=tx))[:, 1] / 5
        fold_results.append({"fold": fold, "best_epochs": best_epochs,
                             "accuracy": float(accuracy_score(y[valid_idx], probability >= 0.5)),
                             "seconds": time.monotonic() - fold_start})
        print(f"{name} fold {fold}: {fold_results[-1]['accuracy']:.6f} epochs={best_epochs} "
              f"({fold_results[-1]['seconds']:.0f}s)", flush=True)
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
