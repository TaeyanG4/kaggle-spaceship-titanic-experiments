"""AutoGluon runner for H-A-34 (run inside `.venv-autogluon`).

Usage: `.venv-autogluon/Scripts/python scripts/autogluon_runner.py <experiment_id> <preset> <time_limit>
<seed> [<seed> ...]` or `... smoke <preset> <time_limit>`.
Per outer frozen fold, a TabularPredictor is fitted on the baseline features of the fit fold only,
with `groups=BagFold`: AutoGluon treats each distinct group value as one bagging fold, so the fit
fold is first split by StratifiedGroupKFold(8) on PassengerId groups and the inner fold number is
passed as the group (travel groups stay together; the column is not a feature). The outer fold is scored once with the final
ensemble; nothing is tuned on it. Output format matches the other runners; cached by fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_autogluon(name: str, seed: int, preset: str, time_limit: int, smoke: bool = False):
    import numpy as np
    from autogluon.tabular import TabularPredictor
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

    config = {"experiment_id": name, "model": "autogluon", "preset": preset,
              "time_limit": time_limit, "random_state": seed, "n_splits": 5,
              "feature_variant": "baseline", "groups": "SGKF(8) BagFold"}
    code = {p: file_hash(ROOT / p) for p in [
        "scripts/autogluon_runner.py", "src/spaceship_titanic/experiments.py",
        "src/spaceship_titanic/versioned_features.py", "src/spaceship_titanic/features.py"]}
    raw = {n: file_hash(ROOT / "data/raw" / n) for n in
           ["train.csv", "test.csv", "sample_submission.csv"]}
    fingerprint = hashlib.sha256(json.dumps({"config": config, "raw": raw, "code": code},
                                            sort_keys=True).encode()).hexdigest()
    metrics_path = ROOT / f"reports/{name}_metrics.json"
    if not smoke and metrics_path.exists():
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        if previous["fingerprint"] == fingerprint:
            print(f"REUSE {name}: {previous['accuracy']:.6f}", flush=True)
            return previous
        raise ValueError(f"Existing experiment differs; use a new ID: {name}")

    started = time.monotonic()
    train, test, sample = load_data()
    folds = frozen_folds(train, seed, 5)
    x, tx, categorical = build_versioned_features(train, test, "baseline")
    x, tx = x.reset_index(drop=True), tx.reset_index(drop=True)
    for frame in (x, tx):
        frame[categorical] = frame[categorical].astype(object)
    y = train.Transported.astype(int).to_numpy()
    groups = train.PassengerId.str[:4].to_numpy()
    x["Transported"] = y
    oof, test_probability, fold_results = np.zeros(len(x)), np.zeros(len(tx)), []
    work = ROOT / "models" / "autogluon_tmp" / name
    for fold in range(1, 6):
        fold_start = time.monotonic()
        fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        shutil.rmtree(work, ignore_errors=True)
        fit_frame = x.iloc[fit_idx].reset_index(drop=True)
        fit_frame["BagFold"] = 0
        inner = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed + fold)
        for k, (_, part) in enumerate(inner.split(fit_idx, y[fit_idx], groups[fit_idx])):
            fit_frame.loc[part, "BagFold"] = k
        predictor = TabularPredictor(label="Transported", eval_metric="accuracy",
                                     groups="BagFold", path=str(work), verbosity=1)
        # "extreme_nc": the extreme preset with the non-commercial 2026-08-05 portfolio, which
        # adds TabPFN-3 to CatBoost, LightGBM, NORI, RealMLP, TabDPT-Turbo and TabICL.
        extra = ({"hyperparameters": "noncommercial_2026_08_05"} if preset == "extreme_nc"
                 else {})
        predictor.fit(fit_frame, presets="extreme" if preset == "extreme_nc" else preset, **extra,
                      time_limit=time_limit, ag_args_fit={"random_seed": seed + fold})
        features = x.drop(columns=["Transported"])
        probability = predictor.predict_proba(features.iloc[valid_idx])[1].to_numpy()
        if smoke:
            print(f"SMOKE autogluon {preset} fold {fold}: acc="
                  f"{accuracy_score(y[valid_idx], probability >= 0.5):.4f} "
                  f"({time.monotonic() - fold_start:.0f}s)", flush=True)
            print(predictor.leaderboard(silent=True).head(8).to_string(), flush=True)
            shutil.rmtree(work, ignore_errors=True)
            return None
        oof[valid_idx] = probability
        test_probability += predictor.predict_proba(tx)[1].to_numpy() / 5
        fold_results.append({"fold": fold,
                             "accuracy": float(accuracy_score(y[valid_idx], probability >= 0.5)),
                             "best_model": predictor.model_best,
                             "seconds": time.monotonic() - fold_start})
        print(f"{name} fold {fold}: {fold_results[-1]['accuracy']:.6f} "
              f"best={predictor.model_best} ({fold_results[-1]['seconds']:.0f}s)", flush=True)
        shutil.rmtree(work, ignore_errors=True)
    artifacts = save_predictions(name, train, test, sample, folds, oof, test_probability)
    result = {
        "experiment_id": name, "config": config, "fingerprint": fingerprint,
        "completed_at": timestamp(), "accuracy": float(accuracy_score(y, oof >= 0.5)),
        "log_loss": float(log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6))),
        "fold_std": float(np.std([f["accuracy"] for f in fold_results])),
        "folds": fold_results, "runtime_seconds": time.monotonic() - started,
        "raw_hashes": raw, "code_hashes": code, "artifacts": artifacts,
        "segments": diagnostics(train, oof),
    }
    write_json(metrics_path, result)
    print(f"DONE {name}: accuracy={result['accuracy']:.6f}, logloss={result['log_loss']:.6f}",
          flush=True)
    return result


if __name__ == "__main__":
    experiment, preset, limit, *seeds = sys.argv[1:]
    if experiment == "smoke":
        run_autogluon("smoke", 42, preset, int(limit), smoke=True)
    else:
        for s in seeds:
            run_autogluon(experiment if int(s) == 42 else f"{experiment}_seed{s}", int(s),
                          preset, int(limit))
