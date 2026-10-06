"""H-A-39/40/41: fine-tuned TabPFN v3.5 with a configurable step budget plus inference variants.

Same recipe as `tabpfn_ft_runner.py` (H-A-23: SGKF(8) slice of the fit fold for early stopping,
fit-fold ordinal encoding) with the fine-tuning settings chosen by arm: `base` (H-A-23 recipe),
`A1` (100 epochs, patience 20, 900 s) or `A2` (lr 3e-5). From the same fine-tuned weights it also
writes `<name>_ctx` (inference context = whole fit fold, H-A-40) and `<name>_ne2` / `<name>_ne16`
(2 or 16 inference estimators instead of 8, H-A-41). The scored fold is only predicted.
Usage: `uv run python scripts/tabpfn_ftbudget_runner.py <arm> <seed> [<seed> ...]`.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BASE_FT = {"epochs": 30, "learning_rate": 1e-5, "early_stopping_patience": 8, "time_limit": 600,
           "eval_metric": "log_loss"}
ARMS = {
    "base": BASE_FT,
    "A1": BASE_FT | {"epochs": 100, "early_stopping_patience": 20, "time_limit": 900},
    "A2": BASE_FT | {"learning_rate": 3e-5},
}
VARIANTS = ("ctx", "ne2", "ne16")


def run_ftbudget(name: str, seed: int, arm: str, threads: int = 2) -> dict:
    import copy
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
    FT = ARMS[arm]
    config = {"experiment_id": name, "model": "tabpfn_finetuned", "model_version": "v3.5",
              "tabpfn_version": tabpfn.__version__, "finetune": FT, "arm": arm,
              "variants": list(VARIANTS), "val_split": "SGKF(8) slice",
              "feature_variant": "baseline", "random_state": seed, "n_splits": 5}
    code = {p: file_hash(ROOT / p) for p in [
        "scripts/tabpfn_ftbudget_runner.py", "src/spaceship_titanic/experiments.py",
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
    extra_oof = {v: np.zeros(len(x)) for v in VARIANTS}
    extra_test = {v: np.zeros(len(tx)) for v in VARIANTS}
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
        frames = []
        for rows in (tr_idx, es_idx, valid_idx, fit_idx):
            frame = x.iloc[rows].copy()
            frame[categorical] = encoder.transform(frame[categorical])
            frames.append(frame.to_numpy(dtype=np.float32))
        test_frame = tx.copy()
        test_frame[categorical] = encoder.transform(test_frame[categorical])
        ft_x, es_x, va_x, fit_x = frames
        model = FinetunedTabPFNClassifier(
            device="cuda", random_state=fit_seed, model_version=ModelVersion.V3_5,
            extra_classifier_kwargs={"categorical_features_indices": cat_idx}, **FT)
        model.fit(ft_x, y[tr_idx], X_val=es_x, y_val=y[es_idx])
        probability = model.predict_proba(va_x)[:, 1]
        oof[valid_idx] = probability
        test_x = test_frame.to_numpy(dtype=np.float32)
        test_probability += model.predict_proba(test_x)[:, 1] / 5
        # Same fine-tuned weights (the inference classifier wraps them as its model spec):
        # ctx = re-conditioned on the whole fit fold (H-A-40); ne2/ne16 = estimator counts (H-A-41).
        for variant in VARIANTS:
            clf = copy.deepcopy(model.finetuned_inference_classifier_)
            if variant == "ctx":
                clf.fit(fit_x, y[fit_idx])
            else:
                clf.n_estimators = int(variant[2:])
                clf.fit(ft_x, y[tr_idx])
            extra_oof[variant][valid_idx] = clf.predict_proba(va_x)[:, 1]
            extra_test[variant] += clf.predict_proba(test_x)[:, 1] / 5
        fold_results.append({"fold": fold,
                             "accuracy": float(accuracy_score(y[valid_idx], probability >= 0.5)),
                             "seconds": time.monotonic() - fold_start})
        print(f"{name} fold {fold}: {fold_results[-1]['accuracy']:.6f} "
              f"({fold_results[-1]['seconds']:.0f}s)", flush=True)
    artifacts = save_predictions(name, train, test, sample, folds, oof, test_probability)
    variant_accuracy = {}
    for variant in VARIANTS:
        save_predictions(f"{name}_{variant}", train, test, sample, folds, extra_oof[variant],
                         extra_test[variant])
        variant_accuracy[variant] = float(accuracy_score(y, extra_oof[variant] >= 0.5))
    result = {
        "experiment_id": name, "config": config, "fingerprint": fingerprint,
        "completed_at": timestamp(), "accuracy": float(accuracy_score(y, oof >= 0.5)),
        "log_loss": float(log_loss(y, oof)),
        "fold_std": float(np.std([f["accuracy"] for f in fold_results])),
        "folds": fold_results, "threads": threads, "runtime_seconds": time.monotonic() - started,
        "raw_hashes": raw, "code_hashes": code, "artifacts": artifacts,
        "segments": diagnostics(train, oof), "variant_accuracy": variant_accuracy,
    }
    write_json(metrics_path, result)
    print(f"DONE {name}: accuracy={result['accuracy']:.6f}, logloss={result['log_loss']:.6f}, "
          f"variants={ {k: round(v, 6) for k, v in variant_accuracy.items()} }", flush=True)
    return result


if __name__ == "__main__":
    import sys

    arm, *seeds = sys.argv[1:]
    for s in seeds:
        run_ftbudget(f"ha39_{arm.lower()}" + ("" if int(s) == 42 else f"_seed{s}"), int(s), arm)
