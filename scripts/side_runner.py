"""Side-venv runner for TabDPT (H-A-28) and TabSTAR (H-A-32).

Run with `.venv-side/Scripts/python scripts/side_runner.py <model> <experiment_id> <seed>`; the side
venv exists because these packages would downgrade huggingface-hub in the main env. Same frozen
folds, baseline features and output format as the main runners. TabDPT gets fit-fold ordinal codes
(in-context, no stopping); TabSTAR gets the baseline frame with readable strings and an SGKF(8)
slice of the fit fold as its early-stopping set. Cached by fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("tabdpt", "tabstar")


def run_side(name: str, seed: int, model: str, device: str = "cuda", limit: int | None = None):
    import numpy as np
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

    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    config = {"experiment_id": name, "model": model, "random_state": seed, "n_splits": 5,
              "feature_variant": "baseline",
              "early_stopping": "SGKF(8) slice" if model == "tabstar" else "none"}
    code = {p: file_hash(ROOT / p) for p in [
        "scripts/side_runner.py", "src/spaceship_titanic/experiments.py",
        "src/spaceship_titanic/versioned_features.py", "src/spaceship_titanic/features.py"]}
    raw = {n: file_hash(ROOT / "data/raw" / n) for n in
           ["train.csv", "test.csv", "sample_submission.csv"]}
    fingerprint = hashlib.sha256(json.dumps({"config": config, "raw": raw, "code": code},
                                            sort_keys=True).encode()).hexdigest()
    metrics_path = ROOT / f"reports/{name}_metrics.json"
    if limit is None and metrics_path.exists():
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
    y = train.Transported.astype(int).to_numpy()
    groups = train.PassengerId.str.split("_").str[0]
    oof, test_probability, fold_results = np.zeros(len(x)), np.zeros(len(tx)), []
    for fold in range(1, 6):
        fold_start = time.monotonic()
        fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        fit_seed = seed + fold
        test_frame = tx
        if limit is not None:  # smoke test only: subsample fit rows, score a slice, skip test
            fit_idx, valid_idx = fit_idx[:limit], valid_idx[: limit // 3]
            test_frame = tx.iloc[:10]
        if model == "tabdpt":
            from tabdpt import TabDPTClassifier
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                     encoded_missing_value=-1)
            encoder.fit(x.iloc[fit_idx][categorical].astype(str))

            def encode(frame, encoder=encoder):
                frame = frame.copy()
                frame[categorical] = encoder.transform(frame[categorical].astype(str))
                return frame.to_numpy(dtype=np.float32)

            est = TabDPTClassifier(device=device, verbose=False, compile=False,
                                   use_flash=False)  # no triton / flash kernel on Windows
            est.fit(encode(x.iloc[fit_idx]), y[fit_idx])
            probability = est.predict_proba(encode(x.iloc[valid_idx]), seed=fit_seed)[:, 1]
            test_part = est.predict_proba(encode(test_frame), seed=fit_seed)[:, 1]
        else:
            import pandas as pd
            from tabstar.tabstar_model import TabSTARClassifier
            a, b = next(StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=fit_seed)
                        .split(fit_idx, y[fit_idx], groups.iloc[fit_idx]))
            tr_idx, es_idx = fit_idx[a], fit_idx[b]
            # TabSTAR's type detection rejects the pandas 3 string dtype; give it plain objects.
            x, test_frame = x.copy(), test_frame.copy()
            for frame in (x, test_frame):
                frame[categorical] = frame[categorical].astype(object)
            est = TabSTARClassifier(device=device, random_state=fit_seed, verbose=False,
                                    keep_model=True,
                                    output_dir=str(ROOT / f"models/tabstar_tmp/{name}_f{fold}"))
            est.fit(x.iloc[tr_idx].reset_index(drop=True), pd.Series(y[tr_idx]),
                    x.iloc[es_idx].reset_index(drop=True), pd.Series(y[es_idx]))
            probability = np.asarray(est.predict_proba(x.iloc[valid_idx].reset_index(drop=True)))[:, 1]
            test_part = np.asarray(est.predict_proba(test_frame.reset_index(drop=True)))[:, 1]
        if limit is not None:
            print(f"SMOKE {model} fold {fold}: acc="
                  f"{accuracy_score(y[valid_idx], probability >= 0.5):.4f} "
                  f"({time.monotonic() - fold_start:.0f}s)", flush=True)
            return None
        oof[valid_idx] = probability
        test_probability += test_part / 5
        fold_results.append({"fold": fold,
                             "accuracy": float(accuracy_score(y[valid_idx], probability >= 0.5)),
                             "seconds": time.monotonic() - fold_start})
        print(f"{name} fold {fold}: {fold_results[-1]['accuracy']:.6f} "
              f"({fold_results[-1]['seconds']:.0f}s)", flush=True)
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
    model, experiment, *rest = sys.argv[1:]
    if experiment == "smoke":
        run_side("smoke", 42, model, device=rest[0] if rest else "cpu", limit=1500)
    else:
        for s in rest:
            run_side(experiment if int(s) == 42 else f"{experiment}_seed{s}", int(s), model)
