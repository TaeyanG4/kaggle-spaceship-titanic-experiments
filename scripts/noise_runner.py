"""H-A-13: champion recipe with fold-local label-noise down-weighting (confident learning).

Inside each outer fit fold F only: a 5-fold inner StratifiedGroupKFold OOF (champion params at a
fixed 400 iterations) gives each fit row a probability p_in. A row "contradicts" its label when
p_in sits on the wrong side of 0.5 with |p_in - 0.5| >= margin. Predeclared arms:
- `noise_drop35`: weight 0 if the contradiction margin >= 0.35;
- `noise_half25`: weight 0.5 if the contradiction margin >= 0.25.
Iterations are then chosen by the champion rule (4-fold inner SGKF Logloss median, weighted) and
the final model is fit on F with those weights. The outer validation fold is never touched.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARAMS = {"iterations": 800, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5.0,
          "random_strength": 0.5, "loss_function": "Logloss", "eval_metric": "Accuracy"}
ARMS = {"noise_drop35": (0.35, 0.0), "noise_half25": (0.25, 0.5)}
INNER_ITERATIONS = 400


def run_noise(name: str, seed: int, arm: str, threads: int) -> dict:
    import numpy as np
    from catboost import CatBoostClassifier, Pool
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

    margin, low_weight = ARMS[arm]
    config = {"experiment_id": name, "arm": arm, "margin": margin, "low_weight": low_weight,
              "inner_iterations": INNER_ITERATIONS, "random_state": seed, "n_splits": 5,
              "stopping": "innercv_logloss", "model_params": PARAMS}
    code = {p: file_hash(ROOT / p) for p in [
        "scripts/noise_runner.py", "src/spaceship_titanic/experiments.py",
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
    oof, test_probability, fold_results = np.zeros(len(x)), np.zeros(len(tx)), []

    def model(**overrides):
        return CatBoostClassifier(**(PARAMS | overrides), thread_count=threads, verbose=False,
                                  allow_writing_files=False)

    for fold in range(1, 6):
        fold_start = time.monotonic()
        fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        fit_seed = seed + fold
        p_in = np.zeros(len(fit_idx))
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=fit_seed + 500)
        for a, b in inner.split(fit_idx, y[fit_idx], groups.iloc[fit_idx]):
            m = model(iterations=INNER_ITERATIONS, random_seed=fit_seed)
            m.fit(x.iloc[fit_idx[a]], y[fit_idx[a]], cat_features=categorical)
            p_in[b] = m.predict_proba(x.iloc[fit_idx[b]], thread_count=threads)[:, 1]
        wrong_side = (p_in >= 0.5) != y[fit_idx].astype(bool)
        contradict = wrong_side & (np.abs(p_in - 0.5) >= margin)
        weights = np.where(contradict, low_weight, 1.0)
        keep = weights > 0
        rows, w = fit_idx[keep], weights[keep]
        bests = []
        probe_split = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=fit_seed)
        for a, b in probe_split.split(rows, y[rows], groups.iloc[rows]):
            probe = model(random_seed=fit_seed, eval_metric="Logloss")
            probe.fit(Pool(x.iloc[rows[a]], y[rows[a]], cat_features=categorical, weight=w[a]),
                      eval_set=Pool(x.iloc[rows[b]], y[rows[b]], cat_features=categorical,
                                    weight=w[b]),
                      early_stopping_rounds=100, verbose=False)
            bests.append(int(probe.get_best_iteration()) + 1)
        best = int(np.median(bests))
        final = model(iterations=best, random_seed=fit_seed)
        final.fit(Pool(x.iloc[rows], y[rows], cat_features=categorical, weight=w))
        probability = final.predict_proba(x.iloc[valid_idx], thread_count=threads)[:, 1]
        oof[valid_idx] = probability
        test_probability += final.predict_proba(tx, thread_count=threads)[:, 1] / 5
        fold_results.append({"fold": fold, "best_iteration": best,
                             "down_weighted": int(contradict.sum()),
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
