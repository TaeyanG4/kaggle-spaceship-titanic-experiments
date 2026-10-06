"""H-Main-09: CatBoost early stopping that never looks at the scored validation fold.

Modes (all predeclared, baseline config and features):
- inner: stop on a stratified group holdout (1/8 of fit-fold groups); predict with that model.
- inner_refit: as inner, then refit on the whole fit fold at the found best iteration.
- fixed300: 300 iterations, no early stopping.
The control is the existing `hm02_base_control` family (stopping on the scored fold).
Screening and confirmation follow `spaceship_titanic.screening` (agents.md decision rule).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from alert import improvement_alert
from run_hm02_grid import BASE_PARAMS, worker_init
from run_versions import Tee

ROOT = Path(__file__).resolve().parents[1]
REPORT = "hm09"
MODES = ("inner", "inner_refit", "fixed300")
CONTROL_ID = "hm02_base_control"


def experiment_id(mode: str, seed: int) -> str:
    return f"hm09_{mode}" + ("" if seed == 42 else f"_seed{seed}")


def run_mode(mode: str, seed: int, threads: int) -> dict:
    import numpy as np
    from catboost import CatBoostClassifier
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

    name = experiment_id(mode, seed)
    config = {"experiment_id": name, "feature_variant": "baseline", "random_state": seed,
              "n_splits": 5, "threshold": 0.5, "model": "catboost", "stopping": mode,
              "inner_holdout": "StratifiedGroupKFold(8) first split", "model_params": BASE_PARAMS}
    code = {p: file_hash(ROOT / p) for p in [
        "scripts/run_hm09_stopping.py", "src/spaceship_titanic/experiments.py",
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
    y = train.Transported.astype(int)
    groups = train.PassengerId.str.split("_").str[0]
    params = {k: v for k, v in BASE_PARAMS.items() if k != "early_stopping_rounds"}
    oof, test_probability, fold_results = np.zeros(len(train)), np.zeros(len(test)), []

    def model(**overrides):
        return CatBoostClassifier(**(params | overrides), thread_count=threads,
                                  verbose=False, allow_writing_files=False)

    for fold in range(1, 6):
        fold_start = time.monotonic()
        fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        fit_seed = seed + fold
        if mode == "fixed300":
            final = model(iterations=300, random_seed=fit_seed)
            final.fit(x.iloc[fit_idx], y.iloc[fit_idx], cat_features=categorical)
            best = 300
        else:
            splitter = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=fit_seed)
            inner_fit, inner_stop = next(splitter.split(
                x.iloc[fit_idx], y.iloc[fit_idx], groups.iloc[fit_idx]))
            inner_fit, inner_stop = fit_idx[inner_fit], fit_idx[inner_stop]
            assert not set(groups.iloc[inner_fit]) & set(groups.iloc[inner_stop])
            assert not set(inner_stop) & set(valid_idx)
            final = model(random_seed=fit_seed)
            final.fit(x.iloc[inner_fit], y.iloc[inner_fit], cat_features=categorical,
                      eval_set=(x.iloc[inner_stop], y.iloc[inner_stop]),
                      early_stopping_rounds=BASE_PARAMS["early_stopping_rounds"], verbose=False)
            best = int(final.get_best_iteration()) + 1
            if mode == "inner_refit":
                final = model(iterations=best, random_seed=fit_seed)
                final.fit(x.iloc[fit_idx], y.iloc[fit_idx], cat_features=categorical)
        probability = final.predict_proba(x.iloc[valid_idx], thread_count=threads)[:, 1]
        oof[valid_idx] = probability
        test_probability += final.predict_proba(tx, thread_count=threads)[:, 1] / 5
        score = float(accuracy_score(y.iloc[valid_idx], probability >= 0.5))
        fold_results.append({"fold": fold, "accuracy": score, "best_iteration": best,
                             "seconds": time.monotonic() - fold_start})
        print(f"{name} fold {fold}/5: {score:.6f}, best={best}", flush=True)
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


def run_job(job: tuple[str, int], threads: int) -> dict:
    mode, seed = job
    if mode == "control":
        from run_hm02_grid import make_config, run_one

        from spaceship_titanic.screening import seeded

        return run_one(seeded(make_config(CONTROL_ID, "baseline", {}), seed), threads)
    return run_mode(mode, seed, threads)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    worker_init(args.threads)
    import numpy as np

    from spaceship_titanic.experiments import file_hash, timestamp, write_json
    from spaceship_titanic.screening import CONFIRM_SEEDS, SCREEN_SEEDS, confirm, screen

    reports = ROOT / "reports"
    sys.stdout = Tee(sys.stdout, reports / f"{REPORT}_training.log")
    status_path = reports / f"{REPORT}_status.json"
    status = {"status": "running", "started_at": timestamp(), "pid": os.getpid()}
    write_json(status_path, status)
    try:
        protected = {n: file_hash(ROOT / n) for n in [
            "data/raw/train.csv", "data/raw/test.csv",
            "outputs/oof/baseline_catboost_oof.csv", "outputs/submissions/baseline_catboost.csv"]}
        write_json(reports / f"{REPORT}_manifest.json", {
            "created_at": timestamp(), "plan_item": "H-Main-09", "modes": list(MODES),
            "control": f"{CONTROL_ID} (stops on the scored fold)",
            "screen_seeds": list(SCREEN_SEEDS), "confirm_seeds": list(CONFIRM_SEEDS),
            "rule": "agents.md multi-seed screen; best passing mode confirmed on 7/99",
            "also_reported": "optimism = control accuracy minus each honest mode, per seed",
        })

        def run_all(jobs):
            with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init,
                                     initargs=(args.threads,)) as pool:
                futures = {job: pool.submit(run_job, job, args.threads) for job in jobs}
                return {job: f.result()["accuracy"] for job, f in futures.items()}

        acc = run_all([(m, s) for s in SCREEN_SEEDS for m in ("control", *MODES)])
        control = {s: acc[("control", s)] for s in SCREEN_SEEDS}
        rows = []
        for mode in MODES:
            outcome = screen(control, {s: acc[(mode, s)] for s in SCREEN_SEEDS})
            rows.append({"mode": mode, **outcome,
                         "accuracy": {s: acc[(mode, s)] for s in SCREEN_SEEDS},
                         "mean_accuracy": float(np.mean([acc[(mode, s)] for s in SCREEN_SEEDS]))})
            print(f"{mode:12s} mean_delta={outcome['mean_delta']:+.6f} "
                  f"positive={outcome['positive_seeds']}/3 passes={outcome['passes']}", flush=True)
        rows.sort(key=lambda r: -r["mean_delta"])
        confirmation = None
        passing = [r for r in rows if r["passes"]]
        if passing:
            mode = passing[0]["mode"]
            cacc = run_all([(m, s) for s in CONFIRM_SEEDS for m in ("control", mode)])
            confirmation = {"mode": mode, **confirm(
                {s: cacc[("control", s)] for s in CONFIRM_SEEDS},
                {s: cacc[(mode, s)] for s in CONFIRM_SEEDS})}
            print(f"CONFIRM {confirmation}", flush=True)
        promoted = bool(confirmation and confirmation["passes"])
        if promoted:
            improvement_alert(f"H-Main-09 promoted stopping mode {confirmation['mode']}")
        assert protected == {n: file_hash(ROOT / n) for n in protected}
        write_json(reports / f"{REPORT}_summary.json", {
            "completed_at": timestamp(), "plan_item": "H-Main-09", "control_accuracy": control,
            "control_mean_accuracy": float(np.mean(list(control.values()))),
            "modes": rows, "confirmation": confirmation, "promoted": promoted,
            "protected_hashes": protected, "remote_actions": "none"})
        status.update(status="results_ready", updated_at=timestamp(), promoted=promoted)
        write_json(status_path, status)
        print(json.dumps({"promoted": promoted, "confirmation": confirmation}), flush=True)
    except BaseException as error:
        status.update(status="failed", updated_at=timestamp(), error=str(error))
        write_json(status_path, status)
        (reports / f"{REPORT}_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
