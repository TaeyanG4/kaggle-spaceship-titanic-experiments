"""H-D-02: small predeclared CatBoost depth/L2/learning-rate grid on frozen SGKF folds.

The grid, selection rule, and confirmation seeds are written to the manifest before any run.
Accuracy stopping stays the control. Earth/non-Earth errors are reported as diagnostics (D-D-021).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from run_versions import Tee, set_low_priority

ROOT = Path(__file__).resolve().parents[1]
REPORT = "hm02"
BASE_PARAMS = {
    "iterations": 800, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5.0,
    "random_strength": 0.5, "early_stopping_rounds": 100,
    "loss_function": "Logloss", "eval_metric": "Accuracy",
}
SETTINGS = {
    "d4_l3": {"depth": 4, "l2_leaf_reg": 3.0},
    "d4_l10": {"depth": 4, "l2_leaf_reg": 10.0},
    "d6_l1": {"depth": 6, "l2_leaf_reg": 1.0},
    "d6_l10": {"depth": 6, "l2_leaf_reg": 10.0},
    "d8_l5": {"depth": 8, "l2_leaf_reg": 5.0},
    "d6_l5_lr03": {"learning_rate": 0.03, "iterations": 1500},
}
VARIANTS = {"base": "baseline", "name": "v1_name"}
SCREENING_GAIN = 0.002
CONFIRMATION_SEEDS = [123, 2026]


def make_config(experiment_id: str, variant: str, overrides: dict, seed: int = 42) -> dict:
    return {
        "experiment_id": experiment_id, "feature_variant": variant, "random_state": seed,
        "n_splits": 5, "threshold": 0.5, "model": "catboost",
        "model_params": BASE_PARAMS | overrides,
    }


def grid() -> tuple[dict, list[dict]]:
    control = make_config("hm02_base_control", "baseline", {})
    runs = [
        make_config(f"hm02_{short}_{setting}", variant, overrides)
        for short, variant in VARIANTS.items()
        for setting, overrides in SETTINGS.items()
    ]
    return control, runs


def worker_init(threads: int) -> None:
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"]:
        os.environ[key] = str(threads)
    set_low_priority()


def run_one(config: dict, threads: int) -> dict:
    from spaceship_titanic.experiments import run_experiment

    return run_experiment(config, threads)


def earth_split(train, probability) -> dict:
    import numpy as np

    earth = train.HomePlanet.eq("Earth").to_numpy()
    wrong = (np.asarray(probability) >= 0.5) != train.Transported.astype(bool).to_numpy()
    return {"earth_errors": int(wrong[earth].sum()), "other_errors": int(wrong[~earth].sum())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=2, help="threads per worker")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1 or args.workers < 1:
        parser.error("--threads and --workers must be positive")
    worker_init(args.threads)
    import pandas as pd

    from spaceship_titanic.experiments import (
        file_hash,
        load_data,
        paired_group_bootstrap,
        timestamp,
        write_json,
    )

    reports = ROOT / "reports"
    sys.stdout = Tee(sys.stdout, reports / f"{REPORT}_training.log")
    status_path = reports / f"{REPORT}_status.json"
    status = {"status": "running", "started_at": timestamp(), "pid": os.getpid(),
              "threads_per_worker": args.threads, "workers": args.workers,
              "priority": "BelowNormal", "completed": []}
    write_json(status_path, status)
    try:
        train, _, _ = load_data()
        protected_paths = [
            "data/raw/train.csv", "data/raw/test.csv", "data/raw/sample_submission.csv",
            "outputs/oof/baseline_catboost_oof.csv", "outputs/submissions/baseline_catboost.csv",
        ]
        protected = {name: file_hash(ROOT / name) for name in protected_paths}
        champion = pd.read_csv(ROOT / "outputs/oof/baseline_catboost_oof.csv")
        assert champion.PassengerId.equals(train.PassengerId)
        champion_prob = champion.probability.to_numpy()
        champion_acc = float((champion.prediction == train.Transported).mean())

        control, runs = grid()
        write_json(reports / f"{REPORT}_manifest.json", {
            "created_at": timestamp(), "plan_item": "H-D-02", "control": control, "runs": runs,
            "name_control_reference": "v1_name_fix (existing, identical params)",
            "selection_rule": (
                "candidate = highest primary accuracy among grid runs (ties: lower log loss); "
                f"screen at >= +{SCREENING_GAIN} vs champion baseline-001 OOF; then matched "
                f"control vs candidate on seeds {CONFIRMATION_SEEDS}; promote only if both "
                "seed deltas are positive"
            ),
            "threads_per_worker": args.threads, "workers": args.workers,
        })

        results = {}
        with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init,
                                 initargs=(args.threads,)) as pool:
            futures = {c["experiment_id"]: pool.submit(run_one, c, args.threads)
                       for c in [control, *runs]}
            for experiment_id, future in futures.items():
                results[experiment_id] = future.result()
                status["completed"].append(experiment_id)
                status["updated_at"] = timestamp()
                write_json(status_path, status)
                print(f"COLLECTED {experiment_id}: {results[experiment_id]['accuracy']:.6f}",
                      flush=True)
        name_control = json.loads(
            (reports / "v1_name_fix_metrics.json").read_text(encoding="utf-8"))

        rows = []
        for experiment_id, result in results.items():
            oof = pd.read_csv(ROOT / result["artifacts"]["oof"])
            assert oof.PassengerId.equals(train.PassengerId)
            variant = result["config"]["feature_variant"]
            reference = (name_control if variant == "v1_name"
                         else results["hm02_base_control"])
            params = result["config"]["model_params"]
            rows.append({
                "id": experiment_id, "variant": variant, "depth": params["depth"],
                "l2_leaf_reg": params["l2_leaf_reg"], "learning_rate": params["learning_rate"],
                "accuracy": result["accuracy"], "log_loss": result["log_loss"],
                "fold_std": result["fold_std"],
                "delta_vs_champion": result["accuracy"] - champion_acc,
                "delta_vs_matched_control": result["accuracy"] - reference["accuracy"],
                "mean_best_iteration": sum(f["best_iteration"] for f in result["folds"]) / 5,
                "min_best_iteration": min(f["best_iteration"] for f in result["folds"]),
                **earth_split(train, oof.probability),
                "champion_bootstrap_ci95": paired_group_bootstrap(
                    train, champion_prob, oof.probability.to_numpy())["ci95"],
            })
        table = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
        table.to_csv(reports / f"{REPORT}_scores.csv", index=False)
        print(table.drop(columns="champion_bootstrap_ci95").to_string(index=False), flush=True)

        grid_rows = table[table.id != "hm02_base_control"].sort_values(
            ["accuracy", "log_loss"], ascending=[False, True])
        candidate = grid_rows.iloc[0].to_dict()
        confirmations = []
        if candidate["delta_vs_champion"] >= SCREENING_GAIN:
            status["stage"] = "confirmation"
            write_json(status_path, status)
            candidate_config = next(c for c in runs if c["experiment_id"] == candidate["id"])
            jobs = []
            for seed in CONFIRMATION_SEEDS:
                matched = copy.deepcopy(control) | {
                    "experiment_id": f"hm02_base_control_seed{seed}", "random_state": seed}
                challenger = copy.deepcopy(candidate_config) | {
                    "experiment_id": f"{candidate['id']}_seed{seed}", "random_state": seed}
                jobs.append((seed, matched, challenger))
            with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init,
                                     initargs=(args.threads,)) as pool:
                pending = [(seed, pool.submit(run_one, m, args.threads),
                            pool.submit(run_one, c, args.threads)) for seed, m, c in jobs]
                for seed, matched_future, challenger_future in pending:
                    matched, challenger = matched_future.result(), challenger_future.result()
                    confirmations.append({
                        "seed": seed, "control": matched["accuracy"],
                        "challenger": challenger["accuracy"],
                        "delta": challenger["accuracy"] - matched["accuracy"],
                    })
        promoted = bool(confirmations) and all(c["delta"] > 0 for c in confirmations)
        assert protected == {name: file_hash(ROOT / name) for name in protected}
        summary = {
            "completed_at": timestamp(), "plan_item": "H-D-02",
            "champion_accuracy": champion_acc,
            "control_accuracy": results["hm02_base_control"]["accuracy"],
            "name_control_accuracy": name_control["accuracy"],
            "candidate": candidate, "screening_gain": SCREENING_GAIN,
            "screened": candidate["delta_vs_champion"] >= SCREENING_GAIN,
            "confirmation": confirmations, "promoted": promoted,
            "results": table.to_dict(orient="records"),
            "protected_hashes": protected, "remote_actions": "none",
            "note": "Max over 12 grid runs on the same folds is optimistic; confirmation decides.",
        }
        write_json(reports / f"{REPORT}_summary.json", summary)
        status.update(status="results_ready", updated_at=timestamp(), promoted=promoted)
        write_json(status_path, status)
        print(json.dumps({k: summary[k] for k in
                          ["champion_accuracy", "control_accuracy", "candidate", "screened",
                           "confirmation", "promoted"]}, indent=2), flush=True)
    except BaseException as error:
        status.update(status="failed", updated_at=timestamp(), error=str(error))
        write_json(status_path, status)
        (reports / f"{REPORT}_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
