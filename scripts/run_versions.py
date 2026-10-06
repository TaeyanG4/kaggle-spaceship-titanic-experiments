"""Sequential v1/v2 experiments with bounded CPU, resume, and completion status."""

from __future__ import annotations

import argparse
import copy
import ctypes
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def set_low_priority() -> str:
    if os.name != "nt":
        os.nice(5)
        return "nice +5"
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
        raise ctypes.WinError(ctypes.get_last_error())
    return "BelowNormal"


class Tee:
    def __init__(self, original, path):
        self.original = original
        self.file = path.open("a", encoding="utf-8", buffering=1)

    def write(self, text):
        self.original.write(text)
        self.file.write(text)

    def flush(self):
        self.original.flush()
        self.file.flush()


def configurations():
    baseline = json.loads((ROOT / "configs/baseline.json").read_text(encoding="utf-8"))
    params = baseline["model_params"] | {"loss_function": "Logloss", "eval_metric": "Accuracy"}
    common = {"random_state": 42, "n_splits": 5, "threshold": 0.5,
              "model": "catboost", "model_params": params}
    runs = []
    for experiment_id, variant in [
        ("v1_name_fix", "v1_name"), ("v1_spend", "v1_spend"),
        ("v1", "v1"), ("v1_rules", "v1_rules"),
        ("v2_spending", "v2_spending"), ("v2_group", "v2_group"), ("v2", "v2"),
    ]:
        runs.append(copy.deepcopy(common) | {
            "experiment_id": experiment_id, "feature_variant": variant
        })
    logloss = copy.deepcopy(runs[-1])
    logloss["experiment_id"] = "v2_logloss"
    logloss["model_params"]["eval_metric"] = "Logloss"
    runs.append(logloss)
    runs.append(common | {"experiment_id": "v2_lightgbm", "feature_variant": "v2",
                         "model": "lightgbm", "model_params": {
        "n_estimators": 1600, "learning_rate": 0.03, "num_leaves": 15,
        "max_depth": 6, "min_child_samples": 40, "reg_lambda": 5.0,
        "subsample": 0.85, "subsample_freq": 1, "colsample_bytree": 0.9,
        "early_stopping_rounds": 100,
    }})
    runs.append(common | {"experiment_id": "v2_xgboost", "feature_variant": "v2",
                         "model": "xgboost", "model_params": {
        "n_estimators": 1200, "learning_rate": 0.04, "max_depth": 4,
        "min_child_weight": 5, "reg_lambda": 5.0, "subsample": 0.85,
        "colsample_bytree": 0.9, "tree_method": "hist", "eval_metric": "logloss",
        "early_stopping_rounds": 100,
    }})
    baseline_config = copy.deepcopy(common) | {
        "experiment_id": "baseline_reference", "feature_variant": "baseline"
    }
    return runs, baseline_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"]:
        os.environ[key] = str(args.threads)
    priority = set_low_priority()
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, log_loss

    from spaceship_titanic.experiments import (
        diagnostics,
        file_hash,
        frozen_folds,
        load_data,
        paired_group_bootstrap,
        run_experiment,
        save_predictions,
        timestamp,
        write_json,
    )

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    sys.stdout = Tee(sys.stdout, reports / "v1_v2_training.log")
    status_path = reports / "v1_v2_status.json"
    status = {"status": "running", "started_at": timestamp(), "pid": os.getpid(),
              "threads": args.threads, "priority": priority, "completed_experiments": [],
              "automation_id": "spaceship-titanic-v1-v2"}

    def update(experiment_id, fold, result):
        status.update(current_experiment=experiment_id, fold=fold,
                      last_fold=result, updated_at=timestamp())
        write_json(status_path, status)

    def blend(experiment_id, components, train, test, sample, folds):
        oof_tables = [pd.read_csv(ROOT / r["artifacts"]["oof"]) for r in components]
        test_tables = [pd.read_csv(ROOT / r["artifacts"]["test_probability"]) for r in components]
        for table in oof_tables:
            assert table.PassengerId.equals(train.PassengerId)
            assert np.array_equal(table.fold, folds)
        for table in test_tables:
            assert table.PassengerId.equals(test.PassengerId)
        probability = np.mean([t.probability.to_numpy() for t in oof_tables], axis=0)
        test_probability = np.mean([t.probability.to_numpy() for t in test_tables], axis=0)
        result = {"experiment_id": experiment_id,
                  "accuracy": float(accuracy_score(train.Transported, probability >= 0.5)),
                  "log_loss": float(log_loss(train.Transported, probability)),
                  "components": [r["experiment_id"] for r in components],
                  "weights": [1 / len(components)] * len(components),
                  "completed_at": timestamp(), "segments": diagnostics(train, probability),
                  "fold_std": float(np.std([
                      accuracy_score(train.Transported[folds == f], probability[folds == f] >= 0.5)
                      for f in range(1, 6)
                  ])),
                  "artifacts": save_predictions(
                      experiment_id, train, test, sample, folds, probability, test_probability
                  )}
        write_json(reports / f"{experiment_id}_metrics.json", result)
        print(f"BLEND {experiment_id}: {result['accuracy']:.6f}", flush=True)
        return result

    write_json(status_path, status)
    try:
        train, test, sample = load_data()
        protected = {
            str(p.relative_to(ROOT)): file_hash(p) for p in [
                ROOT / "data/raw/train.csv", ROOT / "data/raw/test.csv",
                ROOT / "data/raw/sample_submission.csv",
                ROOT / "outputs/oof/baseline_catboost_oof.csv",
                ROOT / "outputs/submissions/baseline_catboost.csv",
            ]
        }
        reference = pd.read_csv(ROOT / "outputs/oof/baseline_catboost_oof.csv")
        assert reference.PassengerId.equals(train.PassengerId)
        baseline_score = float(accuracy_score(train.Transported, reference.prediction))
        runs, baseline_config = configurations()
        write_json(reports / "v1_v2_manifest.json", {"runs": runs, "threads": args.threads,
                   "confirmation_seeds": [123, 2026], "threshold": 0.5,
                   "predeclared_blend": ["v2_logloss", "v2_lightgbm", "v2_xgboost"]})
        results = []
        for config in runs:
            status["stage"] = "v1" if config["experiment_id"].startswith("v1") else "v2"
            update(config["experiment_id"], 0, {})
            result = run_experiment(config, args.threads, update)
            results.append(result)
            status["completed_experiments"].append(
                {"id": result["experiment_id"], "accuracy": result["accuracy"]}
            )
            write_json(status_path, status)
            pd.DataFrame([{"id": r["experiment_id"], "accuracy": r["accuracy"],
                           "log_loss": r["log_loss"], "fold_std": r["fold_std"]}
                          for r in results]).to_csv(reports / "v1_v2_scores.csv", index=False)
        component_ids = ["v2_logloss", "v2_lightgbm", "v2_xgboost"]
        components = [next(r for r in results if r["experiment_id"] == name)
                      for name in component_ids]
        ensemble = blend("v2_blend", components, train, test, sample, frozen_folds(train, 42))
        ensemble_oof = pd.read_csv(ROOT / ensemble["artifacts"]["oof"]).probability.to_numpy()
        ensemble["baseline_comparison"] = paired_group_bootstrap(
            train, reference.probability.to_numpy(), ensemble_oof
        )
        write_json(reports / "v2_blend_metrics.json", ensemble)
        results.append(ensemble)
        v1 = max([r for r in results if r["experiment_id"].startswith("v1")],
                 key=lambda r: (r["accuracy"], -r["log_loss"]))
        v2 = max([r for r in results if r["experiment_id"].startswith("v2")],
                 key=lambda r: (r["accuracy"], -r["log_loss"]))
        candidate = max([v1, v2], key=lambda r: (r["accuracy"], -r["log_loss"]))
        confirmations = []
        if candidate["accuracy"] - baseline_score >= 0.002:
            status["stage"] = "confirmation"
            configs_by_id = {r["experiment_id"]: r for r in runs}
            for seed in [123, 2026]:
                matched_config = copy.deepcopy(baseline_config)
                matched_config.update(experiment_id=f"baseline_seed{seed}", random_state=seed)
                matched = run_experiment(matched_config, args.threads, update)
                if candidate["experiment_id"] == "v2_blend":
                    seed_components = []
                    for name in component_ids:
                        config = copy.deepcopy(configs_by_id[name])
                        config.update(experiment_id=f"{name}_seed{seed}", random_state=seed)
                        seed_components.append(run_experiment(config, args.threads, update))
                    confirmed = blend(f"v2_blend_seed{seed}", seed_components,
                                      train, test, sample, frozen_folds(train, seed))
                else:
                    config = copy.deepcopy(configs_by_id[candidate["experiment_id"]])
                    config.update(experiment_id=f"{candidate['experiment_id']}_seed{seed}",
                                  random_state=seed)
                    confirmed = run_experiment(config, args.threads, update)
                confirmations.append({"seed": seed, "baseline": matched["accuracy"],
                                      "challenger": confirmed["accuracy"],
                                      "delta": confirmed["accuracy"] - matched["accuracy"],
                                      "baseline_id": matched["experiment_id"],
                                      "challenger_id": confirmed["experiment_id"]})
        promoted = bool(confirmations) and all(c["delta"] > 0 for c in confirmations)
        for label, result in [("v1", v1), ("v2", v2)]:
            shutil.copyfile(ROOT / result["artifacts"]["submission"],
                            ROOT / f"outputs/submissions/{label}_best.csv")
        selected = candidate["experiment_id"] if promoted else "baseline-001"
        selected_source = (ROOT / candidate["artifacts"]["submission"] if promoted else
                           ROOT / "outputs/submissions/baseline_catboost.csv")
        shutil.copyfile(selected_source, ROOT / "outputs/submissions/selected_v1_v2.csv")
        assert protected == {name: file_hash(ROOT / name) for name in protected}
        summary = {"completed_at": timestamp(), "baseline_accuracy": baseline_score,
                   "v1": {"best_id": v1["experiment_id"], "accuracy": v1["accuracy"],
                          "delta": v1["accuracy"] - baseline_score,
                          "submission": "outputs/submissions/v1_best.csv"},
                   "v2": {"best_id": v2["experiment_id"], "accuracy": v2["accuracy"],
                          "delta": v2["accuracy"] - baseline_score,
                          "submission": "outputs/submissions/v2_best.csv"},
                   "candidate": candidate["experiment_id"], "promoted": promoted,
                   "selected": selected, "confirmation": confirmations,
                   "selection_rule": "primary +0.002, then positive delta on both seeds 123 and 2026",
                   "selected_submission": "outputs/submissions/selected_v1_v2.csv",
                   "primary_results": [{"id": r["experiment_id"], "accuracy": r["accuracy"],
                       "log_loss": r["log_loss"], "fold_std": r["fold_std"],
                       "delta": r["accuracy"] - baseline_score} for r in results],
                   "protected_hashes": protected, "threads": args.threads, "priority": priority,
                   "remote_actions": "none", "threshold": 0.5,
                   "note": "Primary best scores are selection estimates; confirmation is secondary evidence."}
        write_json(reports / "v1_v2_summary.json", summary)
        pd.DataFrame(summary["primary_results"]).to_csv(reports / "v1_v2_scores.csv", index=False)
        status.update(status="results_ready", stage="finalize_docs", updated_at=timestamp(),
                      selected=selected, promoted=promoted, summary="reports/v1_v2_summary.json")
        write_json(status_path, status)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    except BaseException as error:
        status.update(status="failed", updated_at=timestamp(), error=str(error))
        write_json(status_path, status)
        (reports / "v1_v2_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
