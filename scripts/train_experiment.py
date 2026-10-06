from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--experiment-id")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[key] = str(args.threads)
    from spaceship_titanic.experiments import run_experiment

    config = json.loads(args.config.read_text(encoding="utf-8"))
    config.setdefault("experiment_id", "baseline_replay")
    config.setdefault("feature_variant", "baseline")
    if args.experiment_id:
        config["experiment_id"] = args.experiment_id
    if config["model"] == "CatBoostClassifier":
        config["model"] = "catboost"
        config["model_params"].update(loss_function="Logloss", eval_metric="Accuracy")
    run_experiment(config, args.threads)


if __name__ == "__main__":
    main()
