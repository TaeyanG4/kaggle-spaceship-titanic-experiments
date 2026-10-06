"""H-D-08: multi-seed SGKF screening of the strongest existing candidates.

Seed-42 runs and existing seed 123/2026 runs are reused by fingerprint. The manifest is written
before any training. Rule: `spaceship_titanic.screening` (agents.md experiment decision rule).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from alert import improvement_alert
from run_hm02_grid import make_config, run_one, worker_init
from run_versions import Tee

ROOT = Path(__file__).resolve().parents[1]
REPORT = "hm08"
CONTROL = make_config("hm02_base_control", "baseline", {})
CANDIDATES = [
    make_config("hm02_name_d8_l5", "v1_name", {"depth": 8, "l2_leaf_reg": 5.0}),
    make_config("hm02_name_d6_l1", "v1_name", {"depth": 6, "l2_leaf_reg": 1.0}),
    make_config("hm02_base_d8_l5", "baseline", {"depth": 8, "l2_leaf_reg": 5.0}),
    make_config("hm02_base_d6_l1", "baseline", {"depth": 6, "l2_leaf_reg": 1.0}),
    make_config("v1_name_fix", "v1_name", {}),
]


def run_all(configs: list[dict], workers: int, threads: int) -> dict[str, dict]:
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init,
                             initargs=(threads,)) as pool:
        futures = {c["experiment_id"]: pool.submit(run_one, c, threads) for c in configs}
        return {name: future.result() for name, future in futures.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=2, help="threads per worker")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1 or args.workers < 1:
        parser.error("--threads and --workers must be positive")
    worker_init(args.threads)
    import numpy as np
    import pandas as pd

    from spaceship_titanic.experiments import file_hash, load_data, timestamp, write_json
    from spaceship_titanic.screening import (
        CONFIRM_SEEDS,
        SCREEN_GAIN,
        SCREEN_SEEDS,
        confirm,
        screen,
        seeded,
    )

    reports = ROOT / "reports"
    sys.stdout = Tee(sys.stdout, reports / f"{REPORT}_training.log")
    status_path = reports / f"{REPORT}_status.json"
    status = {"status": "running", "started_at": timestamp(), "pid": os.getpid(),
              "threads_per_worker": args.threads, "workers": args.workers}
    write_json(status_path, status)
    try:
        train, _, _ = load_data()
        earth = train.HomePlanet.eq("Earth").to_numpy()
        target = train.Transported.astype(bool).to_numpy()
        protected = {name: file_hash(ROOT / name) for name in [
            "data/raw/train.csv", "data/raw/test.csv", "data/raw/sample_submission.csv",
            "outputs/oof/baseline_catboost_oof.csv", "outputs/submissions/baseline_catboost.csv",
        ]}
        write_json(reports / f"{REPORT}_manifest.json", {
            "created_at": timestamp(), "plan_item": "H-D-08", "control": CONTROL,
            "candidates": CANDIDATES, "screen_seeds": list(SCREEN_SEEDS),
            "confirm_seeds": list(CONFIRM_SEEDS),
            "rule": (f"screen: mean paired delta >= +{SCREEN_GAIN} and at least 2 of 3 seeds "
                     "positive vs matched control; only the highest passing mean goes to "
                     "confirmation on fresh seeds 7/99; promote only if both deltas > 0"),
            "workers": args.workers, "threads_per_worker": args.threads,
        })

        def errors(result: dict) -> dict:
            oof = pd.read_csv(ROOT / result["artifacts"]["oof"])
            assert oof.PassengerId.equals(train.PassengerId)
            wrong = (oof.probability.to_numpy() >= 0.5) != target
            return {"earth": int(wrong[earth].sum()), "other": int(wrong[~earth].sum())}

        configs = [seeded(c, s) for s in SCREEN_SEEDS for c in [CONTROL, *CANDIDATES]]
        results = run_all(configs, args.workers, args.threads)

        def accuracy(config: dict, seeds) -> dict[int, float]:
            return {s: results[seeded(config, s)["experiment_id"]]["accuracy"] for s in seeds}

        control_acc = accuracy(CONTROL, SCREEN_SEEDS)
        rows = []
        for candidate in CANDIDATES:
            outcome = screen(control_acc, accuracy(candidate, SCREEN_SEEDS))
            per_seed_errors = {s: errors(results[seeded(candidate, s)["experiment_id"]])
                               for s in SCREEN_SEEDS}
            rows.append({"id": candidate["experiment_id"], **outcome,
                         "accuracy": accuracy(candidate, SCREEN_SEEDS),
                         "mean_accuracy": float(np.mean(list(accuracy(candidate, SCREEN_SEEDS)
                                                             .values()))),
                         "errors": per_seed_errors})
        rows.sort(key=lambda r: -r["mean_delta"])
        control_errors = {s: errors(results[seeded(CONTROL, s)["experiment_id"]])
                          for s in SCREEN_SEEDS}
        for row in rows:
            print(f"{row['id']:18s} mean_delta={row['mean_delta']:+.6f} "
                  f"positive={row['positive_seeds']}/3 passes={row['passes']} "
                  f"deltas={ {s: round(d, 6) for s, d in row['deltas'].items()} }", flush=True)

        passing = [row for row in rows if row["passes"]]
        confirmation = None
        if passing:
            best = next(c for c in CANDIDATES if c["experiment_id"] == passing[0]["id"])
            status["stage"] = f"confirmation:{best['experiment_id']}"
            write_json(status_path, status)
            confirm_results = run_all([seeded(c, s) for s in CONFIRM_SEEDS for c in [CONTROL, best]],
                                      args.workers, args.threads)
            results.update(confirm_results)
            confirmation = {"id": best["experiment_id"],
                            **confirm(accuracy(CONTROL, CONFIRM_SEEDS),
                                      accuracy(best, CONFIRM_SEEDS)),
                            "control_accuracy": accuracy(CONTROL, CONFIRM_SEEDS),
                            "challenger_accuracy": accuracy(best, CONFIRM_SEEDS)}
            print(f"CONFIRM {confirmation}", flush=True)
        promoted = bool(confirmation and confirmation["passes"])
        if promoted:
            improvement_alert(f"H-D-08 promoted {confirmation['id']}")
        assert protected == {name: file_hash(ROOT / name) for name in protected}
        summary = {
            "completed_at": timestamp(), "plan_item": "H-D-08",
            "control_accuracy": control_acc,
            "control_mean_accuracy": float(np.mean(list(control_acc.values()))),
            "control_errors": control_errors, "screening": rows,
            "confirmation": confirmation, "promoted": promoted,
            "protected_hashes": protected, "remote_actions": "none",
        }
        write_json(reports / f"{REPORT}_summary.json", summary)
        status.update(status="results_ready", updated_at=timestamp(), promoted=promoted)
        write_json(status_path, status)
        print(json.dumps({"promoted": promoted, "confirmation": confirmation}, indent=2),
              flush=True)
    except BaseException as error:
        status.update(status="failed", updated_at=timestamp(), error=str(error))
        write_json(status_path, status)
        (reports / f"{REPORT}_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
