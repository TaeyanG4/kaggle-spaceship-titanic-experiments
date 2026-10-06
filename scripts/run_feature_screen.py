"""Multi-seed screen of CatBoost feature-subset candidates against a matched control.

Usage: run_feature_screen.py --spec configs/<item>_spec.json
Spec: {"item": "H-D-03", "prefix": "hm03", "stopping": "fixed300", "variant": "baseline",
       "candidates": {"name": ["Col", ...]}}  (a list = columns to drop; control drops none).
A candidate may instead be {"drop": [...], "stopping": "...", "params": {...}, "variant": "...",
"model": "catboost" | "xgboost" | "catboost_knn" | "catboost_extra" | "extratrees" | "histgb" |
"viktor_faithful" | "viktor_plain" | "noise_drop35" | "noise_half25" | "tabpfn" | "tabicl" |
"tabpfn_variant" (drop ["RAW_MINIMAL"] = baseline minus engineered columns),
"extras": [...]}.
Optional "control_id": reuse an existing run family (e.g. "hm10_innercv_logloss") as the control;
it must have the same stopping/variant/params as the spec defaults (the runner cache verifies this).
Rule: spaceship_titanic.screening (agents.md). Sound only on a confirmed promotion.
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
from catboost_runner import run_catboost
from extra_runner import run_catboost_extra
from knn_runner import run_catboost_knn
from member_runner import MODELS as MEMBER_MODELS
from member_runner import run_member
from noise_runner import run_noise
from run_hm02_grid import worker_init
from run_versions import Tee
from sklearn_runner import run_sklearn
from tabicl_runner import run_tabicl
from tabpfn_ft_runner import run_tabpfn_ft
from tabpfn_ftrefit_runner import run_tabpfn_ftrefit
from tabpfn_ne_runner import run_tabpfn_ne
from tabpfn_runner import run_tabpfn
from tabpfn_variant_runner import RAW_MINIMAL_DROP, run_tabpfn_variant
from tabr_runner import run_tabr
from viktor_runner import run_viktor
from xgb_runner import run_xgboost

ROOT = Path(__file__).resolve().parents[1]


def job_id(prefix: str, name: str, seed: int) -> str:
    return f"{prefix}_{name}" + ("" if seed == 42 else f"_seed{seed}")


def candidate(spec: dict, name: str) -> dict:
    value = {} if name == "control" else spec["candidates"][name]
    if isinstance(value, list):
        value = {"drop": value}
    return {"drop": value.get("drop", []), "stopping": value.get("stopping", spec["stopping"]),
            "params": spec.get("params", {}) | value.get("params", {}),
            "variant": value.get("variant", spec.get("variant", "baseline")),
            "model": value.get("model", spec.get("model", "catboost")),
            "extras": value.get("extras", []),
            "version": value.get("version", "v3.5"),
            "n_estimators": value.get("n_estimators")}


def run_job(spec: dict, name: str, seed: int, threads: int) -> dict:
    c = candidate(spec, name)
    if name == "control" and spec.get("control_id"):
        base = spec["control_id"]
        experiment = base if seed == 42 else f"{base}_seed{seed}"
    else:
        experiment = job_id(spec["prefix"], name, seed)
    if c["model"] == "tabpfn_variant":
        drop = RAW_MINIMAL_DROP if c["drop"] == ["RAW_MINIMAL"] else c["drop"]
        return run_tabpfn_variant(experiment, seed, threads, c["variant"], drop)
    if c["model"] == "tabr_gpu":
        return run_tabr(experiment, seed, threads)
    if c["model"] in MEMBER_MODELS:
        return run_member(experiment, seed, c["model"], threads)
    if c["model"] == "tabicl":
        return run_tabicl(experiment, seed, threads)
    if c["model"] == "tabpfn_ftrefit":
        return run_tabpfn_ftrefit(experiment, seed, threads)
    if c["model"] == "tabpfn_ft":
        return run_tabpfn_ft(experiment, seed, threads)
    if c["model"] == "tabpfn" and c["n_estimators"]:
        return run_tabpfn_ne(experiment, seed, threads, c["n_estimators"], c["version"])
    if c["model"] == "tabpfn":
        return run_tabpfn(experiment, seed, threads, c["version"])
    if c["model"] in ("viktor_faithful", "viktor_plain"):
        return run_viktor(experiment, seed, c["model"], threads)
    if c["model"] in ("noise_drop35", "noise_half25"):
        return run_noise(experiment, seed, c["model"], threads)
    if c["model"] in ("extratrees", "histgb"):
        return run_sklearn(experiment, seed, c["model"], threads, c["variant"])
    if c["model"] == "catboost_extra":
        return run_catboost_extra(experiment, seed, c["extras"], threads)
    if c["model"] == "catboost_knn":
        return run_catboost_knn(experiment, seed, threads)
    runner = run_xgboost if c["model"] == "xgboost" else run_catboost
    return runner(experiment, seed, c["stopping"], c["drop"], threads, c["variant"],
                  c["params"] or None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    worker_init(args.threads)
    import numpy as np

    from spaceship_titanic.experiments import file_hash, load_data, timestamp, write_json
    from spaceship_titanic.screening import CONFIRM_SEEDS, SCREEN_SEEDS, confirm, screen

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    prefix = spec["prefix"]
    reports = ROOT / "reports"
    sys.stdout = Tee(sys.stdout, reports / f"{prefix}_training.log")
    status_path = reports / f"{prefix}_status.json"
    status = {"status": "running", "started_at": timestamp(), "pid": os.getpid(),
              "spec": str(args.spec)}
    write_json(status_path, status)
    try:
        import pandas as pd

        train, _, _ = load_data()
        earth = train.HomePlanet.eq("Earth").to_numpy()
        target = train.Transported.astype(bool).to_numpy()
        protected = {n: file_hash(ROOT / n) for n in [
            "data/raw/train.csv", "data/raw/test.csv",
            "outputs/oof/baseline_catboost_oof.csv", "outputs/submissions/baseline_catboost.csv"]}
        write_json(reports / f"{prefix}_manifest.json", {
            "created_at": timestamp(), "spec": spec, "screen_seeds": list(SCREEN_SEEDS),
            "confirm_seeds": list(CONFIRM_SEEDS),
            "rule": "agents.md multi-seed screen; best passing candidate confirmed on 7/99"})

        def run_all(names, seeds):
            with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init,
                                     initargs=(args.threads,)) as pool:
                futures = {(n, s): pool.submit(run_job, spec, n, s, args.threads)
                           for s in seeds for n in names}
                return {key: f.result() for key, f in futures.items()}

        def earth_errors(result):
            oof = pd.read_csv(ROOT / result["artifacts"]["oof"])
            return int(((oof.probability.to_numpy() >= 0.5) != target)[earth].sum())

        names = ["control", *spec["candidates"]]
        results = run_all(names, SCREEN_SEEDS)
        control = {s: results[("control", s)]["accuracy"] for s in SCREEN_SEEDS}
        rows = []
        for name in spec["candidates"]:
            challenger = {s: results[(name, s)]["accuracy"] for s in SCREEN_SEEDS}
            outcome = screen(control, challenger)
            rows.append({"candidate": name, **candidate(spec, name), **outcome,
                         "accuracy": challenger,
                         "earth_errors": {s: earth_errors(results[(name, s)])
                                          for s in SCREEN_SEEDS}})
            print(f"{name:24s} mean_delta={outcome['mean_delta']:+.6f} "
                  f"positive={outcome['positive_seeds']}/3 passes={outcome['passes']}", flush=True)
        rows.sort(key=lambda r: -r["mean_delta"])
        confirmation = None
        passing = [r for r in rows if r["passes"]]
        if passing:
            best = passing[0]["candidate"]
            cres = run_all(["control", best], CONFIRM_SEEDS)
            confirmation = {"candidate": best, **confirm(
                {s: cres[("control", s)]["accuracy"] for s in CONFIRM_SEEDS},
                {s: cres[(best, s)]["accuracy"] for s in CONFIRM_SEEDS})}
            print(f"CONFIRM {confirmation}", flush=True)
        promoted = bool(confirmation and confirmation["passes"])
        if promoted:
            improvement_alert(f"{spec['item']} promoted {confirmation['candidate']}")
        assert protected == {n: file_hash(ROOT / n) for n in protected}
        write_json(reports / f"{prefix}_summary.json", {
            "completed_at": timestamp(), "item": spec["item"], "spec": spec,
            "control_accuracy": control,
            "control_mean_accuracy": float(np.mean(list(control.values()))),
            "control_earth_errors": {s: earth_errors(results[("control", s)])
                                     for s in SCREEN_SEEDS},
            "screening": rows, "confirmation": confirmation, "promoted": promoted,
            "protected_hashes": protected, "remote_actions": "none"})
        status.update(status="results_ready", updated_at=timestamp(), promoted=promoted)
        write_json(status_path, status)
        print(json.dumps({"promoted": promoted, "confirmation": confirmation}), flush=True)
    except BaseException as error:
        status.update(status="failed", updated_at=timestamp(), error=str(error))
        write_json(status_path, status)
        (reports / f"{prefix}_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
