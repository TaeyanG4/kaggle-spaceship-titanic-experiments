"""H-A-07: does a full-data refit beat the current 5-model 80%-data average? (nested proxy)

Inside each outer SGKF fold (fit fold F, recorded champion iteration it_k):
- control: mean of 5 CatBoost models, each on 4/5 of F (inner SGKF), at it_k (current recipe);
- B1: the champion's single model on all of F at it_k (reused champion OOF);
- B2: a single model on all of F at round(1.25 * it_k).
All arms are scored on the same outer validation rows. Rule: agents.md screen + confirm (7/99).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from alert import improvement_alert
from run_hm02_grid import worker_init

ROOT = Path(__file__).resolve().parents[1]
BASE = "hm10_innercv_logloss"


def name(seed: int) -> str:
    return BASE if seed == 42 else f"{BASE}_seed{seed}"


def proxy(seed: int, threads: int) -> dict:
    import numpy as np
    import pandas as pd
    from catboost import CatBoostClassifier
    from catboost_runner import BASE_PARAMS
    from sklearn.model_selection import StratifiedGroupKFold

    from spaceship_titanic.experiments import frozen_folds, load_data, write_json
    from spaceship_titanic.versioned_features import build_versioned_features

    cache = ROOT / f"reports/ha07_proxy_seed{seed}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    train, test, _ = load_data()
    folds = frozen_folds(train, seed, 5)
    x, _, categorical = build_versioned_features(train, test, "baseline")
    y = train.Transported.astype(int).to_numpy()
    groups = train.PassengerId.str.split("_").str[0]
    champion = json.loads((ROOT / f"reports/{name(seed)}_metrics.json").read_text(encoding="utf-8"))
    b1 = pd.read_csv(ROOT / f"outputs/oof/{name(seed)}.csv").probability.to_numpy()
    control, b2 = np.zeros(len(x)), np.zeros(len(x))

    def fit(rows, iterations, random_seed):
        model = CatBoostClassifier(**(BASE_PARAMS | {"iterations": iterations}),
                                   random_seed=random_seed, thread_count=threads, verbose=False,
                                   allow_writing_files=False)
        model.fit(x.iloc[rows], y[rows], cat_features=categorical)
        return model

    for fold in range(1, 6):
        fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        fit_seed = seed + fold
        iterations = champion["folds"][fold - 1]["best_iteration"]
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=fit_seed)
        parts = []
        for j, (a, _) in enumerate(splitter.split(fit_idx, y[fit_idx], groups.iloc[fit_idx])):
            parts.append(fit(fit_idx[a], iterations, fit_seed + 100 + j)
                         .predict_proba(x.iloc[valid_idx], thread_count=threads)[:, 1])
        control[valid_idx] = np.mean(parts, axis=0)
        b2[valid_idx] = fit(fit_idx, round(1.25 * iterations), fit_seed).predict_proba(
            x.iloc[valid_idx], thread_count=threads)[:, 1]

    def acc(p):
        return float(((p >= 0.5) == y).mean())

    result = {"seed": seed, "control": acc(control), "b1": acc(b1), "b2": acc(b2)}
    write_json(cache, result)
    print(f"seed {seed}: {result}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    worker_init(args.threads)
    from spaceship_titanic.experiments import timestamp, write_json
    from spaceship_titanic.screening import CONFIRM_SEEDS, SCREEN_SEEDS, confirm, screen

    def run(seeds):
        with ProcessPoolExecutor(args.workers, initializer=worker_init,
                                 initargs=(args.threads,)) as pool:
            futures = {s: pool.submit(proxy, s, args.threads) for s in seeds}
            return {s: f.result() for s, f in futures.items()}

    res = run(SCREEN_SEEDS)
    rows = []
    for arm in ("b1", "b2"):
        outcome = screen({s: res[s]["control"] for s in SCREEN_SEEDS},
                         {s: res[s][arm] for s in SCREEN_SEEDS})
        rows.append({"arm": arm, **outcome})
        print(f"{arm} mean_delta={outcome['mean_delta']:+.6f} "
              f"positive={outcome['positive_seeds']}/3 passes={outcome['passes']}", flush=True)
    rows.sort(key=lambda r: -r["mean_delta"])
    confirmation = None
    if any(r["passes"] for r in rows):
        arm = next(r for r in rows if r["passes"])["arm"]
        cres = run(CONFIRM_SEEDS)
        res.update(cres)
        confirmation = {"arm": arm, **confirm({s: cres[s]["control"] for s in CONFIRM_SEEDS},
                                              {s: cres[s][arm] for s in CONFIRM_SEEDS})}
        print(f"CONFIRM {confirmation}", flush=True)
    promoted = bool(confirmation and confirmation["passes"])
    if promoted:
        improvement_alert(f"H-A-07 promoted full-train refit arm {confirmation['arm']}")
    write_json(ROOT / "reports/ha07_summary.json", {
        "completed_at": timestamp(), "plan_item": "H-A-07", "per_seed": res, "screening": rows,
        "confirmation": confirmation, "promoted": promoted})
    print(json.dumps({"promoted": promoted, "confirmation": confirmation}), flush=True)


if __name__ == "__main__":
    main()
