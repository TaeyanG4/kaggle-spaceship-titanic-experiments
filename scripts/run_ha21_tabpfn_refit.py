"""H-A-21: does TabPFN conditioned on more rows beat the 5-model 80%-context average? (nested proxy)

Inside each outer SGKF fold (fit fold F, validation V):
- control: mean of 5 TabPFN v3.5 fits, each conditioned on 4/5 of F (inner SGKF) = submission recipe;
- challenger: one TabPFN fit conditioned on all of F = the saved `ha12_tabpfn` OOF (reused).
Both are scored on the same V rows. Rule: agents.md screen (42/123/2026) + confirm (7/99).
If promoted, the final submission is one TabPFN fit conditioned on the whole train set.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from alert import improvement_alert
from run_hm02_grid import worker_init

ROOT = Path(__file__).resolve().parents[1]
BASE = "ha12_tabpfn"


def name(seed: int) -> str:
    return BASE if seed == 42 else f"{BASE}_seed{seed}"


def proxy(seed: int, threads: int) -> dict:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import OrdinalEncoder
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    from spaceship_titanic.experiments import frozen_folds, load_data, write_json
    from spaceship_titanic.versioned_features import build_versioned_features

    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    cache = ROOT / f"reports/ha21_seed{seed}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    train, test, _ = load_data()
    folds = frozen_folds(train, seed, 5)
    x, _, categorical = build_versioned_features(train, test, "baseline")
    y = train.Transported.astype(int).to_numpy()
    groups = train.PassengerId.str.split("_").str[0]
    cat_idx = [x.columns.get_loc(c) for c in categorical]
    challenger = pd.read_csv(ROOT / f"outputs/oof/{name(seed)}.csv").probability.to_numpy()
    control = np.zeros(len(x))

    def fit_predict(rows, valid, random_state):
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(x.iloc[rows][categorical])
        frames = []
        for frame in (x.iloc[rows], x.iloc[valid]):
            frame = frame.copy()
            frame[categorical] = encoder.transform(frame[categorical])
            frames.append(frame.to_numpy(dtype=np.float32))
        model = TabPFNClassifier.create_default_for_version(
            ModelVersion.V3_5, categorical_features_indices=cat_idx, device="cuda",
            random_state=random_state)
        model.fit(frames[0], y[rows])
        return model.predict_proba(frames[1])[:, 1]

    for fold in range(1, 6):
        fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        fit_seed = seed + fold
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=fit_seed)
        parts = [fit_predict(fit_idx[a], valid_idx, fit_seed + 100 + j)
                 for j, (a, _) in enumerate(splitter.split(fit_idx, y[fit_idx],
                                                           groups.iloc[fit_idx]))]
        control[valid_idx] = np.mean(parts, axis=0)

    def acc(p):
        return float(((p >= 0.5) == y).mean())

    result = {"seed": seed, "control": acc(control), "challenger": acc(challenger)}
    write_json(cache, result)
    print(f"seed {seed}: {result}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
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
    outcome = screen({s: res[s]["control"] for s in SCREEN_SEEDS},
                     {s: res[s]["challenger"] for s in SCREEN_SEEDS})
    print(f"full_context mean_delta={outcome['mean_delta']:+.6f} "
          f"positive={outcome['positive_seeds']}/3 passes={outcome['passes']}", flush=True)
    confirmation = None
    if outcome["passes"]:
        cres = run(CONFIRM_SEEDS)
        res.update(cres)
        confirmation = confirm({s: cres[s]["control"] for s in CONFIRM_SEEDS},
                               {s: cres[s]["challenger"] for s in CONFIRM_SEEDS})
        print(f"CONFIRM {confirmation}", flush=True)
    promoted = bool(confirmation and confirmation["passes"])
    if promoted:
        improvement_alert("H-A-21 promoted TabPFN full-context refit")
    write_json(ROOT / "reports/ha21_summary.json", {
        "completed_at": timestamp(), "plan_item": "H-A-21", "per_seed": res, "screen": outcome,
        "confirmation": confirmation, "promoted": promoted})
    print(json.dumps({"promoted": promoted, "confirmation": confirmation}), flush=True)


if __name__ == "__main__":
    main()
