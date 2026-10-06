"""H-A-11 (from H-C-04): nested pseudo-label simulation; never touches the real test set.

Per outer SGKF fold (fit fold F, untouched validation V), using the champion recipe at the
champion's recorded iteration for that fold:
- split F by an inner StratifiedGroupKFold(5): F_lab (4/5, labels used) and F_unl (1/5, labels hidden);
- teachers A and B (different seeds) train on F_lab and score F_unl;
- accept F_unl rows where both teachers give p <= 0.03 or both give p >= 0.97 (predeclared);
- control = teacher A scored on V; challenger = student trained on F_lab + accepted pseudo-rows.
Only V's labels are scored. Rule: agents.md screen (42/123/2026) + confirm (7/99).
A real-test pseudo-label step would additionally need explicit user approval.
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
LOW, HIGH = 0.03, 0.97


def simulate(seed: int, threads: int) -> dict:
    import numpy as np
    from catboost import CatBoostClassifier
    from catboost_runner import BASE_PARAMS
    from sklearn.model_selection import StratifiedGroupKFold

    from spaceship_titanic.experiments import frozen_folds, load_data, write_json
    from spaceship_titanic.versioned_features import build_versioned_features

    cache = ROOT / f"reports/ha11_seed{seed}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    train, test, _ = load_data()
    folds = frozen_folds(train, seed, 5)
    x, _, categorical = build_versioned_features(train, test, "baseline")
    y = train.Transported.astype(int).to_numpy()
    groups = train.PassengerId.str.split("_").str[0]
    name = BASE if seed == 42 else f"{BASE}_seed{seed}"
    champion = json.loads((ROOT / f"reports/{name}_metrics.json").read_text(encoding="utf-8"))
    control, student = np.zeros(len(x)), np.zeros(len(x))
    accepted, pseudo_errors = 0, 0

    def fit(rows_x, rows_y, iterations, random_seed):
        model = CatBoostClassifier(**(BASE_PARAMS | {"iterations": iterations}),
                                   random_seed=random_seed, thread_count=threads, verbose=False,
                                   allow_writing_files=False)
        model.fit(rows_x, rows_y, cat_features=categorical)
        return model

    for fold in range(1, 6):
        fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        fit_seed = seed + fold
        iterations = champion["folds"][fold - 1]["best_iteration"]
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=fit_seed)
        a, b = next(splitter.split(fit_idx, y[fit_idx], groups.iloc[fit_idx]))
        lab, unl = fit_idx[a], fit_idx[b]
        assert not set(groups.iloc[lab]) & set(groups.iloc[unl])
        teacher_a = fit(x.iloc[lab], y[lab], iterations, fit_seed)
        teacher_b = fit(x.iloc[lab], y[lab], iterations, fit_seed + 1000)
        pa = teacher_a.predict_proba(x.iloc[unl], thread_count=threads)[:, 1]
        pb = teacher_b.predict_proba(x.iloc[unl], thread_count=threads)[:, 1]
        keep = ((pa <= LOW) & (pb <= LOW)) | ((pa >= HIGH) & (pb >= HIGH))
        pseudo_y = (pa[keep] >= 0.5).astype(int)
        accepted += int(keep.sum())
        pseudo_errors += int((pseudo_y != y[unl][keep]).sum())  # diagnostic only, not used to select
        control[valid_idx] = teacher_a.predict_proba(x.iloc[valid_idx], thread_count=threads)[:, 1]
        rows = np.r_[lab, unl[keep]]
        labels = np.r_[y[lab], pseudo_y]
        stud = fit(x.iloc[rows], labels, iterations, fit_seed)
        student[valid_idx] = stud.predict_proba(x.iloc[valid_idx], thread_count=threads)[:, 1]

    def acc(p):
        return float(((p >= 0.5) == y).mean())

    result = {"seed": seed, "control": acc(control), "student": acc(student),
              "accepted_pseudo_rows": accepted, "pseudo_label_errors": pseudo_errors}
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
            futures = {s: pool.submit(simulate, s, args.threads) for s in seeds}
            return {s: f.result() for s, f in futures.items()}

    res = run(SCREEN_SEEDS)
    outcome = screen({s: res[s]["control"] for s in SCREEN_SEEDS},
                     {s: res[s]["student"] for s in SCREEN_SEEDS})
    print(f"student mean_delta={outcome['mean_delta']:+.6f} "
          f"positive={outcome['positive_seeds']}/3 passes={outcome['passes']}", flush=True)
    confirmation = None
    if outcome["passes"]:
        cres = run(CONFIRM_SEEDS)
        res.update(cres)
        confirmation = confirm({s: cres[s]["control"] for s in CONFIRM_SEEDS},
                               {s: cres[s]["student"] for s in CONFIRM_SEEDS})
        print(f"CONFIRM {confirmation}", flush=True)
    promoted = bool(confirmation and confirmation["passes"])
    if promoted:
        improvement_alert("H-A-11 nested pseudo-label simulation passed (real-test step needs user approval)")
    write_json(ROOT / "reports/ha11_summary.json", {
        "completed_at": timestamp(), "plan_item": "H-A-11", "rule": f"accept p<={LOW} or p>={HIGH} with teacher agreement",
        "per_seed": res, "screen": outcome, "confirmation": confirmation, "promoted": promoted,
        "note": "simulation only; the real test set was not used"})
    print(json.dumps({"promoted": promoted, "confirmation": confirmation}), flush=True)


if __name__ == "__main__":
    main()
