"""H-A-27: nested multi-member stacker over saved OOF probabilities (no training of members).

For each seed and outer fold k, a logistic regression on the logits of the members' OOF
probabilities is fit on the rows of the other four folds and applied to fold k, so no fold is
scored with a stacker fitted on itself. Compared with the TabPFN v3.5 champion (`ha12_tabpfn`)
under the agents.md screen rule. Also reports the equal mean of all members for reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
CONTROL = "ha12_tabpfn"
MEMBER_SETS = {
    "all10": ["ha12_tabpfn", "ha23_ft", "ha24_ftrefit", "hm10_innercv_logloss", "hm05_xgb_base",
              "ha05_histgb", "ha05_extratrees", "ha18_tabpfn_v25", "ha18b_tabicl",
              "ha17_viktor_plain"],
    "strong4": ["ha12_tabpfn", "ha23_ft", "ha24_ftrefit", "hm10_innercv_logloss"],
    "tabpfn3": ["ha12_tabpfn", "ha23_ft", "ha24_ftrefit"],
}


def exp(base: str, seed: int) -> str:
    return base if seed == 42 else f"{base}_seed{seed}"


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def main() -> None:
    from spaceship_titanic.experiments import load_data, timestamp, write_json
    from spaceship_titanic.screening import SCREEN_SEEDS, screen

    train, _, _ = load_data()
    y = train.Transported.astype(int).to_numpy()
    results = {}
    for name, members in MEMBER_SETS.items():
        control_acc, stack_acc, mean_acc = {}, {}, {}
        for seed in SCREEN_SEEDS:
            control = pd.read_csv(ROOT / f"outputs/oof/{exp(CONTROL, seed)}.csv")
            folds = control.fold.to_numpy()
            probs = []
            for m in members:
                frame = pd.read_csv(ROOT / f"outputs/oof/{exp(m, seed)}.csv")
                assert frame.PassengerId.equals(train.PassengerId)
                assert np.array_equal(frame.fold.to_numpy(), folds)
                probs.append(frame.probability.to_numpy())
            X = np.column_stack([logit(p) for p in probs])
            stacked = np.zeros(len(y))
            for k in np.unique(folds):
                fit, hold = folds != k, folds == k
                lr = LogisticRegression(C=1.0, max_iter=1000).fit(X[fit], y[fit])
                stacked[hold] = lr.predict_proba(X[hold])[:, 1]
            control_acc[seed] = float(((control.probability.to_numpy() >= 0.5) == y).mean())
            stack_acc[seed] = float(((stacked >= 0.5) == y).mean())
            mean_acc[seed] = float(((np.mean(probs, axis=0) >= 0.5) == y).mean())
        results[name] = {"members": members, "stacker": screen(control_acc, stack_acc),
                         "equal_mean": screen(control_acc, mean_acc),
                         "control": control_acc, "stack": stack_acc, "mean": mean_acc}
        s, e = results[name]["stacker"], results[name]["equal_mean"]
        print(f"{name:8s} stacker mean_delta={s['mean_delta']:+.6f} positive={s['positive_seeds']}/3 "
              f"passes={s['passes']} | equal_mean {e['mean_delta']:+.6f} ({e['positive_seeds']}/3)",
              flush=True)
    promoted = any(r["stacker"]["passes"] or r["equal_mean"]["passes"] for r in results.values())
    write_json(ROOT / "reports/ha27_summary.json", {
        "completed_at": timestamp(), "plan_item": "H-A-27", "control": CONTROL,
        "results": results, "any_pass": promoted,
        "note": "nested LR stacker over member OOF logits; a pass would still need 7/99 confirmation"})
    print(json.dumps({"any_pass": promoted}), flush=True)


if __name__ == "__main__":
    main()
