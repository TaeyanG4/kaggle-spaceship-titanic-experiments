"""H-A-27 confirmation: nested stacker over the three TabPFN v3.5 variants on seeds 7/99.

Members must already exist on the confirmation seeds (`ha12_tabpfn`, `ha23_ft`, `ha24_ftrefit`).
Per seed: logistic regression on member OOF logits, fitted on the other four outer folds, scored
on the held-out fold; compared with frozen TabPFN under `screening.confirm`. On a pass, the final
stacker is fitted on all seed-42 OOF rows and applied to the members' seed-42 fold-averaged test
probabilities to write `outputs/submissions/ha27_stack3.csv` (not submitted here).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from alert import improvement_alert
from run_ha27_stacker import CONTROL, exp, logit
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
MEMBERS = ["ha12_tabpfn", "ha23_ft", "ha24_ftrefit"]


def nested_stack(train, y, seed):
    control = pd.read_csv(ROOT / f"outputs/oof/{exp(CONTROL, seed)}.csv")
    folds = control.fold.to_numpy()
    probs = []
    for m in MEMBERS:
        frame = pd.read_csv(ROOT / f"outputs/oof/{exp(m, seed)}.csv")
        assert frame.PassengerId.equals(train.PassengerId)
        assert np.array_equal(frame.fold.to_numpy(), folds)
        probs.append(frame.probability.to_numpy())
    X = np.column_stack([logit(p) for p in probs])
    stacked = np.zeros(len(y))
    for k in np.unique(folds):
        fit, hold = folds != k, folds == k
        stacked[hold] = LogisticRegression(C=1.0, max_iter=1000).fit(X[fit], y[fit]) \
            .predict_proba(X[hold])[:, 1]
    acc = lambda p: float(((p >= 0.5) == y).mean())
    return acc(control.probability.to_numpy()), acc(stacked), X


def main() -> None:
    from spaceship_titanic.experiments import file_hash, load_data, timestamp, write_json
    from spaceship_titanic.screening import CONFIRM_SEEDS, confirm

    train, test, sample = load_data()
    y = train.Transported.astype(int).to_numpy()
    control_acc, stack_acc = {}, {}
    for seed in CONFIRM_SEEDS:
        control_acc[seed], stack_acc[seed], _ = nested_stack(train, y, seed)
        print(f"seed {seed}: control {control_acc[seed]:.6f} stacker {stack_acc[seed]:.6f} "
              f"delta {stack_acc[seed] - control_acc[seed]:+.6f}", flush=True)
    confirmation = confirm(control_acc, stack_acc)
    print(f"CONFIRM {confirmation}", flush=True)
    summary = {"completed_at": timestamp(), "plan_item": "H-A-27", "members": MEMBERS,
               "control_accuracy": control_acc, "stacker_accuracy": stack_acc,
               "confirmation": confirmation, "promoted": bool(confirmation["passes"])}
    if confirmation["passes"]:
        _, _, X42 = nested_stack(train, y, 42)
        final = LogisticRegression(C=1.0, max_iter=1000).fit(X42, y)
        test_probs = []
        for m in MEMBERS:
            frame = pd.read_csv(ROOT / f"outputs/probabilities/{m}.csv")
            assert frame.PassengerId.equals(test.PassengerId)
            test_probs.append(frame.probability.to_numpy())
        test_stacked = final.predict_proba(np.column_stack([logit(p) for p in test_probs]))[:, 1]
        submission = sample.copy()
        submission["Transported"] = test_stacked >= 0.5
        assert submission.PassengerId.equals(test.PassengerId)
        out = ROOT / "outputs/submissions/ha27_stack3.csv"
        submission.to_csv(out, index=False)
        pd.DataFrame({"PassengerId": test.PassengerId, "probability": test_stacked}).to_csv(
            ROOT / "outputs/probabilities/ha27_stack3.csv", index=False)
        summary["submission"] = {"path": str(out.relative_to(ROOT)), "sha256": file_hash(out),
                                 "positive_rate": float(submission.Transported.mean()),
                                 "stacker_coef": final.coef_[0].tolist(),
                                 "stacker_intercept": float(final.intercept_[0])}
        improvement_alert("H-A-27 promoted: nested stacker of TabPFN variants")
    write_json(ROOT / "reports/ha27_confirm_summary.json", summary)
    print(json.dumps({"promoted": summary["promoted"]}), flush=True)


if __name__ == "__main__":
    main()
