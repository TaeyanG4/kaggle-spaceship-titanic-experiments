"""H-D-11: seed bagging of the champion (`hm10_innercv_logloss`) across SGKF splits.

No training. Each seed's OOF probability for a row comes from a model that never saw that row, so
the mean over seeds is a valid OOF estimate of the bag. Predeclared rule:
screen  = acc(bag of 42/123/2026) - acc(single seed s) for s in 42/123/2026 (spaceship_titanic.screening)
confirm = acc(bag of 7/99) - acc(single seed s) for s in 7/99, both positive.
If promoted, the submission averages test probabilities over all five seeds' fold models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from alert import improvement_alert

ROOT = Path(__file__).resolve().parents[1]
BASE = "hm10_innercv_logloss"


def name(seed: int) -> str:
    return BASE if seed == 42 else f"{BASE}_seed{seed}"


def main() -> None:
    from spaceship_titanic.experiments import file_hash, load_data, timestamp, write_json
    from spaceship_titanic.screening import CONFIRM_SEEDS, SCREEN_SEEDS, confirm, screen

    train, test, sample = load_data()
    y = train.Transported.astype(bool).to_numpy()
    earth = train.HomePlanet.eq("Earth").to_numpy()
    seeds = (*SCREEN_SEEDS, *CONFIRM_SEEDS)
    oof, prob = {}, {}
    for seed in seeds:
        frame = pd.read_csv(ROOT / f"outputs/oof/{name(seed)}.csv", dtype={"PassengerId": str})
        assert frame.PassengerId.equals(train.PassengerId)
        oof[seed] = frame.probability.to_numpy()
        tp = pd.read_csv(ROOT / f"outputs/probabilities/{name(seed)}.csv", dtype={"PassengerId": str})
        assert tp.PassengerId.equals(test.PassengerId)
        prob[seed] = tp.probability.to_numpy()

    def acc(p):
        return float(((p >= 0.5) == y).mean())

    def bag(group):
        return np.mean([oof[s] for s in group], axis=0)

    single = {s: acc(oof[s]) for s in seeds}
    screen_bag, confirm_bag = acc(bag(SCREEN_SEEDS)), acc(bag(CONFIRM_SEEDS))
    screened = screen(single_subset := {s: single[s] for s in SCREEN_SEEDS},
                      {s: screen_bag for s in SCREEN_SEEDS})
    confirmation = None
    if screened["passes"]:
        confirmation = confirm({s: single[s] for s in CONFIRM_SEEDS},
                               {s: confirm_bag for s in CONFIRM_SEEDS})
    promoted = bool(confirmation and confirmation["passes"])
    all_bag = bag(seeds)
    summary = {
        "completed_at": timestamp(), "plan_item": "H-D-11", "base": BASE,
        "single_accuracy": single, "screen_bag_accuracy": screen_bag,
        "confirm_bag_accuracy": confirm_bag, "all5_bag_accuracy": acc(all_bag),
        "screen": screened, "screen_singles": single_subset, "confirmation": confirmation,
        "promoted": promoted,
        "earth_errors": {"all5_bag": int(((all_bag >= 0.5) != y)[earth].sum()),
                         **{str(s): int(((oof[s] >= 0.5) != y)[earth].sum()) for s in seeds}},
        "note": "Bag OOF rows are out-of-fold in every component; the 5-seed bag uses the "
                "confirmation seeds too, so all5 accuracy is descriptive only.",
    }
    if promoted:
        test_probability = np.mean([prob[s] for s in seeds], axis=0)
        submission = sample.copy()
        submission.Transported = test_probability >= 0.5
        assert submission.PassengerId.equals(test.PassengerId)
        assert submission.Transported.dtype == bool
        out = ROOT / "outputs/submissions/hm11_bag5_innercv.csv"
        submission.to_csv(out, index=False)
        pd.DataFrame({"PassengerId": test.PassengerId, "probability": test_probability}).to_csv(
            ROOT / "outputs/probabilities/hm11_bag5_innercv.csv", index=False)
        summary["submission"] = {"path": str(out.relative_to(ROOT)), "sha256": file_hash(out),
                                 "positive_rate": float(submission.Transported.mean())}
        improvement_alert("H-D-11 promoted 5-seed bag of hm10_innercv_logloss")
    write_json(ROOT / "reports/hm11_summary.json", summary)
    print({k: summary[k] for k in ["single_accuracy", "screen_bag_accuracy", "confirm_bag_accuracy",
                                   "screen", "confirmation", "promoted"]})


if __name__ == "__main__":
    main()
