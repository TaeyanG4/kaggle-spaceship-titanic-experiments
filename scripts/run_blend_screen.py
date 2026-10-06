"""H-D-06: blends / thresholds over saved OOF, evaluated with nested (outer-fold) selection.

For each SGKF seed, every candidate is scored on outer fold k using parameters chosen only from the
other four folds' OOF rows, so no fold is scored with parameters fitted on itself. The champion
(single model, threshold 0.5) is the matched control; the agents.md screen/confirm rule applies.
Members must exist for the screen seeds, and for seeds 7/99 if a candidate reaches confirmation.

Usage: run_blend_screen.py --spec configs/hm06_spec.json
Spec: {"item": ..., "prefix": ..., "control": "<exp base>", "members": ["<exp base>", ...]}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from alert import improvement_alert

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.1), 2)
THRESHOLDS = np.round(np.arange(0.44, 0.5601, 0.01), 2)


def exp(base: str, seed: int) -> str:
    return base if seed == 42 else f"{base}_seed{seed}"


def load(base: str, seed: int, ids: pd.Series) -> pd.DataFrame:
    frame = pd.read_csv(ROOT / f"outputs/oof/{exp(base, seed)}.csv", dtype={"PassengerId": str})
    assert frame.PassengerId.equals(ids)
    return frame


def nested_accuracy(y, folds, choose, apply) -> float:
    correct = 0
    for k in np.unique(folds):
        out = folds != k
        params = choose(out)
        correct += int((apply(params, folds == k) == y[folds == k]).sum())
    return correct / len(y)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    from spaceship_titanic.experiments import load_data, timestamp, write_json
    from spaceship_titanic.screening import CONFIRM_SEEDS, SCREEN_SEEDS, confirm, screen

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    train, _, _ = load_data()
    y = train.Transported.astype(bool).to_numpy()

    def scores(seed: int) -> dict[str, float]:
        control = load(spec["control"], seed, train.PassengerId)
        folds = control.fold.to_numpy()
        p0 = control.probability.to_numpy()
        members = []
        for base in spec["members"]:
            frame = load(base, seed, train.PassengerId)
            assert np.array_equal(frame.fold.to_numpy(), folds)
            members.append(frame.probability.to_numpy())
        p1 = np.mean(members, axis=0)  # other members, equal weight
        acc = lambda pred: float((pred == y).mean())
        out = {"control": acc(p0 >= 0.5), "equal_blend": acc((p0 + p1 * len(members))
                                                          / (1 + len(members)) >= 0.5)}

        def choose_weight(rows):
            return max(WEIGHTS, key=lambda w: (((w * p0 + (1 - w) * p1)[rows] >= 0.5) == y[rows])
                       .mean() - 1e-9 * abs(w - 0.5))

        out["nested_weight"] = nested_accuracy(
            y, folds, choose_weight, lambda w, rows: (w * p0 + (1 - w) * p1)[rows] >= 0.5)

        def choose_threshold(rows):
            return max(THRESHOLDS, key=lambda t: ((p0[rows] >= t) == y[rows]).mean()
                       - 1e-9 * abs(t - 0.5))

        out["nested_threshold"] = nested_accuracy(
            y, folds, choose_threshold, lambda t, rows: p0[rows] >= t)
        return out

    per_seed = {s: scores(s) for s in SCREEN_SEEDS}
    rows = []
    for cand in ("equal_blend", "nested_weight", "nested_threshold"):
        outcome = screen({s: per_seed[s]["control"] for s in SCREEN_SEEDS},
                         {s: per_seed[s][cand] for s in SCREEN_SEEDS})
        rows.append({"candidate": cand, **outcome})
        print(f"{cand:18s} mean_delta={outcome['mean_delta']:+.6f} "
              f"positive={outcome['positive_seeds']}/3 passes={outcome['passes']}", flush=True)
    rows.sort(key=lambda r: -r["mean_delta"])
    passing = [r for r in rows if r["passes"]]
    confirmation = None
    if passing:
        cand = passing[0]["candidate"]
        per_seed.update({s: scores(s) for s in CONFIRM_SEEDS})
        confirmation = {"candidate": cand, **confirm(
            {s: per_seed[s]["control"] for s in CONFIRM_SEEDS},
            {s: per_seed[s][cand] for s in CONFIRM_SEEDS})}
        print(f"CONFIRM {confirmation}", flush=True)
    promoted = bool(confirmation and confirmation["passes"])
    summary = {"completed_at": timestamp(), "item": spec["item"], "spec": spec,
               "per_seed": per_seed, "screening": rows, "confirmation": confirmation,
               "promoted": promoted}
    if promoted:
        summary["submission_note"] = ("build the final blend from seed-42 member test probabilities "
                                      "with parameters chosen on all seed-42 OOF rows")
        improvement_alert(f"{spec['item']} promoted {confirmation['candidate']}")
    write_json(ROOT / f"reports/{spec['prefix']}_summary.json", summary)
    print(json.dumps({"promoted": promoted, "confirmation": confirmation}), flush=True)


if __name__ == "__main__":
    main()
