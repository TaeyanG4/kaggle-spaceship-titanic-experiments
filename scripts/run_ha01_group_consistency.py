"""H-A-01: group-consistency post-processing of uncertain champion predictions (no training).

A member of a multi-passenger group with |p - 0.5| < B gets the mean probability of its confident
groupmates (|p - 0.5| >= C), if it has any. Uses predictions only, never labels; SGKF keeps whole
groups in one fold, so the champion's OOF stays honest. Predeclared B=0.10, C=0.20.
Rule: agents.md screen on seeds 42/123/2026. If it passes, (B, C) is chosen per outer fold from a
small grid on the other folds' rows, then confirmed on seeds 7/99.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from alert import improvement_alert

ROOT = Path(__file__).resolve().parents[1]
BASE = "hm10_innercv_logloss"
B, C = 0.10, 0.20
GRID = [(b, c) for b in (0.05, 0.10, 0.15) for c in (0.15, 0.20, 0.30)]


def adjust(p: np.ndarray, groups: pd.Series, b: float, c: float) -> np.ndarray:
    out = p.copy()
    frame = pd.DataFrame({"g": groups.to_numpy(), "p": p})
    confident = (frame.p - 0.5).abs() >= c
    sums = frame.p.where(confident, 0.0).groupby(frame.g).transform("sum")
    counts = confident.astype(int).groupby(frame.g).transform("sum")
    own = confident.astype(int)
    peer_count = counts - own
    peer_mean = (sums - frame.p.where(confident, 0.0)) / peer_count.replace(0, np.nan)
    target = ((frame.p - 0.5).abs() < b) & peer_count.gt(0)
    out[target.to_numpy()] = peer_mean[target].to_numpy()
    return out


def main() -> None:
    from spaceship_titanic.experiments import load_data, timestamp, write_json
    from spaceship_titanic.screening import CONFIRM_SEEDS, SCREEN_SEEDS, confirm, screen

    train, _, _ = load_data()
    y = train.Transported.astype(bool).to_numpy()
    groups = train.PassengerId.str.split("_").str[0]
    earth = train.HomePlanet.eq("Earth").to_numpy()

    def load(seed):
        name = BASE if seed == 42 else f"{BASE}_seed{seed}"
        frame = pd.read_csv(ROOT / f"outputs/oof/{name}.csv", dtype={"PassengerId": str})
        assert frame.PassengerId.equals(train.PassengerId)
        return frame.probability.to_numpy(), frame.fold.to_numpy()

    def acc(p, rows=slice(None)):
        return float(((p[rows] >= 0.5) == y[rows]).mean())

    control, fixed, changed, earth_err = {}, {}, {}, {}
    for seed in SCREEN_SEEDS:
        p, _ = load(seed)
        q = adjust(p, groups, B, C)
        control[seed], fixed[seed] = acc(p), acc(q)
        changed[seed] = int(((p >= 0.5) != (q >= 0.5)).sum())
        earth_err[seed] = {"control": int(((p >= 0.5) != y)[earth].sum()),
                           "adjusted": int(((q >= 0.5) != y)[earth].sum())}
    screened = screen(control, fixed)
    print(f"fixed B={B} C={C}: {screened} changed={changed}", flush=True)

    confirmation = None
    if screened["passes"]:
        nested, ctrl = {}, {}
        for seed in CONFIRM_SEEDS:
            p, folds = load(seed)
            correct = 0
            for k in np.unique(folds):
                out_rows = folds != k
                b, c = max(GRID, key=lambda bc: acc(adjust(p, groups, *bc), out_rows))
                correct += int(((adjust(p, groups, b, c)[folds == k] >= 0.5) == y[folds == k]).sum())
            nested[seed], ctrl[seed] = correct / len(y), acc(p)
        confirmation = confirm(ctrl, nested)
        print(f"CONFIRM {confirmation}", flush=True)
    promoted = bool(confirmation and confirmation["passes"])
    if promoted:
        improvement_alert("H-A-01 promoted group-consistency post-processing")
    write_json(ROOT / "reports/ha01_summary.json", {
        "completed_at": timestamp(), "plan_item": "H-A-01", "base": BASE, "B": B, "C": C,
        "control_accuracy": control, "adjusted_accuracy": fixed, "changed_predictions": changed,
        "earth_errors": earth_err, "screen": screened, "confirmation": confirmation,
        "promoted": promoted})


if __name__ == "__main__":
    main()
