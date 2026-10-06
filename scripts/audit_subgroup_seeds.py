"""H-A-43 (re-review of D-D-021): multi-seed Earth / Deck G subgroup audit over saved honest OOF.

Read-only (no training). Predeclared checks:
1. D-D-021 mechanism: Logloss-stopped vs Accuracy-stopped CatBoost under honest inner-CV stopping
   (`hm10_innercv_logloss` vs `hm09_inner`, seeds 42/123/2026), and vs `fixed300` (`hm10_control`,
   seeds 42/123/2026/7/99): paired change in Earth and non-Earth errors, Earth predicted positive
   rate, and the share of changed Earth rows within ±0.1 of 0.5. Stable if the direction (Earth
   errors up, non-Earth down) holds on at least 2/3 screen seeds.
2. Subgroup error ratios for the current champion (nested `ha27_stack3`, five seeds): error rate in
   Earth, Deck G, and CryoSleep with zero observed spend, relative to the overall error rate.
Writes `reports/ha43_subgroup_audit.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_ha27_stacker import exp
from run_stack_member_screen import BASE, member_matrix
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]


def oof(name, seed):
    return pd.read_csv(ROOT / f"outputs/oof/{exp(name, seed)}.csv").probability.to_numpy()


def stack_oof(train, y, seed):
    X, folds = member_matrix(BASE, seed, train)
    out = np.zeros(len(y))
    for k in np.unique(folds):
        fit, hold = folds != k, folds == k
        out[hold] = LogisticRegression(C=1.0, max_iter=1000).fit(X[fit], y[fit]) \
            .predict_proba(X[hold])[:, 1]
    return out


def main() -> None:
    from spaceship_titanic.experiments import load_data, timestamp, write_json

    train, _, _ = load_data()
    y = train.Transported.astype(bool).to_numpy()
    earth = train.HomePlanet.eq("Earth").to_numpy()
    known = train.HomePlanet.notna().to_numpy()
    deck_g = train.Cabin.fillna("").str.startswith("G/").to_numpy()
    spend = train[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]]
    cryo_zero = (train.CryoSleep.astype("boolean").fillna(False).to_numpy()
                 & spend.fillna(0).sum(axis=1).eq(0).to_numpy())

    def errors(p):
        return (p >= 0.5) != y

    result = {"completed_at": timestamp(), "plan_item": "H-A-43", "pairs": {}, "champion": {}}
    pairs = {"logloss_vs_accuracy_inner": ("hm10_innercv_logloss", "hm09_inner", [42, 123, 2026]),
             "logloss_vs_fixed300": ("hm10_innercv_logloss", "hm10_control", [42, 123, 2026, 7, 99])}
    for label, (a, b, seeds) in pairs.items():
        rows = {}
        for s in seeds:
            pa, pb = oof(a, s), oof(b, s)
            ea, eb = errors(pa), errors(pb)
            changed = earth & (ea != eb)
            rows[s] = {
                "earth_error_change": int(ea[earth].sum() - eb[earth].sum()),
                "non_earth_error_change": int(ea[~earth & known].sum() - eb[~earth & known].sum()),
                "earth_pos_rate": [float((pa[earth] >= 0.5).mean()), float((pb[earth] >= 0.5).mean())],
                "changed_earth_rows": int(changed.sum()),
                "changed_earth_near_half": float((np.abs(pa[changed] - 0.5) <= 0.1).mean())
                if changed.any() else None,
            }
        direction = sum(r["earth_error_change"] > 0 and r["non_earth_error_change"] < 0
                        for r in rows.values())
        result["pairs"][label] = {"a": a, "b": b, "seeds": rows,
                                  "direction_holds_on": f"{direction}/{len(seeds)}"}
        print(label, json.dumps(rows), f"direction {direction}/{len(seeds)}", flush=True)
    for s in [42, 123, 2026, 7, 99]:
        e = errors(stack_oof(train, y.astype(int), s))
        overall = e.mean()
        result["champion"][s] = {
            "overall_error": float(overall),
            "earth_ratio": float(e[earth].mean() / overall),
            "deck_g_ratio": float(e[deck_g].mean() / overall),
            "cryo_zero_spend_ratio": float(e[cryo_zero].mean() / overall),
            "non_earth_ratio": float(e[~earth & known].mean() / overall),
        }
        print("champion", s, {k: round(v, 3) for k, v in result["champion"][s].items()}, flush=True)
    write_json(ROOT / "reports/ha43_subgroup_audit.json", result)


if __name__ == "__main__":
    main()
