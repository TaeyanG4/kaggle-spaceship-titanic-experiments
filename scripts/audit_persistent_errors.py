"""H-A-08 (from H-C-01): five-seed persistent-error audit of the champion's honest OOF (read-only).

For each train row: errors across seeds 42/123/2026/7/99 (0-5), mean and std of probability.
Persistent = wrong on >= 4 seeds; split-specific = wrong on 1-3. Segments are target-free.
Predeclared reading: a segment is "materially elevated" if it has >= 100 rows and a persistent-error
rate >= 1.5x the overall rate; if none qualifies beyond already-known HomePlanet/CryoSleep effects,
feature-side hypothesis generation stops.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spaceship_titanic.experiments import ROOT, load_data, timestamp, write_json
from spaceship_titanic.features import SPEND_COLUMNS

BASE = "hm10_innercv_logloss"
SEEDS = (42, 123, 2026, 7, 99)
MIN_ROWS = 100


def main() -> None:
    train, _, _ = load_data()
    y = train.Transported.astype(bool).to_numpy()
    probs = []
    for seed in SEEDS:
        name = BASE if seed == 42 else f"{BASE}_seed{seed}"
        frame = pd.read_csv(ROOT / f"outputs/oof/{name}.csv", dtype={"PassengerId": str})
        assert frame.PassengerId.equals(train.PassengerId)
        probs.append(frame.probability.to_numpy())
    p = np.vstack(probs)
    wrong = ((p >= 0.5) != y).sum(axis=0)
    mean_p, std_p = p.mean(axis=0), p.std(axis=0)
    persistent, split_specific = wrong >= 4, (wrong >= 1) & (wrong <= 3)

    cabin = train.Cabin.str.split("/", expand=True)
    groups = train.PassengerId.str[:4]
    segments = {
        "HomePlanet": train.HomePlanet.fillna("missing"),
        "CryoSleep": train.CryoSleep.astype("string").fillna("missing"),
        "CabinDeck": cabin[0].fillna("missing"),
        "CabinSide": cabin[2].fillna("missing"),
        "CabinNumBand": pd.cut(pd.to_numeric(cabin[1], errors="coerce"),
                               [-1, 300, 600, 900, 1200, 2000]).astype(str),
        "GroupSize": groups.map(groups.value_counts()).clip(upper=4).astype(str),
        "SpendMissingCount": train[SPEND_COLUMNS].isna().sum(axis=1).clip(upper=2).astype(str),
        "NoSpend": train[SPEND_COLUMNS].fillna(0).sum(axis=1).eq(0).astype(str),
        "AgeBand": pd.cut(train.Age, [-1, 12, 17, 25, 40, 200]).astype(str),
        "Destination": train.Destination.fillna("missing"),
    }
    overall = float(persistent.mean())
    rows = []
    for name, values in segments.items():
        frame = pd.DataFrame({"v": values.to_numpy(), "persist": persistent, "split": split_specific})
        for value, g in frame.groupby("v"):
            rate = float(g.persist.mean())
            rows.append({"segment": name, "value": str(value), "rows": len(g),
                         "persistent_rate": rate, "lift": rate / overall,
                         "split_specific_rate": float(g.split.mean()),
                         "elevated": bool(len(g) >= MIN_ROWS and rate >= 1.5 * overall)})
    table = pd.DataFrame(rows).sort_values("lift", ascending=False)
    errors_any = wrong >= 1
    result = {
        "created_at": timestamp(), "plan_item": "H-A-08", "seeds": list(SEEDS),
        "rows": len(y), "persistent_rows": int(persistent.sum()),
        "split_specific_rows": int(split_specific.sum()),
        "persistent_share_of_rows_with_any_error": float(persistent.sum() / errors_any.sum()),
        "overall_persistent_rate": overall,
        "mean_error_count_per_seed": float(wrong.sum() / len(SEEDS)),
        "persistent_mean_prob_distance_from_0.5": float(np.abs(mean_p[persistent] - 0.5).mean()),
        "split_specific_mean_prob_distance_from_0.5": float(np.abs(mean_p[split_specific] - 0.5).mean()),
        "persistent_mean_seed_std": float(std_p[persistent].mean()),
        "split_specific_mean_seed_std": float(std_p[split_specific].mean()),
        "persistent_confident_wrong_rows": int((persistent & (np.abs(mean_p - 0.5) >= 0.3)).sum()),
        "elevated_segments": table[table.elevated].to_dict(orient="records"),
        "top_segments": table.head(15).to_dict(orient="records"),
    }
    write_json(ROOT / "reports/ha08_persistent_errors.json", result)
    table.to_csv(ROOT / "reports/ha08_persistent_segments.csv", index=False)
    print({k: v for k, v in result.items() if k not in ("top_segments", "elevated_segments")})
    print(table[table.elevated][["segment", "value", "rows", "persistent_rate", "lift"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
