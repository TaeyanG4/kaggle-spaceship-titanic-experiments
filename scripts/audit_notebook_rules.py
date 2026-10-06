"""Label-aware diagnostics for rules taken from public notebooks (junaid512 audit, D-A-003).

No model is trained or selected here. Train labels are read only to measure fixed, predeclared
rules: how far junaid512's group-label propagation reaches test rows, how much groupmate labels
agree, and what its CryoSleep+NoSpend hard lock does to the champion's honest OOF predictions.
"""

import pandas as pd

from spaceship_titanic.experiments import ROOT, load_data
from spaceship_titanic.features import SPEND_COLUMNS

CHAMPION_OOF = {
    42: "hm10_innercv_logloss.csv",
    123: "hm10_innercv_logloss_seed123.csv",
    2026: "hm10_innercv_logloss_seed2026.csv",
    7: "hm10_innercv_logloss_seed7.csv",
    99: "hm10_innercv_logloss_seed99.csv",
}


def cryo_no_spend(df: pd.DataFrame) -> pd.Series:
    """junaid512 R1: CryoSleep (observed, or inferred from zero known spend) and no known spend."""
    spend = df[SPEND_COLUMNS].fillna(0).sum(axis=1)
    cryo = df.CryoSleep.astype("object").copy()
    cryo[cryo.isna() & spend.gt(0)] = False
    cryo[cryo.isna() & spend.eq(0)] = True
    return cryo.astype(bool) & spend.eq(0)


def main() -> None:
    train, test, _ = load_data()
    y = train.Transported.astype(int)
    group = train.PassengerId.str[:4]
    test_group = test.PassengerId.str[:4]

    print("## Group label propagation")
    reached = int(test_group.isin(set(group)).sum())
    print(f"test rows whose group has a labelled train member: {reached} of {len(test)}")
    total = y.groupby(group).transform("sum")
    count = y.groupby(group).transform("count")
    multi = count > 1
    loo = (total - y) / (count - 1).clip(lower=1)
    accuracy = ((loo[multi] >= 0.5).astype(int) == y[multi]).mean()
    print(f"multi-member train rows predicted by groupmates' LOO mean >= 0.5: {accuracy:.4f} "
          f"(n={int(multi.sum())})")
    stats = y.groupby(group).agg(["mean", "count"])
    stats = stats[stats["count"] > 1]
    unanimous = (stats["mean"].isin([0, 1])).mean()
    print(f"multi-member train groups with a unanimous label: {unanimous:.3f} (n={len(stats)})")

    print("## Hard lock R1 (CryoSleep & NoSpend -> Transported)")
    mask = cryo_no_spend(train)
    print(f"R1 rows: train {int(mask.sum())}, test {int(cryo_no_spend(test).sum())}; "
          f"train precision {y[mask].mean():.4f}")
    for seed, name in CHAMPION_OOF.items():
        oof = pd.read_csv(ROOT / "outputs/oof" / name)
        assert oof.PassengerId.equals(train.PassengerId)
        pred = (oof.probability >= 0.5).astype(int)
        locked = pred.mask(mask, 1)
        base, after = (pred == y).mean(), (locked == y).mean()
        print(f"seed {seed}: champion {base:.6f} -> locked {after:.6f} ({after - base:+.6f}; "
              f"{int((pred[mask] == 0).sum())} rows flipped)")


if __name__ == "__main__":
    main()
