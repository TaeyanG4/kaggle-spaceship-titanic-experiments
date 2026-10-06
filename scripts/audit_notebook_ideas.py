"""Target-free coverage checks for ideas found in the public-notebook audit (H-A plan items).

Reads only official train/test feature columns (never `Transported`) and prints how many rows each
candidate rule would touch, plus structural facts behind the location and surname hypotheses.
"""

import numpy as np
import pandas as pd

from spaceship_titanic.experiments import load_data
from spaceship_titanic.features import SPEND_COLUMNS


def report(name: str, mask: pd.Series, n_train: int) -> None:
    print(f"{name}: train {int(mask[:n_train].sum())}, test {int(mask[n_train:].sum())}")


def main() -> None:
    train, test, _ = load_data()
    n_train = len(train)
    df = pd.concat([train.drop(columns="Transported"), test], ignore_index=True)
    group = df.PassengerId.str[:4]
    group_size = group.map(group.value_counts())
    spend = df[SPEND_COLUMNS]
    observed_zero = spend.fillna(0).sum(axis=1).eq(0)
    complete = spend.notna().all(axis=1)

    print("## Reverse CryoSleep rule")
    report("CryoSleep missing", df.CryoSleep.isna(), n_train)
    report("  and complete all-zero spend", df.CryoSleep.isna() & observed_zero & complete, n_train)
    for label, extra in [("all ages", True), ("age >= 13", df.Age.ge(13))]:
        m = df.CryoSleep.notna() & observed_zero & complete & extra
        rate = df.loc[m, "CryoSleep"].astype(bool).mean()
        print(f"  P(CryoSleep | observed, complete zero spend, {label}) = {rate:.4f} (n={int(m.sum())})")

    print("## Group fills")
    for col in ["Cabin", "Destination", "VIP", "HomePlanet"]:
        unique = df[col].groupby(group).transform("nunique")
        unanimous_share = (unique[(group_size > 1) & df[col].notna()] == 1).mean()
        report(f"{col} missing", df[col].isna(), n_train)
        report("  fillable from a unanimous group", df[col].isna() & unique.eq(1), n_train)
        print(f"  multi-member rows in a unanimous group: {unanimous_share:.3f}")

    print("## Location")
    cabin = df.Cabin.str.split("/", expand=True)
    deck, number = cabin[0], pd.to_numeric(cabin[1])
    group_id = pd.to_numeric(group)
    for d in sorted(deck.dropna().unique()):
        m = deck.eq(d) & number.notna()
        if m.sum() > 30:
            corr = np.corrcoef(group_id[m], number[m])[0, 1]
            resid = number[m] - np.polyval(np.polyfit(group_id[m], number[m], 1), group_id[m])
            print(f"deck {d}: n={int(m.sum())} corr(GroupId, CabinNum)={corr:.3f} "
                  f"linear-fit MAE={resid.abs().mean():.1f}")
    occupancy = df.Cabin.map(df.Cabin.value_counts())
    print("cabin occupancy (non-missing rows):", occupancy.value_counts().sort_index().to_dict())
    groups_per_cabin = df.dropna(subset=["Cabin"]).groupby("Cabin").PassengerId.agg(
        lambda s: s.str[:4].nunique()
    )
    print(f"cabins shared by >1 PassengerId group: {int((groups_per_cabin > 1).sum())} "
          f"of {len(groups_per_cabin)}")

    print("## Surname")
    surname = df.Name.str.split().str[-1]
    train_surnames = set(surname[:n_train].dropna())
    test_named = surname[n_train:].dropna()
    print(f"test rows with a surname seen in train: {test_named.isin(train_surnames).mean():.3f}")
    s_train = surname[:n_train].dropna()
    spans = s_train.groupby(s_train).transform(lambda s: group[s.index].nunique())
    print(f"named train rows whose surname spans >1 train group: {(spans > 1).mean():.3f}")


if __name__ == "__main__":
    main()
