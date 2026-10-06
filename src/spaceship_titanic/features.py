from __future__ import annotations

import numpy as np
import pandas as pd

SPEND_COLUMNS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def _surname(name: pd.Series) -> pd.Series:
    return (
        name.fillna("__MISSING__")
        .astype(str)
        .str.strip()
        .str.split()
        .str[-1]
        .replace("", "__MISSING__")
    )


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    passenger_parts = out["PassengerId"].str.split("_", n=1, expand=True)
    out["PassengerGroup"] = passenger_parts[0]
    out["GroupMember"] = pd.to_numeric(passenger_parts[1], errors="coerce")

    cabin_parts = out["Cabin"].fillna("__MISSING__/__MISSING__/__MISSING__").str.split(
        "/", expand=True
    )
    out["CabinDeck"] = cabin_parts[0]
    out["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
    out["CabinSide"] = cabin_parts[2]

    out["Surname"] = _surname(out["Name"])

    out["TotalSpend"] = out[SPEND_COLUMNS].fillna(0).sum(axis=1)
    out["NoSpend"] = (out["TotalSpend"] == 0).astype(int)
    out["SpendMissingCount"] = out[SPEND_COLUMNS].isna().sum(axis=1)

    age = out["Age"]
    out["IsChild"] = (age < 13).fillna(False).astype(int)
    out["IsTeen"] = ((age >= 13) & (age < 18)).fillna(False).astype(int)
    out["IsAdult"] = (age >= 18).fillna(False).astype(int)

    return out


def build_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build leakage-safe features using labels from neither train nor test.

    GroupSize and SurnameSize are computed from the concatenated feature-only
    train+test population. This is transductive but target-free and reflects
    information visible in the official competition files.
    """

    train_x = _feature_frame(train)
    test_x = _feature_frame(test)
    combined = pd.concat([train_x, test_x], axis=0, ignore_index=True)

    group_counts = combined["PassengerGroup"].value_counts(dropna=False)
    surname_counts = combined["Surname"].value_counts(dropna=False)

    for frame in (train_x, test_x):
        frame["GroupSize"] = frame["PassengerGroup"].map(group_counts).astype(int)
        frame["SurnameSize"] = frame["Surname"].map(surname_counts).astype(int)
        frame["IsAlone"] = (frame["GroupSize"] == 1).astype(int)

    drop_columns = ["Transported", "Name", "Cabin", "PassengerGroup", "Surname"]
    train_x = train_x.drop(columns=[c for c in drop_columns if c in train_x.columns])
    test_x = test_x.drop(columns=[c for c in drop_columns if c in test_x.columns])

    categorical = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "CabinDeck",
        "CabinSide",
    ]
    for frame in (train_x, test_x):
        for col in categorical:
            frame[col] = frame[col].astype("string").fillna("__MISSING__").astype(str)

    # PassengerId itself is not used as a predictive feature.
    train_x = train_x.drop(columns=["PassengerId"])
    test_x = test_x.drop(columns=["PassengerId"])

    # Defensive conversion for nullable integer-like values.
    for frame in (train_x, test_x):
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)

    return train_x, test_x, categorical
