from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def main() -> None:
    train = pd.read_csv(RAW / "train.csv")
    test = pd.read_csv(RAW / "test.csv")

    train_group = train["PassengerId"].str.split("_").str[0]
    test_group = test["PassengerId"].str.split("_").str[0]
    overlap = set(train_group) & set(test_group)

    print(f"train shape: {train.shape}")
    print(f"test shape: {test.shape}")
    print(f"train unique groups: {train_group.nunique()}")
    print(f"test unique groups: {test_group.nunique()}")
    print(f"train/test overlapping groups: {len(overlap)}")
    print(f"largest train group: {train_group.value_counts().max()}")
    print()
    print("target balance:")
    print(train["Transported"].value_counts(normalize=False))
    print()
    print("missing values:")
    print(train.isna().sum().sort_values(ascending=False))


if __name__ == "__main__":
    main()

