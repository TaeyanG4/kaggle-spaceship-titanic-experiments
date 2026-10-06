"""Versioned target-free features; the original baseline remains reproducible."""

from __future__ import annotations

import numpy as np
import pandas as pd

from spaceship_titanic.features import SPEND_COLUMNS, _surname, build_features

VARIANTS = {
    "baseline": set(),
    "v1_name": {"name"},
    "v1_spend": {"spend"},
    "v1": {"name", "spend"},
    "v1_rules": {"name", "spend", "rules"},
    "v2_spending": {"name", "spend", "composition"},
    "v2_group": {"name", "spend", "peers"},
    "v2": {"name", "spend", "composition", "peers"},
}


def _peer_mean(values: pd.Series, groups: pd.Series) -> pd.Series:
    """Observed peer mean, excluding this row; no peers yields NaN."""
    count = values.notna().astype(int).groupby(groups).transform("sum")
    total = values.fillna(0).groupby(groups).transform("sum")
    denominator = count - values.notna().astype(int)
    return (total - values.fillna(0)) / denominator.replace(0, np.nan)


def build_versioned_features(
    train: pd.DataFrame, test: pd.DataFrame, variant: str = "v1"
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Use only official feature columns, including the unlabeled validation population.

    Counts, peer summaries, unanimous peer fills and deck-relative cabin ranks
    use the combined official train/test feature population. SGKF keeps each
    PassengerId group intact, matching complete unseen test groups. Population
    summaries are transductive but never consume Transported. Learned model
    encoders are fitted separately on each training fold by the experiment runner.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown feature variant: {variant}")
    options = VARIANTS[variant]
    train_raw = train.drop(columns="Transported", errors="ignore")
    test_raw = test.drop(columns="Transported", errors="ignore")
    a, b, categorical = build_features(train_raw, test_raw)
    if not options:
        return a, b, categorical
    raw = pd.concat([train_raw, test_raw], ignore_index=True)
    out = pd.concat([a, b], ignore_index=True)
    groups = raw.PassengerId.str.split("_").str[0]
    observed = raw[SPEND_COLUMNS]
    if "name" in options:
        surname = _surname(raw.Name)
        missing = raw.Name.isna() | raw.Name.fillna("").str.strip().eq("")
        counts = surname[~missing].value_counts()
        out["NameMissing"] = missing.astype(int)
        out["SurnameSize"] = surname.map(counts).fillna(0).astype(int)
        out.loc[missing, "SurnameSize"] = 0
    if "spend" in options:
        complete = observed.notna().all(axis=1)
        zero = observed.fillna(0).sum(axis=1).eq(0)
        out["ObservedNoSpend"] = zero.astype(int)
        out["NoSpend"] = (zero & complete).astype(int)
        out["SpendUnknownZero"] = (zero & ~complete).astype(int)
        out["SpendObservedCount"] = observed.notna().sum(axis=1)
        out["AgeMissing"] = raw.Age.isna().astype(int)
        for column in SPEND_COLUMNS:
            out[f"{column}Missing"] = observed[column].isna().astype(int)
    if "rules" in options:
        eligible = raw.CryoSleep.eq(True) | raw.Age.lt(13)
        out["SpendRuleFilledCount"] = (
            observed.isna().sum(axis=1) * eligible.astype(int)
        )
        for column in SPEND_COLUMNS:
            mask = eligible & raw[column].isna()
            out.loc[mask, column] = 0.0
    if "peers" in options:
        out["NoPeer"] = out.GroupSize.eq(1).astype(int)
        out["PeerCount"] = out.GroupSize - 1
        out["PeerSpendMean"] = _peer_mean(
            observed.sum(axis=1, min_count=1), groups
        )
        out["PeerAgeMean"] = _peer_mean(raw.Age, groups)
        cryo = raw.CryoSleep.map({True: 1.0, False: 0.0})
        out["PeerCryoProportion"] = _peer_mean(cryo, groups)
        for column in ["HomePlanet", "CabinSide"]:
            values = raw.HomePlanet if column == "HomePlanet" else out.CabinSide
            values = values.replace("__MISSING__", np.nan)
            unique = values.groupby(groups).transform("nunique")
            consensus = values.groupby(groups).transform("first")
            mask = values.isna() & unique.eq(1)
            out[f"{column}GroupFilled"] = mask.astype(int)
            out.loc[mask, column] = consensus[mask]
        out["CabinPosition"] = out.groupby("CabinDeck").CabinNum.rank(pct=True)
    if "composition" in options:
        out["PositiveServiceCount"] = observed.gt(0).sum(axis=1)
        out["LuxurySpend"] = observed[["Spa", "VRDeck"]].sum(axis=1, min_count=1)
        out["FoodShoppingSpend"] = observed[["FoodCourt", "ShoppingMall"]].sum(
            axis=1, min_count=1
        )
        denominator = out.TotalSpend.replace(0, np.nan)
        for column in SPEND_COLUMNS:
            share = observed[column] / denominator
            out[f"{column}Share"] = share.mask(
                out.TotalSpend.eq(0) & observed[column].notna(), 0
            )
        for name, left, right in [
            ("PlanetCryo", "HomePlanet", "CryoSleep"),
            ("DeckSide", "CabinDeck", "CabinSide"),
        ]:
            out[name] = out[left].astype(str) + "|" + out[right].astype(str)
            categorical.append(name)
    for column in categorical:
        out[column] = out[column].fillna("__MISSING__").astype(str)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    a = out.iloc[: len(train)].copy()
    b = out.iloc[len(train) :].copy()
    a.index, b.index = train.index, test.index
    return a, b, categorical
