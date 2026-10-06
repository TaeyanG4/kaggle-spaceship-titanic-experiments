"""Target-free extra features for agent A's items (H-A-02/03/04). Never reads `Transported`.

`pre_*` functions edit raw train/test copies before the baseline feature builder runs;
`post_*` functions add columns after it. Train+test are used together only for target-free
population statistics (documented transductive use, allowed by agents.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def _both(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in test.columns]
    return pd.concat([train[cols], test[cols]], ignore_index=True)


def _surname(name: pd.Series) -> pd.Series:
    return name.str.strip().str.split().str[-1]


# ---- pre (raw) transforms -------------------------------------------------------------

def pre_cryo_rules(train, test):
    """Missing CryoSleep: False if any known spend > 0; True if all 5 observed zero and age >= 13."""
    out = []
    for frame in (train, test):
        frame = frame.copy()
        missing = frame.CryoSleep.isna()
        positive = frame[SPEND].gt(0).any(axis=1)
        all_zero = frame[SPEND].notna().all(axis=1) & frame[SPEND].fillna(0).eq(0).all(axis=1)
        frame.loc[missing & positive, "CryoSleep"] = False
        frame.loc[missing & ~positive & all_zero & frame.Age.ge(13), "CryoSleep"] = True
        out.append(frame)
    return tuple(out)


def pre_group_deck_fills(train, test):
    """Unanimous-group fill of Cabin and VIP; deterministic deck -> HomePlanet (A/B/C/T Europa, G Earth)."""
    both = _both(train, test)
    group = both.PassengerId.str[:4]
    fills = {}
    for col in ("Cabin", "VIP"):
        observed = both[col].dropna()
        values = observed.groupby(group[observed.index]).agg(lambda s: s.iloc[0] if s.nunique() == 1 else np.nan)
        fills[col] = values
    out = []
    for frame in (train, test):
        frame = frame.copy()
        g = frame.PassengerId.str[:4]
        for col in ("Cabin", "VIP"):
            fill = g.map(fills[col])
            frame[col] = frame[col].where(frame[col].notna(), fill)
        deck = frame.Cabin.str.split("/").str[0]
        planet = deck.map({"A": "Europa", "B": "Europa", "C": "Europa", "T": "Europa", "G": "Earth"})
        frame["HomePlanet"] = frame.HomePlanet.where(frame.HomePlanet.notna(), planet)
        out.append(frame)
    return tuple(out)


# ---- post (feature) additions --------------------------------------------------------

def post_surname_cat(x, tx, train, test):
    for frame, raw in ((x, train), (tx, test)):
        frame["SurnameCat"] = _surname(raw.Name).fillna("__MISSING__").to_numpy()
    return ["SurnameCat"]


def post_surname_repeat(x, tx, train, test):
    for frame, raw in ((x, train), (tx, test)):
        key = raw.PassengerId.str[:4] + "|" + _surname(raw.Name).fillna("")
        counts = key.map(key.value_counts())
        frame["SurnameRepeatInGroup"] = np.where(raw.Name.isna(), 0, counts).astype(int)
    return []


def post_surname_group_size(x, tx, train, test):
    both = _both(train, test)
    surname = _surname(both.Name)
    sizes = pd.DataFrame({"s": surname, "g": both.PassengerId.str[:4]}).dropna().groupby("s").g.nunique()
    for frame, raw in ((x, train), (tx, test)):
        frame["SurnameGroupSize"] = _surname(raw.Name).map(sizes).fillna(0).astype(int).to_numpy()
    return []


def post_group_id(x, tx, train, test):
    for frame, raw in ((x, train), (tx, test)):
        frame["GroupId"] = raw.PassengerId.str[:4].astype(int).to_numpy()
    return []


def _cabin_fit(train, test):
    both = _both(train, test)
    parts = both.Cabin.str.split("/", expand=True)
    frame = pd.DataFrame({"deck": parts[0], "num": pd.to_numeric(parts[1], errors="coerce"),
                          "gid": both.PassengerId.str[:4].astype(int)})
    coef = {}
    for deck, d in frame.dropna().groupby("deck"):
        if len(d) >= 2:
            coef[deck] = np.polyfit(d.gid, d.num, 1)
    return coef


def post_cabinnum_fill(x, tx, train, test):
    """Fill missing CabinNum from GroupId by a per-deck linear fit.

    A missing CabinNum always comes with a missing deck, so the deck is the row's own or, failing
    that, its groupmates' most common observed deck (train+test, target-free).
    """
    coef = _cabin_fit(train, test)
    both = _both(train, test)
    both_deck = both.Cabin.str.split("/").str[0]
    group_deck = both_deck.groupby(both.PassengerId.str[:4]).agg(
        lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
    for frame, raw in ((x, train), (tx, test)):
        deck = raw.Cabin.str.split("/").str[0]
        deck = deck.where(deck.notna(), raw.PassengerId.str[:4].map(group_deck))
        gid = raw.PassengerId.str[:4].astype(int)
        pred = [np.polyval(coef[d], g) if isinstance(d, str) and d in coef else np.nan
                for d, g in zip(deck, gid, strict=True)]
        frame["CabinNum"] = frame["CabinNum"].where(frame["CabinNum"].notna(), np.round(pred))
    return []


def post_cabin_size(x, tx, train, test):
    both = _both(train, test)
    counts = both.Cabin.value_counts()
    for frame, raw in ((x, train), (tx, test)):
        frame["CabinSize"] = raw.Cabin.map(counts).fillna(0).astype(int).to_numpy()
    return []


def post_cabin_parity(x, tx, train, test):
    for frame in (x, tx):
        frame["CabinNumParity"] = (frame["CabinNum"] % 2).to_numpy()
    return []


PRE = {"cryo_rules": pre_cryo_rules, "group_deck_fills": pre_group_deck_fills}
POST = {
    "surname_cat": post_surname_cat, "surname_repeat": post_surname_repeat,
    "surname_group_size": post_surname_group_size, "group_id": post_group_id,
    "cabinnum_fill": post_cabinnum_fill, "cabin_size": post_cabin_size,
    "cabin_parity": post_cabin_parity,
}


def build(train, test, extras: list[str]):
    """Return (x, tx, categorical) for the baseline features plus the named extras, in order."""
    from spaceship_titanic.versioned_features import build_versioned_features

    unknown = set(extras) - set(PRE) - set(POST)
    if unknown:
        raise ValueError(f"Unknown extras: {sorted(unknown)}")
    raw_train, raw_test = train.copy(), test.copy()
    for name in extras:
        if name in PRE:
            raw_train, raw_test = PRE[name](raw_train, raw_test)
    x, tx, categorical = build_versioned_features(raw_train, raw_test, "baseline")
    x, tx = x.reset_index(drop=True), tx.reset_index(drop=True)
    for name in extras:
        if name in POST:
            categorical = categorical + POST[name](x, tx, raw_train.reset_index(drop=True),
                                                   raw_test.reset_index(drop=True))
    return x, tx, categorical
