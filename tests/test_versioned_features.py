import numpy as np
import pandas as pd

from spaceship_titanic.versioned_features import build_versioned_features


def population():
    rows = 4
    return pd.DataFrame({
        "PassengerId": ["0001_01", "0001_02", "0002_01", "0003_01"],
        "HomePlanet": ["Earth", None, "Mars", "Europa"],
        "CryoSleep": [True, False, False, True],
        "Cabin": ["B/1/P", None, "F/3/S", "B/2/P"],
        "Destination": ["TRAPPIST-1e"] * rows, "VIP": [False] * rows,
        "Age": [12.0, 30.0, np.nan, 20.0],
        "RoomService": [np.nan, 20.0, np.nan, 0.0],
        "FoodCourt": [0.0] * rows, "ShoppingMall": [0.0] * rows,
        "Spa": [0.0] * rows, "VRDeck": [0.0] * rows,
        "Name": [None, "A Smith", None, "B Smith"],
        "Transported": [True, False, True, False],
    })


def test_missing_semantics_and_rule_fill():
    raw = population()
    x, _, _ = build_versioned_features(raw.iloc[:3], raw.iloc[3:], "v1_rules")
    assert x.SurnameSize.tolist() == [0, 2, 0]
    assert x.NoSpend.tolist() == [0, 0, 0]
    assert x.SpendUnknownZero.tolist() == [1, 0, 1]
    assert x.RoomServiceMissing.tolist() == [1, 0, 1]
    assert x.loc[0, "RoomService"] == 0
    assert pd.isna(x.loc[2, "RoomService"])
    assert x.AgeMissing.tolist() == [0, 0, 1]


def test_peer_features_exclude_self_and_do_not_use_targets():
    raw = population()
    x, tx, cats = build_versioned_features(raw.iloc[:3], raw.iloc[3:], "v2")
    assert x.PeerSpendMean.iloc[:2].tolist() == [20.0, 0.0]
    assert pd.isna(x.PeerSpendMean.iloc[2])
    assert x.PeerCryoProportion.iloc[:2].tolist() == [0.0, 1.0]
    assert x.HomePlanet.iloc[1] == "Earth"
    assert x.CabinSide.iloc[1] == "P"
    assert "PlanetCryo" in cats and "Transported" not in x
    raw.Transported = ~raw.Transported
    other, other_test, _ = build_versioned_features(raw.iloc[:3], raw.iloc[3:], "v2")
    pd.testing.assert_frame_equal(x, other)
    pd.testing.assert_frame_equal(tx, other_test)
    assert raw.RoomService.isna().sum() == 2
