import pandas as pd

from spaceship_titanic.features import build_features


def test_build_features_has_expected_columns() -> None:
    train = pd.DataFrame(
        {
            "PassengerId": ["0001_01", "0001_02"],
            "HomePlanet": ["Earth", "Earth"],
            "CryoSleep": [False, True],
            "Cabin": ["B/1/P", "B/1/P"],
            "Destination": ["TRAPPIST-1e", "TRAPPIST-1e"],
            "Age": [20.0, 21.0],
            "VIP": [False, False],
            "RoomService": [0.0, 0.0],
            "FoodCourt": [10.0, 0.0],
            "ShoppingMall": [0.0, 0.0],
            "Spa": [0.0, 0.0],
            "VRDeck": [0.0, 0.0],
            "Name": ["A Smith", "B Smith"],
            "Transported": [True, False],
        }
    )
    test = train.drop(columns=["Transported"]).iloc[[0]].copy()
    test["PassengerId"] = "0001_03"

    train_x, test_x, categorical = build_features(train, test)

    assert "PassengerId" not in train_x.columns
    assert "Transported" not in train_x.columns
    assert train_x["GroupSize"].tolist() == [3, 3]
    assert test_x["GroupSize"].tolist() == [3]
    assert "CabinDeck" in categorical
    assert "CabinSide" in categorical

