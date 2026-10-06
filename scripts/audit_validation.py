"""Reproducible local shift, missing-semantics, and existing OOF diagnostics."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from spaceship_titanic.experiments import (
    ROOT,
    diagnostics,
    frozen_folds,
    load_data,
    timestamp,
    write_json,
)
from spaceship_titanic.features import SPEND_COLUMNS, build_features


def total_variation(left: pd.Series, right: pd.Series) -> float:
    a, b = left.value_counts(normalize=True), right.value_counts(normalize=True)
    index = a.index.union(b.index)
    return float((a.reindex(index, fill_value=0) - b.reindex(index, fill_value=0)).abs().sum() / 2)


def main() -> None:
    train, test, _ = load_data()
    x, tx, _ = build_features(train.drop(columns="Transported"), test)
    oof = pd.read_csv(ROOT / "outputs/oof/baseline_catboost_oof.csv")
    assert oof.PassengerId.equals(train.PassengerId)
    probability = oof.probability.to_numpy()
    categorical = ["HomePlanet", "CryoSleep", "Destination", "VIP", "CabinDeck",
                   "CabinSide", "NoSpend", "GroupSize"]
    semantics = {}
    for name, frame in [("train", train), ("test", test)]:
        known_positive = frame[SPEND_COLUMNS].gt(0).any(axis=1)
        semantics[name] = {
            "zero_observed_with_missing": int((frame[SPEND_COLUMNS].fillna(0).sum(axis=1).eq(0)
                                               & frame[SPEND_COLUMNS].isna().any(axis=1)).sum()),
            "cryo_true_positive_spend": int((frame.CryoSleep.eq(True) & known_positive).sum()),
            "under13_positive_spend": int((frame.Age.lt(13) & known_positive).sum()),
            "missing_name_count": int(frame.Name.isna().sum()),
        }
    fold = frozen_folds(train, 42)
    result = {
        "created_at": timestamp(), "validation": "5-fold SGKF seed 42, PassengerId group",
        "train_test_group_overlap": 0,
        "baseline_accuracy": float(accuracy_score(train.Transported, probability >= 0.5)),
        "baseline_log_loss": float(log_loss(train.Transported, probability)),
        "baseline_segments": diagnostics(train, probability),
        "fold_scores": [float(accuracy_score(train.Transported[fold == f],
                        probability[fold == f] >= 0.5)) for f in range(1, 6)],
        "missing_rates": pd.DataFrame({"train": train.drop(columns="Transported").isna().mean(),
                                       "test": test.isna().mean()}).to_dict(orient="index"),
        "categorical_total_variation": {
            column: total_variation(x[column].astype(str), tx[column].astype(str))
            for column in categorical
        },
        "joint_total_variation": {
            "HomePlanet_CryoSleep": total_variation(x.HomePlanet + "|" + x.CryoSleep,
                                                     tx.HomePlanet + "|" + tx.CryoSleep),
            "CabinDeck_CabinSide": total_variation(x.CabinDeck + "|" + x.CabinSide,
                                                   tx.CabinDeck + "|" + tx.CabinSide),
        },
        "numeric_quantiles": {
            column: {"train": x[column].quantile([0.1, 0.5, 0.9]).tolist(),
                     "test": tx[column].quantile([0.1, 0.5, 0.9]).tolist()}
            for column in ["Age", "TotalSpend", "CabinNum"]
        },
        "spending_semantics": semantics,
        "uncertain_045_055": int(np.sum((probability >= 0.45) & (probability <= 0.55))),
        "limits": "Marginal and two joint distributions only; public LB-gap cause remains unknown.",
    }
    path = ROOT / "reports/v1_v2_validation.json"
    write_json(path, result)
    print(f"Validation audit saved: {path}")
    print(f"Joint total variation: {result['joint_total_variation']}")


if __name__ == "__main__":
    main()
