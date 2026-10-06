import numpy as np
import pandas as pd
import pytest

from spaceship_titanic.experiments import _encoded_frames, frozen_folds


def test_frozen_folds_replay_and_group_isolation(tmp_path, monkeypatch):
    from spaceship_titanic import experiments

    monkeypatch.setattr(experiments, "ROOT", tmp_path)
    train = pd.DataFrame({
        "PassengerId": [f"{group:04d}_{member:02d}"
                        for group in range(16) for member in [1, 2]],
        "Transported": [bool(group % 2) for group in range(16) for _ in [1, 2]],
    })
    folds = frozen_folds(train, 42, 2)
    assert np.array_equal(folds, frozen_folds(train, 42, 2))
    assert (folds[::2] == folds[1::2]).all()


@pytest.mark.parametrize("name", ["lightgbm", "xgboost"])
def test_fold_encoders_and_model_api_handle_unseen_categories(name):
    fit = pd.DataFrame({"category": ["Earth", "Mars"] * 20,
                        "value": np.arange(40, dtype=float)})
    valid = pd.DataFrame({"category": ["Europa", "Earth"], "value": [10.0, np.nan]})
    encoded, validation, testing, encoder = _encoded_frames(
        fit, valid, valid.copy(), ["category"], name
    )
    assert "Europa" not in encoder.categories_[0]
    target = np.arange(40) % 2
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        assert validation.category.iloc[0] == -1
        model = LGBMClassifier(n_estimators=3, n_jobs=1, verbosity=-1, min_child_samples=2)
        model.fit(encoded, target, categorical_feature=["category"])
    else:
        from xgboost import XGBClassifier

        assert pd.isna(validation.category.iloc[0])
        assert encoded.category.dtype == validation.category.dtype
        model = XGBClassifier(n_estimators=3, n_jobs=1, tree_method="hist",
                              enable_categorical=True)
        model.fit(encoded, target)
    probability = model.predict_proba(testing)[:, 1]
    assert np.isfinite(probability).all() and ((0 <= probability) & (probability <= 1)).all()
