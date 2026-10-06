"""The Kaggle notebook's feature and fold code must match the project code exactly."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "kaggle_notebook/spaceship_titanic_tabpfn_stack.ipynb"

pytestmark = pytest.mark.skipif(not (ROOT / "data/raw/train.csv").exists(),
                                reason="official competition data not available")


def notebook_namespace(monkeypatch) -> dict:
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("TABPFN_TOKEN", "unused-in-this-test")
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    sources = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
               for c in cells if c["cell_type"] == "code"]
    namespace: dict = {}
    for source in sources[0:3]:  # setup, features, folds
        exec(source, namespace)  # noqa: S102 - executing our own generated notebook
    return namespace


def test_features_and_folds_match_project(monkeypatch):
    from sklearn.model_selection import StratifiedGroupKFold

    from spaceship_titanic.experiments import frozen_folds, load_data
    from spaceship_titanic.versioned_features import build_versioned_features

    ns = notebook_namespace(monkeypatch)
    train, test, _ = load_data()
    x, tx, categorical = build_versioned_features(train, test, "baseline")
    pd.testing.assert_frame_equal(ns["X"].reset_index(drop=True), x.reset_index(drop=True))
    pd.testing.assert_frame_equal(ns["X_test"].reset_index(drop=True), tx.reset_index(drop=True))
    assert ns["CATEGORICAL"] == categorical
    assert np.array_equal(ns["folds"], frozen_folds(train, 42, 5))

    y = train.Transported.astype(int).to_numpy()
    groups = train.PassengerId.str.split("_").str[0]
    for fold in range(1, 6):
        fit_idx, _, tr_idx, es_idx, _ = ns["fold_parts"](fold)
        a, b = next(StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=42 + fold)
                    .split(fit_idx, y[fit_idx], groups.iloc[fit_idx]))
        assert np.array_equal(tr_idx, fit_idx[a]) and np.array_equal(es_idx, fit_idx[b])
