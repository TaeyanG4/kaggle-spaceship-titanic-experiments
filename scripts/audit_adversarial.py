"""H-D-04: group-split adversarial validation (train vs test), diagnostic only.

Features are the baseline feature set (target-free; PassengerId and group id excluded).
A 5-fold StratifiedGroupKFold by PassengerId group predicts is_test. Predeclared reading:
AUC < 0.55 means no practically exploitable shift; the top features show where any shift lives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from spaceship_titanic.experiments import ROOT, load_data, timestamp, write_json
from spaceship_titanic.versioned_features import build_versioned_features


def main(threads: int = 4) -> None:
    train, test, _ = load_data()
    x, tx, categorical = build_versioned_features(train, test, "baseline")
    data = pd.concat([x, tx], ignore_index=True)
    is_test = np.r_[np.zeros(len(x), dtype=int), np.ones(len(tx), dtype=int)]
    groups = pd.concat([train.PassengerId, test.PassengerId], ignore_index=True).str[:4]
    oof = np.zeros(len(data))
    importance = pd.Series(0.0, index=data.columns)
    for fit, valid in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
            data, is_test, groups):
        model = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, random_seed=42,
                                   thread_count=threads, verbose=False, allow_writing_files=False)
        model.fit(data.iloc[fit], is_test[fit], cat_features=categorical)
        oof[valid] = model.predict_proba(data.iloc[valid], thread_count=threads)[:, 1]
        importance += pd.Series(model.get_feature_importance(), index=data.columns) / 5
    auc = float(roc_auc_score(is_test, oof))
    train_scores = oof[: len(x)]
    result = {
        "created_at": timestamp(), "plan_item": "H-D-04", "auc": auc,
        "reading": "no practically exploitable shift" if auc < 0.55 else "shift present",
        "top_features": importance.sort_values(ascending=False).head(10).round(3).to_dict(),
        "train_test_likeness_quantiles": np.quantile(train_scores, [0.5, 0.9, 0.99]).tolist(),
        "note": "Diagnostic only; no reweighting or selection was applied.",
    }
    write_json(ROOT / "reports/adversarial_validation.json", result)
    print(result)


if __name__ == "__main__":
    main()
