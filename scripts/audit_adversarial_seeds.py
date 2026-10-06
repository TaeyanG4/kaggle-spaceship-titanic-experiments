"""H-A-44 (re-review of D-D-031): multi-seed group-split adversarial validation with a chance reference.

Target-free: predicts is_test from the baseline features (PassengerId and group id excluded) with
the same CatBoost settings as `audit_adversarial.py`, under 5-fold StratifiedGroupKFold by travel
group on seeds 42/123/2026. Row-level OOF is saved to `outputs/oof/ha44_adversarial_seed<s>.csv`.
Chance reference: the same pipeline with is_test permuted at the group level (whole groups keep one
label), seed-matched. Predeclared reading: no exploitable shift if every seed's AUC < 0.55 and the
mean is within 0.02 of the permutation mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from spaceship_titanic.experiments import ROOT, load_data, timestamp, write_json
from spaceship_titanic.versioned_features import build_versioned_features


def adversarial_oof(data, label, groups, categorical, seed, threads):
    oof = np.zeros(len(data))
    importance = pd.Series(0.0, index=data.columns)
    for fit, valid in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
            data, label, groups):
        model = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, random_seed=seed,
                                   thread_count=threads, verbose=False, allow_writing_files=False)
        model.fit(data.iloc[fit], label[fit], cat_features=categorical)
        oof[valid] = model.predict_proba(data.iloc[valid], thread_count=threads)[:, 1]
        importance += pd.Series(model.get_feature_importance(), index=data.columns) / 5
    return oof, importance


def main(threads: int = 4) -> None:
    train, test, _ = load_data()
    x, tx, categorical = build_versioned_features(train, test, "baseline")
    data = pd.concat([x, tx], ignore_index=True)
    is_test = np.r_[np.zeros(len(x), dtype=int), np.ones(len(tx), dtype=int)]
    ids = pd.concat([train.PassengerId, test.PassengerId], ignore_index=True)
    groups = ids.str[:4]
    group_label = pd.Series(is_test).groupby(groups).first()
    result = {"completed_at": timestamp(), "plan_item": "H-A-44", "seeds": {}}
    for seed in (42, 123, 2026):
        oof, importance = adversarial_oof(data, is_test, groups, categorical, seed, threads)
        auc = float(roc_auc_score(is_test, oof))
        rng = np.random.default_rng(seed)
        permuted = pd.Series(rng.permutation(group_label.to_numpy()), index=group_label.index)
        perm_label = groups.map(permuted).to_numpy()
        perm_oof, _ = adversarial_oof(data, perm_label, groups, categorical, seed, threads)
        perm_auc = float(roc_auc_score(perm_label, perm_oof))
        pd.DataFrame({"PassengerId": ids, "is_test": is_test, "probability": oof}).to_csv(
            ROOT / f"outputs/oof/ha44_adversarial_seed{seed}.csv", index=False)
        result["seeds"][seed] = {
            "auc": auc, "permutation_auc": perm_auc,
            "train_test_likeness_p50_p99": [float(np.median(oof[: len(x)])),
                                            float(np.quantile(oof[: len(x)], 0.99))],
            "top_features": importance.sort_values(ascending=False).head(5).round(3).to_dict()}
        print(seed, f"auc={auc:.4f} permutation={perm_auc:.4f}", flush=True)
    aucs = [r["auc"] for r in result["seeds"].values()]
    perms = [r["permutation_auc"] for r in result["seeds"].values()]
    result["mean_auc"], result["mean_permutation_auc"] = float(np.mean(aucs)), float(np.mean(perms))
    result["no_exploitable_shift"] = bool(max(aucs) < 0.55
                                          and abs(np.mean(aucs) - np.mean(perms)) <= 0.02)
    write_json(ROOT / "reports/ha44_adversarial_seeds.json", result)
    print({k: result[k] for k in ("mean_auc", "mean_permutation_auc", "no_exploitable_shift")},
          flush=True)


if __name__ == "__main__":
    main()
