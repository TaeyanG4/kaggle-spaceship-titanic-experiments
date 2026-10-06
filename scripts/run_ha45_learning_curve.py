"""H-A-45 (re-review of D-A-010): group-preserving learning curve of frozen TabPFN v3.5.

Outer frozen SGKF folds stay fixed. Inside each outer fit fold, a StratifiedGroupKFold(5) on travel
groups (seeded per fold) splits the fit rows into five parts; the 40/60/80/100% arms keep the first
2/3/4/5 parts as TabPFN context (fit-fold ordinal encoding, as `tabpfn_runner.py`). The unchanged
outer fold is scored once per arm. Predeclared reading: a materially positive 80->100% slope
(>= +0.002 mean over seeds 42/123/2026) means the plateau is data-limited; a flat slope supports
D-A-010's signal-limited reading. Writes `reports/ha45_learning_curve.json`.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OrdinalEncoder

FRACTIONS = {"40": 2, "60": 3, "80": 4, "100": 5}


def main() -> None:
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    from spaceship_titanic.experiments import ROOT, frozen_folds, load_data, timestamp, write_json
    from spaceship_titanic.versioned_features import build_versioned_features

    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    train, test, _ = load_data()
    x, _, categorical = build_versioned_features(train, test, "baseline")
    y = train.Transported.astype(int).to_numpy()
    groups = train.PassengerId.str[:4].to_numpy()
    cat_idx = [x.columns.get_loc(c) for c in categorical]
    result = {"completed_at": None, "plan_item": "H-A-45", "seeds": {}}
    for seed in (42, 123, 2026):
        folds = frozen_folds(train, seed, 5)
        correct = {f: 0 for f in FRACTIONS}
        for fold in range(1, 6):
            fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            parts = [fit_idx[p] for _, p in StratifiedGroupKFold(
                n_splits=5, shuffle=True, random_state=seed + fold).split(
                    fit_idx, y[fit_idx], groups[fit_idx])]
            for name, k in FRACTIONS.items():
                rows = np.concatenate(parts[:k])
                encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                encoder.fit(x.iloc[rows][categorical])
                frames = []
                for r in (rows, valid_idx):
                    frame = x.iloc[r].copy()
                    frame[categorical] = encoder.transform(frame[categorical])
                    frames.append(frame.to_numpy(dtype=np.float32))
                model = TabPFNClassifier.create_default_for_version(
                    ModelVersion.V3_5, categorical_features_indices=cat_idx, device="cuda",
                    random_state=seed + fold)
                model.fit(frames[0], y[rows])
                pred = model.predict_proba(frames[1])[:, 1] >= 0.5
                correct[name] += int((pred == y[valid_idx]).sum())
        acc = {f: correct[f] / len(y) for f in FRACTIONS}
        result["seeds"][seed] = acc
        print(seed, {f: round(a, 6) for f, a in acc.items()}, flush=True)
    slopes = [r["100"] - r["80"] for r in result["seeds"].values()]
    result["slope_80_100"] = {"per_seed": slopes, "mean": float(np.mean(slopes))}
    result["data_limited"] = bool(np.mean(slopes) >= 0.002)
    result["completed_at"] = timestamp()
    write_json(ROOT / "reports/ha45_learning_curve.json", result)
    print(result["slope_80_100"], "data_limited", result["data_limited"], flush=True)


if __name__ == "__main__":
    main()
