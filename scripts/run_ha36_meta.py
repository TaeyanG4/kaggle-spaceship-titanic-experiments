"""H-A-36: non-linear correction on top of the D-A-027 TabPFN stacker (saved OOF only).

Arms (predeclared in plan.md), both nested: for outer fold k the meta-model is fitted on the other
four folds only and scored once on k; nothing is early-stopped or tuned on k.
- `cat_residual`: CatBoost (depth 4, lr 0.03, 300 iterations) starting from a fixed `baseline` of
  LR-stack logits, with baseline features + member logits as inputs. Training-row baselines are
  inner-nested (LR fitted on three of the four training folds, predicting the fourth), so the
  residual model never sees an in-sample stack score; fold k gets the LR fitted on all four.
- `lr_segments`: L2 LR (C=1, standardised) on member logits, their mean and std, segment flags
  (Earth, Europa, CryoSleep, solo) and member logit x segment interactions.
Compared with `ha27_stack3` (nested LR on the three logits) under the screen rule; the best passing
arm is confirmed on 7/99, and a promotion writes `outputs/submissions/ha36_<arm>.csv`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_ha27_stacker import exp, logit
from run_stack_member_screen import BASE, member_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("cat_residual", "lr_segments")


def segments(raw: pd.DataFrame, group_size: pd.Series) -> np.ndarray:
    return np.column_stack([
        raw.HomePlanet.eq("Earth").to_numpy(float),
        raw.HomePlanet.eq("Europa").to_numpy(float),
        raw.CryoSleep.astype("boolean").fillna(False).to_numpy(float),
        (group_size.to_numpy() == 1).astype(float),
    ])


def lr_design(X: np.ndarray, seg: np.ndarray) -> np.ndarray:
    inter = np.column_stack([X[:, i] * seg[:, j] for i in range(X.shape[1])
                             for j in range(seg.shape[1])])
    return np.column_stack([X, X.mean(1), X.std(1), seg, inter])


def lr_stack(X_fit, y_fit, X_apply):
    model = LogisticRegression(C=1.0, max_iter=1000).fit(X_fit, y_fit)
    return model.decision_function(X_apply)


def nested_baseline(X, y, rows, folds):
    """Out-of-sample LR-stack logits for `rows`, using only those rows' own fold structure."""
    out = np.zeros(len(rows))
    for j in np.unique(folds[rows]):
        hold = folds[rows] == j
        out[hold] = lr_stack(X[rows][~hold], y[rows][~hold], X[rows][hold])
    return out


def fit_predict(arm, X, y, seg, feats, cat_cols, fit_rows, apply_X, apply_seg, apply_feats,
                folds, seed):
    if arm == "lr_segments":
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        model.fit(lr_design(X[fit_rows], seg[fit_rows]), y[fit_rows])
        return model.predict_proba(lr_design(apply_X, apply_seg))[:, 1]
    from catboost import CatBoostClassifier, Pool

    base_fit = nested_baseline(X, y, fit_rows, folds)
    base_apply = lr_stack(X[fit_rows], y[fit_rows], apply_X)
    train_frame = feats.iloc[fit_rows].reset_index(drop=True).copy()
    apply_frame = apply_feats.reset_index(drop=True).copy()
    for i in range(X.shape[1]):
        train_frame[f"m{i}"] = X[fit_rows, i]
        apply_frame[f"m{i}"] = apply_X[:, i]
    model = CatBoostClassifier(depth=4, learning_rate=0.03, iterations=300, random_seed=seed,
                               verbose=False, thread_count=4, allow_writing_files=False)
    model.fit(Pool(train_frame, y[fit_rows], cat_features=cat_cols, baseline=base_fit))
    raw = model.predict(Pool(apply_frame, cat_features=cat_cols, baseline=base_apply),
                        prediction_type="RawFormulaVal")
    return 1 / (1 + np.exp(-raw))


def main() -> None:
    from alert import improvement_alert

    from spaceship_titanic.experiments import file_hash, load_data, timestamp, write_json
    from spaceship_titanic.screening import CONFIRM_SEEDS, SCREEN_SEEDS, confirm, screen
    from spaceship_titanic.versioned_features import build_versioned_features

    train, test, sample = load_data()
    y = train.Transported.astype(int).to_numpy()
    both_groups = pd.concat([train.PassengerId, test.PassengerId]).str[:4]
    sizes = both_groups.value_counts()
    seg = segments(train, train.PassengerId.str[:4].map(sizes))
    tseg = segments(test, test.PassengerId.str[:4].map(sizes))
    feats, tfeats, cat_cols = build_versioned_features(train, test, "baseline")
    for frame in (feats, tfeats):
        frame[cat_cols] = frame[cat_cols].astype(str)

    def nested(arm, seed):
        X, folds = member_matrix(BASE, seed, train)
        pred = np.zeros(len(y))
        for k in np.unique(folds):
            fit_rows, hold = np.flatnonzero(folds != k), folds == k
            if arm == "control":
                pred[hold] = 1 / (1 + np.exp(-lr_stack(X[fit_rows], y[fit_rows], X[hold])))
            else:
                pred[hold] = fit_predict(arm, X, y, seg, feats, cat_cols, fit_rows, X[hold],
                                         seg[hold], feats.iloc[np.flatnonzero(hold)], folds,
                                         seed + int(k))
        return float(((pred >= 0.5) == y).mean())

    control = {s: nested("control", s) for s in SCREEN_SEEDS}
    rows = []
    for arm in ARMS:
        acc = {s: nested(arm, s) for s in SCREEN_SEEDS}
        outcome = screen(control, acc)
        rows.append({"arm": arm, "accuracy": acc, **outcome})
        print(f"{arm:14s} mean_delta={outcome['mean_delta']:+.6f} "
              f"positive={outcome['positive_seeds']}/3 passes={outcome['passes']}", flush=True)
    rows.sort(key=lambda r: -r["mean_delta"])
    summary = {"completed_at": timestamp(), "plan_item": "H-A-36", "control": "ha27_stack3",
               "control_accuracy": control, "screening": rows, "confirmation": None,
               "promoted": False}
    passing = [r for r in rows if r["passes"]]
    if passing:
        arm = passing[0]["arm"]
        confirmation = {"arm": arm, **confirm({s: nested("control", s) for s in CONFIRM_SEEDS},
                                              {s: nested(arm, s) for s in CONFIRM_SEEDS})}
        summary["confirmation"] = confirmation
        print(f"CONFIRM {confirmation}", flush=True)
        if confirmation["passes"]:
            X, folds = member_matrix(BASE, 42, train)
            test_X = []
            for m in BASE:
                frame = pd.read_csv(ROOT / f"outputs/probabilities/{m}.csv")
                assert frame.PassengerId.equals(test.PassengerId)
                test_X.append(logit(frame.probability.to_numpy()))
            test_X = np.column_stack(test_X)
            probability = fit_predict(arm, X, y, seg, feats, cat_cols, np.arange(len(y)),
                                      test_X, tseg, tfeats, folds, 42)
            submission = sample.copy()
            submission["Transported"] = probability >= 0.5
            assert submission.PassengerId.equals(test.PassengerId)
            out = ROOT / f"outputs/submissions/ha36_{arm}.csv"
            submission.to_csv(out, index=False)
            pd.DataFrame({"PassengerId": test.PassengerId, "probability": probability}) \
                .to_csv(ROOT / f"outputs/probabilities/ha36_{arm}.csv", index=False)
            summary["promoted"] = True
            summary["submission"] = {"path": str(out.relative_to(ROOT)), "sha256": file_hash(out),
                                     "positive_rate": float(submission.Transported.mean())}
            improvement_alert(f"H-A-36 promoted: {arm} meta-model")
    write_json(ROOT / "reports/ha36_summary.json", summary)
    print(json.dumps({"promoted": summary["promoted"]}), flush=True)


if __name__ == "__main__":
    _ = exp
    main()
