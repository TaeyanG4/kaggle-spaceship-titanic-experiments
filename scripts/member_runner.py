"""Fold-local runner for the 2026 member candidates (H-A-29/30/31/33).

Models: `realmlp`, `tabm`, `tabr` (pytabkit TD/D/S-D defaults), `xrfm`, `causilo`,
`tabpfn_identity` (TabPFN v3.5 + raw surname and cabin strings as categoricals).
All preprocessing (category levels, ordinal codes, one-hot, scaling, imputation) is fitted on the
outer fit fold only. Models that early-stop get an SGKF(8) slice of the fit fold as validation.
Cached by fingerprint; outputs follow the project OOF/probability/submission format.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("realmlp", "tabm", "tabr", "xrfm", "causilo", "tabpfn_identity")
EARLY_STOPPING = ("realmlp", "tabm", "tabr", "xrfm")


def _features(train, test, model):
    import extra_features

    from spaceship_titanic.versioned_features import build_versioned_features

    if model == "tabpfn_identity":
        x, tx, categorical = extra_features.build(train, test, ["surname_cat"])
        x["CabinStr"] = train.Cabin.fillna("__MISSING__").astype(str).to_numpy()
        tx["CabinStr"] = test.Cabin.fillna("__MISSING__").astype(str).to_numpy()
        categorical = [*categorical, "CabinStr"]
        for col in categorical:
            x[col], tx[col] = x[col].astype(str), tx[col].astype(str)
        return x, tx, categorical
    return build_versioned_features(train, test, "baseline")


def _encode_ordinal(x, categorical, fit_rows, others):
    import numpy as np
    from sklearn.preprocessing import OrdinalEncoder

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(x.iloc[fit_rows][categorical])
    out = []
    for frame in [x.iloc[fit_rows], *others]:
        frame = frame.copy()
        frame[categorical] = encoder.transform(frame[categorical])
        out.append(frame.to_numpy(dtype=np.float32))
    return out


def _encode_numeric(x, categorical, fit_rows, others):
    import numpy as np
    import pandas as pd

    fit = x.iloc[fit_rows]
    numeric = [c for c in x.columns if c not in categorical]
    levels = {c: sorted(fit[c].astype(str).unique()) for c in categorical}
    median = fit[numeric].astype(float).median()
    mean = fit[numeric].astype(float).fillna(median).mean()
    std = fit[numeric].astype(float).fillna(median).std().replace(0, 1)
    out = []
    for frame in [fit, *others]:
        num = (frame[numeric].astype(float).fillna(median) - mean) / std
        dummies = [pd.Series((frame[c].astype(str) == lv).astype(np.float32).to_numpy(),
                             index=frame.index, name=f"{c}={lv}")
                   for c in categorical for lv in levels[c]]
        out.append(pd.concat([num, *dummies], axis=1).to_numpy(dtype=np.float32))
    return out


def _encode_frame(x, categorical, fit_rows, others):
    import pandas as pd

    fit = x.iloc[fit_rows]
    numeric = [c for c in x.columns if c not in categorical]
    median = fit[numeric].astype(float).median()
    cats = {c: pd.CategoricalDtype(sorted(fit[c].astype(str).unique())) for c in categorical}
    out = []
    for frame in [fit, *others]:
        frame = frame.copy()
        frame[numeric] = frame[numeric].astype(float).fillna(median)
        for c in categorical:
            frame[c] = frame[c].astype(str).astype(cats[c])
        out.append(frame)
    return out


def _make_model(model, seed, threads):
    if model == "realmlp":
        from pytabkit import RealMLP_TD_Classifier
        return RealMLP_TD_Classifier(device="cuda", random_state=seed, n_cv=1, n_refit=0,
                                     verbosity=0)
    if model == "tabm":
        from pytabkit import TabM_D_Classifier
        return TabM_D_Classifier(device="cuda", random_state=seed, n_cv=1, n_refit=0, verbosity=0)
    if model == "tabr":
        # pytabkit's TabR re-imputes categoricals with SimpleImputer(fill_value=nan), which drops
        # every column under sklearn 1.9 unless empty features are kept.
        import functools

        from pytabkit import TabR_S_D_Classifier
        from pytabkit.models.alg_interfaces import tabr_interface
        from sklearn.impute import SimpleImputer
        tabr_interface.SimpleImputer = functools.partial(SimpleImputer, keep_empty_features=True)
        return TabR_S_D_Classifier(device="cuda", random_state=seed, n_cv=1, n_refit=0,
                                   verbosity=0)
    if model == "xrfm":
        from xrfm import xRFM
        return xRFM(device="cuda", random_state=seed, tuning_metric="logloss")
    if model == "causilo":
        from causilo import CausiloClassifier
        return CausiloClassifier(random_state=seed, device="cuda")
    raise ValueError(model)


def run_member(name: str, seed: int, model: str, threads: int) -> dict:
    import numpy as np
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.model_selection import StratifiedGroupKFold

    from spaceship_titanic.experiments import (
        diagnostics,
        file_hash,
        frozen_folds,
        load_data,
        save_predictions,
        timestamp,
        write_json,
    )

    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    config = {"experiment_id": name, "model": model, "random_state": seed, "n_splits": 5,
              "feature_variant": "baseline" + ("+surname+cabin_str" if model == "tabpfn_identity"
                                               else ""),
              "early_stopping": "SGKF(8) slice" if model in EARLY_STOPPING else "none"}
    code = {p: file_hash(ROOT / p) for p in [
        "scripts/member_runner.py", "scripts/extra_features.py",
        "src/spaceship_titanic/experiments.py", "src/spaceship_titanic/versioned_features.py",
        "src/spaceship_titanic/features.py"]}
    raw = {n: file_hash(ROOT / "data/raw" / n) for n in
           ["train.csv", "test.csv", "sample_submission.csv"]}
    fingerprint = hashlib.sha256(json.dumps({"config": config, "raw": raw, "code": code},
                                            sort_keys=True).encode()).hexdigest()
    metrics_path = ROOT / f"reports/{name}_metrics.json"
    if metrics_path.exists():
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        if previous["fingerprint"] == fingerprint:
            print(f"REUSE {name}: {previous['accuracy']:.6f}", flush=True)
            return previous
        raise ValueError(f"Existing experiment differs; use a new ID: {name}")

    started = time.monotonic()
    train, test, sample = load_data()
    folds = frozen_folds(train, seed, 5)
    x, tx, categorical = _features(train, test, model)
    y = train.Transported.astype(int).to_numpy()
    groups = train.PassengerId.str.split("_").str[0]
    oof, test_probability, fold_results = np.zeros(len(x)), np.zeros(len(tx)), []
    for fold in range(1, 6):
        fold_start = time.monotonic()
        fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        fit_seed = seed + fold
        if model in EARLY_STOPPING:
            a, b = next(StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=fit_seed)
                        .split(fit_idx, y[fit_idx], groups.iloc[fit_idx]))
            tr_idx, es_idx = fit_idx[a], fit_idx[b]
        else:
            tr_idx, es_idx = fit_idx, None
        if model in ("realmlp", "tabm", "tabr"):
            others = [x.iloc[valid_idx], tx] + ([x.iloc[es_idx]] if es_idx is not None else [])
            fa, fv, ft, fe = _encode_frame(x, categorical, tr_idx, others)
            est = _make_model(model, fit_seed, threads)
            est.fit(fa, y[tr_idx], X_val=fe, y_val=y[es_idx], cat_col_names=categorical)
        elif model == "xrfm":
            fa, fv, ft, fe = _encode_numeric(x, categorical, tr_idx,
                                             [x.iloc[valid_idx], tx, x.iloc[es_idx]])
            est = _make_model(model, fit_seed, threads)
            est.fit(fa, y[tr_idx], fe, y[es_idx])
        elif model == "causilo":
            fa, fv, ft = _encode_ordinal(x, categorical, tr_idx, [x.iloc[valid_idx], tx])
            est = _make_model(model, fit_seed, threads)
            est.fit(fa, y[tr_idx])
        else:  # tabpfn_identity
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
            fa, fv, ft = _encode_ordinal(x, categorical, tr_idx, [x.iloc[valid_idx], tx])
            est = TabPFNClassifier.create_default_for_version(
                ModelVersion.V3_5, device="cuda", random_state=fit_seed,
                categorical_features_indices=[x.columns.get_loc(c) for c in categorical])
            est.fit(fa, y[tr_idx])
        probability = np.asarray(est.predict_proba(fv))[:, 1]
        oof[valid_idx] = probability
        test_probability += np.asarray(est.predict_proba(ft))[:, 1] / 5
        fold_results.append({"fold": fold,
                             "accuracy": float(accuracy_score(y[valid_idx], probability >= 0.5)),
                             "seconds": time.monotonic() - fold_start})
        print(f"{name} fold {fold}: {fold_results[-1]['accuracy']:.6f} "
              f"({fold_results[-1]['seconds']:.0f}s)", flush=True)
    artifacts = save_predictions(name, train, test, sample, folds, oof, test_probability)
    result = {
        "experiment_id": name, "config": config, "fingerprint": fingerprint,
        "completed_at": timestamp(), "accuracy": float(accuracy_score(y, oof >= 0.5)),
        "log_loss": float(log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6))),
        "fold_std": float(np.std([f["accuracy"] for f in fold_results])),
        "folds": fold_results, "threads": threads, "runtime_seconds": time.monotonic() - started,
        "raw_hashes": raw, "code_hashes": code, "artifacts": artifacts,
        "segments": diagnostics(train, oof),
    }
    write_json(metrics_path, result)
    print(f"DONE {name}: accuracy={result['accuracy']:.6f}, logloss={result['log_loss']:.6f}",
          flush=True)
    return result
