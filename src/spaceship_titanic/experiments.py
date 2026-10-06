from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits

from spaceship_titanic.versioned_features import build_versioned_features

ROOT = Path(__file__).resolve().parents[2]


def timestamp() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = ROOT / "data/raw"
    train, test, sample = [
        pd.read_csv(raw / name)
        for name in ["train.csv", "test.csv", "sample_submission.csv"]
    ]
    assert train.PassengerId.is_unique and test.PassengerId.is_unique
    assert test.PassengerId.equals(sample.PassengerId)
    assert not set(train.PassengerId.str.split("_").str[0]) & set(
        test.PassengerId.str.split("_").str[0]
    )
    return train, test, sample


def frozen_folds(train: pd.DataFrame, seed: int, n_splits: int = 5) -> np.ndarray:
    groups = train.PassengerId.str.split("_").str[0]
    folds = np.zeros(len(train), dtype=int)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (fit, valid) in enumerate(
        splitter.split(train, train.Transported.astype(int), groups), 1
    ):
        assert not set(groups.iloc[fit]) & set(groups.iloc[valid])
        folds[valid] = fold
    path = ROOT / f"data/processed/sgkf_{n_splits}_seed{seed}.csv"
    expected = pd.DataFrame(
        {"PassengerId": train.PassengerId, "group": groups, "fold": folds}
    )
    if path.exists():
        saved = pd.read_csv(path, dtype={"PassengerId": str, "group": str})
        pd.testing.assert_frame_equal(saved, expected.reset_index(drop=True), check_dtype=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        expected.to_csv(path, index=False)
    return folds


def diagnostics(train: pd.DataFrame, probability: np.ndarray) -> dict:
    errors = (probability >= 0.5) != train.Transported.astype(bool).to_numpy()
    groups = train.PassengerId.str.split("_").str[0]
    segments = {
        "HomePlanet": train.HomePlanet.fillna("missing"),
        "CryoSleep": train.CryoSleep.astype("string").fillna("missing"),
        "GroupSize": groups.map(groups.value_counts()).clip(upper=4).astype(str),
        "SpendMissing": train[
            ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        ].isna().any(axis=1).astype(str),
    }
    return {
        name: pd.DataFrame({"segment": values, "error": errors})
        .groupby("segment")
        .agg(rows=("error", "size"), errors=("error", "sum"), error_rate=("error", "mean"))
        .reset_index()
        .to_dict(orient="records")
        for name, values in segments.items()
    }


def paired_group_bootstrap(
    train: pd.DataFrame, reference: np.ndarray, challenger: np.ndarray,
    seed: int = 42, draws: int = 2000,
) -> dict:
    target = train.Transported.astype(bool).to_numpy()
    delta = ((challenger >= 0.5) == target).astype(int) - (
        (reference >= 0.5) == target
    ).astype(int)
    groups = train.PassengerId.str.split("_").str[0]
    grouped = pd.DataFrame({"group": groups, "delta": delta}).groupby("group").delta
    total, size = grouped.sum().to_numpy(), grouped.size().to_numpy()
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        selected = rng.integers(0, len(size), len(size))
        samples.append(float(total[selected].sum() / size[selected].sum()))
    return {
        "accuracy_delta": float(delta.mean()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "positive_fraction": float(np.mean(np.asarray(samples) > 0)),
        "draws": draws,
        "note": "Diagnostic conditional on fitted OOF models; not independent retraining uncertainty.",
    }


def _encoded_frames(fit, valid, test, categorical, model_name):
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(fit[categorical])
    frames = []
    for original in [fit, valid, test]:
        frame = original.copy()
        frame[categorical] = encoder.transform(original[categorical]).astype(int)
        if model_name == "xgboost":
            for index, column in enumerate(categorical):
                values = frame[column].replace(-1, np.nan)
                frame[column] = pd.Categorical(
                    values, categories=range(len(encoder.categories_[index]))
                )
        frames.append(frame)
    return (*frames, encoder)


def save_predictions(
    experiment_id: str, train: pd.DataFrame, test: pd.DataFrame,
    sample: pd.DataFrame, folds: np.ndarray, oof: np.ndarray, test_probability: np.ndarray,
) -> dict:
    paths = {
        "oof": ROOT / f"outputs/oof/{experiment_id}.csv",
        "test_probability": ROOT / f"outputs/probabilities/{experiment_id}.csv",
        "submission": ROOT / f"outputs/submissions/{experiment_id}.csv",
    }
    assert np.isfinite(oof).all() and np.isfinite(test_probability).all()
    assert ((0 <= oof) & (oof <= 1)).all()
    assert ((0 <= test_probability) & (test_probability <= 1)).all()
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "PassengerId": train.PassengerId,
        "Transported": train.Transported.astype(bool),
        "fold": folds,
        "probability": oof,
        "prediction": oof >= 0.5,
    }).to_csv(paths["oof"], index=False)
    pd.DataFrame({"PassengerId": test.PassengerId, "probability": test_probability}).to_csv(
        paths["test_probability"], index=False
    )
    submission = sample.copy()
    submission.Transported = test_probability >= 0.5
    assert submission.columns.tolist() == ["PassengerId", "Transported"]
    assert submission.PassengerId.equals(test.PassengerId)
    assert submission.Transported.notna().all() and submission.Transported.dtype == bool
    submission.to_csv(paths["submission"], index=False)
    return {name: str(path.relative_to(ROOT)) for name, path in paths.items()}


def run_experiment(config: dict, threads: int = 2, progress=None) -> dict:
    experiment_id = config["experiment_id"]
    if not experiment_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Experiment ID must contain only letters, digits, _ or -")
    started = time.monotonic()
    train, test, sample = load_data()
    seed = config.get("random_state", 42)
    n_splits = config.get("n_splits", 5)
    folds = frozen_folds(train, seed, n_splits)
    raw_hashes = {
        name: file_hash(ROOT / "data/raw" / name)
        for name in ["train.csv", "test.csv", "sample_submission.csv"]
    }
    code_hashes = {
        name: file_hash(Path(__file__).resolve().parent / name)
        for name in ["features.py", "versioned_features.py", "experiments.py"]
    }
    fingerprint = hashlib.sha256(json.dumps(
        {"config": config, "raw": raw_hashes, "code": code_hashes}, sort_keys=True
    ).encode()).hexdigest()
    metrics_path = ROOT / f"reports/{experiment_id}_metrics.json"
    if metrics_path.exists():
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != fingerprint:
            raise ValueError(f"Existing experiment differs; use a new ID: {experiment_id}")
        if not all((ROOT / p).exists() for p in previous["artifacts"].values()):
            raise ValueError(f"Completed experiment has missing artifacts: {experiment_id}")
        print(f"REUSE {experiment_id}: {previous['accuracy']:.6f}", flush=True)
        return previous
    write_json(ROOT / f"configs/{experiment_id}.json", config)
    x, tx, categorical = build_versioned_features(train, test, config["feature_variant"])
    y = train.Transported.astype(int)
    oof = np.zeros(len(train))
    test_probability = np.zeros(len(test))
    fold_results = []
    model_dir = ROOT / f"models/{experiment_id}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_name = config["model"]
    params = config["model_params"].copy()
    early_stopping = params.pop("early_stopping_rounds", 100)
    print(f"START {experiment_id}: {model_name}, {x.shape[1]} features, threads={threads}", flush=True)
    with threadpool_limits(limits=threads):
        for fold in range(1, n_splits + 1):
            fold_start = time.monotonic()
            fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            fit, valid, testing = x.iloc[fit_idx], x.iloc[valid_idx], tx
            encoder = None
            if model_name == "catboost":
                model = CatBoostClassifier(
                    **params, random_seed=seed + fold, thread_count=threads,
                    verbose=False, allow_writing_files=False,
                )
                model.fit(
                    fit, y.iloc[fit_idx], cat_features=categorical,
                    eval_set=(valid, y.iloc[valid_idx]),
                    early_stopping_rounds=early_stopping, verbose=False,
                )
                best = int(model.get_best_iteration()) + 1
                model.save_model(str(model_dir / f"fold{fold}.cbm"))
            else:
                fit, valid, testing, encoder = _encoded_frames(
                    fit, valid, testing, categorical, model_name
                )
                if model_name == "lightgbm":
                    import lightgbm as lgb

                    model = lgb.LGBMClassifier(
                        **params, random_state=seed + fold, n_jobs=threads, verbosity=-1
                    )
                    model.fit(
                        fit, y.iloc[fit_idx], eval_set=[(valid, y.iloc[valid_idx])],
                        eval_metric="binary_logloss", categorical_feature=categorical,
                        callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
                    )
                    best = int(model.best_iteration_)
                elif model_name == "xgboost":
                    from xgboost import XGBClassifier

                    model = XGBClassifier(
                        **params, random_state=seed + fold, n_jobs=threads,
                        early_stopping_rounds=early_stopping, enable_categorical=True,
                    )
                    model.fit(
                        fit, y.iloc[fit_idx], eval_set=[(valid, y.iloc[valid_idx])], verbose=False
                    )
                    best = int(model.best_iteration) + 1
                else:
                    raise ValueError(f"Unknown model: {model_name}")
                joblib.dump({"model": model, "encoder": encoder, "categorical": categorical},
                            model_dir / f"fold{fold}.joblib")
            prediction_params = {"thread_count": threads} if model_name == "catboost" else {}
            probability = model.predict_proba(valid, **prediction_params)[:, 1]
            oof[valid_idx] = probability
            test_probability += model.predict_proba(testing, **prediction_params)[:, 1] / n_splits
            score = float(accuracy_score(y.iloc[valid_idx], probability >= 0.5))
            result = {"fold": fold, "accuracy": score, "best_iteration": best,
                      "seconds": time.monotonic() - fold_start}
            fold_results.append(result)
            print(f"{experiment_id} fold {fold}/{n_splits}: {score:.6f}, best={best}, "
                  f"{result['seconds']:.1f}s", flush=True)
            if progress:
                progress(experiment_id, fold, result)
    artifacts = save_predictions(
        experiment_id, train, test, sample, folds, oof, test_probability
    )
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    reference = pd.read_csv(ROOT / "outputs/oof/baseline_catboost_oof.csv")
    assert reference.PassengerId.equals(train.PassengerId)
    result = {
        "experiment_id": experiment_id, "config": config, "fingerprint": fingerprint,
        "completed_at": timestamp(), "accuracy": float(accuracy_score(y, oof >= 0.5)),
        "log_loss": float(log_loss(y, oof)),
        "fold_std": float(np.std([f["accuracy"] for f in fold_results])),
        "folds": fold_results, "feature_count": x.shape[1], "categorical": categorical,
        "feature_columns": x.columns.tolist(), "threads": threads,
        "runtime_seconds": time.monotonic() - started,
        "raw_hashes": raw_hashes, "code_hashes": code_hashes,
        "fold_hash": file_hash(ROOT / f"data/processed/sgkf_{n_splits}_seed{seed}.csv"),
        "git_revision": git.stdout.strip() if git.returncode == 0 else "uncommitted repository",
        "python": platform.python_version(),
        "packages": {name: version(name) for name in [
            "pandas", "numpy", "scikit-learn", "catboost", "lightgbm", "xgboost"
        ]},
        "artifacts": artifacts, "segments": diagnostics(train, oof),
        "baseline_comparison": paired_group_bootstrap(train, reference.probability.to_numpy(), oof)
        if seed == 42 else {"note": "Compare against the matched-seed baseline, not seed-42 OOF."},
    }
    assert raw_hashes == {
        name: file_hash(ROOT / "data/raw" / name) for name in raw_hashes
    }
    write_json(metrics_path, result)
    print(f"DONE {experiment_id}: accuracy={result['accuracy']:.6f}, "
          f"logloss={result['log_loss']:.6f}, {result['runtime_seconds']:.1f}s", flush=True)
    return result
