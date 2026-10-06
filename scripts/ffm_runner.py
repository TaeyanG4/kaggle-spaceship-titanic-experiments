"""H-A-37: field-aware factorization machine (FFM) member, fold-local, CPU torch.

Fields are the baseline feature columns. Columns with at most 16 distinct fit-fold values are used
as categories; other numerics are cut into 16 fit-fold quantile bins (missing is its own level).
Level vocabularies are built on the outer fit fold only, with one unknown index per field. Model:
logit = bias + sum_i w[x_i] + sum_{f<g} <V[x_f, g], V[x_g, f]> (k=4). Adam with weight decay; the
epoch is selected on an SGKF(8) slice of the fit fold (best slice log loss), never on the scored
fold. Usage: `uv run python scripts/ffm_runner.py <experiment_id> <seed> [<seed> ...]`.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
K, BINS, EPOCHS, LR, WD, BATCH = 4, 16, 40, 0.01, 1e-5, 256


def encode(x, fit_rows):
    import numpy as np
    import pandas as pd

    fit = x.iloc[fit_rows]
    specs, offset = [], 0
    for col in x.columns:
        values = fit[col]
        if pd.api.types.is_numeric_dtype(values) and values.nunique() > BINS:
            edges = np.unique(np.nanquantile(values.astype(float), np.linspace(0, 1, BINS + 1)[1:-1]))
            specs.append((col, "bin", edges, offset))
            offset += len(edges) + 3  # bins, missing, unknown
        else:
            levels = {v: i for i, v in enumerate(sorted(values.astype(str).unique()))}
            specs.append((col, "cat", levels, offset))
            offset += len(levels) + 1  # levels, unknown

    def transform(frame):
        cols = []
        for col, kind, info, off in specs:
            v = frame[col]
            if kind == "bin":
                arr = v.astype(float).to_numpy()
                idx = np.where(np.isnan(arr), len(info) + 1, np.searchsorted(info, arr))
            else:
                idx = v.astype(str).map(info).fillna(len(info)).to_numpy()
            cols.append(off + idx.astype(np.int64))
        return np.column_stack(cols)

    return transform, offset


def train_ffm(idx_tr, y_tr, idx_es, y_es, n_index, n_fields, seed):
    import numpy as np
    import torch

    torch.manual_seed(seed)

    class FFM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1))
            self.w = torch.nn.Embedding(n_index, 1)
            self.v = torch.nn.Embedding(n_index, n_fields * K)
            torch.nn.init.zeros_(self.w.weight)
            torch.nn.init.normal_(self.v.weight, std=0.01)
            self.pairs = [(f, g) for f in range(n_fields) for g in range(f + 1, n_fields)]

        def forward(self, idx):
            v = self.v(idx).view(idx.shape[0], n_fields, n_fields, K)  # [b, field, target, k]
            f = torch.tensor([p[0] for p in self.pairs])
            g = torch.tensor([p[1] for p in self.pairs])
            inter = (v[:, f, g, :] * v[:, g, f, :]).sum(-1).sum(-1)
            return self.bias + self.w(idx).sum((1, 2)) + inter

    model = FFM()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    xt, yt = torch.tensor(idx_tr), torch.tensor(y_tr, dtype=torch.float32)
    xe, ye = torch.tensor(idx_es), torch.tensor(y_es, dtype=torch.float32)
    best, best_state, best_epoch = np.inf, None, 0
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(EPOCHS):
        model.train()
        for b in torch.randperm(len(xt), generator=gen).split(BATCH):
            opt.zero_grad()
            loss_fn(model(xt[b]), yt[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(xe), ye))
        if val < best:
            best, best_epoch = val, epoch + 1
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def run_ffm(name: str, seed: int) -> dict:
    import numpy as np
    import torch
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
    from spaceship_titanic.versioned_features import build_versioned_features

    torch.set_num_threads(4)
    config = {"experiment_id": name, "model": "ffm", "k": K, "bins": BINS, "epochs": EPOCHS,
              "lr": LR, "weight_decay": WD, "batch": BATCH, "random_state": seed, "n_splits": 5,
              "feature_variant": "baseline", "epoch_selection": "SGKF(8) slice log loss"}
    code = {p: file_hash(ROOT / p) for p in [
        "scripts/ffm_runner.py", "src/spaceship_titanic/experiments.py",
        "src/spaceship_titanic/versioned_features.py", "src/spaceship_titanic/features.py"]}
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
    x, tx, _ = build_versioned_features(train, test, "baseline")
    x, tx = x.reset_index(drop=True), tx.reset_index(drop=True)
    y = train.Transported.astype(int).to_numpy()
    groups = train.PassengerId.str[:4]
    oof, test_probability, fold_results = np.zeros(len(x)), np.zeros(len(tx)), []
    for fold in range(1, 6):
        fold_start = time.monotonic()
        fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        fit_seed = seed + fold
        a, b = next(StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=fit_seed)
                    .split(fit_idx, y[fit_idx], groups.iloc[fit_idx]))
        tr_idx, es_idx = fit_idx[a], fit_idx[b]
        transform, n_index = encode(x, tr_idx)
        model, epoch = train_ffm(transform(x.iloc[tr_idx]), y[tr_idx], transform(x.iloc[es_idx]),
                                 y[es_idx], n_index, x.shape[1], fit_seed)
        with torch.no_grad():
            probability = torch.sigmoid(model(torch.tensor(transform(x.iloc[valid_idx])))).numpy()
            test_probability += torch.sigmoid(model(torch.tensor(transform(tx)))).numpy() / 5
        oof[valid_idx] = probability
        fold_results.append({"fold": fold, "best_epoch": epoch,
                             "accuracy": float(accuracy_score(y[valid_idx], probability >= 0.5)),
                             "seconds": time.monotonic() - fold_start})
        print(f"{name} fold {fold}: {fold_results[-1]['accuracy']:.6f} epoch={epoch} "
              f"({fold_results[-1]['seconds']:.0f}s)", flush=True)
    artifacts = save_predictions(name, train, test, sample, folds, oof, test_probability)
    result = {
        "experiment_id": name, "config": config, "fingerprint": fingerprint,
        "completed_at": timestamp(), "accuracy": float(accuracy_score(y, oof >= 0.5)),
        "log_loss": float(log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6))),
        "fold_std": float(np.std([f["accuracy"] for f in fold_results])),
        "folds": fold_results, "runtime_seconds": time.monotonic() - started,
        "raw_hashes": raw, "code_hashes": code, "artifacts": artifacts,
        "segments": diagnostics(train, oof),
    }
    write_json(metrics_path, result)
    print(f"DONE {name}: accuracy={result['accuracy']:.6f}, logloss={result['log_loss']:.6f}",
          flush=True)
    return result


if __name__ == "__main__":
    experiment, *seeds = sys.argv[1:]
    for s in seeds:
        run_ffm(experiment if int(s) == 42 else f"{experiment}_seed{s}", int(s))
