"""Build an averaged stacker member from saved runs (H-A-35).

Usage: `uv run python scripts/make_avg_member.py <new_name> <member> [<member> ...] --seeds 42 123 ...`.
Per seed, the OOF probabilities of the members (same frozen folds, asserted) are averaged in logit
space and written as `outputs/oof/<new_name>[_seed<s>].csv`; for seed 42 the members' test
probabilities are averaged the same way into `outputs/probabilities/<new_name>.csv`. No labels are
used, so the averaged member is as honest as its inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from run_ha27_stacker import exp, logit

ROOT = Path(__file__).resolve().parents[1]


def mean_prob(frames):
    z = np.mean([logit(f.probability.to_numpy()) for f in frames], axis=0)
    return 1 / (1 + np.exp(-z))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("members", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    for seed in args.seeds:
        frames = [pd.read_csv(ROOT / f"outputs/oof/{exp(m, seed)}.csv") for m in args.members]
        for f in frames[1:]:
            assert f.PassengerId.equals(frames[0].PassengerId)
            assert np.array_equal(f.fold.to_numpy(), frames[0].fold.to_numpy())
        out = frames[0][["PassengerId", "Transported", "fold"]].copy()
        out["probability"] = mean_prob(frames)
        out["prediction"] = out.probability >= 0.5
        out.to_csv(ROOT / f"outputs/oof/{exp(args.name, seed)}.csv", index=False)
        print(f"{exp(args.name, seed)}: OOF accuracy "
              f"{float((out.prediction == out.Transported.astype(bool)).mean()):.6f}", flush=True)
    if 42 in args.seeds:
        tests = [pd.read_csv(ROOT / f"outputs/probabilities/{m}.csv") for m in args.members]
        for t in tests[1:]:
            assert t.PassengerId.equals(tests[0].PassengerId)
        pd.DataFrame({"PassengerId": tests[0].PassengerId, "probability": mean_prob(tests)}) \
            .to_csv(ROOT / f"outputs/probabilities/{args.name}.csv", index=False)


if __name__ == "__main__":
    main()
