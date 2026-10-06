"""Build a submission from a nested-LR stack (or a single member) over saved predictions.

Usage: `uv run python scripts/make_stack_submission.py <name> <member> [<member> ...]
[--test-seeds 42 123 ...]`. The LR stacker (C=1) is fitted on all seed-42 OOF rows of the members
(same recipe as `run_ha27_confirm.py`); with one member no stacker is fitted. Each member's test
probability is the logit mean of its fold-averaged test probabilities over `--test-seeds` (default
42 only), so extra seeds only bag the test-side fold models. Writes
`outputs/submissions/<name>.csv` and `outputs/probabilities/<name>.csv` and prints the share of
test predictions that differ from `ha27_stack3`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from run_ha27_stacker import exp, logit
from run_stack_member_screen import member_matrix
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    from spaceship_titanic.experiments import load_data

    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("members", nargs="+")
    parser.add_argument("--test-seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()
    train, test, sample = load_data()
    y = train.Transported.astype(int).to_numpy()
    test_logits = []
    for m in args.members:
        per_seed = []
        for s in args.test_seeds:
            frame = pd.read_csv(ROOT / f"outputs/probabilities/{exp(m, s)}.csv")
            assert frame.PassengerId.equals(test.PassengerId)
            per_seed.append(logit(frame.probability.to_numpy()))
        test_logits.append(np.mean(per_seed, axis=0))
    T = np.column_stack(test_logits)
    if len(args.members) == 1:
        probability = 1 / (1 + np.exp(-T[:, 0]))
    else:
        X, _ = member_matrix(args.members, 42, train)
        stacker = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
        probability = stacker.predict_proba(T)[:, 1]
        print("coef", np.round(stacker.coef_[0], 3).tolist(), flush=True)
    submission = sample.copy()
    submission["Transported"] = probability >= 0.5
    assert submission.PassengerId.equals(test.PassengerId)
    submission.to_csv(ROOT / f"outputs/submissions/{args.name}.csv", index=False)
    pd.DataFrame({"PassengerId": test.PassengerId, "probability": probability}).to_csv(
        ROOT / f"outputs/probabilities/{args.name}.csv", index=False)
    champion = pd.read_csv(ROOT / "outputs/submissions/ha27_stack3.csv").Transported.to_numpy()
    print(f"{args.name}: positive rate {submission.Transported.mean():.4f}, "
          f"differs from ha27_stack3 on {(submission.Transported.to_numpy() != champion).sum()} rows",
          flush=True)


if __name__ == "__main__":
    main()
