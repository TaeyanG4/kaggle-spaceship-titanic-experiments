"""Screen candidate members for the D-A-027 nested stacker (H-A-28..33 gate).

Usage: `uv run python scripts/run_stack_member_screen.py <item> <member> [<member> ...]`.
A candidate written `old=new` replaces base member `old` with `new` instead of adding `new`.
For each candidate, the nested LR stacker over `ha27_stack3` members plus that candidate is
compared with the `ha27_stack3` nested stacker itself on seeds 42/123/2026 (same folds, same nested
fitting: stacker fitted on the other four outer folds only). Members need saved OOF predictions on
every screen seed. The best passing candidate is confirmed on 7/99 when its OOF exists there; a
promotion writes `outputs/submissions/stack_<member>.csv` from a stacker fitted on all seed-42 OOF
rows and the members' seed-42 test probabilities, and plays the improvement alert.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from run_ha27_stacker import exp, logit
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
BASE = ["ha12_tabpfn", "ha23_ft", "ha24_ftrefit"]


def member_matrix(members, seed, train):
    folds, columns = None, []
    for m in members:
        frame = pd.read_csv(ROOT / f"outputs/oof/{exp(m, seed)}.csv")
        assert frame.PassengerId.equals(train.PassengerId)
        if folds is None:
            folds = frame.fold.to_numpy()
        assert np.array_equal(frame.fold.to_numpy(), folds)
        columns.append(logit(frame.probability.to_numpy()))
    return np.column_stack(columns), folds


def nested_accuracy(members, seed, train, y):
    X, folds = member_matrix(members, seed, train)
    stacked = np.zeros(len(y))
    for k in np.unique(folds):
        fit, hold = folds != k, folds == k
        stacked[hold] = LogisticRegression(C=1.0, max_iter=1000).fit(X[fit], y[fit]) \
            .predict_proba(X[hold])[:, 1]
    return float(((stacked >= 0.5) == y).mean())


def members_for(candidate):
    if "=" in candidate:
        old, new = candidate.split("=", 1)
        assert old in BASE, candidate
        return [new if m == old else m for m in BASE]
    return [*BASE, candidate]


def has_oof(member, seeds):
    return all((ROOT / f"outputs/oof/{exp(member, s)}.csv").exists() for s in seeds)


def main() -> None:
    from alert import improvement_alert

    from spaceship_titanic.experiments import file_hash, load_data, timestamp, write_json
    from spaceship_titanic.screening import CONFIRM_SEEDS, SCREEN_SEEDS, confirm, screen

    item, candidates = sys.argv[1], sys.argv[2:]
    train, test, sample = load_data()
    y = train.Transported.astype(int).to_numpy()
    base = {s: nested_accuracy(BASE, s, train, y) for s in SCREEN_SEEDS}
    rows = []
    for c in candidates:
        acc = {s: nested_accuracy(members_for(c), s, train, y) for s in SCREEN_SEEDS}
        outcome = screen(base, acc)
        rows.append({"candidate": c, "accuracy": acc, **outcome})
        print(f"+{c:28s} mean_delta={outcome['mean_delta']:+.6f} "
              f"positive={outcome['positive_seeds']}/3 passes={outcome['passes']}", flush=True)
    rows.sort(key=lambda r: -r["mean_delta"])
    summary = {"completed_at": timestamp(), "plan_item": item, "base_members": BASE,
               "base_accuracy": base, "screening": rows, "confirmation": None, "promoted": False}
    passing = [r for r in rows if r["passes"]]
    if passing:
        best = passing[0]["candidate"]
        if not all(has_oof(m, CONFIRM_SEEDS) for m in members_for(best)):
            summary["confirmation"] = {"candidate": best, "status": "needs OOF on seeds 7/99"}
            print(f"NEEDS_CONFIRM_RUNS {best}", flush=True)
        else:
            members = members_for(best)
            confirmation = {"candidate": best, **confirm(
                {s: nested_accuracy(BASE, s, train, y) for s in CONFIRM_SEEDS},
                {s: nested_accuracy(members, s, train, y) for s in CONFIRM_SEEDS})}
            summary["confirmation"] = confirmation
            print(f"CONFIRM {confirmation}", flush=True)
            if confirmation["passes"]:
                X42, _ = member_matrix(members, 42, train)
                final = LogisticRegression(C=1.0, max_iter=1000).fit(X42, y)
                test_x = []
                for m in members:
                    frame = pd.read_csv(ROOT / f"outputs/probabilities/{m}.csv")
                    assert frame.PassengerId.equals(test.PassengerId)
                    test_x.append(logit(frame.probability.to_numpy()))
                probability = final.predict_proba(np.column_stack(test_x))[:, 1]
                submission = sample.copy()
                submission["Transported"] = probability >= 0.5
                assert submission.PassengerId.equals(test.PassengerId)
                tag = best.replace("=", "_to_")
                out = ROOT / f"outputs/submissions/stack_{tag}.csv"
                submission.to_csv(out, index=False)
                pd.DataFrame({"PassengerId": test.PassengerId, "probability": probability}) \
                    .to_csv(ROOT / f"outputs/probabilities/stack_{tag}.csv", index=False)
                summary["promoted"] = True
                summary["submission"] = {"path": str(out.relative_to(ROOT)),
                                         "sha256": file_hash(out),
                                         "coef": final.coef_[0].tolist()}
                improvement_alert(f"{item} promoted: stacker + {best}")
    write_json(ROOT / f"reports/{item.lower().replace('-', '')}_stack_summary.json", summary)
    print(json.dumps({"promoted": summary["promoted"]}), flush=True)


if __name__ == "__main__":
    main()
