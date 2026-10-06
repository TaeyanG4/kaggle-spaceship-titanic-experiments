"""H-A-26: fresh-seed extended test of the frozen + fine-tuned TabPFN v3.5 equal blend.

Seeds 11/13/17/19/23 were never used before. Per seed it runs (or reuses) frozen TabPFN v3.5
(`ha12_tabpfn_seed<s>`), fold-local fine-tuning (`ha23_ft_seed<s>`) and fine-tune-then-refit
(`ha24_ftrefit_seed<s>`) with the existing runners, then scores the equal average of the three
OOF probabilities against the frozen model. Predeclared rule: mean paired delta >= +0.002 and
>= 4 of 5 seeds positive. This differs from the agents.md protocol, so a pass is reported for the
user's approval; nothing is submitted here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from alert import improvement_alert
from run_hm02_grid import worker_init
from tabpfn_ft_runner import run_tabpfn_ft
from tabpfn_ftrefit_runner import run_tabpfn_ftrefit
from tabpfn_runner import run_tabpfn

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 13, 17, 19, 23)
GAIN, MIN_POSITIVE = 0.002, 4


def main() -> None:
    worker_init(2)
    import numpy as np
    import pandas as pd
    from run_versions import Tee

    from spaceship_titanic.experiments import load_data, timestamp, write_json

    sys.stdout = Tee(sys.stdout, ROOT / "reports/ha26_training.log")
    write_json(ROOT / "reports/ha26_manifest.json", {
        "created_at": timestamp(), "plan_item": "H-A-26", "seeds": list(SEEDS),
        "members": ["ha12_tabpfn", "ha23_ft", "ha24_ftrefit"], "blend": "equal average",
        "rule": f"mean paired delta >= +{GAIN} and >= {MIN_POSITIVE}/5 positive vs frozen",
        "approval": "a pass is reported to the user before promotion or submission"})
    train, _, _ = load_data()
    y = train.Transported.astype(bool).to_numpy()
    rows = []
    for seed in SEEDS:
        runs = [run_tabpfn(f"ha12_tabpfn_seed{seed}", seed, 2),
                run_tabpfn_ft(f"ha23_ft_seed{seed}", seed, 2),
                run_tabpfn_ftrefit(f"ha24_ftrefit_seed{seed}", seed, 2)]
        probs = [pd.read_csv(ROOT / r["artifacts"]["oof"]).probability.to_numpy() for r in runs]
        frozen = float(((probs[0] >= 0.5) == y).mean())
        blend = float(((np.mean(probs, axis=0) >= 0.5) == y).mean())
        rows.append({"seed": seed, "frozen": frozen, "ft": runs[1]["accuracy"],
                     "ftrefit": runs[2]["accuracy"], "blend": blend, "delta": blend - frozen})
        print(f"seed {seed}: frozen {frozen:.6f} blend {blend:.6f} delta {blend - frozen:+.6f}",
              flush=True)
    mean_delta = float(np.mean([r["delta"] for r in rows]))
    positive = int(sum(r["delta"] > 0 for r in rows))
    passes = mean_delta >= GAIN and positive >= MIN_POSITIVE
    print(f"extended mean_delta={mean_delta:+.6f} positive={positive}/5 passes={passes}", flush=True)
    if passes:
        improvement_alert("H-A-26 extended test passed; awaiting user approval to submit")
    write_json(ROOT / "reports/ha26_summary.json", {
        "completed_at": timestamp(), "plan_item": "H-A-26", "per_seed": rows,
        "mean_delta": mean_delta, "positive_seeds": positive, "passes": passes,
        "status": "awaiting user approval" if passes else "not promoted"})
    print(json.dumps({"passes": passes, "mean_delta": mean_delta, "positive": positive}),
          flush=True)


if __name__ == "__main__":
    main()
