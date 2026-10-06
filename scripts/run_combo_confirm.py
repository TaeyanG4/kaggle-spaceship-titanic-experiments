"""H-A-47: confirm a multi-member stack (chosen post hoc on 42/123/2026) on fresh seeds 7/99.

Usage: `uv run python scripts/run_combo_confirm.py <name> <member> [<member> ...]`. The nested LR
stack over the given members is compared with `ha27_stack3` on seeds 7 and 99 under
`screening.confirm` (both deltas positive). Members need OOF on 7/99. On a pass the improvement
alert plays; the submission itself is built separately with `make_stack_submission.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from run_stack_member_screen import BASE, nested_accuracy

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    from alert import improvement_alert

    from spaceship_titanic.experiments import load_data, timestamp, write_json
    from spaceship_titanic.screening import CONFIRM_SEEDS, confirm

    name, members = sys.argv[1], sys.argv[2:]
    train, _, _ = load_data()
    y = train.Transported.astype(int).to_numpy()
    control = {s: nested_accuracy(BASE, s, train, y) for s in CONFIRM_SEEDS}
    challenger = {s: nested_accuracy(members, s, train, y) for s in CONFIRM_SEEDS}
    result = confirm(control, challenger)
    print(f"{name} CONFIRM {result}", flush=True)
    if result["passes"]:
        improvement_alert(f"H-A-47 confirmed: {name}")
    write_json(ROOT / f"reports/ha47_{name}_confirm.json", {
        "completed_at": timestamp(), "plan_item": "H-A-47", "members": members,
        "control_accuracy": control, "challenger_accuracy": challenger, "confirmation": result})
    print(json.dumps({"passes": result["passes"]}), flush=True)


if __name__ == "__main__":
    main()
