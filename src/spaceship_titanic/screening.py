"""Multi-seed SGKF screening and confirmation rule (agents.md experiment decision rule)."""

from __future__ import annotations

import copy

SCREEN_SEEDS = (42, 123, 2026)
CONFIRM_SEEDS = (7, 99)
SCREEN_GAIN = 0.002


def seeded(config: dict, seed: int) -> dict:
    """Copy a seed-42 config to another SGKF seed; seed 42 keeps its original ID for reuse."""
    out = copy.deepcopy(config)
    base = config["experiment_id"]
    out["random_state"] = seed
    out["experiment_id"] = base if seed == 42 else f"{base}_seed{seed}"
    return out


def paired_deltas(control: dict[int, float], challenger: dict[int, float]) -> dict[int, float]:
    if set(control) != set(challenger):
        raise ValueError("Control and challenger must cover the same seeds")
    return {seed: challenger[seed] - control[seed] for seed in sorted(control)}


def screen(control: dict[int, float], challenger: dict[int, float],
           gain: float = SCREEN_GAIN) -> dict:
    deltas = paired_deltas(control, challenger)
    mean = sum(deltas.values()) / len(deltas)
    positive = sum(delta > 0 for delta in deltas.values())
    return {
        "deltas": deltas, "mean_delta": mean, "positive_seeds": positive,
        "passes": mean >= gain and positive >= len(deltas) - 1,
    }


def confirm(control: dict[int, float], challenger: dict[int, float]) -> dict:
    deltas = paired_deltas(control, challenger)
    return {"deltas": deltas, "mean_delta": sum(deltas.values()) / len(deltas),
            "passes": all(delta > 0 for delta in deltas.values())}
