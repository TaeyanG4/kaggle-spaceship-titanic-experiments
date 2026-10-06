"""H-D-01: read-only diagnosis of Earth-passenger errors from saved OOF predictions.

No model is trained. Segments use target-free raw features only; Transported is used solely
to score the existing OOF predictions. Results are diagnostics, not grounds for segment models
or full-OOF threshold tuning.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from spaceship_titanic.experiments import (
    ROOT,
    frozen_folds,
    load_data,
    paired_group_bootstrap,
    timestamp,
    write_json,
)
from spaceship_titanic.features import SPEND_COLUMNS

MODELS = {
    "baseline": "baseline_catboost_oof.csv",
    "v1_name_fix": "v1_name_fix.csv",
    "v2": "v2.csv",
    "v2_logloss": "v2_logloss.csv",
    "v2_lightgbm": "v2_lightgbm.csv",
    "v2_xgboost": "v2_xgboost.csv",
    "v2_blend": "v2_blend.csv",
}
PROBABILITY_LOSS_MODELS = ["v2_logloss", "v2_lightgbm", "v2_xgboost", "v2_blend"]
MIN_ROWS = 50


def load_probabilities(train: pd.DataFrame) -> pd.DataFrame:
    columns = {}
    for name, filename in MODELS.items():
        oof = pd.read_csv(ROOT / "outputs/oof" / filename, dtype={"PassengerId": str})
        assert oof.PassengerId.equals(train.PassengerId), name
        assert oof.Transported.astype(bool).equals(train.Transported.astype(bool)), name
        columns[name] = oof.probability.to_numpy()
    return pd.DataFrame(columns, index=train.index)


def segments(train: pd.DataFrame) -> dict[str, pd.Series]:
    spend = train[SPEND_COLUMNS]
    observed_total = spend.fillna(0).sum(axis=1)
    any_missing = spend.isna().any(axis=1)
    spend_state = np.select(
        [observed_total.gt(0), any_missing],
        ["positive", "zero_with_missing"],
        default="zero_all_observed",
    )
    groups = train.PassengerId.str.split("_").str[0]
    age = train.Age
    age_band = pd.cut(age, [-1, 12, 17, 25, 40, 200],
                      labels=["0-12", "13-17", "18-25", "26-40", "41+"]).astype("string")
    return {
        "CryoSleep": train.CryoSleep.astype("string").fillna("missing"),
        "AgeBand": age_band.fillna("missing"),
        "SpendState": pd.Series(spend_state, index=train.index),
        "GroupSize": groups.map(groups.value_counts()).clip(upper=4).astype(str),
        "Destination": train.Destination.fillna("missing"),
        "CabinDeck": train.Cabin.str.split("/").str[0].fillna("missing"),
        "CabinSide": train.Cabin.str.split("/").str[2].fillna("missing"),
        "CryoSleep_SpendState": train.CryoSleep.astype("string").fillna("missing")
        + "|" + pd.Series(spend_state, index=train.index),
    }


def segment_table(target: pd.Series, probabilities: pd.DataFrame,
                  values: pd.Series) -> list[dict]:
    rows = []
    for segment, index in values.groupby(values).groups.items():
        y = target.loc[index].to_numpy()
        record = {"segment": str(segment), "rows": len(index),
                  "actual_rate": float(y.mean()), "small": bool(len(index) < MIN_ROWS)}
        for model in probabilities.columns:
            p = probabilities.loc[index, model].to_numpy()
            errors = int(((p >= 0.5) != y).sum())
            record[f"{model}_errors"] = errors
            record[f"{model}_mean_prob"] = float(p.mean())
            record[f"{model}_log_loss"] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6),
                                                         labels=[False, True]))
        base = record["baseline_errors"]
        for model in probabilities.columns:
            record[f"{model}_delta_errors"] = record[f"{model}_errors"] - base
        rows.append(record)
    return sorted(rows, key=lambda r: -r["rows"])


def flip_summary(target: pd.Series, probabilities: pd.DataFrame, mask: pd.Series,
                 seg: dict[str, pd.Series]) -> dict:
    y = target[mask].to_numpy()
    base = probabilities.loc[mask, "baseline"].to_numpy()
    base_correct = (base >= 0.5) == y
    result = {}
    for model in PROBABILITY_LOSS_MODELS + ["v1_name_fix", "v2"]:
        p = probabilities.loc[mask, model].to_numpy()
        correct = (p >= 0.5) == y
        lost = base_correct & ~correct
        gained = ~base_correct & correct
        near = np.abs(p - 0.5) < 0.1
        lost_by = {
            name: pd.Series(values[mask].to_numpy()[lost]).value_counts().head(6).to_dict()
            for name, values in seg.items()
            if name in ("CryoSleep", "SpendState", "AgeBand", "GroupSize")
        }
        result[model] = {
            "lost_vs_baseline": int(lost.sum()),
            "gained_vs_baseline": int(gained.sum()),
            "net_errors_vs_baseline": int(lost.sum() - gained.sum()),
            "lost_with_prob_within_0.1_of_0.5": int((lost & near).sum()),
            "lost_true_label_rate": float(y[lost].mean()) if lost.any() else None,
            "gained_true_label_rate": float(y[gained].mean()) if gained.any() else None,
            "predicted_positive_rate": float((p >= 0.5).mean()),
            "lost_by_segment": lost_by,
        }
    result["actual_positive_rate"] = float(y.mean())
    result["baseline_predicted_positive_rate"] = float((base >= 0.5).mean())
    return result


def confidence_table(target: pd.Series, probabilities: pd.DataFrame, mask: pd.Series) -> dict:
    bins = [0, 0.1, 0.3, 0.45, 0.55, 0.7, 0.9, 1.0]
    y = target[mask]
    out = {}
    for model in ["baseline", "v2_logloss", "v2_blend"]:
        p = probabilities.loc[mask, model]
        band = pd.cut(p, bins, include_lowest=True).astype(str)
        frame = pd.DataFrame({"band": band, "y": y, "error": (p >= 0.5) != y})
        out[model] = (frame.groupby("band", sort=False)
                      .agg(rows=("y", "size"), actual_rate=("y", "mean"), errors=("error", "sum"))
                      .reset_index().to_dict(orient="records"))
    return out


def write_markdown(result: dict) -> None:
    earth = result["earth_overall"]
    lines = [
        "# H-D-01 — Earth error diagnosis (saved OOF only)",
        "",
        f"Created: {result['created_at']}. No model trained; segments are target-free raw features.",
        "",
        "## Earth vs non-Earth errors",
        "",
        ("| Model | Earth errors | Earth Δ vs baseline | Non-Earth errors | Non-Earth Δ | "
         "Earth log loss | Earth bootstrap Δacc 95% CI |"),
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for model, row in earth.items():
        ci = row.get("earth_bootstrap_ci95")
        ci_text = "-" if ci is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        lines.append(
            f"| {model} | {row['earth_errors']} | {row['earth_delta']:+d} | "
            f"{row['other_errors']} | {row['other_delta']:+d} | {row['earth_log_loss']:.4f} | "
            f"{ci_text} |"
        )
    lines += ["", "## Earth error deltas by segment (rows ≥ 50)", ""]
    for name, rows in result["earth_segments"].items():
        header = "| Segment | Rows | Actual rate | Baseline errors | " + " | ".join(
            f"{m} Δ" for m in MODELS if m != "baseline") + " |"
        lines += [f"### {name}", "", header,
                  "| --- | ---: | ---: | ---: |" + " ---: |" * (len(MODELS) - 1)]
        for r in rows:
            if r["small"]:
                continue
            deltas = " | ".join(f"{r[f'{m}_delta_errors']:+d}" for m in MODELS if m != "baseline")
            lines.append(f"| {r['segment']} | {r['rows']} | {r['actual_rate']:.3f} | "
                         f"{r['baseline_errors']} | {deltas} |")
        lines.append("")
    lines += ["## Flips vs baseline on Earth rows", "",
              "| Model | Lost | Gained | Net | Lost near 0.5 (±0.1) | Pred. positive rate |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    flips = result["earth_flips"]
    for model, row in flips.items():
        if not isinstance(row, dict):
            continue
        lines.append(f"| {model} | {row['lost_vs_baseline']} | {row['gained_vs_baseline']} | "
                     f"{row['net_errors_vs_baseline']:+d} | "
                     f"{row['lost_with_prob_within_0.1_of_0.5']} | "
                     f"{row['predicted_positive_rate']:.3f} |")
    lines += ["", (f"Earth actual positive rate {flips['actual_positive_rate']:.3f}; baseline "
                   f"predicted positive rate {flips['baseline_predicted_positive_rate']:.3f}."), "",
              "## Limits", "", result["limits"], ""]
    (ROOT / "reports/earth_error_diagnosis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    train, _, _ = load_data()
    frozen_folds(train, 42)
    target = train.Transported.astype(bool)
    probabilities = load_probabilities(train)
    seg = segments(train)
    earth = train.HomePlanet.eq("Earth")

    overall = {}
    base_pred = probabilities.baseline >= 0.5
    base_earth = int((base_pred[earth] != target[earth]).sum())
    base_other = int((base_pred[~earth] != target[~earth]).sum())
    earth_train = train[earth].reset_index(drop=True)
    for model in probabilities.columns:
        pred = probabilities[model] >= 0.5
        row = {
            "earth_errors": int((pred[earth] != target[earth]).sum()),
            "other_errors": int((pred[~earth] != target[~earth]).sum()),
            "earth_log_loss": float(log_loss(target[earth], probabilities.loc[earth, model])),
        }
        row["earth_delta"] = row["earth_errors"] - base_earth
        row["other_delta"] = row["other_errors"] - base_other
        if model != "baseline":
            row["earth_bootstrap_ci95"] = paired_group_bootstrap(
                earth_train, probabilities.loc[earth, "baseline"].to_numpy(),
                probabilities.loc[earth, model].to_numpy(),
            )["ci95"]
        overall[model] = row

    result = {
        "created_at": timestamp(),
        "plan_item": "H-D-01",
        "earth_rows": int(earth.sum()),
        "models": MODELS,
        "earth_overall": overall,
        "earth_segments": {
            name: segment_table(target[earth], probabilities[earth], values[earth])
            for name, values in seg.items()
        },
        "earth_flips": flip_summary(target, probabilities, earth, seg),
        "earth_confidence_bands": confidence_table(target, probabilities, earth),
        "limits": (
            "Diagnostic on fitted OOF predictions from one seed-42 SGKF split. Segment deltas "
            "of a few rows are within noise; bootstrap CIs are conditional on fitted models. "
            "HomePlanet-missing rows are excluded from Earth. Not evidence for segment models "
            "or full-OOF threshold tuning."
        ),
    }
    write_json(ROOT / "reports/earth_error_diagnosis.json", result)
    write_markdown(result)
    print("Saved reports/earth_error_diagnosis.json and .md")
    for model, row in overall.items():
        print(f"{model:12s} earth={row['earth_errors']:4d} ({row['earth_delta']:+d}) "
              f"other={row['other_errors']:4d} ({row['other_delta']:+d}) "
              f"earth_ll={row['earth_log_loss']:.4f}")


if __name__ == "__main__":
    main()
