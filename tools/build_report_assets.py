"""Build the report charts in `docs/assets/` from committed evidence only.

Inputs: `docs/evidence/experiment-ledger.csv`, `docs/evidence/kaggle-submissions.csv`,
`docs/evidence/final-stackers.json`, `reports/ha43_subgroup_audit.json`,
`reports/ha44_adversarial_seeds.json`, and the per-run `reports/*_metrics.json` files.
No model is trained and no Kaggle call is made. Run: `uv run python tools/build_report_assets.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"
EVIDENCE = ROOT / "docs/evidence"
NAVY, TEAL, SAND, RED, GREY, GOLD = "#16324a", "#2f8f83", "#d2b587", "#c4573f", "#9aa7b0", "#e0a526"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#bdcbd5",
                     "axes.labelcolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY,
                     "axes.titleweight": "bold", "axes.titlecolor": NAVY, "axes.titlesize": 14,
                     "axes.spines.top": False, "axes.spines.right": False})
SEEDS = (42, 123, 2026)


def metric(name: str, seed: int) -> float:
    suffix = "" if seed == 42 else f"_seed{seed}"
    return json.loads((ROOT / f"reports/{name}{suffix}_metrics.json").read_text())["accuracy"]


def save(fig, name: str) -> None:
    fig.savefig(ASSETS / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def score_progression() -> None:
    stackers = json.loads((EVIDENCE / "final-stackers.json").read_text())
    steps = [
        ("CatBoost\n(honest stopping)", [metric("hm10_innercv_logloss", s) for s in SEEDS], 0.81131),
        ("TabPFN v3.5\n(frozen)", [metric("ha12_tabpfn", s) for s in SEEDS], 0.81786),
        ("TabPFN stack\n(official champion)",
         list(stackers["ha27_stack3"]["nested_oof_accuracy"].values()), 0.82020),
        ("+ 100-epoch\nfine-tune member", list(stackers["sub_stack3_plus_a1ne2"]
                                                ["nested_oof_accuracy"].values()), 0.82604),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(steps))
    cv = np.array([np.mean(s[1]) for s in steps])
    lo = cv - np.array([min(s[1]) for s in steps])
    hi = np.array([max(s[1]) for s in steps]) - cv
    lb = np.array([s[2] for s in steps])
    ax.errorbar(x, cv, yerr=[lo, hi], fmt="o-", color=TEAL, lw=2.4, ms=9, capsize=6,
                label="Honest OOF accuracy (mean, min-max over SGKF seeds 42/123/2026)")
    ax.plot(x, lb, "s--", color=NAVY, lw=2, ms=9, label="Kaggle public LB")
    ax.scatter([0], [0.80780], marker="x", s=90, color=RED, zorder=5,
               label="First submission 0.80780 (CatBoost, optimistic stopping)")
    for i, (c, b) in enumerate(zip(cv, lb)):
        ax.annotate(f"{c:.4f}", (i, c), textcoords="offset points", xytext=(12, 6), color=TEAL,
                    fontsize=10, fontweight="bold")
        ax.annotate(f"{b:.5f}", (i, b), textcoords="offset points", xytext=(12, -14), color=NAVY,
                    fontsize=10, fontweight="bold")
    ax.axvspan(2.5, 3.5, color=SAND, alpha=0.18, lw=0)
    ax.text(3, 0.8065, "best of one CV-ranked batch\n(not CV-promoted)", ha="center",
            fontsize=9, color="#7a6440")
    ax.set_xticks(x, [s[0] for s in steps])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.804, 0.836)
    ax.set_title("Score progression of the promoted models")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "score-progression.png")


def experiment_landscape() -> None:
    ledger = pd.read_csv(EVIDENCE / "experiment-ledger.csv")
    colors = {"promoted": TEAL, "rejected": GREY, "inconclusive": SAND, "post-hoc": GOLD}
    stage_names = {"1-catboost": "Stage 1 - CatBoost era (vs CatBoost champion)",
                   "2-tabpfn": "Stage 2 - TabPFN era (vs frozen TabPFN v3.5)",
                   "3-stack": "Stage 3 - Stacker era (member added to the TabPFN stack)"}
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 15),
                             gridspec_kw={"height_ratios": [19, 11, 14]})
    for ax, (stage, part) in zip(axes, ledger.groupby("stage", sort=True)):
        part = part.sort_values("mean_delta")
        y = np.arange(len(part))
        ax.barh(y, part.mean_delta * 100, color=[colors[o] for o in part.outcome],
                edgecolor="white")
        ax.set_yticks(y, [f"{lab}  ({pos})" for lab, pos in zip(part.label_en,
                                                               part.positive_seeds)], fontsize=9)
        ax.axvline(0.2, color=RED, ls="--", lw=1.2)
        ax.axvline(0, color=NAVY, lw=0.8)
        ax.set_title(stage_names[stage], fontsize=12, loc="left")
        ax.set_xlabel("Mean paired accuracy delta, percentage points (3 seeds)")
        ax.grid(axis="x", alpha=0.25)
    axes[0].text(0.21, 0.5, "promotion bar +0.2 pp", color=RED, fontsize=9,
                 transform=axes[0].get_xaxis_transform())
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    fig.legend(handles, ["promoted (passed screen + fresh-seed confirmation)", "rejected",
                         "inconclusive (consistent but below bar)", "post hoc combination"],
               loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("What was tested: every idea against a matched control", fontsize=15,
                 fontweight="bold", color=NAVY, y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.985))
    save(fig, "experiment-landscape.png")


def submission_history() -> None:
    subs = pd.read_csv(EVIDENCE / "kaggle-submissions.csv").iloc[::-1].reset_index(drop=True)
    promoted = {"hm10_innercv_logloss.csv", "ha12_tabpfn.csv", "ha27_stack3.csv"}
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(subs))
    colors = [TEAL if f in promoted else (RED if i == 0 else SAND)
              for i, f in enumerate(subs.fileName)]
    ax.scatter(x, subs.publicScore, c=colors, s=80, zorder=3, edgecolor=NAVY, lw=0.6)
    ax.step(x, subs.publicScore.cummax(), where="post", color=NAVY, lw=1.6,
            label="best so far")
    ax.axvline(3.5, color=GREY, ls=":", lw=1.2)
    ax.text(3.6, 0.8085, "one batch of 8 CV-ranked candidates,\nsubmitted once each, same day",
            fontsize=9, color="#7a6440")
    for i, (f, s) in enumerate(zip(subs.fileName, subs.publicScore)):
        if i < 4 or s == subs.publicScore.max():
            ax.annotate(f"{s:.5f}", (i, s), textcoords="offset points", xytext=(-14, 9),
                        fontsize=9, color=NAVY)
    ax.set_xticks(x, [f"#{i + 1}" for i in x])
    ax.set_ylabel("Public LB accuracy")
    ax.set_title(f"All {len(subs)} Kaggle submissions (chronological)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, mec=NAVY, ms=8)
               for c in (RED, TEAL, SAND)]
    ax.legend(handles + [plt.Line2D([], [], color=NAVY)],
              ["first baseline", "CV-promoted champion", "batch candidate", "best so far"],
              loc="lower right", fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "submission-history.png")


def cv_vs_lb() -> None:
    subs = pd.read_csv(EVIDENCE / "kaggle-submissions.csv").set_index("fileName").publicScore
    cv = {"hm10_innercv_logloss.csv": np.mean([metric("hm10_innercv_logloss", s) for s in SEEDS]),
          "ha12_tabpfn.csv": np.mean([metric("ha12_tabpfn", s) for s in SEEDS]),
          "ha27_stack3.csv": 0.828253, "sub_stack3_plus_a1ne2.csv": 0.829326,
          "sub_a1mix_stack.csv": 0.829633, "sub_avg4.csv": 0.830170, "sub_combo5.csv": 0.830247,
          "sub_combo7.csv": 0.831013, "sub_combo7_bag3.csv": 0.831013,
          "sub_stack_a1ctx.csv": 0.828981, "sub_a1mix_single.csv": 0.829480}
    promoted = {"hm10_innercv_logloss.csv", "ha12_tabpfn.csv", "ha27_stack3.csv"}
    fig, (ax, side) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [3, 1.6]})
    rows = []
    for i, (f, c) in enumerate(sorted(cv.items(), key=lambda kv: kv[1]), 1):
        ax.scatter(c, subs[f], s=170, color=TEAL if f in promoted else SAND, edgecolor=NAVY,
                   zorder=3)
        ax.text(c, subs[f], str(i), ha="center", va="center", fontsize=8, fontweight="bold",
                color="white" if f in promoted else NAVY, zorder=4)
        rows.append(f"{i:>2}  {f.replace('.csv', ''):<22} {c:.4f}  {subs[f]:.5f}")
    xs = np.linspace(0.815, 0.833, 10)
    ax.plot(xs, xs, color=GREY, ls="--", lw=1, label="LB = CV")
    ax.set_xlabel("Honest 3-seed nested OOF accuracy (mean)")
    ax.set_ylabel("Public LB accuracy")
    ax.set_title("CV vs public LB")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=TEAL, mec=NAVY, ms=10),
                       plt.Line2D([], [], marker="o", ls="", color=SAND, mec=NAVY, ms=10),
                       plt.Line2D([], [], color=GREY, ls="--")],
              labels=["CV-promoted", "batch candidate", "LB = CV"], frameon=False,
              loc="upper left")
    ax.grid(alpha=0.25)
    side.axis("off")
    side.text(0, 1, " #  submission              CV      LB\n" + "\n".join(rows),
              family="monospace", fontsize=8.5, va="top", color=NAVY)
    fig.tight_layout()
    save(fig, "cv-vs-lb.png")


def error_concentration() -> None:
    audit = json.loads((ROOT / "reports/ha43_subgroup_audit.json").read_text())["champion"]
    seeds = list(audit)
    keys = [("earth_ratio", "Earth"), ("deck_g_ratio", "Deck G"),
            ("cryo_zero_spend_ratio", "CryoSleep + zero spend"), ("non_earth_ratio", "Non-Earth")]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = 0.15
    for i, seed in enumerate(seeds):
        ax.bar(np.arange(len(keys)) + (i - 2) * width, [audit[seed][k] for k, _ in keys], width,
               label=f"seed {seed}", color=[NAVY, TEAL, SAND, GOLD, GREY][i])
    ax.axhline(1, color=RED, ls="--", lw=1)
    ax.set_xticks(np.arange(len(keys)), [lab for _, lab in keys])
    ax.set_ylabel("Error rate / overall error rate")
    ax.set_title("Where the champion's errors concentrate (stable on all 5 seeds)")
    ax.legend(ncol=5, fontsize=9, frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "error-concentration.png")


def adversarial() -> None:
    data = json.loads((ROOT / "reports/ha44_adversarial_seeds.json").read_text())["seeds"]
    seeds = list(data)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(seeds))
    ax.bar(x - 0.18, [data[s]["auc"] for s in seeds], 0.36, color=TEAL,
           label="train vs test classifier")
    ax.bar(x + 0.18, [data[s]["permutation_auc"] for s in seeds], 0.36, color=GREY,
           label="same pipeline, permuted labels (chance)")
    ax.axhline(0.5, color=NAVY, lw=0.8)
    ax.set_xticks(x, [f"seed {s}" for s in seeds])
    ax.set_ylim(0.45, 0.56)
    ax.set_ylabel("ROC AUC")
    ax.set_title("Adversarial validation: train and test are indistinguishable")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    save(fig, "adversarial-validation.png")


def stacker_weights() -> None:
    stackers = json.loads((EVIDENCE / "final-stackers.json").read_text())
    names = {"ha12_tabpfn": "frozen", "ha23_ft": "fine-tuned\n(30 ep)",
             "ha24_ftrefit": "fine-tune\n+ refit", "ha39_a1_ne2": "fine-tuned\n(100 ep, 2 est.)"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, (key, title) in zip(axes, [("ha27_stack3", "Official champion (LB 0.82020)"),
                                       ("sub_stack3_plus_a1ne2", "Best public LB (0.82604)")]):
        coef = stackers[key]["coef"]
        members = stackers[key]["members"]
        ax.bar(range(len(coef)), coef, color=[RED if c < 0 else TEAL for c in coef])
        ax.set_xticks(range(len(coef)), [names[m] for m in members], fontsize=9)
        ax.axhline(0, color=NAVY, lw=0.8)
        ax.set_title(title, fontsize=12)
    axes[0].set_ylabel("Logistic-regression weight on member logit")
    fig.suptitle("Stacker weights: the model extrapolates along the fine-tuning direction",
                 fontsize=13, fontweight="bold", color=NAVY)
    fig.tight_layout()
    save(fig, "stacker-weights.png")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    score_progression()
    experiment_landscape()
    submission_history()
    cv_vs_lb()
    error_concentration()
    adversarial()
    stacker_weights()


if __name__ == "__main__":
    main()
