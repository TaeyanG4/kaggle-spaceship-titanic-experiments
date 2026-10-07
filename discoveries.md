# Spaceship Titanic — Discoveries

Shared reusable findings live here. Every agent reads this file.

Verification is per **host**, not per agent: a discovery made on one host (for example Codex) is
cross-checked **once** by a different host (for example Claude Code). Sessions on the same host do
not re-review each other.

Discovery IDs use `D-<Agent>-<NNN>`, numbered by the agent that creates the discovery.

Use this exact format:

```markdown
## D-A-001 — Short title
- Source: A
- Host: Codex
- Cross-check: VERIFIED
- Finding: ...
- Evidence: ...
- Implication: ...
- Reviews:
  - Claude Code (D): CLOSED — reproduced independently with ...
```

`Cross-check` states:

- `PENDING`: no different host has reviewed it yet. New discoveries start here, with an empty `Reviews:` list.
- `REVIEWING <Host> (<Agent>)`: a different-host session has claimed the review.
- `VERIFIED`: a different host reviewed it and recorded `CLOSED`.
- `HOLD`: a different host needs specific evidence, a condition change, or an improvement first.
- `CHALLENGED`: a host found a material contradiction, flaw, or missing assumption.

Rules:

- The source does not review its own discovery.
- Review only discoveries whose `Host` differs from yours and whose `Cross-check` is `PENDING` (or
  `HOLD` when its condition is now met). Claim it first with `REVIEWING <your host> (<your agent>)`.
- One cross-host review is enough; add another only for high-impact or disputed discoveries.
- Review lines use `<Host> (<Agent>): CLOSED | HOLD | CHALLENGED — reason`. `HOLD` and
  `CHALLENGED` must say what would make the discovery acceptable.
- When the source revises a `HOLD` or `CHALLENGED` discovery, it updates `Evidence`, keeps the old
  reviews, and resets `Cross-check` to `PENDING`.
- If only one host is available, a different slot on the same host may cross-check; begin its
  reason with `same host —`.
- Use `VERIFIED` discoveries freely. A plan item built on a `PENDING`, `REVIEWING`, or `HOLD`
  discovery must say so in its `Evidence`. Do not build on `CHALLENGED` discoveries until resolved.

Negative results are discoveries too. When a hypothesis fails or is disproved, record it here
(`Finding: <claim> does not hold under <conditions>`) so no agent proposes it again. Before adding a
plan item, every agent searches this file first.

A discovery may generate many plan hypotheses, and one hypothesis may combine many discoveries.
When moving an idea back into `plan.md`, always include the source IDs plus the concrete `Evidence`
and `Improvement` that justify the new attempt.

Do not archive this file merely because it grows; merge duplicate discoveries instead.

> Migration note (2026-10-05): D-D-001 to D-D-020 were made in pre-orchestrator sessions on
> Codex (host inferred from the local Codex session record for this project; no other Claude Code
> session exists for it). D-D-021 onward were made on Claude Code. Earlier self-review lines
> (`Agent Main: CLOSED`) were removed because the source does not review its own discovery. The
> Claude Code reviews below were done by recomputing the claims from local official data and saved
> artifacts (read-only), plus the H-D-02 runner replay of the baseline.

## Environment and data

## D-D-001 — Local toolchain and project environment are ready
- Source: D
- Host: Codex
- Cross-check: VERIFIED
- Finding: Python 3.12.6, Git 2.45.1, Kaggle CLI 2.2.4, and uv 0.11.29 are installed. uv resolves the environment and `uv.lock`; the `spaceship_titanic` package installs from `src/`.
- Evidence: `scripts/check_env.py` passed; pytest and Ruff were clean at bootstrap.
- Implication: `uv run ...` is the standard execution path; no environment work is blocking.
- Reviews:
  - Claude Code (D): CLOSED — system Python 3.12.6, Git 2.45.1, Kaggle CLI 2.2.4, uv 0.11.29 confirmed; pytest 8 passed and Ruff passed via uv. Note: the uv venv runs Python 3.12.13 (`.python-version` pins 3.12), which is consistent with the claim.

## D-D-002 — Official Kaggle download requires accepted competition rules
- Source: D
- Host: Codex
- Cross-check: VERIFIED
- Finding: `kaggle competitions files` listed files, but `download` returned HTTP 403 until the account joined the competition and accepted its rules.
- Evidence: 403 before acceptance; official download succeeded afterwards (2026-10-05).
- Implication: fix access errors at the account level; never substitute unofficial data copies.
- Reviews:
  - Claude Code (A): CLOSED — historical 403 cannot be replayed after acceptance, but the CLI now lists the official files and the raw data hashes are consistent; nothing contradicts the claim or its rule.

## D-D-003 — Official data shape and target balance
- Source: D
- Host: Codex
- Cross-check: VERIFIED
- Finding: train 8,693 x 14; test 4,277 x 13; sample submission 4,277 x 2. Train target is near-balanced: 4,378 True vs 4,315 False.
- Evidence: shape/target checks after extraction into `data/raw/`.
- Implication: accuracy at threshold 0.5 is a reasonable primary metric; no rebalancing needed.
- Reviews:
  - Claude Code (D): CLOSED — recomputed identical shapes and target counts from `data/raw/`.

## D-D-004 — PassengerId groups never cross train/test; group-aware CV is the contract
- Source: D
- Host: Codex
- Cross-check: VERIFIED
- Finding: 6,217 train groups and 3,063 test groups; train/test group overlap is exactly 0; the largest train group has 8 passengers.
- Evidence: `scripts/audit_data.py`.
- Implication: StratifiedGroupKFold by group is the structurally appropriate validation; random row-wise CV would let group members leak across folds.
- Reviews:
  - Claude Code (D): CLOSED — recomputed 6,217 / 3,063 groups, overlap 0, max group 8.

## Baseline and validation

## D-D-005 — CatBoost SGKF baseline is a stable champion (`baseline-001`)
- Source: D
- Host: Codex
- Cross-check: VERIFIED
- Finding: 5-fold StratifiedGroupKFold CatBoost reaches OOF accuracy 0.818820, log loss 0.384949, 1,575 errors. Fold accuracies 0.814836, 0.828637, 0.824037, 0.818757, 0.807825 (std 0.007216).
- Evidence: `outputs/oof/baseline_catboost_oof.csv`; recomputed in a later session with matching PassengerId order and targets.
- Implication: every challenger is compared on the same frozen seed-42 folds.
- Reviews:
  - Claude Code (D): CLOSED — retrained the same config through the runner (`hm02_base_control`) and reproduced 0.818820 / 0.384949 exactly; fold scores and std also recomputed from the saved OOF.

## D-D-006 — Public LB is about 0.011 below OOF
- Source: D
- Host: Codex
- Cross-check: VERIFIED
- Finding: submission 56847813 scored 0.80780 public LB versus 0.818820 OOF (gap about -0.0110); rank 250 / 1,575 at submission time.
- Evidence: Kaggle submission result.
- Implication: worth auditing shift and robustness, but one public LB value cannot invalidate SGKF or identify a cause. Public LB stays secondary evidence.
- Reviews:
  - Claude Code (A): CLOSED — `kaggle competitions submissions` shows submission 56847813 at public 0.80780; gap to OOF 0.818820 recomputed. The rank snapshot is not reproducible and is incidental.

## D-D-007 — Train/test shift checks so far do not explain the LB gap
- Source: D
- Host: Codex
- Cross-check: VERIFIED
- Finding: missingness is similar. Marginal total variation over HomePlanet, CryoSleep, Destination, VIP, CabinDeck, CabinSide, NoSpend, and GroupSize is 0.0016-0.0231. TotalSpend p90 is 3,838.4 (train) vs 3,572.4 (test). Joint TV: HomePlanet/CryoSleep 0.020474, CabinDeck/CabinSide 0.028994.
- Evidence: `scripts/audit_validation.py`.
- Implication: the gap cause remains unresolved; broader adversarial validation is optional.
- Reviews:
  - Claude Code (D): CLOSED — independently recomputed HomePlanet/CryoSleep joint TV 0.020474, spend p90 3,838.4 vs 3,572.4, and max per-column missing-rate difference 0.0037. Marginal and CabinDeck/CabinSide values not separately recomputed.

## D-D-008 — Baseline errors concentrate on Earth and solo passengers
- Source: D
- Host: Codex
- Cross-check: VERIFIED
- Finding: Earth has 1,152 of 1,575 errors (25.03% error) vs Europa 7.88% and Mars 12.05%. Solo passengers have 19.77% error. 815 OOF probabilities fall in [0.45, 0.55], including 365 errors.
- Evidence: baseline OOF error review.
- Implication: diagnostic segments only; they do not justify segment models or full-OOF threshold optimization.
- Reviews:
  - Claude Code (D): CLOSED — recomputed every number from the saved baseline OOF.

## Feature semantics

## D-D-009 — Zero-filled spending mixes unknown with observed zero
- Source: D
- Host: Codex
- Cross-check: VERIFIED
- Finding: `TotalSpend` fills missing services with 0 before summation, so 406 train / 193 test rows get `NoSpend=1` despite at least one unknown service. None have all five services missing. 908 train rows have at least one missing service.
- Evidence: feature audit of `src/spaceship_titanic/features.py`.
- Implication: separate observed all-zero spending from incomplete observations (tested in v1_spend, D-D-013).
- Reviews:
  - Claude Code (D): CLOSED — recomputed 406 / 193 rows, 908 train rows with a missing service, 0 rows with all five missing; confirmed `fillna(0).sum` in `features.py`.

## D-D-010 — Missing names collapse into one fake family
- Source: D
- Host: Codex
- Cross-check: VERIFIED
- Finding: all missing names share one surname token, so combined train+test counting assigns `SurnameSize=294` to 200 train and 94 test rows.
- Evidence: feature audit.
- Implication: feature-semantics bug, not target leakage. Missing names must not imply a shared family (fixed in v1_name, D-D-013).
- Reviews:
  - Claude Code (D): CLOSED — 200 + 94 missing names recomputed; `_surname` maps all of them to `__MISSING__`.

## D-D-011 — Observed CryoSleep or age < 13 never has positive known spending
- Source: D
- Host: Codex
- Cross-check: VERIFIED
- Finding: in both files, observed CryoSleep=True and observed age below 13 never coincide with positive known spending.
- Evidence: target-free audit on official train+test features.
- Implication: rule-based zero-fill of missing spending is target-free; zero spending must not imply CryoSleep=True. Tested as v1_rules (D-D-013).
- Reviews:
  - Claude Code (D): CLOSED — recomputed 0 coinciding rows for both conditions in both files.

## Experiment infrastructure

## D-D-012 — Config-driven runner with frozen folds is in place
- Source: D
- Host: Codex
- Cross-check: VERIFIED
- Finding: the hard-coded `train_baseline.py` (no fold IDs, no test probabilities, overwritten outputs) was superseded by `scripts/train_experiment.py` / `scripts/run_versions.py`: frozen fold storage, per-experiment IDs, fold models, test probabilities, paired group-bootstrap diagnostics, and versioned outputs. pandas CSV read-back stores the group column with a different string dtype, so fold-cache validation compares values/order, not storage dtype. The host's shared pytest temp directory denied access; tests use workspace-local `tmp_pytest/`.
- Evidence: 8 tests pass (group-fold replay, label independence, missing semantics, peer exclusion, unseen-category adapters); Ruff passes; replay of the best candidate reused matching artifacts.
- Implication: new experiments are a new `configs/<id>.json` plus `train_experiment.py`.
- Reviews:
  - Claude Code (D): CLOSED — used the runner for 17 H-D-02 runs; frozen-fold checks, artifact writing, and baseline reproduction all worked; pytest 8 passed.

## v1/v2 screening results

## D-D-013 — Missing-value semantic fixes give small, non-additive, unconfirmed gains
- Source: D
- Host: Codex
- Cross-check: VERIFIED
- Finding: name-only fix 0.820200 (+0.001380; paired group-bootstrap 95% interval [-0.002192, 0.005028]). Spending semantics 0.819395 (+0.000575; log loss 0.382946). Combined v1 0.816979 (fold 5 stopped at iteration 54, accuracy 0.798619). Rule fill on v1 0.819165. That the semantic fixes add up to a material gain does not hold under seed-42 SGKF screening.
- Evidence: `reports/v1_v2_report.md`, `reports/v1_v2_summary.json`; primary-fold screening only.
- Implication: changes do not add up; the stopping criterion may matter but is not proven causal. Minimal name-fix features are the best base for further small tests.
- Reviews:
  - Claude Code (D): CLOSED — recomputed all four accuracies and spend log loss from saved OOF; bootstrap interval and fold-5 iteration 54 / 0.798619 match the saved metrics.

## D-D-014 — Broad v2 feature families do not improve accuracy or Earth errors
- Source: D
- Host: Codex
- Cross-check: VERIFIED
- Finding: that broad v2 features improve accuracy does not hold under seed-42 SGKF: v2 spending composition 0.817554; group peer summaries/fills/cabin position 0.816289; full v2 CatBoost 0.818820 (ties baseline). Full v2 Earth errors 1,163 vs baseline 1,152; best v1 has 1,149.
- Evidence: v1/v2 suite, same seed-42 SGKF folds.
- Implication: do not revisit group/cabin fills or accumulate broad features without a narrow, distinct hypothesis. Group/spending features did not solve the largest error segment.
- Reviews:
  - Claude Code (D): CLOSED — recomputed the three accuracies and Earth errors from saved OOF.

## D-D-015 — Better probability loss does not translate to accuracy at 0.5
- Source: D
- Host: Codex
- Cross-check: VERIFIED
- Finding: that Logloss stopping, LightGBM, XGBoost, or an equal blend beat baseline accuracy does not hold on full v2 features: 0.815599 / 0.376543; 0.815829 / 0.377652; 0.817784 / 0.378051; blend 0.818590 / 0.374598 (two more errors than baseline). No weights or thresholds were tuned.
- Evidence: v1/v2 suite metrics.
- Implication: log loss improvements are real but accuracy is the metric; calibration/threshold work needs nested group validation.
- Reviews:
  - Claude Code (D): CLOSED — recomputed all accuracy/log-loss pairs from saved OOF; H-D-01 analysis (D-D-021) explains where the accuracy is lost.

## D-D-016 — No v1/v2 candidate met promotion; baseline remains champion
- Source: D
- Host: Codex
- Cross-check: VERIFIED
- Finding: 10 trained configurations (50 fold models, 584 summed training seconds) plus an equal-weight blend, 11 primary candidates in total. Best gain +0.001380 is below +0.002 and its interval includes zero, so seed confirmation was skipped. All OOF/probability/submission artifacts passed row/fold/probability/schema checks; raw data and baseline SHA256 hashes are unchanged.
- Evidence: `reports/v1_v2_validation.json`, `reports/v1_v2_status.json`.
- Implication: `baseline-001` stays champion; `selected_v1_v2.csv` equals the baseline submission.
- Reviews:
  - Claude Code (D): CLOSED — 50 fold-model files present across the 10 configs (plus the historical v1_name run); every saved OOF matches the frozen seed-42 fold column; best gain recomputed as +0.001380.

## External notebook research

## D-D-017 — Several high-scoring public notebooks are not clean predictions
- Source: D
- Host: Codex
- Cross-check: VERIFIED
- Finding: 18 source snapshots (from 100 score-sorted and 30 vote-sorted entries) were inspected without execution. bsthere's 0.82137 replaces model predictions with embedded test-length bits when `use_best_public_override=True` (provenance unknown; do not claim they are recovered labels). arunklenin, ravi20076, abdmental01, jbomitchell, gyogyocat, and r2303254 use existing submission CSVs in final inference.
- Evidence: `reports/clean_notebook_catalog.md` / `.json` (source hashes, code locations).
- Implication: excluded under project rules; official-only input metadata is insufficient evidence of clean predictions.
- Reviews:
  - Claude Code (A): CLOSED — static re-read of the cached sources: bsthere sets `use_best_public_override = True` (L80, applied L570); arunklenin, ravi20076, abdmental01, jbomitchell, gyogyocat (R, `blend1/submission.csv`) and r2303254 (L1987-1988) all read external submission CSVs.
  - Claude Code (A): CLOSED — re-read the cached sources. bsthere L80 `use_best_public_override = True`. External submission reads at arunklenin L2061-2063 (ORed at L2070), ravi_models L1007-1008, abdmental01 L268-271, jbomitchell L27-33, gyogyocat L374, r2303254 L1987-1988. The same override mechanism also appears in eyefulye ×2 (D-A-001).

## D-D-018 — Public notebook CV is often optimistic or contaminated
- Source: D
- Host: Codex
- Cross-check: VERIFIED
- Finding: junaid512 builds global leave-one-out group target features before row-wise CV (held-out peers' labels leak), and its stacker/global threshold search is not independent OOF. cv13j0's NN keeps a supervised KNN feature prepared before outer CV (held pending nested reconstruction). Viktor/Tao/twinpilgrim are one XGBoost lineage; Viktor's disabled Optuna branches would SMOTE before CV if enabled. jimliu's Headjack remote features are unverifiable.
- Evidence: `reports/clean_notebook_catalog.md`.
- Implication: adopt notebook methods only after fold-local/nested group reconstruction; never treat notebook CV as our validation.
- Reviews:
  - Claude Code (A): CLOSED — core claims re-read: cv13j0 builds `knn_kfold_extract` features on all train rows before outer CV; Viktor runs SMOTE at cell 67 after the initial CV, and the only later CVs sit behind `optuna_study == *_study` gates that are false ("ON" vs "OFF"); junaid512 LOO leakage independently confirmed by a full read (D-A-003). The lineage and Headjack remarks were not re-checked.
  - Claude Code (A): CLOSED — junaid512 L290-309 builds a group target sum/LOO mean from all train labels, and CV is row-wise StratifiedKFold (L458, L627). cv13j0 L536-546 calls `knn_kfold_extract` with the full-train target, and `KNN_K1_01` stays in the final features (L554, L640). In Viktor, `LGBM_study`/`XGB_study` are "OFF" (L365-366), while SMOTE at L353 rewrites X/y before those CV branches (L404, L469) and the final fit (L506). jimliu was not re-checked; the claim there is only "unverifiable".

## D-D-019 — Reference public Best Scores for clean-looking method families
- Source: D
- Host: Codex
- Cross-check: VERIFIED
- Finding: Viktor 0.81833 (V242), Tao 0.81646 (V2), twinpilgrim 0.81599 (V2), Dmitry 0.81318 (V41), Misael 0.81271 (V8), Samuel 0.80874 (V47).
- Evidence: Kaggle page Best Scores; latest source may differ from the best-score version.
- Implication: method references only, not reproduced scores or selection targets. Our 0.80780 LB is within this band.
- Reviews:
  - Claude Code (A): HOLD — the scores come from Kaggle page Best Scores, which the cached CLI listings do not contain. Acceptable once someone records the page scores (or a kernels API score field) for the six notebooks.
  - Claude Code (A): CLOSED — re-retrieved the key reference from the public notebook page: viktortaran/space-titanic Best Score 0.81833 at V242 (current version 0.81739), saved in `reports/ha46_provenance.json` (D-A-041); the other five references remain descriptive and were not rechecked.

## D-D-020 — Kaggle skill MCP sample lookup can fail while the CLI works
- Source: D
- Host: Codex
- Cross-check: HOLD
- Finding: the Kaggle skill's automatic sample lookup failed on its MCP read surface even with CLI competition access.
- Evidence: submission dry-run validated against the local official `sample_submission.csv` (4 structural checks passed).
- Implication: validate submissions against the local sample file.
- Reviews:
  - Claude Code (A): HOLD — not reproduced in this session; acceptable after a Kaggle-skill MCP sample lookup is attempted and its failure (or success) recorded alongside the CLI result.

## Claude Code experiments

## D-D-021 — Probability-loss models trade Earth accuracy for non-Earth gains at the 0.5 margin
- Source: D
- Host: Claude Code
- Cross-check: CHALLENGED
- Finding: versus baseline (Earth 1,152 errors, non-Earth 423), Logloss-stopped CatBoost, LightGBM, XGBoost, and the blend add +41/+42/+29/+23 Earth errors while removing 13/16/20/21 non-Earth errors. Same-feature v2 with Accuracy stopping is only +11 Earth, so the stopping metric alone accounts for roughly 30 extra Earth errors. 81-89% of the lost Earth rows sit within ±0.1 of 0.5, and 59-64% of them are truly Transported: these models push borderline Earth rows toward False (predicted positive rate 0.457 -> 0.432-0.445; actual 0.424). Losses concentrate in CryoSleep=True with all-zero observed spending (actual rate 0.655; +23 for Logloss CatBoost), CabinDeck G, CabinSide P, GroupSize 4+, and age 0-12. `v1_name_fix` is neutral on Earth (-3) and gains on non-Earth (-9). Baseline Earth errors themselves are mostly borderline: 873 of 1,152 lie in probability 0.3-0.7, and the 0.45-0.55 band is a coin flip (actual 0.505).
- Evidence: `scripts/audit_earth_errors.py` -> `reports/earth_error_diagnosis.md` / `.json` (saved OOF only, no training). Earth-only paired group-bootstrap 95% CI excludes zero for Logloss CatBoost [-0.0166, -0.0009] and LightGBM [-0.0181, -0.0009]; it includes zero for XGBoost, the blend, v2, and v1_name_fix. Conditional on fitted models; one seed-42 split.
- Implication: Earth's weakness is not a missing feature family. The difference between models is a reshuffle of borderline rows, not a fix. Keep Accuracy stopping as the default CatBoost control and report Earth errors as a diagnostic in every comparison. Do not build Earth-only models or tune per-planet thresholds on full OOF; any threshold/calibration work must be nested.
- Reviews:
  - Claude Code (A): CHALLENGED — same host — rechecked on honest multi-seed OOF (D-A-038): the Earth-for-non-Earth trade appears on 0/3 seeds vs inner-CV Accuracy stopping and 1/5 vs fixed300, so the headline mechanism is a seed-42 scored-fold-stopping artefact; acceptable if narrowed to that scope, keeping only the stable parts (Earth rows pushed toward False; Earth/Deck G error concentration).

## D-D-022 — One SGKF split cannot resolve gains of a few tenths of a percent
- Source: D
- Host: Claude Code
- Cross-check: VERIFIED
- Finding: the H-D-02 grid winner `hm02_name_d8_l5` scored +0.005292 over the champion on the seed-42 split, and its paired group-bootstrap CI vs the champion excluded zero ([+0.0012, +0.0092]). It then lost to the matched control on seed 123 (0.817439 vs 0.821351, −0.003911) and won on seed 2026 (0.819625 vs 0.815714, +0.003911); mean confirmation delta 0.000. The identical control config alone ranges 0.815714-0.821351 across seeds 42/123/2026 (spread 0.0056).
- Evidence: `reports/hm02_summary.json`, `reports/hm02_base_control_seed*_metrics.json`, `reports/hm02_name_d8_l5_seed*_metrics.json`.
- Implication: split-to-split noise is as large as the effects being screened, and picking the max of 12 configs on one split adds selection optimism. The paired bootstrap is conditional on fitted models and misses split/refit variance, so it is not sufficient evidence by itself. Screening should use the mean paired delta over several SGKF seeds.
- Reviews:
  - ChatGPT (C): CLOSED — `hm02_summary.json` independently matches the +0.005292 seed-42 selection gain, the -0.003911/+0.003911 confirmation sign flip, and the 0.0056 control spread across seeds.

## D-D-023 — CatBoost grid direction: shallow or heavily regularized trees are worse
- Source: D
- Host: Claude Code
- Cross-check: VERIFIED
- Finding: on seed 42, depth 4 (L2 3 or 10) and depth 6 / L2 10 underperform the control on both feature sets (−0.0003 to −0.0081). Depth 8 / L2 5 and depth 6 / L2 1 are the top two on both feature sets (+0.0012 to +0.0053). Learning rate 0.03 / 1,500 iterations gives no gain. Earth errors move with overall accuracy (1,129-1,201); no setting trades Earth for non-Earth the way the probability-loss models did (D-D-021).
- Evidence: `reports/hm02_scores.csv` (13 runs, Accuracy stopping, frozen seed-42 folds).
- Implication: weak directional evidence only (single split, see D-D-022). If capacity is revisited, compare depth 6-8 with low L2 under multi-seed screening rather than widening the grid on one split.
- Reviews:
  - ChatGPT (C): CLOSED — the saved H-D-02 score table and metrics reproduce the stated seed-42 ordering; depth-4/heavy-L2 arms are worse and lr=0.03 adds no gain.

## D-D-024 — Small CatBoost capacity/L2 tuning does not beat the champion
- Source: D
- Host: Claude Code
- Cross-check: VERIFIED
- Finding: that a small predeclared CatBoost depth/L2/learning-rate grid (H-D-02: 6 settings × {baseline, v1_name} features, Accuracy stopping) yields a promotable improvement does not hold under matched seed 123/2026 confirmation. The best candidate `hm02_name_d8_l5` failed confirmation (−0.003911, +0.003911).
- Evidence: `reports/hm02_manifest.json` (predeclared rule), `reports/hm02_summary.json` (`promoted: false`), `reports/hm02_scores.csv`.
- Implication: do not re-run this grid on one split. A repeat needs a materially better method, such as multi-seed screening (D-D-022) or early stopping that does not use the scored fold.
- Reviews:
  - ChatGPT (C): CLOSED — the predeclared manifest and summary show `hm02_name_d8_l5` failed matched confirmation with -0.003911/+0.003911 and `promoted=false`.

## D-D-025 — CatBoost capacity / name-fix candidates fail multi-seed screening
- Source: D
- Host: Claude Code
- Cross-check: VERIFIED
- Finding: that the four best H-D-02 configs or `v1_name_fix` beat the baseline config does not hold under multi-seed SGKF screening (seeds 42/123/2026, matched control). Mean paired deltas: name_d6_l1 +0.001764, name_d8_l5 +0.001764, base_d6_l1 +0.001342, base_d8_l5 +0.001227, v1_name_fix +0.000077. Every candidate loses on seed 123 (control 0.821351). Seed 42 was also the selection seed for these configs, so the means are biased upward; without seed 42 the means are −0.0006 to +0.0014.
- Evidence: `reports/hm08_summary.json`, `reports/hm08_manifest.json` (rule predeclared before the 8 new runs); `src/spaceship_titanic/screening.py` with tests.
- Implication: closes CatBoost depth/L2 tuning and the missing-name fix as promotion routes under the current stopping method. Control Earth errors vary 1,147-1,177 across seeds, so a single-seed Earth comparison is also noisy.
- Reviews:
  - ChatGPT (C): CLOSED — `hm08_summary.json` reproduces every listed mean delta and positive-seed count; none reaches the +0.002 screen and every candidate loses on seed 123.

## D-D-026 — Early stopping on the scored fold inflates OOF accuracy by about 0.004
- Source: D
- Host: Claude Code
- Cross-check: VERIFIED
- Finding: the runner's control stops CatBoost (Accuracy eval metric) on the same fold it scores. With the same baseline config/features, a fixed 300 iterations scores 0.815944 / 0.813528 / 0.813183 on seeds 42/123/2026 vs control 0.818820 / 0.821351 / 0.815714 (mean −0.004410; log loss better, 0.382-0.384 vs 0.382-0.391). That an inner stratified group holdout (1/8 of the fit fold, Accuracy metric) gives a less noisy honest estimate does not hold. It is unstable (best iterations 35-440 across folds) and scores 0.804555-0.810192 (mean −0.010545; with refit −0.010583).
- Evidence: `reports/hm09_summary.json`, `reports/hm09_*_metrics.json`, `scripts/run_hm09_stopping.py`. Control runs reused from H-D-02/08.
- Implication: every OOF number from the runner (v1/v2, H-D-02, H-D-08, and baseline-001's 0.818820) carries this optimism. Paired comparisons share it, but per-fold stopping adds noise. An honest baseline estimate is about 0.814-0.816, which explains roughly 0.004 of the 0.011 CV-to-LB gap (D-D-006). New screens should use an honest stopping rule; Accuracy-metric stopping on a small holdout should not be used.
- Reviews:
  - ChatGPT (C): CLOSED — `hm09_summary.json` reproduces fixed300 mean delta -0.004410 and the unstable inner-Accuracy arms (-0.010545/-0.010583); the runner design confirms the old control selected iterations on the scored fold.

## Notebook idea audit (Agent A)

## D-A-001 — Unexecuted ideas in the score-sorted top-100 public notebooks
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: 65 previously unaudited score-sorted notebooks (about 61 distinct after clone merging), the 2 cached-but-unaudited eyefulye notebooks, and arunklenin's feature engineering were read statically. Excluded: 9 build final predictions from other submission CSVs or external features.; eyefulye ×2 override predictions with hard-coded 4,277-bit strings and auto-submit (same mechanism as bsthere).; 2 use an external service or a data copy.; About 10 compute target statistics before CV.; The 11-notebook larjeck clone cluster runs SMOTE before CV. Untested, rule-compatible ideas, each queued as a plan item: group-consistency post-processing (H-A-01);; surname as a CatBoost categorical (H-A-02);; group-id location features, CabinNum regression imputation, and cabin occupancy (H-A-03);; reverse CryoSleep rule with Cabin/VIP group fill (H-A-04);; YDF / Lossguide / MLP diversity members with an inner-fold stacker (H-A-05). Not queued, with reasons in the report: monotone transforms and re-binning (no effect on trees); KNN/iterative imputation and target/WOE/count encodings of low-cardinality categoricals; SMOTE, outlier removal and pseudo-labels; Destination group fill (only 42.9% of groups agree).
- Evidence: `reports/notebook_idea_audit_top100.md` (per-notebook line references from five parallel static reads plus spot checks). Sources are in `.kaggle-research/.../kernel-archives/` (Git-ignored, untrusted, never executed). The CLI list has no scores, so the band is inferred from the score sort (titles 0.82137 to 81.1%).
- Implication: notebook mining in this band is now exhausted. The remaining distinct ideas are group-label structure (post-processing, surname), location from group order, and learner diversity. Feature transforms and imputers that public notebooks add are not new evidence for trees.
- Reviews:

## D-A-002 — Target-free structure behind the notebook ideas
- Source: A
- Host: Claude Code
- Cross-check: VERIFIED
- Finding: Within every deck, corr(GroupId, CabinNum) is 0.980-1.000 (per-deck linear-fit MAE 4.5-34.6). The group id is a near-exact location proxy.; No cabin is shared by two PassengerId groups (0 of 9,825). Occupancy is 1-8.; 89.9% of named test rows have a surname seen in train.; 90.4% of named train rows share their surname with another train group.; CryoSleep is missing with complete zero spend in 87 train / 36 test rows. Among observed rows with complete zero spend, P(CryoSleep) is 0.859, or 0.964 at age ≥ 13.; Unanimous-group fills cover Cabin 87/31, VIP 86/36 and HomePlanet 90/41 rows (train/test).; Multi-member group agreement is 100% for HomePlanet, 92.6% for VIP, 66.7% for Cabin and 42.9% for Destination.
- Evidence: `scripts/audit_notebook_ideas.py` -> `reports/notebook_idea_coverage.txt` (official train+test feature columns only; `Transported` never read).
- Implication: supports H-A-02 to H-A-04. Group-id features and surname statistics are honest under SGKF because test groups are disjoint from train groups but share surnames and the id ordering.
- Reviews:
  - ChatGPT (C): CLOSED — reran `audit_notebook_ideas.py` on official train+test features only and reproduced deck correlations/MAE, 0/9,825 cross-group cabins, surname overlap, CryoSleep rates, and all group-fill coverage/agreement figures.

## D-D-027 — Inner-CV Logloss iteration selection beats a fixed 300 iterations (promoted)
- Source: D
- Host: Claude Code
- Cross-check: VERIFIED
- Finding: on baseline config/features with honest stopping, choosing CatBoost iterations as the median Logloss-best iteration over a 4-fold inner SGKF on each fit fold (`innercv_logloss`; chosen iterations about 360-520) beats fixed300. Screen seeds 42/123/2026: 0.815944 / 0.817784 / 0.818475 vs 0.815944 / 0.813528 / 0.813183 (mean +0.003183, 2/3 positive). Fresh confirmation seeds 7/99: +0.002301 / +0.001495, so it was promoted under the `agents.md` rule. fixed500 (+0.003144, 3/3) and inner-holdout Logloss refit (+0.002607, 3/3) also passed the screen; fixed200 did not (−0.001802). Log loss also improves (seed 7 0.3802 vs 0.3842; seed 99 0.3789 vs 0.3834).
- Evidence: `reports/hm10_summary.json` (`promoted: true`), `reports/hm10_manifest.json` (predeclared), `reports/hm10_*_metrics.json`, `configs/hm10_spec.json`.
- Implication: new current best: with honest stopping, the baseline config was under-trained at about 240-300 iterations; about 400-500 is better. Honest mean OOF over three seeds is about 0.8174 vs about 0.8142 for the honest fixed300 control. This is a gain over an honest control, not a like-for-like comparison with baseline-001's optimistic 0.818820, and the public LB is untested. Future screens should use `innercv_logloss` as the control's stopping rule.
- Reviews:
  - ChatGPT (C): CLOSED — recomputed accuracy directly from saved OOF: screen deltas 0/+0.004256/+0.005292 and fresh 7/99 deltas +0.002301/+0.001495 exactly match the promotion record.

## D-D-028 — The honest-stopping gain transfers to the public LB
- Source: D
- Host: Claude Code
- Cross-check: VERIFIED
- Finding: `hm10_innercv_logloss` (seed-42 fold models) scored public LB 0.81131 vs baseline-001's 0.80780 (+0.00351). The CV-to-LB gap fell from −0.0110 (optimistic 0.818820) to −0.0046 (honest 0.815944). Under both, LB still sits below CV.
- Evidence: Kaggle submission 56855038 (user-requested, 2026-10-05); submission 56847813 for baseline-001.
- Implication: supports D-D-026/027. Honest OOF tracks the public LB better than the old optimistic OOF. The project goal (user, 2026-10-05) is the best clean shared-notebook score, 0.81833 (Viktor, D-D-019); +0.007 to go. LB is secondary evidence: only CV-promoted champions are submitted, and LB scores are never used to choose between candidates.
- Reviews:
  - Claude Code (A): CLOSED — same host — the Kaggle API submissions list shows 56855038 at 0.81131 and 56847813 at 0.80780, as claimed (D-A-041).

## D-D-029 — Seed bagging the champion across SGKF splits does not pass screening
- Source: D
- Host: Claude Code
- Cross-check: VERIFIED
- Finding: that averaging `hm10_innercv_logloss` OOF probabilities over SGKF seeds beats a single seed does not hold. Bag(42,123,2026) accuracy 0.818014 vs singles 0.815944 / 0.817784 / 0.818475 (mean +0.000614, 2/3 positive; screen fails). Bag(7,99) 0.815484 vs singles 0.813988 / 0.816749 (descriptive only; no confirmation run because the screen failed).
- Evidence: `reports/hm11_summary.json`, `scripts/run_hm11_bagging.py` (no training).
- Implication: the 5-fold CatBoost average is already stable. Bagging over splits adds little accuracy, so variance reduction is not the route to the LB goal.
- Reviews:
  - ChatGPT (C): CLOSED — `hm11_summary.json` and the saved five champion OOF sets agree with bag3 accuracy 0.818014, mean delta +0.000614, 2/3 positives, and no confirmation because the screen failed.

## D-A-003 — junaid512 does not use test labels; its group-label rules never reach test and its hard lock hurts
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: the full latest source (SHA256 prefix 96f7836e0f386938, same as the 2026-10-05 cache) reads only the official files and builds its submission from its own models plus rules. There are no external CSVs, override bits or test labels. It does have three flaws: **Validation leakage.** A train-wide groupmate-label LOO feature (L274-333) is built before row-wise StratifiedKFold, so groupmates' labels leak into CV.; **In-sample selection.** The LR stacker, blend weight, 0.35-0.65 threshold, MLP blend and calibration are all chosen on the same OOF (L727-744, L930-960, L1043-1057).; **Pseudo-labels.** Confident test predictions (≥ 0.92) are fed back over 2 rounds (L790-849). The "group label propagation" reaches 0 of 4,277 test rows because train/test groups never overlap (D-D-004). Its test value is a constant 0.5, and its R2/R3 locks and the Priority-2 override never fire. Groupmate labels are only weakly shared: 43.6% of multi-member train groups have a unanimous label, and the LOO groupmate mean predicts a row with accuracy 0.578. That a hard lock "CryoSleep (observed or inferred) & NoSpend → Transported" (R1, 3,135 train / 1,582 test rows, precision 0.8147) improves accuracy does not hold. Applied to the champion's honest OOF it changes seeds 42/123/2026/7/99 by −0.001726 / −0.003796 / −0.002761 / −0.002301 / −0.003221.
- Evidence: `scripts/audit_notebook_rules.py` -> `reports/notebook_rule_checks.txt`; source `.kaggle-research/spaceship-titanic/kernel-archives/junaid512/source_code.txt` (read only, never executed). Deck → HomePlanet crosstab over train+test: A/B/C/T only Europa, G only Earth.
- Implication: the notebook is not cheating, but its reported CV ("0.90+" dashboard) is not evidence, and its group-label machinery is inert or harmful on test. Usable ideas were merged into H-A-02 (SurnameGroupSize), H-A-03 (CabinNum parity), H-A-04 (both CryoSleep rules, deck → HomePlanet), H-A-05 (ExtraTrees/HistGB members) and H-A-06 (spend extras). Not queued: group or cabin label propagation (no test reach; cabins never cross groups, D-A-002);; hard rule locks (negative above);; row-wise OOF target encoding of low-cardinality columns;; Optuna on leaked features;; log transforms, age bands and cabin buckets (no effect on trees);; more seeds (D-D-029);; noise injection and FT-Transformer/TabNet (no evidence; heavy). Isotonic calibration and threshold choice belong to H-D-06. Pseudo-labelling stays held until the user decides; it is not label leakage, but it needs nested evaluation.
- Reviews:

## D-D-030 — Removing baseline feature groups does not help; position features matter most
- Source: D
- Host: Claude Code
- Cross-check: VERIFIED
- Finding: that dropping any of five predeclared baseline feature groups improves the champion does not hold (honest `innercv_logloss`, seeds 42/123/2026, matched control = champion runs). Mean deltas: family counts (SurnameSize, IsAlone) +0.000230 (2/3 positive, below +0.002); VIP −0.000230; age flags −0.002147; spending aggregates (TotalSpend, NoSpend) −0.004525; positional ids (GroupMember, CabinNum) −0.007017 (0/3, Earth errors +46 to +73).
- Evidence: `reports/hm03_summary.json`, `configs/hm03_spec.json` (rationale predeclared).
- Implication: the baseline set has no harmful group. Spending aggregates help trees despite their semantic flaw (D-D-009), and cabin/group position is the strongest removable signal, which supports location-style features (consistent with D-A-002, PENDING). Do not repeat subtraction-based selection on these groups.
- Reviews:
  - ChatGPT (C): CLOSED — independently recomputed all five candidate deltas from saved OOF; they match +0.000230/-0.000230/-0.002147/-0.004525/-0.007017 and no arm passes.

## D-D-031 — No train/test shift: adversarial AUC 0.486
- Source: D
- Host: Claude Code
- Cross-check: VERIFIED
- Finding: that a multivariate train-vs-test shift explains the remaining CV-to-LB gap does not hold. A group-split (SGKF by PassengerId group) CatBoost adversarial classifier on the baseline features reaches AUC 0.486 (chance level). Train rows' test-likeness median is 0.33 and 99th percentile 0.45.
- Evidence: `reports/adversarial_validation.json`, `scripts/audit_adversarial.py`.
- Implication: after the honest-stopping correction (D-D-026/028), the remaining about 0.005 CV-to-LB gap is best treated as public-LB sampling noise plus any residual selection optimism, not as shift. Reweighting or shift-specific features are not warranted. Expect single LB readings to move by several thousandths for reasons unrelated to model quality.
- Reviews:
  - Claude Code (A): CLOSED — same host — reproduced on seeds 42/123/2026 with saved row-level OOF and a group-level permutation reference (D-A-040): AUC 0.483-0.495, below the 0.506-0.520 chance reference.

## D-D-032 — Compact-feature XGBoost is clearly worse than the CatBoost champion
- Source: D
- Host: Claude Code
- Cross-check: VERIFIED
- Finding: that XGBoost on compact baseline / v1_name features rivals CatBoost does not hold under honest inner-CV Logloss iteration selection (fold-fitted ordinal encoding; best iterations about 400-570). Mean paired delta vs the champion: −0.005752 (baseline features), −0.005598 (v1_name), 0/3 seeds positive. Earth errors 1,187-1,224 vs champion 1,158-1,185.
- Evidence: `reports/hm05_summary.json`, `scripts/xgb_runner.py`, `configs/hm05_spec.json`.
- Implication: CatBoost's native categorical handling matters here. XGBoost is only useful as a diversity member (see D-D-033).
- Reviews:
  - ChatGPT (C): CLOSED — direct OOF recomputation gives mean deltas -0.005752 (baseline) and -0.005598 (v1_name), both 0/3 positive, matching the report.

## D-D-033 — Blending with XGBoost or a nested threshold does not beat the champion
- Source: D
- Host: Claude Code
- Cross-check: VERIFIED
- Finding: on saved OOF (seeds 42/123/2026), with parameters chosen only on the other outer folds, candidates vs the champion at threshold 0.5: equal CatBoost+XGBoost average −0.002607; nested blend weight (grid 0-1) −0.000077; nested threshold (0.44-0.56) +0.000882 (2/3 positive). None reach +0.002.
- Evidence: `reports/hm06_summary.json`, `scripts/run_blend_screen.py`, `configs/hm06_spec.json`.
- Implication: threshold 0.5 stays. A weaker GBDT adds no complementary signal. Ensembling needs members that are both strong and different, not another tree booster on the same features.
- Reviews:
  - ChatGPT (C): CLOSED — `run_blend_screen.py` selects weight/threshold on the other outer folds only, and `hm06_summary.json` reproduces -0.002607/-0.000077/+0.000882; none passes.

## D-D-034 — v1/v2 features and capacity changes stay flat under honest stopping
- Source: D
- Host: Claude Code
- Cross-check: VERIFIED
- Finding: that v1/v2 feature families or depth/L2 changes improve the champion does not hold even with honest inner-CV Logloss iteration selection and multi-seed screening (control = champion runs). Mean paired deltas: v1_spend +0.000690, v1_rules +0.000345, d8_l5 +0.000192, v2_spending +0.000038, v1_name 0.000000, d6_l1 −0.001227, v2 −0.002147, v2_group −0.003413. None pass.
- Evidence: `reports/hm12_summary.json`, `configs/hm12_spec.json`.
- Implication: confirms D-D-013/014/025 under the stronger method. The earlier null results were not artefacts of optimistic stopping. Group peer summaries/fills (v2_group) hurt. The baseline CatBoost is at a plateau for these feature and capacity directions; further gains need structurally new information, such as the location and group-label structure in D-A-002.
- Reviews:
  - ChatGPT (C): CLOSED — `hm12_summary.json` reproduces the eight stated multi-seed deltas and confirms that every arm fails the normal screen under `innercv_logloss`.

## D-D-035 — In-fold supervised KNN distance features hurt the champion
- Source: D
- Host: Claude Code
- Cross-check: VERIFIED
- Finding: that cv13j0-style supervised KNN features (k=1 standardized distance to the nearest class-0 and class-1 fit-fold row, plus their difference) help once rebuilt strictly inside each outer fold (inner 5-fold StratifiedGroupKFold for fit rows; fit-fold-only scaler/imputer) does not hold. Mean paired delta vs the champion −0.008628 (0/3 seeds; 0.810997 / 0.811457 / 0.803865); Earth errors 1,214-1,270 vs 1,158-1,185.
- Evidence: `reports/hm07_summary.json`, `scripts/knn_runner.py`, `configs/hm07_spec.json`.
- Implication: closes D-D-018's held cv13j0 KNN idea. Label-derived neighbour features add noise that trees overfit; the notebook's apparent value likely came from its non-group-aware construction. Do not reuse supervised neighbour or target-derived features without a materially new mechanism.
- Reviews:
  - ChatGPT (C): CLOSED — `knn_runner.py` builds supervised distances inside group-aware inner folds, and direct OOF recomputation matches mean delta -0.008628 with 0/3 positive.

## D-A-004 — Group-consistency post-processing of uncertain predictions hurts
- Source: A
- Host: Claude Code
- Cross-check: VERIFIED
- Finding: that replacing a multi-member passenger's borderline champion probability (|p − 0.5| < 0.10) with its confident groupmates' mean (|p − 0.5| ≥ 0.20) improves accuracy does not hold. It changes 191-200 predictions per seed and gives deltas −0.001841 / −0.001956 / −0.002301 on seeds 42/123/2026 (0/3 positive). Earth errors rise by 11-27.
- Evidence: `reports/ha01_summary.json`, `scripts/run_ha01_group_consistency.py` (predeclared B=0.10, C=0.20; champion honest OOF; no training).
- Implication: consistent with weak group co-transport (D-A-003): groupmates' predictions are not a reliable tiebreaker. Closes notebook-style group snapping; no grid search was run because the predeclared setting already failed by about 0.002.
- Reviews:
  - ChatGPT (C): CLOSED — the script is prediction-only with predeclared B=0.10/C=0.20, and the saved screen reproduces -0.001841/-0.001956/-0.002301 with 0/3 positive.

## D-A-005 — A full-data refit does not beat the 5-model fold-average recipe
- Source: A
- Host: Claude Code
- Cross-check: VERIFIED
- Finding: in a nested proxy inside each outer SGKF fold at the champion's recorded iteration, the idea that one model on 100% of the fit fold beats the current recipe (mean of 5 models on 80%) does not hold. Control / B1 (single refit, same iterations) / B2 (single refit, 1.25× iterations): seed 42 0.819855 / 0.815944 / 0.816979; seed 123 0.812953 / 0.817784 / 0.817899; seed 2026 0.816289 / 0.818475 / 0.816864. Mean delta B1 +0.001035 and B2 +0.000882 (both 2/3 positive); neither reaches +0.002.
- Evidence: `reports/ha07_summary.json`, `reports/ha07_proxy_seed*.json`, `scripts/run_ha07_refit.py`.
- Implication: keep the 5-fold-average submission recipe. The sign flips across seeds, so the final-fit choice is within split noise and not a route to the LB goal.
- Reviews:
  - ChatGPT (C): CLOSED — inspected the nested proxy construction and reproduced summary deltas +0.001035/+0.000882; both flip sign across seeds and remain below the +0.002 screen.

## D-A-006 — Surname identity (CatBoost categorical) does not help
- Source: A
- Host: Claude Code
- Cross-check: VERIFIED
- Finding: that adding the surname as a CatBoost categorical feature (fold-local ordered target statistics; missing names as their own level) beats the champion does not hold. Mean paired deltas: surname −0.001572 (0/3), + within-group surname repeat count −0.000959 (1/3), + SurnameGroupSize −0.000422 (1/3). Champion recipe, seeds 42/123/2026.
- Evidence: `reports/ha02_summary.json`, `configs/ha02_spec.json`, `scripts/extra_features.py`.
- Implication: the cross-group family-label signal implied by surname overlap (D-A-002) is too weak or too noisy to use. As with group label structure (D-A-003, D-A-004), family identity does not predict transport reliably. Closes surname-based features.
- Reviews:
  - ChatGPT (C): CLOSED — direct OOF recomputation matches surname/repeat/group-size deltas -0.001572/-0.000959/-0.000422; feature code uses no target information.

## D-A-007 — Location features from group order and cabin occupancy do not help
- Source: A
- Host: Claude Code
- Cross-check: VERIFIED
- Finding: that location features (numeric GroupId; plus per-deck GroupId→CabinNum imputation with the deck from the row or its groupmates, filling 100 train / 37 test rows; cabin occupancy; GroupId + CabinNum parity) beat the champion does not hold. Mean paired deltas: group_id −0.000767 (0/3), cabin_size −0.000844 (2/3), group_id_fill −0.001419 (0/3), cabin_parity −0.001841 (0/3).
- Evidence: `reports/ha03_summary.json`, `configs/ha03_spec.json`, `scripts/extra_features.py`.
- Implication: CabinNum/CabinDeck already carry the location signal that D-D-030 found valuable, and GroupId is near-redundant with it (D-A-002). Adding it again, imputing the 2% missing cabins, or parity gives nothing. Closes location engineering beyond the raw cabin fields.
- Reviews:
  - ChatGPT (C): CLOSED — direct OOF recomputation matches all four negative means; the GroupId/CabinNum fill/occupancy/parity code is target-free and uses only documented train+test structure.

## D-A-008 — CryoSleep inference rules and deterministic fills do not help
- Source: A
- Host: Claude Code
- Cross-check: VERIFIED
- Finding: that filling missing CryoSleep from spending (False if any known spend > 0; True if all five are observed zero and age ≥ 13) beats the champion does not hold: −0.001764 (0/3). Adding unanimous-group Cabin/VIP fills and deck → HomePlanet gives +0.000460 (1/3), below +0.002. Earth errors are unchanged within noise.
- Evidence: `reports/ha04_summary.json`, `configs/ha04_spec.json`, `scripts/extra_features.py`.
- Implication: CatBoost already learns these relations from spend splits and the explicit `__MISSING__` level, so hard imputation removes information about missingness. Closes notebook-style rule imputation (with D-D-013, D-A-003).
- Reviews:
  - ChatGPT (C): CLOSED — direct OOF recomputation gives -0.001764 for CryoSleep rules and +0.000460 for rules+fills, and the fill implementation is target-free.

## D-A-009 — Non-CatBoost and Lossguide members neither beat nor usefully complement the champion
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: that a different-inductive-bias member, alone or stacked, beats the champion does not hold. Singles vs the champion (seeds 42/123/2026): HistGB with inner-CV log-loss iterations −0.002991, CatBoost Lossguide (31 leaves) −0.003451, ExtraTrees −0.011619 (all 0/3). Error correlation with the champion is 0.81 / 0.86 / 0.73. A nested stack of HistGB+ExtraTrees with the champion gives equal blend +0.000230, nested weight −0.000729, nested threshold +0.000882; none pass.
- Evidence: `reports/ha05_summary.json`, `reports/ha05_stack_summary.json`, `configs/ha05_spec.json`, `configs/ha05_stack_spec.json`, `scripts/sklearn_runner.py`, `scripts/run_blend_screen.py`.
- Implication: weaker members dilute rather than complement the champion, which agrees with D-D-032/033 for XGBoost. No dependency (ydf/MLP) is worth adding on this evidence. Closes learner diversity as a route to the goal.
- Reviews:

## D-A-010 — The champion's plateau is signal-limited: most errors are persistent and confident
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: across the champion's honest OOF on seeds 42/123/2026/7/99, 1,374 rows (15.8%) are wrong on ≥4 of 5 seeds and 712 on 1-3. Persistent errors are 66% of all rows ever wrong. Persistent errors sit far from 0.5 (mean |p − 0.5| = 0.186, seed std 0.041; 307 rows ≥ 0.3 away), while split-specific errors sit at the boundary (0.037, std 0.060). Only two target-free segments pass the predeclared ≥100 rows and ≥1.5× persistent-rate bar: CabinDeck G (2,559 rows, 1.57×, the Earth deck already covered by D-D-021) and Destination PSO J318.5-22 (796 rows, 1.58×, marginal).
- Evidence: `reports/ha08_persistent_errors.json`, `reports/ha08_persistent_segments.csv`, `scripts/audit_persistent_errors.py` (read-only over saved OOF; no training).
- Implication: the remaining error is mostly signal the current features do not carry (confidently wrong rows), not variance. That explains why bagging, blending, capacity, and feature-family screens all stayed flat (D-D-024/025/029/033/034, D-A-005/009). Further feature-side search on these columns has low expected value. Variance-side fixes can reach at most the ~712 boundary rows.
- Reviews:

## D-A-011 — One-hot encoding the low-cardinality categoricals does not help
- Source: A
- Host: Claude Code
- Cross-check: VERIFIED
- Finding: that raising CatBoost `one_hot_max_size` to 10 (all six low-cardinality categoricals one-hot instead of ordered CTR; everything else fixed) beats the champion does not hold: 0.814794 / 0.820775 / 0.816864 on seeds 42/123/2026, mean delta +0.000077 (1/3 positive).
- Evidence: `reports/ha10_summary.json`, `configs/ha10_spec.json`.
- Implication: the categorical representation is not a bottleneck, which agrees with D-A-010's signal-limited plateau. Closes low-cardinality encoding tuning without a threshold grid, as H-C-03 predeclared.
- Reviews:
  - ChatGPT (C): CLOSED — direct OOF recomputation matches mean delta +0.000077 with only 1/3 seeds positive; the isolated `one_hot_max_size=10` arm does not pass.

## D-A-012 — Nested pseudo-labelling adds no signal
- Source: A
- Host: Claude Code
- Cross-check: VERIFIED
- Finding: that agreement-filtered pseudo-labels (two teachers both ≤0.03 or ≥0.97 on a held-out 1/5 of each fit fold, true labels hidden) improve a student scored on the untouched outer fold does not hold. Student vs teacher-only control: seed 42 0.815139 vs 0.815024, seed 123 0.815944 vs 0.815944, seed 2026 0.809732 vs 0.811227; mean −0.000460 (1/3 positive). 1,166-1,310 pseudo-rows were accepted per seed, with only 3-11 wrong.
- Evidence: `reports/ha11_summary.json`, `reports/ha11_seed*.json`, `scripts/run_ha11_pseudo.py` (simulation only; the real test set was never used).
- Implication: the high-confidence rows are ones the model already gets right, so they carry no new information (consistent with D-A-010). Closes pseudo-labelling; no real-test step is warranted.
- Reviews:
  - ChatGPT (C): CLOSED — inspected the nested simulation: pseudo-labels come only from a held-out part of each fit fold and the real test is never read; saved results match mean delta -0.000460 and screen failure.

## D-A-013 — CatBoost Ordered boosting does not beat Plain boosting
- Source: A
- Host: Claude Code
- Cross-check: VERIFIED
- Finding: that switching only `boosting_type` to `Ordered` (features, depth, L2, LR, folds and inner-CV iteration selection fixed) beats the Plain-boosting champion does not hold: 0.812378 / 0.819165 / 0.817324 on seeds 42/123/2026, mean delta −0.001112 (1/3 positive). It also runs several times slower.
- Evidence: `reports/ha09_summary.json`, `configs/ha09_spec.json`.
- Implication: the boosting scheme is not the bottleneck either, consistent with D-A-010. The CatBoost configuration axes tested so far (capacity, stopping, growth policy, categorical encoding, boosting type) are all closed except iteration selection (D-D-027).
- Reviews:
  - ChatGPT (C): CLOSED — direct OOF recomputation matches -0.001112 mean delta with 1/3 seeds positive; the isolated Ordered arm fails the screen.

## D-A-014 — Viktor's public notebook model scores about 0.805 under honest validation and does not help in a blend
- Source: A
- Host: Claude Code
- Cross-check: VERIFIED
- Finding: that the best clean public notebook's model (page Best Score 0.81833) matches or complements the champion does not hold. Re-implemented from its source with SGKF and fold-local SMOTE (faithful arm: author's fixed drop list + SMOTE), it scores 0.803865 / 0.805016 / 0.806166 (seeds 42/123/2026, mean −0.012385 vs the champion); without drop list and SMOTE 0.806396 / 0.807661 / 0.805246 (−0.010967). Its error correlation with the champion is the lowest seen (0.66-0.70, about 860 disagreements), yet nested blends give at best +0.000230 (faithful, nested weight) and an equal blend −0.000307 / −0.003029.
- Evidence: `reports/ha17_summary.json`, `reports/ha17_blend_viktor_plain_summary.json`, `reports/ha17_blend_viktor_faithful_summary.json`, `scripts/viktor_runner.py`, `configs/ha17_spec.json`.
- Implication: the 0.81833 public score is not reproduced by its model under group-aware CV (its own CV is row-wise, D-D-018), so it is a public-LB reading, not a better model. Low error correlation alone does not make a useful blend partner when the member is about 0.012 weaker. The project goal of 0.81833 should be read as an LB-noise-inflated benchmark.
- Reviews:
  - ChatGPT (C): CLOSED — direct OOF recomputation matches -0.010967/-0.012385 for plain/faithful arms; error correlations independently recompute to 0.661-0.695 and the saved nested-blend summaries show no passing blend.

## D-A-015 — Fold-local label-noise down-weighting hurts
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: that down-weighting fit-fold rows whose inner OOF strongly contradicts their label improves the champion does not hold. Dropping rows with a contradiction margin ≥0.35 (172-195 rows per fold) gives 0.812148 / 0.812378 / 0.814218 (mean −0.004486, 0/3); half-weighting rows with margin ≥0.25 (320-352 per fold) gives 0.814909 / 0.817439 / 0.815484 (−0.001457, 0/3). Iterations were re-chosen by the champion rule (about 480-740).
- Evidence: `reports/ha13_summary.json`, `reports/ha13_*_metrics.json`, `scripts/noise_runner.py`, `configs/ha13_spec.json`.
- Implication: the confidently contradicted rows are not removable noise; they carry signal the model needs, even if it cannot fit them. Together with D-A-010/012, this closes training-signal manipulation (pseudo-labels, noise weighting) as a route.
- Reviews:

## D-A-016 — A lower learning rate (0.02) does not beat the champion's 0.05
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: that learning rate 0.02 with inner-CV Logloss iteration selection (chosen about 910-1,500 iterations) beats the champion does not hold: 0.816979 / 0.817094 / 0.815369 on seeds 42/123/2026, mean −0.000920 (1/3 positive). Log loss improves slightly (0.3781 on seed 42 vs 0.3802-0.3842 for the champion family).
- Evidence: `reports/ha16_summary.json`, `reports/ha16_lr02_metrics.json`, `configs/ha16_spec.json`.
- Implication: closes the last CatBoost configuration axis under the honest rule (with D-D-025/027, D-A-009/011/013). Accuracy at 0.5 is saturated for this learner on these columns.
- Reviews:

## D-A-017 — TabPFN v3.5 beats the CatBoost champion by about 0.008 and is promoted
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: TabPFN v3.5 (tabpfn 9.1.0, `create_default_for_version(V3_5)`, GPU) fitted fold-locally on the baseline features (ordinal-encoded categoricals, encoder fitted on the fit fold) scores 0.826182 / 0.825032 / 0.824341 on SGKF seeds 42/123/2026 vs the CatBoost champion's 0.815944 / 0.817784 / 0.818475 (mean +0.007784, 3/3 positive), and 0.823076 / 0.824571 on fresh confirmation seeds 7/99 (+0.009088 / +0.007822). Log loss 0.356-0.359 vs about 0.38. Earth errors fall to 1,133-1,139 from 1,158-1,185. Error correlation with the champion is 0.79-0.80 (511-533 disagreements). Public LB 0.81786 (submission 56865289) vs 0.81131.
- Evidence: `reports/ha12_summary.json`, `reports/ha12_tabpfn*_metrics.json`, `configs/ha12_spec.json`, `scripts/tabpfn_runner.py`; Kaggle submission 56865289.
- Implication: new current best: `ha12_tabpfn` (TabPFN v3.5, baseline features), honest SGKF OOF mean 0.825185 over seeds 42/123/2026, public LB 0.81786. A prior-based learner broke the tree plateau of D-A-010. The CV-to-LB gap widened to about −0.007 (CatBoost: −0.0046); if TabPFN v3.5's pretraining included the public train set, OOF could be inflated, but the +0.0066 LB gain shows most of the improvement is real. Future screens should use `ha12_tabpfn` as the control.
- Reviews:

## D-A-018 — Blending the CatBoost model into TabPFN v3.5 does not help
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: that a nested blend of TabPFN v3.5 with the former CatBoost champion beats TabPFN alone does not hold: equal blend −0.001956 (0/3), nested weight 0.000000 (1/3), nested threshold on TabPFN −0.002262 (0/3), on seeds 42/123/2026 with TabPFN as the control.
- Evidence: `reports/ha19_summary.json`, `configs/ha19_spec.json` (saved OOF only).
- Implication: despite 511-533 disagreements per seed, CatBoost adds nothing TabPFN lacks; the nested weight search picks (almost) pure TabPFN. Keep threshold 0.5 and TabPFN alone; blending partners must be at least as strong as TabPFN.
- Reviews:

## D-A-019 — Other tabular foundation models (TabPFN v2.5, TabICL) neither beat nor complement TabPFN v3.5
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: that TabPFN v2.5 or TabICL 2.2.0 beats or usefully blends with the TabPFN v3.5 champion does not hold (seeds 42/123/2026, v3.5 runs as control). TabPFN v2.5: 0.819970 / 0.816404 / 0.816749 (mean −0.007477, 0/3; error correlation with v3.5 0.84). TabICL: 0.803635 / 0.811342 / 0.805706 (mean −0.018291, 0/3; error correlation 0.62-0.72). Nested blends with v3.5: v2.5 best −0.000345, TabICL best −0.000383; equal blends −0.003183 / −0.007362.
- Evidence: `reports/ha18_summary.json`, `reports/ha18b_summary.json`, `reports/ha18_blend_summary.json`, `reports/ha18b_blend_summary.json`, `scripts/tabpfn_runner.py` (version v2.5), `scripts/tabicl_runner.py`, `configs/ha18_spec.json`, `configs/ha18b_spec.json`.
- Implication: within the foundation-model family the newest TabPFN (v3.5) is clearly strongest here; older or different in-context models are weaker and do not add signal even when their errors differ. The "Tab" family does not need a wider sweep.
- Reviews:

## D-A-020 — Other feature sets do not help TabPFN v3.5
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: that TabPFN v3.5 does better on v1_spend, v1_rules, or a raw-minimal set (baseline minus TotalSpend/NoSpend/SpendMissingCount/age flags/SurnameSize/IsAlone) than on the baseline features does not hold: v1_rules −0.000652, v1_spend −0.001419, raw_minimal −0.004793 (all 0/3, seeds 42/123/2026, `ha12_tabpfn` as control).
- Evidence: `reports/ha20_summary.json`, `configs/ha20_spec.json`, `scripts/tabpfn_variant_runner.py`.
- Implication: the baseline engineered features help TabPFN too (removing them costs about 0.005); v1 semantics fixes are neutral to slightly negative. Feature-set search for TabPFN is closed on these columns.
- Reviews:

## D-A-021 — TabPFN with full-fold context is consistently but only slightly better than the 5-model 80%-context average
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: inconclusive — in a nested proxy (inside each outer SGKF fold), one TabPFN v3.5 fit conditioned on the whole fit fold beats the mean of five fits each conditioned on 4/5 of it on every seed (42: 0.826182 vs 0.824917; 123: 0.825032 vs 0.822731; 2026: 0.824341 vs 0.823651; mean +0.001419, 3/3 positive) but stays below the +0.002 screening bar, so it was not confirmed or promoted.
- Evidence: `reports/ha21_summary.json`, `reports/ha21_seed*.json`, `scripts/run_ha21_tabpfn_refit.py`.
- Implication: for an in-context learner, more context helps in a consistent direction (unlike CatBoost, D-A-005), so a full-train-context submission is the more natural final recipe; under the project rule it is not promoted on this evidence alone. The current submission (fold average) stays.
- Reviews:

## D-A-022 — A larger TabPFN internal ensemble (32) does not change accuracy
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: that `n_estimators=32` beats TabPFN v3.5's default (`auto`) does not hold: 0.826182 / 0.825032 / 0.824917 vs 0.826182 / 0.825032 / 0.824341 on seeds 42/123/2026 (mean +0.000192, 1/3 positive); log loss is essentially unchanged (0.3569-0.3570 vs 0.3563-0.3585).
- Evidence: `reports/ha22_summary.json`, `configs/ha22_spec.json`, `scripts/tabpfn_ne_runner.py`.
- Implication: the default auto ensemble is already saturated here; internal ensemble size is not a lever.
- Reviews:

## D-A-023 — Fold-local TabPFN fine-tuning is consistently better but just under the bar
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: inconclusive — fine-tuning TabPFN v3.5 inside each fit fold (30 epochs max, lr 1e-5, early stopping on log loss over an SGKF(8) slice of the fit fold) beats the frozen champion on every seed (42: 0.827332 vs 0.826182; 123: 0.827792 vs 0.825032; 2026: 0.826412 vs 0.824341; mean +0.001994, 3/3 positive), lowers log loss on seeds 42/123 (0.3503 / 0.3520 vs 0.3563 / 0.3568), and cuts Earth errors (1,111-1,129 vs 1,133-1,139). The mean is 0.000006 below the +0.002 screening bar, so it was not confirmed or promoted. About 50-150 s per fold on the GPU.
- Evidence: `reports/ha23_summary.json`, `reports/ha23_ft*_metrics.json`, `scripts/tabpfn_ft_runner.py`, `configs/ha23_spec.json`.
- Implication: fine-tuning moves TabPFN in a consistent direction. Its context excluded the 12.5% early-stopping slice, which D-A-021 suggests costs accuracy, so a refit on the whole fit fold at the selected epoch count is the natural, materially different next test (H-A-24). The project rule is not relaxed for a near miss.
- Reviews:

## D-A-024 — Fine-tune-then-refit on the whole fold improves log loss, not accuracy
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: that choosing the fine-tuning epoch count on an SGKF(8) slice and refitting on the whole fit fold (full context) beats the frozen TabPFN v3.5 champion by the screening bar does not hold: 0.826987 / 0.825722 / 0.825147 vs 0.826182 / 0.825032 / 0.824341 on seeds 42/123/2026 (mean +0.000767, 3/3 positive). Log loss is the best seen (0.3483 / 0.3489 / 0.3534 vs about 0.357). Chosen epoch counts vary widely (5-30 per fold).
- Evidence: `reports/ha24_summary.json`, `reports/ha24_ftrefit*_metrics.json`, `scripts/tabpfn_ftrefit_runner.py`, `configs/ha24_spec.json`.
- Implication: full context did not add to the fine-tuning gain (H-A-23 +0.001994 vs H-A-24 +0.000767); both fine-tuning variants sharpen probabilities more than they move rows across 0.5. Fine-tuned and frozen TabPFN are near-equal-strength variants of one family, which makes them the first strong same-level blend candidates (H-A-25).
- Reviews:

## D-A-025 — Blending frozen and fine-tuned TabPFN is consistently but marginally better
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: inconclusive — an equal blend of frozen TabPFN v3.5 (`ha12_tabpfn`) with the two fine-tuned variants (`ha23_ft`, `ha24_ftrefit`) beats the frozen model on every seed (mean +0.001917, 3/3); the nested-weight blend gives +0.001879 (3/3); a nested threshold on the frozen model −0.002262. Both blends sit just under the +0.002 bar, as did H-A-23 (+0.001994, 3/3). Per seed: seed 42: control 0.826182, equal 0.828253, nested weight 0.828138; seed 123: control 0.825032, equal 0.826872, nested weight 0.827102; seed 2026: control 0.824341, equal 0.826182, nested weight 0.825952.
- Evidence: `reports/ha25_summary.json`, `configs/ha25_spec.json` (saved OOF only).
- Implication: the fine-tuning family repeatedly gives about +0.0015-0.002, always 3/3 positive, which three screen seeds cannot separate from the bar. More fresh seeds are needed to settle it (H-A-26); the project rule is not relaxed after the fact.
- Reviews:

## D-A-044 — StratifiedGroupKFold gives a different split under another scikit-learn version
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: `StratifiedGroupKFold(5, shuffle=True, random_state=42)` on the same rows, labels and travel groups assigns 6,881 of 8,693 rows to a different fold under scikit-learn 1.6.1 (Kaggle image: numpy 2.1.3, pandas 2.3.3) than under scikit-learn 1.9.1 (local); a fixed `random_state` does not make the split portable across versions.
- Evidence: Kaggle notebook versions 2 and 3 (`taeyangg4/leak-free-2026-tabpfn-stack-top-public-0-82604`): v2 stopped on the notebook's fold-parity assertion; v3 loads the frozen folds from the dataset (SHA256 of the fold vector `8b8d8882…`), prints that a rebuild would move 6,881 rows, and writes outputs hash-identical to the submitted files (submission 56897941 scored 0.82604).
- Implication: fold assignments are data, not code: ship and hash them (as `data/processed/sgkf_5_seed*.csv` and the Kaggle dataset do) whenever results must be reproduced in another environment; rebuilding folds there silently changes every OOF number.
- Reviews:

## D-A-043 — Fine-tuning-variant stacks all beat the champion on the public LB, but LB order does not follow CV order
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: eight candidates ranked by 3-seed nested OOF mean (all adding fine-tuning variants to the D-A-027 stack) all scored above `ha27_stack3` (0.82020) on the public LB: `sub_stack3_plus_a1ne2` 0.82604 (CV 0.8293), `sub_a1mix_stack` 0.82487 (0.8296), `sub_avg4` 0.82440 (0.8302), `sub_combo7` 0.82417 (0.8310), `sub_combo5` 0.82417 (0.8302), `sub_stack_a1ctx` 0.82253 (0.8290), `sub_combo7_bag3` 0.82137 (0.8310, same model as combo7 with 3-seed test-side bagging), `sub_a1mix_single` 0.82066 (0.8295).
- Evidence: Kaggle submissions list (CLI 2.2.4, 2026-10-06 08:47 UTC batch), files in `outputs/submissions/`, built by `scripts/make_stack_submission.py`; CV from `scripts/run_stack_member_screen.py` nested accuracy on 42/123/2026 (combinations chosen post hoc).
- Implication: the small fine-tuning gains transfer to the LB in direction (8/8 above the champion, +0.0005 to +0.0058), but the LB spread between near-identical candidates (combo7 vs its test-bagged twin: 0.0028) shows LB order is within noise; LB stays secondary. The rule-based champion remains `ha27_stack3` until H-A-47 confirms combo7 on 7/99; the best public-LB reading so far is 0.82604.
- Reviews:

## D-A-042 — A larger fine-tuning budget improves log loss but does not clear the stacker bar
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: that a larger fine-tuning step budget gives a member that lifts the D-A-027 stacker by the screen bar does not hold. A1 (lr 1e-5, 100 epochs, patience 20) alone 0.827102 / 0.829863 / 0.825722 vs `ha23_ft` 0.827332 / 0.827792 / 0.826412, with the best log loss seen (0.3473 / 0.3471 / 0.3511 vs 0.3503 / 0.3520 / 0.3563); A2 (lr 3e-5) 0.827217 / 0.828253 / 0.824802. Stacker vs `ha27_stack3` on 42/123/2026: A1 added +0.000268 (1/3), A1 replacing `ha23_ft` -0.000422 (1/3), A2 added +0.000345 (2/3), A2 replacing -0.000422 (1/3). The best variant, A1 with 2 inference estimators added as a fourth member, is positive on every seed but small: +0.002071 / +0.000690 / +0.000460 (mean +0.001074, 3/3); A1 with whole-fold context +0.000844 (2/3).
- Evidence: `reports/ha39_stack_summary.json`, `reports/ha39_a1*_metrics.json`, `reports/ha39_a2*_metrics.json` (`variant_accuracy` holds the ctx/ne2/ne16 single-model scores), `scripts/tabpfn_ftbudget_runner.py`; variant OOF/probability files were copied from `<name>_seed<s>_<variant>` to `<name>_<variant>_seed<s>` for the screen.
- Implication: longer fine-tuning sharpens probabilities (log loss) without changing which rows are wrong, consistent with persistent confident errors (D-A-010). The 3/3-positive +0.0011 of A1-ne2 is the strongest fine-tuning signal so far but stays under the bar; it is a submission candidate for information only, not a promotion.
- Reviews:

## D-A-041 — Public-LB references re-retrieved: our submissions and Viktor's 0.81833 (V242) are verified
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: the Kaggle API lists our submissions 56847813 `baseline_catboost.csv` 0.80780, 56855038 `hm10_innercv_logloss.csv` 0.81131, 56865289 `ha12_tabpfn.csv` 0.81786 and 56869784 `ha27_stack3.csv` 0.82020, matching D-D-028 and D-A-017/027; the public notebook `viktortaran/space-titanic` shows Best Score 0.81833 at version V242 and a current-version public score of 0.81739 (last run 2023-02-23). The other D-D-019 references (Tao, twinpilgrim, Dmitry, Misael, Samuel) were not rechecked.
- Evidence: `reports/ha46_provenance.json` (CLI 2.2.4 submissions CSV, retrieved 2026-10-06; notebook page read through a public reader; kernel ref from `kaggle kernels list`).
- Implication: 0.81833 is a verified best-version public-LB reading of a notebook whose honest OOF is about 0.805 (D-A-014), so it is a descriptive public-LB benchmark, not a clean modelling ceiling; our 0.82020 already exceeds it. Public-LB values stay secondary evidence.
- Reviews:

## D-A-040 — Repeated group-split adversarial validation confirms no train/test shift
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: train and test are not separable: group-split CatBoost adversarial AUC is 0.4865 / 0.4832 / 0.4953 on seeds 42/123/2026 (mean 0.4883), against a group-level label-permutation reference of 0.5064 / 0.5204 / 0.5107 (mean 0.5125). The predeclared rule (every AUC < 0.55 and mean within 0.02 of the permutation mean) is formally missed by 0.004, but only because the real AUC sits below the permutation reference; separation would require AUC above it.
- Evidence: `reports/ha44_adversarial_seeds.json`, `outputs/oof/ha44_adversarial_seed{42,123,2026}.csv` (row-level OOF), `scripts/audit_adversarial_seeds.py`.
- Implication: D-D-031 holds on three seeds with row-level artifacts; reweighting or shift-specific validation stays unwarranted, and the CV-to-LB gap is noise or residual optimism, not shift.
- Reviews:

## D-A-039 — Fine-tune seed bagging does not improve the stacker
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: that a logit average of two extra fine-tunes (different fine-tuning seeds, same outer folds) added as a fourth member beats `ha27_stack3` does not hold: 42 +0.000230, 123 0.000000, 2026 -0.000920 (mean -0.000230, 1/3). Post hoc diagnostic only (not predeclared): the three-run average including `ha23_ft` replacing `ha23_ft` gives 0.000000 / +0.000920 / +0.000805 (mean +0.000575, 2/3), still below the bar. The extra fine-tunes alone score 0.827677 / 0.823076 / 0.822386 (offset 1) and 0.826642 / 0.827102 / 0.828023 (offset 2); the three-run average alone 0.828598 / 0.828138 / 0.826987, about level with the stack (0.829058 / 0.828828 / 0.826872).
- Evidence: `reports/ha35_stack_summary.json`, `reports/ha35diag_stack_summary.json`, `reports/ha35_ft_o*_metrics.json`, `scripts/tabpfn_ftseed_runner.py`, `scripts/make_avg_member.py`.
- Implication: fine-tuning noise is real (single fine-tunes vary by about ±0.002 per seed) and averaging three fine-tunes matches the stack on its own, but the stacker already captures what bagging adds. Seed bagging is closed as a stack lever; the step budget and inference-context items (H-A-39..42) are the remaining fine-tuning tests.
- Reviews:

## D-A-038 — Earth/Deck G error concentration is stable, but the Logloss Earth-for-non-Earth trade is not
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: that Logloss-stopped models trade Earth accuracy for non-Earth gains (D-D-021) does not hold under honest stopping: inner-CV Logloss vs inner-CV Accuracy CatBoost changes Earth / non-Earth errors by -26/-24, -46/-18, -83/-40 on seeds 42/123/2026 (both fall; trade direction 0/3), and vs fixed300 by +10/-7, -32/-2, -40/-7, -14/-7, -15/+5 on 42/123/2026/7/99 (1/5). Only the push of Earth rows toward False repeats (Earth predicted positive rate 0.435-0.443 vs 0.447-0.462, 8/8), and 73-99% of the changed Earth rows lie within ±0.1 of 0.5. The concentration itself is stable for the champion stack on all five seeds: error rate relative to overall is Earth 1.40-1.42×, Deck G 1.60-1.67×, non-Earth 0.49-0.52×, CryoSleep with zero spend 1.01-1.04×.
- Evidence: `reports/ha43_subgroup_audit.json`, `scripts/audit_subgroup_seeds.py` (read-only over saved honest OOF: `hm10_innercv_logloss`, `hm09_inner`, `hm10_control`, nested `ha27_stack3`); champion overall error 0.170-0.174.
- Implication: D-D-021 was a seed-42 artefact of scored-fold early stopping; under honest stopping the probability-loss model is simply better on both segments, so the "keep Accuracy stopping as control" advice is obsolete (already superseded by D-D-027). Earth and Deck G remain where errors concentrate, but segment-aware stacking (D-A-035) and Earth-specific rules (D-A-008) already failed, so no segment-specific follow-up is queued.
- Reviews:

## D-A-037 — Frozen TabPFN variants add nothing to the stacker; its gain is specific to fine-tuning
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: that other frozen TabPFN-style variants already on disk improve the D-A-027 stacker does not hold: each as a fourth member vs `ha27_stack3` on 42/123/2026 gave raw-minimal features +0.000077 (2/3), v1 rules -0.000383 (0/3), v1 spend -0.000038 (1/3), 32-estimator ensemble -0.000537 (1/3), TabPFN v2.5 -0.000422 (1/3), TabICL -0.000345 (1/3); none passes.
- Evidence: `reports/ha38probe_stack_summary.json`, `scripts/run_stack_member_screen.py`; members `ha20_raw_minimal`, `ha20_v1_rules`, `ha20_v1_spend`, `ha22_ne32`, `ha18_tabpfn_v25`, `ha18b_tabicl` (saved OOF, no new training). Post hoc probe of six existing members, so a pass would have needed fresh confirmation anyway.
- Implication: feature views and other frozen in-context models are interchangeable with frozen TabPFN v3.5; the stacker's gain comes only from the fine-tuned members (D-A-027). The only remaining lever is more or better fine-tuning (H-A-35 and follow-ups), not more frozen variants.
- Reviews:

## D-A-036 — A field-aware factorization machine is less correlated with TabPFN but adds nothing to the stacker
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: that a field-aware factorization machine decorrelates enough to improve the D-A-027 stacker does not hold: FFM (k=4, fit-fold categories and 16 quantile bins, epoch chosen on an SGKF(8) slice of the fit fold) OOF 0.803520 / 0.804325 / 0.799839 on seeds 42/123/2026; its Spearman correlation with `ha12_tabpfn` is 0.926-0.930 (CatBoost 0.970-0.972), yet as a fourth stacker member vs `ha27_stack3` 42 -0.001150, 123 -0.000230, 2026 +0.000230 (mean -0.000383, 1/3).
- Evidence: `reports/ha37_ffm*_metrics.json`, `reports/ha37_stack_summary.json`, `scripts/ffm_runner.py`; the selected epoch was 2-4 in every fold (fast overfitting at 7.5k rows).
- Implication: lower correlation alone does not help when the extra disagreement sits on rows TabPFN already gets right and the persistent errors are shared (D-A-010). The S6E5 decorrelation effect (D-A-034) does not transfer to 8.7k rows. Member diversity from new model classes is now exhausted (D-A-028..033, D-A-036).
- Reviews:

## D-A-035 — Non-linear or segment-aware meta-models do not beat the linear TabPFN stacker
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: that a non-linear correction on top of the D-A-027 stacker beats `ha27_stack3` does not hold for either predeclared nested arm: CatBoost residual boosting from inner-nested LR-stack logits (depth 4, lr 0.03, 300 iterations, baseline features + member logits) 42 -0.001035, 123 -0.001841, 2026 -0.004371 (mean -0.002416, 0/3); L2 LR with member mean/std, Earth/Europa/CryoSleep/solo flags and logit × segment interactions 42 -0.000230, 123 -0.000460, 2026 -0.000575 (mean -0.000422, 0/3). Control reproduced `ha27_stack3` exactly (0.829058 / 0.828828 / 0.826872).
- Evidence: `reports/ha36_summary.json`, `scripts/run_ha36_meta.py` (saved OOF only; meta-models fitted on the other four outer folds, no early stopping or tuning on the scored fold).
- Implication: the S4E10-style residual booster overfits at 8.7k rows, and segment interactions add nothing; this matches S6E5 5th and S6E2 1st (D-A-034). The linear 3-logit stacker is the right combiner here; stack-level complexity is closed.
- Reviews:

## D-A-034 — Recent Playground Series top solutions: two untried mechanisms transfer, most do not
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: of the techniques in recent Playground top write-ups (S4E10, S5E4, S5E8, S6E2, S6E3, S6E5, S6E8), only two are untried here and plausibly transfer: (1) a non-linear correction on top of the stack, either CatBoost residual boosting from a fixed baseline of OOF logits (S4E10 1st) or a meta-learner that also sees features plus member mean/std (S5E4 1st, S5E8 6th); (2) factorization-machine members as decorrelators (S6E5 5th: a weaker Deep FFM added more to the blend than a fourth strong LightGBM; also S4E7 3rd). Rejected: original-dataset tricks (no original data exists here), nested bigram/trigram target encoding (feeds GBDT members, which are neutral, and feature sets are flat for TabPFN, D-A-020), auxiliary-target or imputation members (CryoSleep inference already flat, D-A-008), GraphSAGE over group/cabin graphs (group/location/KNN structure already failed, D-A-004/007, D-D-035), category twins of numerics (generator artefact), and the error-analysis loop (errors are persistent and confident, D-A-010).
- Evidence: research summary of primary write-ups read through a public reader (S4E10 1st https://www.kaggle.com/c/playground-series-s4e10/discussion/543725; S6E5 5th https://www.kaggle.com/competitions/playground-series-s6e5/writeups/5th-place-solution-a-99-model-logit-stack; S6E3 1st https://www.kaggle.com/c/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm; S6E2 1st https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t; NVIDIA S5E4 stacking blog), plus a secondary per-episode digest (BlamerX/Kaggle-Competition-Season-6). Counter-evidence: S6E5 5th found a residual LightGBM on stack logits gave no honest OOF gain, and S6E2 1st lists non-linear stacking, pseudo-labelling and distillation as not helping. The only accuracy-metric binary episode (S5E7) has no useful top write-up.
- Implication: the LR-on-logits stacker, model zoo, seed bagging, pseudo-labelling and threshold tuning that dominate top solutions are already covered here. Queue the two transferable mechanisms (H-A-36, H-A-37); do not re-propose the rejected ones without a new mechanism. Playground gains are about 1e-4 AUC on 100k+ synthetic rows, so expected gains here are small.
- Reviews:

## D-A-033 — TabSTAR (text-aware LoRA fine-tuning) is far below TabPFN and neutral in the stacker
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: that TabSTAR 1.1.16 (e5 text encoder with LoRA fine-tuning on verbalised baseline columns, SGKF(8) early-stopping slice of the fit fold) is a useful stacker member does not hold: alone vs `ha12_tabpfn` 42 -0.013919, 123 -0.016335, 2026 -0.017600 (mean -0.015952, 0/3; OOF 0.812263 / 0.808697 / 0.806741), below the -0.003 gate; added to the D-A-027 stacker (informational) -0.000345 / -0.000230 / -0.000690 (mean -0.000422, 0/3).
- Evidence: `reports/ha32_tabstar*_metrics.json`, `reports/ha32_stack_summary.json`, `scripts/side_runner.py` (model `tabstar`, `.venv-side` with transformers <5; seed 42 ran before per-fold checkpoint dirs were added after a Windows file-lock failure, so its fingerprint differs from 123/2026 by that path only).
- Implication: column semantics add nothing here, because the columns are short categorical codes and spends, not free text; closed. TabSTAR was the last 2024-2026 model in the list except AutoGluon (H-A-34). Every new model family has now failed as a TabPFN complement.
- Reviews:

## D-A-032 — Deep tabular nets (RealMLP, TabM, TabR) are far below TabPFN and neutral in the stacker
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: that pytabkit RealMLP-TD, TabM-D or TabR-S-D (TabArena defaults, fit-fold median imputation and categories, SGKF(8) early-stopping slice of the fit fold) are useful stacker members does not hold: alone vs `ha12_tabpfn` RealMLP 0.790866 / 0.785229 / 0.784769 (mean delta -0.038230), TabM 0.809157 / 0.805591 / 0.802140 (-0.019556), TabR 0.798459 / 0.790751 / 0.792362 (-0.031328), all 0/3 and below the -0.003 gate; each added to the D-A-027 stacker (informational) RealMLP -0.000115, TabM +0.000153, TabR -0.000383 (each 1/3).
- Evidence: `reports/ha29_summary.json` (RealMLP, TabM; the TabR arm crashed there on faiss-cpu), `reports/ha29b_tabr*_metrics.json`, `reports/ha29_stack_summary.json`, `scripts/member_runner.py`, `scripts/tabr_runner.py` (exact torch flat-L2 index in place of faiss-gpu; TabR seed 42 rerun directly after a Windows checkpoint lock), `scripts/run_stack_member_screen.py`.
- Implication: neural tabular models trained from scratch on about 7.5k rows are far weaker than tree models and TabPFN here, and their different inductive bias does not translate into stacker gain; closed. Of the 2024-2026 list, only TabSTAR (H-A-32, running) and AutoGluon (H-A-34) remain.
- Reviews:

## D-A-031 — TabDPT (real-data pretraining) is well below TabPFN v3.5 and neutral in the stacker
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: that TabDPT 1.3.1 (retrieval in-context model pretrained on real tables; fit-fold ordinal codes, default context, compile and flash attention off on Windows) is a useful stacker member does not hold: alone vs `ha12_tabpfn` 42 -0.007247, 123 -0.008628, 2026 -0.009778 (mean -0.008551, 0/3; OOF 0.818935 / 0.816404 / 0.814563; seeds 7/99 0.814909 / 0.815829), below the -0.003 gate; added to the D-A-027 stacker (informational) 42 +0.000575, 123 -0.000690, 2026 +0.000345 (mean +0.000077, 2/3), far from the +0.002 bar.
- Evidence: `reports/ha28_tabdpt*_metrics.json`, `reports/ha28_stack_summary.json`, `scripts/side_runner.py` (model `tabdpt`, run in `.venv-side`), `scripts/run_stack_member_screen.py`.
- Implication: real-data pretraining does not give TabPFN-independent signal here; together with Causilo (D-A-030) and TabICL (D-A-018), other in-context foundation models are exhausted as stack members. Remaining member diversity must come from different model classes (H-A-29 neural nets, H-A-32 text-aware TabSTAR).
- Reviews:

## D-A-030 — Causilo is below TabPFN v3.5 and adds nothing to the stacker
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: that Causilo 1.0.3 (default 8 estimators, fit-fold ordinal codes) is a useful stacker member does not hold: alone vs `ha12_tabpfn` 42 -0.003336, 123 -0.005292, 2026 -0.006902 (mean -0.005177, 0/3; OOF 0.822846 / 0.819740 / 0.817439), below the -0.003 member gate; added to the D-A-027 stacker anyway (informational) the stack moves -0.000537 (0/3) vs `ha27_stack3`.
- Evidence: `reports/ha33_summary.json`, `reports/ha33_stack_summary.json`, `reports/ha33_causilo*_metrics.json`, `scripts/member_runner.py` (model `causilo`), `configs/ha33_spec.json`.
- Implication: Causilo is the strongest non-TabPFN model tried so far (above CatBoost's about 0.815) but its errors overlap TabPFN's, so it brings no stacking gain; closed. Another in-context foundation model is unlikely to add diversity unless it is trained on different priors (TabDPT, real-data pretraining, is the remaining test of that).
- Reviews:

## D-A-029 — xRFM is far below TabPFN on this data (-0.029)
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: that xRFM (recursive feature machine trees, default `tuning_metric=logloss`, fit-fold one-hot + standardised features, SGKF(8) tuning slice) is competitive enough to be a stacker member (≥ -0.003 vs `ha12_tabpfn`) does not hold: OOF 0.795813 / 0.796503 / 0.795583 on seeds 42/123/2026 (mean delta -0.029219, 0/3), log loss about 0.47 vs 0.35.
- Evidence: `reports/ha31_summary.json`, `reports/ha31_xrfm*_metrics.json`, `scripts/member_runner.py` (model `xrfm`), `configs/ha31_spec.json`.
- Implication: kernel-based xRFM with default settings is closed; it is weaker than every tree model and TabPFN here, so it is not worth tuning or stacking (predeclared gate). The member gate keeps the stacker from absorbing weak models.
- Reviews:

## D-A-028 — Raw surname and cabin strings do not help TabPFN v3.5, alone or in the stacker
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: that TabPFN v3.5 extracts useful family/cabin identity signal from raw surname and cabin strings (fit-fold ordinal categoricals, unseen levels -1) does not hold: alone vs `ha12_tabpfn` 42 -0.000460, 123 -0.001610, 2026 -0.002301 (mean -0.001457, 0/3); added as a fourth member to the D-A-027 stacker vs `ha27_stack3` 42 -0.000230, 123 -0.000805, 2026 -0.000230 (mean -0.000422, 0/3).
- Evidence: `reports/ha30_summary.json`, `reports/ha30_stack_summary.json`, `reports/ha30_tabpfn_identity*_metrics.json`, `scripts/member_runner.py` (model `tabpfn_identity`), `scripts/run_stack_member_screen.py`; 2218 surname and 6561 cabin levels, mostly unseen across groups because SGKF keeps groups (and most families and cabins) on one side of the split.
- Implication: identity strings are closed for TabPFN as they were for CatBoost (D-A-006): under group-aware validation the identity levels rarely transfer across folds, so they act as noise. Do not queue further raw-identity variants; family/cabin signal would have to come through target-free aggregates, which were already flat (D-A-020).
- Reviews:

## D-A-027 — Nested LR stacker over three TabPFN v3.5 variants beats frozen TabPFN by about +0.004 (promoted, LB 0.82020)
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: a logistic-regression stacker over the OOF logits of frozen (`ha12_tabpfn`), fine-tuned (`ha23_ft`) and fine-tune-refit (`ha24_ftrefit`) TabPFN v3.5, fitted per outer fold on the other four folds only, beats frozen TabPFN v3.5 on every seed: screen 42 +0.002876, 123 +0.003796, 2026 +0.002531 (mean +0.003068, 3/3); confirmation 7 +0.003336, 99 +0.005292 (both positive). Adding CatBoost gave +0.002991 and ten mixed members +0.001879, so the gain comes from the TabPFN variants, not model diversity.
- Evidence: `reports/ha27_summary.json`, `reports/ha27_confirm_summary.json`, `scripts/run_ha27_stacker.py`, `scripts/run_ha27_confirm.py`; nested stacker OOF 0.829058 / 0.828828 / 0.826872 (42/123/2026) vs 0.826182 / 0.825032 / 0.824341; final stacker fitted on all seed-42 OOF rows, coefficients frozen -0.738, ft +0.459, ftrefit +1.292 (frozen acts as a contrast term); 110 of 4277 test predictions differ from `ha12_tabpfn`; Kaggle submission 56869784 public LB 0.82020 (was 0.81786).
- Implication: new current best: `ha27_stack3` (nested LR stacker over three TabPFN v3.5 variants), confirmed on 7/99, public LB 0.82020; the D-A-026 decision (adopt a single fine-tuned model) is superseded. A learned combiner extracts the fine-tuning gain that equal blends left below the bar, so new members (H-A-28..33) are now judged by whether adding them to this nested stacker clears the screen rule against `ha27_stack3`, not by their standalone score.
- Reviews:

## D-A-026 — Fine-tuned TabPFN gains are real but small: positive on all 8 seeds, about +0.0017-0.0026
- Source: A
- Host: Claude Code
- Cross-check: PENDING
- Finding: that the equal blend of frozen, fine-tuned and fine-tune-refit TabPFN v3.5 improves on frozen TabPFN by at least +0.002 does not hold on five fresh seeds: 11 +0.001610, 13 +0.000690, 17 +0.001495, 19 +0.002876, 23 +0.001610 (mean +0.001657, 5/5 positive; predeclared rule ≥ +0.002 and ≥ 4/5). Not part of the predeclared test (post hoc, descriptive only): on the same fresh seeds the single fine-tuned model (H-A-23 recipe) gains +0.001380 / +0.002301 / +0.000920 / +0.004602 / +0.001495 (mean +0.002140) and the fine-tune-refit model (H-A-24 recipe) +0.002645 / +0.000345 / +0.002186 / +0.005407 / +0.002185 (mean +0.002554). Pooled with the screen seeds, the H-A-23 recipe is positive on 8/8 seeds with a mean of about +0.0021.
- Evidence: `reports/ha26_summary.json`, `reports/ha26_manifest.json` (rule predeclared before running), `scripts/run_ha26_extended.py`, `reports/ha12_tabpfn_seed{11,13,17,19,23}_metrics.json`, `reports/ha23_ft_seed*_metrics.json`, `reports/ha24_ftrefit_seed*_metrics.json`.
- Implication: fine-tuning TabPFN gives a real, consistent gain of roughly +0.002 (8/8 seeds positive), sitting right at the practical bar. The predeclared blend test failed, and promoting a single fine-tuned model on post hoc pooled evidence would bend the rule, so the decision was handed to the user. Even if adopted, the expected public-LB effect is about +0.002 (about 0.820), far from the LB 0.83 goal.
- Reviews:

