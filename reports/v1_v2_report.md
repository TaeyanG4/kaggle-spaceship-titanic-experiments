# Spaceship Titanic - v1 / v2 experiment report

Completed on 2026-10-05 (Asia/Seoul). All runs used official local data, threshold 0.5,
and the same persisted 5-fold StratifiedGroupKFold assignments (seed 42, PassengerId group).
The primary suite ran sequentially with two CPU threads and BelowNormal Windows priority;
its summed training runtime was 584 seconds. No Kaggle submission was made.

## Outcome

- Baseline: accuracy 0.818820; public LB 0.80780 from the previous authorized submission.
- Best v1 candidate: `v1_name_fix`, accuracy 0.820200 (+0.138 percentage points).
- Best v2 candidate: `v2`, accuracy 0.818820 (no overall accuracy change).
- Selected champion: `baseline-001`. The best gain did not meet the predeclared +0.002
  screening threshold. Seeds 123/2026 were therefore not run; no stable gain is claimed.
- v1_name_fix's paired group-bootstrap 95% difference interval is [-0.002192, +0.005028].
  This includes zero and conditions on the already-fitted OOF models.

## Primary results

Delta is in percentage points relative to the baseline. These are screening/selection scores,
not independent confirmation estimates.

| Experiment | OOF accuracy | Delta, pp | Log loss | Fold std |
| --- | ---: | ---: | ---: | ---: |
| v1_name_fix | 0.820200 | +0.138 | 0.385184 | 0.008088 |
| v1_spend | 0.819395 | +0.058 | 0.382946 | 0.009115 |
| v1 | 0.816979 | -0.184 | 0.394056 | 0.013520 |
| v1_rules | 0.819165 | +0.035 | 0.386043 | 0.005726 |
| v2_spending | 0.817554 | -0.127 | 0.386126 | 0.009068 |
| v2_group | 0.816289 | -0.253 | 0.386459 | 0.007972 |
| v2 | 0.818820 | +0.000 | 0.384964 | 0.007535 |
| v2_logloss | 0.815599 | -0.322 | 0.376543 | 0.005677 |
| v2_lightgbm | 0.815829 | -0.299 | 0.377652 | 0.007186 |
| v2_xgboost | 0.817784 | -0.104 | 0.378051 | 0.008908 |
| v2_blend | 0.818590 | -0.023 | 0.374598 | 0.007926 |

## What changed and what was learned

v1 separated missing names from surname frequency, distinguished incomplete spending from
fully observed zero spending, added spending/age missingness flags, and tested zero fills
under observed CryoSleep=True or age below 13. The missing-name-only change corrected feature
semantics and produced the highest primary accuracy, but its small gain is uncertain.
Combining the corrections regressed to 0.816979: feature changes were not additive.

v2 added spending composition/category interactions, peer means excluding the passenger,
unanimous observed group-based fills, and deck-relative cabin position. All population
summaries used the official combined feature-only train/test population and never Transported.
Entire groups stayed within one validation fold, matching complete unseen test groups.
Learned category encoders for LightGBM/XGBoost fitted on training folds only.

The full feature set matched baseline accuracy. Logloss-based CatBoost stopping, LightGBM,
and XGBoost improved probability log loss but did not improve accuracy at threshold 0.5.
The predeclared equal-weight CatBoost/LightGBM/XGBoost blend reached log loss 0.374598,
but accuracy 0.818590 (two additional errors). Blend weights and thresholds were not optimized.
Earth errors were 1,149 for the best v1 candidate versus 1,152 for the baseline; the full v2
model had 1,163 Earth errors, so it did not solve the largest error segment.

## Verification and reproducibility

- 8 tests passed; Ruff passed. Tests cover missing semantics, peer exclusion, target
  independence, persisted group-fold replay, and unseen categories in both model adapters.
- All 11 primary OOF/probability/submission artifacts have correct IDs, rows, folds,
  finite probabilities, boolean predictions, and sample-submission schema.
- All 50 primary fold models were saved. Original raw data and baseline artifact SHA256
  hashes match their pre-run values. Config, data/code hashes, package versions, fold
  iterations/scores, runtime, and segment diagnostics are recorded per experiment.
- Replaying `configs/v1_name_fix.json` reused its matching completed artifacts successfully.
- The first v1_name run is retained separately. A pandas string-storage mismatch initially
  stopped the next run's fold-cache check; the corrected comparison preserves exact values
  and row order and was tested before the successful suite.

## Artifacts and next action

- `reports/v1_v2_summary.json`: selection, scores, hashes, verification, and confirmation decision.
- `reports/v1_v2_scores.csv`: complete primary score table.
- `reports/v1_v2_validation.json`: reproducible local error/marginal/joint shift audit.
- `reports/v1_v2_training.log`: fold-by-fold log.
- `outputs/submissions/v1_best.csv`: best v1 screening candidate (not promoted).
- `outputs/submissions/v2_best.csv`: best v2 screening candidate (not promoted).
- `outputs/submissions/selected_v1_v2.csv`: unchanged baseline champion predictions.
- Per-run configs, fold models, OOFs and test probabilities use the experiment ID.

Next hypothesis: a small predeclared CatBoost depth/L2 grid using the baseline and minimal
name correction, rather than adding every proposed feature. Compare on frozen folds and
confirm a material gain with matched additional seeds. Threshold/calibration work requires
nested group-aware evaluation before claiming an accuracy gain.

Existing replay command:

```powershell
uv run python scripts/train_experiment.py --config configs/v1_name_fix.json --threads 2
```
