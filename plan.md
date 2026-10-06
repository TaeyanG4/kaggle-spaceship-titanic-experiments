# Spaceship Titanic — Plan

Keep only active unfinished work here. Completed work moves to `handoff.md` and to
`discoveries.md` (always for a failed hypothesis, otherwise when reusable).

Each agent edits only its own `## Agent: <name>` section. Plan item IDs use `H-<Agent>-<NN>`, for
example `H-D-01` or `H-A-07`.

Before adding an item, confirm it is not already settled in `discoveries.md` (including negative
results), already attempted in the `handoff.md` log, or already queued under another agent (check
headings and `Hypothesis` lines only). A repeat needs a real `Improvement` and a citation of the
earlier ID.

Priority formula:

```text
Priority = 2*Impact + 2*Information + Confidence + Unblock + Diversity + (3-Cost)
```

All factors are 0-3. Cost is 0 for cheap and 3 for expensive.

Use this exact item format:

```markdown
### H-D-01 — Short title
- Sources: D-A-001, D-D-004 / none
- Hypothesis: ...
- Evidence: ...
- Improvement: ...
- Impact: 0-3
- Information: 0-3
- Confidence: 0-3
- Unblock: 0-3
- Diversity: 0-3
- Cost: 0-3
- Priority: ...
- Resource: CPU | GPU | EITHER
- Parallel: YES | NO
- Other: NONE | <external executor>
- Next test: ...
```

Example external executor: `Other: Kaggle` (only with user authorization in this project).


## Agent: D

_No active D items (slot named `Main` until 2026-10-06). All D hypotheses are completed (see the `handoff.md` log and D-D-024 to D-D-035)._

## Agent: A

### H-A-47 — Confirm the combined small-gain fine-tuning stack on 7/99
- Sources: D-A-042, D-A-039, D-A-027, D-A-037
- Hypothesis: adding A1-ne2, A1-ctx, A2 and the two-run fine-tune bag together to the D-A-027 stacker beats `ha27_stack3`, although each alone stayed under the bar.
- Evidence: at the user's suggestion (2026-10-06) to combine the small consistent gains, 11 member combinations were screened post hoc on 42/123/2026; only `ha27_stack3 + ha39_a1_ne2 + ha39_a1_ctx + ha39_a2 + ha35_ftbag` passed (+0.003797 / +0.001956 / +0.002531, mean +0.002761, 3/3); runners-up `+a1ne2+a1ctx` +0.001994 (3/3) and `frozen + ft_all_ne2 + ftrefit + a1ctx` +0.001917 (3/3). Single members were +0.0003..+0.0011 (D-A-042, D-A-039, PENDING).
- Improvement: first multi-member combination of fine-tuning variants; because it was chosen among 11 combinations the screen is optimistic, so fresh seeds 7/99 decide (both deltas positive vs `ha27_stack3`), as the rule requires.
- Impact: 2
- Information: 2
- Confidence: 1
- Unblock: 0
- Diversity: 0
- Cost: 2
- Priority: 10
- Resource: GPU
- Parallel: NO
- Other: NONE
- Next test: run `tabpfn_ftbudget_runner.py A1 7 99`, `A2 7 99`, `tabpfn_ftseed_runner.py 1 7 99` and `2 7 99`, copy variant names, `make_avg_member.py ha35_ftbag ha35_ft_o1 ha35_ft_o2 --seeds 7 99`, then `uv run python scripts/run_combo_confirm.py combo7 ha12_tabpfn ha23_ft ha24_ftrefit ha39_a1_ne2 ha39_a1_ctx ha39_a2 ha35_ftbag`; submission file `outputs/submissions/sub_combo7.csv` is already built.

### H-A-40 — Fine-tuned weights re-conditioned on the whole fit fold
- Sources: D-A-021, D-A-023, D-A-024, H-A-38
- Hypothesis: re-conditioning the `ha23`-recipe fine-tuned weights on the whole fit fold (no second fine-tune) gives the full-context gain without changing the weights, and improves the stacker.
- Evidence: verified in source that the fine-tuned model's inference context is only the rows passed to `fit()`, so `ha23_ft` never sees its 12.5% early-stopping slice; for frozen TabPFN, full context gave +0.001419 on 3/3 seeds (D-A-021, PENDING); `ha24` changed context and weights together, so the context effect alone is untested.
- Improvement: isolates the context effect; much cheaper than H-A-38's full refit. After fine-tuning, clone the fine-tuned estimator for evaluation (`clone_model_for_evaluation`, private API, pin tabpfn==9.1.0) and fit it on all of the fit fold; screen as a fourth member and as a replacement for `ha23_ft` vs `ha27_stack3` on 42/123/2026; confirm on 7/99.
- Impact: 1
- Information: 2
- Confidence: 2
- Unblock: 1
- Diversity: 0
- Cost: 1
- Priority: 11
- Resource: GPU
- Parallel: NO
- Other: NONE
- Next test: add the second inference pass to the H-A-39 runner so the baseline recipe and each arm write both outputs (about 30 min for 3 seeds on its own).

### H-A-41 — Consistent estimator counts between fine-tuning and inference
- Sources: D-A-022, D-A-023, D-A-027
- Hypothesis: predicting with 8 estimators when only 2 were fine-tuned blurs the fine-tuning gain, and matching the counts gives a sharper fine-tuned member.
- Evidence: verified in source that `FinetunedTabPFNClassifier` defaults to n_estimators 2 for fine-tuning and validation and 8 for final inference, and the base class warns when they differ; D-A-022 (frozen 8 vs 32 estimators) does not test this mismatch.
- Improvement: stage 1 diagnostic from the same fine-tuned weights: predict the held-out fold with 2, 8 and 16 estimators; stage 2 only if 2 >= 8 on 2/3 seeds: an 8/8/8 arm as a stack member vs `ha27_stack3`, normal screen, then 7/99.
- Impact: 1
- Information: 2
- Confidence: 1
- Unblock: 1
- Diversity: 1
- Cost: 1
- Priority: 11
- Resource: GPU
- Parallel: NO
- Other: NONE
- Next test: write the stage-1 predictions from the H-A-39/40 runner as extra OOF files.

### H-A-42 — Schedule-matched refit with inner-CV median epoch count
- Sources: D-A-024, D-D-026, D-D-027, D-A-027
- Hypothesis: the `ha24` refit member (stacker weight +1.29) is degraded by a schedule mismatch and an unstable epoch choice, and fixing both makes a stronger stack.
- Evidence: verified in source that the refit runs its own warmup and cosine over k epochs, a different learning-rate path from the one that reached epoch k in stage 1; the chosen epochs are bimodal (1-10 vs 27-30) on one 870-row slice, the same instability D-D-026 found and D-D-027 (VERIFIED) fixed with an inner-CV median.
- Improvement: constant learning rate in both stages (`use_lr_scheduler=False`), k = median best epoch over 4 inner SGKF slices, then refit on the whole fold; screen replacing `ha24_ftrefit` vs `ha27_stack3` on 42/123/2026; confirm on 7/99; adopt H-A-39's lr and budget if that item passes.
- Impact: 2
- Information: 2
- Confidence: 1
- Unblock: 0
- Diversity: 0
- Cost: 2
- Priority: 10
- Resource: GPU
- Parallel: NO
- Other: NONE
- Next test: `scripts/tabpfn_ftrefit2_runner.py` built from `tabpfn_ftrefit_runner.py`, reusing its epoch tracking (about 3 GPU hours for 3 seeds).

### H-A-34 — AutoGluon 1.6.3 `extreme` in a separate venv (Mitra access + reference ceiling)
- Sources: D-A-025, D-A-026, D-A-001
- Hypothesis: AutoGluon's multi-layer stack of strong members (Mitra, TabM, RealMLP, TabPFN, GBDTs), run inside our outer SGKF folds, beats the TabPFN stacker.
- Evidence: rescored 2026-10-06 after D-A-028..033: every model family AutoGluon would add (TabM, RealMLP, GBDTs, other TFMs) failed as a D-A-027 stacker member (all within ±0.0006), so confidence is now 0; learned stacking of champion-strength members is the only thing that moved the needle after TabPFN (H-A-27 screen +0.003, D-A-025/026, PENDING); Mitra (2025/2026 TFM) is reachable only through AutoGluon; the earlier gate that closed H-A-14 (no blend-diversity gain) is now met. `autogluon.tabular` 1.6.3 pins numpy<2.6, pandas<2.4, sklearn<1.10, so it must live in its own venv to keep every cached run reproducible.
- Improvement: replaces hand-built stacking with a tuned multi-layer stack and adds Mitra; outer SGKF with `groups=` so no PassengerId group crosses a fold. Gate (predeclared for every new member): single model vs `ha12_tabpfn` on seeds 42/123/2026; mean paired delta >= -0.003 makes it an eligible stacker member (re-run the H-A-27 stacker with it; normal screen/confirm/submit rule applies to the stack); >= +0.002 alone follows the normal promotion rule. Members below -0.003 are closed with a negative discovery. applied to the whole predictor as one member.
- Impact: 2
- Information: 2
- Confidence: 0
- Unblock: 0
- Diversity: 2
- Cost: 3
- Priority: 10
- Resource: GPU
- Parallel: NO
- Other: NONE
- Next test: venv and `scripts/autogluon_runner.py` are ready (groups = SGKF(8) inner fold id; a raw PassengerGroup column made AutoGluon run leave-one-group-out); smoke `extreme` with 600 s, then `.venv-autogluon/Scripts/python scripts/autogluon_runner.py ha34_ag_extreme extreme <limit> 42 123 2026` and `uv run python scripts/run_stack_member_screen.py H-A-34 ha34_ag_extreme`.

### H-A-38 — Full-train refit of the stacker members for the test predictions
- Sources: D-A-021, D-A-005, D-A-027, D-A-024
- Hypothesis: producing the `ha27_stack3` members' test predictions from one fit on all training rows (full context, and fine-tunes run on all rows) instead of the mean of five fold models on 80% beats the current submission recipe.
- Evidence: for frozen TabPFN v3.5, one full-fold-context fit beat the 5-model 80%-context average on every seed in a nested proxy (+0.001265 / +0.002301 / +0.000690, mean +0.001419, 3/3; D-A-021, PENDING), unlike CatBoost where a full refit was within noise (D-A-005, PENDING). The current champion uses fold-averaged member test predictions and a stacker fitted on all seed-42 OOF rows. Risk: the stacker's coefficients (frozen -0.74, ft +0.46, ftrefit +1.29) were learned on 80%-context OOF predictions, and sharper full-context predictions may shift the logit scale they assume.
- Improvement: extends D-A-021 from the frozen model to the whole stack, including the fine-tuned members and the stacker-calibration risk. Predeclared nested proxy: inside each outer fold, compare stack accuracy on the held-out fold when the members' held-out predictions come from (a) the mean of five inner-fold fits on 4/5 of the fit fold vs (b) one fit on the whole fit fold, with the stacker coefficients fixed from the inner OOF; screen on 42/123/2026, confirm on 7/99.
- Impact: 1
- Information: 2
- Confidence: 1
- Unblock: 0
- Diversity: 0
- Cost: 3
- Priority: 7
- Resource: GPU
- Parallel: NO
- Other: NONE
- Next test: estimate cost first (inner 5-fold fine-tunes for two members ≈ 25 fine-tunes per member per seed, roughly 4-5 GPU hours per seed); if too costly, run the proxy for the frozen member only and keep the fine-tuned members fold-averaged.

### H-A-45 — Re-review D-A-010: group-preserving learning curve of the TabPFN champion (from H-C-05)
- Sources: D-A-010, D-D-022, D-D-027, D-A-009
- Hypothesis: measuring how the champion scales with additional fit groups distinguishes a data-limited plateau from a model-limited one and tells whether the remaining fine-tuning items (H-A-39..42) can still pay off.
- Evidence: taken over from H-C-05 (user instruction 2026-10-06: Claude continues the ChatGPT queue as re-review); re-review reason — D-A-010's counts are reproducible (1,374 persistent and 712 split-specific errors), but `signal-limited` is a causal interpretation rather than an observed fact; label noise, sample-size limits, and model bias remain alternatives. D-D-022 and D-D-027 are VERIFIED. The tested tree-diversity numbers in D-A-009 were also reproduced, but its family-wide implication remains PENDING. This diagnostic can change whether H-A-12/H-A-14/H-A-15 deserve expensive compute.
- Improvement: instead of inferring the cause of persistent errors from OOF confidence, directly intervene on training-data quantity: keep the outer SGKF validation rows fixed and subsample only fit-fold PassengerId groups at predeclared fractions, preserving group boundaries and class balance while holding features, model family, metric, and honest iteration selection constant.
- Impact: 1
- Information: 3
- Confidence: 2
- Unblock: 3
- Diversity: 1
- Cost: 1
- Priority: 16
- Resource: CPU
- Parallel: YES
- Other: NONE
- Next test: (taken over 2026-10-06; the champion is now TabPFN v3.5, so frozen TabPFN context subsampling is the cheap proxy, about 1 min per fraction) on seed 42 first, train the champion recipe using 40/60/80/100% of each outer fit fold's groups with deterministic stratified group subsampling and score the unchanged outer fold. If the 80→100% slope is materially positive, repeat on seeds 123/2026 and keep H-A-12/H-A-14 high priority; if the curve saturates early, lower the priority of expensive capacity/representation experiments rather than adding more tuning.

## Agent: B

### H-B-01 — CatBoost categorical combinations and HomePlanet×Destination interaction
- Sources: D-D-034, D-A-010
- Hypothesis: higher-order categorical interactions that the current `max_ctr_complexity=1` champion cannot form, especially HomePlanet×Destination, add signal that is missing from the current baseline columns.
- Evidence: the baseline CatBoost runtime resolves `max_ctr_complexity=1`, so automatic combination CTRs are disabled; broad feature/capacity changes are already flat (D-D-034), while persistent errors indicate the remaining plateau is primarily missing signal rather than variance (D-A-010). This item was already recorded as queued in `handoff.md` but its plan block was lost.
- Improvement: test CatBoost-native combination CTRs rather than another manual broad feature bundle; include an explicit HomePlanet×Destination arm so the most interpretable interaction can be isolated from the general CTR-combination setting.
- Impact: 1
- Information: 2
- Confidence: 1
- Unblock: 0
- Diversity: 2
- Cost: 1
- Priority: 11
- Resource: CPU
- Parallel: YES
- Other: NONE
- Next test: write `configs/hb01_spec.json` with three matched arms vs `hm10_innercv_logloss`: `max_ctr_complexity=2`, `max_ctr_complexity=3`, and explicit HomePlanet×Destination; run seeds 42/123/2026 with `innercv_logloss`, then apply the normal screen/confirm rule.

## Agent: C

_No active C items (H-C-05..08 taken over by A on 2026-10-06 at the user's request)._

