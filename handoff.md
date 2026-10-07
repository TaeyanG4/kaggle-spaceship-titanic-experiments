# Spaceship Titanic — Handoff

Everything operational is recorded here. Keep unfinished agent-specific detail inside that agent's
active section; completed/reviewed work goes into the shared log.

## Shared state

- Active agent(s): A, B, C, D
- Current best: `ha27_stack3` — nested LR stacker over frozen / fine-tuned / fine-tune-refit TabPFN v3.5 (baseline features, SGKF fold-local); nested OOF 0.829058 / 0.828828 / 0.826872 (seeds 42/123/2026), confirmed on 7/99, public LB 0.82020 (D-A-027)
- Best public LB: 0.82604 — `sub_stack3_plus_a1ne2` (`ha27_stack3` + fine-tuned A1 member with 2 inference estimators), secondary evidence only; CV-promotion of fine-tuning combinations pending H-A-47 (D-A-043)
- Open consistency issues: B's active handoff section and 01:25 planning event still use wrapped/legacy field formatting; owner to fix (found by A)
- Current project state: champion `ha12_tabpfn` (TabPFN v3.5), public LB 0.81786. Goal raised by the user on 2026-10-06 to LB 0.83 if possible (previous goal 0.81833, the best clean public notebook). Only CV-promoted champions are submitted.
- Active compute: none
- Important shared artifacts/metrics:
  - Champion (since H-D-10): `hm10_innercv_logloss` — baseline config/features with inner-CV
    Logloss iteration selection. Honest OOF 0.815944 / 0.817784 / 0.818475 (seeds 42/123/2026),
    confirmation +0.002301 / +0.001495 on seeds 7/99. Submission candidate:
    `outputs/submissions/hm10_innercv_logloss.csv` (seed-42 fold models). Public LB **0.81131**
    (submission 56855038).
  - Goal (user, 2026-10-05): reach the best clean shared-notebook score, public LB 0.81833.
  - Previous champion: `baseline-001` — OOF 0.818820 (optimistic stopping), public LB 0.80780
    (submission 56847813, rank 250 / 1,575 at submission time). Still the only LB-scored model.
  - Validation: 5-fold StratifiedGroupKFold by PassengerId group, frozen folds in `data/processed/`
    (`sgkf_5_seed42.csv`, plus seed 123/2026 files from H-D-02).
  - Environment: Python 3.12 via uv (`pyproject.toml`, `uv.lock`, `.venv/`). Tests: 8 passed. Ruff: passed.
  - Baseline OOF / submission: `outputs/oof/baseline_catboost_oof.csv`,
    `outputs/submissions/baseline_catboost.csv` (dry-run SHA256 prefix 2b8451c2e7d30577)
  - v1/v2 report: `reports/v1_v2_report.md`; summary `reports/v1_v2_summary.json`
  - Screening aliases: `outputs/submissions/v1_best.csv`, `outputs/submissions/v2_best.csv`;
    selected fallback `outputs/submissions/selected_v1_v2.csv` (same SHA256 as baseline)
  - H-D-01 Earth diagnosis: `reports/earth_error_diagnosis.md` / `.json` (`scripts/audit_earth_errors.py`)
  - H-D-08 multi-seed screen: `reports/hm08_summary.json`, `reports/hm08_manifest.json`
    (`scripts/run_hm08_screen.py`, `src/spaceship_titanic/screening.py`)
  - H-D-09 stopping screen: `reports/hm09_summary.json` (`scripts/run_hm09_stopping.py`)
  - General runner + screen driver: `scripts/catboost_runner.py`, `scripts/run_feature_screen.py`
    (spec files `configs/<prefix>_spec.json`)
  - H-D-10 honest stopping screen: `reports/hm10_summary.json`, `configs/hm10_spec.json`
  - Improvement-only alert: `scripts/alert.py` (sound only on promotion, per user request)
  - H-D-02 CatBoost grid: `reports/hm02_scores.csv`, `reports/hm02_summary.json`,
    `reports/hm02_manifest.json` (`scripts/run_hm02_grid.py`)
  - Notebook audit: `reports/clean_notebook_catalog.md` / `.json`; source cache
    `.kaggle-research/spaceship-titanic/kernel-archives/` (Git-ignored, untrusted text)

Agent names are work slots (`A`, `B`, `C`, ...), never host or model names. A new concurrent
session takes the next unused name and adds it to `Active agent(s)`.

### Score ledger

| ID | Date | Model / change | CV accuracy | LB accuracy | Status | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| setup-001 | 2026-10-05 | Workspace bootstrap | - | - | complete | No model trained |
| baseline-001 | 2026-10-05 | CatBoost + basic target-free features + 5-fold SGKF | 0.818820 | 0.80780 | champion / submitted | rank 250/1,575; submission 56847813 |
| v1_name | 2026-10-05 | Missing-name count correction only | 0.820200 | - | historical run | first suite stopped at next fold-cache check |
| v1_name_fix | 2026-10-05 | Same name ablation, corrected fold-cache verification | 0.820200 | - | complete | +0.001380; below +0.002 screening |
| v1_spend | 2026-10-05 | Spending semantics + age/spend missing indicators | 0.819395 | - | complete | +0.000575; not promoted |
| v1 | 2026-10-05 | Combined name/spending semantics | 0.816979 | - | complete | changes not additive |
| v1_rules | 2026-10-05 | v1 + observed CryoSleep/child zero fill | 0.819165 | - | complete | +0.000345; not promoted |
| v2_spending | 2026-10-05 | v1 + composition and category interactions | 0.817554 | - | complete | no overall gain |
| v2_group | 2026-10-05 | v1 + peer summaries/fills/cabin position | 0.816289 | - | complete | no overall gain |
| v2 | 2026-10-05 | v1 + all v2 feature families, CatBoost | 0.818820 | - | complete | ties baseline |
| v2_logloss | 2026-10-05 | Full v2, CatBoost Logloss stopping | 0.815599 | - | complete | log loss 0.376543 |
| v2_lightgbm | 2026-10-05 | Full v2, fold-fitted categorical encoder, LightGBM | 0.815829 | - | complete | log loss 0.377652 |
| v2_xgboost | 2026-10-05 | Full v2, fold-fitted categories, XGBoost | 0.817784 | - | complete | log loss 0.378051 |
| v2_blend | 2026-10-05 | Equal-weight v2 Logloss CatBoost/LightGBM/XGBoost | 0.818590 | - | complete | log loss 0.374598; two more errors than baseline |
| hm02_base_control | 2026-10-05 | Runner replay of baseline config (d6, L2 5) | 0.818820 | - | complete | exactly reproduces baseline-001 OOF/log loss |
| hm02_name_d8_l5 | 2026-10-05 | v1_name features, CatBoost depth 8 / L2 5 | 0.824111 | - | rejected | best of 12-run grid; seed 123 −0.003911, seed 2026 +0.003911 vs matched control |
| hm02_name_d6_l1 | 2026-10-05 | v1_name features, depth 6 / L2 1 | 0.821926 | - | complete | not confirmed |
| hm02_base_d8_l5 | 2026-10-05 | baseline features, depth 8 / L2 5 | 0.821235 | - | complete | not confirmed |
| hm02 (other 9) | 2026-10-05 | depth 4, L2 10, lr 0.03 variants | 0.810997-0.819970 | - | complete | see `reports/hm02_scores.csv` |
| hm08 screen | 2026-10-05 | 5 candidates × seeds 42/123/2026 vs matched control | mean Δ +0.000077 to +0.001764 | - | rejected | no candidate passed; all lose on seed 123 |
| hm09 stopping | 2026-10-05 | baseline config, honest stopping (fixed300 / inner / inner_refit), 3 seeds | 0.813183-0.815944 (fixed300) | - | complete | fixed300 mean −0.0044 vs optimistic control: measures optimism |
| hm10_innercv_logloss | 2026-10-05 | baseline config, inner 4-fold Logloss iteration selection (honest) | 0.815944 (s42); mean 0.817401 | 0.81131 | **champion / submitted** | screen +0.003183 vs fixed300; confirm +0.002301 / +0.001495 (seeds 7/99) |
| hm10 others | 2026-10-05 | fixed200 / fixed500 / inner_logloss_refit vs fixed300 | mean Δ −0.001802 / +0.003144 / +0.002607 | - | complete | fixed500 and inner_logloss_refit also passed the screen |
| hm11 bag | 2026-10-05 | 3-seed / 5-seed OOF average of hm10_innercv_logloss | bag3 0.818014 | - | rejected | screen mean +0.000614 vs singles |
| hm03 removal | 2026-10-06 | drop 5 predeclared baseline feature groups (innercv_logloss) | mean Δ −0.007017 to +0.000230 | - | rejected | positional ids most valuable |
| hm05 xgboost | 2026-10-06 | XGBoost baseline / v1_name, innercv logloss | mean Δ −0.005752 / −0.005598 | - | rejected | worse than CatBoost |
| hm06 blend | 2026-10-06 | equal blend / nested weight / nested threshold on OOF | mean Δ −0.002607 / −0.000077 / +0.000882 | - | rejected | threshold 0.5 kept |
| hm12 rescreen | 2026-10-06 | v1/v2 variants + d8_l5 / d6_l1 with innercv logloss | mean Δ −0.003413 to +0.000690 | - | rejected | plateau |
| hm07 knn | 2026-10-06 | champion + in-fold supervised k=1 KNN distances | mean Δ −0.008628 | - | rejected | hurts, esp. Earth |
| ha01 group snap | 2026-10-06 | champion OOF + group-consistency post-processing (B=0.10, C=0.20) | mean Δ −0.002032 | - | rejected | no training |
| ha07 refit proxy | 2026-10-06 | single full-fit-fold refit (B1 / B2 ×1.25 iter) vs 5-model 80% average | mean Δ +0.001035 / +0.000882 | - | rejected | keep fold-average recipe |
| ha02 surname | 2026-10-06 | + Surname categorical (± repeat count / SurnameGroupSize) | mean Δ −0.001572 to −0.000422 | - | rejected | family identity weak |
| ha03 location | 2026-10-06 | + GroupId / CabinNum fill / CabinSize / parity | mean Δ −0.001841 to −0.000767 | - | rejected | location already in cabin fields |
| ha04 cryo/fills | 2026-10-06 | CryoSleep rules (± Cabin/VIP group fill, deck→HomePlanet) | mean Δ −0.001764 / +0.000460 | - | rejected | imputation removes missingness signal |

## Active handoff — Agent sections

Each research agent reads and edits only its own active handoff section. Do not copy another
agent's unfinished plan here.

`Current host` records which host (Claude Code, Codex, Antigravity, ...) currently owns the slot. A
slot is free when it reads `unassigned` or `released`. Set it to your host when you take or resume
the slot, and to `released` when you close the session. `Last active` is updated with every log
event. Host means the platform (Claude Code, Codex, ...), never the model (Fable/Opus/Sonnet). A new
session of the same platform may continue a slot its platform holds (see `agents.md`). A new session
may read only these two lines of other agents' sections, to choose a slot.

### Agent: D
- Current host: released
- Last active: 2026-10-06 01:10
- Current thread: none — queue exhausted; the user moved this Claude Code session to slot A (2026-10-06)
- Resumable state: none
- Blocker: none
- Next action: none — no D items remain (slot renamed from Main on 2026-10-06 at the user's request)


### Agent: A
- Current host: released
- Last active: 2026-10-06 18:03
- Current thread: none — research stopped at the user's request (2026-10-06) to publish the write-up
- Resumable state: champion `ha27_stack3` (LB 0.82020, D-A-027); best public LB 0.82604 (`sub_stack3_plus_a1ne2`, D-A-043). Stopped mid-run: `tabpfn_ftbudget_runner.py base 42 123 2026` (H-A-40/41 inputs, no output written) and H-A-45 (not started). H-A-47 needs `tabpfn_ftbudget_runner.py A1 7 99`, `A2 7 99`, `tabpfn_ftseed_runner.py 1 7 99`, `2 7 99`, `make_avg_member.py ha35_ftbag ha35_ft_o1 ha35_ft_o2 --seeds 7 99`, then `run_combo_confirm.py combo7 ...`. Variant files from the budget runner must be copied to `<name>_<variant>_seed<s>` before screening. `TABPFN_TOKEN` is a user-scope env var (load per process, never print).
- Blocker: none
- Next action: if research resumes, H-A-47 confirmation first, then H-A-40/41 (base arm), H-A-45, H-A-42

### Agent: B
- Current host: released
- Last active: 2026-10-06 01:25
- Current thread: none (H-B-01 queued, not started)
- Resumable state: read `## Agent: B` in `plan.md`. Use `scripts/run_feature_screen.py` with
  `control_id: hm10_innercv_logloss` and `innercv_logloss` stopping. The `max_ctr_complexity` arms
  need only a spec file (candidate `params` override).
- Blocker: none. The HomeDest arm needs a B-owned post-feature hook; agent A owns the hooks in
  `scripts/extra_features.py`.
- Next action: write `configs/hb01_spec.json` (arms max_ctr_complexity 2 and 3, plus HomeDest) and run
  seeds 42/123/2026. Expect each run to be slower than the champion, because combination CTRs add
  features.

### Agent: C
- Current host: released
- Last active: 2026-10-06 03:13
- Current thread: none — queue taken over by A (user instruction 2026-10-06)
- Resumable state: none — the four re-review items now live in A's plan section
- Blocker: none
- Next action: none — queue taken over by A (user instruction 2026-10-06)

## Completed / review log

All agents may read this shared log. Append completed work, meaningful resource-routing changes,
and discovery-review events here. Entries are append-only.

Entries before 2026-10-05 21:08 were written on Codex before agent names existed and are
attributed to slot `D` (named `Main` until 2026-10-06). Their times come from artifact modification times (approximate); IDs and
fields were converted to the current format on 2026-10-05 23:15.

### 2026-10-05 17:25 — D — none
- Host: Codex
- Action: [Bootstrap (pre-orchestrator)] confirmed the empty workspace and local tools; initialized Git; created continuity docs, directory structure, `pyproject.toml`, package skeleton, environment check, tests, baseline config, README, gitignore; created `.venv` via uv; generated `uv.lock`.
- Result: `check_env.py`, pytest, and Ruff passed. The first Kaggle download returned HTTP 403; after the user joined/accepted the rules, the official download succeeded.; see D-D-001, D-D-002, D-D-003 Evidence: train (8693, 14), test (4277, 13), sample (4277, 2); target 4378/4315.
- Artifacts: `pyproject.toml`, `uv.lock`, `data/raw/`
- Discovery updates: D-D-001, D-D-002, D-D-003
- Review verdict: none
- Resource: CPU
- Other executor: Kaggle (official data download)
- New plan items: none — no follow-up queued in this step

### 2026-10-05 17:37 — D — none
- Host: Codex
- Action: [Baseline and first submission (pre-orchestrator)] added `src/spaceship_titanic/features.py`, `scripts/audit_data.py`, `scripts/train_baseline.py`; confirmed zero train/test group overlap; trained 5-fold SGKF CatBoost; saved OOF and submission; dry-run passed 4 structural checks; submitted on user request.
- Result: OOF 0.818820; public LB 0.80780; rank 250/1,575.; see D-D-004, D-D-005, D-D-006, D-D-020 Evidence: pytest 3 passed; Ruff passed.
- Artifacts: `outputs/oof/baseline_catboost_oof.csv`, `outputs/submissions/baseline_catboost.csv`
- Discovery updates: D-D-004, D-D-005, D-D-006, D-D-020
- Review verdict: none
- Resource: CPU
- Other executor: Kaggle (submission 56847813, user-authorized)
- New plan items: none — shift audit, feature-semantics review (pre-orchestrator, not ID'd

### 2026-10-05 18:27 — D — none
- Host: Codex
- Action: [OOF review and improvement planning (pre-orchestrator)] recomputed baseline OOF; reviewed features, training, config, and audit code using official local data only. No training, raw-data edits, predictions, or remote actions.
- Result: found zero-filled spending ambiguity, the 294 missing-name surname count, Earth/solo error concentration, and only small marginal shift.; see D-D-007, D-D-008, D-D-009, D-D-010, D-D-011 Evidence: OOF 0.818820, log loss 0.384949, 1,575 errors reproduced.
- Artifacts: `plan.md`, `discoveries.md`, `handoff.md`
- Discovery updates: D-D-007, D-D-008, D-D-009, D-D-010, D-D-011
- Review verdict: none
- Resource: CPU
- Other executor: none
- New plan items: none — P0-P4 proposals (pre-orchestrator, not ID'd

### 2026-10-05 19:05 — D — none
- Host: Codex
- Action: [P0 infrastructure and v1/v2 screening suite (pre-orchestrator)] implemented the config-driven runner, frozen fold storage, fold models, test probabilities, paired group-bootstrap diagnostics, and versioned outputs. Ran `scripts/run_versions.py --threads 2` at BelowNormal priority. The first attempt stopped after v1_name on a pandas string-dtype fold-cache mismatch; it was fixed and repeated as v1_name_fix. pytest temp moved to `tmp_pytest/`. Completion heartbeat `spaceship-titanic-v1-v2` paused after completion.
- Result: 10 trained configs + 1 blend; best v1 0.820200, best v2 0.818820. No candidate met +0.002, seed confirmation was skipped, and the baseline was retained.; see D-D-012, D-D-013, D-D-014, D-D-015, D-D-016 Evidence: all artifact checks passed; raw/baseline hashes unchanged; pytest 8 passed; Ruff passed.
- Artifacts: `reports/v1_v2_*`, `reports/*_metrics.json`, `configs/v1*.json`, `configs/v2*.json`
- Discovery updates: D-D-012, D-D-013, D-D-014, D-D-015, D-D-016
- Review verdict: none
- Resource: CPU (2 threads)
- Other executor: none
- New plan items: none — CatBoost depth/L2 grid; Earth regression diagnosis; optional adversarial validation

### 2026-10-05 19:39 — D — none
- Host: Codex
- Action: [Shared notebook provenance and validation audit (pre-orchestrator)] searched 100 score-sorted and 30 vote-sorted notebooks via the Kaggle skill's read-only CLI pulls; inspected 18 source/metadata snapshots without executing them.
- Result: six direct-training method references cataloged; prediction-override and submission-CSV-reuse notebooks excluded; supervised group/KNN contamination and unverifiable remote features identified. No training, submission, or scheduling.; see D-D-017, D-D-018, D-D-019 Evidence: source hashes and code locations in the catalog.
- Artifacts: `reports/clean_notebook_catalog.md`, `reports/clean_notebook_catalog.json`
- Discovery updates: D-D-017, D-D-018, D-D-019
- Review verdict: none
- Resource: none
- Other executor: Kaggle (read-only source pulls)
- New plan items: none — compact XGBoost; feature-removal ablations; nested threshold/blend; cv13j0 KNN nested reconstruction

### 2026-10-05 21:08 — D — none
- Host: Claude Code
- Action: [Migration to Research Orchestrator v7.1.0] installed the skill to `~/.claude/skills/` and `~/.agents/skills/`; rewrote `agents.md`, converted discoveries to numbered entries, rebuilt `plan.md` as a scored active queue, and restructured this file. No experiments, data, or code changed.
- Result: active queue of seven items (now H-D-01 to H-D-07). The Earth diagnosis (15) was scored ahead of the CatBoost grid (14).
- Artifacts: `agents.md`, `plan.md`, `discoveries.md`, `handoff.md`, `README.md`
- Discovery updates: none — no experiment in this step
- Review verdict: none
- Resource: none
- Other executor: none
- New plan items: H-D-01 to H-D-07

### 2026-10-05 22:15 — D — H-D-01
- Host: Claude Code
- Action: added `scripts/audit_earth_errors.py`; compared the saved OOF of baseline, v1_name_fix, v2, v2_logloss, v2_lightgbm, v2_xgboost, and v2_blend on Earth rows by target-free segments, flips vs baseline, confidence bands, and Earth-only paired group bootstrap. No training.
- Result: probability-loss models add +23 to +42 Earth errors while removing 13-21 non-Earth errors. Logloss stopping alone costs about 30 Earth errors; most lost rows are borderline and truly Transported.; see D-D-021 Evidence: `reports/earth_error_diagnosis.md`; Earth CI excludes zero for Logloss CatBoost and LightGBM only.
- Artifacts: `scripts/audit_earth_errors.py`, `reports/earth_error_diagnosis.md` / `.json`; Ruff passed; pytest 8 passed
- Discovery updates: D-D-021
- Review verdict: none
- Resource: CPU
- Other executor: none
- New plan items: none (rescored H-D-02, H-D-05 Confidence 1 -> 0, H-D-06)

### 2026-10-05 22:26 — D — H-D-02
- Host: Claude Code
- Action: added `scripts/run_hm02_grid.py`; predeclared manifest (6 settings × {baseline, v1_name} plus a baseline-config control; Accuracy stopping; +0.002 screen; seeds 123/2026 confirmation). Ran 13 primary and 4 confirmation runs with 2 workers × 2 threads at BelowNormal priority. No Kaggle actions.
- Result: the control reproduced baseline-001 exactly. Best grid run `hm02_name_d8_l5` 0.824111 (+0.005292) passed screening but failed confirmation (−0.003911 seed 123, +0.003911 seed 2026). Not promoted; `baseline-001` remains champion.; see D-D-022, D-D-023, D-D-024 Evidence: `reports/hm02_summary.json` (`promoted: false`); protected raw/champion hashes unchanged; pytest 8 passed; Ruff passed.
- Artifacts: `reports/hm02_*`, `outputs/oof/hm02_*.csv`, `configs/hm02_*.json`, `models/hm02_*`
- Discovery updates: D-D-022, D-D-023, D-D-024 (negative result)
- Review verdict: none
- Resource: CPU (2 workers × 2 threads, about 7.5 minutes wall time)
- Other executor: none
- New plan items: H-D-08 (18), H-D-09 (16); H-D-03 now waits for H-D-08

### 2026-10-05 23:15 — D — none
- Host: Claude Code
- Action: [Migration to the host-aware orchestrator format + cross-checks] converted all four files to the updated skill: `H-D-NN` / `D-D-NNN` IDs, `Host` and `Cross-check` fields, a `Current host` slot field, plan items with Hypothesis / Evidence / Improvement, and this event format. Removed self-review verdicts. Recorded H-D-02's failure as the negative-result discovery D-D-024. Cross-checked the Codex discoveries that can be recomputed locally (read-only scratch script over `data/raw/` and saved OOF/metrics; no training). Renamed plan-item labels in `scripts/audit_earth_errors.py`, `scripts/run_hm02_grid.py`, and their report files.
- Result: 14 Codex discoveries VERIFIED. 6 Codex discoveries (Kaggle-account or notebook-source claims) and 4 Claude Code discoveries remain PENDING. Fixed a migration transcription error in D-D-016 (10 trained configs + 1 blend, not 11 configs + blend).; see D-D-024, D-D-016 Evidence: every recomputed shape, count, accuracy, log loss, and fold score matched the Codex claims; the uv venv runs Python 3.12.13 while system Python is 3.12.6, consistent with D-D-001.
- Artifacts: `agents.md`, `plan.md`, `discoveries.md`, `handoff.md`
- Discovery updates: D-D-024 (new); D-D-016 (text fix)
- Review verdict: D-D-001, 003, 004, 005, 007, 008, 009, 010, 011, 012, 013, 014, 015, 016 CLOSED
- Resource: CPU (read-only checks)
- Other executor: none
- New plan items: none — all items re-ID'd and given Hypothesis / Evidence / Improvement; scores unchanged

### 2026-10-05 23:23 — D — H-D-08
- Host: Claude Code
- Action: user approved changing the screening rule; `agents.md` now uses a multi-seed screen (mean paired delta ≥ +0.002 over seeds 42/123/2026 with ≥2 seeds positive; confirmation on fresh seeds 7/99). Added `src/spaceship_titanic/screening.py` + `tests/test_screening.py` and `scripts/run_hm08_screen.py`; re-screened the four best H-D-02 configs plus `v1_name_fix` (8 new runs, 10 reused) with 4 workers × 2 threads. Synced the updated skill to `~/.agents/skills/` (old copy backed up in the scratchpad). Added `scripts/alert.py` so a sound plays only on promotion; the always-on Stop/Notification sound hooks were added then removed at the user's request (`~/.claude/settings.json` restored to its original content).
- Result: no candidate passed (best mean +0.001764); no confirmation run; not promoted.; see D-D-025 Evidence: `reports/hm08_summary.json`; pytest 11 passed; Ruff passed; protected hashes unchanged.
- Artifacts: `agents.md`, `src/spaceship_titanic/screening.py`, `tests/test_screening.py`, `scripts/run_hm08_screen.py`, `scripts/alert.py`, `reports/hm08_*`, `outputs/oof/*_seed{123,2026}.csv`
- Discovery updates: D-D-025 (negative result)
- Review verdict: none
- Resource: CPU (4 workers × 2 threads)
- Other executor: none
- New plan items: none (H-D-09 and H-D-03 updated to use the harness)

### 2026-10-05 23:28 — D — H-D-09
- Host: Claude Code
- Action: added `scripts/run_hm09_stopping.py`; screened three honest stopping modes (inner holdout, inner holdout + refit, fixed300) on the baseline config/features over seeds 42/123/2026 against the existing optimistic control. Added `scripts/catboost_runner.py` and `scripts/run_feature_screen.py` for later items; `agents.md` now requires honest stopping for new screens.
- Result: fixed300 mean −0.004410, inner −0.010545, inner_refit −0.010583 vs control; none passed (expected, because the control is optimistic). Accuracy-metric inner-holdout stopping is unstable.; see D-D-026 Evidence: `reports/hm09_summary.json`; protected hashes unchanged; pytest 11 passed; Ruff passed.
- Artifacts: `scripts/run_hm09_stopping.py`, `scripts/catboost_runner.py`, `scripts/run_feature_screen.py`, `configs/hm10_spec.json`, `reports/hm09_*`, `agents.md`
- Discovery updates: D-D-026
- Review verdict: none
- Resource: CPU (4 workers × 2 threads)
- Other executor: none
- New plan items: H-D-10 (15); H-D-04 rescored 12 -> 10 (part of the gap explained)

### 2026-10-05 23:45 — A — none
- Host: Claude Code
- Action: [Notebook idea audit (score-sorted top 100, about 0.81-0.83)] user asked to find unexecuted ideas in the 0.81-0.83 shared notebooks and add them to the plan. Took slot A because Main is held by a concurrent Claude Code session running H-D-10. Pulled 65 uncached score-sorted sources read-only (`kaggle kernels pull --metadata`, never executed) into the Git-ignored archive; merged clones; five parallel static reads plus spot checks. Added the target-free script `scripts/audit_notebook_ideas.py`. Cross-checked the Codex notebook discoveries D-D-017/018 against the cached sources. No training, submission, or account change.
- Result: about 61 distinct notebooks reviewed. Many are excluded (submission-CSV reuse, hard-coded overrides with auto-submit, external services, pre-CV target statistics, SMOTE before CV). Five untested rule-compatible directions were queued. Transforms and imputers were reviewed and left out with reasons.; see D-A-001, D-A-002 Evidence: `reports/notebook_idea_audit_top100.md`, `reports/notebook_idea_coverage.txt`
- Artifacts: `plan.md` (`## Agent: A`), `discoveries.md`, `scripts/audit_notebook_ideas.py` (Ruff passed), `reports/notebook_idea_*`
- Discovery updates: D-A-001, D-A-002 (new)
- Review verdict: D-D-017 CLOSED, D-D-018 CLOSED
- Resource: CPU (read-only)
- Other executor: Kaggle (read-only source pulls)
- New plan items: H-A-01 (15), H-A-02 (14), H-A-03 (11), H-A-04 (9), H-A-05 (8)

### 2026-10-05 23:48 — D — H-D-10
- Host: Claude Code
- Action: screened honest iteration-selection modes (fixed200, fixed500, inner-holdout Logloss refit, inner 4-fold CV Logloss median) against fixed300 on seeds 42/123/2026 with `scripts/run_feature_screen.py` (`configs/hm10_spec.json`, predeclared manifest), then confirmed the best passing mode on fresh seeds 7/99. 19 runs, 4 workers × 2 threads.
- Result: `innercv_logloss` passed (mean +0.003183) and confirmed (+0.002301 / +0.001495), so it was PROMOTED. The improvement alert sounded, and work paused per the user's instruction.; see D-D-027 Evidence: `reports/hm10_summary.json` (`promoted: true`); protected hashes unchanged.
- Artifacts: `reports/hm10_*`, `outputs/oof/hm10_*.csv`, `outputs/submissions/hm10_innercv_logloss.csv`, `agents.md` (stopping rule), `configs/hm03_spec.json` (stopping switched)
- Discovery updates: D-D-027
- Review verdict: none
- Resource: CPU (4 workers × 2 threads)
- Other executor: none
- New plan items: none (H-D-10 removed; H-D-03 updated)

### 2026-10-05 23:52 — D — none
- Host: Claude Code
- Action: [Submission of hm10_innercv_logloss] at the user's request, submitted `outputs/submissions/hm10_innercv_logloss.csv` (SHA256 prefix 750f5c45bd6f46ae) to Kaggle. The user also approved the honest-stopping rule change and set the goal LB 0.81833.
- Result: public LB 0.81131 (submission 56855038), +0.00351 over baseline-001.; see D-D-028 Evidence: `kaggle competitions submissions -c spaceship-titanic`
- Artifacts: none changed
- Discovery updates: D-D-028
- Review verdict: none
- Resource: Other
- Other executor: Kaggle (user-authorized submission; 8 submissions left today at submit time)
- New plan items: H-D-11, H-D-12, H-D-13

### 2026-10-05 23:54 — D — H-D-11
- Host: Claude Code
- Action: averaged the champion's OOF/test probabilities over SGKF seeds with no training (`scripts/run_hm11_bagging.py`; predeclared rule in the docstring). Found agent A's section and items in `plan.md` (A released). Moved H-D-12/13, which an append had placed under A's section, back into Main's section. A's items were not edited.
- Result: screen failed (mean +0.000614); not promoted, no submission.; see D-D-029 Evidence: `reports/hm11_summary.json`
- Artifacts: `scripts/run_hm11_bagging.py`, `reports/hm11_summary.json`
- Discovery updates: D-D-029 (negative result)
- Review verdict: none
- Resource: CPU
- Other executor: none
- New plan items: none (H-D-11 removed)

### 2026-10-06 00:40 — A — none
- Host: Claude Code
- Action: [Junaid512 leakage audit and idea coverage] user asked whether junaid512 (second in the score sort, after the excluded bsthere) leaks or cheats, and whether all its ideas are in the plan. Re-pulled the source read-only (hash unchanged from the cache), read all 34 cells without executing them, and added the label-aware diagnostic `scripts/audit_notebook_rules.py` (fixed rules, no model selection). No training, submission, or account change.
- Result: no test-label use or external predictions. CV is leaked (train-wide groupmate LOO feature, row-wise folds, in-sample stacker/threshold), and pseudo-labels are used. Group-label propagation reaches 0 test rows. The R1 hard lock lowers the champion OOF on all 5 seeds (−0.0017 to −0.0038). Only 43.6% of multi-member groups are label-unanimous.; see D-A-003 Evidence: `reports/notebook_rule_checks.txt`
- Artifacts: `plan.md` (A section: H-A-01 rescored 15 -> 12; H-A-02/03/04/05 extended; H-A-06 new), `discoveries.md`, `scripts/audit_notebook_rules.py` (Ruff passed)
- Discovery updates: D-A-003 (new)
- Review verdict: none
- Resource: CPU (read-only)
- Other executor: Kaggle (read-only source pull)
- New plan items: H-A-06 (8)

### 2026-10-06 00:34 — D — H-D-03
- Host: Claude Code
- Action: screened five predeclared feature-group removals against the champion runs (`configs/hm03_spec.json`, `scripts/run_feature_screen.py`; control reused via `control_id`). 15 new runs, 4 workers × 2 threads, alongside H-D-12/13.
- Result: none passed (best +0.000230); not promoted.; see D-D-030 Evidence: `reports/hm03_summary.json`; protected hashes unchanged.
- Artifacts: `reports/hm03_*`, `outputs/oof/hm03_*.csv`, `scripts/run_feature_screen.py` (control reuse)
- Discovery updates: D-D-030 (negative result)
- Review verdict: none
- Resource: CPU
- Other executor: none
- New plan items: none (H-D-03 removed)

### 2026-10-06 00:36 — D — H-D-04
- Host: Claude Code
- Action: group-split adversarial validation on baseline features (`scripts/audit_adversarial.py`), diagnostic only, 4 threads, alongside H-D-12/13.
- Result: AUC 0.486, so there is no exploitable shift.; see D-D-031 Evidence: `reports/adversarial_validation.json`
- Artifacts: `scripts/audit_adversarial.py`, `reports/adversarial_validation.json`
- Discovery updates: D-D-031 (negative result)
- Review verdict: none
- Resource: CPU
- Other executor: none
- New plan items: none (H-D-04 removed)

### 2026-10-06 00:40 — D — H-D-05
- Host: Claude Code
- Action: removed H-D-06 — completed in this same event. added `scripts/xgb_runner.py` (honest inner-CV Logloss XGBoost) and XGBoost dispatch in `scripts/run_feature_screen.py`. Screened XGBoost on baseline/v1_name (`configs/hm05_spec.json`). Then added `scripts/run_blend_screen.py` and screened an equal blend, nested blend weight, and nested threshold over saved OOF (`configs/hm06_spec.json`).
- Result: XGBoost −0.0058 / −0.0056; blends/threshold best +0.000882; none passed; not promoted.; see D-D-032, D-D-033 Evidence: `reports/hm05_summary.json`, `reports/hm06_summary.json`
- Artifacts: the scripts and reports above, `outputs/oof/hm05_*.csv`
- Discovery updates: D-D-032, D-D-033 (negative results)
- Review verdict: none
- Resource: CPU
- Other executor: none
- New plan items: none (H-D-05, H-D-06 removed)

### 2026-10-06 00:55 — D — H-D-12
- Host: Claude Code
- Action: removed H-D-13 — completed in this same event. re-screened six v1/v2 feature variants and two capacity settings with honest inner-CV Logloss iteration selection (`configs/hm12_spec.json`; champion runs as control). 24 new runs; slowed by other projects' processes (H1N1 AutoGluon, rsna) sharing the CPU, which were not touched.
- Result: none passed (best +0.000690); not promoted.; see D-D-034 Evidence: `reports/hm12_summary.json`
- Artifacts: `reports/hm12_*`, `outputs/oof/hm12_*.csv`
- Discovery updates: D-D-034 (negative result)
- Review verdict: none
- Resource: CPU (4 workers × 2 threads)
- Other executor: none
- New plan items: none (H-D-12, H-D-13 removed)

### 2026-10-06 01:05 — A — H-A-06
- Host: Claude Code
- Action: [Plan update: full-train refit item; dropped] at the user's request, added a full-train refit item for the most promising model. That is still the champion `hm10_innercv_logloss`: every later Main screen failed (D-D-030 to 034). Confirmed from `scripts/catboost_runner.py` that the submission averages five 80%-train fold models (:120). The champion's OOF already comes from a single refit on each whole fit fold (:115-117), so the refit arm can reuse it. Removed H-A-06 because its stated gate failed: v2_spending's final mean was +0.000038 (D-D-034). It was never run, so there is no new discovery. No training.
- Result: plan queue for A is H-A-07 (14), H-A-02 (14), H-A-01 (12), H-A-03 (11), H-A-04 (9), H-A-05 (8). Evidence: `scripts/catboost_runner.py:84-124`, `reports/hm12_summary.json`
- Artifacts: `plan.md` (A section)
- Discovery updates: none — no experiment in this step
- Review verdict: none
- Resource: none
- Other executor: none
- New plan items: H-A-07 (14)

### 2026-10-06 01:10 — D — H-D-07
- Host: Claude Code
- Action: rebuilt cv13j0's supervised KNN features strictly in-fold (`scripts/knn_runner.py`; `catboost_knn` model in `scripts/run_feature_screen.py`; `configs/hm07_spec.json`) on the champion config, then screened them against the champion runs.
- Result: mean −0.008628 (0/3); not promoted. Main's plan section is now empty.; see D-D-035 Evidence: `reports/hm07_summary.json`
- Artifacts: `scripts/knn_runner.py`, `reports/hm07_*`, `outputs/oof/hm07_*.csv`
- Discovery updates: D-D-035 (negative result)
- Review verdict: none
- Resource: CPU (3 workers × 2 threads)
- Other executor: none
- New plan items: none (H-D-07 removed)

### 2026-10-06 01:20 — A — H-A-01
- Host: Claude Code
- Action: [Slot takeover, same-platform rule] at the user's choice this session released Main (queue exhausted) and took slot A. At the user's request, `agents.md` now treats the host as the platform (not the model) and lets any same-platform session continue a slot its platform holds; added `Last active` to the agent sections. Ran H-A-01 on the champion's honest OOF (`scripts/run_ha01_group_consistency.py`). Added `scripts/extra_features.py` (target-free extras for H-A-02/03/04). (model: Opus)
- Result: H-A-01 mean −0.002032 (0/3); not promoted.; see D-A-004 Evidence: `reports/ha01_summary.json`
- Artifacts: `agents.md`, `handoff.md`, `scripts/run_ha01_group_consistency.py`, `scripts/extra_features.py`
- Discovery updates: D-A-004 (negative result)
- Review verdict: none
- Resource: CPU
- Other executor: none
- New plan items: none — D-A-004 closes group snapping (predeclared setting failed by about 0.002, so no grid follow-up)

### 2026-10-06 01:25 — B — slot opened; user EDA mapped to the plan
- Host: Claude Code
- Action: opened slot B at the user's request; Main was also free, but the user named B. Compared
  the user's EDA (shared in chat) with `plan.md` and `discoveries.md`. Every recommendation except
  categorical interactions was already in the baseline, settled, or queued. Checked that CatBoost
  1.2.10 resolves `max_ctr_complexity=1` on the baseline features, so it builds no automatic
  categorical combinations. No training.
- Result: one new item, H-B-01 (11). One-hot + logistic regression was not queued: a weak member
  does not help (D-D-033), so it is diagnostic only.
- Evidence: EDA mapping in chat; CatBoost `get_all_params()` on the baseline features.
- Discovery updates: none
- Review verdict: none
- Files/metrics: `plan.md` (`## Agent: B`), `handoff.md`
- Resource: none
- Other executor: none
- New plan items: H-B-01 (11)
- Next resumable action: H-B-01 spec and screen.

### 2026-10-06 01:27 — A — H-A-07
- Host: Claude Code
- Action: nested refit proxy (`scripts/run_ha07_refit.py`) on seeds 42/123/2026, with the champion's recorded per-fold iterations; B1 reused the champion OOF. Also added `scripts/extra_runner.py` (`catboost_extra` model in `run_feature_screen.py`) and specs `configs/ha02/ha03/ha04_spec.json`. (model: Opus)
- Result: B1 +0.001035, B2 +0.000882 (2/3 each); not promoted.; see D-A-005 Evidence: `reports/ha07_summary.json`
- Artifacts: `scripts/run_ha07_refit.py`, `scripts/extra_runner.py`, `scripts/extra_features.py`, `reports/ha07_*`
- Discovery updates: D-A-005 (negative result)
- Review verdict: none
- Resource: CPU (3 workers × 2 threads, alongside H-A-02)
- Other executor: none
- New plan items: none — D-A-005 shows the final-fit recipe is within split noise; keep fold averaging

### 2026-10-06 01:45 — A — H-A-02
- Host: Claude Code
- Action: screened Surname as a CatBoost categorical, with optional repeat-count / SurnameGroupSize arms (`configs/ha02_spec.json`), against the champion runs. (model: Opus)
- Result: all three arms negative (−0.001572 to −0.000422); not promoted.; see D-A-006 Evidence: `reports/ha02_summary.json`
- Artifacts: `reports/ha02_*`, `outputs/oof/ha02_*.csv`
- Discovery updates: D-A-006 (negative result)
- Review verdict: none
- Resource: CPU (3 workers × 2 threads)
- Other executor: none
- New plan items: none — D-A-006 closes surname identity; H-A-03/04 already cover the remaining group/location ideas

### 2026-10-06 01:55 — A — none
- Host: Claude Code
- Action: [Rules synced to the skill update (01:40)] the skill gained A-first slot names, host = platform, same-host item takeover, mandatory follow-up planning, and a "queue runs out" order. Merged these into `agents.md`, keeping `Main` as a legacy slot name (renaming it would collide with existing A IDs) and keeping the user's same-platform whole-slot continuation rule. Re-synced `~/.agents/skills/research-orchestrator-skill` (previous copy backed up in the scratchpad). Added reasons to the `New plan items` lines of earlier A events. (model: Opus)
- Result: rules updated; no experiment changed.
- Artifacts: `agents.md`, `handoff.md`
- Discovery updates: none — no experiment in this step
- Review verdict: none
- Resource: none
- Other executor: none
- New plan items: none — rules only

### 2026-10-06 02:00 — A — H-A-03
- Host: Claude Code
- Action: screened four location-feature arms (`configs/ha03_spec.json`) against the champion runs. Before running, the CabinNum fill was corrected to take the deck from groupmates (a missing CabinNum always comes with a missing deck). Launched H-A-05 (`configs/ha05_spec.json`, `scripts/sklearn_runner.py`) on the freed workers. (model: Opus)
- Result: all four arms negative (−0.001841 to −0.000767); not promoted.; see D-A-007 Evidence: `reports/ha03_summary.json`
- Artifacts: `reports/ha03_*`, `outputs/oof/ha03_*.csv`, `scripts/sklearn_runner.py`
- Discovery updates: D-A-007 (negative result)
- Review verdict: none
- Resource: CPU
- Other executor: none
- New plan items: none — D-A-007 closes location engineering; H-A-04 (running) is the last feature-side item

### 2026-10-06 02:03 — A — H-A-04
- Host: Claude Code
- Action: [And cross-checks of Codex discoveries] recorded H-A-04 (`configs/ha04_spec.json`). Cross-checked the six PENDING Codex discoveries (queue-runs-out order step 2) by reading the `kaggle competitions submissions` output and the cached notebook sources (static, never executed). (model: Opus)
- Result: H-A-04 negative (−0.001764 / +0.000460); not promoted. Cross-checks: D-D-002, 006, 017, 018 VERIFIED; D-D-019, 020 HOLD (need page scores / an MCP lookup attempt).; see D-A-008 Evidence: `reports/ha04_summary.json`; source line references in the review lines.
- Artifacts: `discoveries.md`, `reports/ha04_*`
- Discovery updates: D-A-008 (negative result)
- Review verdict: D-D-002 CLOSED, D-D-006 CLOSED, D-D-017 CLOSED, D-D-018 CLOSED, D-D-019 HOLD, D-D-020 HOLD
- Resource: CPU
- Other executor: none
- New plan items: none — D-A-008 closes rule imputation; the cross-checks confirm existing exclusions and suggest no new test

### 2026-10-06 02:11 — C — none
- Host: ChatGPT
- Action: reviewed `agents.md`, all discoveries, active plan state, handoff history, the current CatBoost runner, and baseline feature code; checked duplicates; confirmed runtime `Plain`, `max_ctr_complexity=1`, `one_hot_max_size=2`; added H-C-01 to H-C-04 and corrected the stale champion label.
- Result: planning only; added H-C-01 persistent-error audit (19), H-C-02 Ordered boosting (13), H-C-03 one-hot categorical handling (11), and H-C-04 nested pseudo-label simulation (10); no experiment run.
- Artifacts: `plan.md`, `handoff.md`
- Discovery updates: none — planning only, no experiment
- Review verdict: none
- Resource: CPU (read-only checks only; no training)
- Other executor: none
- New plan items: H-C-01 (19), H-C-02 (13), H-C-03 (11), H-C-04 (10)

### 2026-10-06 02:08 — A — none
- Host: Claude Code
- Action: applied the 01:59 skill update (model: Opus). Ran `check_project.py` (303 problems), unwrapped and converted every Main/A discovery, event and section to the single-line formats, moved the current best from `plan.md` into Shared state (`Current best` citing D-D-027, whose Implication now starts with `new current best:`), added `Open consistency issues`, updated `agents.md` (one discovery per experiment, event format, consistency check), and re-synced the Codex skill copy (old copy backed up). B and C sections were not edited; their issues are recorded under Open consistency issues.
- Result: checker down to 110 problems: 1 disputed for Main (legacy name), 6 for B, 103 for C; none open in A's sections
- Artifacts: agents.md, plan.md, discoveries.md, handoff.md; pre-change copies in the session scratchpad (backup-docs-0215)
- Discovery updates: none — format and rules only, no experiment
- Review verdict: none
- Resource: none
- Other executor: none
- New plan items: none — documentation step; H-A-05 still running

### 2026-10-06 02:12 — A — H-A-05
- Host: Claude Code
- Action: screened HistGB / CatBoost Lossguide / ExtraTrees single members (`configs/ha05_spec.json`) and a nested stack of the two least-correlated members (`configs/ha05_stack_spec.json`) (model: Opus). While clearing A's finished item, an over-broad cut removed Agent C's whole plan section; it was restored byte-identical from the pre-normalization backup (C's plan edits were complete by 02:02), and the restore was verified by diff. Also corrected two of A's event times that had been written ahead of the clock (02:10→02:03, 02:20→02:08).
- Result: no single member or stack passes (best stack +0.000882); not promoted; see D-A-009
- Artifacts: scripts/sklearn_runner.py, configs/ha05_spec.json, configs/ha05_stack_spec.json, reports/ha05_summary.json, reports/ha05_stack_summary.json, outputs/oof/ha05_*.csv
- Discovery updates: D-A-009
- Review verdict: none
- Resource: CPU
- Other executor: none
- New plan items: none — D-A-009 closes learner diversity; A's queue is now empty and the remaining queued ideas belong to C (ChatGPT host)

### 2026-10-06 02:15 — A — none
- Host: Claude Code
- Action: renamed legacy slot Main to D at the user's request (A-ZZ rule): H-Main-NN → H-D-NN and D-Main-NNN → D-D-NNN across the four docs, configs, reports, tests and non-fingerprinted scripts; `Active agent(s)` now lists A, B, C, D. Old IDs remain only in the fingerprinted runner scripts so cached runs stay reusable (model: Opus).
- Result: checker no longer flags D or A; remaining problems are B's and C's formatting
- Artifacts: agents.md, plan.md, discoveries.md, handoff.md, configs/*_spec.json, reports/*.json, reports/*.md, scripts (non-fingerprinted), tests/test_screening.py; pre-rename copies in the session scratchpad (backup-before-main-rename)
- Discovery updates: none — rename only, no experiment
- Review verdict: none
- Resource: none
- Other executor: none
- New plan items: none — rename only

### 2026-10-06 02:16 — A — H-A-08
- Host: Claude Code
- Action: took over H-C-01 from C (cross-host: ChatGPT → Claude Code; reason: user-authorized take-over of C's whole queue on 2026-10-06 after A's queue ran out) as H-A-08
- Result: item moved into A's plan section and rescored
- Artifacts: plan.md
- Discovery updates: none — take-over, no experiment
- Review verdict: none
- Resource: none
- Other executor: none
- New plan items: none — take-over only

### 2026-10-06 02:16 — A — H-A-09
- Host: Claude Code
- Action: took over H-C-02 from C (cross-host: ChatGPT → Claude Code; reason: user-authorized take-over of C's whole queue on 2026-10-06 after A's queue ran out) as H-A-09
- Result: item moved into A's plan section and rescored
- Artifacts: plan.md
- Discovery updates: none — take-over, no experiment
- Review verdict: none
- Resource: none
- Other executor: none
- New plan items: none — take-over only

### 2026-10-06 02:16 — A — H-A-10
- Host: Claude Code
- Action: took over H-C-03 from C (cross-host: ChatGPT → Claude Code; reason: user-authorized take-over of C's whole queue on 2026-10-06 after A's queue ran out) as H-A-10
- Result: item moved into A's plan section and rescored
- Artifacts: plan.md
- Discovery updates: none — take-over, no experiment
- Review verdict: none
- Resource: none
- Other executor: none
- New plan items: none — take-over only

### 2026-10-06 02:16 — A — H-A-11
- Host: Claude Code
- Action: took over H-C-04 from C (cross-host: ChatGPT → Claude Code; reason: user-authorized take-over of C's whole queue on 2026-10-06 after A's queue ran out) as H-A-11
- Result: item moved into A's plan section and rescored
- Artifacts: plan.md
- Discovery updates: none — take-over, no experiment
- Review verdict: none
- Resource: none
- Other executor: none
- New plan items: none — take-over only

### 2026-10-06 02:16 — A — H-A-08
- Host: Claude Code
- Action: wrote and ran a read-only five-seed persistent-error audit over the champion's saved OOF (model: Opus)
- Result: errors are mostly persistent and confident, so the plateau is signal-limited; see D-A-010
- Artifacts: scripts/audit_persistent_errors.py, reports/ha08_persistent_errors.json, reports/ha08_persistent_segments.csv
- Discovery updates: D-A-010
- Review verdict: none
- Resource: CPU
- Other executor: none
- New plan items: none — the two elevated segments are the known Earth deck G (D-D-021) and a marginal PSO destination; no target-free feature is suggested, and H-A-09/10/11 already cover the remaining model-side directions

### 2026-10-06 02:16 — A — H-A-10
- Host: Claude Code
- Action: screened `one_hot_max_size=10` against the champion runs (`configs/ha10_spec.json`) (model: Opus)
- Result: flat, not promoted; see D-A-011
- Artifacts: configs/ha10_spec.json, reports/ha10_summary.json, outputs/oof/ha10_*.csv
- Discovery updates: D-A-011
- Review verdict: none
- Resource: CPU
- Other executor: none
- New plan items: none — D-A-011 closes categorical encoding, as the item predeclared

### 2026-10-06 02:25 — A — H-A-11
- Host: Claude Code
- Action: ran a nested pseudo-label simulation inside outer SGKF folds on seeds 42/123/2026 (`scripts/run_ha11_pseudo.py`; teacher agreement at 0.03/0.97 predeclared; the real test set was never used) (model: Opus)
- Result: flat to negative, not promoted; see D-A-012
- Artifacts: scripts/run_ha11_pseudo.py, reports/ha11_summary.json, reports/ha11_seed42.json, reports/ha11_seed123.json, reports/ha11_seed2026.json
- Discovery updates: D-A-012
- Review verdict: none
- Resource: CPU
- Other executor: none
- New plan items: none — D-A-012 closes pseudo-labelling; with D-A-010 it shows confident rows add no signal

### 2026-10-06 02:29 — A — H-A-09
- Host: Claude Code
- Action: screened CatBoost `boosting_type=Ordered` against the champion runs (`configs/ha09_spec.json`), then followed the queue-runs-out order: no different-host PENDING cross-checks, no same-host items, C's items already taken over, and no new hypothesis that could change a decision given D-A-010. Ran the consistency check and released slot A (model: Opus).
- Result: negative, not promoted; see D-A-013
- Artifacts: configs/ha09_spec.json, reports/ha09_summary.json, outputs/oof/ha09_*.csv
- Discovery updates: D-A-013
- Review verdict: none
- Resource: CPU
- Other executor: none
- New plan items: none — D-A-013 closes boosting type; with D-A-010 nothing worthwhile remains on the current features

### 2026-10-06 02:45 — A — none
- Host: Claude Code
- Action: at the user's request, derived new hypotheses from the discoveries (queue-runs-out step 5) after confirming no duplicate in discoveries.md or any agent's queue (model: Fable). Queued H-A-12 TabPFN member/blend (17), H-A-13 fold-local label-noise weighting (15), H-A-14 AutoGluon extreme inside outer SGKF (11), H-A-15 embedding MLP (10), H-A-16 learning-rate axis (8). Slot left released until the user approves the dependency changes.
- Result: plan has five active A items; no experiment run
- Artifacts: plan.md, handoff.md
- Discovery updates: none — planning only, no experiment
- Review verdict: none
- Resource: none
- Other executor: none
- New plan items: H-A-12, H-A-13, H-A-14, H-A-15, H-A-16

### 2026-10-06 02:55 — A — none
- Host: Claude Code
- Action: resumed slot A (model: Opus). The user approved adding `tabpfn` + CUDA `torch`, running H-A-12, and a 7/99 confirmation then submission, and asked to re-implement Viktor's notebook model under honest validation. Statically read Viktor's source and queued H-A-17 (Viktor honest re-implementation, 13) and H-A-18 (other tabular foundation models after H-A-12, 10). Duplicate check: no overlap with H-B-01 or H-C-05.
- Result: two items queued; no experiment run
- Artifacts: plan.md, handoff.md
- Discovery updates: none — planning only, no experiment
- Review verdict: none
- Resource: none
- Other executor: none
- New plan items: H-A-17, H-A-18

### 2026-10-06 02:57 — A — H-A-17
- Host: Claude Code
- Action: added `tabpfn` 9.1.0, CUDA `torch` 2.14.1+cu126 and `imbalanced-learn` via uv (user-approved; no existing package version changed; TabPFN weights are license-gated, so H-A-12 waits for the user's `TABPFN_TOKEN`). Re-implemented Viktor's notebook model (`scripts/viktor_runner.py`, fold-local SMOTE) and screened two arms alone and in nested blends with the champion (model: Opus)
- Result: about 0.805 honest OOF, no blend gain; see D-A-014
- Artifacts: pyproject.toml, uv.lock, scripts/viktor_runner.py, scripts/tabpfn_runner.py, configs/ha17_spec.json, configs/ha17_blend_viktor_plain_spec.json, configs/ha17_blend_viktor_faithful_spec.json, configs/ha12_spec.json, reports/ha17_*
- Discovery updates: D-A-014
- Review verdict: none
- Resource: CPU
- Other executor: none
- New plan items: none — D-A-014 closes public-notebook reproduction as a route; H-A-12/13 already cover the remaining member and training-signal ideas

### 2026-10-06 03:06 — C — none
- Host: ChatGPT
- Action: independently audited the PENDING discovery backlog against saved OOF, summaries, configs and runner code; directly recomputed multi-seed deltas/correlations for the main model claims and reran the target-free notebook-idea structural audit.
- Result: 22 bounded experiment/structure claims are now VERIFIED; D-D-021, D-A-009 and D-A-010 intentionally remain PENDING because their numeric observations reproduce but their implications overreach the evidence; D-D-028 and D-D-031 still need an independent external-LB lookup / adversarial rerun respectively.
- Artifacts: discoveries.md, plan.md, handoff.md, reports/hm02_summary.json, reports/hm08_summary.json, reports/hm09_summary.json, reports/hm10_summary.json, reports/hm11_summary.json, reports/hm03_summary.json, reports/hm05_summary.json, reports/hm06_summary.json, reports/hm12_summary.json, reports/hm07_summary.json, reports/ha01_summary.json, reports/ha07_summary.json, reports/ha02_summary.json, reports/ha03_summary.json, reports/ha04_summary.json, reports/ha10_summary.json, reports/ha11_summary.json, reports/ha09_summary.json, reports/ha17_summary.json
- Discovery updates: D-D-022, D-D-023, D-D-024, D-D-025, D-D-026, D-A-002, D-D-027, D-D-029, D-D-030, D-D-032, D-D-033, D-D-034, D-D-035, D-A-004, D-A-005, D-A-006, D-A-007, D-A-008, D-A-011, D-A-012, D-A-013, D-A-014
- Review verdict: CLOSED — 22 discoveries verified; no CHALLENGED discovery created; D-D-021/D-A-009/D-A-010 left PENDING rather than over-accepting their interpretations
- Resource: CPU (read-only audits and lightweight local recomputation; no model training)
- Other executor: none
- New plan items: none — H-A-12/H-A-13/H-B-01/H-C-05 already cover the only unresolved model/causal directions with enough evidence to change a decision

### 2026-10-06 03:13 — C — none
- Host: ChatGPT
- Action: converted unresolved high-impact discovery interpretations into explicit re-review plan items after duplicate checking against A/B/C queues; strengthened H-C-05 to state why D-A-010 needs causal re-review and added H-C-06/H-C-07/H-C-08 for D-D-021, D-D-031, and D-D-019/D-D-028 respectively.
- Result: planning only; three re-review items added with explicit evidence gaps, improved test designs, and closure/reopen criteria; D-A-009 was not duplicated because H-A-12/H-A-15/H-A-18 already cover its unresolved learner-diversity scope.
- Artifacts: plan.md, handoff.md
- Discovery updates: none — planning only, no discovery state changed
- Review verdict: none
- Resource: none
- Other executor: none
- New plan items: H-C-06, H-C-07, H-C-08; H-C-05 evidence/improvement updated for D-A-010 re-review

### 2026-10-06 03:27 — A — H-A-13
- Host: Claude Code
- Action: screened two predeclared fold-local label-noise arms (`scripts/noise_runner.py`, `configs/ha13_spec.json`) against the champion runs; launched H-A-16 (`configs/ha16_spec.json`) on the freed CPU (model: Opus)
- Result: both arms negative, not promoted; see D-A-015
- Artifacts: scripts/noise_runner.py, configs/ha13_spec.json, reports/ha13_summary.json, outputs/oof/ha13_*.csv
- Discovery updates: D-A-015
- Review verdict: none
- Resource: CPU
- Other executor: none
- New plan items: none — D-A-015 closes training-signal manipulation; remaining items (H-A-12/14/15/16/18) cover learners and the LR axis

### 2026-10-06 03:44 — A — H-A-16
- Host: Claude Code
- Action: screened learning rate 0.02 (iteration cap 3000, champion inner-CV rule) against the champion runs (`configs/ha16_spec.json`); released slot A while H-A-12 waits for the user's token (model: Opus)
- Result: flat to negative, not promoted; see D-A-016
- Artifacts: configs/ha16_spec.json, reports/ha16_summary.json, outputs/oof/ha16_*.csv
- Discovery updates: D-A-016
- Review verdict: none
- Resource: CPU
- Other executor: none
- New plan items: none — D-A-016 closes the last CatBoost axis; the remaining A items depend on H-A-12

### 2026-10-06 09:22 — A — H-A-12
- Host: Claude Code
- Action: the user set `TABPFN_TOKEN`; verified token validity and per-version license acceptance without printing the token, pinned the runner to TabPFN v3.5, screened it against the CatBoost champion on seeds 42/123/2026, confirmed on 7/99, and submitted under the standing rule (pre-approved for H-A-12) (model: Opus)
- Result: promoted and submitted, public LB 0.81786; see D-A-017
- Artifacts: scripts/tabpfn_runner.py, configs/ha12_spec.json, reports/ha12_summary.json, reports/ha12_tabpfn*_metrics.json, outputs/oof/ha12_tabpfn*.csv, outputs/submissions/ha12_tabpfn.csv
- Discovery updates: D-A-017
- Review verdict: none
- Resource: GPU
- Other executor: Kaggle (submission 56865289, user-authorized)
- New plan items: H-A-19, H-A-20 (H-A-18 rescored 10 -> 13)

### 2026-10-06 09:22 — A — H-A-19
- Host: Claude Code
- Action: nested blend screen of TabPFN v3.5 (control) with the CatBoost model over saved OOF (`configs/ha19_spec.json`) (model: Opus)
- Result: no blend gain, not promoted; see D-A-018
- Artifacts: configs/ha19_spec.json, reports/ha19_summary.json
- Discovery updates: D-A-018
- Review verdict: none
- Resource: CPU
- Other executor: none
- New plan items: none — D-A-018 closes CatBoost+TabPFN blending; H-A-18 covers stronger foundation-model partners

### 2026-10-06 09:27 — A — H-A-18
- Host: Claude Code
- Action: installed `tabicl` 2.2.0 (BSD-3, open weights; no existing package version changed); screened TabPFN v2.5 and TabICL against the TabPFN v3.5 champion runs and in nested blends with it (model: Opus)
- Result: both weaker and no blend gain; see D-A-019
- Artifacts: pyproject.toml, uv.lock, scripts/tabicl_runner.py, configs/ha18_spec.json, configs/ha18b_spec.json, configs/ha18_blend_spec.json, configs/ha18b_blend_spec.json, reports/ha18*_summary.json, outputs/oof/ha18*.csv
- Discovery updates: D-A-019
- Review verdict: none
- Resource: GPU
- Other executor: none
- New plan items: none — D-A-019 closes further foundation-model members; H-A-20 (running) tests feature sets for TabPFN v3.5

### 2026-10-06 09:38 — A — H-A-20
- Host: Claude Code
- Action: screened TabPFN v3.5 on v1_spend, v1_rules and a raw-minimal feature set against `ha12_tabpfn` (`configs/ha20_spec.json`). Also removed H-A-14 — its predeclared gate (a usable blend-diversity gain from H-A-12) was not met: D-A-018/019 show no partner adds to TabPFN. Also removed H-A-15 — same unmet gate (run only if H-A-12's blend gains ≥ +0.001) (model: Opus)
- Result: no feature set beats baseline for TabPFN; see D-A-020
- Artifacts: scripts/tabpfn_variant_runner.py, configs/ha20_spec.json, reports/ha20_summary.json, outputs/oof/ha20_*.csv
- Discovery updates: D-A-020
- Review verdict: none
- Resource: GPU
- Other executor: none
- New plan items: H-A-21, H-A-22

### 2026-10-06 09:43 — A — H-A-21
- Host: Claude Code
- Action: nested full-context proxy for TabPFN v3.5 on seeds 42/123/2026 (`scripts/run_ha21_tabpfn_refit.py`; challenger reused the `ha12_tabpfn` OOF); launched H-A-22 on the freed GPU (model: Opus)
- Result: consistent but below the bar (+0.001419, 3/3), not promoted; see D-A-021
- Artifacts: scripts/run_ha21_tabpfn_refit.py, reports/ha21_summary.json, reports/ha21_seed42.json, reports/ha21_seed123.json, reports/ha21_seed2026.json
- Discovery updates: D-A-021
- Review verdict: none
- Resource: GPU
- Other executor: none
- New plan items: none — D-A-021 is a final-recipe question already answered for now by the project rule; H-A-22 tests the remaining TabPFN knob

### 2026-10-06 09:50 — A — H-A-22
- Host: Claude Code
- Action: screened TabPFN v3.5 with `n_estimators=32` against `ha12_tabpfn` (`configs/ha22_spec.json`). Queue-runs-out check: no different-host PENDING discoveries exist (B and C have none); B's H-B-01 is a same-host item but targets CatBoost, which no longer decides anything (D-A-018), so it was not taken; derived H-A-23 instead (model: Opus)
- Result: no change in accuracy; see D-A-022
- Artifacts: scripts/tabpfn_ne_runner.py, configs/ha22_spec.json, reports/ha22_summary.json, outputs/oof/ha22_*.csv
- Discovery updates: D-A-022
- Review verdict: none
- Resource: GPU
- Other executor: none
- New plan items: H-A-23

### 2026-10-06 10:15 — A — H-A-23
- Host: Claude Code
- Action: fold-local fine-tuning of TabPFN v3.5 (`scripts/tabpfn_ft_runner.py`, `configs/ha23_spec.json`) on seeds 42/123/2026 against `ha12_tabpfn`, one GPU worker (model: Opus)
- Result: 3/3 positive but mean +0.001994, just under the +0.002 bar; not promoted; see D-A-023
- Artifacts: scripts/tabpfn_ft_runner.py, configs/ha23_spec.json, reports/ha23_summary.json, reports/ha23_ft*_metrics.json, outputs/oof/ha23_*.csv
- Discovery updates: D-A-023
- Review verdict: none
- Resource: GPU
- Other executor: none
- New plan items: H-A-24

### 2026-10-06 10:50 — A — H-A-24
- Host: Claude Code
- Action: fine-tune-then-refit of TabPFN v3.5 (`scripts/tabpfn_ftrefit_runner.py`, `configs/ha24_spec.json`) on seeds 42/123/2026 against `ha12_tabpfn`; the user raised the goal to LB 0.83 during the run (model: Opus)
- Result: 3/3 positive but mean +0.000767, not promoted; see D-A-024
- Artifacts: scripts/tabpfn_ftrefit_runner.py, configs/ha24_spec.json, reports/ha24_summary.json, reports/ha24_ftrefit*_metrics.json, outputs/oof/ha24_*.csv
- Discovery updates: D-A-024
- Review verdict: none
- Resource: GPU
- Other executor: none
- New plan items: H-A-25

### 2026-10-06 10:50 — A — H-A-25
- Host: Claude Code
- Action: nested blend screen of frozen TabPFN v3.5 (control) with the two fine-tuned variants over saved OOF (`configs/ha25_spec.json`) (model: Opus)
- Result: 3/3 positive but +0.001917, just under the bar; not promoted; see D-A-025
- Artifacts: configs/ha25_spec.json, reports/ha25_summary.json
- Discovery updates: D-A-025
- Review verdict: none
- Resource: CPU
- Other executor: none
- New plan items: H-A-26

### 2026-10-06 12:28 — A — H-A-26
- Host: Claude Code
- Action: ran the predeclared fresh-seed extended test (seeds 11/13/17/19/23) of the frozen + fine-tuned TabPFN v3.5 blend (`scripts/run_ha26_extended.py`). Queue-runs-out check: no different-host PENDING discoveries, B's H-B-01 still targets CatBoost (no decision value), no worthwhile new hypothesis on the current columns. Released slot A pending the user's decision (model: Opus)
- Result: blend 5/5 positive but mean +0.001657, not promoted; single fine-tuned models look better post hoc; see D-A-026
- Artifacts: scripts/run_ha26_extended.py, reports/ha26_summary.json, reports/ha26_manifest.json, reports/ha26_training.log, outputs/oof/ha12_tabpfn_seed{11,13,17,19,23}.csv, outputs/oof/ha23_ft_seed*.csv, outputs/oof/ha24_ftrefit_seed*.csv
- Discovery updates: D-A-026
- Review verdict: none
- Resource: GPU
- Other executor: none
- New plan items: none — D-A-026 leaves only a user decision; no further test on the current columns can change it

### 2026-10-06 12:31 — A — none
- Host: Claude Code
- Action: at the user's suggestion to ensemble different models, checked the record (every pairwise/equal blend already tested, D-A-009/014/018/019/025/026) and ran the one untried form: a nested logistic-regression stacker over member OOF logits (`scripts/run_ha27_stacker.py`, no training). Re-took slot A and queued H-A-27; launched the fine-tuned members on seeds 7/99 for confirmation (model: Fable)
- Result: screen passed for the three TabPFN variants (+0.003068, 3/3) and for those plus CatBoost (+0.002991, 3/3); ten members +0.001879; equal means unchanged from D-A-025. Confirmation pending
- Artifacts: scripts/run_ha27_stacker.py, reports/ha27_summary.json
- Discovery updates: none — screening step of H-A-27; the discovery is written after confirmation
- Review verdict: none
- Resource: GPU
- Other executor: none
- New plan items: H-A-27

### 2026-10-06 12:38 — A — none
- Host: Claude Code
- Action: the user supplied a 2024-2026 tabular model list and asked for the next plans. Audited availability on PyPI and in the installed tabpfn package (read-only), queued H-A-28..H-A-34 with one predeclared membership gate (>= -0.003 vs `ha12_tabpfn` on seeds 42/123/2026 to enter the H-A-27 stacker; normal rule for promotion), and recorded every other listed model under "Not queued" in plan.md with the reason (model: Fable). " + " ".join(line.strip() for line in NOT_QUEUED.splitlines() if line.strip() and not line.startswith("## ")) + "
- Result: seven new items; installs await the user's approval; no experiment run
- Artifacts: plan.md, handoff.md
- Discovery updates: none — planning only, no experiment
- Review verdict: none
- Resource: none
- Other executor: none
- New plan items: H-A-28, H-A-29, H-A-30, H-A-31, H-A-32, H-A-33, H-A-34

### 2026-10-06 13:08 — A — H-A-27
- Host: Claude Code
- Action: fitted `ha23_ft` and `ha24_ftrefit` on seeds 7/99, ran the nested stacker confirmation (`scripts/run_ha27_confirm.py`), and, as it passed, submitted `outputs/submissions/ha27_stack3.csv` (SHA256 prefix 03b62c4c5c75ed12) under the standing user rule; also wired and smoke-tested the H-A-28..33 member runners (model: Opus)
- Result: promoted (7 +0.003336, 99 +0.005292), public LB 0.82020; see D-A-027
- Artifacts: reports/ha27_confirm_summary.json, outputs/submissions/ha27_stack3.csv, outputs/probabilities/ha27_stack3.csv, scripts/member_runner.py, scripts/side_runner.py, configs/ha29_spec.json, configs/ha30_spec.json, configs/ha31_spec.json, configs/ha33_spec.json
- Discovery updates: D-A-027 (new current best)
- Review verdict: none
- Resource: GPU
- Other executor: Kaggle (standing user authorization for CV-promoted candidates; 8 submissions left today at submit time)
- New plan items: none — H-A-28..33 already queued; their gate now screens the stack with each member added against `ha27_stack3`

### 2026-10-06 13:11 — A — H-A-30
- Host: Claude Code
- Action: screened TabPFN v3.5 with raw surname and cabin strings (`configs/ha30_spec.json`) vs `ha12_tabpfn`, then as a fourth stacker member vs `ha27_stack3` with the new `scripts/run_stack_member_screen.py` (model: Opus)
- Result: negative alone (-0.001457, 0/3) and in the stack (-0.000422, 0/3); see D-A-028
- Artifacts: reports/ha30_summary.json, reports/ha30_stack_summary.json, scripts/run_stack_member_screen.py, configs/ha30_spec.json
- Discovery updates: D-A-028 (negative)
- Review verdict: none
- Resource: GPU
- Other executor: none
- New plan items: none — identity strings are closed for both CatBoost and TabPFN; the remaining member items (H-A-28/29/31/32/33/34) already cover new model families

### 2026-10-06 13:13 — A — H-A-31
- Host: Claude Code
- Action: screened xRFM (`configs/ha31_spec.json`, `scripts/member_runner.py`) vs `ha12_tabpfn` on seeds 42/123/2026 (model: Opus)
- Result: negative, mean delta -0.029219 (0/3), below the -0.003 member gate; see D-A-029
- Artifacts: reports/ha31_summary.json, configs/ha31_spec.json
- Discovery updates: D-A-029 (negative)
- Review verdict: none
- Resource: GPU
- Other executor: none
- New plan items: none — below the predeclared member gate, so no stack test or tuning follow-up

### 2026-10-06 13:13 — A — H-A-33
- Host: Claude Code
- Action: screened Causilo (`configs/ha33_spec.json`) vs `ha12_tabpfn` on seeds 42/123/2026, and as a fourth member of the D-A-027 stacker vs `ha27_stack3` (model: Opus)
- Result: negative alone (-0.005177, 0/3) and in the stack (-0.000537, 0/3); see D-A-030
- Artifacts: reports/ha33_summary.json, reports/ha33_stack_summary.json, configs/ha33_spec.json
- Discovery updates: D-A-030 (negative)
- Review verdict: none
- Resource: GPU
- Other executor: none
- New plan items: none — below the member gate and no stack gain; TabDPT (H-A-28) already tests a differently pretrained foundation model

### 2026-10-06 13:14 — A — H-A-28
- Host: Claude Code
- Action: ran TabDPT on seeds 42/123/2026/7/99 in `.venv-side` (`scripts/side_runner.py`; compile and flash attention disabled because Triton and the flash kernel are unavailable on Windows), compared with `ha12_tabpfn`, and screened it as a fourth D-A-027 stacker member (model: Opus)
- Result: negative alone (-0.008551, 0/3) and neutral in the stack (+0.000077, 2/3); see D-A-031
- Artifacts: scripts/side_runner.py, reports/ha28_tabdpt_metrics.json, reports/ha28_stack_summary.json
- Discovery updates: D-A-031 (negative)
- Review verdict: none
- Resource: GPU
- Other executor: none
- New plan items: none — in-context foundation models are exhausted as members (D-A-018/030/031); remaining diversity items H-A-29/32 are already queued

### 2026-10-06 13:46 — A — H-A-29
- Host: Claude Code
- Action: screened RealMLP-TD and TabM-D (`configs/ha29_spec.json`) and TabR-S-D (`configs/ha29b_spec.json`, torch flat-L2 shim in `scripts/tabr_runner.py`) vs `ha12_tabpfn` on seeds 42/123/2026, then each as a fourth D-A-027 stacker member vs `ha27_stack3` (model: Opus)
- Result: negative alone (RealMLP -0.038230, TabM -0.019556, TabR -0.031328, all 0/3) and neutral in the stack (all within ±0.0004); see D-A-032
- Artifacts: reports/ha29_summary.json, reports/ha29_stack_summary.json, scripts/tabr_runner.py, configs/ha29_spec.json, configs/ha29b_spec.json
- Discovery updates: D-A-032 (negative)
- Review verdict: none
- Resource: GPU
- Other executor: none
- New plan items: none — neural nets from scratch are far below the gate on this data size; no tuning follow-up would close a 0.02-0.04 gap

### 2026-10-06 13:49 — A — H-A-32
- Host: Claude Code
- Action: ran TabSTAR in `.venv-side` (`scripts/side_runner.py`) on seeds 42/123/2026, compared with `ha12_tabpfn`, and screened it as a fourth D-A-027 stacker member; fixed a Windows checkpoint lock with per-fold output dirs (model: Opus)
- Result: negative alone (-0.015952, 0/3) and in the stack (-0.000422, 0/3); see D-A-033
- Artifacts: scripts/side_runner.py, reports/ha32_tabstar_metrics.json, reports/ha32_stack_summary.json
- Discovery updates: D-A-033 (negative)
- Review verdict: none
- Resource: GPU
- Other executor: none
- New plan items: none — text-aware modelling has no free text to exploit here; only H-A-34 remains from the 2024-2026 list

### 2026-10-06 14:01 — A — none
- Host: Claude Code
- Action: at the user's request, surveyed top-solution write-ups of recent Kaggle Playground Series episodes (S4E10-S6E8) for strategies not yet tried, via a research subagent reading primary write-ups through a public reader; checked each against discoveries.md (model: Opus)
- Result: two transferable untried mechanisms queued, six rejected with reasons; see D-A-034
- Artifacts: discoveries.md (D-A-034 sources), plan.md
- Discovery updates: D-A-034 (survey)
- Review verdict: none
- Resource: none
- Other executor: none
- New plan items: H-A-36, H-A-37

### 2026-10-06 14:03 — A — H-A-36
- Host: Claude Code
- Action: screened two predeclared nested meta-models on top of the D-A-027 stacker (`scripts/run_ha36_meta.py`, saved OOF only) vs `ha27_stack3` on 42/123/2026 (model: Opus)
- Result: negative for CatBoost residual boosting (-0.002416, 0/3) and segment-aware LR (-0.000422, 0/3); see D-A-035
- Artifacts: scripts/run_ha36_meta.py, reports/ha36_summary.json
- Discovery updates: D-A-035 (negative)
- Review verdict: none
- Resource: CPU
- Other executor: none
- New plan items: none — stack-level complexity is closed; member diversity (H-A-35, H-A-37) remains queued

### 2026-10-06 14:06 — A — H-A-37
- Host: Claude Code
- Action: ran a field-aware factorization machine member (`scripts/ffm_runner.py`, CPU) on seeds 42/123/2026 and screened it as a fourth D-A-027 stacker member vs `ha27_stack3` (model: Opus)
- Result: negative (stack -0.000383, 1/3) despite lower correlation with TabPFN (Spearman about 0.93 vs 0.97 for CatBoost); see D-A-036
- Artifacts: scripts/ffm_runner.py, reports/ha37_ffm_metrics.json, reports/ha37_stack_summary.json
- Discovery updates: D-A-036 (negative)
- Review verdict: none
- Resource: CPU
- Other executor: none
- New plan items: none — new model classes are exhausted as members; remaining queue is H-A-35 (running) and H-A-34

### 2026-10-06 14:30 — A — none
- Host: Claude Code
- Action: while answering the user's question about remaining strategies, probed six existing frozen TabPFN-style OOF sets as fourth stacker members (no training) with `scripts/run_stack_member_screen.py` (model: Opus)
- Result: all neutral (best +0.000077); see D-A-037
- Artifacts: reports/ha38probe_stack_summary.json
- Discovery updates: D-A-037 (negative)
- Review verdict: none
- Resource: CPU
- Other executor: none
- New plan items: none — follow-ups depend on H-A-35 (fine-tuning is the only lever left); decided after its result

### 2026-10-06 14:37 — A — none
- Host: Claude Code
- Action: at the user's request ran a read-only planning subagent over discoveries.md, the handoff log, tabpfn 9.1.0 fine-tuning source and fine-tuning literature; verified its key source claims (cosine schedule spans steps_per_epoch * epochs; estimator defaults 2/2/8 with a mismatch warning) and queued four fine-tuning items (model: Opus)
- Result: four fine-tuning hypotheses queued (H-A-39..42); rejected LoRA and label smoothing (not in tabpfn 9.1.0), smaller chunks, kNN-context retrieval and pooled-seed stacking with reasons; subagent expects at most about +0.002..0.004 OOF in total
- Artifacts: plan.md
- Discovery updates: none — planning step; sources are D-A-021..027, D-A-037, D-D-026/027
- Review verdict: none
- Resource: none
- Other executor: none
- New plan items: H-A-39, H-A-40, H-A-41, H-A-42

### 2026-10-06 14:41 — A — H-A-43
- Host: Claude Code
- Action: took over H-C-06 from C (cross-host: ChatGPT → Claude Code; reason: user instruction to continue the ChatGPT queue as re-review) as H-A-43 (model: Opus)
- Result: item moved into A's queue; reviews of Claude Code discoveries are marked same host
- Artifacts: plan.md, handoff.md
- Discovery updates: none — take-over step only
- Review verdict: none
- Resource: none
- Other executor: none
- New plan items: H-A-43

### 2026-10-06 14:41 — A — H-A-44
- Host: Claude Code
- Action: took over H-C-07 from C (cross-host: ChatGPT → Claude Code; reason: user instruction to continue the ChatGPT queue as re-review) as H-A-44 (model: Opus)
- Result: item moved into A's queue; reviews of Claude Code discoveries are marked same host
- Artifacts: plan.md, handoff.md
- Discovery updates: none — take-over step only
- Review verdict: none
- Resource: none
- Other executor: none
- New plan items: H-A-44

### 2026-10-06 14:41 — A — H-A-45
- Host: Claude Code
- Action: took over H-C-05 from C (cross-host: ChatGPT → Claude Code; reason: user instruction to continue the ChatGPT queue as re-review) as H-A-45; reframed to the TabPFN champion (model: Opus)
- Result: item moved into A's queue; reviews of Claude Code discoveries are marked same host
- Artifacts: plan.md, handoff.md
- Discovery updates: none — take-over step only
- Review verdict: none
- Resource: none
- Other executor: none
- New plan items: H-A-45

### 2026-10-06 14:41 — A — H-A-46
- Host: Claude Code
- Action: took over H-C-08 from C (cross-host: ChatGPT → Claude Code; reason: user instruction to continue the ChatGPT queue as re-review) as H-A-46; D-D-019 (Codex) is a true cross-host review (model: Opus)
- Result: item moved into A's queue; reviews of Claude Code discoveries are marked same host
- Artifacts: plan.md, handoff.md
- Discovery updates: none — take-over step only
- Review verdict: none
- Resource: none
- Other executor: none
- New plan items: H-A-46

### 2026-10-06 14:43 — A — H-A-43
- Host: Claude Code
- Action: re-reviewed D-D-021 with a read-only multi-seed subgroup audit (`scripts/audit_subgroup_seeds.py`) over honest OOF of Logloss vs Accuracy vs fixed300 CatBoost and the champion stack (model: Opus)
- Result: the Logloss Earth-for-non-Earth trade does not replicate (0/3, 1/5); Earth and Deck G error concentration is stable on 5/5 seeds; see D-A-038
- Artifacts: scripts/audit_subgroup_seeds.py, reports/ha43_subgroup_audit.json
- Discovery updates: D-A-038 (new), D-D-021 review
- Review verdict: D-D-021 CHALLENGED
- Resource: CPU
- Other executor: none
- New plan items: none — segment-specific stacking and Earth rules already failed (D-A-035, D-A-008)

### 2026-10-06 14:49 — A — H-A-35
- Host: Claude Code
- Action: ran fine-tune seed offsets 1/2 on seeds 42/123/2026 (`scripts/tabpfn_ftseed_runner.py`), averaged them (`scripts/make_avg_member.py`) and screened the average as a fourth stacker member vs `ha27_stack3`; a post hoc three-run replacement as diagnostic (model: Opus)
- Result: negative (-0.000230, 1/3; diagnostic +0.000575, 2/3); see D-A-039
- Artifacts: scripts/tabpfn_ftseed_runner.py, scripts/make_avg_member.py, reports/ha35_stack_summary.json, reports/ha35diag_stack_summary.json
- Discovery updates: D-A-039 (negative)
- Review verdict: none
- Resource: GPU
- Other executor: none
- New plan items: none — fine-tuning step budget and context items H-A-39..42 already queued and running

### 2026-10-06 14:49 — A — H-A-44
- Host: Claude Code
- Action: re-reviewed D-D-031 with multi-seed group-split adversarial validation, saved row-level OOF and a group-level permutation reference (`scripts/audit_adversarial_seeds.py`) (model: Opus)
- Result: AUC 0.483-0.495 vs permutation 0.506-0.520; no shift; see D-A-040
- Artifacts: scripts/audit_adversarial_seeds.py, reports/ha44_adversarial_seeds.json, outputs/oof/ha44_adversarial_seed42.csv
- Discovery updates: D-A-040 (new), D-D-031 review
- Review verdict: D-D-031 CLOSED
- Resource: CPU
- Other executor: none
- New plan items: none — no shift, so no reweighting or validation change

### 2026-10-06 14:51 — A — H-A-46
- Host: Claude Code
- Action: re-reviewed D-D-019 and D-D-028 with a read-only provenance audit: Kaggle CLI submissions CSV and the public page of `viktortaran/space-titanic` (model: Opus)
- Result: our four submission scores and Viktor's Best Score 0.81833 (V242) verified; see D-A-041
- Artifacts: reports/ha46_provenance.json
- Discovery updates: D-A-041 (new), D-D-019 and D-D-028 reviews
- Review verdict: D-D-019 CLOSED, D-D-028 CLOSED
- Resource: none
- Other executor: Kaggle (read-only API and public page)
- New plan items: none — provenance settled; 0.81833 stays a descriptive benchmark

### 2026-10-06 17:42 — A — H-A-39
- Host: Claude Code
- Action: ran fine-tuning arms A1 (100 epochs) and A2 (lr 3e-5) on seeds 42/123/2026 with ctx/ne2/ne16 inference variants (`scripts/tabpfn_ftbudget_runner.py`; the first background job hit its 2 h limit after A2 and A1 seed 42 and was resumed) and screened them in the stacker (model: Opus)
- Result: negative; best variant A1-ne2 +0.001074 (3/3), below the bar; see D-A-042
- Artifacts: scripts/tabpfn_ftbudget_runner.py, scripts/make_stack_submission.py, reports/ha39_stack_summary.json, reports/ha39_a1_metrics.json
- Discovery updates: D-A-042 (negative)
- Review verdict: none
- Resource: GPU
- Other executor: none
- New plan items: none — H-A-40/41 use the same runner (base arm running) and H-A-42 stays queued

### 2026-10-06 17:44 — A — none
- Host: Claude Code
- Action: at the user's request, screened 11 post hoc combinations of the small-gain fine-tuning members in the nested stacker on 42/123/2026 and built submission files for the best ones with `scripts/make_stack_submission.py` (model: Opus)
- Result: one combination passed the screen (+0.002761, 3/3), confirmation on 7/99 queued as H-A-47; user also asked to treat any small CV gain as a submission candidate
- Artifacts: scripts/make_stack_submission.py, scripts/run_combo_confirm.py, outputs/submissions/sub_combo7.csv, outputs/submissions/sub_combo5.csv, outputs/submissions/sub_avg4.csv
- Discovery updates: none — screening step of H-A-47; the discovery is written after confirmation
- Review verdict: none
- Resource: CPU
- Other executor: none
- New plan items: H-A-47

### 2026-10-06 17:48 — A — none
- Host: Claude Code
- Action: at the user's request (use all remaining submissions, ranked by likely score), submitted eight CV-ranked candidates (model: Opus)
- Result: all eight above the champion's 0.82020, best 0.82604 (`sub_stack3_plus_a1ne2`); see D-A-043
- Artifacts: outputs/submissions/sub_combo7.csv, outputs/submissions/sub_combo7_bag3.csv, outputs/submissions/sub_stack3_plus_a1ne2.csv, scripts/make_stack_submission.py
- Discovery updates: D-A-043 (LB readings)
- Review verdict: none
- Resource: none
- Other executor: Kaggle (user-requested batch; 0 submissions left today)
- New plan items: none — H-A-47 (7/99 confirmation of combo7) already queued

### 2026-10-06 18:03 — A — none
- Host: Claude Code
- Action: stopped the running base-arm job and closed the research session at the user's request to publish a GitHub write-up and a Kaggle notebook; released slot A (model: Opus)
- Result: session closed; H-A-40/41/45/47 remain queued with resumable commands in A's handoff section
- Artifacts: handoff.md
- Discovery updates: none — session close only
- Review verdict: none
- Resource: none
- Other executor: none
- New plan items: none — no new hypotheses at close

### 2026-10-07 11:41 — A — none
- Host: Claude Code
- Action: at the user's request published the Kaggle dataset `taeyangg4/spaceship-titanic-tabpfn-member-predictions` and the notebook `taeyangg4/leak-free-2026-tabpfn-stack-top-public-0-82604`; v1 failed on the competition data path, v2 on the fold-parity check (scikit-learn 1.6.1 splits differently), v3 uses the hashed frozen folds and ran clean; submitted the notebook output once to attach its score (model: Opus)
- Result: Kaggle outputs hash-identical to the submitted files; notebook submission 56897941 scored 0.82604; see D-A-044
- Artifacts: kaggle_notebook/build_notebook.py, kaggle_notebook/README.md, kaggle_dataset/README.md
- Discovery updates: D-A-044 (new)
- Review verdict: none
- Resource: none
- Other executor: Kaggle (user-requested publication and one notebook submission)
- New plan items: none — publication step; no new hypothesis

## Archived history

When this file becomes hard to scan, move older completed-log detail to `docs/<focused-name>.md`
and leave a short summary/link here. Keep shared state, active agent handoffs, and recent
completed/review records in this file.
