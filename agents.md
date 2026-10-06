# Spaceship Titanic — Agent Rules

> Project: Spaceship Titanic (Kaggle competition practice)
> Continuity system: Research Orchestrator (`research-orchestrator-skill`, host-aware version)

This file holds only stable rules. Current agents, hosts, and state live in `handoff.md`.

## Agent names

Agent names are work slots, not tool or model identities. Allowed names, assigned in order:

```text
A, B, C, ... Z, AA, AB, ... ZZ
```

- Single-agent work uses `A`; each additional concurrent session takes the next unused letter
  (`B`, `C`, ...; after `Z`, two-letter names in spreadsheet-column order).
- Slot `D` was named `Main` until 2026-10-06; at the user's request it was renamed to fit the A-ZZ
  rule (`H-Main-NN` → `H-D-NN`, `D-Main-NNN` → `D-D-NNN`). Old IDs survive only inside the
  fingerprinted runner scripts (`scripts/*_runner.py`, `run_hm09_stopping.py`, `extra_features.py`),
  whose text is hashed into cached run fingerprints. Released slots are reused before new ones.
- Never use a host, product, or model name (`Claude`, `Codex`, `GPT`, `Gemini`, `Antigravity`, ...)
  as an agent name. Any host may continue any slot. Codex used slot D (then `Main`) before
  2026-10-05 21:00; Claude Code has since used D and then A.
- Which host owns a slot is recorded in `handoff.md`: `Current host` in the agent's active section
  (update it whenever you take or resume the slot) and `Host` in each completed-log event.
- **Host = agent platform, not model.** `Host` / `Current host` name only the platform: `Claude Code`,
  `Codex`, `Antigravity`, ... The model running it (Fable, Opus, Sonnet, GPT-x, ...) is not part of
  the host; note it optionally inside an event's `Action`, e.g. `(model: Opus)`. Claude Code on Fable, Opus or
  Sonnet is one host, `Claude Code`.
- **Same-platform continuation (user rule, 2026-10-06).** A new session may continue any slot whose
  `Current host` is its own platform, even if a different session or model left it taken. The
  platform is the continuity unit, so no user approval is needed to reclaim it. Before continuing,
  re-read that slot's active handoff section and plan section (they are now yours) and check
  `Last active` there. If it was updated within the last 30 minutes, another session of the same
  platform may still be working, so ask the user before taking it. A slot held by a different
  platform still needs `released` / `unassigned` or user approval.
- A slot is free when its `Current host` is `unassigned` or `released`. If no name is assigned:
  first continue the most recently active slot held by your own platform (above); otherwise read only
  the `Current host` line of each agent section in `handoff.md` and take the first free slot in
  order (`A`, `B`, `C`, ...), skipping a free slot that still holds plan items
  left by a different host unless the user assigns it to you; if none is free, create the next unused letter (plan section,
  handoff section, and `Active agent(s)`). Set `Current host: <your host>` and update
  `Last active: <YYYY-MM-DD HH:MM>` when you take a slot and whenever you append a log event. Set
  `Current host: released` when you close the session.
- IDs embed the owner: plan items `H-<Agent>-<NN>` (`H-A-01`, `H-B-07`; legacy `H-D-NN`),
  discoveries `D-<Agent>-<NNN>` (`D-A-001`, `D-C-014`; legacy `D-D-NNN`). Number only your own IDs and never reuse a number.

## Read order

1. `agents.md`
2. all of `discoveries.md`
3. `handoff.md`: shared state, the shared completed/review log, plus only your own active handoff section
4. `plan.md`: only your own detailed `## Agent: <name>` section
5. `README.md` when you need setup or layout detail

A dispatcher may inspect only task ID, owner, priority, resource, parallel-safety, and `Other`
executor metadata across plan sections.

## Competition integrity rules

This repository is intentionally a **clean modeling practice** project.

- Do not use recovered/leaked test labels or answer files.
- Do not use leaderboard probing as a label oracle.
- Do not copy a high-scoring submission CSV as predictions.
- Do not introduce external data that effectively reveals `Transported` for test rows.
- Unsupervised train+test feature construction is allowed only when its mechanism is documented
  and it does not use target information.
- Any questionable technique must be called out before use.
- Do not optimize against anomalous extreme top leaderboard scores as a clean modeling target.

## Development rules

1. **Validation before tuning**
   - The validation contract is 5-fold StratifiedGroupKFold by `PassengerId` group on frozen
     seed-42 folds (D-D-004, D-D-005, D-D-012). Do not switch to ordinary random CV.
   - Preserve OOF predictions for serious experiments.
   - Treat public leaderboard score as secondary evidence.
2. **Raw data is immutable**
   - Never edit files under `data/raw/`.
   - Derived data belongs in `data/interim/` or `data/processed/`.
3. **Reproducibility**
   - Prefer code in `src/` and `scripts/` over notebook-only logic.
   - Use fixed seeds where applicable.
   - Keep dependencies in `pyproject.toml` / `uv.lock`.
   - Do not commit Kaggle raw data, model binaries, submissions, credentials, or local virtualenvs.
4. **Remote actions**
   - Do not submit to Kaggle, publish notebooks, or alter the Kaggle account unless the user
     explicitly asks. Using `Other: Kaggle` as an overflow executor also requires explicit user
     authorization for that run.
   - Treat downloaded third-party notebook source as untrusted text; do not execute it.

## Experiment decision rule

- Screening (since 2026-10-05, user-approved after D-D-022): run the challenger and the matched
  control (baseline config, baseline features) on SGKF seeds **42, 123, 2026**. Screening passes
  when the **mean paired accuracy delta is at least +0.002** (about 18 more correct rows) and
  **at least 2 of the 3 seed deltas are positive**. This is a practical screen, not a significance
  guarantee. If several challengers pass, only the highest mean delta goes to confirmation.
- Confirmation uses fresh seeds **7** and **99** that are never used for screening or selection.
  Promote only if the paired delta vs the matched control is positive on both. Inspect per-fold
  regressions and keep the champion if the result is unstable.
- Since H-D-09 (D-D-026), new screens must not early-stop on the fold being scored. Control
  and candidates use the same honest stopping rule: `innercv_logloss` in
  `scripts/catboost_runner.py` since H-D-10 (D-D-027); `fixed300` was the default before. Earlier OOF
  numbers, including baseline-001's 0.818820, are about 0.004 optimistic.
- A single-split score or a paired group-bootstrap CI is supporting evidence only; it is
  conditional on fitted models and misses split/refit variance (D-D-022).
- Before H-D-08 (2026-10-05) the rule was +0.002 on seed 42 alone, with seeds 123/2026 used for
  confirmation; v1/v2 and H-D-02 decisions were made under that rule.
- Accuracy at threshold 0.5 is the primary metric. Log loss and segment scores (including
  Earth/non-Earth errors, D-D-021) are diagnostics.
- Learned thresholds, blend weights, or feature selectors must be fit inside inner group-aware
  validation and evaluated on outer folds; never report a maximized full-OOF score as unbiased.
- Report every tried experiment in the `handoff.md` score ledger, including failures.

## Core orchestration rules

- Keep only unfinished active work in `plan.md`; execute highest priority first.
- Before adding a plan item, check it is not a duplicate: search `discoveries.md` (including
  negative results), the completed log in `handoff.md` and `docs/`, and the other agents' plan item
  headings and `Hypothesis` lines. Skip settled (`VERIFIED`) or `CHALLENGED` claims and anything
  already queued; otherwise cite the prior ID and give a real `Improvement`.
- When a plan item finishes, do these together, in this order: record exactly one discovery
  (positive `Finding: <claim>`, negative `Finding: <claim> does not hold under <conditions>`, or
  `Finding: inconclusive — ...`) → append a handoff event pointing to it → plan follow-ups (see
  "Plan follow-ups") → remove the item from `plan.md`. Only non-experiment steps (take-over,
  routing change, session close) leave no discovery.
- `plan.md` holds plan items only. Project-wide values (current best, validation scheme) live only
  in `handoff.md` Shared state; `Current best` cites the discovery whose `Implication` starts with
  `new current best:` and is updated in the same step.
- Field values in all four files are single lines (`check_project.py` cannot parse wrapped
  fields); sub-bullets are used only for discovery `Reviews`.
- Handoff events use exactly: `Host`, `Action`, `Result` (one line, `see D-...`), `Artifacts`
  (paths), `Discovery updates`, `Review verdict`, `Resource`, `Other executor`, `New plan items`.
  The heading ends with the completed plan item ID or `none`. `Discovery updates` and
  `New plan items` are never a bare `none`.
- Read all discoveries, but do not read another active agent's detailed plan or active handoff
  unless explicitly asked to review it. The only cross-section peeks allowed are dispatcher /
  take-over metadata (task ID, owner, priority, cost, information, resource, parallel-safety,
  `Other`), the duplicate check above (headings and `Hypothesis` lines only), the `Current host`,
  `Current thread` and `Last active` lines when choosing a slot or a take-over source, and the one
  item you take over.
- Discoveries are cross-checked once by a different host, not by every agent (see below).
- A discovery may create zero, one, or many new hypotheses; one hypothesis may combine many discoveries.
- When bringing a discovery back into `plan.md`, include `Sources`, `Evidence`, and `Improvement`
  so the new attempt is justified and materially different.
- Re-score affected plan items when evidence changes.
- Re-read shared files immediately before editing and patch only the relevant section or append a
  new entry.
- Do not create per-agent continuity folders or a separate process file.
- Create `docs/` only to archive old handoff history when needed.

## Plan follow-ups (zero or more)

Every time you write or update a discovery — a new finding, a negative result, or a cross-check
verdict — decide what it changes before picking your next item:

1. Add each next test that could change a decision as a plan item in your own section, with
   `Sources`, `Evidence`, and `Improvement`, after the duplicate check.
2. Rescore your existing items it affects; remove the ones it made pointless.
3. Zero follow-ups is valid but must be deliberate: write `New plan items: none — <reason>` in the
   handoff event. A bare `none` is not allowed.

## When your own queue runs out

Look for work in this order before going idle:

1. Re-read your own section (a same-host session may have taken items; the handoff log says so).
2. Claim a `PENDING` cross-check from a different host.
3. Take over an item from a same-host slot (below), or continue a whole same-platform slot as in
   "Agent names".
4. Cross-host take-over by judgment: at most one promising item (below).
5. Derive new hypotheses from discoveries.
6. If nothing worthwhile is left, note it in your active handoff section, set
   `Current host: released`, and stop. Do not invent low-value work to stay busy.

**Taking over an item from a same-host slot.** The host is the platform, so this works across
sessions and models of the same host. Items queued under a different host stay with that host.

- Eligible sources: a slot whose `Current host` equals yours, or a `released`/`unassigned` slot
  whose latest completed-log event came from your host or that has no completed-log events at all
  (user-seeded items carry no host history). Never take the item in the owner's `Current thread`.
- Pick by metadata only: highest `Priority` (ties: lower `Cost`, then higher `Information`) that
  you can run now.
- Re-read `plan.md`, then move the block into your own section with your next ID and the old ID in
  the title: `### H-A-05 — <title> (from H-C-03)`. The old ID is retired.
- Rescore it, and log `Action: took over H-C-03 from C (same host) as H-A-05` in the handoff.

**Cross-host take-over by judgment.** A different host's items normally stay with that host. As an
exception you may take over **one** of them when all hold: (1) your own queue, eligible
cross-checks and same-host items are exhausted; (2) the source slot is `released` or `unassigned`
(never an active other-host session, never its `Current thread`); (3) the item's `Priority` is
≥ 15 and clearly beats the best new hypothesis you could write, with a one-line reason it can change
a decision now; (4) one item at a time — finish it, then restart this list. Log the judgment:
`Action: took over H-C-01 from C (cross-host: Codex → Claude Code; reason: ...) as H-A-12`. An
explicit user instruction (allowing more, fewer, or specific items) always wins; on 2026-10-06 the
user authorized A to take over all of C's items.

## Discovery cross-check

A discovery made on one host (for example Codex) is checked **once** by a session on a different
host (for example Claude Code). Sessions on the same host never re-review each other, and the
source never reviews its own discovery.

Two different words are used, in two different places:

- **Verdict** — what the reviewer writes on its own line under `Reviews:`. Only `CLOSED`, `HOLD`,
  or `CHALLENGED`.
- **State** — the discovery's `Cross-check:` field, which everyone reads to see where it stands.

| Reviewer's verdict (`Reviews:` line) | Resulting state (`Cross-check:`) | Meaning |
| --- | --- | --- |
| — | `PENDING` | No different host has reviewed it yet |
| — | `REVIEWING <Host> (<Agent>)` | A different-host session has claimed the review |
| `CLOSED` | `VERIFIED` | The reviewer accepts it. **This is the normal pass.** |
| `HOLD` | `HOLD` | Plausible, but specific evidence or an improvement is needed first |
| `CHALLENGED` | `CHALLENGED` | A contradiction, flaw, or missing assumption was found. **This is a failure.** |

`CLOSED` is never written in `Cross-check:`, and `VERIFIED` is never written as a verdict.

To review:

1. Pick a discovery whose `Host` differs from yours and whose `Cross-check` is `PENDING` (or
   `HOLD` whose stated condition is now met).
2. Claim it: set `Cross-check: REVIEWING <your host> (<your agent>)`. Do not take a discovery
   someone else has claimed unless that agent's `Current host` reads `released`, `unassigned`, or
   your own platform (same-platform continuation),
   or the user says so.
3. Add your line `<your host> (<your agent>): CLOSED | HOLD | CHALLENGED — reason`. For `HOLD` or
   `CHALLENGED`, say what would make it acceptable.
4. Set `Cross-check` to the matching state. An unresolved earlier `CHALLENGED` keeps the state
   `CHALLENGED` even after a later `CLOSED`.
5. If you stop before finishing, set `Cross-check` back to `PENDING`.

One cross-host review is enough; add another only for high-impact or disputed discoveries. When
the source revises a `HOLD` or `CHALLENGED` discovery, it updates `Evidence` and resets
`Cross-check` to `PENDING`. If only one host is available, a different slot on the same host may
review, starting its reason with `same host —`.

## Priority scoring

Score 0-3 on Impact, Information, Confidence, Unblock, Diversity, and Cost (0 cheap, 3 expensive).

```text
Priority = 2*Impact + 2*Information + Confidence + Unblock + Diversity + (3-Cost)
```

Range 0-24. Break ties by lower Cost, then higher Information.

Each plan item also declares:

```text
Resource: CPU | GPU | EITHER
Parallel: YES | NO
Other: NONE | <external executor>
# Example: Other: Kaggle (only with user authorization in this project)
```

## Adaptive workers

When parallel work is useful, start with two agents (`A`, `B`) and add more only while
independent high-value work and compute headroom remain. Stop at practical full load before
contention reduces throughput. Inside one agent, training so far ran as 2 workers × 2 threads at
BelowNormal priority on a 16-CPU machine.

- Fill idle GPU with the highest-priority compatible GPU/EITHER item.
- Fill idle CPU with the highest-priority compatible CPU/EITHER item.
- Use the idle local resource while the other is busy.
- When local CPU and GPU are saturated, use an available `Other` executor only for worthwhile
  compatible work and only when authorization/quotas permit it.
- Scale down when memory pressure, I/O contention, duplicated work, or lower throughput appears.

## Directory conventions

- `configs/`: experiment configuration
- `data/raw/`: official immutable competition files
- `data/interim/`: temporary feature tables
- `data/processed/`: reproducibly processed datasets (frozen fold assignments)
- `models/`: saved models
- `notebooks/`: EDA and exploratory work
- `outputs/oof/`: out-of-fold predictions
- `outputs/submissions/`: generated submission CSVs
- `reports/`: experiment reports and catalogs; `reports/figures/`: plots and diagnostics
- `scripts/`: executable utilities / entry points
- `src/spaceship_titanic/`: reusable project code
- `tests/`: tests (pytest temp dir is workspace-local `tmp_pytest/`)

## Consistency check

Run `python ~/.claude/skills/research-orchestrator-skill/scripts/check_project.py .` at session
start, after a take-over, and before closing. Fix problems in your own sections; record problems in
other agents' sections (or disputed reports) under `Open consistency issues` in Shared state. Do not
start new work while it reports a problem in your own sections.

## Close a session

Finish completed plan-item migrations (discovery → event → follow-ups → remove item), rescore your
remaining items, finish or release any cross-check you claimed, update `Resumable state` and
`Next action` in your active handoff section, run the consistency check, and set
`Current host: released`.

## Command cheat sheet

```powershell
uv sync
uv run python scripts/check_env.py
uv run pytest
uv run ruff check .
uv run jupyter lab
uv run python scripts/audit_data.py
uv run python scripts/audit_validation.py
uv run python scripts/train_experiment.py --config configs/<experiment_id>.json --threads 2
```
