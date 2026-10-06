# Skill snapshots

Copies of the agent skill documents as they were used in this research, kept byte-for-byte for the record (excluded from linting). See [docs/08-skills.md](../08-skills.md) for how each skill was used.

| Folder | Skill | Version | Upstream | License |
|---|---|---|---|---|
| `research-orchestrator-skill/` | Research Orchestrator: four-file research protocol (agents / plan / discoveries / handoff), cross-host review, consistency check | copy synced 2026-10-06 02:21 local time, which governed most of the research (SKILL.md SHA256 `eebe653dc2c48253…`) | https://github.com/TaeyanG4/research-orchestrator-skill (latest at publication: commit `f5d9a40`) | MIT |
| `kaggle/` | Kaggle (unofficial): competition research, submission checks and logging, notebooks | 3.0.1, `SKILL.md` only (SHA256 `8740474db360a271…`) | https://github.com/shepsci/kaggle-skill (commit `a47bff9`) | MIT |

The Kaggle skill's modules and scripts are not copied; see the upstream repository. Claude Code's built-in `update-config` skill was also used once to review notification settings; it is part of Claude Code and not reproduced here.
