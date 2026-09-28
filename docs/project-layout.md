# Project layout and migration

Active research path should be obvious from the directory structure.
Research internals are configurable, normal workflow is one command.

`run_natural_train.sh` → `src.natural.trainer` → bank / curriculum / difficulty scan → `src.evaluation.natural`.

The configured active sequence is R80 pretrained → NC12 → NC16 → NC20 → NC24 → NC32 → NC48 → NC64 → NC72 → NC80 → NC84 → unconditional natural full-game evaluation.
NC4/NC8 remain rehearsal and scan anchors. No game, action, observation, reward or promotion thresholds were changed.

## File mapping

| Before | After |
| --- | --- |
| `the_game_env.py` | `src/env.py` |
| `curriculum.py` | `src/synthetic/curriculum.py` |
| `train_replay_curriculum.py` | `src/synthetic/trainer.py` |
| `train_natural_reverse.py` | `src/natural/trainer.py` |
| `natural_bank.py` | `src/natural/bank.py` |
| `natural_curriculum.py` | `src/natural/curriculum.py` |
| `natural_run_state.py` | `src/common/checkpoints.py` |
| `natural_usage.py` | `src/common/source_usage.py` |
| `natural_dashboard.py` | `src/common/terminal_ui.py` |
| `natural_evaluation.py` | `src/evaluation/natural.py` |
| `evaluate.py` | `src/evaluation/synthetic.py` |
| `natural_distribution_audit.py` | `src/evaluation/distribution_audit.py` |
| `standalone_natural_heuristic.py` | `src/natural/heuristics.py` |
| `evaluate_standalone_natural.py` | `src/evaluation/heuristic.py` |
| `plot_history.py` | `src/tools/plot_history.py` |
| `html_model.py` | `src/tools/html_model.py` |
| `random_baseline.py` | `src/evaluation/random_baseline.py` |
| `natural_conditioned_env.py` | `archive/nc84_direct_mix/env.py` |
| `train_natural_conditioned.py` | `archive/nc84_direct_mix/trainer.py` |
| `audit_nc_direct_mix.py` | `archive/nc84_direct_mix/audit.py` |
| `run_natural_campaign.py` | `archive/nc84_direct_mix/campaign.py` |
| `natural_teacher.py` | `archive/privileged_teacher/teacher.py` |
| `natural_prefix.py` | `archive/privileged_teacher/prefix.py` |
| `build_natural_bank.py` | `archive/privileged_teacher/build_bank.py` |
| `build_natural_bank_v3.py` | `archive/handoff_v3/build_bank.py` |
| `dietrich_teacher.py` | `archive/handoff_v3/teacher.py` |
| `evaluate_dietrich_teacher.py` | `archive/handoff_v3/evaluate.py` |
| `config.yaml` | `configs/synthetic.yaml` |
| `configs/natural_reverse.yaml` | `configs/natural_replay.yaml` |
| `configs/nc84.yaml` | `archive/nc84_direct_mix/config.yaml` |
| `natural_evaluation.py:difficulty_scan` | `src/natural/difficulty_scan.py` |
| `train_natural_reverse.py:read_checkpoint/save_checkpoint` | `src/common/checkpoints.py` |
| `train_natural_conditioned.py:validate_start` | `src/common/checkpoints.py` |
| `train_replay_curriculum.py:resolve_checkpoint/load_or_create/save_model` | `src/common/checkpoints.py` |
| `natural_conditioned_env.py:SourceUsageWrapper` | `src/common/source_usage.py` |
| `natural_conditioned_env.py:NC_SYNTHETIC_*` | `src/natural/curriculum.py` |
| `docs/natural-reverse.md` | `docs/natural-replay-curriculum.md` |
| `docs/usage.md` | `docs/synthetic-curriculum.md` |
| Past experiment reports | `docs/experiments/` |

## Resume and provenance

New manifests hash repository-relative Python paths under `src/`, including shared dependencies.
Existing run files and schemas are not migrated or rewritten. Model, metadata, bank, config,
manifest, scan, log-prefix and continuation hashes retain their checks.

`archive/pre_package/source.zip` preserves the exact pre-move sources and configs.
`archive/pre_package/migration.json` records that entire old source hash set and the entire
reviewed replacement hash set. The common provenance validator allows only that exact pair;
it verifies archived bytes, rejects missing/changed/unknown sources, and never imports archive code.
A run already incompatible before this move stays incompatible. Checkpoints predating
`continuation_schema: 2` lack the environment/RNG and evaluation state needed for exact
in-place continuation; this migration does not fabricate those missing fields.
Future code changes require
an explicit reviewed compatibility decision rather than disabling source validation.

Python imports and research CLIs now use their package names (`python -m ...`).
There are no root-level Python shims or shell PYTHONPATH overrides. Use the preserved snapshot
for historical flat-module scripts. Default assets resolve against `src/common/paths.py:ROOT`;
explicit relative CLI paths remain relative to the invoking working directory. The shell launchers
change to the repository root before execution, including when invoked from another directory.
